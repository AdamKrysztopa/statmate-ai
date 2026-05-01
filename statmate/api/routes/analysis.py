"""Analysis execution API routes."""

import asyncio
import io
import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from config.settings import settings
from database.models import AnalysisStatus, User
from database.session import get_db
from statmate.api.dependencies import get_current_user_optional
from statmate.api.models.analysis import (
    AnalysisCommentUpdate,
    AnalysisCreate,
    AnalysisResponse,
    AnalysisResultResponse,
    AnalysisStatusResponse,
)
from statmate.api.services.analysis_service import AnalysisService
from statmate.api.services.dataset_service import DatasetService
from statmate.api.services.export_service import ExportService
from statmate.api.services.storage_service import StorageService
from statmate.core.config import NodeName

router = APIRouter(prefix="/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)

_VALID_ROUTE_OVERRIDES = {value for key, value in vars(NodeName).items() if key.isupper() and isinstance(value, str)}


@router.post("/run", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def run_analysis(
    request: AnalysisCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> AnalysisResponse:
    """Run a statistical analysis on a dataset.

    The analysis is executed in the background. Use GET /analysis/{id} to check status.

    Args:
        request: Analysis creation parameters
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        AnalysisResponse with analysis details

    Raises:
        HTTPException: If dataset not found or validation fails
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    if request.route_override and request.route_override not in _VALID_ROUTE_OVERRIDES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid route_override: {request.route_override}",
        )

    try:
        # Create analysis record
        analysis = AnalysisService.create_analysis(
            db=db,
            dataset_id=request.dataset_id,
            selected_columns=request.selected_columns,
            configuration=request.configuration,
            route_override=request.route_override,
            model_name=request.model_name,
            provider=request.provider,
            user_id=current_user.id if current_user else None,
            overwrite=request.overwrite,
        )

        # Run analysis in background
        background_tasks.add_task(run_analysis_background, analysis.id)

        return AnalysisResponse.model_validate(analysis)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to start analysis: {str(e)}"
        )


def run_analysis_background(analysis_id: str) -> None:
    """Background task to execute analysis.

    Args:
        analysis_id: Analysis UUID
    """
    from database.session import SessionLocal

    db = SessionLocal()
    try:
        AnalysisService.run_analysis(db, analysis_id)
    except Exception as e:
        print(f"Background analysis failed: {e}")
    finally:
        db.close()


def _parse_trace_from_log(log_content: str) -> list[dict[str, str]]:
    """Extract lightweight execution trace entries from the log."""
    trace: list[dict[str, str]] = []
    for line in log_content.splitlines():
        if "Trace step:" not in line:
            continue
        try:
            _, payload = line.split("Trace step:", 1)
            step_part, detail_part = payload.split("|", 1) if "|" in payload else (payload, "")
            trace.append(
                {
                    "step": step_part.strip() or f"Step {len(trace) + 1}",
                    "detail": detail_part.strip(),
                }
            )
        except ValueError:
            continue
    return trace


@router.get("/{analysis_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> AnalysisStatusResponse:
    """Get the status of an analysis.

    Args:
        analysis_id: Analysis UUID
        db: Database session

    Returns:
        AnalysisStatusResponse with current status

    Raises:
        HTTPException: If analysis not found
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    analysis = AnalysisService.get_analysis(db, analysis_id, user_id=current_user.id if current_user else None)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    analysis = AnalysisService.sanitize_analysis(analysis)

    execution_trace: list[dict[str, str]] | None = None
    if analysis.log_path:
        log_content = AnalysisService.get_analysis_log(
            db, analysis_id, user_id=current_user.id if current_user else None
        )
        if log_content:
            execution_trace = _parse_trace_from_log(log_content)

    progress_pct = None
    if analysis.decision_steps:
        last_step = analysis.decision_steps[-1]
        progress_pct = last_step.get("progress_pct") if isinstance(last_step, dict) else None
        if progress_pct is None:
            progress_pct = min(100.0, len(analysis.decision_steps) / AnalysisService.WORKFLOW_STEP_TARGET * 100)
    if analysis.status == AnalysisStatus.COMPLETED:
        progress_pct = 100.0

    workflow_graph = AnalysisService.workflow_graph_for_analysis(analysis)

    return AnalysisStatusResponse(
        id=analysis.id,
        status=analysis.status.value,
        progress=progress_pct,
        message=analysis.summary or analysis.error_message,
        log_available=bool(analysis.log_path),
        version=analysis.version,
        superseded_at=analysis.superseded_at,
        comment=analysis.comment,
        execution_trace=execution_trace,
        decision_steps=analysis.decision_steps,
        intermediate_log=analysis.intermediate_log,
        assumption_log=analysis.assumption_log,
        workflow_graph=workflow_graph,
    )


@router.get("/{analysis_id}/stream")
async def stream_analysis_events(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> StreamingResponse:
    """Server-sent events for live analysis updates."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    analysis = AnalysisService.get_analysis(db, analysis_id, user_id=current_user.id if current_user else None)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    analysis_owner = current_user.id if current_user else None
    from database.session import SessionLocal

    async def event_generator():
        last_step_idx = 0
        last_log_len = 0
        last_assumption_idx = 0
        session = SessionLocal()
        try:
            while True:
                fresh = AnalysisService.get_analysis(session, analysis_id, user_id=analysis_owner)
                if not fresh:
                    break

                steps = fresh.decision_steps or []
                while last_step_idx < len(steps):
                    payload = dict(steps[last_step_idx])
                    step_index = last_step_idx + 1
                    total = payload.get("total_steps") or AnalysisService.WORKFLOW_STEP_TARGET
                    payload.setdefault("step_index", step_index)
                    payload.setdefault("total_steps", total)
                    payload.setdefault("progress_pct", round(min(1.0, step_index / total) * 100, 2))
                    payload.setdefault("analysis_id", fresh.id)
                    payload.setdefault("version", fresh.version)
                    payload["workflow_graph"] = AnalysisService.build_workflow_graph_state(
                        steps[: last_step_idx + 1], None
                    )
                    yield f"event: step\ndata: {json.dumps(payload, default=str)}\n\n"
                    last_step_idx += 1

                assumptions = fresh.assumption_log or []
                while last_assumption_idx < len(assumptions):
                    payload = dict(assumptions[last_assumption_idx])
                    payload.setdefault("analysis_id", fresh.id)
                    payload.setdefault("version", fresh.version)
                    yield f"event: assumptions\ndata: {json.dumps(payload, default=str)}\n\n"
                    last_assumption_idx += 1

                log_text = fresh.intermediate_log or ""
                if len(log_text) > last_log_len:
                    chunk = log_text[last_log_len:]
                    log_payload = {"analysis_id": fresh.id, "chunk": chunk, "version": fresh.version}
                    yield f"event: log\ndata: {json.dumps(log_payload)}\n\n"
                    last_log_len = len(log_text)

                status_value = fresh.status.value if hasattr(fresh.status, "value") else str(fresh.status)
                if fresh.status in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED):
                    done_payload = {
                        "analysis_id": fresh.id,
                        "status": status_value,
                        "version": fresh.version,
                        "progress_pct": 100.0,
                        "workflow_graph": AnalysisService.build_workflow_graph_state(fresh.decision_steps or [], None),
                    }
                    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
                    break

                await asyncio.sleep(1.0)

            # Final flush of any remaining log content
            fresh = AnalysisService.get_analysis(session, analysis_id, user_id=analysis_owner)
            if fresh:
                log_text = fresh.intermediate_log or ""
                if len(log_text) > last_log_len:
                    chunk = log_text[last_log_len:]
                    log_payload = {"analysis_id": fresh.id, "chunk": chunk}
                    yield f"event: log\ndata: {json.dumps(log_payload)}\n\n"
                assumptions = fresh.assumption_log or []
                while last_assumption_idx < len(assumptions):
                    payload = dict(assumptions[last_assumption_idx])
                    payload.setdefault("analysis_id", fresh.id)
                    payload.setdefault("version", fresh.version)
                    yield f"event: assumptions\ndata: {json.dumps(payload, default=str)}\n\n"
                    last_assumption_idx += 1
        finally:
            session.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/workflow-graph")
async def get_workflow_graph(
    analysis_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> dict:
    """Return the canonical workflow graph and optional per-analysis progress."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    if not analysis_id:
        return AnalysisService.build_workflow_graph_state([], None)

    analysis = AnalysisService.get_analysis(db, analysis_id, user_id=current_user.id if current_user else None)
    if not analysis:
        logger.warning("Workflow graph requested for missing analysis_id=%s", analysis_id)
        return {
            **AnalysisService.build_workflow_graph_state([], None),
            "analysis_id": analysis_id,
            "error": "Analysis not found",
        }

    analysis = AnalysisService.sanitize_analysis(analysis)
    return AnalysisService.workflow_graph_for_analysis(analysis)


@router.get("/{analysis_id}/results", response_model=AnalysisResultResponse)
async def get_analysis_results(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> AnalysisResultResponse:
    """Get the results of a completed analysis.

    Args:
        analysis_id: Analysis UUID
        db: Database session

    Returns:
        AnalysisResultResponse with detailed results

    Raises:
        HTTPException: If analysis not found or not completed
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    results = AnalysisService.get_analysis_results(db, analysis_id, user_id=current_user.id if current_user else None)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found or not yet completed",
        )

    return AnalysisResultResponse(**results)


@router.get("/{analysis_id}/log")
async def get_analysis_log(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> dict[str, str]:
    """Get the execution log for an analysis.

    Args:
        analysis_id: Analysis UUID
        db: Database session

    Returns:
        Dictionary with log content

    Raises:
        HTTPException: If log not found
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    log_content = AnalysisService.get_analysis_log(db, analysis_id, user_id=current_user.id if current_user else None)
    if log_content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")

    return {
        "analysis_id": analysis_id,
        "log_content": log_content,
        "log_lines": str(len(log_content.split("\n"))),
    }


@router.get("/{analysis_id}/export/{export_format}")
async def export_analysis_report(
    analysis_id: str,
    export_format: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> StreamingResponse:
    """Export an analysis as PDF, DOCX, or CSV."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    results = AnalysisService.get_analysis_results(db, analysis_id, user_id=current_user.id if current_user else None)
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found or incomplete")

    dataset_df = None
    try:
        dataset = DatasetService.get_dataset(
            db, results["dataset_id"], user_id=current_user.id if current_user else None
        )
        if dataset:
            dataset_df = StorageService.read_dataset(settings.get_upload_path(dataset.filename))
    except Exception:
        dataset_df = None

    try:
        html = ExportService.render_html_report(results)
        filename = f"analysis-{analysis_id}.{export_format}"
        if export_format == "pdf":
            payload = ExportService.render_pdf(html)
            media_type = "application/pdf"
        elif export_format in ("docx", "word"):
            payload = ExportService.render_docx(results)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            filename = f"analysis-{analysis_id}.docx"
        elif export_format == "csv":
            payload = ExportService.render_csv(results)
            media_type = "text/csv"
        elif export_format in ("latex", "tex"):
            payload = ExportService.render_latex_report(results)
            media_type = "application/x-tex"
            filename = f"analysis-{analysis_id}.tex"
        elif export_format in ("bundle", "zip"):
            log_content = AnalysisService.get_analysis_log(
                db, analysis_id, user_id=current_user.id if current_user else None
            )
            payload = ExportService.build_repro_bundle(results, dataset_df, log_content)
            media_type = "application/zip"
            filename = f"analysis-{analysis_id}-bundle.zip"
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported export format")
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(io.BytesIO(payload), media_type=media_type, headers=headers)


@router.get("/", response_model=list[AnalysisResponse])
async def list_analyses(
    skip: int = 0,
    limit: int = 100,
    dataset_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> list[AnalysisResponse]:
    """List all analyses with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        dataset_id: Optional dataset filter
        db: Database session

    Returns:
        List of AnalysisResponse objects
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    analyses = AnalysisService.list_analyses(
        db,
        skip=skip,
        limit=limit,
        user_id=current_user.id if current_user else None,
        dataset_id=dataset_id,
    )
    return [AnalysisResponse.model_validate(AnalysisService.sanitize_analysis(a)) for a in analyses]


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> None:
    """Delete an analysis version and its stored artifacts."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    deleted = AnalysisService.delete_analysis(db, analysis_id, user_id=current_user.id if current_user else None)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    # Files are removed inside the service via StorageService


@router.patch("/{analysis_id}/comment", response_model=AnalysisResponse)
async def update_analysis_comment(
    analysis_id: str,
    payload: AnalysisCommentUpdate,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> AnalysisResponse:
    """Upsert a user comment on an analysis."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    updated = AnalysisService.update_comment(
        db, analysis_id, comment=payload.comment, user_id=current_user.id if current_user else None
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return AnalysisResponse.model_validate(updated)
