"""Analysis execution API routes."""

import asyncio
import io
import json

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
from statmate.api.services.export_service import ExportService

router = APIRouter(prefix='/analysis', tags=['analysis'])


@router.post('/run', response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    try:
        # Create analysis record
        analysis = AnalysisService.create_analysis(
            db=db,
            dataset_id=request.dataset_id,
            selected_columns=request.selected_columns,
            configuration=request.configuration,
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to start analysis: {str(e)}'
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
        print(f'Background analysis failed: {e}')
    finally:
        db.close()


def _parse_trace_from_log(log_content: str) -> list[dict[str, str]]:
    """Extract lightweight execution trace entries from the log."""
    trace: list[dict[str, str]] = []
    for line in log_content.splitlines():
        if 'Trace step:' not in line:
            continue
        try:
            _, payload = line.split('Trace step:', 1)
            step_part, detail_part = payload.split('|', 1) if '|' in payload else (payload, '')
            trace.append(
                {
                    'step': step_part.strip() or f'Step {len(trace) + 1}',
                    'detail': detail_part.strip(),
                }
            )
        except ValueError:
            continue
    return trace


@router.get('/{analysis_id}', response_model=AnalysisStatusResponse)
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    analysis = AnalysisService.get_analysis(db, analysis_id, user_id=current_user.id if current_user else None)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Analysis not found')

    execution_trace: list[dict[str, str]] | None = None
    if analysis.log_path:
        log_content = AnalysisService.get_analysis_log(
            db, analysis_id, user_id=current_user.id if current_user else None
        )
        if log_content:
            execution_trace = _parse_trace_from_log(log_content)

    return AnalysisStatusResponse(
        id=analysis.id,
        status=analysis.status.value,
        message=analysis.summary or analysis.error_message,
        log_available=bool(analysis.log_path),
        version=analysis.version,
        superseded_at=analysis.superseded_at,
        comment=analysis.comment,
        execution_trace=execution_trace,
        decision_steps=analysis.decision_steps,
        intermediate_log=analysis.intermediate_log,
    )


@router.get('/{analysis_id}/stream')
async def stream_analysis_events(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> StreamingResponse:
    """Server-sent events for live analysis updates."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    analysis = AnalysisService.get_analysis(db, analysis_id, user_id=current_user.id if current_user else None)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Analysis not found')

    async def event_generator():
        last_step_idx = 0
        last_log_len = 0

        while True:
            db.refresh(analysis)

            steps = analysis.decision_steps or []
            while last_step_idx < len(steps):
                payload = dict(steps[last_step_idx])
                payload.setdefault('analysis_id', analysis.id)
                payload.setdefault('version', analysis.version)
                yield f'event: step\ndata: {json.dumps(payload, default=str)}\n\n'
                last_step_idx += 1

            log_text = analysis.intermediate_log or ''
            if len(log_text) > last_log_len:
                chunk = log_text[last_log_len:]
                log_payload = {'analysis_id': analysis.id, 'chunk': chunk, 'version': analysis.version}
                yield f'event: log\ndata: {json.dumps(log_payload)}\n\n'
                last_log_len = len(log_text)

            status_value = analysis.status.value if hasattr(analysis.status, 'value') else str(analysis.status)
            if analysis.status in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED):
                done_payload = {'analysis_id': analysis.id, 'status': status_value, 'version': analysis.version}
                yield f'event: done\ndata: {json.dumps(done_payload)}\n\n'
                break

            await asyncio.sleep(1.0)

        # Final flush of any remaining log content
        db.refresh(analysis)
        log_text = analysis.intermediate_log or ''
        if len(log_text) > last_log_len:
            chunk = log_text[last_log_len:]
            log_payload = {'analysis_id': analysis.id, 'chunk': chunk}
            yield f'event: log\ndata: {json.dumps(log_payload)}\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@router.get('/{analysis_id}/results', response_model=AnalysisResultResponse)
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    results = AnalysisService.get_analysis_results(db, analysis_id, user_id=current_user.id if current_user else None)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Analysis not found or not yet completed',
        )

    return AnalysisResultResponse(**results)


@router.get('/{analysis_id}/log')
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    log_content = AnalysisService.get_analysis_log(db, analysis_id, user_id=current_user.id if current_user else None)
    if log_content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Log not found')

    return {
        'analysis_id': analysis_id,
        'log_content': log_content,
        'log_lines': str(len(log_content.split('\n'))),
    }


@router.get('/{analysis_id}/export/{export_format}')
async def export_analysis_report(
    analysis_id: str,
    export_format: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> StreamingResponse:
    """Export an analysis as PDF, DOCX, or CSV."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    results = AnalysisService.get_analysis_results(db, analysis_id, user_id=current_user.id if current_user else None)
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Analysis not found or incomplete')

    try:
        html = ExportService.render_html_report(results)
        filename = f'analysis-{analysis_id}.{export_format}'
        if export_format == 'pdf':
            payload = ExportService.render_pdf(html)
            media_type = 'application/pdf'
        elif export_format in ('docx', 'word'):
            payload = ExportService.render_docx(results)
            media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            filename = f'analysis-{analysis_id}.docx'
        elif export_format == 'csv':
            payload = ExportService.render_csv(results)
            media_type = 'text/csv'
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Unsupported export format')
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    headers = {'Content-Disposition': f'attachment; filename={filename}'}
    return StreamingResponse(io.BytesIO(payload), media_type=media_type, headers=headers)


@router.get('/', response_model=list[AnalysisResponse])
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    analyses = AnalysisService.list_analyses(
        db,
        skip=skip,
        limit=limit,
        user_id=current_user.id if current_user else None,
        dataset_id=dataset_id,
    )
    return [AnalysisResponse.model_validate(a) for a in analyses]


@router.delete('/{analysis_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> None:
    """Delete an analysis version and its stored artifacts."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    deleted = AnalysisService.delete_analysis(db, analysis_id, user_id=current_user.id if current_user else None)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Analysis not found')
    # Files are removed inside the service via StorageService


@router.patch('/{analysis_id}/comment', response_model=AnalysisResponse)
async def update_analysis_comment(
    analysis_id: str,
    payload: AnalysisCommentUpdate,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> AnalysisResponse:
    """Upsert a user comment on an analysis."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    updated = AnalysisService.update_comment(
        db, analysis_id, comment=payload.comment, user_id=current_user.id if current_user else None
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Analysis not found')
    return AnalysisResponse.model_validate(updated)
