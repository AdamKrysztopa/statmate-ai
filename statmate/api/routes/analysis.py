"""Analysis execution API routes."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from config.settings import settings
from database.session import get_db
from statmate.api.dependencies import get_current_user_optional
from statmate.api.models.analysis import (
    AnalysisCreate,
    AnalysisResponse,
    AnalysisResultResponse,
    AnalysisStatusResponse,
)
from statmate.api.services.analysis_service import AnalysisService
from database.models import User

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

    return AnalysisStatusResponse(
        id=analysis.id,
        status=analysis.status.value,
        message=analysis.summary or analysis.error_message,
    )


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


@router.get('/', response_model=list[AnalysisResponse])
async def list_analyses(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> list[AnalysisResponse]:
    """List all analyses with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of AnalysisResponse objects
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    analyses = AnalysisService.list_analyses(
        db, skip=skip, limit=limit, user_id=current_user.id if current_user else None
    )
    return [AnalysisResponse.model_validate(a) for a in analyses]
