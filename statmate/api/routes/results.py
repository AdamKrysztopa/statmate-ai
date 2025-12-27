"""Results retrieval API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from config.settings import settings
from database.session import get_db
from statmate.api.dependencies import get_current_user_optional
from statmate.api.models.result import LogResponse, ResultDetailResponse, ResultListItem, ResultListResponse
from statmate.api.services.analysis_service import AnalysisService
from statmate.api.services.dataset_service import DatasetService
from database.models import User

router = APIRouter(prefix='/results', tags=['results'])


@router.get('/', response_model=ResultListResponse)
async def list_results(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> ResultListResponse:
    """List all analysis results with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        ResultListResponse with list of results
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    analyses = AnalysisService.list_analyses(
        db, skip=skip, limit=limit, user_id=current_user.id if current_user else None
    )

    items = []
    for analysis in analyses:
        dataset = DatasetService.get_dataset(db, analysis.dataset_id)
        items.append(
            ResultListItem(
                id=analysis.id,
                dataset_id=analysis.dataset_id,
                dataset_name=dataset.original_filename if dataset else 'Unknown',
                status=analysis.status.value,
                start_time=analysis.start_time,
                end_time=analysis.end_time,
                summary=analysis.summary,
            )
        )

    # Get total count
    from database.models import Analysis

    total_query = db.query(Analysis)
    if current_user:
        total_query = total_query.filter(Analysis.user_id == current_user.id)
    total = total_query.count()

    return ResultListResponse(
        results=items,
        total=total,
        page=skip // limit + 1 if limit > 0 else 1,
        page_size=limit,
    )


@router.get('/{result_id}', response_model=ResultDetailResponse)
async def get_result_detail(
    result_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> ResultDetailResponse:
    """Get detailed result information.

    Args:
        result_id: Analysis/Result UUID
        db: Database session

    Returns:
        ResultDetailResponse with detailed information

    Raises:
        HTTPException: If result not found
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    analysis = AnalysisService.get_analysis(db, result_id, user_id=current_user.id if current_user else None)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Result not found')

    dataset = DatasetService.get_dataset(db, analysis.dataset_id)
    dataset_name = dataset.original_filename if dataset else 'Unknown'

    # Calculate duration
    duration = None
    if analysis.start_time and analysis.end_time:
        duration = (analysis.end_time - analysis.start_time).total_seconds()

    # Try to load detailed results
    results_data = None
    effect_sizes = None
    if analysis.result_path:
        from statmate.api.services.storage_service import StorageService

        results_data = StorageService.read_results(result_id)
        if results_data:
            effect_sizes = results_data.get('effect_sizes')

    return ResultDetailResponse(
        id=analysis.id,
        dataset_id=analysis.dataset_id,
        dataset_name=dataset_name,
        status=analysis.status.value,
        selected_columns=analysis.selected_columns,
        start_time=analysis.start_time,
        end_time=analysis.end_time,
        duration_seconds=duration,
        summary=analysis.summary,
        probabilities=analysis.probabilities,
        results_data=results_data,
        effect_sizes=effect_sizes,
        error_message=analysis.error_message,
    )


@router.get('/{result_id}/log', response_model=LogResponse)
async def get_result_log(
    result_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> LogResponse:
    """Get execution log for a result.

    Args:
        result_id: Analysis/Result UUID
        db: Database session

    Returns:
        LogResponse with log content

    Raises:
        HTTPException: If log not found
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    log_content = AnalysisService.get_analysis_log(db, result_id, user_id=current_user.id if current_user else None)
    if log_content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Log not found')

    return LogResponse(
        analysis_id=result_id,
        log_content=log_content,
        log_lines=len(log_content.split('\n')),
        log_size_bytes=len(log_content.encode('utf-8')),
    )
