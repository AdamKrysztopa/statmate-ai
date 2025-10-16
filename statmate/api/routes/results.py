"""Results retrieval API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.session import get_db
from statmate.api.models.result import LogResponse, ResultDetailResponse, ResultListItem, ResultListResponse
from statmate.api.services.analysis_service import AnalysisService
from statmate.api.services.dataset_service import DatasetService

router = APIRouter(prefix='/results', tags=['results'])


@router.get('/', response_model=ResultListResponse)
async def list_results(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> ResultListResponse:
    """List all analysis results with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        ResultListResponse with list of results
    """
    analyses = AnalysisService.list_analyses(db, skip=skip, limit=limit)

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

    total = db.query(Analysis).count()

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
    analysis = AnalysisService.get_analysis(db, result_id)
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
    if analysis.result_path:
        from statmate.api.services.storage_service import StorageService

        results_data = StorageService.read_results(result_id)

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
        error_message=analysis.error_message,
    )


@router.get('/{result_id}/log', response_model=LogResponse)
async def get_result_log(
    result_id: str,
    db: Session = Depends(get_db),
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
    log_content = AnalysisService.get_analysis_log(db, result_id)
    if log_content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Log not found')

    return LogResponse(
        analysis_id=result_id,
        log_content=log_content,
        log_lines=len(log_content.split('\n')),
        log_size_bytes=len(log_content.encode('utf-8')),
    )
