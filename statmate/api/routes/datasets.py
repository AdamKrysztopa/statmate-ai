"""Dataset management API routes."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from database.session import get_db
from statmate.api.models.dataset import (
    DatasetPreviewResponse,
    DatasetResponse,
    DatasetUploadResponse,
)
from statmate.api.services.dataset_service import DatasetService

router = APIRouter(prefix='/datasets', tags=['datasets'])


@router.post('/upload', response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    description: str | None = None,
    db: Session = Depends(get_db),
) -> DatasetUploadResponse:
    """Upload a new dataset.

    Args:
        file: CSV or Excel file
        description: Optional dataset description
        db: Database session

    Returns:
        DatasetUploadResponse with dataset details

    Raises:
        HTTPException: If file format is unsupported or upload fails
    """
    # Validate file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    if not any(file.filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Unsupported file format. Allowed: {", ".join(allowed_extensions)}',
        )

    try:
        dataset = DatasetService.create_dataset(
            db=db,
            file=file.file,
            original_filename=file.filename,
            description=description,
        )

        return DatasetUploadResponse(
            dataset_id=dataset.id,
            message='Dataset uploaded successfully',
            dataset=DatasetResponse.model_validate(dataset),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Upload failed: {str(e)}')


@router.get('/', response_model=list[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[DatasetResponse]:
    """List all datasets with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of DatasetResponse objects
    """
    datasets = DatasetService.list_datasets(db, skip=skip, limit=limit)
    return [DatasetResponse.model_validate(d) for d in datasets]


@router.get('/{dataset_id}', response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
) -> DatasetResponse:
    """Get a specific dataset by ID.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Returns:
        DatasetResponse object

    Raises:
        HTTPException: If dataset not found
    """
    dataset = DatasetService.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')

    return DatasetResponse.model_validate(dataset)


@router.get('/{dataset_id}/preview', response_model=DatasetPreviewResponse)
async def preview_dataset(
    dataset_id: str,
    num_rows: int = 10,
    db: Session = Depends(get_db),
) -> DatasetPreviewResponse:
    """Get a preview of dataset contents.

    Args:
        dataset_id: Dataset UUID
        num_rows: Number of rows to preview (default: 10)
        db: Database session

    Returns:
        DatasetPreviewResponse with preview data

    Raises:
        HTTPException: If dataset not found
    """
    preview = DatasetService.get_dataset_preview(db, dataset_id, num_rows=num_rows)
    if not preview:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')

    return DatasetPreviewResponse(**preview)


@router.delete('/{dataset_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
) -> None:
    """Delete a dataset and its associated data.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Raises:
        HTTPException: If dataset not found
    """
    deleted = DatasetService.delete_dataset(db, dataset_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')
