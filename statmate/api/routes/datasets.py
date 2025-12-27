"""Dataset management API routes."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from config.settings import settings
from database.models import User
from database.session import get_db
from statmate.api.dependencies import get_current_user_optional
from statmate.api.models.dataset import (
    ColumnRenameRequest,
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
    current_user: User | None = Depends(get_current_user_optional),
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
    if not DatasetService.is_supported_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Unsupported file format. Allowed: {", ".join(sorted(DatasetService.SUPPORTED_EXTENSIONS))}',
        )

    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    try:
        dataset = DatasetService.create_dataset(
            db=db,
            file=file.file,
            original_filename=file.filename,
            description=description,
            user_id=current_user.id if current_user else None,
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
    current_user: User | None = Depends(get_current_user_optional),
) -> list[DatasetResponse]:
    """List all datasets with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of DatasetResponse objects
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    if current_user:
        datasets = DatasetService.list_user_datasets(db, current_user.id, skip=skip, limit=limit)
    else:
        datasets = DatasetService.list_datasets(db, skip=skip, limit=limit)
    return [DatasetResponse.model_validate(d) for d in datasets]


@router.get('/{dataset_id}', response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
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
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    dataset = DatasetService.get_dataset(db, dataset_id, user_id=current_user.id if current_user else None)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')

    return DatasetResponse.model_validate(dataset)


@router.get('/{dataset_id}/preview', response_model=DatasetPreviewResponse)
async def preview_dataset(
    dataset_id: str,
    num_rows: int = 10,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
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
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    preview = DatasetService.get_dataset_preview(
        db, dataset_id, num_rows=num_rows, user_id=current_user.id if current_user else None
    )
    if not preview:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')

    return DatasetPreviewResponse(**preview)


@router.delete('/{dataset_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> None:
    """Delete a dataset and its associated data.

    Args:
        dataset_id: Dataset UUID
        db: Database session

    Raises:
        HTTPException: If dataset not found
    """
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    deleted = DatasetService.delete_dataset(db, dataset_id, user_id=current_user.id if current_user else None)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')


@router.patch('/{dataset_id}/columns', response_model=DatasetPreviewResponse)
async def rename_columns(
    dataset_id: str,
    payload: ColumnRenameRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> DatasetPreviewResponse:
    """Rename dataset columns and return refreshed preview metadata."""
    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')

    renames_raw = payload.renames
    renames: dict[str, str] = {}
    if isinstance(renames_raw, list):
        renames = {item.from_name: item.to_name for item in renames_raw}
    else:
        renames = dict(renames_raw)

    try:
        preview = DatasetService.rename_columns(
            db,
            dataset_id,
            renames,
            user_id=current_user.id if current_user else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not preview:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')

    return DatasetPreviewResponse(**preview)
