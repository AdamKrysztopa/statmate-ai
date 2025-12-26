"""Dataset service for managing uploaded datasets."""

import logging
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sqlalchemy.orm import Session

from database.models import Dataset
from statmate.api.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for dataset management operations."""

    @staticmethod
    def create_dataset(
        db: Session,
        file: BinaryIO,
        original_filename: str,
        description: str | None = None,
        user_id: str | None = None,
    ) -> Dataset:
        """Create a new dataset from uploaded file.

        Args:
            db: Database session
            file: Uploaded file object
            original_filename: Original filename
            description: Optional description
            user_id: Optional owner user ID

        Returns:
            Created Dataset model
        """
        # Generate unique filename
        filename = StorageService.generate_filename(original_filename)

        # Read the file into DataFrame for analysis
        file_content = file.read()
        file.seek(0)  # Reset for saving

        # Determine file type and read
        suffix = Path(original_filename).suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(pd.io.common.BytesIO(file_content))
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(pd.io.common.BytesIO(file_content))
        else:
            msg = f'Unsupported file format: {suffix}'
            raise ValueError(msg)

        # Extract metadata
        row_count = len(df)
        column_names = df.columns.tolist()
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Save as parquet
        file_path = StorageService.save_dataset(df, filename)
        file_size = StorageService.get_file_size(file_path)

        # Create database record
        dataset = Dataset(
            filename=filename,
            original_filename=original_filename,
            file_size=file_size,
            row_count=row_count,
            column_names=column_names,
            data_types=data_types,
            description=description,
            user_id=user_id,
        )

        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        logger.info(f'Created dataset: {dataset.id} ({original_filename})')
        return dataset

    @staticmethod
    def get_dataset(db: Session, dataset_id: str, *, user_id: str | None = None) -> Dataset | None:
        """Get a dataset by ID.

        Args:
            db: Database session
            dataset_id: Dataset UUID
            user_id: Optional owner filter

        Returns:
            Dataset model or None if not found
        """
        query = db.query(Dataset).filter(Dataset.id == dataset_id)
        if user_id:
            query = query.filter(Dataset.user_id == user_id)
        return query.first()

    @staticmethod
    def list_datasets(db: Session, skip: int = 0, limit: int = 100) -> list[Dataset]:
        """List all datasets with pagination.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of Dataset models
        """
        return db.query(Dataset).order_by(Dataset.upload_timestamp.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def list_user_datasets(db: Session, user_id: str, skip: int = 0, limit: int = 100) -> list[Dataset]:
        """List datasets for a specific user."""
        return (
            db.query(Dataset)
            .filter(Dataset.user_id == user_id)
            .order_by(Dataset.upload_timestamp.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def delete_dataset(db: Session, dataset_id: str, *, user_id: str | None = None) -> bool:
        """Delete a dataset and its file.

        Args:
            db: Database session
            dataset_id: Dataset UUID
            user_id: Optional owner filter

        Returns:
            True if deleted, False if not found
        """
        dataset = DatasetService.get_dataset(db, dataset_id, user_id=user_id)
        if not dataset:
            return False

        # Delete file
        try:
            StorageService.delete_upload(dataset.filename)
        except Exception as e:
            logger.warning(f'Could not delete file {dataset.filename}: {e}')

        # Delete database record (cascades to analyses and tasks)
        db.delete(dataset)
        db.commit()

        logger.info(f'Deleted dataset: {dataset_id}')
        return True

    @staticmethod
    def get_dataset_preview(
        db: Session, dataset_id: str, num_rows: int = 10, *, user_id: str | None = None
    ) -> dict | None:
        """Get a preview of dataset contents.

        Args:
            db: Database session
            dataset_id: Dataset UUID
            num_rows: Number of rows to preview
            user_id: Optional owner filter

        Returns:
            Dictionary with preview data or None if not found
        """
        dataset = DatasetService.get_dataset(db, dataset_id, user_id=user_id)
        if not dataset:
            return None

        # Read dataset file
        file_path = Path(dataset.filename)
        if not file_path.is_absolute():
            from config.settings import settings

            file_path = settings.get_upload_path(dataset.filename)

        df = StorageService.read_dataset(file_path)

        # Get preview rows
        preview_df = df.head(num_rows)
        preview_data = preview_df.to_dict(orient='records')

        return {
            'dataset_id': dataset.id,
            'original_filename': dataset.original_filename,
            'row_count': dataset.row_count,
            'column_names': dataset.column_names,
            'data_types': dataset.data_types,
            'preview_data': preview_data,
            'preview_rows': len(preview_data),
        }

    @staticmethod
    def load_dataset_dataframe(db: Session, dataset_id: str, *, user_id: str | None = None) -> pd.DataFrame | None:
        """Load dataset as DataFrame for analysis.

        Args:
            db: Database session
            dataset_id: Dataset UUID
            user_id: Optional owner filter

        Returns:
            DataFrame or None if not found
        """
        dataset = DatasetService.get_dataset(db, dataset_id, user_id=user_id)
        if not dataset:
            return None

        file_path = Path(dataset.filename)
        if not file_path.is_absolute():
            from config.settings import settings

            file_path = settings.get_upload_path(dataset.filename)

        return StorageService.read_dataset(file_path)
