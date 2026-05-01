"""Dataset service for managing uploaded datasets."""

import logging
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sqlalchemy.orm import Session

from database.models import Analysis, AnalysisStatus, Dataset, ScheduledTask
from statmate.api.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for dataset management operations."""

    SUPPORTED_EXTENSIONS = {
        '.csv',
        '.tsv',
        '.txt',
        '.xlsx',
        '.xls',
        '.json',
        '.parquet',
        '.md',
        '.doc',
        '.docx',
    }

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
        df = DatasetService._load_dataframe(file_content, suffix)

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
    def _load_dataframe(file_content: bytes, suffix: str) -> pd.DataFrame:
        """Load an uploaded file into a DataFrame based on extension."""
        buffer = pd.io.common.BytesIO(file_content)  # type: ignore[attr-defined]

        if suffix == '.csv':
            return pd.read_csv(buffer)
        if suffix == '.tsv':
            buffer.seek(0)
            return pd.read_csv(buffer, sep='\t')
        if suffix == '.txt':
            buffer.seek(0)
            try:
                return pd.read_csv(buffer, sep=None, engine='python')  # auto-detect delimiter
            except Exception:
                text = file_content.decode('utf-8', errors='replace')
                lines = [line for line in text.splitlines() if line.strip()] or [text]
                return pd.DataFrame({'text': lines})
        if suffix in {'.xlsx', '.xls'}:
            buffer.seek(0)
            return pd.read_excel(buffer)
        if suffix == '.json':
            buffer.seek(0)
            return pd.read_json(buffer)
        if suffix == '.parquet':
            buffer.seek(0)
            return pd.read_parquet(buffer)
        if suffix in {'.md', '.doc', '.docx'}:
            text = file_content.decode('utf-8', errors='replace')
            lines = [line for line in text.splitlines() if line.strip()] or [text]
            return pd.DataFrame({'text': lines})

        msg = f'Unsupported file format: {suffix}'
        raise ValueError(msg)

    @classmethod
    def is_supported_extension(cls, filename: str) -> bool:
        """Check whether the given filename has a supported extension."""
        suffix = Path(filename).suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS

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
    def _resolve_dataset_path(dataset: Dataset) -> Path:
        file_path = Path(str(dataset.filename))
        if not file_path.is_absolute():
            from config.settings import settings

            file_path = settings.get_upload_path(str(dataset.filename))
        return file_path

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
    def dataset_file_exists(dataset: Dataset) -> bool:
        file_path = DatasetService._resolve_dataset_path(dataset)
        return file_path.exists()

    @staticmethod
    def purge_missing_datasets(db: Session, *, user_id: str | None = None) -> list[str]:
        query = db.query(Dataset)
        if user_id:
            query = query.filter(Dataset.user_id == user_id)
        datasets = query.all()

        deleted_ids: list[str] = []
        for dataset in datasets:
            if not DatasetService.dataset_file_exists(dataset):
                db.delete(dataset)
                deleted_ids.append(str(dataset.id))

        if deleted_ids:
            db.commit()

        return deleted_ids

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
            StorageService.delete_upload(str(dataset.filename))
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
        file_path = DatasetService._resolve_dataset_path(dataset)
        if not file_path.exists():
            raise FileNotFoundError(f'Dataset file missing: {file_path}')

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
            'description': dataset.description,
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

        file_path = DatasetService._resolve_dataset_path(dataset)
        return StorageService.read_dataset(file_path)

    @staticmethod
    def rename_columns(
        db: Session, dataset_id: str, renames: dict[str, str], *, user_id: str | None = None, preview_rows: int = 10
    ) -> dict | None:
        """Rename columns for a dataset and propagate to pending tasks/analyses."""
        dataset = DatasetService.get_dataset(db, dataset_id, user_id=user_id)
        if not dataset:
            return None

        if not renames:
            raise ValueError('No column renames provided')

        existing_cols = dataset.column_names or []
        missing = [col for col in renames.keys() if col not in existing_cols]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")

        new_names = [renames.get(str(col), str(col)) for col in existing_cols]
        if len(set(new_names)) != len(new_names):
            raise ValueError('Duplicate target column names detected')

        if any((name is None) or (str(name).strip() == '') for name in new_names):
            raise ValueError('Column names cannot be empty')

        # Read dataset
        file_path = DatasetService._resolve_dataset_path(dataset)

        df = StorageService.read_dataset(file_path)
        df = df.rename(columns=renames)

        dataset.column_names = df.columns.tolist()  # type: ignore[assignment]
        dataset.data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}  # type: ignore[assignment]

        # Persist dataset
        StorageService.save_dataset(df, str(dataset.filename))

        # Update scheduled tasks
        tasks = db.query(ScheduledTask).filter(ScheduledTask.dataset_id == dataset_id).all()
        for task in tasks:
            if task.selected_columns:  # type: ignore[truthy-function]
                task.selected_columns = [  # type: ignore[assignment]
                    renames.get(str(col), str(col)) for col in task.selected_columns
                ]

        # Update pending/running analyses
        analyses = (
            db.query(Analysis)
            .filter(
                Analysis.dataset_id == dataset_id,
                Analysis.status.in_([AnalysisStatus.PENDING, AnalysisStatus.RUNNING]),
            )
            .all()
        )
        for analysis in analyses:
            if analysis.selected_columns:  # type: ignore[truthy-function]
                analysis.selected_columns = [  # type: ignore[assignment]
                    renames.get(str(col), str(col)) for col in analysis.selected_columns
                ]

        db.commit()

        preview_df = df.head(preview_rows)
        preview_data = preview_df.to_dict(orient='records')

        return {
            'dataset_id': dataset.id,
            'original_filename': dataset.original_filename,
            'row_count': dataset.row_count,
            'column_names': dataset.column_names,
            'data_types': dataset.data_types,
            'preview_data': preview_data,
            'preview_rows': len(preview_data),
            'description': dataset.description,
        }

    @staticmethod
    def update_description(
        db: Session, dataset_id: str, description: str | None, *, user_id: str | None = None
    ) -> Dataset | None:
        """Update dataset notes/description."""
        dataset = DatasetService.get_dataset(db, dataset_id, user_id=user_id)
        if not dataset:
            return None

        dataset.description = description  # type: ignore[assignment]
        db.commit()
        db.refresh(dataset)
        return dataset
