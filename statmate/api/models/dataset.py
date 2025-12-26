"""Pydantic models for dataset-related API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    """Request model for creating a dataset (not used directly - file upload via multipart/form-data)."""

    description: str | None = Field(default=None, description='Optional dataset description')


class DatasetResponse(BaseModel):
    """Response model for dataset information."""

    id: str = Field(description='Dataset unique identifier')
    original_filename: str = Field(description='Original filename from user')
    upload_timestamp: datetime = Field(description='When the file was uploaded')
    file_size: int = Field(description='File size in bytes')
    row_count: int | None = Field(description='Number of rows in the dataset')
    column_names: list[str] | None = Field(description='List of column names')
    data_types: dict[str, str] | None = Field(description='Data types for each column')
    description: str | None = Field(description='User-provided description')
    user_id: str | None = Field(description='Owner user ID')

    class Config:
        """Pydantic config."""

        from_attributes = True


class DatasetUploadResponse(BaseModel):
    """Response model after successful dataset upload."""

    dataset_id: str = Field(description='Unique identifier for the uploaded dataset')
    message: str = Field(description='Success message')
    dataset: DatasetResponse = Field(description='Dataset details')


class DatasetPreviewResponse(BaseModel):
    """Response model for dataset preview."""

    dataset_id: str = Field(description='Dataset identifier')
    original_filename: str = Field(description='Original filename')
    row_count: int = Field(description='Total number of rows')
    column_names: list[str] = Field(description='List of column names')
    data_types: dict[str, str] = Field(description='Data types for each column')
    preview_data: list[dict[str, Any]] = Field(description='First N rows of data')
    preview_rows: int = Field(description='Number of preview rows returned')

    class Config:
        """Pydantic config."""

        from_attributes = True
