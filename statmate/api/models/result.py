"""Pydantic models for result-related API responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResultListItem(BaseModel):
    """Individual result item in list response."""

    id: str = Field(description='Analysis identifier')
    dataset_id: str = Field(description='Related dataset ID')
    dataset_name: str = Field(description='Dataset original filename')
    status: str = Field(description='Analysis status')
    start_time: datetime | None = Field(description='Analysis start time')
    end_time: datetime | None = Field(description='Analysis end time')
    summary: str | None = Field(description='Brief summary')


class ResultListResponse(BaseModel):
    """Response model for list of results."""

    results: list[ResultListItem] = Field(description='List of analysis results')
    total: int = Field(description='Total number of results')
    page: int = Field(default=1, description='Current page number')
    page_size: int = Field(default=50, description='Items per page')


class ResultDetailResponse(BaseModel):
    """Response model for detailed result information."""

    id: str = Field(description='Analysis identifier')
    dataset_id: str = Field(description='Related dataset ID')
    dataset_name: str = Field(description='Dataset original filename')
    status: str = Field(description='Analysis status')
    selected_columns: list[str] | None = Field(description='Columns analyzed')
    start_time: datetime | None = Field(description='Analysis start time')
    end_time: datetime | None = Field(description='Analysis end time')
    duration_seconds: float | None = Field(description='Analysis duration')
    summary: str | None = Field(description='Analysis summary')
    probabilities: dict[str, float] | None = Field(description='Statistical test p-values')
    results_data: dict[str, Any] | None = Field(
        description='Detailed results and statistical information'
    )
    effect_sizes: dict[str, float] | None = Field(
        default=None, description='Computed effect sizes when available (e.g., Cohen’s d)'
    )
    error_message: str | None = Field(description='Error details if failed')

    class Config:
        """Pydantic config."""

        from_attributes = True


class LogResponse(BaseModel):
    """Response model for analysis execution log."""

    analysis_id: str = Field(description='Analysis identifier')
    log_content: str = Field(description='Full log file content')
    log_lines: int = Field(description='Number of lines in log')
    log_size_bytes: int = Field(description='Log file size in bytes')
