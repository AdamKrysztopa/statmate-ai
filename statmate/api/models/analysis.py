"""Pydantic models for analysis-related API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalysisCreate(BaseModel):
    """Request model for creating a new analysis."""

    dataset_id: str = Field(description='UUID of the dataset to analyze')
    selected_columns: list[str] | None = Field(
        default=None, description='Columns to include in analysis (None = all columns)'
    )
    model_name: str | None = Field(
        default=None, description='AI model to use for analysis (optional, uses default if not specified)'
    )
    provider: str | None = Field(default=None, description='Model provider (optional, auto-detected from model_name)')
    configuration: dict[str, Any] | None = Field(default=None, description='Optional analysis configuration parameters')


class AnalysisResponse(BaseModel):
    """Response model for analysis information."""

    id: str = Field(description='Analysis unique identifier')
    dataset_id: str = Field(description='Related dataset ID')
    status: str = Field(description='Current status (pending, running, completed, failed)')
    selected_columns: list[str] | None = Field(description='Columns selected for analysis')
    model_name: str | None = Field(description='AI model used for analysis')
    provider: str | None = Field(description='Model provider used')
    configuration: dict[str, Any] | None = Field(description='Analysis configuration')
    start_time: datetime | None = Field(description='When analysis started')
    end_time: datetime | None = Field(description='When analysis completed')
    user_id: str | None = Field(description='Owner user ID')
    summary: str | None = Field(description='Brief summary of results')
    error_message: str | None = Field(description='Error details if failed')
    probabilities: dict[str, float] | None = Field(description='Test p-values')
    scheduled_task_id: str | None = Field(description='Related scheduled task if any')

    class Config:
        """Pydantic config."""

        from_attributes = True


class AnalysisStatusResponse(BaseModel):
    """Response model for analysis status check."""

    id: str = Field(description='Analysis identifier')
    status: str = Field(description='Current status')
    progress: float | None = Field(default=None, description='Progress percentage (0-100)', ge=0, le=100)
    message: str | None = Field(default=None, description='Status message')
    log_available: bool | None = Field(default=None, description='Whether a live execution log is available')
    execution_trace: list[dict[str, Any]] | None = Field(
        default=None, description='Live step-by-step trace (may be partial while running)'
    )


class AnalysisResultResponse(BaseModel):
    """Response model for analysis results."""

    id: str = Field(description='Analysis identifier')
    status: str = Field(description='Analysis status')
    dataset_id: str = Field(description='Related dataset ID')
    dataset_name: str = Field(description='Original dataset filename')
    model_name: str | None = Field(description='AI model used for analysis')
    provider: str | None = Field(description='Model provider used')
    user_id: str | None = Field(description='Owner user ID')
    start_time: datetime | None = Field(description='Analysis start time')
    end_time: datetime | None = Field(description='Analysis end time')
    duration_seconds: float | None = Field(description='Analysis duration in seconds')
    summary: str | None = Field(description='Analysis summary')
    probabilities: dict[str, float] | None = Field(description='Statistical test p-values')
    results_detail: dict[str, Any] | None = Field(description='Detailed results and statistical tree')
    execution_trace: list[dict[str, Any]] | None = Field(
        default=None, description='Ordered list of agent/node outputs for UI display'
    )
    plots: list[dict[str, Any]] | None = Field(default=None, description='Base64-encoded diagnostic plots')
    log_available: bool = Field(description='Whether execution log is available')

    class Config:
        """Pydantic config."""

        from_attributes = True
