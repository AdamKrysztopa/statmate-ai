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
    route_override: str | None = Field(
        default=None,
        description='Optional user override for routing choice (must match NodeName values)',
    )
    overwrite: bool = Field(
        default=False,
        description='If true, mark the latest analysis for this dataset/user as superseded and create a new version',
    )


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
    version: int = Field(description='Monotonic version number scoped to dataset/user')
    superseded_at: datetime | None = Field(description='When this run was superseded by a newer overwrite')
    user_id: str | None = Field(description='Owner user ID')
    summary: str | None = Field(description='Brief summary of results')
    comment: str | None = Field(description='User-supplied comment')
    error_message: str | None = Field(description='Error details if failed')
    probabilities: dict[str, float] | None = Field(description='Test p-values')
    decision_steps: list[dict[str, Any]] | None = Field(
        default=None, description='Intermediate decision steps captured while streaming'
    )
    intermediate_log: str | None = Field(
        default=None, description='Rolling workflow log captured during streaming execution'
    )
    assumption_log: list[dict[str, Any]] | None = Field(
        default=None, description='Assumption diagnostics captured during the workflow'
    )
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
    version: int | None = Field(default=None, description='Version number for this analysis')
    superseded_at: datetime | None = Field(default=None, description='When this run was superseded, if applicable')
    comment: str | None = Field(default=None, description='User-supplied comment')
    decision_steps: list[dict[str, Any]] | None = Field(
        default=None, description='Ordered list of streamed node/agent decisions'
    )
    intermediate_log: str | None = Field(default=None, description='Rolling log output persisted during streaming')
    execution_trace: list[dict[str, Any]] | None = Field(
        default=None, description='Live step-by-step trace (may be partial while running)'
    )
    assumption_log: list[dict[str, Any]] | None = Field(
        default=None, description='Assumption diagnostics captured so far'
    )
    workflow_graph: dict[str, Any] | None = Field(
        default=None, description='Workflow graph metadata plus visited/active nodes'
    )


class AnalysisPlotResponse(BaseModel):
    """Structured chart payload returned for an analysis result."""

    title: str = Field(description='Chart title shown in the client.')
    description: str = Field(description='Short technical description of the plot.')
    image_base64: str = Field(description='Base64-encoded PNG bytes for the rendered chart.')
    type: str = Field(description='Plot type identifier.')
    column: str = Field(description='Column or column pair represented in the chart.')
    content_type: str = Field(description='MIME type for the plot image.')
    caption: str | None = Field(default=None, description='Plain-language narrative caption for the chart.')


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
    comment: str | None = Field(description='User-supplied comment')
    probabilities: dict[str, float] | None = Field(description='Statistical test p-values')
    results_detail: dict[str, Any] | None = Field(description='Detailed results and statistical tree')
    execution_trace: list[dict[str, Any]] | None = Field(
        default=None, description='Ordered list of agent/node outputs for UI display'
    )
    decision_steps: list[dict[str, Any]] | None = Field(
        default=None, description='Intermediate decisions collected while streaming'
    )
    intermediate_log: str | None = Field(
        default=None, description='Rolling workflow log captured during streaming execution'
    )
    plots: list[AnalysisPlotResponse] | None = Field(default=None, description='Base64-encoded diagnostic plots')
    effect_sizes: dict[str, float] | None = Field(
        default=None, description="Computed effect sizes (e.g., Cohen's d) when a binary grouping exists"
    )
    test_hierarchy: dict[str, Any] | None = Field(
        default=None, description='Tree of attempted tests, assumption failures, and chosen path'
    )
    reviewer_report: dict[str, Any] | None = Field(
        default=None, description='Reviewer/consensus agent verdict and adjusted summary'
    )
    assumption_log: list[dict[str, Any]] | None = Field(
        default=None, description='All assumption diagnostics captured during the workflow'
    )
    workflow_graph: dict[str, Any] | None = Field(
        default=None, description='Workflow graph metadata plus visited/active nodes'
    )
    log_available: bool = Field(description='Whether execution log is available')
    version: int = Field(description='Monotonic version number scoped to dataset/user')
    superseded_at: datetime | None = Field(description='When this run was superseded by an overwrite')

    class Config:
        """Pydantic config."""

        from_attributes = True


class AnalysisCommentUpdate(BaseModel):
    """Request body for updating an analysis comment."""

    comment: str | None = Field(default=None, description='New comment text')
