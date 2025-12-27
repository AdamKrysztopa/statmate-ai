"""Pydantic models for scheduled task-related API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskCreate(BaseModel):
    """Request model for creating a scheduled task."""

    name: str = Field(description='User-friendly task name')
    task_type: str = Field(description='Task type: one_time or recurring')
    dataset_id: str = Field(description='UUID of the dataset to analyze')
    selected_columns: list[str] | None = Field(
        default=None, description='Columns to include in analysis (None = all columns)'
    )
    configuration: dict[str, Any] | None = Field(
        default=None, description='Optional analysis configuration parameters'
    )
    schedule: str = Field(description='Cron expression (recurring) or ISO datetime (one_time)')

    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task type."""
        allowed = ['one_time', 'recurring']
        if v not in allowed:
            msg = f'task_type must be one of {allowed}'
            raise ValueError(msg)
        return v


class TaskResponse(BaseModel):
    """Response model for scheduled task information."""

    id: str = Field(description='Task unique identifier')
    name: str = Field(description='Task name')
    task_type: str = Field(description='Task type (one_time or recurring)')
    dataset_id: str = Field(description='Related dataset ID')
    dataset_name: str | None = Field(description='Dataset original filename')
    selected_columns: list[str] | None = Field(description='Columns selected for analysis')
    configuration: dict[str, Any] | None = Field(description='Task configuration')
    schedule: str = Field(description='Cron expression or ISO datetime')
    status: str = Field(description='Current status (active, paused, completed, failed)')
    next_run: datetime | None = Field(description='Next scheduled execution time')
    last_run: datetime | None = Field(description='Last execution time')
    run_count: int = Field(description='Number of times executed')
    created_at: datetime = Field(description='When task was created')
    updated_at: datetime = Field(description='When task was last modified')

    class Config:
        """Pydantic config."""

        from_attributes = True


class TaskUpdateResponse(BaseModel):
    """Response model after updating a task."""

    task_id: str = Field(description='Task identifier')
    message: str = Field(description='Success message')
    task: TaskResponse = Field(description='Updated task details')


class TaskPauseResponse(BaseModel):
    """Response model after pausing/resuming a task."""

    task_id: str = Field(description='Task identifier')
    status: str = Field(description='New status (active or paused)')
    message: str = Field(description='Success message')
