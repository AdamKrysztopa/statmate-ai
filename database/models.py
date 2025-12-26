"""SQLAlchemy database models for StatmateAI.

This module defines the database schema including:
- Dataset: Uploaded data files
- Analysis: Statistical analysis runs
- ScheduledTask: Background and recurring tasks
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class AnalysisStatus(str, PyEnum):
    """Status of an analysis run."""

    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class TaskStatus(str, PyEnum):
    """Status of a scheduled task."""

    ACTIVE = 'active'
    PAUSED = 'paused'
    COMPLETED = 'completed'
    FAILED = 'failed'


class TaskType(str, PyEnum):
    """Type of scheduled task."""

    ONE_TIME = 'one_time'
    RECURRING = 'recurring'


def generate_uuid() -> str:
    """Generate a UUID string for use as primary key."""
    return str(uuid.uuid4())


class User(Base):
    """User accounts for authentication and ownership tracking."""

    __tablename__ = 'users'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    # Encrypted email for display and notifications
    email = Column(String(255), nullable=False, unique=True, index=True)
    # Deterministic hash of email for lookups without storing plaintext
    email_hash = Column(String(64), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Integer, default=1, nullable=False)

    datasets = relationship('Dataset', back_populates='owner')
    analyses = relationship('Analysis', back_populates='owner')
    scheduled_tasks = relationship('ScheduledTask', back_populates='owner')

    def __repr__(self) -> str:
        """String representation of User."""
        return f'<User(id={self.id}, email={self.email})>'


class Dataset(Base):
    """Dataset model representing uploaded data files.

    Attributes:
        id: Unique identifier (UUID)
        filename: Internal storage filename
        original_filename: User's original filename
        upload_timestamp: When the file was uploaded
        file_size: Size in bytes
        row_count: Number of rows in the dataset
        column_names: List of column names (JSON)
        data_types: Data types for each column (JSON)
        description: Optional user description
        analyses: Related analysis runs
    """

    __tablename__ = 'datasets'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False, unique=True, index=True)
    original_filename = Column(String(255), nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    row_count = Column(Integer, nullable=True)
    column_names = Column(JSON, nullable=True)  # ["col1", "col2", ...]
    data_types = Column(JSON, nullable=True)  # {"col1": "int64", "col2": "float64", ...}
    description = Column(Text, nullable=True)

    # Relationships
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    analyses = relationship('Analysis', back_populates='dataset', cascade='all, delete-orphan')
    scheduled_tasks = relationship('ScheduledTask', back_populates='dataset', cascade='all, delete-orphan')
    owner = relationship('User', back_populates='datasets')

    def __repr__(self) -> str:
        """String representation of Dataset."""
        return f'<Dataset(id={self.id}, original_filename={self.original_filename})>'

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'upload_timestamp': self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            'file_size': self.file_size,
            'row_count': self.row_count,
            'column_names': self.column_names,
            'data_types': self.data_types,
            'description': self.description,
            'user_id': self.user_id,
        }


class Analysis(Base):
    """Analysis model representing a statistical analysis run.

    Attributes:
        id: Unique identifier (UUID)
        dataset_id: Foreign key to Dataset
        status: Current status (pending, running, completed, failed)
        selected_columns: Columns selected for analysis (JSON)
        configuration: Analysis configuration parameters (JSON)
        start_time: When analysis started
        end_time: When analysis completed
        result_path: Path to result files
        log_path: Path to execution log
        error_message: Error details if failed
        summary: Brief summary of results
        probabilities: Test p-values (JSON)
        dataset: Related dataset
        scheduled_task: Related scheduled task (if from scheduler)
    """

    __tablename__ = 'analyses'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey('datasets.id'), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False, index=True)
    selected_columns = Column(JSON, nullable=True)  # ["col1", "col3"]
    model_name = Column(String(100), nullable=True)  # AI model used (e.g., 'gpt-4o', 'deepseek-r1:8b')
    provider = Column(String(50), nullable=True)  # Model provider (e.g., 'openai', 'ollama')
    configuration = Column(JSON, nullable=True)  # Analysis parameters
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    result_path = Column(String(500), nullable=True)
    log_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    probabilities = Column(JSON, nullable=True)  # {"test_name": p_value, ...}

    # Relationships
    dataset = relationship('Dataset', back_populates='analyses')
    scheduled_task_id = Column(String(36), ForeignKey('scheduled_tasks.id'), nullable=True)
    scheduled_task = relationship('ScheduledTask', back_populates='analyses')
    owner = relationship('User', back_populates='analyses')

    def __repr__(self) -> str:
        """String representation of Analysis."""
        return f'<Analysis(id={self.id}, status={self.status}, dataset_id={self.dataset_id})>'

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'status': self.status.value if isinstance(self.status, PyEnum) else self.status,
            'selected_columns': self.selected_columns,
            'model_name': self.model_name,
            'provider': self.provider,
            'configuration': self.configuration,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'result_path': self.result_path,
            'log_path': self.log_path,
            'error_message': self.error_message,
            'summary': self.summary,
            'probabilities': self.probabilities,
            'scheduled_task_id': self.scheduled_task_id,
            'user_id': self.user_id,
        }


class ScheduledTask(Base):
    """Scheduled task model for background and recurring analyses.

    Attributes:
        id: Unique identifier (UUID)
        name: User-friendly task name
        task_type: ONE_TIME or RECURRING
        dataset_id: Foreign key to Dataset
        selected_columns: Columns to analyze (JSON)
        configuration: Task configuration (JSON)
        schedule: Cron expression or ISO datetime
        status: Current status (active, paused, completed, failed)
        next_run: Next scheduled execution time
        last_run: Last execution time
        run_count: Number of times executed
        created_at: When task was created
        updated_at: When task was last modified
        dataset: Related dataset
        analyses: Related analysis runs
    """

    __tablename__ = 'scheduled_tasks'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    task_type = Column(Enum(TaskType), nullable=False)
    dataset_id = Column(String(36), ForeignKey('datasets.id'), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    selected_columns = Column(JSON, nullable=True)
    configuration = Column(JSON, nullable=True)
    schedule = Column(String(255), nullable=False)  # Cron or ISO datetime
    status = Column(Enum(TaskStatus), default=TaskStatus.ACTIVE, nullable=False, index=True)
    next_run = Column(DateTime, nullable=True, index=True)
    last_run = Column(DateTime, nullable=True)
    run_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    dataset = relationship('Dataset', back_populates='scheduled_tasks')
    analyses = relationship('Analysis', back_populates='scheduled_task')
    owner = relationship('User', back_populates='scheduled_tasks')

    def __repr__(self) -> str:
        """String representation of ScheduledTask."""
        return f'<ScheduledTask(id={self.id}, name={self.name}, status={self.status})>'

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'task_type': self.task_type.value if isinstance(self.task_type, PyEnum) else self.task_type,
            'dataset_id': self.dataset_id,
            'selected_columns': self.selected_columns,
            'configuration': self.configuration,
            'schedule': self.schedule,
            'status': self.status.value if isinstance(self.status, PyEnum) else self.status,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'run_count': self.run_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'user_id': self.user_id,
        }
