"""Task service for managing scheduled tasks."""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from database.models import ScheduledTask, TaskStatus, TaskType
from statmate.api.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


class TaskService:
    """Service for scheduled task management operations."""

    @staticmethod
    def create_task(
        db: Session,
        name: str,
        task_type: str,
        dataset_id: str,
        schedule: str,
        selected_columns: list[str] | None = None,
        configuration: dict | None = None,
    ) -> ScheduledTask:
        """Create a new scheduled task.

        Args:
            db: Database session
            name: Task name
            task_type: 'one_time' or 'recurring'
            dataset_id: UUID of dataset to analyze
            schedule: Cron expression or ISO datetime
            selected_columns: Columns to analyze (None = all)
            configuration: Optional task configuration

        Returns:
            Created ScheduledTask model

        Raises:
            ValueError: If dataset not found or invalid task_type
        """
        # Validate dataset exists
        dataset = DatasetService.get_dataset(db, dataset_id)
        if not dataset:
            msg = f'Dataset not found: {dataset_id}'
            raise ValueError(msg)

        # Validate task type
        try:
            task_type_enum = TaskType(task_type)
        except ValueError as e:
            msg = f'Invalid task_type: {task_type}'
            raise ValueError(msg) from e

        # Parse schedule to determine next_run
        # TODO: Implement proper schedule parsing (cron or ISO datetime)
        next_run = datetime.utcnow()  # Placeholder

        task = ScheduledTask(
            name=name,
            task_type=task_type_enum,
            dataset_id=dataset_id,
            selected_columns=selected_columns,
            configuration=configuration or {},
            schedule=schedule,
            status=TaskStatus.ACTIVE,
            next_run=next_run,
        )

        db.add(task)
        db.commit()
        db.refresh(task)

        logger.info(f'Created scheduled task: {task.id} ({name})')
        return task

    @staticmethod
    def get_task(db: Session, task_id: str) -> ScheduledTask | None:
        """Get a task by ID.

        Args:
            db: Database session
            task_id: Task UUID

        Returns:
            ScheduledTask model or None if not found
        """
        return db.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()

    @staticmethod
    def list_tasks(
        db: Session,
        status: TaskStatus | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ScheduledTask]:
        """List scheduled tasks with optional filtering.

        Args:
            db: Database session
            status: Filter by status (None = all)
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of ScheduledTask models
        """
        query = db.query(ScheduledTask)

        if status:
            query = query.filter(ScheduledTask.status == status)

        return query.order_by(ScheduledTask.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def pause_task(db: Session, task_id: str) -> ScheduledTask:
        """Pause a scheduled task.

        Args:
            db: Database session
            task_id: Task UUID

        Returns:
            Updated ScheduledTask model

        Raises:
            ValueError: If task not found
        """
        task = TaskService.get_task(db, task_id)
        if not task:
            msg = f'Task not found: {task_id}'
            raise ValueError(msg)

        task.status = TaskStatus.PAUSED
        task.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(task)

        logger.info(f'Paused task: {task_id}')
        return task

    @staticmethod
    def resume_task(db: Session, task_id: str) -> ScheduledTask:
        """Resume a paused task.

        Args:
            db: Database session
            task_id: Task UUID

        Returns:
            Updated ScheduledTask model

        Raises:
            ValueError: If task not found
        """
        task = TaskService.get_task(db, task_id)
        if not task:
            msg = f'Task not found: {task_id}'
            raise ValueError(msg)

        task.status = TaskStatus.ACTIVE
        task.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(task)

        logger.info(f'Resumed task: {task_id}')
        return task

    @staticmethod
    def delete_task(db: Session, task_id: str) -> bool:
        """Delete a scheduled task.

        Args:
            db: Database session
            task_id: Task UUID

        Returns:
            True if deleted, False if not found
        """
        task = TaskService.get_task(db, task_id)
        if not task:
            return False

        db.delete(task)
        db.commit()

        logger.info(f'Deleted task: {task_id}')
        return True

    @staticmethod
    def update_task_execution(db: Session, task_id: str, success: bool = True) -> None:
        """Update task after execution.

        Args:
            db: Database session
            task_id: Task UUID
            success: Whether execution was successful
        """
        task = TaskService.get_task(db, task_id)
        if not task:
            return

        task.last_run = datetime.utcnow()
        task.run_count += 1

        # Update status if one-time task
        if task.task_type == TaskType.ONE_TIME:
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED

        # TODO: Calculate next_run for recurring tasks

        task.updated_at = datetime.utcnow()
        db.commit()

        logger.info(f'Updated task execution: {task_id} (success={success})')
