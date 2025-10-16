"""Scheduled job definitions and execution logic."""

import logging
from datetime import datetime

from database.models import TaskStatus
from database.session import SessionLocal
from statmate.api.services.analysis_service import AnalysisService
from statmate.api.services.task_service import TaskService

logger = logging.getLogger(__name__)


def execute_scheduled_analysis(task_id: str) -> None:
    """Execute a scheduled analysis task.

    This function is called by APScheduler to run a scheduled analysis.

    Args:
        task_id: UUID of the scheduled task
    """
    db = SessionLocal()
    logger.info(f'Executing scheduled task: {task_id}')

    try:
        # Get task details
        task = TaskService.get_task(db, task_id)
        if not task:
            logger.error(f'Task not found: {task_id}')
            return

        # Check if task is still active
        if task.status != TaskStatus.ACTIVE:
            logger.info(f'Task {task_id} is not active (status: {task.status}), skipping')
            return

        # Create and run analysis
        analysis = AnalysisService.create_analysis(
            db=db,
            dataset_id=task.dataset_id,
            selected_columns=task.selected_columns,
            configuration=task.configuration,
        )

        # Link analysis to task
        analysis.scheduled_task_id = task_id
        db.commit()

        # Run analysis
        AnalysisService.run_analysis(db, analysis.id)

        # Update task execution record
        TaskService.update_task_execution(db, task_id, success=True)
        logger.info(f'Scheduled task completed successfully: {task_id}')

    except Exception as e:
        logger.error(f'Scheduled task failed: {task_id} - {e}', exc_info=True)
        TaskService.update_task_execution(db, task_id, success=False)

    finally:
        db.close()


def register_task_with_scheduler(task_id: str, schedule: str, task_type: str) -> None:
    """Register a task with APScheduler.

    Args:
        task_id: UUID of the task
        schedule: Cron expression or ISO datetime
        task_type: 'one_time' or 'recurring'
    """
    from statmate.api.scheduler.scheduler import get_scheduler

    scheduler = get_scheduler()

    # Remove existing job if any
    try:
        scheduler.remove_job(task_id)
    except Exception:
        pass

    if task_type == 'one_time':
        # Schedule as a one-time job
        try:
            run_date = datetime.fromisoformat(schedule)
            scheduler.add_job(
                execute_scheduled_analysis,
                trigger='date',
                run_date=run_date,
                args=[task_id],
                id=task_id,
                replace_existing=True,
            )
            logger.info(f'Registered one-time task: {task_id} at {run_date}')
        except ValueError as e:
            logger.error(f'Invalid datetime for one-time task {task_id}: {schedule} - {e}')
            raise

    elif task_type == 'recurring':
        # Schedule as a recurring job using cron
        try:
            scheduler.add_job(
                execute_scheduled_analysis,
                trigger='cron',
                **parse_cron_expression(schedule),
                args=[task_id],
                id=task_id,
                replace_existing=True,
            )
            logger.info(f'Registered recurring task: {task_id} with schedule {schedule}')
        except ValueError as e:
            logger.error(f'Invalid cron expression for recurring task {task_id}: {schedule} - {e}')
            raise


def unregister_task_from_scheduler(task_id: str) -> None:
    """Remove a task from APScheduler.

    Args:
        task_id: UUID of the task
    """
    from statmate.api.scheduler.scheduler import get_scheduler

    try:
        scheduler = get_scheduler()
        scheduler.remove_job(task_id)
        logger.info(f'Unregistered task from scheduler: {task_id}')
    except Exception as e:
        logger.warning(f'Could not unregister task {task_id}: {e}')


def parse_cron_expression(cron_str: str) -> dict[str, str]:
    """Parse cron expression into APScheduler parameters.

    Args:
        cron_str: Cron expression (e.g., '0 2 * * *')

    Returns:
        Dictionary of cron parameters

    Example:
        '0 2 * * *' -> {'minute': '0', 'hour': '2', 'day': '*', 'month': '*', 'day_of_week': '*'}
    """
    parts = cron_str.split()
    if len(parts) != 5:
        msg = f'Invalid cron expression: {cron_str}. Expected 5 parts (minute hour day month day_of_week)'
        raise ValueError(msg)

    return {
        'minute': parts[0],
        'hour': parts[1],
        'day': parts[2],
        'month': parts[3],
        'day_of_week': parts[4],
    }
