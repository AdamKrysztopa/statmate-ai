"""APScheduler configuration and management."""

import logging

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from config.settings import settings

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    """Get the global scheduler instance.

    Returns:
        BackgroundScheduler instance

    Raises:
        RuntimeError: If scheduler not initialized
    """
    if _scheduler is None:
        msg = 'Scheduler not initialized. Call init_scheduler() first.'
        raise RuntimeError(msg)
    return _scheduler


def init_scheduler() -> BackgroundScheduler:
    """Initialize and start the APScheduler.

    Returns:
        BackgroundScheduler instance
    """
    global _scheduler

    if _scheduler is not None:
        logger.warning('Scheduler already initialized')
        return _scheduler

    # Configure job stores
    jobstores = {}
    if settings.SCHEDULER_JOBSTORE == 'sqlite':
        # Use SQLite for persistent job storage
        jobstores['default'] = SQLAlchemyJobStore(url=settings.DATABASE_URL)
    else:
        # Use in-memory storage (jobs lost on restart)
        jobstores['default'] = MemoryJobStore()

    # Configure executors
    executors = {
        'default': ThreadPoolExecutor(max_workers=5),
    }

    # Job defaults
    job_defaults = {
        'coalesce': True,  # Combine multiple missed runs into one
        'max_instances': 3,  # Max concurrent instances of same job
        'misfire_grace_time': 300,  # 5 minutes grace period for missed jobs
    }

    # Create scheduler
    _scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=settings.SCHEDULER_TIMEZONE,
    )

    # Start scheduler
    _scheduler.start()
    logger.info('APScheduler started successfully')

    return _scheduler


def shutdown_scheduler() -> None:
    """Shutdown the APScheduler gracefully."""
    global _scheduler

    if _scheduler is None:
        return

    _scheduler.shutdown(wait=True)
    _scheduler = None
    logger.info('APScheduler shut down')
