"""Scheduled task management API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.models import TaskStatus
from database.session import get_db
from statmate.api.models.task import (
    TaskCreate,
    TaskPauseResponse,
    TaskResponse,
)
from statmate.api.services.dataset_service import DatasetService
from statmate.api.services.task_service import TaskService

router = APIRouter(prefix='/tasks', tags=['tasks'])


@router.post('/schedule', response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def schedule_task(
    request: TaskCreate,
    db: Session = Depends(get_db),
) -> TaskResponse:
    """Schedule a new analysis task.

    Args:
        request: Task creation parameters
        db: Database session

    Returns:
        TaskResponse with task details

    Raises:
        HTTPException: If dataset not found or validation fails
    """
    try:
        task = TaskService.create_task(
            db=db,
            name=request.name,
            task_type=request.task_type,
            dataset_id=request.dataset_id,
            schedule=request.schedule,
            selected_columns=request.selected_columns,
            configuration=request.configuration,
        )

        # Get dataset name for response
        dataset = DatasetService.get_dataset(db, task.dataset_id)
        dataset_name = dataset.original_filename if dataset else None

        response_dict = task.to_dict()
        response_dict['dataset_name'] = dataset_name

        return TaskResponse(**response_dict)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to create task: {str(e)}'
        )


@router.get('/', response_model=list[TaskResponse])
async def list_tasks(
    status_filter: str | None = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[TaskResponse]:
    """List scheduled tasks with optional filtering.

    Args:
        status_filter: Filter by status (active, paused, completed, failed)
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of TaskResponse objects

    Raises:
        HTTPException: If invalid status filter
    """
    status_enum = None
    if status_filter:
        try:
            status_enum = TaskStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid status: {status_filter}',
            )

    tasks = TaskService.list_tasks(db, status=status_enum, skip=skip, limit=limit)

    # Enhance with dataset names
    responses = []
    for task in tasks:
        dataset = DatasetService.get_dataset(db, task.dataset_id)
        task_dict = task.to_dict()
        task_dict['dataset_name'] = dataset.original_filename if dataset else None
        responses.append(TaskResponse(**task_dict))

    return responses


@router.get('/{task_id}', response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> TaskResponse:
    """Get a specific task by ID.

    Args:
        task_id: Task UUID
        db: Database session

    Returns:
        TaskResponse object

    Raises:
        HTTPException: If task not found
    """
    task = TaskService.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Task not found')

    dataset = DatasetService.get_dataset(db, task.dataset_id)
    task_dict = task.to_dict()
    task_dict['dataset_name'] = dataset.original_filename if dataset else None

    return TaskResponse(**task_dict)


@router.put('/{task_id}/pause', response_model=TaskPauseResponse)
async def pause_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> TaskPauseResponse:
    """Pause a scheduled task.

    Args:
        task_id: Task UUID
        db: Database session

    Returns:
        TaskPauseResponse

    Raises:
        HTTPException: If task not found
    """
    try:
        task = TaskService.pause_task(db, task_id)
        return TaskPauseResponse(
            task_id=task.id,
            status=task.status.value,
            message='Task paused successfully',
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put('/{task_id}/resume', response_model=TaskPauseResponse)
async def resume_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> TaskPauseResponse:
    """Resume a paused task.

    Args:
        task_id: Task UUID
        db: Database session

    Returns:
        TaskPauseResponse

    Raises:
        HTTPException: If task not found
    """
    try:
        task = TaskService.resume_task(db, task_id)
        return TaskPauseResponse(
            task_id=task.id,
            status=task.status.value,
            message='Task resumed successfully',
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete('/{task_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: str,
    db: Session = Depends(get_db),
) -> None:
    """Delete a scheduled task.

    Args:
        task_id: Task UUID
        db: Database session

    Raises:
        HTTPException: If task not found
    """
    deleted = TaskService.delete_task(db, task_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Task not found')
