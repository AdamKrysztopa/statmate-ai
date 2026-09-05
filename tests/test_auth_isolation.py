"""Tests that authentication and per-user isolation fail closed.

Before this suite, every data route depended on ``get_current_user_optional`` and
enforced access with a hand-written ``if settings.AUTH_REQUIRED and not current_user``
guard. An anonymous caller therefore reached the service layer with ``user_id=None``,
where the ownership filters were opt-in (``if user_id:``) and were simply skipped —
widening each query to every user's rows.

These tests pin both halves: routes reject anonymous callers, and the service-layer
filters deny rather than widen when no owner is supplied.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from database.models import (
    Analysis,
    AnalysisStatus,
    Base,
    Dataset,
    ScheduledTask,
    TaskType,
    TaskStatus,
    User,
)
from statmate.api.services.analysis_service import AnalysisService
from statmate.api.services.dataset_service import DatasetService
from statmate.api.services.task_service import TaskService


@pytest.fixture()
def db() -> Generator[Session, None, None]:
    """An isolated in-memory database seeded with two users' rows."""
    engine = create_engine(
        'sqlite://',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed(session: Session) -> dict[str, Any]:
    """Create two users, each owning one dataset, analysis and task."""
    created: dict[str, Any] = {}
    for tag in ('alice', 'bob'):
        user = User(
            id=f'user-{tag}',
            email=f'{tag}@example.com',
            email_hash=f'hash-{tag}',
            hashed_password='x',
        )
        dataset = Dataset(
            id=f'ds-{tag}',
            filename=f'{tag}.parquet',
            original_filename=f'{tag}.csv',
            upload_timestamp=datetime.now(UTC),
            file_size=1,
            user_id=user.id,
        )
        analysis = Analysis(
            id=f'an-{tag}',
            dataset_id=dataset.id,
            status=AnalysisStatus.COMPLETED,
            version=1,
            user_id=user.id,
        )
        task = ScheduledTask(
            id=f'tk-{tag}',
            name=f'{tag}-task',
            task_type=TaskType.RECURRING,
            dataset_id=dataset.id,
            schedule='0 0 * * *',
            status=TaskStatus.ACTIVE,
            run_count=0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            user_id=user.id,
        )
        session.add_all([user, dataset, analysis, task])
        created[tag] = {'user': user, 'dataset': dataset, 'analysis': analysis, 'task': task}
    session.commit()
    return created


# --- service layer: no owner must deny, never widen -------------------------------


def test_get_analysis_without_owner_returns_nothing(db: Session) -> None:
    """An ownerless lookup must not resolve another user's analysis."""
    _seed(db)
    assert AnalysisService.get_analysis(db, 'an-alice', user_id=None) is None


def test_list_analyses_without_owner_returns_nothing(db: Session) -> None:
    """An ownerless listing must not enumerate every user's analyses."""
    _seed(db)
    assert AnalysisService.list_analyses(db, user_id=None) == []


def test_get_dataset_without_owner_returns_nothing(db: Session) -> None:
    """An ownerless lookup must not resolve another user's dataset."""
    _seed(db)
    assert DatasetService.get_dataset(db, 'ds-alice', user_id=None) is None


def test_get_task_without_owner_returns_nothing(db: Session) -> None:
    """An ownerless lookup must not resolve another user's scheduled task."""
    _seed(db)
    assert TaskService.get_task(db, 'tk-alice', user_id=None) is None


# --- service layer: cross-tenant access must be denied ----------------------------


def test_user_cannot_read_another_users_analysis(db: Session) -> None:
    """Bob must not resolve Alice's analysis by id."""
    _seed(db)
    assert AnalysisService.get_analysis(db, 'an-alice', user_id='user-bob') is None
    assert AnalysisService.get_analysis(db, 'an-alice', user_id='user-alice') is not None


def test_user_cannot_read_another_users_dataset(db: Session) -> None:
    """Bob must not resolve Alice's dataset by id."""
    _seed(db)
    assert DatasetService.get_dataset(db, 'ds-alice', user_id='user-bob') is None
    assert DatasetService.get_dataset(db, 'ds-alice', user_id='user-alice') is not None


def test_listing_is_scoped_to_the_caller(db: Session) -> None:
    """Each user sees only their own analyses."""
    _seed(db)
    alice = AnalysisService.list_analyses(db, user_id='user-alice')
    assert [a.id for a in alice] == ['an-alice']


# --- the deliberate internal escape hatch -----------------------------------------


def test_unscoped_helpers_are_explicit(db: Session) -> None:
    """Trusted internal lookups exist under names that say so."""
    _seed(db)
    assert AnalysisService.get_analysis_unscoped(db, 'an-alice') is not None
    assert TaskService.get_task_unscoped(db, 'tk-alice') is not None


# --- routes: anonymous callers are rejected ---------------------------------------


@pytest.mark.parametrize(
    'method,path',
    [
        ('get', '/api/v1/analysis/list'),
        ('get', '/api/v1/analysis/an-alice'),
        ('get', '/api/v1/datasets'),
        ('get', '/api/v1/datasets/ds-alice'),
        ('get', '/api/v1/tasks'),
        ('post', '/api/v1/analysis/run'),
    ],
)
def test_data_routes_reject_anonymous_callers(method: str, path: str) -> None:
    """Every data route answers 401 without a bearer token."""
    from fastapi.testclient import TestClient

    from statmate.api.main import app

    client = TestClient(app)
    response = client.request(method, path, json={} if method == 'post' else None)
    assert response.status_code == 401, f'{method.upper()} {path} returned {response.status_code}'


def test_no_route_uses_the_optional_dependency() -> None:
    """Data routes must depend on the enforcing dependency, not the optional one.

    ``get_current_user_optional`` is legitimate only in ``auth.py`` (``/auth/me``
    reports the caller, if any). Anywhere else it reintroduces the ``None`` owner.
    """
    import pathlib

    offenders = [
        path.name
        for path in pathlib.Path('statmate/api/routes').glob('*.py')
        if path.name != 'auth.py' and 'get_current_user_optional' in path.read_text()
    ]
    assert offenders == [], f'optional auth dependency used in: {offenders}'
