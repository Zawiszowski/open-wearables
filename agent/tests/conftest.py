"""
Main pytest configuration for Open Wearables Agent tests.

Follows the same patterns as the backend test suite:
- PostgreSQL via testcontainers (or TEST_DATABASE_URL)
- Per-test async transaction rollback
- FastAPI TestClient with dependency overrides
- Autouse mocks for Celery, LLM, and external HTTP
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from jose import jwt
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import BaseDbModel, _get_async_db_dependency
from tests import factories

# Ensure env is configured before importing app modules
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _postgres_url() -> Generator[str, None, None]:
    """Provide a PostgreSQL connection URL for tests."""
    explicit_url = os.environ.get("TEST_DATABASE_URL")
    if explicit_url:
        yield explicit_url
        return

    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        image="postgres:16",
        username="open-wearables",
        password="open-wearables",
        dbname="agent_test",
        driver="psycopg",
    ) as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="session")
def async_engine(_postgres_url: str) -> Any:
    """Create async test engine and schema."""
    engine = create_async_engine(_postgres_url, pool_pre_ping=True)
    return engine


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy for session-scoped async fixtures."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(scope="session")
async def _create_schema(async_engine: Any) -> AsyncGenerator[None, None]:
    """Create all tables once per session."""
    async with async_engine.begin() as conn:
        await conn.run_sync(BaseDbModel.metadata.create_all)
    yield
    async with async_engine.begin() as conn:
        await conn.run_sync(BaseDbModel.metadata.drop_all)


@pytest_asyncio.fixture
async def db(_create_schema: None, async_engine: Any) -> AsyncGenerator[AsyncSession, None]:
    """
    Per-test async session with savepoint-based rollback.
    Each test sees a clean slate without touching committed data.
    """
    async with async_engine.connect() as conn:
        await conn.begin()
        nested = await conn.begin_nested()

        session_factory = async_sessionmaker(conn, class_=AsyncSession, expire_on_commit=False)
        session = session_factory()

        @event.listens_for(session.sync_session, "after_transaction_end")
        def restart_savepoint(session: Any, transaction: Any) -> None:
            nonlocal nested
            if not nested._transaction.is_active:
                import asyncio
                nested = asyncio.get_event_loop().run_until_complete(conn.begin_nested())

        try:
            yield session
        finally:
            await session.close()
            await conn.rollback()


@pytest.fixture(autouse=True)
def set_factory_session(db: AsyncSession) -> Generator[None, None, None]:
    """Inject the per-test DB session into all factories."""
    for name in dir(factories):
        obj = getattr(factories, name)
        if isinstance(obj, type) and hasattr(obj, "_meta") and hasattr(obj._meta, "sqlalchemy_session"):
            obj._meta.sqlalchemy_session = db
    yield
    for name in dir(factories):
        obj = getattr(factories, name)
        if isinstance(obj, type) and hasattr(obj, "_meta") and hasattr(obj._meta, "sqlalchemy_session"):
            obj._meta.sqlalchemy_session = None


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def client(db: AsyncSession) -> Generator[TestClient, None, None]:
    """TestClient with async DB dependency overridden."""
    from app.main import api

    async def override_db() -> AsyncGenerator[AsyncSession, None]:
        yield db

    api.dependency_overrides[_get_async_db_dependency] = override_db

    with TestClient(api) as c:
        yield c

    api.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def user_id() -> Any:
    return uuid4()


@pytest.fixture
def auth_token(user_id: Any) -> str:
    """Generate a valid JWT for the test user."""
    payload = {
        "sub": str(user_id),
        "exp": int((datetime(2099, 1, 1, tzinfo=timezone.utc)).timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


@pytest.fixture
def auth_headers(auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_token}"}


# ---------------------------------------------------------------------------
# Global autouse mocks
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_celery(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Prevent Celery tasks from actually dispatching."""
    mock_task = MagicMock()
    mock_task.delay.return_value = MagicMock(id="test-task-id")
    mock_task.apply_async.return_value = MagicMock(id="test-task-id")

    with patch("app.integrations.celery.tasks.process_message.process_message", mock_task):
        yield mock_task


@pytest.fixture(autouse=True)
def mock_llm() -> Generator[dict[str, MagicMock], None, None]:
    """Mock all pydantic-ai Agent.run calls to avoid real LLM calls."""
    mock_run_result = MagicMock()
    mock_run_result.data = "This is a test assistant response."

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_run_result)

    with patch("app.agent.engines.reasoning.Agent", return_value=mock_agent) as mock_reasoning, \
         patch("app.agent.engines.router.Agent", return_value=mock_agent) as mock_router, \
         patch("app.agent.engines.guardrails.Agent", return_value=mock_agent) as mock_guardrails:
        yield {
            "agent": mock_agent,
            "reasoning": mock_reasoning,
            "router": mock_router,
            "guardrails": mock_guardrails,
            "run_result": mock_run_result,
        }


@pytest.fixture(autouse=True)
def mock_ow_client() -> Generator[MagicMock, None, None]:
    """Mock the OW backend REST client."""
    with patch("app.integrations.ow_backend.client.ow_client") as mock:
        mock.get_user_profile = AsyncMock(return_value={"id": str(uuid4()), "first_name": "Test"})
        mock.get_body_summary = AsyncMock(return_value={"slow_changing": {}, "averaged": {}})
        mock.get_activity_summaries = AsyncMock(return_value={"data": []})
        mock.get_sleep_summaries = AsyncMock(return_value={"data": []})
        mock.get_recovery_summaries = AsyncMock(return_value={"data": []})
        mock.get_workout_events = AsyncMock(return_value={"data": []})
        mock.get_sleep_events = AsyncMock(return_value={"data": []})
        mock.get_timeseries = AsyncMock(return_value={"data": []})
        yield mock
