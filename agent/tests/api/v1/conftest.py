"""API v1 specific fixtures."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import ConversationFactory, SessionFactory


@pytest.fixture
def user_id():
    return uuid4()


@pytest.fixture
def active_conversation(db: AsyncSession, user_id):
    from app.schemas.agent import ConversationStatus

    conv = ConversationFactory(user_id=user_id, status=ConversationStatus.ACTIVE)
    return conv


@pytest.fixture
def active_session(db: AsyncSession, active_conversation):
    sess = SessionFactory(conversation=active_conversation, active=True)
    return sess
