"""API v1 specific fixtures."""

from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import Session
from app.models.conversation import Conversation
from tests.factories import ConversationFactory, SessionFactory


@pytest.fixture
def user_id() -> UUID:
    return uuid4()


@pytest.fixture
def active_conversation(db: AsyncSession, user_id: UUID) -> Conversation:
    from app.schemas.agent import ConversationStatus

    return ConversationFactory(user_id=user_id, status=ConversationStatus.ACTIVE)


@pytest.fixture
def active_session(db: AsyncSession, active_conversation: Conversation) -> Session:
    return SessionFactory(conversation=active_conversation, active=True)
