"""Tests for POST /api/v1/chat/{session_id} route."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.schemas.agent import ConversationStatus
from tests.factories import ConversationFactory, SessionFactory


CALLBACK_URL = "https://example.com/callback"


class TestSendMessage:
    def test_queues_task_and_returns_task_id(
        self,
        client: TestClient,
        auth_headers: dict,
        active_session,
        mock_celery: MagicMock,
    ) -> None:
        response = client.post(
            f"/api/v1/chat/{active_session.id}",
            json={"message": "How was my sleep?", "callback_url": CALLBACK_URL},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_id"] == "test-task-id"

    def test_calls_process_message_delay_with_correct_args(
        self,
        client: TestClient,
        auth_headers: dict,
        active_session,
        active_conversation,
        mock_celery: MagicMock,
    ) -> None:
        client.post(
            f"/api/v1/chat/{active_session.id}",
            json={"message": "Steps today?", "callback_url": CALLBACK_URL},
            headers=auth_headers,
        )

        mock_celery.delay.assert_called_once()
        call_kwargs = mock_celery.delay.call_args.kwargs
        assert call_kwargs["message"] == "Steps today?"
        assert call_kwargs["session_id"] == str(active_session.id)
        assert call_kwargs["conversation_id"] == str(active_conversation.id)

    def test_returns_404_for_unknown_session(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.post(
            f"/api/v1/chat/{uuid4()}",
            json={"message": "Hello", "callback_url": CALLBACK_URL},
            headers=auth_headers,
        )

        assert response.status_code == 404

    def test_returns_410_for_closed_conversation(
        self,
        client: TestClient,
        auth_headers: dict,
        user_id,
        db,
    ) -> None:
        closed_conv = ConversationFactory(
            user_id=user_id, status=ConversationStatus.CLOSED
        )
        sess = SessionFactory(conversation=closed_conv, active=True)

        response = client.post(
            f"/api/v1/chat/{sess.id}",
            json={"message": "Hello", "callback_url": CALLBACK_URL},
            headers=auth_headers,
        )

        assert response.status_code == 410

    def test_returns_410_for_inactive_session(
        self,
        client: TestClient,
        auth_headers: dict,
        user_id,
        db,
    ) -> None:
        conv = ConversationFactory(
            user_id=user_id, status=ConversationStatus.ACTIVE
        )
        inactive_sess = SessionFactory(conversation=conv, active=False)

        response = client.post(
            f"/api/v1/chat/{inactive_sess.id}",
            json={"message": "Hello", "callback_url": CALLBACK_URL},
            headers=auth_headers,
        )

        assert response.status_code == 410

    def test_rejects_empty_message(
        self,
        client: TestClient,
        auth_headers: dict,
        active_session,
    ) -> None:
        response = client.post(
            f"/api/v1/chat/{active_session.id}",
            json={"message": "", "callback_url": CALLBACK_URL},
            headers=auth_headers,
        )

        assert response.status_code == 422

    def test_requires_auth(self, client: TestClient, active_session) -> None:
        response = client.post(
            f"/api/v1/chat/{active_session.id}",
            json={"message": "Hello", "callback_url": CALLBACK_URL},
        )

        assert response.status_code == 401
