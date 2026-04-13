"""Tests for POST /api/v1/session and PATCH /api/v1/session/{id} routes."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.agent import ConversationStatus
from tests.factories import ConversationFactory, SessionFactory


class TestCreateOrGetSession:
    def test_creates_new_session_for_new_user(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.post("/api/v1/session", json={}, headers=auth_headers)

        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert "conversation_id" in data
        assert "created_at" in data

    def test_returns_existing_session_when_passed_valid_session_id(
        self,
        client: TestClient,
        auth_headers: dict,
        active_session,
        active_conversation,
        user_id,
    ) -> None:
        response = client.post(
            "/api/v1/session",
            json={"session_id": str(active_session.id)},
            headers=auth_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == str(active_session.id)
        assert data["conversation_id"] == str(active_conversation.id)

    def test_creates_new_session_when_session_id_is_unknown(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.post(
            "/api/v1/session",
            json={"session_id": str(uuid4())},
            headers=auth_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.post("/api/v1/session", json={})

        assert response.status_code == 401


class TestDeactivateSession:
    def test_deactivates_own_session(
        self,
        client: TestClient,
        auth_headers: dict,
        active_session,
    ) -> None:
        response = client.patch(
            f"/api/v1/session/{active_session.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(active_session.id)

    def test_returns_404_for_unknown_session(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.patch(
            f"/api/v1/session/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == 404

    def test_returns_403_for_session_owned_by_other_user(
        self,
        client: TestClient,
        auth_headers: dict,
        db: AsyncSession,
    ) -> None:
        other_conv = ConversationFactory(
            user_id=uuid4(), status=ConversationStatus.ACTIVE
        )
        other_sess = SessionFactory(conversation=other_conv, active=True)

        response = client.patch(
            f"/api/v1/session/{other_sess.id}",
            headers=auth_headers,
        )

        assert response.status_code == 403

    def test_requires_auth(self, client: TestClient, active_session) -> None:
        response = client.patch(f"/api/v1/session/{active_session.id}")

        assert response.status_code == 401
