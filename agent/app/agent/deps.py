"""Agent dependency types for pydantic-ai RunContext injection."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pygentic_ai.engines.base import BaseAgentDeps

from app.schemas.agent import AccessRole


@dataclass
class HealthAgentDeps(BaseAgentDeps):
    """Dependencies injected into every tool call via RunContext.

    Extends BaseAgentDeps (which carries language) with the resolved
    user_id and access_role so tools can enforce the user/admin boundary
    without the model supplying either.
    """

    user_id: UUID | None = None
    access_role: AccessRole = AccessRole.USER
