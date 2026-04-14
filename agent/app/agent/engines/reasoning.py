"""Health-domain reasoning agent — wraps pygentic-ai BaseAgent."""

from __future__ import annotations

from pydantic_ai import UsageLimits
from pygentic_ai import BaseAgent

from app.agent.prompts.agent_prompts import build_system_prompt
from app.agent.utils.model_utils import get_llm
from app.config import settings
from app.schemas.agent import AgentMode
from app.schemas.language import LANGUAGE_NAMES, Language


class HealthReasoningAgent(BaseAgent):
    """ReAct-style reasoning agent for the Open Wearables health domain.

    Wraps pygentic-ai BaseAgent with health-specific instructions and
    the configured LLM provider from app settings.
    """

    def __init__(
        self,
        mode: AgentMode = AgentMode.GENERAL,
        tools: list | None = None,
        language: Language | None = None,
    ) -> None:
        vendor, model, api_key = get_llm()
        lang_name = LANGUAGE_NAMES[language] if language else LANGUAGE_NAMES[Language.english]
        instructions = build_system_prompt(mode, language)

        super().__init__(
            llm_vendor=vendor,
            llm_model=model,
            api_key=api_key,
            tool_list=tools or [],
            system_prompt=instructions,
            language=lang_name,
            usage_limits=UsageLimits(request_limit=settings.max_tool_calls),
        )
