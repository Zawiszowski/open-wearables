"""Tests for WorkflowEngine pipeline (router → reasoning → guardrails)."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.agent.workflows.agent_workflow import WorkflowEngine
from app.agent.static.default_msgs import GUARDRAILS_REFUSAL_MSG, WORKFLOW_ERROR_MSG


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def answer_decision():
    decision = MagicMock()
    decision.route = "answer"
    decision.reasoning = "health question"
    return decision


@pytest.fixture
def refuse_decision():
    decision = MagicMock()
    decision.route = "refuse"
    decision.reasoning = "off-topic"
    return decision


class TestWorkflowEngineRun:
    async def test_full_pipeline_returns_formatted_response(
        self, engine: WorkflowEngine, answer_decision: MagicMock
    ) -> None:
        router_result = MagicMock()
        router_result.data = answer_decision

        reasoning_result = MagicMock()
        reasoning_result.data = "Raw response from reasoning agent"

        guardrails_result = MagicMock()
        guardrails_result.data = "Polished response"

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            side_effect=[router_result, reasoning_result, guardrails_result]
        )

        with patch("app.agent.engines.reasoning.Agent", return_value=mock_agent), \
             patch("app.agent.engines.router.Agent", return_value=mock_agent), \
             patch("app.agent.engines.guardrails.Agent", return_value=mock_agent):
            result = await engine.run(
                user_id=uuid4(),
                message="How was my sleep last week?",
                history=[],
            )

        assert result == "Polished response"

    async def test_returns_refusal_message_when_router_refuses(
        self, engine: WorkflowEngine, refuse_decision: MagicMock
    ) -> None:
        router_result = MagicMock()
        router_result.data = refuse_decision

        mock_router_agent = MagicMock()
        mock_router_agent.run = AsyncMock(return_value=router_result)

        with patch("app.agent.engines.router.Agent", return_value=mock_router_agent):
            result = await engine.run(
                user_id=uuid4(),
                message="What is the capital of France?",
                history=[],
            )

        assert result == GUARDRAILS_REFUSAL_MSG

    async def test_router_failure_defaults_to_answer(
        self, engine: WorkflowEngine
    ) -> None:
        """When router raises, the pipeline falls through to reasoning."""
        mock_router_agent = MagicMock()
        mock_router_agent.run = AsyncMock(side_effect=Exception("LLM timeout"))

        reasoning_result = MagicMock()
        reasoning_result.data = "Reasoning response"

        guardrails_result = MagicMock()
        guardrails_result.data = "Final response"

        mock_reasoning_agent = MagicMock()
        mock_reasoning_agent.run = AsyncMock(
            side_effect=[reasoning_result, guardrails_result]
        )

        with patch("app.agent.engines.router.Agent", return_value=mock_router_agent), \
             patch("app.agent.engines.reasoning.Agent", return_value=mock_reasoning_agent), \
             patch("app.agent.engines.guardrails.Agent", return_value=mock_reasoning_agent):
            result = await engine.run(
                user_id=uuid4(),
                message="How many steps did I walk?",
                history=[],
            )

        assert result == "Final response"

    async def test_guardrails_failure_returns_raw_reasoning_output(
        self, engine: WorkflowEngine, answer_decision: MagicMock
    ) -> None:
        router_result = MagicMock()
        router_result.data = answer_decision

        reasoning_result = MagicMock()
        reasoning_result.data = "Raw reasoning output"

        mock_router_agent = MagicMock()
        mock_router_agent.run = AsyncMock(return_value=router_result)

        mock_reasoning_agent = MagicMock()
        mock_reasoning_agent.run = AsyncMock(return_value=reasoning_result)

        mock_guardrails_agent = MagicMock()
        mock_guardrails_agent.run = AsyncMock(side_effect=Exception("guardrails down"))

        with patch("app.agent.engines.router.Agent", return_value=mock_router_agent), \
             patch("app.agent.engines.reasoning.Agent", return_value=mock_reasoning_agent), \
             patch("app.agent.engines.guardrails.Agent", return_value=mock_guardrails_agent):
            result = await engine.run(
                user_id=uuid4(),
                message="Test message",
                history=[],
            )

        assert result == "Raw reasoning output"

    async def test_reasoning_failure_raises(
        self, engine: WorkflowEngine, answer_decision: MagicMock
    ) -> None:
        router_result = MagicMock()
        router_result.data = answer_decision

        mock_router_agent = MagicMock()
        mock_router_agent.run = AsyncMock(return_value=router_result)

        mock_reasoning_agent = MagicMock()
        mock_reasoning_agent.run = AsyncMock(side_effect=RuntimeError("API error"))

        with patch("app.agent.engines.router.Agent", return_value=mock_router_agent), \
             patch("app.agent.engines.reasoning.Agent", return_value=mock_reasoning_agent):
            with pytest.raises(RuntimeError, match="API error"):
                await engine.run(
                    user_id=uuid4(),
                    message="Test",
                    history=[],
                )

    async def test_history_is_passed_to_reasoning_agent(
        self, engine: WorkflowEngine, answer_decision: MagicMock
    ) -> None:
        router_result = MagicMock()
        router_result.data = answer_decision

        reasoning_result = MagicMock()
        reasoning_result.data = "Answer"

        guardrails_result = MagicMock()
        guardrails_result.data = "Polished"

        mock_router_agent = MagicMock()
        mock_router_agent.run = AsyncMock(return_value=router_result)

        mock_reasoning_agent = MagicMock()
        mock_reasoning_agent.run = AsyncMock(
            side_effect=[reasoning_result, guardrails_result]
        )

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        with patch("app.agent.engines.router.Agent", return_value=mock_router_agent), \
             patch("app.agent.engines.reasoning.Agent", return_value=mock_reasoning_agent), \
             patch("app.agent.engines.guardrails.Agent", return_value=mock_reasoning_agent):
            await engine.run(
                user_id=uuid4(),
                message="Follow-up question",
                history=history,
            )

        reasoning_call = mock_reasoning_agent.run.call_args_list[0]
        assert "message_history" in reasoning_call.kwargs


class TestWorkflowEngineSummarize:
    async def test_returns_summary_string(self, engine: WorkflowEngine) -> None:
        summary_result = MagicMock()
        summary_result.data = "Summary of the conversation."

        mock_summarizer = MagicMock()
        mock_summarizer.run = AsyncMock(return_value=summary_result)

        with patch("app.agent.workflows.agent_workflow.Agent", return_value=mock_summarizer):
            result = await engine.summarize(
                [
                    {"role": "user", "content": "How many steps?"},
                    {"role": "assistant", "content": "You walked 8000 steps."},
                ]
            )

        assert result == "Summary of the conversation."

    async def test_formats_transcript_correctly(self, engine: WorkflowEngine) -> None:
        summary_result = MagicMock()
        summary_result.data = "OK"

        mock_summarizer = MagicMock()
        mock_summarizer.run = AsyncMock(return_value=summary_result)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        with patch("app.agent.workflows.agent_workflow.Agent", return_value=mock_summarizer):
            await engine.summarize(messages)

        prompt = mock_summarizer.run.call_args[0][0]
        assert "USER: Hello" in prompt
        assert "ASSISTANT: Hi there" in prompt
