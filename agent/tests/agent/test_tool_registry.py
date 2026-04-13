"""Tests for ToolManager / tool registry."""

import pytest

from app.agent.tools.tool_registry import ToolManager, tool_manager
from app.schemas.agent import AgentMode


class TestToolManager:
    def test_general_mode_includes_ow_tools(self) -> None:
        tools = tool_manager.get_tools_for_mode(AgentMode.GENERAL)

        tool_names = [t.__name__ for t in tools]
        assert "get_user_profile" in tool_names
        assert "get_recent_activity" in tool_names
        assert "get_recent_sleep" in tool_names
        assert "get_recovery_data" in tool_names
        assert "get_workouts" in tool_names

    def test_general_mode_includes_date_tools(self) -> None:
        tools = tool_manager.get_tools_for_mode(AgentMode.GENERAL)

        tool_names = [t.__name__ for t in tools]
        assert "get_today_date" in tool_names
        assert "get_current_week" in tool_names

    def test_returns_list(self) -> None:
        tools = tool_manager.get_tools_for_mode(AgentMode.GENERAL)
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_all_tools_are_callable(self) -> None:
        tools = tool_manager.get_tools_for_mode(AgentMode.GENERAL)
        for tool in tools:
            assert callable(tool)
