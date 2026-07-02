from app.agent.tools.ow_tools import lookup_user
from app.agent.tools.tool_registry import tool_manager
from app.schemas.agent import AccessRole, AgentMode


def test_user_mode_excludes_lookup_user():
    tools = tool_manager.get_tools(AgentMode.GENERAL, AccessRole.USER)
    assert lookup_user not in tools


def test_admin_mode_includes_lookup_user():
    tools = tool_manager.get_tools(AgentMode.GENERAL, AccessRole.ADMIN)
    assert lookup_user in tools


def test_user_mode_still_has_data_tools():
    from app.agent.tools.ow_tools import get_recent_sleep

    tools = tool_manager.get_tools(AgentMode.GENERAL, AccessRole.USER)
    assert get_recent_sleep in tools
