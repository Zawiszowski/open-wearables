"""Tool registry — maps AgentMode + AccessRole to the list of tool functions."""

from enum import Enum

from app.agent.tools.date_tools import DATE_TOOLS
from app.agent.tools.ow_tools import OW_ADMIN_TOOLS, OW_SELF_TOOLS
from app.agent.tools.stats_tools import STATS_TOOLS
from app.schemas.agent import AccessRole, AgentMode


class Toolpack(str, Enum):
    OW_DATA = "ow_data"
    OW_ADMIN = "ow_admin"
    DATE_UTILS = "date_utils"
    STATS = "stats"


_TOOLPACKS: dict[Toolpack, list] = {
    Toolpack.OW_DATA: OW_SELF_TOOLS,
    Toolpack.OW_ADMIN: OW_ADMIN_TOOLS,
    Toolpack.DATE_UTILS: DATE_TOOLS,
    Toolpack.STATS: STATS_TOOLS,
}

_MODE_MAPPING: dict[AgentMode, list[Toolpack]] = {
    AgentMode.GENERAL: [Toolpack.OW_DATA, Toolpack.DATE_UTILS, Toolpack.STATS],
}

_ADMIN_TOOLPACKS: list[Toolpack] = [Toolpack.OW_ADMIN]


class ToolManager:
    def get_tools(self, mode: AgentMode, access_role: AccessRole) -> list:
        packs = list(_MODE_MAPPING.get(mode, []))
        if access_role is AccessRole.ADMIN:
            packs += _ADMIN_TOOLPACKS
        tools: list = []
        for pack in packs:
            tools.extend(_TOOLPACKS.get(pack, []))
        return tools


tool_manager = ToolManager()
