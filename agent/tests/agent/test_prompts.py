from app.agent.prompts.agent_prompts import build_system_prompt
from app.agent.prompts.worker_prompts import WorkerType, build_worker_prompt
from app.schemas.agent import AccessRole, AgentMode


def test_user_system_prompt_has_no_cross_user_language():
    prompt = build_system_prompt(AgentMode.GENERAL, None, AccessRole.USER)
    lowered = prompt.lower()
    assert "lookup_user" not in lowered
    assert "target_user_id" not in lowered
    assert "any other platform user" not in lowered


def test_admin_system_prompt_has_cross_user_language():
    prompt = build_system_prompt(AgentMode.GENERAL, None, AccessRole.ADMIN)
    assert "lookup_user" in prompt


def test_user_router_prompt_omits_cross_user_bullet():
    prompt = build_worker_prompt(WorkerType.ROUTER, access_role=AccessRole.USER)
    assert "another user" not in prompt.lower()


def test_admin_router_prompt_includes_cross_user_bullet():
    prompt = build_worker_prompt(WorkerType.ROUTER, access_role=AccessRole.ADMIN)
    assert "another user" in prompt.lower()
