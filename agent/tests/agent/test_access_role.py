from app.schemas.agent import AccessRole


def test_access_role_values() -> None:
    assert AccessRole.USER.value == "user"
    assert AccessRole.ADMIN.value == "admin"


def test_access_role_is_str_enum() -> None:
    # Must serialize cleanly as a Celery task arg / JSON
    assert AccessRole("user") is AccessRole.USER
    assert str(AccessRole.ADMIN) == "admin"
