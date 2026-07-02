from uuid import uuid4

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.config import settings
from app.schemas.agent import AccessRole
from app.utils.auth import jwt_auth


def _creds(claims: dict) -> HTTPAuthorizationCredentials:
    token = jwt.encode(claims, settings.secret_key, algorithm=settings.algorithm)
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


async def test_sdk_scope_is_user_role() -> None:
    uid = uuid4()
    principal = await jwt_auth.get_principal(_creds({"sub": str(uid), "scope": "sdk"}))
    assert principal.subject_id == uid
    assert principal.access_role is AccessRole.USER


async def test_no_scope_is_admin_role() -> None:
    did = uuid4()
    principal = await jwt_auth.get_principal(_creds({"sub": str(did)}))
    assert principal.subject_id == did
    assert principal.access_role is AccessRole.ADMIN


async def test_unknown_scope_falls_closed_to_user() -> None:
    principal = await jwt_auth.get_principal(_creds({"sub": str(uuid4()), "scope": "something-else"}))
    assert principal.access_role is AccessRole.USER


async def test_missing_sub_raises_401() -> None:
    with pytest.raises(HTTPException) as exc:
        await jwt_auth.get_principal(_creds({"scope": "sdk"}))
    assert exc.value.status_code == 401
