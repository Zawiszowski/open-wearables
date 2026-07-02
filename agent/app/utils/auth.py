from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.schemas.agent import AccessRole


@dataclass
class Principal:
    subject_id: UUID
    access_role: AccessRole


_bearer_scheme = HTTPBearer(auto_error=False)

_CredentialsDep = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)]

_AUTH_HEADERS = {"WWW-Authenticate": "Bearer"}


def _unauthorized(detail: str = "Could not validate credentials") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail, headers=_AUTH_HEADERS)


class JWTAuth:
    def __init__(self) -> None:
        self._secret_key = settings.secret_key
        self._algorithm = settings.algorithm

    def _decode(self, credentials: HTTPAuthorizationCredentials | None) -> dict:
        if not credentials:
            raise _unauthorized()
        try:
            return jwt.decode(credentials.credentials, self._secret_key, algorithms=[self._algorithm])
        except jwt.ExpiredSignatureError:
            raise _unauthorized("Token has expired")
        except jwt.PyJWTError:
            raise _unauthorized()

    async def validate_token(self, credentials: _CredentialsDep) -> None:
        self._decode(credentials)

    async def get_user_id(self, credentials: _CredentialsDep) -> UUID:
        payload = self._decode(credentials)
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise _unauthorized()
        try:
            return UUID(user_id)
        except ValueError:
            raise _unauthorized()

    async def get_principal(self, credentials: _CredentialsDep) -> Principal:
        payload = self._decode(credentials)
        sub: str | None = payload.get("sub")
        if sub is None:
            raise _unauthorized()
        try:
            subject_id = UUID(sub)
        except ValueError:
            raise _unauthorized()
        # No scope => developer login token => admin. Any scope (incl. "sdk" and
        # unknown scopes) => user, the least-privileged role (fail-closed).
        access_role = AccessRole.ADMIN if payload.get("scope") is None else AccessRole.USER
        return Principal(subject_id=subject_id, access_role=access_role)


jwt_auth = JWTAuth()

ValidToken = Annotated[None, Depends(jwt_auth.validate_token)]
CurrentUserId = Annotated[UUID, Depends(jwt_auth.get_user_id)]
CurrentPrincipal = Annotated[Principal, Depends(jwt_auth.get_principal)]
