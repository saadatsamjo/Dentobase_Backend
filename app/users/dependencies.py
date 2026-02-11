# app/users/dependencies.py
from typing import Optional


from app.users.security import decode_token, verify_password
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config.appconfig import settings
from app.database.connection import get_db
from app.helpers.time import utcnow
from app.users.user_models.user_model import User

security_scheme = HTTPBearer()


# ===========================================
# ✅ Get Current User Using Access Token
# ===========================================
async def get_current_user(
    token: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
)-> User: 
    payload = decode_token(token.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user_email = payload.get("sub")
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
        
    return user


# # ===========================================
# # ✅ Get Refresh Token User
# # ===========================================
# async def get_refresh_token_user(
#     token: HTTPAuthorizationCredentials = Depends(security_scheme),
#     db: AsyncSession = Depends(get_db)
# ) -> User:
#     # This dependency is slightly redundant if we pass refresh token string in body,
#     # but valid if we pass it in Authorization header.
#     # The auth service `refresh_access_token` takes a string. 
#     # Usually refresh token endpoint takes token in body.
#     # However, if we want to use this dependency, we assume it's in the header.
#     # But for typical OAuth2 flow, refresh token is often a POST parameter.
#     # Let's support both or stick to standard.
#     # Given the schemas, `refresh_access_token` in services takes `refresh_token: str`.
#     # Let's assume we decode it here just like access token if provided in header?
#     # Or maybe this dependency is for specific routes.
    
#     payload = decode_token(token.credentials)
#     if not payload or payload.get("type") != "refresh":
#          raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid refresh token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     user_email = payload.get("sub")
#     if not user_email:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Could not validate credentials",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
        
#     result = await db.execute(select(User).where(User.email == user_email))
#     user = result.scalars().first()
    
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="User not found",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
        
#     return user