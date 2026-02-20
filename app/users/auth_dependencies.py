# app/users/auth_dependencies.py
# Centralized Authentication Dependencies

from typing import Optional
from fastapi import Depends, HTTPException, status, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.users.user_models.user_model import User
from app.users.auth_token_model.token_model import Token
from app.users.security import decode_token

# Security schemes
security_scheme = HTTPBearer(auto_error=False)  # Don't auto-raise for cookie fallback


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token"),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user with PROPER token revocation check.
    
    Accepts token from EITHER:
    - Authorization: Bearer header (for mobile/Postman)
    - httpOnly cookie (for web browsers)
    
    Validates:
    1. JWT signature and expiry
    2. Token exists in database
    3. Token is not revoked (THE CRITICAL FIX)
    4. User exists and is active
    
    Raises 401 if any validation fails.
    """
    # Extract token from header or cookie
    token_string = None
    if credentials:
        token_string = credentials.credentials
    elif access_token_cookie:
        token_string = access_token_cookie
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide token in Authorization header or cookie.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 1. Decode JWT (validates signature + expiry)
    payload = decode_token(token_string)
    if not payload or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 2. Extract user identifier
    user_email = payload.get("sub")
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identifier",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # ═══════════════════════════════════════════════════════════════
    # 3. THE FIX: Check token revocation status in database
    # ═══════════════════════════════════════════════════════════════
    token_record = await db.execute(
        select(Token).where(
            and_(
                Token.token_string == token_string,
                Token.token_type == "access"
            )
        )
    )
    token_obj = token_record.scalars().first()
    
    # Token not in DB → invalid (shouldn't happen but be defensive)
    if not token_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token not found in database. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Token revoked → user logged out
    if token_obj.is_revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # ═══════════════════════════════════════════════════════════════
    
    # 4. Fetch user from database
    result = await db.execute(
        select(User).where(User.email == user_email)
    )
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 5. Check user status
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Optional: Check email verification
    # if not user.is_verified:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Email not verified"
    #     )
    
    return user


async def get_current_user_id(
    current_user: User = Depends(get_current_user)
) -> int:
    """
    Get just the user ID.
    Convenience wrapper around get_current_user.
    """
    return current_user.id


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Require admin role.
    Raises 403 if user is not admin.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token"),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get user if authenticated, otherwise None.
    For endpoints that work with or without auth.
    """
    if not credentials and not access_token_cookie:
        return None
    
    try:
        return await get_current_user(credentials, access_token_cookie, db)
    except HTTPException:
        return None  # Invalid/expired token → treat as anonymous