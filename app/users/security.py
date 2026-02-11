# app/users/security.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config.appconfig import settings
from app.helpers.time import utcnow
from jose import JWTError, jwt
import secrets
import random


# Password hashing context
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


# ============================================================
# ✅ Verify Password
# ============================================================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)




# ============================================================
# ✅ Get Password Hash
# ============================================================
def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)



# ============================================================
# ✅ Create Access Token
# ============================================================
async def create_access_token(
    data: Dict[str, Any], 
    db: AsyncSession,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token and store it as active."""
    to_encode = data.copy()

    if expires_delta:
        expire = utcnow() + expires_delta
    else:
        expire = utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRY)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    # Store as active token
    from app.system_models.token_model.token_model import Token
    token = Token(
        token_string=encoded_jwt,
        token_type="access",
        user_id=data.get("user_id"),
        expires_at=expire
    )
    db.add(token)
    await db.commit()   
    
    return encoded_jwt




# ============================================================
# ✅ Create Refresh Token
# ============================================================
async def create_refresh_token(
    data: Dict[str, Any], 
    db: AsyncSession,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT refresh token and store it as active."""
    to_encode = data.copy()

    if expires_delta:
        expire = utcnow() + expires_delta
    else:
        expire = utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRY)

    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    # Store token
    from app.system_models.token_model.token_model import Token
    token = Token(
        token_string=encoded_jwt,
        token_type="refresh",
        user_id=data.get("user_id"),
        expires_at=expire
    )
    db.add(token)
    await db.commit()   

    
    return encoded_jwt



# ============================================================
# ✅ Decode Token
# ============================================================
def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None
    
    
# ============================================================
# ✅ Generate Password Reset Token
# ============================================================
def generate_password_reset_token() -> str:
    """Generate a secure random token for password reset."""
    return secrets.token_urlsafe(32)



# ============================================================
# ✅ Get Token Expiry
# ============================================================
def get_token_expiry(token_type: str = "access") -> datetime:
    """Get expiry datetime for a token."""
    if token_type == "refresh":
        return utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRY)
    return utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRY)



# ============================================================
# ✅ Generate Verification Code
# ============================================================
def generate_verification_code():
    """Generate a random 6-digit verification code."""
    return str(random.randint(100000, 999999))