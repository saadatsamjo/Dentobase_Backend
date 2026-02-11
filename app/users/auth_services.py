from fastapi import Depends, HTTPException, status
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.users.user_models.schemas import UserRegisterResponse, UserLogin, UserResponse, UserRegister
from app.users.user_models.user_model import User
from app.database.connection import get_db
from app.users.security import (
    get_password_hash, 
    verify_password, 
    create_access_token, 
    create_refresh_token, 
    generate_password_reset_token,
    decode_token,
    generate_verification_code,
    get_password_hash
)
from app.users.auth_token_model.token_model import Token
from app.users.password_reset_token.password_reset_token_model import PasswordResetToken
from app.users.auth_emails import send_registration_email_with_verification_code, send_reset_password_link_with_token_in_email
from config.appconfig import settings
from app.helpers.time import utcnow
from datetime import timedelta


# ============================================================
# ✅ REGISTER A NEW USER
# ============================================================

async def registering_user (user_data: UserRegister, db: AsyncSession = Depends(get_db)) -> UserRegisterResponse:
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Hash password
    hashed_password = get_password_hash(user_data.password)
    
    # Create new user
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        is_active=user_data.is_active,
        is_verified=user_data.is_verified,
        role=user_data.role,
        verification_code=generate_verification_code() # Generate verification code on signup
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Send verification email
    try:
        send_registration_email_with_verification_code(new_user.email, new_user.verification_code, new_user.first_name)
    except Exception as e:
        print(f"Error sending registration email: {e}")
    
    return new_user


# ============================================================
# ✅ AUTHENTICATE USER
# ============================================================
async def authenticate_user(
    email: str, password: str, db: AsyncSession
) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    
    if not user:
        return None
        
    if not verify_password(password, user.hashed_password):
        return None
        
    return user


# ============================================================
# ✅ LOGIN USER
# ============================================================
async def login_user(user_data: UserLogin, db: AsyncSession) -> tuple[str, str, UserResponse]:
    user = await authenticate_user(user_data.email, user_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    # Create tokens
    access_token = await create_access_token(
        data={"sub": user.email, "user_id": user.id, "role": user.role}, 
        db=db
    )
    refresh_token = await create_refresh_token(
        data={"sub": user.email, "user_id": user.id, "role": user.role}, 
        db=db
    )
    
    return access_token, refresh_token, user


# ============================================================
# ✅ REFRESH ACCESS TOKEN
# ============================================================
async def refresh_access_token(
    refresh_token: str, db: AsyncSession
) -> tuple[str, str]:
    # Verify refresh token
    payload = decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user_email = payload.get("sub")
    if not user_email:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if token exists and is valid in DB (optional but recommended)
    result = await db.execute(select(Token).where(Token.token_string == refresh_token))
    stored_token = result.scalars().first()
    
    if not stored_token or stored_token.is_revoked:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user to ensure they still exist/active
    result = await db.execute(select(User).where(User.email == user_email))
    user = result.scalars().first()
    
    if not user or not user.is_active:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User inactive or not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create new access token
    access_token = await create_access_token(
        data={"sub": user.email, "user_id": user.id, "role": user.role}, 
        db=db
    )
    
    # Optionally rotate refresh token
    new_refresh_token = await create_refresh_token(
        data={"sub": user.email, "user_id": user.id, "role": user.role}, 
        db=db
    )
    
    # Revoke old refresh token
    stored_token.is_revoked = True
    await db.commit()
    
    return access_token, new_refresh_token


# ============================================================
# ✅ LOGOUT USER (Global Revocation)
# ============================================================
async def logout_user(user: User, db: AsyncSession) -> None:
    # Revoke all tokens (access and refresh) for this user
    from sqlalchemy import update
    
    await db.execute(
        update(Token)
        .where(Token.user_id == user.id)
        .values(is_revoked=True)
    )
    await db.commit()


# ============================================================
# ✅ CREATE PASSWORD RESET LINK WITH THE RESET TOKEN ON IT
# ============================================================
async def create_password_reset_link(
    email: str, db: AsyncSession
) -> tuple[str, str | None]:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    
    if not user:
        # Don't reveal user existence
        return "If your email is registered, you will receive a password reset link.", None
        
    reset_token = generate_password_reset_token()
    
    
    # Store token in DB
    
    expires_at = utcnow() + timedelta(minutes=15) # 15 minutes expiry
    
    new_reset_token = PasswordResetToken(
        user_id=user.id,
        token=reset_token,
        expires_at=expires_at
    )
    db.add(new_reset_token)
    await db.commit()
    
    reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
    try:
        send_reset_password_link_with_token_in_email(email, reset_link, user.first_name)
    except Exception as e:
        print(f"Error sending reset password email: {e}")
        
    return "Password reset link sent to your email", reset_token


# ============================================================
# ✅ CHANGE/UPDATE PASSWORD
# ============================================================
async def update_password(
    user: User, current_password: str, new_password: str, db: AsyncSession
) -> None:
    if not verify_password(current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
        
    user.hashed_password = get_password_hash(new_password)
    db.add(user)
    await db.commit()


# ============================================================
# ✅ VERIFY EMAIL
# ============================================================
async def verify_email_with_code(
    user: User, verification_code: str, db: AsyncSession
) -> None:
    if user.is_verified:
        return # Already verified
        
    if user.verification_code != verification_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code"
        )
        
    user.is_verified = True
    # Clear code after verification
    user.verification_code = None 
    db.add(user)
    await db.commit()


# ============================================================
# ✅ RESET PASSWORD WITH TOKEN
# ============================================================
async def reset_password_with_token(token: str, new_password: str, db: AsyncSession) -> None:
    # 1. Verify token exists and is valid
    
    result = await db.execute(select(PasswordResetToken).where(PasswordResetToken.token == token))
    stored_token = result.scalars().first()
    
    if not stored_token:
        raise HTTPException(status_code=400, detail="Invalid reset token")
        
    if stored_token.is_used:
        raise HTTPException(status_code=400, detail="Token already used")
        
    if stored_token.expires_at < utcnow():
        raise HTTPException(status_code=400, detail="Token expired")
        
    # 2. Get User
    result = await db.execute(select(User).where(User.id == stored_token.user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # 3. Update Password
    user.hashed_password = get_password_hash(new_password)
    stored_token.is_used = True # Mark token as used
    
    db.add(user)
    db.add(stored_token)
    await db.commit()



