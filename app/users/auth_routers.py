# app/users/auth_routers.py

from fastapi import APIRouter, Depends, HTTPException, Body
from app.users.auth_services import (
    registering_user,
    authenticate_user,
    login_user,
    refresh_access_token,
    logout_user,
    create_password_reset_link,
    verify_email_with_code,
    update_password,
    reset_password_with_token,
)
from app.users.user_models.schemas import (
    UserRegister,
    UserLogin,
    UserResponse,
    UserRegisterResponse,
    UserLoginResponse,
    UserLogoutResponse,
    UserRefreshResponse,
    UserForgotPassword,
    UserResetPassword,
    UserVerifyEmail,
    UserChangePassword,
)
from app.users.user_models.user_model import User
from typing import Optional, Tuple
from app.database.connection import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.users.dependencies import get_current_user

router = APIRouter()

# ============================================================
# ✅ REGISTER
# ============================================================
@router.post("/register", response_model=UserRegisterResponse)
async def register_user(user_data: UserRegister, db: AsyncSession = Depends(get_db)) -> UserRegisterResponse:
    return await registering_user(user_data, db)


# ============================================================
# ✅ AUTHENTICATE USER (LOGIN)
# ============================================================
@router.post("/login", response_model=UserLoginResponse)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)) -> UserLoginResponse:
    access_token, refresh_token, user = await login_user(user_data, db)
    return UserLoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=user
    )


# ============================================================
# ✅ REFRESH TOKEN
# ============================================================
@router.post("/refresh", response_model=UserRefreshResponse)
async def refresh_token(
    refresh_token: str = Body(..., embed=True), 
    db: AsyncSession = Depends(get_db)
) -> UserRefreshResponse:
    access_token, new_refresh_token = await refresh_access_token(refresh_token, db)
    return UserRefreshResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer"
    )


# ============================================================
# ✅ LOGOUT USER
# ============================================================
# Note: This requires the token to be passed. Since we use Bearer token, we can extract it.
# We will use a dependency to get the token string if we want to revoke it.
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer()

@router.post("/logout", response_model=UserLogoutResponse)
async def logout(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> UserLogoutResponse:
    await logout_user(current_user, db)
    return UserLogoutResponse(message="Successfully logged out of all devices")


# ============================================================
# ✅ CHANGE PASSWORD
# ============================================================
@router.post("/change-password", response_model=UserLogoutResponse)
async def change_password(
    data: UserChangePassword,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> UserLogoutResponse:
    await update_password(current_user, data.current_password, data.new_password, db)
    return {"message": "Password changed successfully"}



# ============================================================
# ✅ FORGOT PASSWORD
# ============================================================
@router.post("/forgot-password")
async def forgot_password(
    data: UserForgotPassword,
    db: AsyncSession = Depends(get_db)
):
    message, token = await create_password_reset_link(data.email, db)
    # In production, send email here.
    return {"message": message, "reset_token": token} 


# ============================================================
# ✅ RESET PASSWORD
# ============================================================
@router.post("/reset-password")
async def reset_password(
    data: UserResetPassword,
    db: AsyncSession = Depends(get_db)
):
    # Verify token and reset password
    await reset_password_with_token(data.token, data.new_password, db)
    return {"message": "Password reset successful"}


# ============================================================
# ✅ VERIFY EMAIL
# ============================================================
@router.post("/verify-email")
async def verify_email(
    data: UserVerifyEmail,
    db: AsyncSession = Depends(get_db)
):
    from sqlalchemy.future import select
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    await verify_email_with_code(user, data.verification_code, db)
    return {"message": "Email verified successfully"}
