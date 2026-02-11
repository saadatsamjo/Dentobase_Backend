# app/users/user_models/schemas.py


from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# Allowed values as constants
ROLES = Literal["admin", "doctor"]


class UserBase(BaseModel):
    email: EmailStr
    model_config = ConfigDict(from_attributes=True)


# ✅ Request schema for registration
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    role: ROLES = "doctor"

    @field_validator("email", mode="before")
    def normalize_email(cls, v):
        # sourcery skip: assign-if-exp, reintroduce-else
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("role")
    def validate_role_field(cls, v):
        if v not in ["admin", "doctor"]:
            raise ValueError(f"Invalid role value: {v}")
        return v


# ✅ Response schema for user registration
class UserRegisterResponse(UserBase):
    id: int
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


# ✅ User login request
class UserLogin(BaseModel):
    email: EmailStr
    password: str


# ✅ Response schema for user info
class UserResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class UserInDB(UserRegister):
    id: int
    is_active: bool
    role: ROLES

    hashed_password: str  # For internal use, not API responses


class UserList(BaseModel):
    users: List[UserResponse]
    total: int


class UserPublic(BaseModel):
    id: int
    role: ROLES


# ✅ Response schema for user login
class UserLoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user: UserResponse


# ✅ Response schema for user logout
class UserLogoutResponse(BaseModel):
    message: str


# ✅ Response schema for token refresh
class UserRefreshResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


# ✅ Request schema for forgot password
class UserForgotPassword(BaseModel):
    email: EmailStr


# ✅ Request schema for reset password
class UserResetPassword(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)


# ✅ Request schema for verify email
class UserVerifyEmail(BaseModel):
    email: EmailStr
    verification_code: str


# ✅ Request schema for change password (authenticated)
class UserChangePassword(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)