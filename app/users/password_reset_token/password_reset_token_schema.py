# app/users/password_reset_token/password_reset_token_schema.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class PasswordResetTokenBase(BaseModel):
    token: str
    is_used: bool = False
    expires_at: datetime
    
class PasswordResetTokenCreate(PasswordResetTokenBase):
    user_id: int

class PasswordResetTokenResponse(PasswordResetTokenBase):
    id: int
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True