# app/users/auth_token_model/token_schemas.py
from typing import  Literal
from datetime import date, datetime
from pydantic import BaseModel



# Allowed token types as constants
TOKEN_TYPE = Literal["access", "refresh"]   


# tokens schemas
class TokenBase(BaseModel):
    token_string: str
    token_type: TOKEN_TYPE
    user_id: int
    expires_at: datetime
    is_revoked: bool = False

class TokenCreate(TokenBase):
    pass

class TokenResponse(TokenBase):
    pass
