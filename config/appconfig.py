# app/core/config.py

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, model_validator
from typing import Optional

class AppConfig(BaseSettings):
    # General Settings
    APP_NAME: str = Field(default="My FastAPI App", env="APP_NAME")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # Database Settings
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: str = Field(default="5432", env="DB_PORT")
    DB_NAME: str = Field(..., env="DB_NAME")
    
    # JWT Settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRY: int = Field(default=30, env="ACCESS_TOKEN_EXPIRY")
    REFRESH_TOKEN_EXPIRY: int = Field(default=60, env="REFRESH_TOKEN_EXPIRY")
    
    # URLs
    BASE_URL: str = Field(..., env="BASE_URL")
    FRONTEND_URL: str = Field(..., env="FRONTEND_URL")
    
    # Email Settings
    RESEND_API_KEY: str = Field(..., env="RESEND_API_KEY")
    
    # Cookie Settings
    COOKIE_DOMAIN: Optional[str] = Field(default=None, env="COOKIE_DOMAIN")
    COOKIE_SECURE: bool = Field(default=False, env="COOKIE_SECURE")
    COOKIE_SAMESITE: str = Field(default="lax", env="COOKIE_SAMESITE")
    ACCESS_TOKEN_COOKIE_NAME: str = Field(default="access_token", env="ACCESS_TOKEN_COOKIE_NAME")
    REFRESH_TOKEN_COOKIE_NAME: str = Field(default="refresh_token", env="REFRESH_TOKEN_COOKIE_NAME")

    @model_validator(mode='after')
    def adjust_for_environment(self):
        """Automatically adjust settings based on ENVIRONMENT variable from .env file"""
        if  self.ENVIRONMENT == "production":
            # Production settings
            self.COOKIE_SECURE = True
            self.COOKIE_SAMESITE = "strict"
            self.ACCESS_TOKEN_COOKIE_NAME = "__Host-access_token"
            self.REFRESH_TOKEN_COOKIE_NAME = "__Host-refresh_token"
            self.COOKIE_DOMAIN = None
            
            # You can add more production overrides here
            # if self.ACCESS_TOKEN_EXPIRY > 30:
            #     self.ACCESS_TOKEN_EXPIRY = 15  # Force shorter tokens in prod
        
        return self

    @property
    def DB_URL(self) -> str:
        """Async URL for your FastAPI application"""
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DB_URL_SYNC(self) -> str:
        """Sync URL for Alembic migrations"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = ConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore",
        case_sensitive=False
    )

    LOGGING_CONFIG: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelname)s:     %(message)s",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "": {"handlers": ["default"], "level": "INFO"}, # Root logger
            "uvicorn.error": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.access": {"propagate": False}, # Let uvicorn handle its own access logs
        },
    }
        
settings = AppConfig()