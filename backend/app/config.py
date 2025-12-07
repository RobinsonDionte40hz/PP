from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from typing import List, Union, Optional

class Settings(BaseSettings):
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "development-secret-key-CHANGE-IN-PRODUCTION"
    
    # Database - Use SQLite file for development (shared between backend and celery)
    DATABASE_URL: str = "sqlite:///./pp_dev.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_SESSION_URL: str = "redis://localhost:6379/1"  # New: separate DB for sessions
    REDIS_MAX_CONNECTIONS: int = 50
    
    # Session Management (Redis DB 1, separate from Celery)
    SESSION_REDIS_URL: str = "redis://localhost:6379/1"
    SESSION_REDIS_PREFIX: str = "session:"
    SESSION_EXPIRE_MINUTES: int = 30
    SESSION_TTL_SECONDS: int = 1800  # 30 minutes
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    WORKER_CLASS: str = "uvicorn.workers.UvicornWorker"
    
    # PP System Integration
    PP_RESULTS_DIR: str = "./results"
    PP_CHECKPOINTS_DIR: str = "./checkpoints"
    PP_PDB_CACHE_DIR: str = "./pdb_cache"
    PP_VISUALIZATION_OUTPUT_DIR: str = "./visualization_output"
    PP_DEFAULT_ITERATIONS: int = 1000
    PP_DEFAULT_AGENTS: int = 10
    PP_MAX_SEQUENCE_LENGTH: int = 1000
    
    # Security Settings - CORS_ORIGINS can be comma-separated string or JSON array
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS from comma-separated string or JSON array."""
        if isinstance(v, str):
            # Remove quotes if present
            v = v.strip().strip("'\"")
            # Try JSON first
            if v.startswith('['):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Fall back to comma-separated
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    CORS_ALLOW_CREDENTIALS: bool = True
    ENABLE_HSTS: bool = False  # Enable in production with HTTPS
    HSTS_MAX_AGE: int = 31536000
    ENABLE_CSRF: bool = True
    ENABLE_API_KEYS: bool = False  # Enable if API key auth is needed
    
    # Rate Limiting
    RATE_LIMIT_REGISTER: int = 5
    RATE_LIMIT_LOGIN: int = 10
    RATE_LIMIT_REFRESH: int = 20
    RATE_LIMIT_API: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "console"
    
    # Email Configuration (for feedback feature)
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    FEEDBACK_EMAIL: Optional[str] = None  # Email to receive feedback
    
    # JWT Settings
    JWT_SECRET_KEY: str = "jwt-secret-CHANGE-IN-PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Quota Settings - User prediction limits
    DEFAULT_DAILY_QUOTA: int = 20  # Free tier daily limit
    DEFAULT_MONTHLY_QUOTA: int = 100  # Free tier monthly limit
    PRO_DAILY_QUOTA: int = 100  # Pro tier daily limit
    PRO_MONTHLY_QUOTA: int = 500  # Pro tier monthly limit
    ENTERPRISE_DAILY_QUOTA: int = -1  # Enterprise tier (-1 = unlimited)
    ENTERPRISE_MONTHLY_QUOTA: int = -1  # Enterprise tier (-1 = unlimited)
    
    # CAPTCHA Settings - Bot protection on registration
    RECAPTCHA_ENABLED: bool = False  # Enable in production
    RECAPTCHA_SITE_KEY: Optional[str] = None  # Public key for frontend
    RECAPTCHA_SECRET_KEY: Optional[str] = None  # Secret key for backend verification
    CAPTCHA_PROVIDER: str = "recaptcha"  # 'recaptcha' or 'hcaptcha'
    RECAPTCHA_MIN_SCORE: float = 0.5  # Minimum score for v3 (0.0-1.0)
    
    # Email Verification Settings
    EMAIL_VERIFICATION_EXPIRE_HOURS: int = 24  # Verification links expire after 24 hours
    REQUIRE_EMAIL_VERIFICATION: bool = True  # Require email verification for predictions
    FRONTEND_URL: str = "http://localhost:5173"  # Frontend URL for verification links
    
    # OAuth Settings - Google
    GOOGLE_CLIENT_ID: Optional[str] = None  # Google OAuth client ID
    GOOGLE_CLIENT_SECRET: Optional[str] = None  # Google OAuth client secret
    GOOGLE_REDIRECT_URI: Optional[str] = None  # Google OAuth redirect URI (defaults to FRONTEND_URL/auth/google/callback)
    
    # OAuth Settings - GitHub
    GITHUB_CLIENT_ID: Optional[str] = None  # GitHub OAuth client ID
    GITHUB_CLIENT_SECRET: Optional[str] = None  # GitHub OAuth client secret
    GITHUB_REDIRECT_URI: Optional[str] = None  # GitHub OAuth redirect URI (defaults to FRONTEND_URL/auth/github/callback)
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_TIME_LIMIT: int = 3600
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3300
    
    # Session Storage and Cleanup
    USER_DATA_DIR: str = "./user_data"
    SESSION_RETENTION_DAYS: int = 90  # Sessions inactive for 90+ days are deleted
    SHARE_LINK_MAX_HOURS: int = 168  # Share links expire after 7 days (168 hours)
    CLEANUP_SCHEDULE_CRON: str = "0 2 * * *"  # Run daily at 2 AM
    
    # Development Settings
    SQL_ECHO: bool = False
    SHOW_ERROR_DETAILS: bool = True
    RELOAD: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
