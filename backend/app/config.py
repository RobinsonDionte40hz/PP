from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "development-secret-key-CHANGE-IN-PRODUCTION"
    
    # Database - Use SQLite file for development (shared between backend and celery)
    DATABASE_URL: str = "sqlite:///./pp_dev.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Session Management (Redis DB 1, separate from Celery)
    SESSION_REDIS_URL: str = "redis://localhost:6379/1"
    SESSION_REDIS_PREFIX: str = "session:"
    SESSION_EXPIRE_MINUTES: int = 30
    
    # PP System Integration
    PP_RESULTS_DIR: str = "./results"
    PP_CHECKPOINTS_DIR: str = "./checkpoints"
    PP_PDB_CACHE_DIR: str = "./pdb_cache"
    
    # Security Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    ENABLE_HSTS: bool = False  # Enable in production with HTTPS
    ENABLE_CSRF: bool = True
    ENABLE_API_KEYS: bool = False  # Enable if API key auth is needed
    
    # JWT Settings
    JWT_SECRET_KEY: str = "jwt-secret-CHANGE-IN-PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
