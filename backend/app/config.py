from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "development-secret-key"
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/pp_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # PP System Integration
    PP_RESULTS_DIR: str = "./results"
    PP_CHECKPOINTS_DIR: str = "./checkpoints"
    PP_PDB_CACHE_DIR: str = "./pdb_cache"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
