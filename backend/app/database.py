"""
Database configuration and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Try to create engine, but make it optional for development
try:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        echo=settings.APP_ENV == "development"
    )
    # Test connection
    with engine.connect() as conn:
        pass
    logger.info("Database connection successful")
except Exception as e:
    logger.warning(f"Database connection failed: {e}")
    logger.warning("Running without database - using in-memory storage")
    # Create SQLite in-memory database as fallback
    engine = create_engine("sqlite:///:memory:", echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all tables"""
    try:
        # Import models to register them with Base
        from app.models import prediction, campaign
        
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.warning(f"Could not create tables: {e}")