"""
Database migration script for user authentication system
Creates the users table with proper indexes
"""
import logging
from sqlalchemy import create_engine, text, inspect
from app.config import settings
from app.database import Base
from app.models.user import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_table_exists(engine, table_name: str) -> bool:
    """Check if a table exists in the database"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def migrate_database():
    """
    Run database migration to add user authentication tables
    """
    logger.info("Starting database migration...")
    
    try:
        # Create engine
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            echo=True
        )
        
        # Check if users table already exists
        if check_table_exists(engine, "users"):
            logger.info("Users table already exists. Skipping creation.")
        else:
            logger.info("Creating users table...")
            
            # Create only the users table
            User.__table__.create(engine, checkfirst=True)
            
            logger.info("✅ Users table created successfully")
            logger.info(f"   - Columns: key_id, username, email, password_hash, is_active, created_at, updated_at, last_login")
            logger.info(f"   - Indexes: key_id (PK), username (unique), email (unique)")
            logger.info(f"   - Composite indexes: (username, is_active), (email, is_active)")
        
        # Verify table structure
        inspector = inspect(engine)
        columns = inspector.get_columns("users")
        indexes = inspector.get_indexes("users")
        
        logger.info(f"✅ Migration completed successfully")
        logger.info(f"   - Table: users")
        logger.info(f"   - Columns: {len(columns)}")
        logger.info(f"   - Indexes: {len(indexes)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    migrate_database()
