"""
Migrate existing predictions to session-based storage

This script migrates predictions that don't have a session_id to the new
session-based storage system by creating default sessions and updating
prediction records.

Usage:
    # Dry run (preview only)
    python scripts/migrate_predictions_to_sessions.py --dry-run
    
    # Migrate all users
    python scripts/migrate_predictions_to_sessions.py
    
    # Migrate specific user
    python scripts/migrate_predictions_to_sessions.py --user-id user_abc123
    
    # Migrate and move files to new structure
    python scripts/migrate_predictions_to_sessions.py --move-files
"""
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session

from app.database import get_db
from app.models.work_session import WorkSession
from app.models.prediction import Prediction
from app.services.work_session_service import work_session_service
from app.services.file_storage_service import FileStorageService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionMigrator:
    """Migrates predictions to session-based storage"""
    
    def __init__(
        self,
        db: Session,
        dry_run: bool = False,
        move_files: bool = False
    ):
        self.db = db
        self.dry_run = dry_run
        self.move_files = move_files
        self.file_storage = FileStorageService()
        
        self.stats = {
            "unmigrated_predictions": 0,
            "users_affected": 0,
            "sessions_created": 0,
            "predictions_updated": 0,
            "files_moved": 0,
            "errors": []
        }
    
    def run(
        self,
        user_id: Optional[str] = None,
        session_name: str = "Migrated Predictions",
        batch_size: int = 100
    ) -> Dict:
        """
        Run migration process
        
        Args:
            user_id: Migrate specific user only (None = all users)
            session_name: Name for created sessions
            batch_size: Database batch size
            
        Returns:
            Dictionary with migration statistics
        """
        logger.info("=== Prediction Migration to Sessions ===")
        logger.info(f"Mode: {'DRY RUN (no changes will be made)' if self.dry_run else 'LIVE'}")
        logger.info(f"Move files: {self.move_files}")
        logger.info("")
        
        # Step 1: Find unmigrated predictions
        logger.info("Finding unmigrated predictions...")
        unmigrated = self._find_unmigrated_predictions(user_id)
        self.stats["unmigrated_predictions"] = len(unmigrated)
        logger.info(f"Found {len(unmigrated)} predictions without sessions")
        
        if not unmigrated:
            logger.info("No predictions to migrate")
            return self.stats
        
        # Step 2: Group by user
        logger.info("")
        logger.info("Analyzing by user:")
        by_user = self._group_by_user(unmigrated)
        self.stats["users_affected"] = len(by_user)
        
        for uid, preds in by_user.items():
            logger.info(f"  - User {uid}: {len(preds)} predictions")
        
        # Step 3: Show migration plan
        logger.info("")
        logger.info("Plan:")
        logger.info(f"  - Create {len(by_user)} new sessions")
        logger.info(f"  - Update {len(unmigrated)} predictions")
        if self.move_files:
            logger.info(f"  - Move files to new structure")
        else:
            logger.info(f"  - Move 0 files (--move-files not enabled)")
        
        if self.dry_run:
            logger.info("")
            logger.info("DRY RUN COMPLETE - no changes made")
            logger.info("Run without --dry-run to apply changes")
            return self.stats
        
        # Step 4: Execute migration
        logger.info("")
        logger.info("Starting migration...")
        
        for uid, predictions in by_user.items():
            self._migrate_user_predictions(
                user_id=uid,
                predictions=predictions,
                session_name=session_name
            )
        
        # Step 5: Summary
        logger.info("")
        logger.info("=== Migration Complete ===")
        logger.info(f"Sessions created: {self.stats['sessions_created']}")
        logger.info(f"Predictions updated: {self.stats['predictions_updated']}")
        logger.info(f"Files moved: {self.stats['files_moved']}")
        if self.stats["errors"]:
            logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats["errors"]:
                logger.warning(f"  - {error}")
        
        return self.stats
    
    def _find_unmigrated_predictions(
        self,
        user_id: Optional[str] = None
    ) -> List[Prediction]:
        """Find predictions without session_id"""
        query = self.db.query(Prediction).filter(
            Prediction.session_id.is_(None)
        )
        
        if user_id:
            query = query.filter(Prediction.user_id == user_id)
        
        return query.all()
    
    def _group_by_user(
        self,
        predictions: List[Prediction]
    ) -> Dict[str, List[Prediction]]:
        """Group predictions by user_id"""
        by_user = {}
        for pred in predictions:
            if pred.user_id not in by_user:
                by_user[pred.user_id] = []
            by_user[pred.user_id].append(pred)
        return by_user
    
    def _migrate_user_predictions(
        self,
        user_id: str,
        predictions: List[Prediction],
        session_name: str
    ):
        """Migrate all predictions for a user"""
        try:
            # Create session
            logger.info(f"Creating session for user {user_id}...")
            session = work_session_service.create_session(
                user_id=user_id,
                name=session_name,
                db=self.db
            )
            self.stats["sessions_created"] += 1
            logger.info(f"  Created session {session.id}")
            
            # Update predictions
            for pred in predictions:
                try:
                    pred.session_id = session.id
                    self.db.commit()
                    self.stats["predictions_updated"] += 1
                    
                    logger.info(f"  Updated prediction {pred.id}")
                    
                    # Move files if requested
                    if self.move_files:
                        moved = self._move_prediction_files(
                            prediction_id=pred.id,
                            user_id=user_id,
                            session_id=session.id
                        )
                        if moved:
                            self.stats["files_moved"] += 1
                
                except Exception as e:
                    error_msg = f"Failed to migrate prediction {pred.id}: {str(e)}"
                    logger.error(error_msg)
                    self.stats["errors"].append(error_msg)
                    self.db.rollback()
            
        except Exception as e:
            error_msg = f"Failed to create session for user {user_id}: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            self.db.rollback()
    
    def _move_prediction_files(
        self,
        prediction_id: str,
        user_id: str,
        session_id: str
    ) -> bool:
        """Move files from old to new location"""
        try:
            # Old location
            old_path = Path(f"results/{prediction_id}")
            
            if not old_path.exists():
                logger.debug(f"    No files to move for {prediction_id}")
                return False
            
            # New location
            new_path = self.file_storage.get_prediction_directory(
                user_id=user_id,
                session_id=session_id,
                prediction_id=prediction_id
            )
            
            # Create parent directory
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move files
            shutil.move(str(old_path), str(new_path))
            
            logger.info(f"    Moved files {old_path} -> {new_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to move files for {prediction_id}: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing predictions to session-based storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without applying
  python scripts/migrate_predictions_to_sessions.py --dry-run
  
  # Migrate all users
  python scripts/migrate_predictions_to_sessions.py
  
  # Migrate specific user
  python scripts/migrate_predictions_to_sessions.py --user-id user_abc123
  
  # Migrate and move files to new structure
  python scripts/migrate_predictions_to_sessions.py --move-files
  
  # Full migration with file moves
  python scripts/migrate_predictions_to_sessions.py --move-files --session-name "Legacy Data"
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying (recommended first step)"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="Migrate specific user only (omit to migrate all users)"
    )
    parser.add_argument(
        "--move-files",
        action="store_true",
        help="Move files from results/ to user_data/ structure"
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default="Migrated Predictions",
        help="Name for created sessions (default: 'Migrated Predictions')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Database batch size (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Confirmation for live migration
    if not args.dry_run:
        logger.warning("")
        logger.warning("⚠️  WARNING: This will modify the database and potentially move files!")
        logger.warning("")
        response = input("Continue with live migration? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Migration cancelled")
            return 0
        logger.info("")
    
    # Get database session
    db = next(get_db())
    
    try:
        # Run migration
        migrator = PredictionMigrator(
            db=db,
            dry_run=args.dry_run,
            move_files=args.move_files
        )
        
        stats = migrator.run(
            user_id=args.user_id,
            session_name=args.session_name,
            batch_size=args.batch_size
        )
        
        # Return success exit code
        return 0 if not stats["errors"] else 1
        
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Migration interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Migration failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        db.close()


if __name__ == "__main__":
    exit(main())
