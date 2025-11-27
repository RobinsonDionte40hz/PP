"""
Session Cleanup Service - manages expired session removal

This service identifies and removes expired work sessions, including:
- Database records (WorkSession and related SharedExport)
- File system data (session directories with predictions)
- Comprehensive logging for audit trails
"""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple
import shutil

from sqlalchemy.orm import Session

from app.database import get_db
from app.models.work_session import WorkSession
from app.models.shared_export import SharedExport
from app.services.file_storage_service import FileStorageService

logger = logging.getLogger(__name__)


class SessionCleanupService:
    """Service for cleaning up expired sessions"""
    
    def __init__(self, file_storage: FileStorageService = None):
        """
        Initialize cleanup service
        
        Args:
            file_storage: FileStorageService instance (optional, creates new if None)
        """
        self.file_storage = file_storage or FileStorageService()
    
    def identify_expired_sessions(
        self,
        retention_days: int,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        Identify sessions that have expired based on retention period
        
        A session is considered expired if:
        - last_active_at is older than retention_days
        
        Args:
            retention_days: Number of days to retain inactive sessions
            db: Database session (optional, creates new if None)
            
        Returns:
            List of session dictionaries with id, user_id, name, last_active_at
            
        Example:
            >>> service = SessionCleanupService()
            >>> expired = service.identify_expired_sessions(retention_days=90)
            >>> print(f"Found {len(expired)} expired sessions")
        """
        should_close_db = db is None
        if db is None:
            db = next(get_db())
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            expired_sessions = db.query(WorkSession).filter(
                WorkSession.last_active_at < cutoff_date
            ).all()
            
            result = []
            for session in expired_sessions:
                # Handle both naive and aware datetimes
                last_active = session.last_active_at
                if last_active.tzinfo is None:
                    # SQLite stores naive datetimes
                    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
                    days_inactive = (now_naive - last_active).days
                else:
                    days_inactive = (datetime.now(timezone.utc) - last_active).days
                
                result.append({
                    "id": session.id,
                    "user_id": session.user_id,
                    "name": session.name,
                    "last_active_at": session.last_active_at,
                    "days_inactive": days_inactive
                })
            
            
            logger.info(
                f"Identified {len(result)} expired sessions "
                f"(retention_days={retention_days}, cutoff={cutoff_date})"
            )
            
            return result
            
        finally:
            if should_close_db:
                db.close()
    
    def delete_expired_sessions(
        self,
        retention_days: int,
        dry_run: bool = False,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Delete expired sessions including DB records and files
        
        Performs:
        1. Identifies expired sessions
        2. Deletes shared export records
        3. Deletes session database records
        4. Removes session file directories
        
        Args:
            retention_days: Number of days to retain inactive sessions
            dry_run: If True, only identify sessions without deleting
            db: Database session (optional, creates new if None)
            
        Returns:
            Dictionary with cleanup statistics:
            - sessions_identified: Count of expired sessions found
            - sessions_deleted: Count of sessions deleted from DB
            - directories_deleted: Count of session directories removed
            - errors: List of error messages encountered
            - dry_run: Whether this was a dry run
            
        Example:
            >>> service = SessionCleanupService()
            >>> # Dry run to preview
            >>> stats = service.delete_expired_sessions(retention_days=90, dry_run=True)
            >>> print(f"Would delete {stats['sessions_identified']} sessions")
            >>> 
            >>> # Actual deletion
            >>> stats = service.delete_expired_sessions(retention_days=90)
            >>> print(f"Deleted {stats['sessions_deleted']} sessions")
        """
        should_close_db = db is None
        if db is None:
            db = next(get_db())
        
        stats = {
            "sessions_identified": 0,
            "sessions_deleted": 0,
            "directories_deleted": 0,
            "errors": [],
            "dry_run": dry_run
        }
        
        try:
            # Step 1: Identify expired sessions
            expired_sessions = self.identify_expired_sessions(
                retention_days=retention_days,
                db=db
            )
            stats["sessions_identified"] = len(expired_sessions)
            
            if dry_run:
                logger.info(
                    f"DRY RUN: Would delete {len(expired_sessions)} expired sessions"
                )
                return stats
            
            # Step 2: Delete each expired session
            for session_info in expired_sessions:
                session_id = session_info["id"]
                user_id = session_info["user_id"]
                
                try:
                    # Delete shared exports for this session
                    deleted_shares = db.query(SharedExport).filter(
                        SharedExport.session_id == session_id
                    ).delete(synchronize_session=False)
                    
                    # Delete session record
                    deleted_sessions = db.query(WorkSession).filter(
                        WorkSession.id == session_id
                    ).delete(synchronize_session=False)
                    
                    db.commit()
                    
                    if deleted_sessions > 0:
                        stats["sessions_deleted"] += 1
                        logger.info(
                            f"Deleted session {session_id} "
                            f"(user={user_id}, shares={deleted_shares})"
                        )
                    
                    # Delete session directory
                    success, directory_deleted = self._delete_session_directory(
                        user_id=user_id,
                        session_id=session_id
                    )
                    
                    if directory_deleted:
                        stats["directories_deleted"] += 1
                    
                    if not success:
                        stats["errors"].append(
                            f"Failed to delete directory for session {session_id}"
                        )
                
                except Exception as e:
                    error_msg = f"Error deleting session {session_id}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    db.rollback()
            
            logger.info(
                f"Cleanup complete: deleted {stats['sessions_deleted']} sessions, "
                f"{stats['directories_deleted']} directories, "
                f"{len(stats['errors'])} errors"
            )
            
            return stats
            
        except Exception as e:
            error_msg = f"Cleanup failed: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            if db:
                db.rollback()
            return stats
            
        finally:
            if should_close_db:
                db.close()
    
    def _delete_session_directory(
        self,
        user_id: str,
        session_id: str
    ) -> Tuple[bool, bool]:
        """
        Delete session directory from file system
        
        Args:
            user_id: User's key_id
            session_id: Session ID
            
        Returns:
            Tuple of (success, directory_existed)
            - success: True if deletion succeeded or directory didn't exist
            - directory_existed: True if directory existed and was deleted
        """
        try:
            session_dir = self.file_storage.get_session_directory(
                user_id=user_id,
                session_id=session_id
            )
            
            if session_dir.exists():
                shutil.rmtree(session_dir)
                logger.info(f"Deleted session directory: {session_dir}")
                return True, True
            else:
                logger.debug(f"Session directory does not exist: {session_dir}")
                return True, False
                
        except Exception as e:
            logger.error(
                f"Failed to delete session directory "
                f"(user={user_id}, session={session_id}): {str(e)}"
            )
            return False, False
    
    def cleanup_expired_shares(
        self,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Delete expired share links (separate from session cleanup)
        
        This removes SharedExport records where expires_at has passed,
        without deleting the associated session.
        
        Args:
            db: Database session (optional, creates new if None)
            
        Returns:
            Dictionary with cleanup statistics:
            - shares_deleted: Count of expired shares removed
            - errors: List of error messages
            
        Example:
            >>> service = SessionCleanupService()
            >>> stats = service.cleanup_expired_shares()
            >>> print(f"Deleted {stats['shares_deleted']} expired share links")
        """
        should_close_db = db is None
        if db is None:
            db = next(get_db())
        
        stats = {
            "shares_deleted": 0,
            "errors": []
        }
        
        try:
            now = datetime.now(timezone.utc)
            
            deleted_count = db.query(SharedExport).filter(
                SharedExport.expires_at < now
            ).delete(synchronize_session=False)
            
            db.commit()
            
            stats["shares_deleted"] = deleted_count
            logger.info(f"Deleted {deleted_count} expired share links")
            
            return stats
            
        except Exception as e:
            error_msg = f"Share cleanup failed: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            if db:
                db.rollback()
            return stats
            
        finally:
            if should_close_db:
                db.close()


# Singleton instance
_cleanup_service = None

def get_cleanup_service() -> SessionCleanupService:
    """Get or create singleton cleanup service instance"""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = SessionCleanupService()
    return _cleanup_service
