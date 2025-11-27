"""
Work Session service - business logic for work session management
"""
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import tempfile
import shutil

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc

from app.models.work_session import WorkSession
from app.models.shared_export import SharedExport
from app.models.prediction import Prediction
from app.database import SessionLocal
from app.services.file_storage_service import FileStorageService

logger = logging.getLogger(__name__)


class WorkSessionService:
    """Service for managing work sessions"""
    
    def __init__(self, file_storage: Optional[FileStorageService] = None, db: Optional[Session] = None):
        """
        Initialize WorkSessionService
        
        Args:
            file_storage: Optional FileStorageService instance (creates default if None)
            db: Optional database session for testing (creates new session if None)
        """
        self.file_storage = file_storage or FileStorageService()
        self._db = db
    
    def _get_db(self) -> Session:
        """Get database session"""
        if self._db:
            return self._db
        return SessionLocal()
    
    # ========== Session CRUD Operations ==========
    
    def create_session(self, user_id: str, name: str) -> WorkSession:
        """
        Create a new work session with file system directory
        
        Args:
            user_id: User's key_id from authentication
            name: Human-readable session name
            
        Returns:
            Created WorkSession instance
            
        Raises:
            ValueError: If name is invalid
            OSError: If directory creation fails
        """
        # Validate name
        if not name or len(name.strip()) == 0:
            raise ValueError("Session name cannot be empty")
        if len(name) > 255:
            raise ValueError("Session name cannot exceed 255 characters")
        
        db = self._get_db()
        close_db = not self._db  # Only close if we created the session
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Create database record
            now = datetime.now(timezone.utc)
            session = WorkSession(
                id=session_id,
                user_id=user_id,
                name=name.strip(),
                created_at=now,
                updated_at=now,
                last_active_at=now
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            # Create file system directory
            try:
                self.file_storage.create_session_directory(user_id, session_id)
                logger.info(f"Created work session {session_id} for user {user_id}: {name}")
            except OSError as e:
                # Rollback database if directory creation fails
                db.delete(session)
                db.commit()
                logger.error(f"Failed to create directory for session {session_id}, rolled back DB: {e}")
                raise
            
            return session
        finally:
            if close_db:
                db.close()
    
    def get_session_by_id(self, session_id: str) -> Optional[WorkSession]:
        """
        Get session by ID without ownership validation (for internal use)
        
        Args:
            session_id: Work session ID
            
        Returns:
            WorkSession if found, None otherwise
        """
        db = self._get_db()
        close_db = not self._db
        try:
            session = db.query(WorkSession).filter(
                WorkSession.id == session_id
            ).first()
            
            return session
        finally:
            if close_db:
                db.close()
    
    def get_session(self, session_id: str, user_id: str) -> Optional[WorkSession]:
        """
        Get session by ID with ownership validation
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            
        Returns:
            WorkSession if found and owned by user, None otherwise
        """
        db = self._get_db()
        close_db = not self._db
        try:
            session = db.query(WorkSession).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            return session
        finally:
            if close_db:
                db.close()
    
    def list_sessions(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "last_active_at",
        sort_order: str = "desc"
    ) -> Tuple[List[WorkSession], int]:
        """
        List sessions for a user with pagination and sorting
        
        Args:
            user_id: User's key_id to filter by
            page: Page number (1-indexed)
            page_size: Number of items per page
            sort_by: Field to sort by (created_at, updated_at, last_active_at, name)
            sort_order: Sort order (asc, desc)
            
        Returns:
            Tuple of (list of sessions, total count)
        """
        db = self._get_db()
        try:
            # Base query with user filter
            query = db.query(WorkSession).filter(WorkSession.user_id == user_id)
            
            # Apply sorting
            sort_field = getattr(WorkSession, sort_by, WorkSession.last_active_at)
            if sort_order.lower() == "desc":
                query = query.order_by(desc(sort_field))
            else:
                query = query.order_by(sort_field)
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            offset = (page - 1) * page_size
            sessions = query.offset(offset).limit(page_size).all()
            
            logger.debug(f"Listed {len(sessions)} sessions for user {user_id} (page {page}/{(total + page_size - 1) // page_size})")
            
            return sessions, total
        finally:
            if not self._db:
                db.close()
    
    def update_session(self, session_id: str, user_id: str, name: str) -> Optional[WorkSession]:
        """
        Update session name with ownership validation
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            name: New session name
            
        Returns:
            Updated WorkSession if successful, None if not found or not owned
            
        Raises:
            ValueError: If name is invalid
        """
        # Validate name
        if not name or len(name.strip()) == 0:
            raise ValueError("Session name cannot be empty")
        if len(name) > 255:
            raise ValueError("Session name cannot exceed 255 characters")
        
        db = self._get_db()
        try:
            session = db.query(WorkSession).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            if not session:
                logger.warning(f"Session {session_id} not found or not owned by user {user_id}")
                return None
            
            # Update name and timestamp
            session.name = name.strip()
            session.updated_at = datetime.now(timezone.utc)
            
            db.commit()
            db.refresh(session)
            
            logger.info(f"Updated session {session_id} name to: {name}")
            
            return session
        finally:
            if not self._db:
                db.close()
    
    def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete session with ownership validation and cascade to files
        
        This will:
        1. Delete all related predictions (cascade)
        2. Delete all shared exports (cascade)
        3. Delete the session database record
        4. Delete the file system directory
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            
        Returns:
            True if deleted successfully, False if not found or not owned
        """
        db = self._get_db()
        try:
            session = db.query(WorkSession).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            if not session:
                logger.warning(f"Session {session_id} not found or not owned by user {user_id}")
                return False
            
            # Delete database record (cascades to predictions and shared_exports)
            db.delete(session)
            db.commit()
            
            # Delete file system directory (gracefully handle missing directories)
            try:
                self.file_storage.delete_session_directory(user_id, session_id)
            except Exception as e:
                logger.warning(f"Failed to delete directory for session {session_id}: {e}")
                # Continue - database is already deleted
            
            logger.info(f"Deleted session {session_id} for user {user_id}")
            
            return True
        finally:
            if not self._db:
                db.close()
    
    # ========== Prediction-in-Session Operations ==========
    
    def create_prediction_in_session(
        self,
        session_id: str,
        user_id: str,
        prediction: Prediction
    ) -> bool:
        """
        Link a prediction to a session and update activity timestamp
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            prediction: Prediction instance to link
            
        Returns:
            True if linked successfully, False if session not found or not owned
        """
        db = self._get_db()
        try:
            # Validate session ownership
            session = db.query(WorkSession).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            if not session:
                logger.warning(f"Session {session_id} not found or not owned by user {user_id}")
                return False
            
            # Get the prediction in this session's context (it may be from another session)
            db_prediction = db.query(Prediction).filter(Prediction.id == prediction.id).first()
            if not db_prediction:
                logger.warning(f"Prediction {prediction.id} not found in database")
                return False
            
            # Link prediction to session
            db_prediction.session_id = session_id
            db_prediction.updated_at = datetime.now(timezone.utc)
            
            # Update session activity
            session.last_active_at = datetime.now(timezone.utc)
            session.updated_at = datetime.now(timezone.utc)
            
            db.commit()
            
            logger.info(f"Linked prediction {prediction.id} to session {session_id}")
            
            return True
        finally:
            if not self._db:
                db.close()
    
    def get_session_predictions(
        self,
        session_id: str,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Prediction], int]:
        """
        Get predictions for a session with pagination
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            page: Page number (1-indexed)
            page_size: Number of items per page
            
        Returns:
            Tuple of (list of predictions, total count), or ([], 0) if not found/owned
        """
        db = self._get_db()
        try:
            # Validate session ownership
            session = db.query(WorkSession).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            if not session:
                logger.warning(f"Session {session_id} not found or not owned by user {user_id}")
                return [], 0
            
            # Query predictions
            query = db.query(Prediction).filter(Prediction.session_id == session_id)
            
            # Get total count
            total = query.count()
            
            # Sort by created_at descending and paginate
            offset = (page - 1) * page_size
            predictions = query.order_by(desc(Prediction.created_at)).offset(offset).limit(page_size).all()
            
            logger.debug(f"Retrieved {len(predictions)} predictions for session {session_id} (page {page})")
            
            return predictions, total
        finally:
            if not self._db:
                db.close()
    
    def update_session_activity(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """
        Update session's last_active_at timestamp
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation (optional, for internal use)
            
        Returns:
            True if updated, False if not found or not owned
        """
        db = self._get_db()
        try:
            if user_id:
                # With ownership validation
                session = db.query(WorkSession).filter(
                    WorkSession.id == session_id,
                    WorkSession.user_id == user_id
                ).first()
            else:
                # Without ownership validation (internal use)
                session = db.query(WorkSession).filter(
                    WorkSession.id == session_id
                ).first()
            
            if not session:
                return False
            
            session.last_active_at = datetime.now(timezone.utc)
            db.commit()
            
            return True
        finally:
            if not self._db:
                db.close()
    
    # ========== Archive Creation ==========
    
    def create_session_archive(
        self,
        session_id: str,
        user_id: str
    ) -> Optional[Path]:
        """
        Create a ZIP archive of the session with metadata
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            
        Returns:
            Path to temporary ZIP file, or None if session not found/owned
            Caller is responsible for cleaning up the temporary file
            
        Raises:
            OSError: If archive creation fails
        """
        db = self._get_db()
        try:
            # Validate session ownership and get details
            session = db.query(WorkSession).options(
                joinedload(WorkSession.predictions)
            ).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            if not session:
                logger.warning(f"Session {session_id} not found or not owned by user {user_id}")
                return None
            
            # Create metadata file in session directory
            predictions_metadata = [
                {
                    "id": pred.id,
                    "sequence": pred.sequence,
                    "status": pred.status,
                    "created_at": pred.created_at.isoformat() if pred.created_at else None,
                }
                for pred in session.predictions
            ]
            
            self.file_storage.create_session_metadata(
                user_id=user_id,
                session_id=session_id,
                session_name=session.name,
                created_at=session.created_at,
                last_active_at=session.last_active_at,
                predictions=predictions_metadata
            )
            
            # Create temporary ZIP file
            temp_dir = Path(tempfile.gettempdir())
            zip_path = temp_dir / f"session_{session_id}.zip"
            
            # Create archive
            self.file_storage.create_zip_archive(user_id, session_id, zip_path)
            
            logger.info(f"Created archive for session {session_id}: {zip_path}")
            
            return zip_path
            
        finally:
            if not self._db:
                db.close()
    
    # ========== Share Link Operations ==========
    
    def create_share_link(
        self,
        session_id: str,
        user_id: str,
        expires_in_hours: int = 24
    ) -> Optional[SharedExport]:
        """
        Generate a public share link for a session
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            expires_in_hours: Hours until link expires (default 24, max 168 = 7 days)
            
        Returns:
            SharedExport instance with share_id, or None if session not found/owned
            
        Raises:
            ValueError: If expires_in_hours is invalid
        """
        # Validate expiration time
        if expires_in_hours < 1:
            raise ValueError("Expiration time must be at least 1 hour")
        if expires_in_hours > 168:  # 7 days
            raise ValueError("Expiration time cannot exceed 168 hours (7 days)")
        
        db = self._get_db()
        try:
            # Validate session ownership
            session = db.query(WorkSession).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            if not session:
                logger.warning(f"Session {session_id} not found or not owned by user {user_id}")
                return None
            
            # Generate share ID
            share_id = str(uuid.uuid4())
            
            # Calculate expiration
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(hours=expires_in_hours)
            
            # Create shared export
            shared_export = SharedExport(
                share_id=share_id,
                session_id=session_id,
                created_at=now,
                expires_at=expires_at,
                access_count=0
            )
            
            db.add(shared_export)
            db.commit()
            db.refresh(shared_export)
            
            logger.info(f"Created share link {share_id} for session {session_id}, expires at {expires_at}")
            
            return shared_export
        finally:
            if not self._db:
                db.close()
    
    def get_shared_session(self, share_id: str) -> Optional[Dict[str, Any]]:
        """
        Get shared session data (public access, no authentication required)
        
        This increments the access count and validates expiration.
        Returns read-only session data.
        
        Args:
            share_id: Share link identifier
            
        Returns:
            Dictionary with session data and predictions, or None if not found/expired
        """
        db = self._get_db()
        try:
            # Get shared export with session
            shared_export = db.query(SharedExport).options(
                joinedload(SharedExport.work_session).joinedload(WorkSession.predictions)
            ).filter(
                SharedExport.share_id == share_id
            ).first()
            
            if not shared_export:
                logger.warning(f"Share link {share_id} not found")
                return None
            
            # Check expiration
            if shared_export.is_expired():
                logger.warning(f"Share link {share_id} has expired")
                return None
            
            # Increment access count
            shared_export.access_count += 1
            shared_export.last_accessed_at = datetime.now(timezone.utc)
            db.commit()
            
            # Build response data
            session = shared_export.work_session
            session_data = {
                "id": session.id,
                "name": session.name,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "last_active_at": session.last_active_at.isoformat() if session.last_active_at else None,
                "prediction_count": len(session.predictions),
                "predictions": [
                    {
                        "id": pred.id,
                        "sequence": pred.sequence,
                        "status": pred.status,
                        "created_at": pred.created_at.isoformat() if pred.created_at else None,
                        "completed_at": pred.completed_at.isoformat() if pred.completed_at else None,
                    }
                    for pred in session.predictions
                ],
                "shared_link": {
                    "share_id": shared_export.share_id,
                    "created_at": shared_export.created_at.isoformat() if shared_export.created_at else None,
                    "expires_at": shared_export.expires_at.isoformat() if shared_export.expires_at else None,
                    "access_count": shared_export.access_count,
                }
            }
            
            logger.info(f"Served shared session {session.id} via share link {share_id} (access #{shared_export.access_count})")
            
            return session_data
            
        finally:
            if not self._db:
                db.close()
    
    def cleanup_expired_shares(self) -> int:
        """
        Delete expired share links (maintenance task)
        
        Returns:
            Number of expired shares deleted
        """
        db = self._get_db()
        try:
            now = datetime.now(timezone.utc)
            
            # Find expired shares
            expired = db.query(SharedExport).filter(
                SharedExport.expires_at < now
            ).all()
            
            count = len(expired)
            
            # Delete them
            for share in expired:
                db.delete(share)
            
            db.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired share links")
            
            return count
            
        finally:
            if not self._db:
                db.close()
    
    # ========== Utility Methods ==========
    
    def get_session_size(self, session_id: str, user_id: str) -> int:
        """
        Get total size of session files in bytes
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            
        Returns:
            Size in bytes, or 0 if session not found/owned
        """
        db = self._get_db()
        try:
            # Validate session ownership
            session = db.query(WorkSession).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            if not session:
                return 0
            
            return self.file_storage.get_session_size(user_id, session_id)
            
        finally:
            if not self._db:
                db.close()
    
    def get_session_with_stats(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session with additional statistics (prediction count, total size)
        
        Args:
            session_id: Work session ID
            user_id: User's key_id for ownership validation
            
        Returns:
            Dictionary with session data and stats, or None if not found/owned
        """
        db = self._get_db()
        try:
            session = db.query(WorkSession).options(
                joinedload(WorkSession.predictions)
            ).filter(
                WorkSession.id == session_id,
                WorkSession.user_id == user_id
            ).first()
            
            if not session:
                return None
            
            # Calculate statistics
            prediction_count = len(session.predictions)
            total_size = self.file_storage.get_session_size(user_id, session_id)
            
            return {
                "id": session.id,
                "user_id": session.user_id,
                "name": session.name,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "last_active_at": session.last_active_at.isoformat() if session.last_active_at else None,
                "prediction_count": prediction_count,
                "total_size_bytes": total_size,
            }
            
        finally:
            if not self._db:
                db.close()


# Global service instance
work_session_service = WorkSessionService()
