"""
Prediction service - business logic for predictions
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.models.prediction import Prediction, PredictionStatus
from app.models.work_session import WorkSession
from app.schemas.prediction import PredictionCreateSchema, PredictionUpdateSchema
from app.database import SessionLocal
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for managing predictions"""
    
    def __init__(self):
        pass  # Database-backed now
    
    def _get_db(self) -> Session:
        """Get database session"""
        return SessionLocal()
    
    def create_prediction(self, data: PredictionCreateSchema, user_id: Optional[str] = None) -> Prediction:
        """Create a new prediction
        
        Args:
            data: Prediction configuration
            user_id: User ID to associate prediction with (will create/use default session)
        """
        db = self._get_db()
        try:
            prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
            
            # Build configuration
            config = data.configuration.model_dump() if data.configuration else {}
            
            # Get or create default session for user
            session_id = None
            if user_id:
                session_id = self._get_or_create_default_session(db, user_id)
            
            prediction = Prediction(
                id=prediction_id,
                sequence=data.sequence,
                session_id=session_id,
                status=PredictionStatus.PENDING.value,  # Store as string
                configuration=config,
                total_iterations=config.get("iterations", 1000),
            )
            
            db.add(prediction)
            db.commit()
            db.refresh(prediction)
            
            logger.info(f"Created prediction {prediction_id} for sequence length {len(data.sequence)} (session: {session_id})")
            
            return prediction
        finally:
            db.close()
    
    def _get_or_create_default_session(self, db: Session, user_id: str) -> str:
        """Get or create a default session for the user"""
        # Look for existing default session
        default_session = db.query(WorkSession).filter(
            WorkSession.user_id == user_id,
            WorkSession.name == "Default Session"
        ).first()
        
        if default_session:
            return default_session.id
        
        # Create new default session
        from datetime import timezone
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        new_session = WorkSession(
            id=session_id,
            user_id=user_id,
            name="Default Session",
            created_at=now,
            updated_at=now,
            last_active_at=now
        )
        db.add(new_session)
        db.commit()
        logger.info(f"Created default session {session_id} for user {user_id}")
        return session_id
    
    def get_prediction(self, prediction_id: str, user_id: Optional[str] = None) -> Optional[Prediction]:
        """Get prediction by ID, optionally filtered by user ownership"""
        db = self._get_db()
        try:
            query = db.query(Prediction).filter(Prediction.id == prediction_id)
            
            # If user_id provided, validate ownership through session
            # But also allow predictions without session_id (legacy/standalone predictions)
            if user_id:
                prediction = query.first()
                if prediction:
                    # If prediction has no session, allow access (standalone prediction)
                    if prediction.session_id is None:
                        return prediction
                    # If prediction has session, verify ownership
                    session = db.query(WorkSession).filter(
                        WorkSession.id == prediction.session_id,
                        WorkSession.user_id == user_id
                    ).first()
                    if session:
                        return prediction
                    return None  # User doesn't own this prediction
                return None
            
            return query.first()
        finally:
            db.close()
    
    def list_predictions(
        self,
        user_id: Optional[str] = None,
        status: Optional[PredictionStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Prediction], int]:
        """List predictions with optional filtering and pagination
        
        Args:
            user_id: Filter predictions to only those belonging to this user's sessions
            status: Filter by prediction status
            page: Page number (1-indexed)
            page_size: Number of items per page
            
        Returns:
            Tuple of (list of predictions, total count)
        """
        db = self._get_db()
        try:
            query = db.query(Prediction)
            
            # Filter by user ownership through session join
            # Also include predictions with no session (legacy/standalone)
            if user_id:
                from sqlalchemy import or_
                query = query.outerjoin(WorkSession, Prediction.session_id == WorkSession.id).filter(
                    or_(
                        WorkSession.user_id == user_id,
                        Prediction.session_id.is_(None)  # Include sessionless predictions
                    )
                )
            
            # Filter by status
            if status:
                query = query.filter(Prediction.status == status.value)
            
            # Get total count
            total = query.count()
            
            # Sort by created_at descending and paginate
            predictions = query.order_by(Prediction.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
            
            return predictions, total
        finally:
            db.close()
    
    def update_prediction(self, prediction_id: str, data: PredictionUpdateSchema) -> Optional[Prediction]:
        """Update prediction"""
        db = self._get_db()
        try:
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
            if not prediction:
                return None
            
            # Update fields
            if data.status is not None:
                prediction.status = data.status.value if hasattr(data.status, 'value') else data.status
                
                # Set timestamps based on status
                if data.status == PredictionStatus.RUNNING and not prediction.started_at:
                    prediction.started_at = datetime.utcnow()
                elif data.status in [PredictionStatus.COMPLETED, PredictionStatus.FAILED, PredictionStatus.STOPPED]:
                    prediction.completed_at = datetime.utcnow()
            
            if data.task_id is not None:
                prediction.task_id = data.task_id
            
            if data.current_iteration is not None:
                prediction.current_iteration = data.current_iteration
            
            if data.progress_percentage is not None:
                prediction.progress_percentage = data.progress_percentage
            
            if data.metrics is not None:
                # Ensure metrics is a dict and properly merge
                if prediction.metrics is None:
                    prediction.metrics = {}
                if isinstance(prediction.metrics, dict):
                    prediction.metrics.update(data.metrics)
                else:
                    prediction.metrics = data.metrics
                # Mark as modified for SQLAlchemy to detect change
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(prediction, "metrics")
            
            if data.error_message is not None:
                prediction.error_message = data.error_message
            
            if data.checkpoint_path is not None:
                prediction.checkpoint_path = data.checkpoint_path
            
            if data.result_path is not None:
                prediction.result_path = data.result_path
            
            prediction.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(prediction)
            
            logger.info(f"Updated prediction {prediction_id}: status={prediction.status}")
            
            return prediction
        finally:
            db.close()
    
    def delete_prediction(self, prediction_id: str, user_id: Optional[str] = None) -> bool:
        """Delete prediction, optionally validated by user ownership"""
        db = self._get_db()
        try:
            query = db.query(Prediction).filter(Prediction.id == prediction_id)
            
            # If user_id provided, validate ownership through session
            if user_id:
                query = query.join(WorkSession, Prediction.session_id == WorkSession.id).filter(
                    WorkSession.user_id == user_id
                )
            
            prediction = query.first()
            if prediction:
                db.delete(prediction)
                db.commit()
                logger.info(f"Deleted prediction {prediction_id}")
                return True
            return False
        finally:
            db.close()
    
    def pause_prediction(self, prediction_id: str, user_id: Optional[str] = None) -> Optional[Prediction]:
        """Pause a running prediction, optionally validated by user ownership"""
        db = self._get_db()
        try:
            query = db.query(Prediction).filter(Prediction.id == prediction_id)
            
            # If user_id provided, validate ownership through session
            if user_id:
                query = query.join(WorkSession, Prediction.session_id == WorkSession.id).filter(
                    WorkSession.user_id == user_id
                )
            
            prediction = query.first()
            if not prediction:
                return None
            
            if prediction.status != PredictionStatus.RUNNING.value:
                logger.warning(f"Cannot pause prediction {prediction_id} - status is {prediction.status}")
                return None
            
            prediction.status = PredictionStatus.PAUSED.value
            prediction.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(prediction)
            
            logger.info(f"Paused prediction {prediction_id}")
            return prediction
        finally:
            db.close()
    
    def resume_prediction(self, prediction_id: str, user_id: Optional[str] = None) -> Optional[Prediction]:
        """Resume a paused prediction, optionally validated by user ownership"""
        db = self._get_db()
        try:
            query = db.query(Prediction).filter(Prediction.id == prediction_id)
            
            # If user_id provided, validate ownership through session
            if user_id:
                query = query.join(WorkSession, Prediction.session_id == WorkSession.id).filter(
                    WorkSession.user_id == user_id
                )
            
            prediction = query.first()
            if not prediction:
                return None
            
            if prediction.status != PredictionStatus.PAUSED.value:
                logger.warning(f"Cannot resume prediction {prediction_id} - status is {prediction.status}")
                return None
            
            prediction.status = PredictionStatus.RUNNING.value
            prediction.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(prediction)
            
            logger.info(f"Resumed prediction {prediction_id}")
            return prediction
        finally:
            db.close()
    
    def stop_prediction(self, prediction_id: str, user_id: Optional[str] = None) -> Optional[Prediction]:
        """Stop a running or paused prediction, optionally validated by user ownership"""
        db = self._get_db()
        try:
            query = db.query(Prediction).filter(Prediction.id == prediction_id)
            
            # If user_id provided, validate ownership through session
            if user_id:
                query = query.join(WorkSession, Prediction.session_id == WorkSession.id).filter(
                    WorkSession.user_id == user_id
                )
            
            prediction = query.first()
            if not prediction:
                return None
            
            if prediction.status not in [PredictionStatus.RUNNING.value, PredictionStatus.PAUSED.value]:
                logger.warning(f"Cannot stop prediction {prediction_id} - status is {prediction.status}")
                return None
            
            prediction.status = PredictionStatus.STOPPED.value
            prediction.completed_at = datetime.utcnow()
            prediction.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(prediction)
            
            logger.info(f"Stopped prediction {prediction_id}")
            return prediction
        finally:
            db.close()


# Global service instance
prediction_service = PredictionService()
