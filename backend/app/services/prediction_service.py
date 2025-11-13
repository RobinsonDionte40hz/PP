"""
Prediction service - business logic for predictions
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.models.prediction import Prediction, PredictionStatus
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
    
    def create_prediction(self, data: PredictionCreateSchema) -> Prediction:
        """Create a new prediction"""
        db = self._get_db()
        try:
            prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
            
            # Build configuration
            config = data.configuration.model_dump() if data.configuration else {}
            
            prediction = Prediction(
                id=prediction_id,
                sequence=data.sequence,
                status=PredictionStatus.PENDING.value,  # Store as string
                configuration=config,
                total_iterations=config.get("iterations", 1000),
            )
            
            db.add(prediction)
            db.commit()
            db.refresh(prediction)
            
            logger.info(f"Created prediction {prediction_id} for sequence length {len(data.sequence)}")
            
            return prediction
        finally:
            db.close()
    
    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Get prediction by ID"""
        db = self._get_db()
        try:
            return db.query(Prediction).filter(Prediction.id == prediction_id).first()
        finally:
            db.close()
    
    def list_predictions(
        self,
        status: Optional[PredictionStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Prediction], int]:
        """List predictions with optional filtering and pagination"""
        db = self._get_db()
        try:
            query = db.query(Prediction)
            
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
                prediction.metrics.update(data.metrics)
            
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
    
    def delete_prediction(self, prediction_id: str) -> bool:
        """Delete prediction"""
        db = self._get_db()
        try:
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
            if prediction:
                db.delete(prediction)
                db.commit()
                logger.info(f"Deleted prediction {prediction_id}")
                return True
            return False
        finally:
            db.close()
    
    def pause_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Pause a running prediction"""
        db = self._get_db()
        try:
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
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
    
    def resume_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Resume a paused prediction"""
        db = self._get_db()
        try:
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
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
    
    def stop_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Stop a running or paused prediction"""
        db = self._get_db()
        try:
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
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
