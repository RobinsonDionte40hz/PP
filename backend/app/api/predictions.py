"""
Prediction API endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks, Request, Depends
from typing import Optional, List, Dict, Any
from app.schemas.prediction import (
    PredictionCreateSchema,
    PredictionResponseSchema,
    PredictionListResponseSchema,
)
from app.models.prediction import PredictionStatus
from app.services.prediction_service import prediction_service
from app.services.quota_service import quota_service
from app.security import require_auth_with_session, require_verified_email
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
import os

logger = logging.getLogger(__name__)

# Initialize rate limiter (disabled in testing)
IS_TESTING = os.getenv("TESTING", "false").lower() == "true"
limiter = Limiter(key_func=get_remote_address, enabled=not IS_TESTING)

router = APIRouter()


def get_user_id(user: Dict[str, Any]) -> str:
    """Extract user_id from JWT token payload"""
    user_id = user.get("sub") or user.get("key_id")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="User ID not found in token"
        )
    return user_id


@router.post(
    "",
    response_model=PredictionResponseSchema,
    status_code=201,
    summary="Create new prediction",
    description="Submit a new protein structure prediction job"
)
@limiter.limit("10/minute")
async def create_prediction(
    request: Request,
    data: PredictionCreateSchema,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_verified_email),
):
    """
    Create a new prediction job.
    
    The prediction will be queued and executed asynchronously.
    If Celery/Redis is not available, prediction is created in pending state.
    
    NOTE: Consider using POST /api/sessions/{session_id}/predictions for better organization.
    """
    user_id = get_user_id(user)
    
    # Check user quota before creating prediction
    has_quota, error_message = quota_service.check_quota(user_id)
    if not has_quota:
        quota_info = quota_service.get_user_quota(user_id)
        raise HTTPException(
            status_code=429,
            detail={
                "message": error_message,
                "quota": quota_info
            },
            headers={"X-Quota-Exceeded": "true"}
        )
    
    try:
        # Create prediction with user's session
        prediction = prediction_service.create_prediction(data, user_id=user_id)
        
        # Increment user's quota count
        quota_service.increment_quota(user_id)
        
        # Try to queue Celery task
        celery_available = False
        try:
            # Use V2 task with unified PredictionRunner
            from app.tasks import run_prediction_v2
            task = run_prediction_v2.delay(prediction.id)
            
            # Update with task ID
            from app.schemas.prediction import PredictionUpdateSchema
            prediction_service.update_prediction(
                prediction.id,
                PredictionUpdateSchema(task_id=task.id, status=PredictionStatus.QUEUED)
            )
            
            logger.info(f"Queued prediction {prediction.id} with task {task.id} (using PredictionRunner V2)")
            celery_available = True
        except Exception as celery_error:
            # Celery/Redis not available - keep prediction in pending state
            logger.warning(f"Celery not available: {celery_error}")
            logger.info(f"Prediction {prediction.id} created in PENDING state. Start Redis and Celery worker to process it.")
        
        # Refresh prediction to get updated data
        prediction = prediction_service.get_prediction(prediction.id)
        
        response = PredictionResponseSchema(**prediction.to_dict())
        
        # Add warning header if Celery is not available
        if not celery_available:
            logger.warning(f"Prediction {prediction.id} will remain pending until Celery worker is started")
        
        return response
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "",
    response_model=PredictionListResponseSchema,
    summary="List predictions",
    description="Get list of predictions with optional filtering (filtered by authenticated user)"
)
@limiter.limit("30/minute")
async def list_predictions(
    request: Request,
    status: Optional[PredictionStatus] = Query(
        None,
        description="Filter by status"
    ),
    page: int = Query(
        1,
        ge=1,
        description="Page number"
    ),
    page_size: int = Query(
        20,
        ge=1,
        le=100,
        description="Items per page"
    ),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    List predictions with pagination and optional status filter.
    
    Only returns predictions belonging to sessions owned by the authenticated user.
    """
    try:
        user_id = get_user_id(user)
        predictions, total = prediction_service.list_predictions(
            user_id=user_id,
            status=status,
            page=page,
            page_size=page_size
        )
        
        return PredictionListResponseSchema(
            predictions=[PredictionResponseSchema(**p.to_dict()) for p in predictions],
            total=total,
            page=page,
            page_size=page_size
        )
    
    except Exception as e:
        logger.error(f"Error listing predictions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{prediction_id}",
    response_model=PredictionResponseSchema,
    summary="Get prediction details",
    description="Get detailed information about a specific prediction"
)
async def get_prediction(
    prediction_id: str = Path(..., description="Prediction ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Get detailed information about a specific prediction.
    
    Only returns prediction if it belongs to a session owned by the authenticated user.
    """
    user_id = get_user_id(user)
    prediction = prediction_service.get_prediction(prediction_id, user_id=user_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return PredictionResponseSchema(**prediction.to_dict())


@router.delete(
    "/{prediction_id}",
    status_code=204,
    summary="Delete prediction",
    description="Delete a prediction and its associated data"
)
async def delete_prediction(
    prediction_id: str = Path(..., description="Prediction ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Delete a prediction and all associated data.
    
    Only allows deletion if prediction belongs to a session owned by the authenticated user.
    This cannot be undone.
    """
    user_id = get_user_id(user)
    success = prediction_service.delete_prediction(prediction_id, user_id=user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # TODO: Delete associated files (checkpoints, results)
    
    return None


@router.post(
    "/{prediction_id}/pause",
    response_model=PredictionResponseSchema,
    summary="Pause prediction",
    description="Pause a running prediction"
)
async def pause_prediction(
    prediction_id: str = Path(..., description="Prediction ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Pause a running prediction.
    
    Only allows pausing if prediction belongs to a session owned by the authenticated user.
    The prediction can be resumed later from the last checkpoint.
    """
    user_id = get_user_id(user)
    prediction = prediction_service.pause_prediction(prediction_id, user_id=user_id)
    
    if not prediction:
        raise HTTPException(
            status_code=400,
            detail="Cannot pause prediction - not found or not in running state"
        )
    
    # TODO: Send signal to Celery task to pause
    
    return PredictionResponseSchema(**prediction.to_dict())


@router.post(
    "/{prediction_id}/resume",
    response_model=PredictionResponseSchema,
    summary="Resume prediction",
    description="Resume a paused prediction"
)
async def resume_prediction(
    prediction_id: str = Path(..., description="Prediction ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Resume a paused prediction from the last checkpoint.
    
    Only allows resuming if prediction belongs to a session owned by the authenticated user.
    """
    user_id = get_user_id(user)
    prediction = prediction_service.resume_prediction(prediction_id, user_id=user_id)
    
    if not prediction:
        raise HTTPException(
            status_code=400,
            detail="Cannot resume prediction - not found or not in paused state"
        )
    
    # TODO: Send signal to Celery task to resume or create new task
    
    return PredictionResponseSchema(**prediction.to_dict())


@router.post(
    "/{prediction_id}/stop",
    response_model=PredictionResponseSchema,
    summary="Stop prediction",
    description="Stop a running or paused prediction"
)
async def stop_prediction(
    prediction_id: str = Path(..., description="Prediction ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Stop a running or paused prediction.
    
    Only allows stopping if prediction belongs to a session owned by the authenticated user.
    The prediction will be marked as stopped and cannot be resumed.
    """
    user_id = get_user_id(user)
    prediction = prediction_service.stop_prediction(prediction_id, user_id=user_id)
    
    if not prediction:
        raise HTTPException(
            status_code=400,
            detail="Cannot stop prediction - not found or not in running/paused state"
        )
    
    # TODO: Send signal to Celery task to stop
    
    return PredictionResponseSchema(**prediction.to_dict())


@router.get(
    "/{prediction_id}/queue-status",
    summary="Get queue status",
    description="Get the queue position and estimated wait time for a prediction"
)
async def get_queue_status(
    prediction_id: str = Path(..., description="Prediction ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Get queue status for a prediction.
    
    Returns:
    - queue_position: Position in the queue (1 = next to run, 0 = running)
    - estimated_wait_minutes: Estimated minutes until processing starts
    - total_queued: Total number of predictions in queue
    - status: Current prediction status
    """
    user_id = get_user_id(user)
    prediction = prediction_service.get_prediction(prediction_id, user_id=user_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Get queue information
    queue_info = prediction_service.get_queue_position(prediction_id)
    
    return {
        "prediction_id": prediction_id,
        "status": prediction.status,
        "queue_position": queue_info["queue_position"],
        "total_queued": queue_info["total_queued"],
        "estimated_wait_minutes": queue_info["estimated_wait_minutes"],
        "message": queue_info["message"]
    }


@router.get(
    "/{prediction_id}/checkpoint",
    summary="Download checkpoint",
    description="Download the latest checkpoint file for a prediction"
)
async def get_checkpoint(
    prediction_id: str = Path(..., description="Prediction ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Download the latest checkpoint file.
    
    Only allows download if prediction belongs to a session owned by the authenticated user.
    Returns the checkpoint JSON file if available.
    """
    from pathlib import Path as FilePath
    from fastapi.responses import FileResponse
    import os
    
    user_id = get_user_id(user)
    prediction = prediction_service.get_prediction(prediction_id, user_id=user_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if not prediction.checkpoint_path:
        raise HTTPException(status_code=404, detail="No checkpoint available")
    
    # Find latest checkpoint
    checkpoint_dir = FilePath(prediction.checkpoint_path)
    if not checkpoint_dir.exists():
        raise HTTPException(status_code=404, detail="Checkpoint directory not found")
    
    # Get all checkpoint files sorted by modification time
    checkpoint_files = sorted(
        checkpoint_dir.glob("checkpoint_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not checkpoint_files:
        raise HTTPException(status_code=404, detail="No checkpoint files found")
    
    latest_checkpoint = checkpoint_files[0]
    
    return FileResponse(
        path=str(latest_checkpoint),
        media_type="application/json",
        filename=f"{prediction_id}_checkpoint.json"
    )
