"""
Prediction API endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from typing import Optional, List
from app.schemas.prediction import (
    PredictionCreateSchema,
    PredictionResponseSchema,
    PredictionListResponseSchema,
)
from app.models.prediction import PredictionStatus
from app.services.prediction_service import prediction_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=PredictionResponseSchema,
    status_code=201,
    summary="Create new prediction",
    description="Submit a new protein structure prediction job"
)
async def create_prediction(
    data: PredictionCreateSchema,
    background_tasks: BackgroundTasks,
):
    """
    Create a new prediction job.
    
    The prediction will be queued and executed asynchronously.
    If Celery/Redis is not available, prediction is created in pending state.
    """
    try:
        # Create prediction
        prediction = prediction_service.create_prediction(data)
        
        # Try to queue Celery task
        celery_available = False
        try:
            from app.tasks import run_prediction
            task = run_prediction.delay(prediction.id)
            
            # Update with task ID
            from app.schemas.prediction import PredictionUpdateSchema
            prediction_service.update_prediction(
                prediction.id,
                PredictionUpdateSchema(task_id=task.id, status=PredictionStatus.QUEUED)
            )
            
            logger.info(f"Queued prediction {prediction.id} with task {task.id}")
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
    description="Get list of predictions with optional filtering"
)
async def list_predictions(
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
):
    """
    List predictions with pagination and optional status filter.
    """
    try:
        predictions, total = prediction_service.list_predictions(
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
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Get detailed information about a specific prediction.
    """
    prediction = prediction_service.get_prediction(prediction_id)
    
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
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Delete a prediction and all associated data.
    
    This cannot be undone.
    """
    success = prediction_service.delete_prediction(prediction_id)
    
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
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Pause a running prediction.
    
    The prediction can be resumed later from the last checkpoint.
    """
    prediction = prediction_service.pause_prediction(prediction_id)
    
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
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Resume a paused prediction from the last checkpoint.
    """
    prediction = prediction_service.resume_prediction(prediction_id)
    
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
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Stop a running or paused prediction.
    
    The prediction will be marked as stopped and cannot be resumed.
    """
    prediction = prediction_service.stop_prediction(prediction_id)
    
    if not prediction:
        raise HTTPException(
            status_code=400,
            detail="Cannot stop prediction - not found or not in running/paused state"
        )
    
    # TODO: Send signal to Celery task to stop
    
    return PredictionResponseSchema(**prediction.to_dict())


@router.get(
    "/{prediction_id}/checkpoint",
    summary="Download checkpoint",
    description="Download the latest checkpoint file for a prediction"
)
async def get_checkpoint(
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Download the latest checkpoint file.
    
    Returns the checkpoint JSON file if available.
    """
    from pathlib import Path as FilePath
    from fastapi.responses import FileResponse
    import os
    
    prediction = prediction_service.get_prediction(prediction_id)
    
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
