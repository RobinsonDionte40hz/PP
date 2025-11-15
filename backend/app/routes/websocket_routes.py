"""
WebSocket emission endpoints for Celery workers to trigger emissions
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from app.websocket.socket_manager import socket_manager
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


class EmitProgressRequest(BaseModel):
    prediction_id: str
    data: Dict[str, Any]


@router.post("/emit/progress")
async def emit_progress(request: EmitProgressRequest):
    """Emit progress update via WebSocket (called by Celery workers)"""
    try:
        await socket_manager.emit_progress_update(
            request.prediction_id,
            request.data
        )
        subscriber_count = socket_manager.get_subscriber_count(request.prediction_id)
        logger.info(f"✓ Emitted progress to {subscriber_count} subscribers for {request.prediction_id}")
        return {"status": "success", "subscribers": subscriber_count}
    except Exception as e:
        logger.error(f"Failed to emit progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emit/metrics")
async def emit_metrics(request: EmitProgressRequest):
    """Emit metrics update via WebSocket (called by Celery workers)"""
    try:
        await socket_manager.emit_metrics_update(
            request.prediction_id,
            request.data
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to emit metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emit/completion")
async def emit_completion(request: EmitProgressRequest):
    """Emit completion notification via WebSocket (called by Celery workers)"""
    try:
        await socket_manager.emit_completion(
            request.prediction_id,
            request.data
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to emit completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emit/log")
async def emit_log(request: EmitProgressRequest):
    """Emit log event via WebSocket (called by Celery workers)"""
    try:
        await socket_manager.emit_event_log(
            request.prediction_id,
            request.data
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to emit log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subscribers/{prediction_id}")
async def get_subscribers(prediction_id: str):
    """Get subscriber count for a prediction"""
    count = socket_manager.get_subscriber_count(prediction_id)
    return {"prediction_id": prediction_id, "subscribers": count}
