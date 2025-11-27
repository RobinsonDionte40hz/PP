"""
Work Session API endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends, Request, status
from typing import Dict, Any
import os
from datetime import datetime, timezone
from app.schemas.work_session import (
    WorkSessionCreateSchema,
    WorkSessionUpdateSchema,
    WorkSessionResponseSchema,
    WorkSessionListResponseSchema,
)
from app.schemas.prediction import (
    PredictionCreateSchema,
    PredictionResponseSchema,
    PredictionListResponseSchema,
)
from app.services.work_session_service import work_session_service
from app.services.prediction_service import PredictionService
from app.security import require_auth_with_session
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

logger = logging.getLogger(__name__)

# Initialize rate limiter (disabled in testing)
IS_TESTING = os.getenv("TESTING", "false").lower() == "true"
limiter = Limiter(key_func=get_remote_address, enabled=not IS_TESTING)

router = APIRouter()


@router.get(
    "",
    response_model=WorkSessionListResponseSchema,
    summary="List work sessions",
    description="Get list of work sessions for authenticated user with pagination"
)
@limiter.limit("30/minute")
async def list_sessions(
    request: Request,
    page: int = Query(
        1,
        ge=1,
        description="Page number (1-indexed)"
    ),
    page_size: int = Query(
        20,
        ge=1,
        le=100,
        description="Items per page (max 100)"
    ),
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    List work sessions for the authenticated user.
    
    Sessions are returned in descending order by last_active_at (most recent first).
    Only sessions belonging to the authenticated user are returned.
    
    Requirements: 8.1, 1.3
    """
    try:
        # JWT tokens use "sub" (subject) for user ID
        user_id = user.get("sub") or user.get("key_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        # Get sessions for user
        sessions, total = work_session_service.list_sessions(
            user_id=user_id,
            page=page,
            page_size=page_size
        )
        
        # Convert to response schemas with prediction count and size
        session_responses = []
        for session in sessions:
            # get_session_predictions returns (predictions_list, count) tuple
            _, prediction_count = work_session_service.get_session_predictions(
                session_id=session.id,  # type: ignore[arg-type]
                user_id=user_id,
                page=1,
                page_size=1  # We only need count, not data
            )
            total_size = work_session_service.get_session_size(session.id, user_id)  # type: ignore[arg-type]
            
            session_responses.append(
                WorkSessionResponseSchema(  # type: ignore[arg-type]
                    id=session.id,
                    user_id=session.user_id,
                    name=session.name,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    last_active_at=session.last_active_at,
                    prediction_count=prediction_count,
                    total_size=total_size
                )
            )
        
        return WorkSessionListResponseSchema(
            sessions=session_responses,
            total=total,
            page=page,
            page_size=page_size
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing sessions for user {user.get('key_id')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "",
    response_model=WorkSessionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create new work session",
    description="Create a new work session for organizing predictions"
)
@limiter.limit("10/minute")
async def create_session(
    request: Request,
    data: WorkSessionCreateSchema,
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    Create a new work session for the authenticated user.
    
    A work session is a logical grouping of related protein structure predictions.
    The session directory is automatically created in the file system.
    
    Requirements: 8.2, 1.1, 1.2
    """
    try:
        # JWT tokens use "sub" (subject) for user ID
        user_id = user.get("sub") or user.get("key_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        # Create session
        session = work_session_service.create_session(
            user_id=user_id,
            name=data.name
        )
        
        logger.info(f"Created work session {session.id} for user {user_id}")
        
        # Return response with initial counts
        return WorkSessionResponseSchema(  # type: ignore[arg-type]
            id=session.id,
            user_id=session.user_id,
            name=session.name,
            created_at=session.created_at,
            updated_at=session.updated_at,
            last_active_at=session.last_active_at,
            prediction_count=0,
            total_size=0
        )
    
    except ValueError as e:
        logger.error(f"Validation error creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating session for user {user.get('key_id')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/{session_id}",
    response_model=WorkSessionResponseSchema,
    summary="Get work session details",
    description="Get detailed information about a specific work session"
)
async def get_session(
    session_id: str = Path(..., description="Session ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    Get detailed information about a specific work session.
    
    Only the session owner can access the session details.
    Includes prediction count and total storage size.
    
    Requirements: 8.3, 3.1, 3.2
    """
    try:
        # JWT tokens use "sub" (subject) for user ID
        user_id = user.get("sub") or user.get("key_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        # Get session with ownership validation
        session = work_session_service.get_session(session_id, user_id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or access denied"
            )
        
        # Get prediction count and total size
        _, prediction_count = work_session_service.get_session_predictions(
            session_id=session_id,
            user_id=user_id,
            page=1,
            page_size=1  # We only need count, not data
        )
        total_size = work_session_service.get_session_size(session_id, user_id)
        
        return WorkSessionResponseSchema(  # type: ignore[arg-type]
            id=session.id,
            user_id=session.user_id,
            name=session.name,
            created_at=session.created_at,
            updated_at=session.updated_at,
            last_active_at=session.last_active_at,
            prediction_count=prediction_count,
            total_size=total_size
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id} for user {user.get('key_id')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put(
    "/{session_id}",
    response_model=WorkSessionResponseSchema,
    summary="Update work session",
    description="Update work session name"
)
@limiter.limit("10/minute")
async def update_session(
    request: Request,
    data: WorkSessionUpdateSchema,
    session_id: str = Path(..., description="Session ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    Update a work session's name.
    
    Only the session owner can update the session.
    
    Requirements: 1.2
    """
    try:
        # JWT tokens use "sub" (subject) for user ID
        user_id = user.get("sub") or user.get("key_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        # Update session with ownership validation
        session = work_session_service.update_session(
            session_id=session_id,
            user_id=user_id,
            name=data.name
        )
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or access denied"
            )
        
        logger.info(f"Updated session {session_id} for user {user_id}")
        
        # Get additional metrics
        _, prediction_count = work_session_service.get_session_predictions(
            session_id=session_id,
            user_id=user_id,
            page=1,
            page_size=1  # We only need count, not data
        )
        total_size = work_session_service.get_session_size(session_id, user_id)
        
        return WorkSessionResponseSchema(  # type: ignore[arg-type]
            id=session.id,
            user_id=session.user_id,
            name=session.name,
            created_at=session.created_at,
            updated_at=session.updated_at,
            last_active_at=session.last_active_at,
            prediction_count=prediction_count,
            total_size=total_size
        )
    
    except ValueError as e:
        logger.error(f"Validation error updating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session {session_id} for user {user.get('key_id')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete work session",
    description="Delete a work session and all associated predictions and files"
)
@limiter.limit("10/minute")
async def delete_session(
    request: Request,
    session_id: str = Path(..., description="Session ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    Delete a work session.
    
    Only the session owner can delete the session.
    This will cascade delete all predictions within the session and remove all files.
    
    Requirements: 6.5
    """
    try:
        # JWT tokens use "sub" (subject) for user ID
        user_id = user.get("sub") or user.get("key_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        # Delete session with ownership validation
        success = work_session_service.delete_session(session_id, user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or access denied"
            )
        
        logger.info(f"Deleted session {session_id} for user {user_id}")
        
        # 204 No Content - no response body
        return None
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id} for user {user.get('key_id')}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ========== Session Predictions Endpoints ==========

@router.get(
    "/{session_id}/predictions",
    response_model=PredictionListResponseSchema,
    summary="List predictions in session",
    description="Get list of predictions for a specific work session"
)
async def list_session_predictions(
    session_id: str = Path(..., description="Session ID"),
    page: int = Query(
        1,
        ge=1,
        description="Page number (1-indexed)"
    ),
    page_size: int = Query(
        20,
        ge=1,
        le=100,
        description="Items per page (max 100)"
    ),
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    List predictions for a specific work session.
    
    Only the session owner can access the predictions.
    Predictions are returned in descending order by created_at (most recent first).
    
    Requirements: 8.3, 2.3
    """
    try:
        # JWT tokens use "sub" (subject) for user ID
        user_id = user.get("sub") or user.get("key_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        # Get predictions for session (includes ownership validation)
        predictions, total = work_session_service.get_session_predictions(
            session_id=session_id,
            user_id=user_id,
            page=page,
            page_size=page_size
        )
        
        # Check if empty result means session not found or just no predictions
        if total == 0:
            session = work_session_service.get_session(session_id, user_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found or access denied"
                )
        
        # Convert to response schemas
        prediction_responses = [
            PredictionResponseSchema(  # type: ignore[arg-type]
                id=pred.id,
                sequence=pred.sequence,
                status=pred.status,
                configuration=pred.configuration,
                created_at=pred.created_at,
                updated_at=pred.updated_at,
                started_at=pred.started_at,
                completed_at=pred.completed_at,
                error_message=pred.error_message,
                task_id=pred.task_id,
                checkpoint_path=pred.checkpoint_path,
                result_path=pred.result_path,
                current_iteration=pred.current_iteration,
                total_iterations=pred.total_iterations,
                progress_percentage=pred.progress_percentage,
                metrics=pred.metrics
            )
            for pred in predictions
        ]
        
        return PredictionListResponseSchema(
            predictions=prediction_responses,
            total=total,
            page=page,
            page_size=page_size
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing predictions for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/{session_id}/predictions",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create prediction in session",
    description="Create a new prediction within a specific work session"
)
@limiter.limit("20/minute")
async def create_session_prediction(
    request: Request,
    data: PredictionCreateSchema,
    session_id: str = Path(..., description="Session ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    Create a new prediction within a work session.
    
    Only the session owner can create predictions in the session.
    The session's last_active_at timestamp is automatically updated.
    
    Requirements: 8.4, 2.1, 1.5
    """
    try:
        # JWT tokens use "sub" (subject) for user ID
        user_id = user.get("sub") or user.get("key_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        # Validate session ownership first
        session = work_session_service.get_session(session_id, user_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or access denied"
            )
        
        # Create prediction through prediction service
        prediction_service = PredictionService()
        
        # Create prediction record
        prediction = prediction_service.create_prediction(data)
        
        # Link prediction to session and update activity timestamp
        success = work_session_service.create_prediction_in_session(
            session_id=session_id,
            user_id=user_id,
            prediction=prediction
        )
        
        if not success:
            # Rollback prediction creation if linking failed
            prediction_service.delete_prediction(prediction.id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to link prediction to session"
            )
        
        logger.info(f"Created prediction {prediction.id} in session {session_id} for user {user_id}")
        
        return PredictionResponseSchema(  # type: ignore[arg-type]
            id=prediction.id,
            sequence=prediction.sequence,
            status=prediction.status,
            configuration=prediction.configuration,
            created_at=prediction.created_at,
            updated_at=prediction.updated_at,
            started_at=prediction.started_at,
            completed_at=prediction.completed_at,
            error_message=prediction.error_message,
            task_id=prediction.task_id,
            checkpoint_path=prediction.checkpoint_path,
            result_path=prediction.result_path,
            current_iteration=prediction.current_iteration,
            total_iterations=prediction.total_iterations,
            progress_percentage=prediction.progress_percentage,
            metrics=prediction.metrics
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error creating prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating prediction in session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
