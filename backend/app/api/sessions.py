"""
Work Session API endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends, Request, status
from typing import Dict, Any
import os
from app.schemas.work_session import (
    WorkSessionCreateSchema,
    WorkSessionUpdateSchema,
    WorkSessionResponseSchema,
    WorkSessionListResponseSchema,
)
from app.services.work_session_service import work_session_service
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
                session_id=session.id,
                user_id=user_id,
                page=1,
                page_size=1  # We only need count, not data
            )
            total_size = work_session_service.get_session_size(session.id, user_id)
            
            session_responses.append(
                WorkSessionResponseSchema(
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
        return WorkSessionResponseSchema(
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
        
        return WorkSessionResponseSchema(
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
        
        return WorkSessionResponseSchema(
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
