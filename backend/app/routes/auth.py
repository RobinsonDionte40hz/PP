"""
Authentication API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.auth import (
    UserRegisterRequest,
    UserRegisterResponse,
    UserResponse,
    ErrorResponse
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "User registered successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        409: {"model": ErrorResponse, "description": "Username or email already exists"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Register a new user",
    description="""
    Register a new user account with username, password, and optional email.
    
    **Username Requirements:**
    - 3-50 characters
    - Must start with letter or number
    - Can contain letters, numbers, underscores, and hyphens
    - Must be unique
    
    **Password Requirements:**
    - 8-72 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    
    **Email Requirements:**
    - Must be valid email format (if provided)
    - Must be unique (if provided)
    """
)
async def register(
    request: UserRegisterRequest,
    db: Session = Depends(get_db)
):
    """
    Register a new user.
    
    This endpoint creates a new user account with the provided credentials.
    The password is securely hashed using bcrypt before storage.
    """
    success, message, user = AuthService.register_user(
        db=db,
        username=request.username,
        password=request.password,
        email=request.email
    )
    
    if not success:
        # Determine status code based on error message
        if "already exists" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=message
            )
        elif any(keyword in message.lower() for keyword in ["cannot be empty", "must contain", "must be at least"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message
            )
    
    # user should never be None here if success is True, but add assertion for type checker
    assert user is not None, "User should not be None after successful registration"
    
    # Return user profile (without password)
    user_profile = user.to_profile()
    
    return UserRegisterResponse(
        message="User registered successfully",
        user=UserResponse(**user_profile)
    )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the authentication service is running"
)
async def health_check():
    """Health check endpoint for authentication service"""
    return {"status": "healthy", "service": "authentication"}
