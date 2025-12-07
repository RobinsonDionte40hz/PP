"""
Authentication API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.auth import (
    UserRegisterRequest,
    UserRegisterResponse,
    UserLoginRequest,
    UserLoginResponse,
    LogoutResponse,
    TokenRefreshRequest,
    TokenRefreshResponse,
    UserResponse,
    TokenResponse,
    ErrorResponse
)
from app.services.auth_service import AuthService
from app.utils.rate_limit import RateLimiter
from app.services.session_manager import get_session_manager

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "User registered successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        409: {"model": ErrorResponse, "description": "Username or email already exists"},
        429: {"model": ErrorResponse, "description": "Too many registration attempts"},
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
    db: Session = Depends(get_db),
    req: Request = None,
    session_manager = Depends(get_session_manager)
):
    """
    Register a new user.
    
    This endpoint creates a new user account with the provided credentials.
    The password is securely hashed using bcrypt before storage.
    
    Rate limit: 5 registrations per hour per IP address.
    Requirement: 6.4
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    # Get client IP for logging (Requirement 6.5 - no sensitive data)
    client_ip = req.client.host if req and req.client else "unknown"
    
    # Check rate limit (Requirement 6.4: 5 registrations per hour)
    # Skip rate limiting in test mode
    import os
    if os.environ.get("TESTING") != "true":
        try:
            rate_limiter = RateLimiter(session_manager.redis_client)
            allowed, retry_after = rate_limiter.check_rate_limit(
                endpoint="register",
                identifier=client_ip,
                max_attempts=5,
                window_seconds=3600
            )
            if not allowed:
                logger.warning(
                    f"Registration rate limit exceeded from IP: {client_ip}, Retry after: {retry_after}s"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many registration attempts. Try again in {retry_after} seconds",
                    headers={"Retry-After": str(retry_after)}
                )
        except HTTPException:
            raise
        except Exception as e:
            # Fail open if rate limiting fails
            logger.error(f"Rate limiting check failed: {str(e)}")
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
    
    # Secure logging - no passwords (Requirement 6.5)
    logger.info(
        f"User registered successfully: username={user.username}, "
        f"ip={client_ip}, key_id={user.key_id}"
    )
    
    return UserRegisterResponse(
        message="User registered successfully",
        user=UserResponse(**user_profile)
    )


@router.post(
    "/login",
    response_model=UserLoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Login successful"},
        400: {"model": ErrorResponse, "description": "Invalid credentials"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Account inactive"},
        429: {"model": ErrorResponse, "description": "Too many login attempts"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="User login",
    description="""
    Authenticate user and create session.
    
    This endpoint:
    1. Validates credentials against database
    2. Checks for existing active session
    3. Terminates old session if exists (single-session enforcement)
    4. Generates new JWT tokens (access + refresh)
    5. Creates session in Redis with user info
    6. Updates user's last_login timestamp
    
    Returns JWT tokens that must be included in subsequent API requests.
    """
)
async def login(
    request: Request,
    credentials: UserLoginRequest,
    db: Session = Depends(get_db),
    session_manager = Depends(get_session_manager)
):
    """
    Authenticate user and create session.
    
    This endpoint implements single-session-per-user enforcement,
    automatically terminating any existing session.
    
    Requirements: 2.1, 2.2, 2.4, 2.5, 3.1, 3.2, 3.4
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    # Extract client info for session tracking (Requirement 6.4)
    ip_address = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Check rate limit (Requirement 6.4: 10 logins per 15 minutes)
    # Skip rate limiting in test mode
    import os
    if os.environ.get("TESTING") != "true":
        try:
            rate_limiter = RateLimiter(session_manager.redis_client)
            allowed, retry_after = rate_limiter.check_rate_limit(
                endpoint="login",
                identifier=ip_address,
                max_attempts=10,
                window_seconds=900
            )
            if not allowed:
                logger.warning(
                    f"Login rate limit exceeded from IP: {ip_address}, Retry after: {retry_after}s"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many login attempts. Try again in {retry_after} seconds",
                    headers={"Retry-After": str(retry_after)}
                )
        except HTTPException:
            raise
        except Exception as e:
            # Fail open if rate limiting fails
            logger.error(f"Rate limiting check failed: {str(e)}")
    
    # Log login attempt - no passwords (Requirement 6.5)
    logger.info(
        f"Login attempt: username={credentials.username}, ip={ip_address}"
    )
    
    # Attempt login
    success, message, data = await AuthService.login_user(
        db=db,
        username=credentials.username,
        password=credentials.password,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        # Secure logging - failed login (Requirement 6.5)
        logger.warning(
            f"Login failed: username={credentials.username}, ip={ip_address}, reason={message}"
        )
        
        # Determine status code based on error message
        if "cannot be empty" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        elif "inactive" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=message
            )
        elif "invalid username or password" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=message
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message
            )
    
    # data should never be None here if success is True
    assert data is not None, "Data should not be None after successful login"
    
    user = data["user"]
    
    # Secure logging - successful login, no tokens (Requirement 6.5)
    logger.info(
        f"Login successful: username={user.username}, ip={ip_address}, key_id={user.key_id}"
    )
    
    # Return user profile and tokens
    return UserLoginResponse(
        message="Login successful",
        user=UserResponse(**user.to_profile()),
        tokens=TokenResponse(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            token_type="bearer",
            expires_in=data["expires_in"]
        )
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Logout successful"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="User logout",
    description="""
    Logout user and terminate session.
    
    This endpoint:
    1. Extracts token JTI from Authorization header
    2. Deletes session from Redis
    3. Removes user's active session mapping
    4. Invalidates JWT token (session cannot be reused)
    
    Requires valid JWT token in Authorization header.
    """
)
async def logout(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Logout user and terminate session.
    
    This endpoint requires authentication and terminates the user's
    active session, ensuring it cannot be reused.
    
    Requirements: 3.3, 5.1, 5.2, 5.4
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    # Extract client info
    client_ip = request.client.host if request.client else "unknown"
    
    # Extract token from Authorization header
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Remove "Bearer " prefix
    token = auth_header.replace("Bearer ", "").replace("bearer ", "")
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Decode token to get user info (allow expired tokens for logout)
    try:
        from app.security import decode_token, extract_jti_from_token
        
        # Try to decode token normally first
        try:
            payload = decode_token(token)
            user_key_id = payload.get("sub")
        except HTTPException:
            # Token is expired - that's OK for logout
            # Extract JTI without full validation
            from jose import jwt
            from app.config import settings
            
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": False}  # Allow expired tokens
            )
            user_key_id = payload.get("sub")
        
        if not user_key_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Logout user
        success, message = await AuthService.logout_user(
            token=token,
            user_key_id=user_key_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message
            )
        
        # Secure logging - no tokens (Requirement 6.5)
        logger.info(
            f"Logout successful: user_id={user_key_id}, ip={client_ip}"
        )
        
        return LogoutResponse(message=message)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )


@router.post(
    "/refresh",
    response_model=TokenRefreshResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Token refreshed successfully"},
        401: {"model": ErrorResponse, "description": "Invalid or expired refresh token"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Refresh access token",
    description="""
    Refresh access token using refresh token.
    
    This endpoint:
    1. Validates refresh token (signature, expiration, type)
    2. Checks that session still exists in Redis
    3. Generates new access token with same claims
    4. Updates session with new token JTI
    5. Returns new access token
    
    Refresh tokens have longer expiration (7 days vs 30 minutes for access tokens).
    """
)
async def refresh(
    refresh_request: TokenRefreshRequest,
    db: Session = Depends(get_db)
):
    """
    Refresh access token.
    
    This endpoint allows clients to obtain a new access token
    without requiring the user to log in again.
    
    Requirement: 4.1
    """
    # Attempt token refresh
    success, message, data = await AuthService.refresh_token(
        refresh_token=refresh_request.refresh_token
    )
    
    if not success:
        # Determine status code based on error message
        if "Invalid token" in message or "expired" in message.lower() or "not found" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=message
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message
            )
    
    # data should never be None here if success is True
    assert data is not None, "Data should not be None after successful refresh"
    
    # Return new access token
    return TokenRefreshResponse(
        access_token=data["access_token"],
        token_type="bearer",
        expires_in=data["expires_in"]
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


# ==================== Email Verification Endpoints ====================

from app.schemas.auth import (
    SendVerificationRequest,
    SendVerificationResponse,
    VerifyEmailRequest,
    VerifyEmailResponse,
    VerificationStatusResponse
)
from app.services.email_verification_service import EmailVerificationService
from app.security import require_auth_with_session


@router.post(
    "/send-verification",
    response_model=SendVerificationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Verification email sent or status returned"},
        400: {"model": ErrorResponse, "description": "Email already verified or no email"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        429: {"model": ErrorResponse, "description": "Too many requests"},
        500: {"model": ErrorResponse, "description": "Failed to send email"}
    },
    summary="Send verification email",
    description="""
    Send or resend email verification link to the authenticated user.
    
    Rate limited to once every 5 minutes unless force_resend is True.
    """
)
async def send_verification_email(
    request: SendVerificationRequest = None,
    current_user=Depends(require_auth_with_session),
    db: Session = Depends(get_db)
):
    """
    Send verification email to authenticated user.
    
    Requires authentication. The verification email contains a link
    that expires after EMAIL_VERIFICATION_EXPIRE_HOURS (default 24h).
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    # Default request if not provided
    if request is None:
        request = SendVerificationRequest()
    
    verification_service = EmailVerificationService()
    success, message = verification_service.send_verification_email(
        db=db,
        user=current_user,
        force_resend=request.force_resend
    )
    
    if not success:
        # Check if it's a "wait before resending" message (not an error)
        if "wait" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=message
            )
        elif "already verified" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        elif "no email" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=message
            )
    
    logger.info(f"Verification email sent to user {current_user.key_id}")
    return SendVerificationResponse(message=message, success=True)


@router.post(
    "/verify-email",
    response_model=VerifyEmailResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Email verified successfully"},
        400: {"model": ErrorResponse, "description": "Invalid or expired token"},
    },
    summary="Verify email with token",
    description="""
    Verify email address using the token from verification email.
    
    This endpoint does not require authentication - the token itself
    proves the user has access to the email.
    """
)
async def verify_email(
    request: VerifyEmailRequest,
    db: Session = Depends(get_db)
):
    """
    Verify email using token from verification email.
    
    The token is valid for EMAIL_VERIFICATION_EXPIRE_HOURS after sending.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    verification_service = EmailVerificationService()
    success, message, user = verification_service.verify_email(db, request.token)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    logger.info(f"Email verified for user {user.key_id if user else 'unknown'}")
    
    return VerifyEmailResponse(
        message=message,
        success=True,
        user=UserResponse(**user.to_profile()) if user else None
    )


@router.get(
    "/verify-email/{token}",
    response_model=VerifyEmailResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Email verified successfully"},
        400: {"model": ErrorResponse, "description": "Invalid or expired token"},
    },
    summary="Verify email with token (GET)",
    description="""
    Verify email address using the token from verification email.
    
    This is a GET endpoint for convenience when clicking links in emails.
    """
)
async def verify_email_get(
    token: str,
    db: Session = Depends(get_db)
):
    """
    Verify email using token from URL (GET request).
    
    This endpoint is for clicking links directly from emails.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    verification_service = EmailVerificationService()
    success, message, user = verification_service.verify_email(db, token)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    logger.info(f"Email verified via link for user {user.key_id if user else 'unknown'}")
    
    return VerifyEmailResponse(
        message=message,
        success=True,
        user=UserResponse(**user.to_profile()) if user else None
    )


@router.get(
    "/verification-status",
    response_model=VerificationStatusResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Verification status returned"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
    summary="Get verification status",
    description="""
    Get current email verification status for authenticated user.
    
    Returns whether email is verified, if verification is required,
    and whether user can request a new verification email.
    """
)
async def get_verification_status(
    current_user=Depends(require_auth_with_session),
):
    """
    Get email verification status for authenticated user.
    """
    verification_service = EmailVerificationService()
    status_info = verification_service.get_verification_status(current_user)
    
    return VerificationStatusResponse(**status_info)
