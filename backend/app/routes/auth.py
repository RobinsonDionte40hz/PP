"""
Authentication API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
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
    ErrorResponse,
    OAuthConfigResponse,
    OAuthInitiateResponse,
    OAuthCallbackRequest,
    OAuthLoginResponse,
    OAuthUnlinkRequest,
    OAuthLinkedAccountsResponse,
    SetPasswordRequest,
)
from app.services.auth_service import AuthService
from app.utils.rate_limit import RateLimiter
from app.services.session_manager import get_session_manager
from app.security import require_auth_with_session

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
    
    # Verify CAPTCHA token (bot protection)
    from app.services.captcha_service import CaptchaService
    if CaptchaService.is_enabled():
        captcha_valid, captcha_message, captcha_score = await CaptchaService.verify_token(
            token=request.captcha_token or "",
            remote_ip=client_ip,
            expected_action="register"
        )
        if not captcha_valid:
            logger.warning(
                f"CAPTCHA verification failed: ip={client_ip}, message={captcha_message}, score={captcha_score}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=captcha_message
            )
        logger.debug(f"CAPTCHA verified: ip={client_ip}, score={captcha_score}")
    
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
    db: Session = Depends(get_db),
    current_user=Depends(require_auth_with_session),
):
    """
    Get email verification status for authenticated user.
    """
    from app.models.user import User
    
    user = db.query(User).filter(User.key_id == current_user["sub"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    verification_service = EmailVerificationService()
    status_info = verification_service.get_verification_status(user)
    
    return VerificationStatusResponse(**status_info)


@router.get(
    "/captcha-config",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "CAPTCHA configuration returned"},
    },
    summary="Get CAPTCHA configuration",
    description="""
    Get CAPTCHA configuration for frontend integration.
    
    Returns whether CAPTCHA is enabled, the provider type,
    and the public site key for widget initialization.
    """
)
async def get_captcha_config():
    """
    Get CAPTCHA configuration for frontend use.
    
    This endpoint provides the public configuration needed
    to initialize the CAPTCHA widget on the frontend.
    No authentication required.
    """
    from app.services.captcha_service import CaptchaService
    
    return {
        "enabled": CaptchaService.is_enabled(),
        "provider": CaptchaService.get_provider(),
        "site_key": CaptchaService.get_site_key()
    }


# -------------------- OAuth Endpoints --------------------


@router.get(
    "/oauth-config",
    response_model=OAuthConfigResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "OAuth configuration returned"},
    },
    summary="Get OAuth configuration",
    description="""
    Get OAuth provider configuration for frontend integration.
    
    Returns which OAuth providers are enabled and their client IDs
    for initializing OAuth buttons on the frontend.
    """
)
async def get_oauth_config():
    """
    Get OAuth configuration for frontend use.
    
    This endpoint provides the public configuration needed
    to display OAuth login buttons on the frontend.
    No authentication required.
    """
    from app.services.oauth_service import OAuthService
    
    return OAuthService.get_oauth_config()


@router.get(
    "/google",
    response_model=OAuthInitiateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Authorization URL generated"},
        400: {"model": ErrorResponse, "description": "Google OAuth not configured"},
    },
    summary="Initiate Google OAuth",
    description="""
    Generate Google OAuth authorization URL.
    
    Returns a URL to redirect the user to Google's login page,
    along with a state token for CSRF protection.
    
    The state token should be stored (e.g., in session storage)
    and validated when the callback is received.
    """
)
async def initiate_google_oauth(
    redirect_uri: str = None,
    session_manager = Depends(get_session_manager)
):
    """
    Initiate Google OAuth flow.
    
    Generates authorization URL and state token.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    from app.services.oauth_service import OAuthService
    
    if not OAuthService.is_google_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google OAuth is not configured"
        )
    
    # Generate state for CSRF protection
    state = OAuthService.generate_state()
    
    # Store state in Redis with 10-minute TTL
    try:
        state_key = f"oauth_state:{state}"
        session_manager.redis_client.setex(state_key, 600, "google")
    except Exception as e:
        logger.error(f"Failed to store OAuth state: {str(e)}")
    
    # Generate authorization URL
    url = OAuthService.get_google_authorization_url(state=state, redirect_uri=redirect_uri)
    
    logger.info(f"Google OAuth initiated, state={state[:8]}...")
    
    return OAuthInitiateResponse(
        authorization_url=url,
        state=state
    )


@router.post(
    "/google/callback",
    response_model=OAuthLoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Login successful"},
        400: {"model": ErrorResponse, "description": "Invalid code or state"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Account inactive"},
    },
    summary="Google OAuth callback",
    description="""
    Handle Google OAuth callback.
    
    Exchanges authorization code for tokens, retrieves user info,
    and creates or authenticates the user account.
    
    If the user doesn't exist, a new account is created.
    If a user with the same email exists, the Google account is linked.
    """
)
async def google_oauth_callback(
    request: OAuthCallbackRequest,
    redirect_uri: str = None,
    db: Session = Depends(get_db),
    session_manager = Depends(get_session_manager)
):
    """
    Handle Google OAuth callback.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    from app.services.oauth_service import OAuthService
    from app.security import create_access_token, create_refresh_token
    from app.config import settings
    
    # Validate state (CSRF protection)
    try:
        state_key = f"oauth_state:{request.state}"
        stored_provider = session_manager.redis_client.get(state_key)
        if not stored_provider:
            logger.warning(f"Invalid OAuth state: {request.state[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state token"
            )
        # Handle both bytes and string from Redis
        provider_str = stored_provider.decode() if isinstance(stored_provider, bytes) else stored_provider
        if provider_str != "google":
            logger.warning(f"Invalid OAuth state provider: {provider_str}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state token"
            )
        # Delete state after use
        session_manager.redis_client.delete(state_key)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate OAuth state: {str(e)}")
    
    # Build redirect URI - must match what was used in authorization
    actual_redirect_uri = redirect_uri or f"{settings.FRONTEND_URL}/auth/google/callback"
    
    # Exchange code for user info
    success, message, user_info = await OAuthService.exchange_google_code(
        code=request.code,
        redirect_uri=actual_redirect_uri
    )
    
    if not success or not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )
    
    # Authenticate or create user
    success, message, user, is_new_user = OAuthService.authenticate_or_create_oauth_user(
        db=db,
        provider="google",
        user_info=user_info
    )
    
    if not success or not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Create JWT tokens
    access_token = create_access_token(
        data={"sub": user.key_id, "username": user.username, "role": user.role}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.key_id, "username": user.username, "role": user.role}
    )
    
    # Create session
    try:
        session_manager.create_session(
            user_id=user.key_id,
            user_data={
                "username": user.username,
                "role": user.role,
                "email": user.email
            },
            access_token=access_token
        )
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
    
    logger.info(
        f"Google OAuth login: username={user.username}, is_new={is_new_user}"
    )
    
    return OAuthLoginResponse(
        message="Login successful" if not is_new_user else "Account created successfully",
        user=UserResponse(**user.to_profile()),
        tokens=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        ),
        is_new_user=is_new_user
    )


@router.get(
    "/github",
    response_model=OAuthInitiateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Authorization URL generated"},
        400: {"model": ErrorResponse, "description": "GitHub OAuth not configured"},
    },
    summary="Initiate GitHub OAuth",
    description="""
    Generate GitHub OAuth authorization URL.
    
    Returns a URL to redirect the user to GitHub's login page,
    along with a state token for CSRF protection.
    """
)
async def initiate_github_oauth(
    redirect_uri: str = None,
    session_manager = Depends(get_session_manager)
):
    """
    Initiate GitHub OAuth flow.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    from app.services.oauth_service import OAuthService
    
    if not OAuthService.is_github_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub OAuth is not configured"
        )
    
    # Generate state for CSRF protection
    state = OAuthService.generate_state()
    
    # Store state in Redis with 10-minute TTL
    try:
        state_key = f"oauth_state:{state}"
        session_manager.redis_client.setex(state_key, 600, "github")
    except Exception as e:
        logger.error(f"Failed to store OAuth state: {str(e)}")
    
    # Generate authorization URL
    url = OAuthService.get_github_authorization_url(state=state, redirect_uri=redirect_uri)
    
    logger.info(f"GitHub OAuth initiated, state={state[:8]}...")
    
    return OAuthInitiateResponse(
        authorization_url=url,
        state=state
    )


@router.post(
    "/github/callback",
    response_model=OAuthLoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Login successful"},
        400: {"model": ErrorResponse, "description": "Invalid code or state"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Account inactive"},
    },
    summary="GitHub OAuth callback",
    description="""
    Handle GitHub OAuth callback.
    
    Exchanges authorization code for tokens, retrieves user info,
    and creates or authenticates the user account.
    """
)
async def github_oauth_callback(
    request: OAuthCallbackRequest,
    redirect_uri: str = None,
    db: Session = Depends(get_db),
    session_manager = Depends(get_session_manager)
):
    """
    Handle GitHub OAuth callback.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    from app.services.oauth_service import OAuthService
    from app.security import create_access_token, create_refresh_token
    from app.config import settings
    
    # Validate state (CSRF protection)
    try:
        state_key = f"oauth_state:{request.state}"
        stored_provider = session_manager.redis_client.get(state_key)
        if not stored_provider:
            logger.warning(f"Invalid OAuth state: {request.state[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state token"
            )
        # Handle both bytes and string from Redis
        provider_str = stored_provider.decode() if isinstance(stored_provider, bytes) else stored_provider
        if provider_str != "github":
            logger.warning(f"Invalid OAuth state provider: {provider_str}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state token"
            )
        # Delete state after use
        session_manager.redis_client.delete(state_key)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate OAuth state: {str(e)}")
    
    # Build redirect URI - must match what was used in authorization
    actual_redirect_uri = redirect_uri or f"{settings.FRONTEND_URL}/auth/github/callback"
    
    # Exchange code for user info
    success, message, user_info = await OAuthService.exchange_github_code(
        code=request.code,
        redirect_uri=actual_redirect_uri
    )
    
    if not success or not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )
    
    # Authenticate or create user
    success, message, user, is_new_user = OAuthService.authenticate_or_create_oauth_user(
        db=db,
        provider="github",
        user_info=user_info
    )
    
    if not success or not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Create JWT tokens
    access_token = create_access_token(
        data={"sub": user.key_id, "username": user.username, "role": user.role}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.key_id, "username": user.username, "role": user.role}
    )
    
    # Create session
    try:
        session_manager.create_session(
            user_id=user.key_id,
            user_data={
                "username": user.username,
                "role": user.role,
                "email": user.email
            },
            access_token=access_token
        )
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
    
    logger.info(
        f"GitHub OAuth login: username={user.username}, is_new={is_new_user}"
    )
    
    return OAuthLoginResponse(
        message="Login successful" if not is_new_user else "Account created successfully",
        user=UserResponse(**user.to_profile()),
        tokens=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        ),
        is_new_user=is_new_user
    )


# -------------------- Account Linking Endpoints --------------------


@router.get(
    "/linked-accounts",
    response_model=OAuthLinkedAccountsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Linked accounts returned"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
    summary="Get linked OAuth accounts",
    description="""
    Get list of OAuth accounts linked to the current user.
    
    Returns which OAuth providers are linked and whether
    the user has a password set (for account security).
    """
)
async def get_linked_accounts(
    db: Session = Depends(get_db),
    current_user=Depends(require_auth_with_session),
):
    """
    Get linked OAuth accounts for current user.
    """
    from app.models.user import User
    
    user = db.query(User).filter(User.key_id == current_user["sub"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return OAuthLinkedAccountsResponse(
        google=user.google_id is not None,
        github=user.github_id is not None,
        has_password=user.password_hash is not None
    )


@router.post(
    "/link/{provider}",
    response_model=OAuthInitiateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Authorization URL generated for linking"},
        400: {"model": ErrorResponse, "description": "Invalid provider or already linked"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
    summary="Initiate OAuth account linking",
    description="""
    Generate OAuth authorization URL for linking an account.
    
    Use this to add Google or GitHub login to an existing account.
    The user will be redirected to the OAuth provider to authorize.
    """
)
async def initiate_oauth_link(
    provider: str,
    redirect_uri: str = None,
    db: Session = Depends(get_db),
    session_manager = Depends(get_session_manager),
    current_user=Depends(require_auth_with_session),
):
    """
    Initiate OAuth account linking flow.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    from app.services.oauth_service import OAuthService
    from app.models.user import User
    
    provider = provider.lower()
    if provider not in ['google', 'github']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider must be 'google' or 'github'"
        )
    
    # Check if provider is enabled
    if provider == "google" and not OAuthService.is_google_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google OAuth is not configured"
        )
    if provider == "github" and not OAuthService.is_github_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub OAuth is not configured"
        )
    
    # Check if already linked
    user = db.query(User).filter(User.key_id == current_user["sub"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if provider == "google" and user.google_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account is already linked"
        )
    if provider == "github" and user.github_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub account is already linked"
        )
    
    # Generate state for CSRF protection (includes user ID for linking)
    state = OAuthService.generate_state()
    
    # Store state with user ID for linking
    try:
        state_key = f"oauth_state:{state}"
        state_data = f"link:{provider}:{user.key_id}"
        session_manager.redis_client.setex(state_key, 600, state_data)
    except Exception as e:
        logger.error(f"Failed to store OAuth state: {str(e)}")
    
    # Generate authorization URL
    if provider == "google":
        url = OAuthService.get_google_authorization_url(state=state, redirect_uri=redirect_uri)
    else:
        url = OAuthService.get_github_authorization_url(state=state, redirect_uri=redirect_uri)
    
    logger.info(f"OAuth link initiated: user={user.username}, provider={provider}")
    
    return OAuthInitiateResponse(
        authorization_url=url,
        state=state
    )


@router.post(
    "/link/{provider}/callback",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Account linked successfully"},
        400: {"model": ErrorResponse, "description": "Invalid code or state"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        409: {"model": ErrorResponse, "description": "OAuth account already linked to another user"},
    },
    summary="Complete OAuth account linking",
    description="""
    Complete OAuth account linking after authorization.
    
    Exchanges code for tokens, retrieves OAuth user info,
    and links the OAuth account to the current user.
    """
)
async def complete_oauth_link(
    provider: str,
    request: OAuthCallbackRequest,
    redirect_uri: str = None,
    db: Session = Depends(get_db),
    session_manager = Depends(get_session_manager),
):
    """
    Complete OAuth account linking.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    from app.services.oauth_service import OAuthService
    from app.models.user import User
    
    provider = provider.lower()
    if provider not in ['google', 'github']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider must be 'google' or 'github'"
        )
    
    # Validate state (CSRF protection and get user ID)
    user_id = None
    try:
        state_key = f"oauth_state:{request.state}"
        stored_data = session_manager.redis_client.get(state_key)
        if not stored_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state token"
            )
        
        # Handle both bytes and string from Redis
        data_str = stored_data.decode() if isinstance(stored_data, bytes) else stored_data
        parts = data_str.split(":")
        if len(parts) != 3 or parts[0] != "link" or parts[1] != provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state token"
            )
        
        user_id = parts[2]
        session_manager.redis_client.delete(state_key)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate OAuth state: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to validate state token"
        )
    
    # Get user
    user = db.query(User).filter(User.key_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Build redirect URI - must match what was used in authorization
    actual_redirect_uri = redirect_uri or f"{settings.FRONTEND_URL}/auth/{provider}/callback"
    
    # Exchange code for user info
    if provider == "google":
        success, message, user_info = await OAuthService.exchange_google_code(
            code=request.code,
            redirect_uri=actual_redirect_uri
        )
    else:
        success, message, user_info = await OAuthService.exchange_github_code(
            code=request.code,
            redirect_uri=actual_redirect_uri
        )
    
    if not success or not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )
    
    # Link account
    oauth_id = user_info.get("id")
    email = user_info.get("email")
    email_verified = user_info.get("email_verified", False)
    
    success, message = OAuthService.link_oauth_account(
        db=db,
        user=user,
        provider=provider,
        oauth_id=oauth_id,
        email=email,
        email_verified=email_verified
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=message
        )
    
    logger.info(f"OAuth account linked: user={user.username}, provider={provider}")
    
    return {"message": message, "success": True}


@router.delete(
    "/unlink/{provider}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Account unlinked successfully"},
        400: {"model": ErrorResponse, "description": "Cannot unlink only auth method"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
    summary="Unlink OAuth account",
    description="""
    Unlink an OAuth account from the current user.
    
    Cannot unlink if it's the user's only authentication method
    (no password and no other OAuth accounts linked).
    """
)
async def unlink_oauth_account(
    provider: str,
    db: Session = Depends(get_db),
    current_user=Depends(require_auth_with_session),
):
    """
    Unlink OAuth account from current user.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    from app.services.oauth_service import OAuthService
    from app.models.user import User
    
    provider = provider.lower()
    if provider not in ['google', 'github']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider must be 'google' or 'github'"
        )
    
    user = db.query(User).filter(User.key_id == current_user["sub"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Check if provider is linked
    if provider == "google" and not user.google_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account is not linked"
        )
    if provider == "github" and not user.github_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub account is not linked"
        )
    
    success, message = OAuthService.unlink_oauth_account(
        db=db,
        user=user,
        provider=provider
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    logger.info(f"OAuth account unlinked: user={user.username}, provider={provider}")
    
    return {"message": message, "success": True}


@router.post(
    "/set-password",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Password set successfully"},
        400: {"model": ErrorResponse, "description": "Invalid password or already has password"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
    summary="Set password for OAuth user",
    description="""
    Set a password for an OAuth-only user.
    
    Allows users who signed up via OAuth to also use
    username/password login. Required before unlinking
    all OAuth accounts.
    """
)
async def set_password(
    request: SetPasswordRequest,
    db: Session = Depends(get_db),
    current_user=Depends(require_auth_with_session),
):
    """
    Set password for OAuth-only user.
    """
    import logging
    logger = logging.getLogger("security.auth")
    
    from app.models.user import User
    from app.utils.password import hash_password, validate_password_strength
    
    user = db.query(User).filter(User.key_id == current_user["sub"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Check if user already has a password
    if user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has a password. Use change-password endpoint instead."
        )
    
    # Validate password strength
    valid, errors = validate_password_strength(request.password)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(errors)
        )
    
    # Set password
    user.password_hash = hash_password(request.password)
    db.commit()
    
    logger.info(f"Password set for OAuth user: {user.username}")
    
    return {"message": "Password set successfully", "success": True}

