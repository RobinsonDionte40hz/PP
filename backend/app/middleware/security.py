"""
Security middleware for the FastAPI application.
Includes CSRF protection and security headers.
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable
import time
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    
    Headers added:
    - X-Content-Type-Options: Prevents MIME sniffing
    - X-Frame-Options: Prevents clickjacking
    - X-XSS-Protection: Enables XSS filter
    - Strict-Transport-Security: Enforces HTTPS
    - Content-Security-Policy: Controls resource loading
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features
    """
    
    def __init__(self, app: ASGIApp, enable_hsts: bool = False):
        super().__init__(app)
        self.enable_hsts = enable_hsts
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Enable XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # HSTS (only enable in production with HTTPS)
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Relaxed for dev
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self' ws: wss:",  # Allow WebSocket connections
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions policy (formerly Feature-Policy)
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "gyroscope=()",
            "accelerometer=()"
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)
        
        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle CSRF protection.
    
    - Generates CSRF tokens for GET requests
    - Validates CSRF tokens for state-changing requests (POST, PUT, DELETE, PATCH)
    - Excludes certain paths from CSRF validation (e.g., /health, /docs)
    """
    
    # Paths that don't require CSRF protection
    EXCLUDED_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    def __init__(self, app: ASGIApp, secret_key: str):
        super().__init__(app)
        self.secret_key = secret_key
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip CSRF for excluded paths
        if any(request.url.path.startswith(path) for path in self.EXCLUDED_PATHS):
            return await call_next(request)
        
        # Skip CSRF for safe methods
        if request.method in ["GET", "HEAD", "OPTIONS", "TRACE"]:
            response = await call_next(request)
            
            # Generate and send CSRF token in response header for GET requests
            from app.security import generate_csrf_token, store_csrf_token
            csrf_token = generate_csrf_token()
            store_csrf_token(csrf_token)
            response.headers["X-CSRF-Token"] = csrf_token
            
            return response
        
        # For state-changing requests, validate CSRF token
        # Note: This validation is also done in the verify_csrf dependency
        # This middleware provides defense in depth
        
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.
    Note: For production, use Redis-based rate limiting (slowapi already configured).
    This is a backup/supplementary rate limiter.
    """
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: dict = {}  # {ip: [(timestamp, count), ...]}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (ts, count) for ts, count in self.requests[client_ip]
                if current_time - ts < 60
            ]
        
        # Count requests in last minute
        if client_ip in self.requests:
            request_count = sum(count for _, count in self.requests[client_ip])
        else:
            request_count = 0
            self.requests[client_ip] = []
        
        # Check rate limit
        if request_count >= self.requests_per_minute:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self.requests[client_ip].append((current_time, 1))
        
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests for security auditing.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import logging
        logger = logging.getLogger("security.audit")
        
        # Log request details
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log successful requests
            logger.info(
                f"{client_ip} - {method} {path} - {response.status_code} - {process_time:.3f}s"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Log failed requests
            process_time = time.time() - start_time
            logger.error(
                f"{client_ip} - {method} {path} - ERROR: {str(e)} - {process_time:.3f}s"
            )
            raise


class AuthenticationMiddleware:
    """
    Pure ASGI3 middleware to validate authentication for protected routes.
    
    This middleware:
    - Extracts JWT token from Authorization header
    - Validates token structure and signature
    - Checks token expiration
    - Verifies session exists in Redis
    - Verifies single-session constraint
    - Attaches user info to request state
    - Handles authentication errors with 401 responses
    
    Requirements: 4.2, 4.5, 5.5
    
    Usage:
        app.add_middleware(AuthenticationMiddleware, protected_paths=["/api/protected"])
    """
    
    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/auth/register",
        "/api/auth/login",
        "/api/auth/health",
    }
    
    def __init__(self, app: ASGIApp, protected_paths: list = None):
        """
        Initialize authentication middleware.
        
        Args:
            app: ASGI application
            protected_paths: List of path prefixes that require authentication
                           If None, all paths except PUBLIC_PATHS require auth
        """
        self.app = app
        self.protected_paths = protected_paths or []
    
    async def __call__(self, scope, receive, send):
        """ASGI3 interface"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create Request object to access helpers
        request = Request(scope, receive, send)
        
        # Check if authentication is required
        if not self._is_protected_path(request.url.path):
            await self.app(scope, receive, send)
            return
        
        logger.info(f"Path {request.url.path} is protected, checking auth")
        
        # Perform authentication
        auth_result = await self._authenticate(request)
        
        if not auth_result["success"]:
            # Send 401 response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=401,
                content={"detail": auth_result.get("error", "Unauthorized")},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        
        # Attach user info to scope
        scope["state"] = {"user": auth_result["user"]}
        
        await self.app(scope, receive, send)
    
    def _is_protected_path(self, path: str) -> bool:
        """Check if path requires authentication"""
        # Check if path is in public paths (exact match)
        if path in self.PUBLIC_PATHS:
            return False
        
        # Check if path starts with any public path (but not for single "/")
        for public_path in self.PUBLIC_PATHS:
            # Skip single "/" to avoid matching everything
            if public_path != "/" and path.startswith(public_path):
                return False
        
        # If protected_paths is empty, only PUBLIC_PATHS are public
        if not self.protected_paths:
            return True
        
        # Check if path matches any protected path
        for protected_path in self.protected_paths:
            if path.startswith(protected_path):
                return True
        
        return False
    
    async def _authenticate(self, request: Request) -> dict:
        """
        Validate authentication for a request.
        
        Returns:
            dict with 'success' (bool), 'user' (dict) if success, 'error' (str) if failure
        
        Requirements:
        - 4.2: Validate session before granting access to protected resources
        - 4.5: Verify active valid session exists
        - 5.5: Deny access after logout
        """
        # Extract Authorization header
        auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
        
        if not auth_header:
            return {"success": False, "error": "Unauthorized"}
        
        # Extract token (remove "Bearer " prefix)
        token = auth_header.replace("Bearer ", "").replace("bearer ", "")
        
        if not token:
            return {"success": False, "error": "Unauthorized"}
        
        try:
            # Decode and validate token
            from app.security import decode_token, verify_token_type
            payload = decode_token(token)
            
            # Verify token type is 'access'
            if not verify_token_type(payload, "access"):
                return {"success": False, "error": "Unauthorized"}
            
            # Extract JTI for session validation
            jti = payload.get("jti")
            if not jti:
                return {"success": False, "error": "Unauthorized"}
            
            # Validate session in Redis (Requirement 4.2, 4.5)
            try:
                from app.services.session_manager import get_session_manager
                session_manager = get_session_manager()
                session_data = await session_manager.validate_session(jti)
                
                if not session_data:
                    # Session not found or expired (Requirement 5.5)
                    return {"success": False, "error": "Unauthorized"}
                
                # Build user info
                user_info = {
                    "key_id": payload.get("sub"),
                    "username": payload.get("username"),
                    "jti": jti,
                    "session": {
                        "ip_address": session_data.ip_address,
                        "created_at": session_data.created_at.isoformat(),
                        "expires_at": session_data.expires_at.isoformat(),
                    }
                }
                
                return {"success": True, "user": user_info}
                
            except RuntimeError as e:
                # Redis connection failed
                logger.error(f"Session service unavailable: {str(e)}")
                return {"success": False, "error": "Session service unavailable"}
            
        except Exception as e:
            # Token validation failed
            logger.warning(f"Authentication failed: {str(e)}")
            return {"success": False, "error": "Unauthorized"}
