"""
Security configuration and utilities for the API
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
import re

class SecurityConfig:
    """Security settings for the API"""
    
    # Rate Limits (requests per minute)
    RATE_LIMIT_CREATE_PREDICTION = "10/minute"  # Creating predictions (compute-intensive)
    RATE_LIMIT_LIST_PREDICTIONS = "30/minute"   # Listing predictions (less intensive)
    RATE_LIMIT_GET_PREDICTION = "60/minute"     # Getting single prediction (read-only)
    RATE_LIMIT_DEFAULT = "100/minute"           # Default for other endpoints
    
    # Sequence Validation
    MAX_SEQUENCE_LENGTH = 1000    # Maximum protein length (prevents crashes)
    MIN_SEQUENCE_LENGTH = 3       # Minimum protein length
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")  # Standard 20 amino acids
    
    # Configuration Limits
    MAX_ITERATIONS = 10000        # Maximum iterations per prediction
    MIN_ITERATIONS = 100          # Minimum iterations
    MAX_AGENTS = 100              # Maximum number of agents
    MIN_AGENTS = 1                # Minimum number of agents
    MAX_CHECKPOINT_INTERVAL = 1000
    MIN_CHECKPOINT_INTERVAL = 10
    
    # Pagination Limits
    MAX_PAGE_SIZE = 100           # Maximum items per page
    DEFAULT_PAGE_SIZE = 20
    
    # File Upload (future use)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_FILE_TYPES = {".pdb", ".fasta", ".fa"}


def validate_sequence_security(sequence: str) -> tuple[bool, Optional[str]]:
    """
    Additional security validation for protein sequences.
    
    Returns:
        (is_valid, error_message)
    """
    # Check for SQL injection patterns
    sql_patterns = [
        r"(union|select|insert|update|delete|drop|create|alter)\s",
        r"(--|\*\/|\/\*)",
        r"(;|\||&&|\$\()"
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, sequence.lower()):
            return False, "Sequence contains suspicious patterns"
    
    # Check for script injection
    script_patterns = [
        r"<script",
        r"javascript:",
        r"onerror=",
        r"onload="
    ]
    
    for pattern in script_patterns:
        if re.search(pattern, sequence.lower()):
            return False, "Sequence contains invalid characters"
    
    # Check for excessive repetition (potential DoS)
    max_repetition = 50  # Max consecutive same character
    for char in set(sequence):
        if char * max_repetition in sequence:
            return False, f"Excessive repetition of amino acid '{char}' detected"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal attacks.
    """
    # Remove any path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    
    # Remove parent directory references
    filename = filename.replace("..", "_")
    
    # Keep only alphanumeric, dash, underscore, and dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename


def get_rate_limit_message(endpoint: str, limit: str) -> str:
    """
    Generate user-friendly rate limit message.
    """
    return f"Rate limit exceeded for {endpoint}. Limit: {limit}. Please try again later."


# ==================== JWT Authentication ====================

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration (loaded from settings)
# These are kept for backwards compatibility but settings take precedence

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


def get_secret_key() -> str:
    """Get JWT secret key from config."""
    from app.config import settings
    return settings.SECRET_KEY


def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against one provided by user."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token with user claims.
    
    Args:
        data: Dictionary containing claims to encode in the token.
              Should include: sub (user_key_id), username, jti (token ID)
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
        
    Example:
        token = create_access_token({
            "sub": user_key_id,
            "username": "john_doe",
            "jti": str(uuid.uuid4())
        })
    
    Requirements: 2.1, 4.1, 6.2
    """
    from app.config import settings
    
    to_encode = data.copy()
    
    # Ensure jti is present (required for session management)
    if "jti" not in to_encode:
        import uuid
        to_encode["jti"] = str(uuid.uuid4())
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, get_secret_key(), algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a JWT refresh token with longer expiration.
    
    Args:
        data: Dictionary containing claims to encode in the token.
              Should include: sub (user_key_id), jti (token ID)
        
    Returns:
        Encoded JWT refresh token string
        
    Example:
        refresh_token = create_refresh_token({
            "sub": user_key_id,
            "jti": str(uuid.uuid4())
        })
    
    Requirements: 4.1
    """
    from app.config import settings
    
    to_encode = data.copy()
    
    # Ensure jti is present (required for session management)
    if "jti" not in to_encode:
        import uuid
        to_encode["jti"] = str(uuid.uuid4())
    
    expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, get_secret_key(), algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload containing: sub, username, jti, type, iat, exp
        
    Raises:
        HTTPException: If token is invalid, expired, or malformed
        
    Requirements: 2.1, 4.1, 6.2
    """
    from app.config import settings
    
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[settings.JWT_ALGORITHM])
        
        # Validate required claims
        if "sub" not in payload:
            raise HTTPException(
                status_code=401,
                detail="Token missing required claims",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_token_type(payload: Dict[str, Any], expected_type: str) -> bool:
    """
    Verify that a decoded token has the expected type.
    
    Args:
        payload: Decoded JWT token payload
        expected_type: Expected token type ('access' or 'refresh')
        
    Returns:
        True if type matches, False otherwise
        
    Requirement: 2.1
    """
    return payload.get("type") == expected_type


def extract_jti_from_token(token: str) -> Optional[str]:
    """
    Extract JTI (JWT ID) from a token without full validation.
    Used for logout when token might be expired.
    
    Args:
        token: JWT token string
        
    Returns:
        JTI string if present, None otherwise
    """
    try:
        from app.config import settings
        # Decode without verification for logout purposes
        payload = jwt.decode(
            token, 
            get_secret_key(), 
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False}  # Allow expired tokens for logout
        )
        return payload.get("jti")
    except JWTError:
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to get current authenticated user from JWT token.
    Optional authentication - returns None if no token provided.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: Dict = Depends(get_current_user)):
            if not user:
                raise HTTPException(401, "Authentication required")
            return {"user": user}
    
    Requirement: 4.2
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = decode_token(token)
    
    # Verify token type
    if not verify_token_type(payload, "access"):
        raise HTTPException(
            status_code=401,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


async def require_auth(
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires authentication.
    Raises 401 if no valid token provided.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: Dict = Depends(require_auth)):
            return {"user": user}
    
    Requirement: 4.2
    """
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_auth_with_session(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires both valid JWT and active session.
    Integrates with SessionManager for session validation.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: Dict = Depends(require_auth_with_session)):
            return {"user": user}
    
    Requirements: 4.2, 4.5, 5.5
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = decode_token(token)
    
    # Verify token type
    if not verify_token_type(payload, "access"):
        raise HTTPException(
            status_code=401,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract JTI for session validation
    jti = payload.get("jti")
    if not jti:
        raise HTTPException(
            status_code=401,
            detail="Token missing session identifier",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Validate session in Redis
    try:
        from app.services.session_manager import get_session_manager
        session_manager = get_session_manager()
        session_data = await session_manager.validate_session(jti)
        
        if not session_data:
            raise HTTPException(
                status_code=401,
                detail="Session not found or expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Attach session info to payload
        payload["session"] = {
            "ip_address": session_data.ip_address,
            "created_at": session_data.created_at.isoformat(),
            "expires_at": session_data.expires_at.isoformat(),
        }
        
        return payload
        
    except RuntimeError as e:
        # Redis connection failed
        raise HTTPException(
            status_code=503,
            detail="Session service unavailable",
        )


# ==================== CSRF Protection ====================

# Store for CSRF tokens (in production, use Redis or database)
_csrf_tokens: Dict[str, datetime] = {}

def generate_csrf_token() -> str:
    """Generate a secure CSRF token."""
    return secrets.token_urlsafe(32)


def validate_csrf_token(token: str) -> bool:
    """
    Validate a CSRF token.
    
    Args:
        token: CSRF token to validate
        
    Returns:
        True if valid, False otherwise
    """
    if token not in _csrf_tokens:
        return False
    
    # Check expiration (tokens valid for 1 hour)
    if datetime.utcnow() - _csrf_tokens[token] > timedelta(hours=1):
        del _csrf_tokens[token]
        return False
    
    return True


def store_csrf_token(token: str) -> None:
    """Store a CSRF token with timestamp."""
    _csrf_tokens[token] = datetime.utcnow()
    
    # Clean up old tokens (keep last 1000)
    if len(_csrf_tokens) > 1000:
        # Remove oldest 100 tokens
        oldest = sorted(_csrf_tokens.items(), key=lambda x: x[1])[:100]
        for token, _ in oldest:
            del _csrf_tokens[token]


async def verify_csrf(request: Request) -> None:
    """
    FastAPI dependency to verify CSRF token.
    
    Usage:
        @app.post("/api/something", dependencies=[Depends(verify_csrf)])
        async def create_something():
            return {"status": "created"}
    """
    # Skip CSRF for GET, HEAD, OPTIONS
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return
    
    # Get CSRF token from header
    csrf_token = request.headers.get("X-CSRF-Token")
    
    if not csrf_token:
        raise HTTPException(
            status_code=403,
            detail="CSRF token missing"
        )
    
    if not validate_csrf_token(csrf_token):
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired CSRF token"
        )


# ==================== API Key Management ====================

# Store for API keys (in production, use database)
_api_keys: Dict[str, Dict[str, Any]] = {}


def generate_api_key(name: str, permissions: Optional[list] = None) -> str:
    """
    Generate a new API key.
    
    Args:
        name: Human-readable name for the API key
        permissions: Optional list of permissions
        
    Returns:
        Generated API key string
    """
    api_key = f"pp_{secrets.token_urlsafe(32)}"
    
    _api_keys[api_key] = {
        "name": name,
        "created": datetime.utcnow(),
        "permissions": permissions or ["read", "write"],
        "last_used": None
    }
    
    return api_key


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    if api_key in _api_keys:
        _api_keys[api_key]["last_used"] = datetime.utcnow()
        return True
    return False


def revoke_api_key(api_key: str) -> bool:
    """
    Revoke an API key.
    
    Args:
        api_key: API key to revoke
        
    Returns:
        True if revoked, False if key didn't exist
    """
    if api_key in _api_keys:
        del _api_keys[api_key]
        return True
    return False


async def verify_api_key(request: Request) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to verify API key from header.
    
    Usage:
        @app.get("/api/data")
        async def get_data(api_key_info: Dict = Depends(verify_api_key)):
            return {"data": "..."}
    """
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        return None
    
    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return _api_keys[api_key]
