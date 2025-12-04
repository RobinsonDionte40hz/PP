"""
Authentication service for user registration and management
"""
import uuid
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.models.user import User
from app.utils.password import hash_password, validate_password_strength, validate_credentials, verify_password
from app.services.session_manager import get_session_manager
from app.security import create_access_token, create_refresh_token


class AuthService:
    """Service for handling user authentication operations"""
    
    @staticmethod
    def register_user(
        db: Session,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: str = "user"
    ) -> Tuple[bool, str, Optional[User]]:
        """
        Register a new user with validation.
        
        Args:
            db: Database session
            username: Username for the new user
            password: Plain text password
            email: Optional email address
            
        Returns:
            Tuple of (success: bool, message: str, user: Optional[User])
            
        Example:
            >>> success, message, user = AuthService.register_user(db, "john_doe", "Pass123!", "john@example.com")
            >>> if success:
            ...     print(f"User created: {user.username}")
        """
        # Validate credentials are not empty
        valid, errors = validate_credentials(username, password)
        if not valid:
            return False, "; ".join(errors), None
        
        # Validate password strength
        valid, errors = validate_password_strength(password)
        if not valid:
            return False, "; ".join(errors), None
        
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            return False, "Username already exists", None
        
        # Check if email already exists (if provided)
        if email:
            existing_email = db.query(User).filter(User.email == email).first()
            if existing_email:
                return False, "Email already exists", None
        
        try:
            # Generate UUID for key_id
            key_id = str(uuid.uuid4())
            
            # Hash password
            password_hash = hash_password(password)
            
            # Create user
            new_user = User(
                key_id=key_id,
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            return True, "User registered successfully", new_user
            
        except IntegrityError as e:
            db.rollback()
            # Handle race condition where username was taken between check and insert
            if "username" in str(e.orig).lower():
                return False, "Username already exists", None
            elif "email" in str(e.orig).lower():
                return False, "Email already exists", None
            else:
                return False, f"Database error: {str(e.orig)}", None
        except Exception as e:
            db.rollback()
            return False, f"Registration failed: {str(e)}", None
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            db: Database session
            username: Username to search for
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """
        Get user by email.
        
        Args:
            db: Database session
            email: Email to search for
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_id(db: Session, key_id: str) -> Optional[User]:
        """
        Get user by key_id.
        
        Args:
            db: Database session
            key_id: User UUID to search for
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(User).filter(User.key_id == key_id).first()
    
    @staticmethod
    async def login_user(
        db: Session,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Authenticate user and create session.
        
        This method:
        1. Validates credentials against database
        2. Checks for existing active session
        3. Terminates old session if exists (single-session enforcement)
        4. Generates new JWT tokens (access + refresh)
        5. Creates session in Redis with user info
        6. Updates user's last_login timestamp
        
        Args:
            db: Database session
            username: Username to authenticate
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            Tuple of (success: bool, message: str, data: Optional[Dict])
            data contains: user (User object), access_token, refresh_token
            
        Requirements: 2.1, 2.2, 2.4, 2.5, 3.1, 3.2, 3.4
        
        Example:
            >>> success, message, data = await AuthService.login_user(
            ...     db, "john_doe", "Pass123!", "192.168.1.1", "Mozilla/5.0"
            ... )
            >>> if success:
            ...     print(f"Logged in: {data['user'].username}")
            ...     print(f"Access token: {data['access_token']}")
        """
        # Validate credentials are not empty
        valid, errors = validate_credentials(username, password)
        if not valid:
            return False, "; ".join(errors), None
        
        # Get user by username
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            # User doesn't exist - return generic error to prevent username enumeration
            return False, "Invalid username or password", None
        
        # Check for account lockout (Requirement 6.4: brute force protection)
        try:
            from app.utils.rate_limit import BruteForceProtection
            session_manager_temp = get_session_manager()
            brute_force = BruteForceProtection(session_manager_temp.redis_client)
            
            is_locked, retry_after = brute_force.is_locked_out(username)
            if is_locked:
                return False, f"Account temporarily locked. Try again in {retry_after} seconds", None
        except Exception as e:
            # Fail open if brute force check fails
            pass
        
        # Verify password (Requirement 2.4: constant-time comparison)
        if not verify_password(password, user.password_hash):
            # Record failed attempt (Requirement 6.4)
            try:
                brute_force.record_failed_attempt(username)
            except Exception:
                pass
            
            # Wrong password - return same generic error
            return False, "Invalid username or password", None
        
        # Check if user account is active
        if not user.is_active:
            return False, "Account is inactive", None
        
        try:
            # Generate JWT token IDs
            access_jti = str(uuid.uuid4())
            refresh_jti = str(uuid.uuid4())
            
            # Create token claims (Requirement 2.1, 2.5)
            token_data = {
                "sub": user.key_id,  # Subject: user's key_id
                "username": user.username,
                "role": user.role,  # User role for authorization
                "jti": access_jti,  # JWT ID for session tracking
            }
            
            # Generate JWT tokens (Requirement 2.1)
            access_token = create_access_token(token_data)
            refresh_token = create_refresh_token({
                "sub": user.key_id,
                "jti": refresh_jti,
            })
            
            # Get session manager
            session_manager = get_session_manager()
            
            # Enforce single-session-per-user (Requirements 3.1, 3.4)
            # This will terminate any existing session before creating new one
            await session_manager.enforce_single_session(user.key_id, access_jti)
            
            # Create new session in Redis (Requirement 3.2)
            session_data = await session_manager.create_session(
                token_jti=access_jti,
                user_key_id=user.key_id,
                username=user.username,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            
            # Reset failed attempts on successful login (Requirement 6.4)
            try:
                brute_force.reset_failed_attempts(username)
            except Exception:
                pass
            
            # Update user's last_login timestamp
            user.last_login = datetime.now(timezone.utc)
            db.commit()
            db.refresh(user)
            
            # Return success with user and tokens
            from app.config import settings
            return True, "Login successful", {
                "user": user,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # In seconds
            }
            
        except RuntimeError as e:
            db.rollback()
            # Session/Redis error
            return False, f"Session creation failed: {str(e)}", None
        except Exception as e:
            db.rollback()
            return False, f"Login failed: {str(e)}", None
    
    @staticmethod
    async def logout_user(
        token: str,
        user_key_id: str,
    ) -> Tuple[bool, str]:
        """
        Logout user and terminate session.
        
        This method:
        1. Extracts token JTI from JWT (even if expired)
        2. Deletes session from Redis
        3. Removes user's active session mapping
        4. Invalidates JWT token (session deletion prevents reuse)
        
        Args:
            token: JWT access token (can be expired)
            user_key_id: User's unique key ID for verification
            
        Returns:
            Tuple of (success: bool, message: str)
            
        Requirements: 3.3, 5.1, 5.2, 5.4
        
        Example:
            >>> success, message = await AuthService.logout_user(
            ...     token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            ...     user_key_id="550e8400-e29b-41d4-a716-446655440000"
            ... )
            >>> if success:
            ...     print("User logged out successfully")
        """
        try:
            # Extract JTI from token (allow expired tokens for logout)
            from app.security import extract_jti_from_token
            
            jti = extract_jti_from_token(token)
            
            if not jti:
                # Token is invalid or malformed, but still try to succeed
                # (graceful logout even with invalid token)
                return True, "Logout successful"
            
            # Get session manager
            session_manager = get_session_manager()
            
            # Terminate session (Requirements 3.3, 5.1, 5.2, 5.4)
            # This will:
            # - Delete session data from Redis
            # - Remove user's active session mapping
            # - Ensure session cannot be reused
            terminated = await session_manager.terminate_session(jti)
            
            if terminated:
                return True, "Logout successful"
            else:
                # Session not found (already logged out or expired)
                # Still return success for idempotency
                return True, "Logout successful"
                
        except RuntimeError as e:
            # Session/Redis error
            return False, f"Logout failed: {str(e)}"
        except Exception as e:
            return False, f"Logout failed: {str(e)}"
    
    @staticmethod
    async def refresh_token(
        refresh_token: str,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Refresh access token using refresh token.
        
        This method:
        1. Validates refresh token (signature, expiration, type)
        2. Checks that session still exists in Redis
        3. Generates new access token with same claims
        4. Updates session expiration in Redis
        5. Returns new access token
        
        Args:
            refresh_token: JWT refresh token
            
        Returns:
            Tuple of (success: bool, message: str, data: Optional[Dict])
            data contains: access_token, expires_in
            
        Requirement: 4.1
        
        Example:
            >>> success, message, data = await AuthService.refresh_token(
            ...     refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            ... )
            >>> if success:
            ...     print(f"New access token: {data['access_token']}")
        """
        try:
            # Decode and validate refresh token
            from app.security import decode_token, verify_token_type, create_access_token
            
            payload = decode_token(refresh_token)
            
            # Verify token type is 'refresh'
            if not verify_token_type(payload, "refresh"):
                return False, "Invalid token type", None
            
            # Extract user info
            user_key_id = payload.get("sub")
            refresh_jti = payload.get("jti")
            
            if not user_key_id or not refresh_jti:
                return False, "Invalid token claims", None
            
            # Get session manager
            session_manager = get_session_manager()
            
            # Check if user still has an active session
            # Note: Refresh tokens are associated with the user, not a specific session
            # We just verify the user has some active session
            active_session_jti = await session_manager.get_active_session(user_key_id)
            
            if not active_session_jti:
                return False, "No active session found", None
            
            # Get session data to retrieve username
            session_data = await session_manager.get_session(active_session_jti)
            
            if not session_data:
                return False, "Session not found or expired", None
            
            # Generate new access token with new JTI
            new_access_jti = str(uuid.uuid4())
            token_data = {
                "sub": user_key_id,
                "username": session_data.username,
                "jti": new_access_jti,
            }
            
            new_access_token = create_access_token(token_data)
            
            # Update the active session to use the new access token JTI
            # First terminate old session
            await session_manager.terminate_session(active_session_jti)
            
            # Create new session with new JTI
            await session_manager.create_session(
                token_jti=new_access_jti,
                user_key_id=user_key_id,
                username=session_data.username,
                ip_address=session_data.ip_address,
                user_agent=session_data.user_agent,
            )
            
            # Set as active session
            await session_manager.set_active_session(user_key_id, new_access_jti)
            
            # Return new access token
            from app.config import settings
            return True, "Token refreshed successfully", {
                "access_token": new_access_token,
                "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            }
            
        except Exception as e:
            return False, f"Token refresh failed: {str(e)}", None
