"""
Session management service with Redis backend.

Implements session storage, validation, and single-session-per-user enforcement.
Uses Redis for fast session lookups and automatic expiration via TTL.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import redis
from app.config import settings
from app.models.session import SessionData

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions in Redis.
    
    Session storage schema:
    - session:{jti} -> Session data (TTL: SESSION_EXPIRE_MINUTES)
    - user_session:{user_key_id} -> Active session JTI (TTL: SESSION_EXPIRE_MINUTES)
    
    Requirements: 3.1, 3.2, 3.4, 3.5, 4.1
    """

    def __init__(self, redis_url: Optional[str] = None, session_expire_minutes: Optional[int] = None):
        """
        Initialize session manager with Redis connection.
        
        Args:
            redis_url: Redis connection URL (defaults to settings.SESSION_REDIS_URL)
            session_expire_minutes: Session expiration time in minutes (defaults to settings.SESSION_EXPIRE_MINUTES)
        """
        self.redis_url = redis_url or settings.SESSION_REDIS_URL
        self.session_expire_minutes = session_expire_minutes or settings.SESSION_EXPIRE_MINUTES
        self.session_ttl = self.session_expire_minutes * 60  # Convert to seconds
        
        # Create Redis client
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,  # Automatically decode bytes to strings
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Session manager connected to Redis at {self.redis_url}")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis for session management: {e}")
            raise RuntimeError(f"Redis connection failed: {e}")

    def _get_session_key(self, token_jti: str) -> str:
        """Generate Redis key for session data"""
        return f"{settings.SESSION_REDIS_PREFIX}{token_jti}"

    def _get_user_session_key(self, user_key_id: str) -> str:
        """Generate Redis key for user's active session mapping"""
        return f"user_session:{user_key_id}"

    async def create_session(
        self,
        token_jti: str,
        user_key_id: str,
        username: str,
        ip_address: str,
        user_agent: str,
    ) -> SessionData:
        """
        Create a new session in Redis.
        
        Args:
            token_jti: JWT token ID (jti claim)
            user_key_id: User's unique key ID
            username: Username
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            Created SessionData object
            
        Requirement: 3.2 - Associate session with user's unique key ID
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.session_expire_minutes)

        session_data = SessionData(
            user_key_id=user_key_id,
            username=username,
            created_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        try:
            # Store session data with TTL
            session_key = self._get_session_key(token_jti)
            self.redis_client.setex(
                session_key,
                self.session_ttl,
                json.dumps(session_data.to_dict())
            )

            logger.info(
                f"Session created: user={username}, jti={token_jti[:8]}..., "
                f"ip={ip_address}, expires_at={expires_at.isoformat()}"
            )

            return session_data

        except redis.RedisError as e:
            logger.error(f"Failed to create session for user {username}: {e}")
            raise RuntimeError(f"Session creation failed: {e}")

    async def get_session(self, token_jti: str) -> Optional[SessionData]:
        """
        Retrieve session data from Redis.
        
        Args:
            token_jti: JWT token ID (jti claim)
            
        Returns:
            SessionData if session exists and is valid, None otherwise
            
        Requirement: 4.1 - Session persists until logout or expiration
        """
        try:
            session_key = self._get_session_key(token_jti)
            session_json = self.redis_client.get(session_key)

            if not session_json:
                return None

            session_dict = json.loads(session_json)
            session_data = SessionData.from_dict(session_dict)

            # Check if session is expired (shouldn't happen with TTL, but double-check)
            if session_data.expires_at < datetime.now(timezone.utc):
                logger.warning(f"Expired session found: jti={token_jti[:8]}...")
                await self.terminate_session(token_jti)
                return None

            return session_data

        except (redis.RedisError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to retrieve session {token_jti[:8]}...: {e}")
            return None

    async def terminate_session(self, token_jti: str) -> bool:
        """
        Terminate a session and clean up all related data.
        
        Args:
            token_jti: JWT token ID (jti claim)
            
        Returns:
            True if session was terminated, False if session didn't exist
            
        Requirements: 
            3.3 - Logout terminates active session
            5.1 - Session cannot be reused after logout
            5.2 - Remove all session data
            5.4 - Ensure session cannot be reused
        """
        try:
            session_key = self._get_session_key(token_jti)
            
            # Get session data first to find user_key_id
            session_json = self.redis_client.get(session_key)
            
            if not session_json:
                logger.debug(f"Session not found for termination: jti={token_jti[:8]}...")
                return False

            # Parse session to get user_key_id
            session_dict = json.loads(session_json)
            user_key_id = session_dict["user_key_id"]
            username = session_dict["username"]

            # Delete session data
            deleted_session = self.redis_client.delete(session_key)
            
            # Delete user's active session mapping if it matches this token
            user_session_key = self._get_user_session_key(user_key_id)
            active_jti = self.redis_client.get(user_session_key)
            
            # Redis returns bytes, need to decode for comparison
            if active_jti and (active_jti.decode('utf-8') if isinstance(active_jti, bytes) else active_jti) == token_jti:
                self.redis_client.delete(user_session_key)
                logger.info(f"Terminated session: user={username}, jti={token_jti[:8]}...")
            else:
                logger.debug(
                    f"Session terminated but wasn't active session: "
                    f"user={username}, jti={token_jti[:8]}..."
                )

            return bool(deleted_session)

        except (redis.RedisError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to terminate session {token_jti[:8]}...: {e}")
            return False

    async def get_active_session(self, user_key_id: str) -> Optional[str]:
        """
        Get the active session token JTI for a user.
        
        Args:
            user_key_id: User's unique key ID
            
        Returns:
            Active session token JTI if exists, None otherwise
            
        Requirement: 3.4 - Track active sessions by user key ID
        """
        try:
            user_session_key = self._get_user_session_key(user_key_id)
            active_jti = self.redis_client.get(user_session_key)
            # Redis returns bytes, decode to string
            if active_jti:
                return active_jti.decode('utf-8') if isinstance(active_jti, bytes) else active_jti
            return None

        except redis.RedisError as e:
            logger.error(f"Failed to get active session for user {user_key_id}: {e}")
            return None

    async def set_active_session(self, user_key_id: str, token_jti: str) -> bool:
        """
        Set the active session for a user.
        
        Args:
            user_key_id: User's unique key ID
            token_jti: JWT token ID to set as active
            
        Returns:
            True if successful, False otherwise
        """
        try:
            user_session_key = self._get_user_session_key(user_key_id)
            self.redis_client.setex(
                user_session_key,
                self.session_ttl,
                token_jti
            )
            return True

        except redis.RedisError as e:
            logger.error(f"Failed to set active session for user {user_key_id}: {e}")
            return False

    async def enforce_single_session(self, user_key_id: str, new_token_jti: str) -> None:
        """
        Enforce single-session-per-user constraint.
        Terminates any existing session before setting the new one.
        
        Args:
            user_key_id: User's unique key ID
            new_token_jti: New session token JTI
            
        Requirements:
            3.1 - Terminate existing session before creating new one
            3.4 - Single-session enforcement
        """
        try:
            # Get existing active session
            old_token_jti = await self.get_active_session(user_key_id)

            # Terminate old session if it exists
            if old_token_jti and old_token_jti != new_token_jti:
                logger.info(
                    f"Terminating old session for user {user_key_id}: "
                    f"old_jti={old_token_jti[:8]}..., new_jti={new_token_jti[:8]}..."
                )
                await self.terminate_session(old_token_jti)

            # Set new active session
            await self.set_active_session(user_key_id, new_token_jti)

        except Exception as e:
            logger.error(f"Failed to enforce single session for user {user_key_id}: {e}")
            raise RuntimeError(f"Single session enforcement failed: {e}")

    async def validate_session(self, token_jti: str) -> Optional[SessionData]:
        """
        Validate a session exists and is active.
        
        Args:
            token_jti: JWT token ID (jti claim)
            
        Returns:
            SessionData if valid, None if invalid or expired
            
        Requirement: 4.2 - Validate session before granting access
        """
        session_data = await self.get_session(token_jti)
        
        if not session_data:
            return None

        # Verify this is still the active session for the user
        active_jti = await self.get_active_session(session_data.user_key_id)
        
        if active_jti != token_jti:
            logger.warning(
                f"Session validation failed: token is not active session for user "
                f"{session_data.username}, jti={token_jti[:8]}..."
            )
            return None

        return session_data

    async def refresh_session(self, token_jti: str) -> bool:
        """
        Refresh session expiration time.
        
        Args:
            token_jti: JWT token ID (jti claim)
            
        Returns:
            True if session was refreshed, False if session doesn't exist
            
        Requirement: 4.1 - Session persists across multiple requests
        """
        try:
            session_key = self._get_session_key(token_jti)
            
            # Check if session exists
            if not self.redis_client.exists(session_key):
                return False

            # Update TTL
            self.redis_client.expire(session_key, self.session_ttl)

            # Also refresh user's active session mapping
            session_json = self.redis_client.get(session_key)
            if session_json:
                session_dict = json.loads(session_json)
                user_key_id = session_dict["user_key_id"]
                user_session_key = self._get_user_session_key(user_key_id)
                self.redis_client.expire(user_session_key, self.session_ttl)

            logger.debug(f"Session refreshed: jti={token_jti[:8]}...")
            return True

        except (redis.RedisError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to refresh session {token_jti[:8]}...: {e}")
            return False

    async def cleanup_expired_sessions(self) -> int:
        """
        Manual cleanup of expired sessions (Redis TTL should handle this automatically).
        This is a backup mechanism.
        
        Returns:
            Number of sessions cleaned up
            
        Requirement: 3.5 - Remove expired/terminated sessions from registry
        """
        cleaned = 0
        try:
            # Scan for all session keys
            for key in self.redis_client.scan_iter(f"{settings.SESSION_REDIS_PREFIX}*"):
                try:
                    session_json = self.redis_client.get(key)
                    if session_json:
                        session_dict = json.loads(session_json)
                        expires_at = datetime.fromisoformat(session_dict["expires_at"])
                        
                        if expires_at < datetime.now(timezone.utc):
                            # Extract JTI from key
                            token_jti = key.replace(settings.SESSION_REDIS_PREFIX, "")
                            await self.terminate_session(token_jti)
                            cleaned += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Invalid session data found in {key}: {e}")
                    self.redis_client.delete(key)
                    cleaned += 1

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions")

            return cleaned

        except redis.RedisError as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return cleaned

    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
            logger.info("Session manager Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get or create the singleton SessionManager instance.
    
    Returns:
        SessionManager instance
    """
    global _session_manager
    
    if _session_manager is None:
        _session_manager = SessionManager()
    
    return _session_manager
