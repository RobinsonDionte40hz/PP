"""
Rate limiting utilities for authentication endpoints.

Implements configurable rate limits to prevent abuse and brute force attacks.
Requirements: 6.4
"""
from typing import Optional
from datetime import datetime, timedelta
import redis
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Redis-based rate limiter for authentication endpoints.
    
    Tracks attempts per IP address with configurable windows and limits.
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize rate limiter.
        
        Args:
            redis_client: Redis client for storing rate limit data
        """
        self.redis = redis_client
    
    def _get_key(self, endpoint: str, identifier: str) -> str:
        """Generate Redis key for rate limiting"""
        return f"ratelimit:{endpoint}:{identifier}"
    
    def check_rate_limit(
        self,
        endpoint: str,
        identifier: str,
        max_attempts: int,
        window_seconds: int
    ) -> tuple[bool, Optional[int]]:
        """
        Check if rate limit has been exceeded.
        
        Args:
            endpoint: Endpoint name (e.g., "register", "login")
            identifier: Unique identifier (usually IP address)
            max_attempts: Maximum attempts allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed: bool, retry_after: Optional[int])
            retry_after is seconds until rate limit resets (None if allowed)
        """
        key = self._get_key(endpoint, identifier)
        
        try:
            # Get current count
            current_count = self.redis.get(key)
            
            if current_count is None:
                # First attempt in window
                self.redis.setex(key, window_seconds, 1)
                return True, None
            
            current_count = int(current_count)
            
            if current_count >= max_attempts:
                # Rate limit exceeded
                ttl = self.redis.ttl(key)
                logger.warning(
                    f"Rate limit exceeded: endpoint={endpoint}, "
                    f"identifier={identifier}, count={current_count}, "
                    f"max={max_attempts}"
                )
                return False, ttl if ttl > 0 else window_seconds
            
            # Increment count
            self.redis.incr(key)
            return True, None
            
        except redis.RedisError as e:
            # Redis error - fail open (allow request)
            logger.error(f"Rate limit check failed: {e}")
            return True, None
    
    def reset_rate_limit(self, endpoint: str, identifier: str) -> None:
        """
        Reset rate limit for an identifier.
        
        Args:
            endpoint: Endpoint name
            identifier: Unique identifier
        """
        key = self._get_key(endpoint, identifier)
        try:
            self.redis.delete(key)
        except redis.RedisError as e:
            logger.error(f"Failed to reset rate limit: {e}")


class BruteForceProtection:
    """
    Brute force protection for login attempts.
    
    Tracks failed login attempts per username and implements progressive delays.
    Requirements: 6.4
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize brute force protection.
        
        Args:
            redis_client: Redis client for storing attempt data
        """
        self.redis = redis_client
        
        # Progressive lockout thresholds
        self.thresholds = [
            (3, 60),      # 3 attempts: 1 minute lockout
            (5, 300),     # 5 attempts: 5 minutes lockout
            (10, 900),    # 10 attempts: 15 minutes lockout
            (20, 3600),   # 20 attempts: 1 hour lockout
        ]
    
    def _get_key(self, username: str) -> str:
        """Generate Redis key for failed attempts"""
        return f"bruteforce:login:{username}"
    
    def _get_lockout_key(self, username: str) -> str:
        """Generate Redis key for lockout status"""
        return f"bruteforce:lockout:{username}"
    
    def record_failed_attempt(self, username: str) -> None:
        """
        Record a failed login attempt.
        
        Args:
            username: Username that failed authentication
        """
        key = self._get_key(username)
        
        try:
            # Increment failed attempts (24 hour window)
            count = self.redis.incr(key)
            self.redis.expire(key, 86400)  # 24 hours
            
            # Check if lockout threshold reached
            for threshold, lockout_seconds in self.thresholds:
                if count >= threshold:
                    lockout_key = self._get_lockout_key(username)
                    self.redis.setex(lockout_key, lockout_seconds, count)
                    logger.warning(
                        f"Account locked due to failed attempts: "
                        f"username={username}, attempts={count}, "
                        f"lockout_duration={lockout_seconds}s"
                    )
                    break
                    
        except redis.RedisError as e:
            logger.error(f"Failed to record login attempt: {e}")
    
    def is_locked_out(self, username: str) -> tuple[bool, Optional[int]]:
        """
        Check if account is locked out.
        
        Args:
            username: Username to check
            
        Returns:
            Tuple of (is_locked: bool, retry_after: Optional[int])
            retry_after is seconds until lockout expires (None if not locked)
        """
        lockout_key = self._get_lockout_key(username)
        
        try:
            if self.redis.exists(lockout_key):
                ttl = self.redis.ttl(lockout_key)
                return True, ttl if ttl > 0 else 60
            return False, None
            
        except redis.RedisError as e:
            logger.error(f"Failed to check lockout status: {e}")
            return False, None  # Fail open
    
    def reset_failed_attempts(self, username: str) -> None:
        """
        Reset failed attempts after successful login.
        
        Args:
            username: Username to reset
        """
        key = self._get_key(username)
        lockout_key = self._get_lockout_key(username)
        
        try:
            self.redis.delete(key)
            self.redis.delete(lockout_key)
        except redis.RedisError as e:
            logger.error(f"Failed to reset failed attempts: {e}")
    
    def get_failed_attempts(self, username: str) -> int:
        """
        Get number of failed attempts for a username.
        
        Args:
            username: Username to check
            
        Returns:
            Number of failed attempts
        """
        key = self._get_key(username)
        
        try:
            count = self.redis.get(key)
            return int(count) if count else 0
        except redis.RedisError as e:
            logger.error(f"Failed to get failed attempts: {e}")
            return 0
