"""
Quota Service - Manages user prediction quotas

This service handles:
- Checking if user has quota remaining
- Incrementing prediction counts
- Resetting daily/monthly quotas
- Providing quota status information
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session
import logging

from app.database import get_db
from app.models.user import User

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when user has exceeded their quota"""
    def __init__(self, message: str, quota_type: str, used: int, limit: int):
        self.message = message
        self.quota_type = quota_type  # 'daily' or 'monthly'
        self.used = used
        self.limit = limit
        super().__init__(self.message)


class QuotaService:
    """Service for managing user prediction quotas"""
    
    # Tier definitions
    TIER_LIMITS = {
        'free': {'daily': 20, 'monthly': 100},
        'pro': {'daily': 100, 'monthly': 500},
        'enterprise': {'daily': -1, 'monthly': -1},  # -1 = unlimited
    }
    
    def __init__(self, db: Optional[Session] = None):
        self._db = db
    
    def _get_db(self) -> Session:
        """Get database session"""
        if self._db:
            return self._db
        return next(get_db())
    
    def get_user_quota(self, user_id: str) -> Optional[dict]:
        """
        Get current quota status for a user
        
        Args:
            user_id: User's key_id
            
        Returns:
            Dictionary with quota info or None if user not found
        """
        db = self._get_db()
        try:
            user = db.query(User).filter(User.key_id == user_id).first()
            if not user:
                return None
            
            # Check if quotas need reset
            self._check_and_reset_quotas(user, db)
            
            return user.to_quota_info()
        finally:
            if not self._db:
                db.close()
    
    def check_quota(self, user_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if user has remaining quota
        
        Args:
            user_id: User's key_id
            
        Returns:
            Tuple of (has_quota, error_message)
        """
        db = self._get_db()
        try:
            user = db.query(User).filter(User.key_id == user_id).first()
            if not user:
                return False, "User not found"
            
            # Check if quotas need reset
            self._check_and_reset_quotas(user, db)
            
            # Enterprise tier has unlimited quota
            if user.account_tier == 'enterprise':
                return True, None
            
            # Check daily limit
            if user.daily_prediction_count >= user.daily_prediction_limit:
                return False, f"Daily quota exceeded ({user.daily_prediction_count}/{user.daily_prediction_limit})"
            
            # Check monthly limit
            if user.monthly_prediction_count >= user.monthly_prediction_limit:
                return False, f"Monthly quota exceeded ({user.monthly_prediction_count}/{user.monthly_prediction_limit})"
            
            return True, None
        finally:
            if not self._db:
                db.close()
    
    def increment_quota(self, user_id: str) -> bool:
        """
        Increment user's prediction count (call after successful prediction creation)
        
        Args:
            user_id: User's key_id
            
        Returns:
            True if successful, False if user not found
        """
        db = self._get_db()
        try:
            user = db.query(User).filter(User.key_id == user_id).first()
            if not user:
                logger.warning(f"Cannot increment quota: user {user_id} not found")
                return False
            
            # Check if quotas need reset first
            self._check_and_reset_quotas(user, db)
            
            # Increment counts
            user.daily_prediction_count += 1
            user.monthly_prediction_count += 1
            
            db.commit()
            
            logger.info(
                f"Quota incremented for user {user_id}: "
                f"daily={user.daily_prediction_count}/{user.daily_prediction_limit}, "
                f"monthly={user.monthly_prediction_count}/{user.monthly_prediction_limit}"
            )
            
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error incrementing quota for user {user_id}: {e}")
            raise
        finally:
            if not self._db:
                db.close()
    
    def _check_and_reset_quotas(self, user: User, db: Session) -> None:
        """
        Check if quotas need to be reset based on time
        
        Args:
            user: User model instance
            db: Database session
        """
        now = datetime.now(timezone.utc)
        needs_commit = False
        
        # Check daily reset
        if user.daily_quota_reset_at is None or self._is_new_day(user.daily_quota_reset_at, now):
            user.daily_prediction_count = 0
            user.daily_quota_reset_at = now
            needs_commit = True
            logger.info(f"Daily quota reset for user {user.key_id}")
        
        # Check monthly reset
        if user.monthly_quota_reset_at is None or self._is_new_month(user.monthly_quota_reset_at, now):
            user.monthly_prediction_count = 0
            user.monthly_quota_reset_at = now
            needs_commit = True
            logger.info(f"Monthly quota reset for user {user.key_id}")
        
        if needs_commit:
            db.commit()
    
    def _is_new_day(self, last_reset: datetime, now: datetime) -> bool:
        """Check if we're in a new day (UTC) since last reset"""
        if last_reset.tzinfo is None:
            last_reset = last_reset.replace(tzinfo=timezone.utc)
        return last_reset.date() < now.date()
    
    def _is_new_month(self, last_reset: datetime, now: datetime) -> bool:
        """Check if we're in a new month since last reset"""
        if last_reset.tzinfo is None:
            last_reset = last_reset.replace(tzinfo=timezone.utc)
        return (last_reset.year, last_reset.month) < (now.year, now.month)
    
    def reset_daily_quotas_all_users(self) -> int:
        """
        Reset daily quotas for all users (called by Celery task)
        
        Returns:
            Number of users updated
        """
        db = self._get_db()
        try:
            now = datetime.now(timezone.utc)
            
            # Update all users at once
            result = db.query(User).update({
                User.daily_prediction_count: 0,
                User.daily_quota_reset_at: now
            })
            
            db.commit()
            logger.info(f"Daily quota reset for {result} users")
            return result
        except Exception as e:
            db.rollback()
            logger.error(f"Error resetting daily quotas: {e}")
            raise
        finally:
            if not self._db:
                db.close()
    
    def reset_monthly_quotas_all_users(self) -> int:
        """
        Reset monthly quotas for all users (called by Celery task on 1st of month)
        
        Returns:
            Number of users updated
        """
        db = self._get_db()
        try:
            now = datetime.now(timezone.utc)
            
            # Update all users at once
            result = db.query(User).update({
                User.monthly_prediction_count: 0,
                User.monthly_quota_reset_at: now
            })
            
            db.commit()
            logger.info(f"Monthly quota reset for {result} users")
            return result
        except Exception as e:
            db.rollback()
            logger.error(f"Error resetting monthly quotas: {e}")
            raise
        finally:
            if not self._db:
                db.close()
    
    def set_user_tier(self, user_id: str, tier: str) -> bool:
        """
        Set user's account tier and update their limits
        
        Args:
            user_id: User's key_id
            tier: New tier ('free', 'pro', 'enterprise')
            
        Returns:
            True if successful
        """
        if tier not in self.TIER_LIMITS:
            raise ValueError(f"Invalid tier: {tier}. Must be one of {list(self.TIER_LIMITS.keys())}")
        
        db = self._get_db()
        try:
            user = db.query(User).filter(User.key_id == user_id).first()
            if not user:
                return False
            
            limits = self.TIER_LIMITS[tier]
            user.account_tier = tier
            user.daily_prediction_limit = limits['daily']
            user.monthly_prediction_limit = limits['monthly']
            
            db.commit()
            logger.info(f"User {user_id} tier set to {tier}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error setting tier for user {user_id}: {e}")
            raise
        finally:
            if not self._db:
                db.close()


# Singleton instance
quota_service = QuotaService()
