"""
Email Verification Service

This service handles:
- Generating verification tokens
- Sending verification emails
- Verifying email tokens
- Resending verification emails
"""
import secrets
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from app.models.user import User
from app.config import settings
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


class EmailVerificationError(Exception):
    """Raised when email verification fails"""
    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class EmailVerificationService:
    """Service for managing email verification"""
    
    TOKEN_LENGTH = 32  # Length in bytes - token_urlsafe(32) produces ~43 char string
    
    def __init__(self, db: Optional[Session] = None):
        self._db = db
        self._email_service = EmailService()
    
    def _generate_token(self) -> str:
        """Generate a secure random verification token"""
        return secrets.token_urlsafe(self.TOKEN_LENGTH)
    
    def _is_token_expired(self, sent_at: Optional[datetime]) -> bool:
        """Check if verification token has expired"""
        if not sent_at:
            return True
        
        expire_hours = settings.EMAIL_VERIFICATION_EXPIRE_HOURS
        expiration = sent_at + timedelta(hours=expire_hours)
        return datetime.now(timezone.utc) > expiration
    
    def create_verification_token(self, db: Session, user: User) -> str:
        """
        Create a new verification token for a user.
        
        Args:
            db: Database session
            user: User to create token for
            
        Returns:
            The verification token
        """
        token = self._generate_token()
        user.email_verification_token = token
        user.email_verification_sent_at = datetime.now(timezone.utc)
        db.commit()
        
        logger.info(f"Created verification token for user {user.key_id}")
        return token
    
    def send_verification_email(
        self,
        db: Session,
        user: User,
        force_resend: bool = False
    ) -> Tuple[bool, str]:
        """
        Send verification email to user.
        
        Args:
            db: Database session
            user: User to send email to
            force_resend: If True, resend even if recently sent
            
        Returns:
            Tuple of (success, message)
        """
        # Check if user already verified
        if user.email_verified:
            return False, "Email already verified"
        
        # Check if user has email
        if not user.email:
            return False, "No email address on account"
        
        # Rate limit: don't resend if sent within last 5 minutes
        if not force_resend and user.email_verification_sent_at:
            time_since_sent = datetime.now(timezone.utc) - user.email_verification_sent_at
            if time_since_sent < timedelta(minutes=5):
                remaining = 5 - int(time_since_sent.total_seconds() / 60)
                return False, f"Please wait {remaining} minute(s) before requesting another verification email"
        
        # Generate new token
        token = self.create_verification_token(db, user)
        
        # Build verification URL
        verification_url = f"{settings.FRONTEND_URL}/verify-email/{token}"
        
        # Send email
        try:
            success = self._email_service.send_verification_email(
                to_email=user.email,
                username=user.username,
                verification_url=verification_url,
                expire_hours=settings.EMAIL_VERIFICATION_EXPIRE_HOURS
            )
            
            if success:
                logger.info(f"Verification email sent to user {user.key_id}")
                return True, "Verification email sent"
            else:
                logger.error(f"Failed to send verification email to user {user.key_id}")
                return False, "Failed to send verification email. Please try again later."
                
        except Exception as e:
            logger.exception(f"Error sending verification email: {e}")
            return False, "Failed to send verification email. Please try again later."
    
    def verify_email(self, db: Session, token: str) -> Tuple[bool, str, Optional[User]]:
        """
        Verify email using token.
        
        Args:
            db: Database session
            token: Verification token
            
        Returns:
            Tuple of (success, message, user)
        """
        if not token:
            return False, "Invalid verification token", None
        
        # Find user by token
        user = db.query(User).filter(
            User.email_verification_token == token
        ).first()
        
        if not user:
            logger.warning(f"Verification attempt with invalid token")
            return False, "Invalid or expired verification token", None
        
        # Check if already verified
        if user.email_verified:
            return True, "Email already verified", user
        
        # Check if token expired
        if self._is_token_expired(user.email_verification_sent_at):
            logger.warning(f"Verification attempt with expired token for user {user.key_id}")
            return False, "Verification token has expired. Please request a new one.", None
        
        # Mark as verified
        user.email_verified = True
        user.email_verification_token = None  # Clear token
        user.email_verification_sent_at = None
        db.commit()
        
        logger.info(f"Email verified for user {user.key_id}")
        return True, "Email verified successfully", user
    
    def check_verification_required(self, user: User) -> Tuple[bool, Optional[str]]:
        """
        Check if verification is required for a user to perform actions.
        
        Args:
            user: User to check
            
        Returns:
            Tuple of (is_allowed, error_message if not allowed)
        """
        # If verification not required in settings, allow all
        if not settings.REQUIRE_EMAIL_VERIFICATION:
            return True, None
        
        # Admin users bypass verification requirement
        if user.role in ('admin', 'developer'):
            return True, None
        
        # Users with no email can't verify - allow them for now
        # (They should be prompted to add email)
        if not user.email:
            return True, None
        
        # Check if verified
        if user.email_verified:
            return True, None
        
        return False, "Email verification required. Please verify your email to continue."
    
    def get_verification_status(self, user: User) -> dict:
        """
        Get verification status for a user.
        
        Args:
            user: User to check
            
        Returns:
            Dictionary with verification status info
        """
        return {
            "email": user.email,
            "email_verified": user.email_verified,
            "verification_required": settings.REQUIRE_EMAIL_VERIFICATION,
            "can_resend": (
                user.email and 
                not user.email_verified and
                (
                    not user.email_verification_sent_at or
                    (datetime.now(timezone.utc) - user.email_verification_sent_at) > timedelta(minutes=5)
                )
            ),
            "token_expires_at": (
                (user.email_verification_sent_at + timedelta(hours=settings.EMAIL_VERIFICATION_EXPIRE_HOURS)).isoformat()
                if user.email_verification_sent_at and not user.email_verified
                else None
            )
        }
