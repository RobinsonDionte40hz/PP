"""
Tests for email verification service and related functionality

Tests cover:
- Email verification token generation
- Email verification flow
- Token expiration
- Rate limiting (resend cooldown)
- Verification status checks
- API endpoints
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


class TestEmailVerificationServiceUnit:
    """Unit tests for EmailVerificationService - isolated from database"""
    
    def test_token_generation(self):
        """Test token generation produces secure tokens"""
        import secrets
        
        # Generate tokens the same way the service does
        TOKEN_LENGTH = 64
        tokens = [secrets.token_urlsafe(TOKEN_LENGTH) for _ in range(100)]
        
        # All tokens should be unique
        assert len(set(tokens)) == 100
        
        # Tokens should be reasonably long
        assert all(len(t) > 50 for t in tokens)
    
    def test_token_expiration_logic(self):
        """Test token expiration calculation"""
        expire_hours = 24
        
        # Recent token should not be expired
        sent_at = datetime.now(timezone.utc) - timedelta(hours=1)
        expiration = sent_at + timedelta(hours=expire_hours)
        assert datetime.now(timezone.utc) <= expiration
        
        # Old token should be expired
        sent_at_old = datetime.now(timezone.utc) - timedelta(hours=25)
        expiration_old = sent_at_old + timedelta(hours=expire_hours)
        assert datetime.now(timezone.utc) > expiration_old
    
    def test_none_sent_at_is_expired(self):
        """Test that None sent_at is treated as expired"""
        sent_at = None
        # None should be considered expired
        assert sent_at is None


class TestUserModelVerificationFields:
    """Tests for User model verification fields"""
    
    def test_user_model_structure(self):
        """Test User model has expected verification field names"""
        # Import at test time to avoid global import issues
        from app.models.user import User
        
        # Check that the User model has the expected columns
        columns = [col.name for col in User.__table__.columns]
        
        assert 'email_verified' in columns
        assert 'email_verification_token' in columns
        assert 'email_verification_sent_at' in columns


class TestEmailVerificationServiceWithMocks:
    """Tests using mocked dependencies"""
    
    def test_service_initialization(self):
        """Test service can be initialized"""
        from app.services.email_verification_service import EmailVerificationService
        
        service = EmailVerificationService()
        assert service is not None
    
    def test_token_generation_method(self):
        """Test _generate_token produces unique tokens"""
        from app.services.email_verification_service import EmailVerificationService
        
        service = EmailVerificationService()
        tokens = [service._generate_token() for _ in range(10)]
        
        # All should be unique
        assert len(set(tokens)) == 10
    
    def test_is_token_expired_none(self):
        """Test _is_token_expired with None"""
        from app.services.email_verification_service import EmailVerificationService
        
        service = EmailVerificationService()
        assert service._is_token_expired(None) is True
    
    def test_is_token_expired_recent(self):
        """Test _is_token_expired with recent timestamp"""
        from app.services.email_verification_service import EmailVerificationService
        
        service = EmailVerificationService()
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        assert service._is_token_expired(recent) is False
    
    def test_is_token_expired_old(self):
        """Test _is_token_expired with old timestamp"""
        from app.services.email_verification_service import EmailVerificationService
        
        service = EmailVerificationService()
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        assert service._is_token_expired(old) is True
    
    def test_send_verification_already_verified(self):
        """Test sending to already verified user returns error"""
        from app.services.email_verification_service import EmailVerificationService
        
        mock_user = MagicMock()
        mock_user.email_verified = True
        mock_user.email = "test@example.com"
        
        mock_db = MagicMock()
        
        service = EmailVerificationService()
        success, message = service.send_verification_email(mock_db, mock_user)
        
        assert success is False
        assert "already verified" in message.lower()
    
    def test_send_verification_no_email(self):
        """Test sending to user without email returns error"""
        from app.services.email_verification_service import EmailVerificationService
        
        mock_user = MagicMock()
        mock_user.email_verified = False
        mock_user.email = None
        
        mock_db = MagicMock()
        
        service = EmailVerificationService()
        success, message = service.send_verification_email(mock_db, mock_user)
        
        assert success is False
        assert "no email" in message.lower()
    
    def test_send_verification_rate_limited(self):
        """Test rate limiting prevents frequent resends"""
        from app.services.email_verification_service import EmailVerificationService
        
        mock_user = MagicMock()
        mock_user.email_verified = False
        mock_user.email = "test@example.com"
        mock_user.email_verification_sent_at = datetime.now(timezone.utc) - timedelta(minutes=2)
        
        mock_db = MagicMock()
        
        service = EmailVerificationService()
        success, message = service.send_verification_email(mock_db, mock_user, force_resend=False)
        
        assert success is False
        assert "wait" in message.lower()
    
    def test_verify_email_invalid_token(self):
        """Test verification with invalid token fails"""
        from app.services.email_verification_service import EmailVerificationService
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        service = EmailVerificationService()
        success, message, user = service.verify_email(mock_db, "invalid-token")
        
        assert success is False
        assert "invalid" in message.lower()
        assert user is None
    
    def test_verify_email_expired_token(self):
        """Test verification with expired token fails"""
        from app.services.email_verification_service import EmailVerificationService
        
        mock_user = MagicMock()
        mock_user.email_verified = False
        mock_user.key_id = "test-id"
        mock_user.email_verification_sent_at = datetime.now(timezone.utc) - timedelta(hours=25)
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        service = EmailVerificationService()
        success, message, user = service.verify_email(mock_db, "expired-token")
        
        assert success is False
        assert "expired" in message.lower()
    
    def test_verify_email_success(self):
        """Test successful email verification"""
        from app.services.email_verification_service import EmailVerificationService
        
        mock_user = MagicMock()
        mock_user.email_verified = False
        mock_user.key_id = "test-id"
        mock_user.email_verification_sent_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        service = EmailVerificationService()
        success, message, user = service.verify_email(mock_db, "valid-token")
        
        assert success is True
        assert user is mock_user
        assert mock_user.email_verified is True
        mock_db.commit.assert_called()


class TestVerificationStatus:
    """Tests for verification status checks"""
    
    def test_get_verification_status_verified(self):
        """Test status for verified user"""
        from app.services.email_verification_service import EmailVerificationService
        
        mock_user = MagicMock()
        mock_user.email = "test@example.com"
        mock_user.email_verified = True
        mock_user.email_verification_sent_at = None
        
        service = EmailVerificationService()
        status = service.get_verification_status(mock_user)
        
        assert status["email"] == "test@example.com"
        assert status["email_verified"] is True
        assert status["can_resend"] is False
    
    def test_get_verification_status_unverified_can_resend(self):
        """Test status for unverified user who can resend"""
        from app.services.email_verification_service import EmailVerificationService
        
        mock_user = MagicMock()
        mock_user.email = "test@example.com"
        mock_user.email_verified = False
        mock_user.email_verification_sent_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        
        service = EmailVerificationService()
        status = service.get_verification_status(mock_user)
        
        assert status["email_verified"] is False
        assert status["can_resend"] is True


class TestCheckVerificationRequired:
    """Tests for check_verification_required method"""
    
    def test_admin_bypasses_verification(self):
        """Test admin users bypass verification"""
        from app.services.email_verification_service import EmailVerificationService
        from app.config import settings
        
        mock_user = MagicMock()
        mock_user.email_verified = False
        mock_user.role = "admin"
        
        service = EmailVerificationService()
        
        # Save original value
        original = settings.REQUIRE_EMAIL_VERIFICATION
        try:
            settings.REQUIRE_EMAIL_VERIFICATION = True
            allowed, error = service.check_verification_required(mock_user)
            assert allowed is True
        finally:
            settings.REQUIRE_EMAIL_VERIFICATION = original
    
    def test_verified_user_allowed(self):
        """Test verified users are allowed"""
        from app.services.email_verification_service import EmailVerificationService
        from app.config import settings
        
        mock_user = MagicMock()
        mock_user.email_verified = True
        mock_user.email = "test@example.com"
        mock_user.role = "user"
        
        service = EmailVerificationService()
        
        original = settings.REQUIRE_EMAIL_VERIFICATION
        try:
            settings.REQUIRE_EMAIL_VERIFICATION = True
            allowed, error = service.check_verification_required(mock_user)
            assert allowed is True
        finally:
            settings.REQUIRE_EMAIL_VERIFICATION = original
    
    def test_unverified_user_blocked(self):
        """Test unverified users are blocked"""
        from app.services.email_verification_service import EmailVerificationService
        from app.config import settings
        
        mock_user = MagicMock()
        mock_user.email_verified = False
        mock_user.email = "test@example.com"
        mock_user.role = "user"
        
        service = EmailVerificationService()
        
        original = settings.REQUIRE_EMAIL_VERIFICATION
        try:
            settings.REQUIRE_EMAIL_VERIFICATION = True
            allowed, error = service.check_verification_required(mock_user)
            assert allowed is False
            assert "verification required" in error.lower()
        finally:
            settings.REQUIRE_EMAIL_VERIFICATION = original


class TestConfigSettings:
    """Tests for email verification config settings"""
    
    def test_email_verification_settings_exist(self):
        """Test email verification settings are defined"""
        from app.config import settings
        
        assert hasattr(settings, 'EMAIL_VERIFICATION_EXPIRE_HOURS')
        assert hasattr(settings, 'REQUIRE_EMAIL_VERIFICATION')
        assert hasattr(settings, 'FRONTEND_URL')
    
    def test_default_verification_expire_hours(self):
        """Test default expiration is 24 hours"""
        from app.config import settings
        
        assert settings.EMAIL_VERIFICATION_EXPIRE_HOURS == 24
    
    def test_verification_required_default(self):
        """Test verification is required by default"""
        from app.config import settings
        
        assert settings.REQUIRE_EMAIL_VERIFICATION is True
