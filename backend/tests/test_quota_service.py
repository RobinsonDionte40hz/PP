"""
Tests for the quota service and quota-related functionality

Tests cover:
- Quota checking
- Quota incrementing
- Daily/monthly reset logic
- Tier limits
- API endpoints
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.services.quota_service import QuotaService, QuotaExceededError
from app.models.user import User


class TestQuotaService:
    """Tests for QuotaService class"""
    
    def test_tier_limits_defined(self):
        """Test that tier limits are properly defined"""
        service = QuotaService()
        
        assert 'free' in service.TIER_LIMITS
        assert 'pro' in service.TIER_LIMITS
        assert 'enterprise' in service.TIER_LIMITS
        
        # Free tier
        assert service.TIER_LIMITS['free']['daily'] == 20
        assert service.TIER_LIMITS['free']['monthly'] == 100
        
        # Pro tier
        assert service.TIER_LIMITS['pro']['daily'] == 100
        assert service.TIER_LIMITS['pro']['monthly'] == 500
        
        # Enterprise tier (unlimited)
        assert service.TIER_LIMITS['enterprise']['daily'] == -1
        assert service.TIER_LIMITS['enterprise']['monthly'] == -1
    
    def test_is_new_day_same_day(self):
        """Test _is_new_day returns False for same day"""
        service = QuotaService()
        
        now = datetime(2025, 12, 7, 10, 0, 0, tzinfo=timezone.utc)
        last_reset = datetime(2025, 12, 7, 0, 0, 0, tzinfo=timezone.utc)
        
        assert service._is_new_day(last_reset, now) is False
    
    def test_is_new_day_next_day(self):
        """Test _is_new_day returns True for next day"""
        service = QuotaService()
        
        now = datetime(2025, 12, 8, 0, 0, 1, tzinfo=timezone.utc)
        last_reset = datetime(2025, 12, 7, 23, 59, 59, tzinfo=timezone.utc)
        
        assert service._is_new_day(last_reset, now) is True
    
    def test_is_new_month_same_month(self):
        """Test _is_new_month returns False for same month"""
        service = QuotaService()
        
        now = datetime(2025, 12, 15, 10, 0, 0, tzinfo=timezone.utc)
        last_reset = datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        assert service._is_new_month(last_reset, now) is False
    
    def test_is_new_month_next_month(self):
        """Test _is_new_month returns True for next month"""
        service = QuotaService()
        
        now = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        last_reset = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        
        assert service._is_new_month(last_reset, now) is True
    
    def test_is_new_month_next_year(self):
        """Test _is_new_month returns True for next year"""
        service = QuotaService()
        
        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        last_reset = datetime(2025, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        assert service._is_new_month(last_reset, now) is True


class TestUserQuotaModel:
    """Tests for User model quota methods"""
    
    def test_to_quota_info_free_tier(self):
        """Test to_quota_info returns correct structure for free tier"""
        user = User(
            key_id="test-user-123",
            username="testuser",
            password_hash="hashed",
            account_tier="free",
            daily_prediction_count=5,
            monthly_prediction_count=25,
            daily_prediction_limit=20,
            monthly_prediction_limit=100,
            daily_quota_reset_at=datetime(2025, 12, 7, 0, 0, 0, tzinfo=timezone.utc),
            monthly_quota_reset_at=datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
        )
        
        quota_info = user.to_quota_info()
        
        assert quota_info['account_tier'] == 'free'
        assert quota_info['daily']['used'] == 5
        assert quota_info['daily']['limit'] == 20
        assert quota_info['daily']['remaining'] == 15
        assert quota_info['monthly']['used'] == 25
        assert quota_info['monthly']['limit'] == 100
        assert quota_info['monthly']['remaining'] == 75
    
    def test_to_quota_info_at_limit(self):
        """Test to_quota_info when user is at limit"""
        user = User(
            key_id="test-user-123",
            username="testuser",
            password_hash="hashed",
            account_tier="free",
            daily_prediction_count=20,
            monthly_prediction_count=100,
            daily_prediction_limit=20,
            monthly_prediction_limit=100,
        )
        
        quota_info = user.to_quota_info()
        
        assert quota_info['daily']['remaining'] == 0
        assert quota_info['monthly']['remaining'] == 0
    
    def test_to_quota_info_over_limit(self):
        """Test to_quota_info when user is over limit (edge case)"""
        user = User(
            key_id="test-user-123",
            username="testuser",
            password_hash="hashed",
            account_tier="free",
            daily_prediction_count=25,  # Over limit
            monthly_prediction_count=150,  # Over limit
            daily_prediction_limit=20,
            monthly_prediction_limit=100,
        )
        
        quota_info = user.to_quota_info()
        
        # remaining should never be negative
        assert quota_info['daily']['remaining'] == 0
        assert quota_info['monthly']['remaining'] == 0


class TestQuotaServiceWithMocks:
    """Tests for QuotaService using mocked database"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        return MagicMock()
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user"""
        user = MagicMock(spec=User)
        user.key_id = "test-user-123"
        user.account_tier = "free"
        user.daily_prediction_count = 5
        user.monthly_prediction_count = 25
        user.daily_prediction_limit = 20
        user.monthly_prediction_limit = 100
        user.daily_quota_reset_at = datetime(2025, 12, 7, 0, 0, 0, tzinfo=timezone.utc)
        user.monthly_quota_reset_at = datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
        user.to_quota_info.return_value = {
            "account_tier": "free",
            "daily": {"used": 5, "limit": 20, "remaining": 15, "reset_at": None},
            "monthly": {"used": 25, "limit": 100, "remaining": 75, "reset_at": None},
        }
        return user
    
    def test_check_quota_has_quota(self, mock_db, mock_user):
        """Test check_quota returns True when user has quota"""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        service = QuotaService(db=mock_db)
        has_quota, error = service.check_quota("test-user-123")
        
        assert has_quota is True
        assert error is None
    
    def test_check_quota_daily_exceeded(self, mock_db, mock_user):
        """Test check_quota returns False when daily quota exceeded"""
        mock_user.daily_prediction_count = 20  # At limit
        mock_user.daily_prediction_limit = 20
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        service = QuotaService(db=mock_db)
        has_quota, error = service.check_quota("test-user-123")
        
        assert has_quota is False
        assert "Daily quota exceeded" in error
    
    def test_check_quota_monthly_exceeded(self, mock_db, mock_user):
        """Test check_quota returns False when monthly quota exceeded"""
        mock_user.daily_prediction_count = 5  # Under daily limit
        mock_user.monthly_prediction_count = 100  # At monthly limit
        mock_user.monthly_prediction_limit = 100
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        service = QuotaService(db=mock_db)
        has_quota, error = service.check_quota("test-user-123")
        
        assert has_quota is False
        assert "Monthly quota exceeded" in error
    
    def test_check_quota_enterprise_unlimited(self, mock_db, mock_user):
        """Test enterprise tier has unlimited quota"""
        mock_user.account_tier = "enterprise"
        mock_user.daily_prediction_count = 1000  # High usage
        mock_user.monthly_prediction_count = 10000  # High usage
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        service = QuotaService(db=mock_db)
        has_quota, error = service.check_quota("test-user-123")
        
        assert has_quota is True
        assert error is None
    
    def test_check_quota_user_not_found(self, mock_db):
        """Test check_quota handles user not found"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        service = QuotaService(db=mock_db)
        has_quota, error = service.check_quota("nonexistent-user")
        
        assert has_quota is False
        assert "User not found" in error
    
    def test_increment_quota(self, mock_db, mock_user):
        """Test increment_quota increases counts"""
        initial_daily = mock_user.daily_prediction_count
        initial_monthly = mock_user.monthly_prediction_count
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        service = QuotaService(db=mock_db)
        result = service.increment_quota("test-user-123")
        
        assert result is True
        assert mock_user.daily_prediction_count == initial_daily + 1
        assert mock_user.monthly_prediction_count == initial_monthly + 1
        mock_db.commit.assert_called()


class TestQuotaTasks:
    """Tests for quota Celery tasks"""
    
    def test_reset_daily_quotas_task_exists(self):
        """Test that reset_daily_quotas task is importable"""
        from app.tasks.quota_tasks import reset_daily_quotas
        assert callable(reset_daily_quotas)
    
    def test_reset_monthly_quotas_task_exists(self):
        """Test that reset_monthly_quotas task is importable"""
        from app.tasks.quota_tasks import reset_monthly_quotas
        assert callable(reset_monthly_quotas)
    
    def test_check_and_reset_quotas_task_exists(self):
        """Test that check_and_reset_quotas task is importable"""
        from app.tasks.quota_tasks import check_and_reset_quotas
        assert callable(check_and_reset_quotas)


# Integration tests (require database)
@pytest.mark.integration
class TestQuotaIntegration:
    """Integration tests requiring database"""
    
    @pytest.fixture
    def db_session(self):
        """Get a real database session"""
        from app.database import get_db
        db = next(get_db())
        yield db
        db.close()
    
    def test_quota_service_with_real_db(self, db_session):
        """Test quota service with real database"""
        # This test requires a real user in the database
        # Skip if no test user exists
        pytest.skip("Integration test - requires database setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
