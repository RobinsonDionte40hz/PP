"""
Unit tests for OAuth service

Tests cover:
- Configuration checks
- State generation
- Authorization URL generation
- Token exchange (mocked)
- User creation and linking
- Account unlinking
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import uuid

# Test configuration
import os
os.environ["TESTING"] = "true"

from app.services.oauth_service import OAuthService, OAuthConfig


class TestOAuthConfiguration:
    """Tests for OAuth configuration checks"""
    
    def test_google_disabled_when_not_configured(self):
        """Test Google OAuth shows as disabled when credentials not set"""
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GOOGLE_CLIENT_ID = None
            mock_settings.GOOGLE_CLIENT_SECRET = None
            
            assert OAuthService.is_google_enabled() == False
    
    def test_google_enabled_when_configured(self):
        """Test Google OAuth shows as enabled when credentials set"""
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
            mock_settings.GOOGLE_CLIENT_SECRET = "test-client-secret"
            
            assert OAuthService.is_google_enabled() == True
    
    def test_github_disabled_when_not_configured(self):
        """Test GitHub OAuth shows as disabled when credentials not set"""
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GITHUB_CLIENT_ID = None
            mock_settings.GITHUB_CLIENT_SECRET = None
            
            assert OAuthService.is_github_enabled() == False
    
    def test_github_enabled_when_configured(self):
        """Test GitHub OAuth shows as enabled when credentials set"""
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GITHUB_CLIENT_ID = "test-client-id"
            mock_settings.GITHUB_CLIENT_SECRET = "test-client-secret"
            
            assert OAuthService.is_github_enabled() == True
    
    def test_get_oauth_config_returns_both_providers(self):
        """Test OAuth config returns status for both providers"""
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GOOGLE_CLIENT_ID = "google-id"
            mock_settings.GOOGLE_CLIENT_SECRET = "google-secret"
            mock_settings.GITHUB_CLIENT_ID = None
            mock_settings.GITHUB_CLIENT_SECRET = None
            
            config = OAuthService.get_oauth_config()
            
            assert "google" in config
            assert "github" in config
            assert config["google"]["enabled"] == True
            assert config["google"]["client_id"] == "google-id"
            assert config["github"]["enabled"] == False


class TestStateGeneration:
    """Tests for OAuth state token generation"""
    
    def test_generate_state_returns_string(self):
        """Test state generation returns a string"""
        state = OAuthService.generate_state()
        assert isinstance(state, str)
        assert len(state) > 0
    
    def test_generate_state_is_unique(self):
        """Test each state generation is unique"""
        states = [OAuthService.generate_state() for _ in range(100)]
        assert len(set(states)) == 100
    
    def test_generate_state_is_url_safe(self):
        """Test state is URL-safe"""
        state = OAuthService.generate_state()
        # Should only contain URL-safe characters
        import re
        assert re.match(r'^[A-Za-z0-9_-]+$', state)


class TestGoogleAuthorizationURL:
    """Tests for Google authorization URL generation"""
    
    def test_google_auth_url_raises_when_not_configured(self):
        """Test error raised when Google not configured"""
        with patch.object(OAuthService, 'is_google_enabled', return_value=False):
            with pytest.raises(ValueError, match="Google OAuth is not configured"):
                OAuthService.get_google_authorization_url(state="test")
    
    def test_google_auth_url_contains_required_params(self):
        """Test authorization URL contains required parameters"""
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GOOGLE_CLIENT_ID = "test-client-id"
            mock_settings.GOOGLE_CLIENT_SECRET = "test-secret"
            mock_settings.FRONTEND_URL = "http://localhost:5173"
            
            with patch.object(OAuthService, 'is_google_enabled', return_value=True):
                url = OAuthService.get_google_authorization_url(state="test-state")
                
                assert "accounts.google.com" in url
                assert "test-client-id" in url
                assert "test-state" in url
                assert "scope" in url


class TestGitHubAuthorizationURL:
    """Tests for GitHub authorization URL generation"""
    
    def test_github_auth_url_raises_when_not_configured(self):
        """Test error raised when GitHub not configured"""
        with patch.object(OAuthService, 'is_github_enabled', return_value=False):
            with pytest.raises(ValueError, match="GitHub OAuth is not configured"):
                OAuthService.get_github_authorization_url(state="test")
    
    def test_github_auth_url_contains_required_params(self):
        """Test authorization URL contains required parameters"""
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GITHUB_CLIENT_ID = "test-client-id"
            mock_settings.GITHUB_CLIENT_SECRET = "test-secret"
            mock_settings.FRONTEND_URL = "http://localhost:5173"
            
            with patch.object(OAuthService, 'is_github_enabled', return_value=True):
                url = OAuthService.get_github_authorization_url(state="test-state")
                
                assert "github.com" in url
                assert "test-client-id" in url
                assert "test-state" in url


class TestGoogleCodeExchange:
    """Tests for Google authorization code exchange"""
    
    @pytest.mark.asyncio
    async def test_exchange_fails_when_not_configured(self):
        """Test code exchange fails when not configured"""
        with patch.object(OAuthService, 'is_google_enabled', return_value=False):
            success, message, user_info = await OAuthService.exchange_google_code(code="test")
            
            assert success == False
            assert "not configured" in message
            assert user_info is None
    
    @pytest.mark.asyncio
    async def test_exchange_returns_user_info_on_success(self):
        """Test successful code exchange returns user info"""
        mock_token_response = MagicMock()
        mock_token_response.status_code = 200
        mock_token_response.json.return_value = {"access_token": "test-token"}
        
        mock_userinfo_response = MagicMock()
        mock_userinfo_response.status_code = 200
        mock_userinfo_response.json.return_value = {
            "id": "12345",
            "email": "test@example.com",
            "verified_email": True,
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
            "picture": "https://example.com/photo.jpg"
        }
        
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GOOGLE_CLIENT_ID = "test-id"
            mock_settings.GOOGLE_CLIENT_SECRET = "test-secret"
            mock_settings.FRONTEND_URL = "http://localhost:5173"
            
            with patch.object(OAuthService, 'is_google_enabled', return_value=True):
                with patch('httpx.AsyncClient') as mock_client:
                    mock_instance = AsyncMock()
                    mock_instance.post.return_value = mock_token_response
                    mock_instance.get.return_value = mock_userinfo_response
                    mock_client.return_value.__aenter__.return_value = mock_instance
                    
                    success, message, user_info = await OAuthService.exchange_google_code(
                        code="test-code"
                    )
                    
                    assert success == True
                    assert user_info["id"] == "12345"
                    assert user_info["email"] == "test@example.com"
                    assert user_info["email_verified"] == True
                    assert user_info["provider"] == "google"
    
    @pytest.mark.asyncio
    async def test_exchange_handles_token_error(self):
        """Test code exchange handles token exchange error"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Invalid code"
        
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GOOGLE_CLIENT_ID = "test-id"
            mock_settings.GOOGLE_CLIENT_SECRET = "test-secret"
            mock_settings.FRONTEND_URL = "http://localhost:5173"
            
            with patch.object(OAuthService, 'is_google_enabled', return_value=True):
                with patch('httpx.AsyncClient') as mock_client:
                    mock_instance = AsyncMock()
                    mock_instance.post.return_value = mock_response
                    mock_client.return_value.__aenter__.return_value = mock_instance
                    
                    success, message, user_info = await OAuthService.exchange_google_code(
                        code="invalid-code"
                    )
                    
                    assert success == False
                    assert user_info is None


class TestGitHubCodeExchange:
    """Tests for GitHub authorization code exchange"""
    
    @pytest.mark.asyncio
    async def test_exchange_fails_when_not_configured(self):
        """Test code exchange fails when not configured"""
        with patch.object(OAuthService, 'is_github_enabled', return_value=False):
            success, message, user_info = await OAuthService.exchange_github_code(code="test")
            
            assert success == False
            assert "not configured" in message
            assert user_info is None
    
    @pytest.mark.asyncio
    async def test_exchange_fetches_email_separately(self):
        """Test GitHub exchange fetches email when not in user info"""
        mock_token_response = MagicMock()
        mock_token_response.status_code = 200
        mock_token_response.json.return_value = {"access_token": "test-token"}
        
        mock_userinfo_response = MagicMock()
        mock_userinfo_response.status_code = 200
        mock_userinfo_response.json.return_value = {
            "id": 12345,
            "login": "testuser",
            "name": "Test User",
            "avatar_url": "https://github.com/photo.jpg",
            "email": None  # Email not in user info
        }
        
        mock_emails_response = MagicMock()
        mock_emails_response.status_code = 200
        mock_emails_response.json.return_value = [
            {"email": "test@example.com", "primary": True, "verified": True}
        ]
        
        with patch('app.services.oauth_service.settings') as mock_settings:
            mock_settings.GITHUB_CLIENT_ID = "test-id"
            mock_settings.GITHUB_CLIENT_SECRET = "test-secret"
            mock_settings.FRONTEND_URL = "http://localhost:5173"
            
            with patch.object(OAuthService, 'is_github_enabled', return_value=True):
                with patch('httpx.AsyncClient') as mock_client:
                    mock_instance = AsyncMock()
                    mock_instance.post.return_value = mock_token_response
                    # First call returns user info, second returns emails
                    mock_instance.get.side_effect = [mock_userinfo_response, mock_emails_response]
                    mock_client.return_value.__aenter__.return_value = mock_instance
                    
                    success, message, user_info = await OAuthService.exchange_github_code(
                        code="test-code"
                    )
                    
                    assert success == True
                    assert user_info["email"] == "test@example.com"
                    assert user_info["email_verified"] == True
                    assert user_info["login"] == "testuser"
                    assert user_info["provider"] == "github"


class TestUserAccountManagement:
    """Tests for OAuth user creation and linking"""
    
    def test_find_user_by_oauth_google(self):
        """Test finding user by Google ID"""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        result = OAuthService.find_user_by_oauth(mock_db, "google", "12345")
        
        assert result == mock_user
        mock_db.query.return_value.filter.assert_called_once()
    
    def test_find_user_by_oauth_github(self):
        """Test finding user by GitHub ID"""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        result = OAuthService.find_user_by_oauth(mock_db, "github", "12345")
        
        assert result == mock_user
    
    def test_find_user_by_oauth_invalid_provider(self):
        """Test finding user by invalid provider returns None"""
        mock_db = MagicMock()
        
        result = OAuthService.find_user_by_oauth(mock_db, "invalid", "12345")
        
        assert result is None
    
    def test_generate_unique_username_base_available(self):
        """Test username generation when base is available"""
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        username = OAuthService._generate_unique_username(mock_db, "testuser")
        
        assert username == "testuser"
    
    def test_generate_unique_username_adds_number(self):
        """Test username generation adds number when base taken"""
        mock_db = MagicMock()
        # First call (base) returns user, second call (_1) returns None
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            MagicMock(),  # "testuser" taken
            None  # "testuser_1" available
        ]
        
        username = OAuthService._generate_unique_username(mock_db, "testuser")
        
        assert username == "testuser_1"
    
    def test_generate_unique_username_cleans_special_chars(self):
        """Test username generation cleans special characters"""
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        username = OAuthService._generate_unique_username(mock_db, "test@user.com")
        
        assert "@" not in username
        assert "." not in username
    
    def test_create_oauth_user_success(self):
        """Test successful OAuth user creation"""
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        success, message, user = OAuthService.create_oauth_user(
            db=mock_db,
            provider="google",
            oauth_id="12345",
            email="test@example.com",
            name="Test User",
            email_verified=True
        )
        
        assert success == True
        assert "successfully" in message.lower()
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    def test_link_oauth_account_already_linked_to_other(self):
        """Test linking fails if OAuth already linked to another user"""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.key_id = "user-1"
        
        mock_existing = MagicMock()
        mock_existing.key_id = "user-2"  # Different user
        
        with patch.object(OAuthService, 'find_user_by_oauth', return_value=mock_existing):
            success, message = OAuthService.link_oauth_account(
                db=mock_db,
                user=mock_user,
                provider="google",
                oauth_id="12345"
            )
            
            assert success == False
            assert "already linked to another user" in message
    
    def test_link_oauth_account_success(self):
        """Test successful OAuth account linking"""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.key_id = "user-1"
        mock_user.google_id = None
        mock_user.email = None
        
        with patch.object(OAuthService, 'find_user_by_oauth', return_value=None):
            with patch.object(OAuthService, 'find_user_by_email', return_value=None):
                success, message = OAuthService.link_oauth_account(
                    db=mock_db,
                    user=mock_user,
                    provider="google",
                    oauth_id="12345",
                    email="test@example.com",
                    email_verified=True
                )
                
                assert success == True
                assert mock_user.google_id == "12345"
                assert mock_user.email == "test@example.com"
                mock_db.commit.assert_called_once()
    
    def test_unlink_oauth_account_fails_if_only_auth_method(self):
        """Test unlinking fails if it's the only auth method"""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.password_hash = None  # No password
        mock_user.google_id = "12345"
        mock_user.github_id = None  # No other OAuth
        
        success, message = OAuthService.unlink_oauth_account(
            db=mock_db,
            user=mock_user,
            provider="google"
        )
        
        assert success == False
        assert "only authentication method" in message
    
    def test_unlink_oauth_account_success_with_password(self):
        """Test successful unlink when user has password"""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.password_hash = "hashed-password"
        mock_user.google_id = "12345"
        mock_user.github_id = None
        
        success, message = OAuthService.unlink_oauth_account(
            db=mock_db,
            user=mock_user,
            provider="google"
        )
        
        assert success == True
        assert mock_user.google_id is None
        mock_db.commit.assert_called_once()


class TestAuthenticateOrCreateOAuthUser:
    """Tests for the main OAuth authentication flow"""
    
    def test_existing_user_by_oauth_id(self):
        """Test login with existing OAuth-linked user"""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.key_id = "user-1"
        
        with patch.object(OAuthService, 'find_user_by_oauth', return_value=mock_user):
            success, message, user, is_new = OAuthService.authenticate_or_create_oauth_user(
                db=mock_db,
                provider="google",
                user_info={"id": "12345", "email": "test@example.com"}
            )
            
            assert success == True
            assert user == mock_user
            assert is_new == False
    
    def test_existing_user_by_email_links_account(self):
        """Test login with existing email links OAuth account"""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.key_id = "user-1"
        
        with patch.object(OAuthService, 'find_user_by_oauth', return_value=None):
            with patch.object(OAuthService, 'find_user_by_email', return_value=mock_user):
                with patch.object(OAuthService, 'link_oauth_account', return_value=(True, "Linked")):
                    success, message, user, is_new = OAuthService.authenticate_or_create_oauth_user(
                        db=mock_db,
                        provider="google",
                        user_info={"id": "12345", "email": "test@example.com"}
                    )
                    
                    assert success == True
                    assert user == mock_user
                    assert is_new == False
    
    def test_new_user_creation(self):
        """Test new user creation when no match found"""
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with patch.object(OAuthService, 'find_user_by_oauth', return_value=None):
            with patch.object(OAuthService, 'find_user_by_email', return_value=None):
                success, message, user, is_new = OAuthService.authenticate_or_create_oauth_user(
                    db=mock_db,
                    provider="google",
                    user_info={
                        "id": "12345",
                        "email": "new@example.com",
                        "name": "New User",
                        "email_verified": True
                    }
                )
                
                assert success == True
                assert is_new == True
    
    def test_missing_oauth_id_fails(self):
        """Test authentication fails without OAuth ID"""
        mock_db = MagicMock()
        
        success, message, user, is_new = OAuthService.authenticate_or_create_oauth_user(
            db=mock_db,
            provider="google",
            user_info={"email": "test@example.com"}  # No id
        )
        
        assert success == False
        assert "No user ID" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
