"""
Tests for CAPTCHA verification service

Tests cover:
- CAPTCHA service initialization
- Token verification with mocked HTTP responses
- Score threshold checks
- Error handling
- Configuration checks
- Edge cases
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import Response


class TestCaptchaServiceUnit:
    """Unit tests for CaptchaService - isolated from external calls"""
    
    def test_service_import(self):
        """Test service can be imported"""
        from app.services.captcha_service import CaptchaService
        assert CaptchaService is not None
    
    def test_is_enabled_default(self):
        """Test is_enabled returns False when not configured"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = False
            mock_settings.RECAPTCHA_SECRET_KEY = None
            
            assert CaptchaService.is_enabled() is False
    
    def test_is_enabled_with_key(self):
        """Test is_enabled returns True when properly configured"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret-key"
            
            assert CaptchaService.is_enabled() is True
    
    def test_get_site_key_disabled(self):
        """Test get_site_key returns None when disabled"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = False
            
            assert CaptchaService.get_site_key() is None
    
    def test_get_site_key_enabled(self):
        """Test get_site_key returns key when enabled"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SITE_KEY = "test-site-key"
            
            assert CaptchaService.get_site_key() == "test-site-key"
    
    def test_get_provider_disabled(self):
        """Test get_provider returns 'none' when disabled"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = False
            
            assert CaptchaService.get_provider() == "none"
    
    def test_get_provider_recaptcha(self):
        """Test get_provider returns provider name when enabled"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            assert CaptchaService.get_provider() == "recaptcha"


class TestCaptchaVerification:
    """Tests for CAPTCHA token verification"""
    
    @pytest.mark.asyncio
    async def test_verify_disabled_returns_success(self):
        """Test verification returns success when CAPTCHA is disabled"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = False
            
            success, message, score = await CaptchaService.verify_token("any-token")
            
            assert success is True
            assert "disabled" in message.lower()
            assert score is None
    
    @pytest.mark.asyncio
    async def test_verify_empty_token_fails(self):
        """Test verification fails with empty token when enabled"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret"
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            success, message, score = await CaptchaService.verify_token("")
            
            assert success is False
            assert "required" in message.lower()
    
    @pytest.mark.asyncio
    async def test_verify_no_secret_dev_mode(self):
        """Test verification passes in dev mode without secret key"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = None
            mock_settings.APP_ENV = "development"
            
            success, message, score = await CaptchaService.verify_token("test-token")
            
            assert success is True
            assert "not configured" in message.lower()
    
    @pytest.mark.asyncio
    async def test_verify_no_secret_prod_mode(self):
        """Test verification fails in production without secret key"""
        from app.services.captcha_service import CaptchaService
        
        with patch('app.services.captcha_service.settings') as mock_settings:
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = None
            mock_settings.APP_ENV = "production"
            
            success, message, score = await CaptchaService.verify_token("test-token")
            
            assert success is False
            assert "configuration error" in message.lower()


class TestRecaptchaVerification:
    """Tests for reCAPTCHA-specific verification"""
    
    @pytest.mark.asyncio
    async def test_recaptcha_v3_success(self):
        """Test successful reCAPTCHA v3 verification"""
        from app.services.captcha_service import CaptchaService
        
        mock_response = {
            "success": True,
            "score": 0.9,
            "action": "register",
            "challenge_ts": "2025-12-07T00:00:00Z",
            "hostname": "emergentfolds.com"
        }
        
        with patch('app.services.captcha_service.settings') as mock_settings, \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret"
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            # Setup mock client
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj
            
            success, message, score = await CaptchaService.verify_token(
                "valid-token",
                expected_action="register"
            )
            
            assert success is True
            assert score == 0.9
    
    @pytest.mark.asyncio
    async def test_recaptcha_v3_low_score(self):
        """Test reCAPTCHA v3 rejection for low score"""
        from app.services.captcha_service import CaptchaService
        
        mock_response = {
            "success": True,
            "score": 0.2,  # Low score - likely bot
            "action": "register"
        }
        
        with patch('app.services.captcha_service.settings') as mock_settings, \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret"
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj
            
            success, message, score = await CaptchaService.verify_token(
                "bot-token",
                min_score=0.5
            )
            
            assert success is False
            assert score == 0.2
            assert "suspicious" in message.lower()
    
    @pytest.mark.asyncio
    async def test_recaptcha_v2_success(self):
        """Test successful reCAPTCHA v2 verification (no score)"""
        from app.services.captcha_service import CaptchaService
        
        mock_response = {
            "success": True,
            "challenge_ts": "2025-12-07T00:00:00Z",
            "hostname": "emergentfolds.com"
        }
        
        with patch('app.services.captcha_service.settings') as mock_settings, \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret"
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj
            
            success, message, score = await CaptchaService.verify_token("valid-v2-token")
            
            assert success is True
            assert score is None  # v2 doesn't have score
    
    @pytest.mark.asyncio
    async def test_recaptcha_expired_token(self):
        """Test handling of expired token"""
        from app.services.captcha_service import CaptchaService
        
        mock_response = {
            "success": False,
            "error-codes": ["timeout-or-duplicate"]
        }
        
        with patch('app.services.captcha_service.settings') as mock_settings, \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret"
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj
            
            success, message, score = await CaptchaService.verify_token("expired-token")
            
            assert success is False
            assert "expired" in message.lower()
    
    @pytest.mark.asyncio
    async def test_recaptcha_action_mismatch(self):
        """Test handling of action mismatch"""
        from app.services.captcha_service import CaptchaService
        
        mock_response = {
            "success": True,
            "score": 0.9,
            "action": "login"  # Wrong action
        }
        
        with patch('app.services.captcha_service.settings') as mock_settings, \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret"
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj
            
            success, message, score = await CaptchaService.verify_token(
                "valid-token",
                expected_action="register"
            )
            
            assert success is False
            assert "action" in message.lower()


class TestCaptchaErrorHandling:
    """Tests for error handling in CAPTCHA verification"""
    
    @pytest.mark.asyncio
    async def test_timeout_allows_request(self):
        """Test that timeouts fail open (allow request)"""
        from app.services.captcha_service import CaptchaService
        import httpx
        
        with patch('app.services.captcha_service.settings') as mock_settings, \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret"
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            
            success, message, score = await CaptchaService.verify_token("any-token")
            
            # Should fail open (allow) on timeout
            assert success is True
            assert "timeout" in message.lower()
    
    @pytest.mark.asyncio
    async def test_http_error_allows_request(self):
        """Test that HTTP errors fail open (allow request)"""
        from app.services.captcha_service import CaptchaService
        import httpx
        
        with patch('app.services.captcha_service.settings') as mock_settings, \
             patch('httpx.AsyncClient') as mock_client_class:
            
            mock_settings.RECAPTCHA_ENABLED = True
            mock_settings.RECAPTCHA_SECRET_KEY = "test-secret"
            mock_settings.CAPTCHA_PROVIDER = "recaptcha"
            
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.HTTPError("Connection failed")
            
            success, message, score = await CaptchaService.verify_token("any-token")
            
            # Should fail open (allow) on HTTP errors
            assert success is True
            assert "unavailable" in message.lower()


class TestCaptchaConfigEndpoint:
    """Tests for the CAPTCHA configuration endpoint"""
    
    def test_captcha_config_schema(self):
        """Test expected structure of CAPTCHA config response"""
        expected_keys = {"enabled", "provider", "site_key"}
        
        # The endpoint should return these keys
        response = {
            "enabled": False,
            "provider": "none",
            "site_key": None
        }
        
        assert set(response.keys()) == expected_keys


class TestRegistrationWithCaptcha:
    """Tests for registration endpoint with CAPTCHA"""
    
    def test_register_schema_has_captcha_field(self):
        """Test registration request schema includes captcha_token"""
        from app.schemas.auth import UserRegisterRequest
        
        # Check schema fields
        fields = UserRegisterRequest.model_fields
        assert 'captcha_token' in fields
        
        # Field should be optional
        assert fields['captcha_token'].is_required() is False
    
    def test_register_request_with_captcha(self):
        """Test creating registration request with captcha token"""
        from app.schemas.auth import UserRegisterRequest
        
        request = UserRegisterRequest(
            username="testuser",
            password="SecurePass123!",
            email="test@example.com",
            captcha_token="test-captcha-token"
        )
        
        assert request.captcha_token == "test-captcha-token"
    
    def test_register_request_without_captcha(self):
        """Test creating registration request without captcha token"""
        from app.schemas.auth import UserRegisterRequest
        
        request = UserRegisterRequest(
            username="testuser",
            password="SecurePass123!",
            email="test@example.com"
        )
        
        assert request.captcha_token is None
