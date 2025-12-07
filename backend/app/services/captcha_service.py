"""
CAPTCHA verification service for bot protection.

Supports reCAPTCHA v3 (invisible) and v2 (checkbox).
reCAPTCHA v3 is recommended for better UX.

Usage:
    from app.services.captcha_service import CaptchaService
    
    # Verify a token
    is_valid = await CaptchaService.verify_token(token)
    
    # With score threshold (v3 only)
    is_valid = await CaptchaService.verify_token(token, min_score=0.5)
"""
import logging
from typing import Optional, Tuple
import httpx
from app.config import settings

logger = logging.getLogger("security.captcha")


class CaptchaService:
    """
    Service for verifying CAPTCHA tokens.
    
    Supports:
    - reCAPTCHA v3 (invisible, score-based)
    - reCAPTCHA v2 (checkbox)
    - hCaptcha (alternative to reCAPTCHA)
    """
    
    RECAPTCHA_VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"
    HCAPTCHA_VERIFY_URL = "https://hcaptcha.com/siteverify"
    
    # Minimum score for reCAPTCHA v3 (0.0 to 1.0)
    # 0.0 = likely bot, 1.0 = likely human
    # 0.5 is a good default threshold
    DEFAULT_MIN_SCORE = 0.5
    
    @classmethod
    async def verify_token(
        cls,
        token: str,
        remote_ip: Optional[str] = None,
        expected_action: Optional[str] = None,
        min_score: float = DEFAULT_MIN_SCORE
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Verify a CAPTCHA token with the provider.
        
        Args:
            token: The CAPTCHA response token from the frontend
            remote_ip: Client IP address (optional, for additional security)
            expected_action: Expected action name for reCAPTCHA v3
            min_score: Minimum score threshold for reCAPTCHA v3 (0.0-1.0)
            
        Returns:
            Tuple of (success: bool, message: str, score: Optional[float])
            - success: True if verification passed
            - message: Human-readable status message
            - score: reCAPTCHA v3 score (None for v2/hCaptcha)
        """
        # Check if CAPTCHA is enabled
        if not settings.RECAPTCHA_ENABLED:
            logger.debug("CAPTCHA verification skipped (disabled)")
            return True, "CAPTCHA disabled", None
        
        # Check for missing secret key
        if not settings.RECAPTCHA_SECRET_KEY:
            logger.error("CAPTCHA secret key not configured")
            # Fail open in development, fail closed in production
            if settings.APP_ENV == "production":
                return False, "CAPTCHA configuration error", None
            return True, "CAPTCHA not configured (dev mode)", None
        
        # Validate token
        if not token or not token.strip():
            logger.warning("Empty CAPTCHA token received")
            return False, "CAPTCHA token is required", None
        
        # Determine which provider to use
        if settings.CAPTCHA_PROVIDER == "hcaptcha":
            return await cls._verify_hcaptcha(token, remote_ip)
        else:
            return await cls._verify_recaptcha(
                token, remote_ip, expected_action, min_score
            )
    
    @classmethod
    async def _verify_recaptcha(
        cls,
        token: str,
        remote_ip: Optional[str],
        expected_action: Optional[str],
        min_score: float
    ) -> Tuple[bool, str, Optional[float]]:
        """Verify token with Google reCAPTCHA."""
        payload = {
            "secret": settings.RECAPTCHA_SECRET_KEY,
            "response": token,
        }
        if remote_ip:
            payload["remoteip"] = remote_ip
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    cls.RECAPTCHA_VERIFY_URL,
                    data=payload
                )
                response.raise_for_status()
                result = response.json()
        except httpx.TimeoutException:
            logger.error("reCAPTCHA verification timeout")
            # Fail open on timeout to prevent blocking users
            return True, "CAPTCHA verification timeout (allowed)", None
        except httpx.HTTPError as e:
            logger.error(f"reCAPTCHA HTTP error: {e}")
            return True, "CAPTCHA service unavailable (allowed)", None
        except Exception as e:
            logger.error(f"reCAPTCHA verification error: {e}")
            return True, "CAPTCHA verification error (allowed)", None
        
        # Check basic success
        if not result.get("success"):
            error_codes = result.get("error-codes", [])
            logger.warning(f"reCAPTCHA verification failed: {error_codes}")
            
            # Map error codes to user-friendly messages
            if "timeout-or-duplicate" in error_codes:
                return False, "CAPTCHA expired. Please try again.", None
            elif "invalid-input-response" in error_codes:
                return False, "Invalid CAPTCHA. Please try again.", None
            elif "invalid-input-secret" in error_codes:
                logger.error("Invalid reCAPTCHA secret key!")
                return False, "CAPTCHA configuration error", None
            else:
                return False, "CAPTCHA verification failed", None
        
        # For reCAPTCHA v3, check score
        score = result.get("score")
        if score is not None:
            # v3 response includes score
            logger.info(f"reCAPTCHA v3 score: {score}")
            
            # Check action if expected
            if expected_action:
                actual_action = result.get("action")
                if actual_action != expected_action:
                    logger.warning(
                        f"reCAPTCHA action mismatch: expected={expected_action}, "
                        f"actual={actual_action}"
                    )
                    return False, "Invalid CAPTCHA action", score
            
            # Check score threshold
            if score < min_score:
                logger.warning(f"reCAPTCHA score too low: {score} < {min_score}")
                return False, "Suspicious activity detected. Please try again.", score
            
            return True, "CAPTCHA verified", score
        
        # v2 response (no score)
        return True, "CAPTCHA verified", None
    
    @classmethod
    async def _verify_hcaptcha(
        cls,
        token: str,
        remote_ip: Optional[str]
    ) -> Tuple[bool, str, Optional[float]]:
        """Verify token with hCaptcha."""
        payload = {
            "secret": settings.RECAPTCHA_SECRET_KEY,  # Reuse same config
            "response": token,
        }
        if remote_ip:
            payload["remoteip"] = remote_ip
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    cls.HCAPTCHA_VERIFY_URL,
                    data=payload
                )
                response.raise_for_status()
                result = response.json()
        except Exception as e:
            logger.error(f"hCaptcha verification error: {e}")
            return True, "CAPTCHA verification error (allowed)", None
        
        if not result.get("success"):
            error_codes = result.get("error-codes", [])
            logger.warning(f"hCaptcha verification failed: {error_codes}")
            return False, "CAPTCHA verification failed", None
        
        return True, "CAPTCHA verified", None
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if CAPTCHA is currently enabled."""
        return settings.RECAPTCHA_ENABLED and bool(settings.RECAPTCHA_SECRET_KEY)
    
    @classmethod
    def get_site_key(cls) -> Optional[str]:
        """Get the public site key for frontend use."""
        if not settings.RECAPTCHA_ENABLED:
            return None
        return settings.RECAPTCHA_SITE_KEY
    
    @classmethod
    def get_provider(cls) -> str:
        """Get the current CAPTCHA provider name."""
        return settings.CAPTCHA_PROVIDER if settings.RECAPTCHA_ENABLED else "none"
