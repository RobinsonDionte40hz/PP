"""
Property-based tests for JWT token security (Task 5.1).

Tests Property 14 from design.md:
- Property 14: Session tokens are cryptographically secure
"""
import pytest
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume
import uuid
import jwt
from jose import JWTError

from app.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token_type,
    extract_jti_from_token,
    get_secret_key,
)
from app.config import settings


# Custom strategies for generating test data
@st.composite
def user_key_id_strategy(draw):
    """Generate valid user key IDs (UUID format)"""
    return str(uuid.uuid4())


@st.composite
def username_strategy(draw):
    """Generate valid usernames (3-50 chars, alphanumeric + underscore/dash)"""
    length = draw(st.integers(min_value=3, max_value=50))
    chars = st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-")
    return ''.join(draw(st.lists(chars, min_size=length, max_size=length)))


@st.composite
def token_data_strategy(draw):
    """Generate valid token data"""
    return {
        "sub": draw(user_key_id_strategy()),
        "username": draw(username_strategy()),
        "jti": str(uuid.uuid4()),
    }


# ==================== Property 14: Session tokens are cryptographically secure ====================
# Feature: user-authentication, Property 14: Session tokens are cryptographically secure

@pytest.mark.asyncio
@settings(max_examples=100, deadline=2000)
@given(token_data=token_data_strategy())
def test_property_14_tokens_have_sufficient_entropy(token_data: dict):
    """
    Property 14: For any generated session token, it should have sufficient entropy
    (at least 128 bits), be unpredictable, and not reveal information about the user.
    
    Validates: Requirements 6.2
    """
    # Create access token
    access_token = create_access_token(token_data)
    
    # Property 1: Token should be non-empty and reasonably long
    assert len(access_token) > 100, "Token should have sufficient length"
    
    # Property 2: Token should be different from input data
    assert token_data["sub"] not in access_token, "Token should not contain plaintext user ID"
    assert token_data["username"] not in access_token, "Token should not contain plaintext username"
    
    # Property 3: Token should have 3 parts (header.payload.signature)
    parts = access_token.split('.')
    assert len(parts) == 3, "JWT should have 3 parts (header.payload.signature)"
    
    # Property 4: Each part should be base64 encoded and non-empty
    for i, part in enumerate(parts):
        assert len(part) > 0, f"JWT part {i} should not be empty"
        assert part.replace('-', '').replace('_', '').isalnum(), f"JWT part {i} should be base64url encoded"
    
    # Property 5: Signature should have high entropy (at least 32 bytes)
    signature = parts[2]
    assert len(signature) >= 43, "JWT signature should have at least 256 bits of entropy (43+ base64 chars)"


@pytest.mark.asyncio
@settings(max_examples=100, deadline=2000)
@given(token_data=token_data_strategy())
def test_property_14_tokens_are_unpredictable(token_data: dict):
    """
    Property 14: Tokens should be unpredictable - same input should produce different tokens
    due to different timestamps and jti values.
    
    Validates: Requirements 6.2
    """
    # Create two tokens with same base data but different jti
    token_data_1 = token_data.copy()
    token_data_1["jti"] = str(uuid.uuid4())
    
    token_data_2 = token_data.copy()
    token_data_2["jti"] = str(uuid.uuid4())
    
    token_1 = create_access_token(token_data_1)
    token_2 = create_access_token(token_data_2)
    
    # Property: Different jti should produce different tokens
    assert token_1 != token_2, "Tokens with different jti should be different"
    
    # Property: Signatures should be different
    sig_1 = token_1.split('.')[2]
    sig_2 = token_2.split('.')[2]
    assert sig_1 != sig_2, "Signatures should be different for different tokens"


@pytest.mark.asyncio
@settings(max_examples=100, deadline=2000)
@given(token_data=token_data_strategy())
def test_property_14_tokens_contain_required_claims(token_data: dict):
    """
    Property 14: Tokens should contain all required claims including jti for session tracking.
    
    Validates: Requirements 2.1, 4.1, 6.2
    """
    # Create access token
    access_token = create_access_token(token_data)
    
    # Decode token
    payload = decode_token(access_token)
    
    # Property: Required claims should be present
    assert "sub" in payload, "Token should contain 'sub' (user key ID)"
    assert "jti" in payload, "Token should contain 'jti' (JWT ID for session management)"
    assert "type" in payload, "Token should contain 'type' (access/refresh)"
    assert "iat" in payload, "Token should contain 'iat' (issued at)"
    assert "exp" in payload, "Token should contain 'exp' (expiration)"
    
    # Property: Token type should be correct
    assert payload["type"] == "access", "Access token should have type 'access'"
    
    # Property: User data should match
    assert payload["sub"] == token_data["sub"], "Token 'sub' should match user key ID"
    assert payload["username"] == token_data["username"], "Token should contain username"
    
    # Property: JTI should be a valid UUID
    jti = payload["jti"]
    try:
        uuid.UUID(jti)
    except ValueError:
        pytest.fail("JTI should be a valid UUID")


@pytest.mark.asyncio
@settings(max_examples=100, deadline=2000)
@given(token_data=token_data_strategy())
def test_property_14_refresh_tokens_have_different_expiry(token_data: dict):
    """
    Property 14: Refresh tokens should have longer expiration than access tokens.
    
    Validates: Requirements 4.1
    """
    # Create both token types
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    # Decode both
    access_payload = decode_token(access_token)
    refresh_payload = decode_token(refresh_token)
    
    # Property: Refresh token should expire much later than access token
    access_exp = datetime.fromtimestamp(access_payload["exp"])
    refresh_exp = datetime.fromtimestamp(refresh_payload["exp"])
    
    time_diff = (refresh_exp - access_exp).total_seconds()
    assert time_diff > 86400, "Refresh token should expire at least 1 day after access token"
    
    # Property: Token types should be different
    assert access_payload["type"] == "access", "Access token should have type 'access'"
    assert refresh_payload["type"] == "refresh", "Refresh token should have type 'refresh'"


@pytest.mark.asyncio
@settings(max_examples=50, deadline=2000)
@given(token_data=token_data_strategy())
def test_property_14_tokens_cannot_be_forged(token_data: dict):
    """
    Property 14: Tokens signed with wrong secret should be rejected.
    
    Validates: Requirements 6.2
    """
    # Create valid token
    valid_token = create_access_token(token_data)
    
    # Try to create token with different secret
    wrong_secret = "WRONG_SECRET_KEY_12345"
    
    # Manually create token with wrong secret
    to_encode = token_data.copy()
    to_encode.update({
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "iat": datetime.utcnow(),
        "type": "access",
        "jti": str(uuid.uuid4())
    })
    forged_token = jwt.encode(to_encode, wrong_secret, algorithm=settings.JWT_ALGORITHM)
    
    # Property: Forged token should be rejected
    with pytest.raises(Exception) as exc_info:
        decode_token(forged_token)
    
    assert exc_info.value.status_code == 401, "Forged token should result in 401 error"


@pytest.mark.asyncio
@settings(max_examples=50, deadline=2000)
@given(token_data=token_data_strategy())
def test_property_14_expired_tokens_are_rejected(token_data: dict):
    """
    Property 14: Expired tokens should be rejected during validation.
    
    Validates: Requirements 4.1
    """
    # Create token with negative expiry (already expired)
    expired_token = create_access_token(
        token_data,
        expires_delta=timedelta(seconds=-10)
    )
    
    # Property: Expired token should be rejected
    with pytest.raises(Exception) as exc_info:
        decode_token(expired_token)
    
    assert exc_info.value.status_code == 401, "Expired token should result in 401 error"
    assert "expired" in str(exc_info.value.detail).lower(), "Error should mention expiration"


# ==================== Additional Unit Tests ====================

def test_verify_token_type_access():
    """Test token type verification for access tokens"""
    payload = {"type": "access", "sub": "user123"}
    assert verify_token_type(payload, "access") is True
    assert verify_token_type(payload, "refresh") is False


def test_verify_token_type_refresh():
    """Test token type verification for refresh tokens"""
    payload = {"type": "refresh", "sub": "user123"}
    assert verify_token_type(payload, "refresh") is True
    assert verify_token_type(payload, "access") is False


def test_extract_jti_from_valid_token():
    """Test JTI extraction from valid token"""
    token_data = {
        "sub": str(uuid.uuid4()),
        "username": "test_user",
        "jti": str(uuid.uuid4())
    }
    token = create_access_token(token_data)
    
    extracted_jti = extract_jti_from_token(token)
    assert extracted_jti == token_data["jti"]


def test_extract_jti_from_expired_token():
    """Test JTI extraction from expired token (for logout)"""
    token_data = {
        "sub": str(uuid.uuid4()),
        "username": "test_user",
        "jti": str(uuid.uuid4())
    }
    # Create expired token
    expired_token = create_access_token(
        token_data,
        expires_delta=timedelta(seconds=-10)
    )
    
    # Should still extract JTI even though token is expired
    extracted_jti = extract_jti_from_token(expired_token)
    assert extracted_jti == token_data["jti"]


def test_extract_jti_from_invalid_token():
    """Test JTI extraction from invalid token"""
    invalid_token = "invalid.token.here"
    
    extracted_jti = extract_jti_from_token(invalid_token)
    assert extracted_jti is None


def test_token_without_jti_gets_auto_generated():
    """Test that tokens without jti get one auto-generated"""
    token_data = {
        "sub": str(uuid.uuid4()),
        "username": "test_user",
        # No jti provided
    }
    
    token = create_access_token(token_data)
    payload = decode_token(token)
    
    # Should have auto-generated jti
    assert "jti" in payload
    try:
        uuid.UUID(payload["jti"])
    except ValueError:
        pytest.fail("Auto-generated JTI should be a valid UUID")


def test_access_token_expiration_time():
    """Test that access token has correct expiration time"""
    token_data = {
        "sub": str(uuid.uuid4()),
        "username": "test_user",
        "jti": str(uuid.uuid4())
    }
    
    token = create_access_token(token_data)
    payload = decode_token(token)
    
    # Check expiration is approximately 30 minutes from now
    exp = datetime.fromtimestamp(payload["exp"])
    iat = datetime.fromtimestamp(payload["iat"])
    
    duration = (exp - iat).total_seconds()
    expected = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    
    # Allow 2 second tolerance for test execution time
    assert abs(duration - expected) < 2, f"Access token should expire in {expected} seconds"


def test_refresh_token_expiration_time():
    """Test that refresh token has correct expiration time"""
    token_data = {
        "sub": str(uuid.uuid4()),
        "jti": str(uuid.uuid4())
    }
    
    token = create_refresh_token(token_data)
    payload = decode_token(token)
    
    # Check expiration is approximately 7 days from now
    exp = datetime.fromtimestamp(payload["exp"])
    iat = datetime.fromtimestamp(payload["iat"])
    
    duration = (exp - iat).total_seconds()
    expected = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 86400
    
    # Allow 2 second tolerance for test execution time
    assert abs(duration - expected) < 2, f"Refresh token should expire in {expected} seconds"


def test_token_contains_issued_at():
    """Test that tokens contain issued at timestamp"""
    token_data = {
        "sub": str(uuid.uuid4()),
        "username": "test_user",
        "jti": str(uuid.uuid4())
    }
    
    before = datetime.utcnow()
    token = create_access_token(token_data)
    after = datetime.utcnow()
    
    payload = decode_token(token)
    iat = datetime.fromtimestamp(payload["iat"])
    
    # iat should be between before and after
    assert before <= iat <= after, "Token issued at should be current time"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
