"""
Tests for password hashing and validation utilities
"""
import pytest
from app.utils.password import (
    hash_password,
    verify_password,
    validate_password_strength,
    validate_credentials,
    needs_rehash
)


class TestPasswordHashing:
    """Tests for password hashing functionality"""
    
    def test_hash_password_returns_valid_hash(self):
        """Test that hashing returns a valid bcrypt hash"""
        password = "TestPassword123!"
        hashed = hash_password(password)
        
        # Bcrypt hashes are 60 characters
        assert len(hashed) == 60
        # Should start with $2b$ (bcrypt identifier)
        assert hashed.startswith("$2b$")
        
    def test_hash_password_is_different_each_time(self):
        """Test that same password produces different hashes (due to salt)"""
        password = "TestPassword123!"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        assert hash1 != hash2
        
    def test_hash_password_empty_raises_error(self):
        """Test that empty password raises ValueError"""
        with pytest.raises(ValueError, match="Password cannot be empty"):
            hash_password("")
            
    def test_hash_password_too_long_raises_error(self):
        """Test that overly long password raises ValueError"""
        long_password = "a" * 73  # Max is 72 bytes
        with pytest.raises(ValueError, match="cannot exceed 72 bytes"):
            hash_password(long_password)
            
    def test_hash_password_max_length_succeeds(self):
        """Test that max length password is accepted"""
        max_password = "A" * 72  # Bcrypt max is 72 bytes
        hashed = hash_password(max_password)
        assert len(hashed) == 60


class TestPasswordVerification:
    """Tests for password verification functionality"""
    
    def test_verify_correct_password(self):
        """Test that correct password verifies successfully"""
        password = "CorrectPassword123!"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
        
    def test_verify_incorrect_password(self):
        """Test that incorrect password fails verification"""
        password = "CorrectPassword123!"
        hashed = hash_password(password)
        
        assert verify_password("WrongPassword123!", hashed) is False
        
    def test_verify_case_sensitive(self):
        """Test that password verification is case-sensitive"""
        password = "TestPassword123!"
        hashed = hash_password(password)
        
        assert verify_password("testpassword123!", hashed) is False
        assert verify_password("TESTPASSWORD123!", hashed) is False
        
    def test_verify_empty_password_returns_false(self):
        """Test that empty password returns False (not exception)"""
        hashed = hash_password("TestPassword123!")
        assert verify_password("", hashed) is False
        
    def test_verify_empty_hash_returns_false(self):
        """Test that empty hash returns False (not exception)"""
        assert verify_password("TestPassword123!", "") is False
        
    def test_verify_invalid_hash_returns_false(self):
        """Test that invalid hash format returns False gracefully"""
        assert verify_password("TestPassword123!", "invalid_hash") is False
        assert verify_password("TestPassword123!", "not_a_bcrypt_hash") is False


class TestPasswordStrength:
    """Tests for password strength validation"""
    
    def test_strong_password_validates(self):
        """Test that strong password passes all checks"""
        valid, errors = validate_password_strength("StrongPass123!")
        assert valid is True
        assert len(errors) == 0
        
    def test_password_too_short(self):
        """Test that short password fails validation"""
        valid, errors = validate_password_strength("Sh0rt!")
        assert valid is False
        assert any("at least 8 characters" in err for err in errors)
        
    def test_password_too_long(self):
        """Test that overly long password fails validation"""
        long_password = "A" * 73 + "a1!"  # Over 72 bytes
        valid, errors = validate_password_strength(long_password)
        assert valid is False
        assert any("cannot exceed 72 bytes" in err for err in errors)
        
    def test_password_no_uppercase(self):
        """Test that password without uppercase fails"""
        valid, errors = validate_password_strength("lowercase123!")
        assert valid is False
        assert any("uppercase letter" in err for err in errors)
        
    def test_password_no_lowercase(self):
        """Test that password without lowercase fails"""
        valid, errors = validate_password_strength("UPPERCASE123!")
        assert valid is False
        assert any("lowercase letter" in err for err in errors)
        
    def test_password_no_digit(self):
        """Test that password without digit fails"""
        valid, errors = validate_password_strength("NoDigitsHere!")
        assert valid is False
        assert any("digit" in err for err in errors)
        
    def test_password_no_special_char(self):
        """Test that password without special character fails"""
        valid, errors = validate_password_strength("NoSpecial123")
        assert valid is False
        assert any("special character" in err for err in errors)
        
    def test_empty_password(self):
        """Test that empty password fails validation"""
        valid, errors = validate_password_strength("")
        assert valid is False
        assert any("cannot be empty" in err for err in errors)
        
    def test_multiple_errors(self):
        """Test that multiple validation errors are returned"""
        valid, errors = validate_password_strength("weak")
        assert valid is False
        assert len(errors) >= 3  # Multiple requirements failed
        
    def test_various_special_characters(self):
        """Test that various special characters are accepted"""
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        for char in special_chars:
            password = f"Test123{char}abc"
            valid, errors = validate_password_strength(password)
            assert valid is True, f"Failed for character: {char}"


class TestCredentialsValidation:
    """Tests for credentials validation"""
    
    def test_valid_credentials(self):
        """Test that valid credentials pass validation"""
        valid, errors = validate_credentials("testuser", "password123")
        assert valid is True
        assert len(errors) == 0
        
    def test_empty_username(self):
        """Test that empty username fails validation"""
        valid, errors = validate_credentials("", "password123")
        assert valid is False
        assert any("Username cannot be empty" in err for err in errors)
        
    def test_whitespace_only_username(self):
        """Test that whitespace-only username fails validation"""
        valid, errors = validate_credentials("   ", "password123")
        assert valid is False
        assert any("Username cannot be empty" in err for err in errors)
        
    def test_empty_password(self):
        """Test that empty password fails validation"""
        valid, errors = validate_credentials("testuser", "")
        assert valid is False
        assert any("Password cannot be empty" in err for err in errors)
        
    def test_both_empty(self):
        """Test that both empty fails with multiple errors"""
        valid, errors = validate_credentials("", "")
        assert valid is False
        assert len(errors) == 2


class TestPasswordRehashing:
    """Tests for password rehash detection"""
    
    def test_current_hash_does_not_need_rehash(self):
        """Test that current hash format doesn't need rehashing"""
        password = "TestPassword123!"
        hashed = hash_password(password)
        
        # Fresh hash with current settings should not need rehash
        assert needs_rehash(hashed) is False
        
    def test_invalid_hash_needs_rehash(self):
        """Test that invalid hash returns True"""
        assert needs_rehash("invalid_hash") is True
        assert needs_rehash("") is True


class TestSecurityProperties:
    """Property-based tests for security requirements"""
    
    def test_password_hashing_is_irreversible(self):
        """
        Property: Password hashing should be one-way (irreversible)
        We cannot recover the original password from the hash
        """
        password = "TestPassword123!"
        hashed = hash_password(password)
        
        # Hash should not contain the original password
        assert password not in hashed
        # Hash should be significantly different from password
        assert hashed != password
        # Cannot reverse the hash
        assert len(hashed) == 60  # Fixed bcrypt length
        
    def test_empty_credentials_always_rejected(self):
        """
        Property: Empty credentials should always be rejected
        """
        # Empty username
        valid, _ = validate_credentials("", "password")
        assert valid is False
        
        # Empty password
        valid, _ = validate_credentials("username", "")
        assert valid is False
        
        # Both empty
        valid, _ = validate_credentials("", "")
        assert valid is False
        
    def test_constant_time_comparison(self):
        """
        Property: Password verification timing should not leak information
        This is a basic test - actual timing analysis would be more complex
        """
        password = "TestPassword123!"
        hashed = hash_password(password)
        
        # Both should return False, regardless of how wrong the password is
        assert verify_password("", hashed) is False
        assert verify_password("a", hashed) is False
        assert verify_password("completely_wrong", hashed) is False
        assert verify_password(password[:-1], hashed) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
