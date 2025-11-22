"""
Password hashing and validation utilities
Uses bcrypt with cost factor 12 for secure password storage
"""
import re
import bcrypt
from typing import Tuple

# Bcrypt cost factor: 12 rounds (2^12 = 4096 rounds)
# This provides strong security while maintaining reasonable performance
BCRYPT_ROUNDS = 12

# Password strength requirements
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 72  # Bcrypt limitation is 72 bytes
REQUIRE_UPPERCASE = True
REQUIRE_LOWERCASE = True
REQUIRE_DIGIT = True
REQUIRE_SPECIAL = True


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt with cost factor 12.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        Hashed password string (bcrypt format)
        
    Raises:
        ValueError: If password is empty or exceeds max length
        
    Example:
        >>> hashed = hash_password("MySecurePass123!")
        >>> len(hashed)
        60
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    # Check byte length (bcrypt has 72-byte limit)
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password cannot exceed {MAX_PASSWORD_LENGTH} bytes")
    
    # Hash with bcrypt
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Return as string
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash using constant-time comparison.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Bcrypt hash to verify against
        
    Returns:
        True if password matches, False otherwise
        
    Note:
        Uses constant-time comparison to prevent timing attacks.
        
    Example:
        >>> hashed = hash_password("MySecurePass123!")
        >>> verify_password("MySecurePass123!", hashed)
        True
        >>> verify_password("WrongPassword", hashed)
        False
    """
    if not plain_password or not hashed_password:
        return False
    
    try:
        # bcrypt.checkpw uses constant-time comparison internally
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception:
        # Invalid hash format or other errors
        return False


def validate_password_strength(password: str) -> Tuple[bool, list[str]]:
    """
    Validate password meets strength requirements.
    
    Requirements:
    - Minimum 8 characters
    - Maximum 128 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid: bool, errors: list[str])
        
    Example:
        >>> validate_password_strength("weak")
        (False, ['Password must be at least 8 characters', ...])
        >>> validate_password_strength("StrongPass123!")
        (True, [])
    """
    errors = []
    
    if not password:
        errors.append("Password cannot be empty")
        return False, errors
    
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    
    # Check byte length (bcrypt limitation)
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > MAX_PASSWORD_LENGTH:
        errors.append(f"Password cannot exceed {MAX_PASSWORD_LENGTH} bytes")
    
    if REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    if REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    if REQUIRE_DIGIT and not re.search(r'\d', password):
        errors.append("Password must contain at least one digit")
    
    if REQUIRE_SPECIAL and not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
        errors.append("Password must contain at least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)")
    
    return len(errors) == 0, errors


def validate_credentials(username: str, password: str) -> Tuple[bool, list[str]]:
    """
    Validate both username and password are not empty.
    
    Args:
        username: Username to validate
        password: Password to validate
        
    Returns:
        Tuple of (is_valid: bool, errors: list[str])
        
    Example:
        >>> validate_credentials("", "password")
        (False, ['Username cannot be empty'])
        >>> validate_credentials("user", "")
        (False, ['Password cannot be empty'])
        >>> validate_credentials("user", "pass")
        (True, [])
    """
    errors = []
    
    if not username or not username.strip():
        errors.append("Username cannot be empty")
    
    if not password:
        errors.append("Password cannot be empty")
    
    return len(errors) == 0, errors


def needs_rehash(hashed_password: str) -> bool:
    """
    Check if a password hash needs to be rehashed.
    
    This is useful when upgrading security parameters (e.g., increasing cost factor).
    
    Args:
        hashed_password: Existing password hash
        
    Returns:
        True if hash should be updated, False otherwise
        
    Example:
        >>> old_hash = "$2a$10$..."  # Old hash with rounds=10
        >>> needs_rehash(old_hash)
        True
    """
    try:
        # Extract rounds from hash (format: $2b$rounds$...)
        if not hashed_password or not hashed_password.startswith('$2'):
            return True
        
        parts = hashed_password.split('$')
        if len(parts) < 4:
            return True
            
        current_rounds = int(parts[2])
        return current_rounds < BCRYPT_ROUNDS
    except Exception:
        # Invalid hash format
        return True
