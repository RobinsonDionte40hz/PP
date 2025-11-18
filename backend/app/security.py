"""
Security configuration and utilities for the API
"""
from typing import Optional
import re

class SecurityConfig:
    """Security settings for the API"""
    
    # Rate Limits (requests per minute)
    RATE_LIMIT_CREATE_PREDICTION = "10/minute"  # Creating predictions (compute-intensive)
    RATE_LIMIT_LIST_PREDICTIONS = "30/minute"   # Listing predictions (less intensive)
    RATE_LIMIT_GET_PREDICTION = "60/minute"     # Getting single prediction (read-only)
    RATE_LIMIT_DEFAULT = "100/minute"           # Default for other endpoints
    
    # Sequence Validation
    MAX_SEQUENCE_LENGTH = 1000    # Maximum protein length (prevents crashes)
    MIN_SEQUENCE_LENGTH = 3       # Minimum protein length
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")  # Standard 20 amino acids
    
    # Configuration Limits
    MAX_ITERATIONS = 10000        # Maximum iterations per prediction
    MIN_ITERATIONS = 100          # Minimum iterations
    MAX_AGENTS = 100              # Maximum number of agents
    MIN_AGENTS = 1                # Minimum number of agents
    MAX_CHECKPOINT_INTERVAL = 1000
    MIN_CHECKPOINT_INTERVAL = 10
    
    # Pagination Limits
    MAX_PAGE_SIZE = 100           # Maximum items per page
    DEFAULT_PAGE_SIZE = 20
    
    # File Upload (future use)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_FILE_TYPES = {".pdb", ".fasta", ".fa"}


def validate_sequence_security(sequence: str) -> tuple[bool, Optional[str]]:
    """
    Additional security validation for protein sequences.
    
    Returns:
        (is_valid, error_message)
    """
    # Check for SQL injection patterns
    sql_patterns = [
        r"(union|select|insert|update|delete|drop|create|alter)\s",
        r"(--|\*\/|\/\*)",
        r"(;|\||&&|\$\()"
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, sequence.lower()):
            return False, "Sequence contains suspicious patterns"
    
    # Check for script injection
    script_patterns = [
        r"<script",
        r"javascript:",
        r"onerror=",
        r"onload="
    ]
    
    for pattern in script_patterns:
        if re.search(pattern, sequence.lower()):
            return False, "Sequence contains invalid characters"
    
    # Check for excessive repetition (potential DoS)
    max_repetition = 50  # Max consecutive same character
    for char in set(sequence):
        if char * max_repetition in sequence:
            return False, f"Excessive repetition of amino acid '{char}' detected"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal attacks.
    """
    # Remove any path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    
    # Remove parent directory references
    filename = filename.replace("..", "_")
    
    # Keep only alphanumeric, dash, underscore, and dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename


def get_rate_limit_message(endpoint: str, limit: str) -> str:
    """
    Generate user-friendly rate limit message.
    """
    return f"Rate limit exceeded for {endpoint}. Limit: {limit}. Please try again later."
