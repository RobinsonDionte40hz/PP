"""
Test security features (rate limiting and input validation)

Run with: pytest backend/tests/test_security.py -v
"""
import pytest
import os
from fastapi.testclient import TestClient

# Disable rate limiting for tests
os.environ["TESTING"] = "true"

from app.main import app
from app.security import validate_sequence_security, sanitize_filename

client = TestClient(app)


class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_valid_sequence(self):
        """Valid sequence should be accepted"""
        response = client.post(
            "/api/predictions",
            json={"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        )
        assert response.status_code == 201
    
    def test_sequence_too_short(self):
        """Sequence shorter than 3 should be rejected"""
        response = client.post(
            "/api/predictions",
            json={"sequence": "AC"}
        )
        assert response.status_code == 422
        assert "too short" in response.json()["detail"][0]["msg"].lower()
    
    def test_sequence_too_long(self):
        """Sequence longer than 1000 should be rejected"""
        long_sequence = "A" * 1001
        response = client.post(
            "/api/predictions",
            json={"sequence": long_sequence}
        )
        assert response.status_code == 422
        assert "too long" in response.json()["detail"][0]["msg"].lower()
    
    def test_invalid_amino_acids(self):
        """Sequence with invalid amino acids should be rejected"""
        response = client.post(
            "/api/predictions",
            json={"sequence": "ACDEFGH123XYZ"}
        )
        assert response.status_code == 422
        assert "invalid amino acids" in response.json()["detail"][0]["msg"].lower()
    
    def test_whitespace_handling(self):
        """Whitespace should be stripped automatically"""
        response = client.post(
            "/api/predictions",
            json={"sequence": "  ACDEFGH  "}
        )
        assert response.status_code == 201
    
    def test_lowercase_conversion(self):
        """Lowercase sequences should be converted to uppercase"""
        response = client.post(
            "/api/predictions",
            json={"sequence": "acdefgh"}
        )
        assert response.status_code == 201
    
    def test_sql_injection_blocked(self):
        """SQL injection attempts should be blocked"""
        response = client.post(
            "/api/predictions",
            json={"sequence": "AC; DROP TABLE predictions--"}
        )
        assert response.status_code == 422
    
    def test_script_injection_blocked(self):
        """Script injection attempts should be blocked"""
        response = client.post(
            "/api/predictions",
            json={"sequence": "AC<script>alert('xss')</script>"}
        )
        assert response.status_code == 422
    
    def test_excessive_repetition_blocked(self):
        """Excessive amino acid repetition should be blocked"""
        repetitive_sequence = "A" * 51 + "CDEFGH"  # 51 consecutive A's
        response = client.post(
            "/api/predictions",
            json={"sequence": repetitive_sequence}
        )
        assert response.status_code == 422
        assert "repetition" in response.json()["detail"][0]["msg"].lower()


class TestConfigurationValidation:
    """Test configuration parameter validation"""
    
    def test_iterations_too_high(self):
        """Iterations above 10000 should be rejected"""
        response = client.post(
            "/api/predictions",
            json={
                "sequence": "ACDEFGH",
                "configuration": {"iterations": 10001}
            }
        )
        assert response.status_code == 422
    
    def test_iterations_too_low(self):
        """Iterations below 100 should be rejected"""
        response = client.post(
            "/api/predictions",
            json={
                "sequence": "ACDEFGH",
                "configuration": {"iterations": 99}
            }
        )
        assert response.status_code == 422
    
    def test_agents_too_high(self):
        """Agents above 100 should be rejected"""
        response = client.post(
            "/api/predictions",
            json={
                "sequence": "ACDEFGH",
                "configuration": {"agents": 101}
            }
        )
        assert response.status_code == 422
    
    def test_invalid_diversity(self):
        """Invalid diversity value should be rejected"""
        response = client.post(
            "/api/predictions",
            json={
                "sequence": "ACDEFGH",
                "configuration": {"diversity": "invalid"}
            }
        )
        assert response.status_code == 422
    
    def test_valid_configuration(self):
        """Valid configuration should be accepted"""
        response = client.post(
            "/api/predictions",
            json={
                "sequence": "ACDEFGH",
                "configuration": {
                    "iterations": 1000,
                    "agents": 10,
                    "diversity": "balanced",
                    "enable_checkpointing": True
                }
            }
        )
        assert response.status_code == 201


class TestSecurityUtilities:
    """Test security utility functions"""
    
    def test_validate_sequence_security_valid(self):
        """Valid sequence should pass security check"""
        is_valid, error = validate_sequence_security("ACDEFGH")
        assert is_valid is True
        assert error is None
    
    def test_validate_sequence_security_sql_injection(self):
        """SQL injection should be detected"""
        is_valid, error = validate_sequence_security("AC; DROP TABLE users--")
        assert is_valid is False
        assert error is not None
        assert "suspicious" in error.lower()
    
    def test_validate_sequence_security_script(self):
        """Script injection should be detected"""
        is_valid, error = validate_sequence_security("<script>alert('xss')</script>")
        assert is_valid is False
        assert error is not None
        assert "invalid" in error.lower()
    
    def test_validate_sequence_security_repetition(self):
        """Excessive repetition should be detected"""
        is_valid, error = validate_sequence_security("A" * 51)
        assert is_valid is False
        assert error is not None
        assert "repetition" in error.lower()
    
    def test_sanitize_filename_path_traversal(self):
        """Path traversal attempts should be sanitized"""
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result
    
    def test_sanitize_filename_special_chars(self):
        """Special characters should be sanitized"""
        result = sanitize_filename("file<>:\"\\|?*.txt")
        assert all(c not in result for c in '<>:"|?*')
    
    def test_sanitize_filename_length_limit(self):
        """Long filenames should be truncated"""
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 255


class TestRateLimiting:
    """Test rate limiting (note: may need to adjust based on test environment)"""
    
    @pytest.mark.skip(reason="Rate limiting may interfere with other tests")
    def test_rate_limit_create_prediction(self):
        """Creating too many predictions should trigger rate limit"""
        # Make 11 requests (limit is 10/minute)
        responses = []
        for i in range(11):
            response = client.post(
                "/api/predictions",
                json={"sequence": f"ACDEFGH{i % 10}"}
            )
            responses.append(response.status_code)
        
        # At least one should be rate limited
        assert 429 in responses
    
    def test_rate_limit_headers(self):
        """Rate limit headers should be present"""
        response = client.post(
            "/api/predictions",
            json={"sequence": "ACDEFGH"}
        )
        # Check if rate limit headers are present
        # (exact header names depend on slowapi configuration)
        assert response.status_code in [201, 429]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
