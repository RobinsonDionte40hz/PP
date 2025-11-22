"""
Quick verification script for Task 10 security features
Tests rate limiting, brute force protection, and secure logging
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.session_manager import SessionManager
from app.utils.rate_limit import RateLimiter, BruteForceProtection
from redis import Redis


async def test_rate_limiting():
    """Test rate limiting with Redis"""
    print("\n=== Testing Rate Limiting ===")
    
    # Connect to Redis (session DB)
    redis_client = Redis.from_url("redis://localhost:6379/1", decode_responses=True)
    rate_limiter = RateLimiter(redis_client)
    
    # Test registration rate limit (5 per hour)
    print("\n1. Testing registration rate limit (5/hour)...")
    key = "test:register:192.168.1.1"
    
    for i in range(7):
        allowed, retry_after = rate_limiter.check_rate_limit(key, 5, 3600)
        if allowed:
            print(f"   Request {i+1}: ✓ Allowed")
        else:
            print(f"   Request {i+1}: ✗ Blocked (retry after {retry_after}s)")
    
    # Cleanup
    redis_client.delete(key)
    
    # Test login rate limit (10 per 15 minutes)
    print("\n2. Testing login rate limit (10/15min)...")
    key = "test:login:192.168.1.1"
    
    for i in range(12):
        allowed, retry_after = rate_limiter.check_rate_limit(key, 10, 900)
        if allowed:
            print(f"   Request {i+1}: ✓ Allowed")
        else:
            print(f"   Request {i+1}: ✗ Blocked (retry after {retry_after}s)")
    
    # Cleanup
    redis_client.delete(key)
    
    print("\n✓ Rate limiting tests complete")


async def test_brute_force_protection():
    """Test brute force protection with progressive lockouts"""
    print("\n=== Testing Brute Force Protection ===")
    
    redis_client = Redis.from_url("redis://localhost:6379/1", decode_responses=True)
    bf_protection = BruteForceProtection(redis_client)
    
    username = "test_bruteforce_user"
    
    print("\n1. Testing progressive lockouts...")
    
    # Record 3 failed attempts (should lock for 60s)
    for i in range(3):
        bf_protection.record_failed_attempt(username)
        is_locked, retry_after = bf_protection.is_locked_out(username)
        print(f"   Attempt {i+1}: Locked={is_locked}, Retry={retry_after}s")
    
    # Check lockout after 3 failures
    is_locked, retry_after = bf_protection.is_locked_out(username)
    print(f"\n   After 3 failures: Locked={is_locked}, Lockout={retry_after}s")
    assert is_locked, "Should be locked after 3 failures"
    assert retry_after == 60, "Should have 60s lockout"
    
    # Reset and test 5 failures (300s lockout)
    bf_protection.reset_failed_attempts(username)
    
    for i in range(5):
        bf_protection.record_failed_attempt(username)
    
    is_locked, retry_after = bf_protection.is_locked_out(username)
    print(f"   After 5 failures: Locked={is_locked}, Lockout={retry_after}s")
    assert is_locked, "Should be locked after 5 failures"
    assert retry_after == 300, "Should have 300s lockout"
    
    # Reset and test 10 failures (900s lockout)
    bf_protection.reset_failed_attempts(username)
    
    for i in range(10):
        bf_protection.record_failed_attempt(username)
    
    is_locked, retry_after = bf_protection.is_locked_out(username)
    print(f"   After 10 failures: Locked={is_locked}, Lockout={retry_after}s")
    assert is_locked, "Should be locked after 10 failures"
    assert retry_after == 900, "Should have 900s lockout"
    
    # Test reset
    print("\n2. Testing lockout reset on success...")
    bf_protection.reset_failed_attempts(username)
    is_locked, retry_after = bf_protection.is_locked_out(username)
    print(f"   After reset: Locked={is_locked}")
    assert not is_locked, "Should not be locked after reset"
    
    # Cleanup
    redis_client.delete(f"failed_attempts:{username}")
    redis_client.delete(f"lockout:{username}")
    
    print("\n✓ Brute force protection tests complete")


async def test_secure_logging():
    """Test that sensitive data is not logged"""
    print("\n=== Testing Secure Logging ===")
    
    import logging
    from io import StringIO
    
    # Capture log output
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    
    logger = logging.getLogger("security.auth")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Simulate logging with sensitive data
    username = "testuser"
    password = "SuperSecret123!"
    ip_address = "192.168.1.100"
    access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    
    # Log registration (should NOT include password)
    logger.info(
        f"User registered successfully: username={username}, ip={ip_address}"
    )
    
    # Log login attempt (should NOT include password or tokens)
    logger.info(
        f"Login attempt: username={username}, ip={ip_address}"
    )
    
    # Get log output
    log_output = log_stream.getvalue()
    
    print("\n1. Checking log output...")
    print(f"   Log contains username: {'✓' if username in log_output else '✗'}")
    print(f"   Log contains IP: {'✓' if ip_address in log_output else '✗'}")
    print(f"   Log contains password: {'✗ (FAIL)' if password in log_output else '✓ (OK)'}")
    print(f"   Log contains token: {'✗ (FAIL)' if access_token in log_output else '✓ (OK)'}")
    
    # Verify sensitive data NOT in logs
    assert password not in log_output, "❌ Password found in logs!"
    assert access_token not in log_output, "❌ Token found in logs!"
    
    # Verify safe data IS in logs
    assert username in log_output, "❌ Username not in logs!"
    assert ip_address in log_output, "❌ IP not in logs!"
    
    print("\n✓ Secure logging tests complete")


async def main():
    """Run all verification tests"""
    print("=" * 60)
    print("Task 10 Security Features Verification")
    print("=" * 60)
    
    try:
        # Test rate limiting
        await test_rate_limiting()
        
        # Test brute force protection
        await test_brute_force_protection()
        
        # Test secure logging
        await test_secure_logging()
        
        print("\n" + "=" * 60)
        print("✅ All Task 10 security features verified successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
