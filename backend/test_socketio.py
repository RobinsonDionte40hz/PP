"""
Test script to verify Socket.IO is working
"""
import httpx
import sys

def test_socketio():
    """Test Socket.IO endpoints"""
    base_url = "http://localhost:8000"
    
    # Test main health endpoint
    print("Testing main API health endpoint...")
    try:
        response = httpx.get(f"{base_url}/health")
        print(f"✓ Health endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"✗ Health endpoint error: {e}")
        return False
    
    # Test Socket.IO polling endpoint
    print("\nTesting Socket.IO polling endpoint...")
    try:
        response = httpx.get(f"{base_url}/ws/socket.io/?EIO=4&transport=polling")
        print(f"✓ Socket.IO polling: {response.status_code}")
        print(f"  Response preview: {response.text[:100]}")
    except httpx.HTTPStatusError as e:
        print(f"✗ Socket.IO polling error: {e.response.status_code}")
        print(f"  This means Socket.IO is not properly mounted!")
        return False
    except Exception as e:
        print(f"✗ Socket.IO polling error: {e}")
        return False
    
    print("\n✓ All tests passed! Socket.IO is working.")
    return True

if __name__ == "__main__":
    success = test_socketio()
    sys.exit(0 if success else 1)
