import asyncio
from app.services.auth_service import AuthService
from app.database import SessionLocal
from app.services.session_manager import get_session_manager

async def test():
    db = SessionLocal()
    sm = get_session_manager()
    
    AuthService.register_user(db, 'testuser', 'TestPass123!', 'test@example.com')
    success, message, data = await AuthService.login_user(
        db, 'testuser', 'TestPass123!', '192.168.1.1', 'TestAgent'
    )
    
    print(f'Login: success={success}, message={message}, data_exists={data is not None}')
    
    if success and data:
        token = data['access_token']
        user_key_id = data['user'].key_id
        print(f'Token: {token[:50]}...')
        print(f'User key_id: {user_key_id}')
        
        success2, message2 = await AuthService.logout_user(token, user_key_id)
        print(f'Logout: success={success2}, message={message2}')
    else:
        print(f'Login failed: {message}')

asyncio.run(test())
