"""
WSGI/ASGI entry point with Socket.IO integration
"""
import socketio
from app.main import app
from app.websocket.socket_manager import socket_manager

# Wrap FastAPI app with Socket.IO
socket_app = socketio.ASGIApp(
    socket_manager.sio,
    other_asgi_app=app,
    socketio_path='socket.io'
)

import logging
logger = logging.getLogger(__name__)
logger.info("✓ Socket.IO ASGI app created and wrapping FastAPI")
