"""
WebSocket manager for real-time updates
"""
import socketio
from typing import Dict, Set, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


class SocketManager:
    """Manages WebSocket connections and rooms for real-time updates"""
    
    def __init__(self):
        # Create Socket.IO server
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins='*',
            logger=False,
            engineio_logger=False
        )
        
        # Track active connections
        self.active_connections: Dict[str, Set[str]] = {}  # prediction_id -> set of session_ids
        
        # Register event handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register Socket.IO event handlers"""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            """Handle client connection"""
            logger.info(f"Client connected: {sid}")
            return True
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {sid}")
            
            # Remove from all rooms
            for prediction_id, sessions in self.active_connections.items():
                if sid in sessions:
                    sessions.remove(sid)
                    await self.sio.leave_room(sid, prediction_id)
        
        @self.sio.event
        async def subscribe(sid, data):
            """Subscribe to prediction updates"""
            prediction_id = data.get('prediction_id')
            if not prediction_id:
                await self.sio.emit('error', {'message': 'prediction_id required'}, room=sid)
                return
            
            # Add to room
            await self.sio.enter_room(sid, prediction_id)
            
            # Track connection
            if prediction_id not in self.active_connections:
                self.active_connections[prediction_id] = set()
            self.active_connections[prediction_id].add(sid)
            
            logger.info(f"Client {sid} subscribed to prediction {prediction_id}")
            
            await self.sio.emit('subscribed', {'prediction_id': prediction_id}, room=sid)
        
        @self.sio.event
        async def unsubscribe(sid, data):
            """Unsubscribe from prediction updates"""
            prediction_id = data.get('prediction_id')
            if not prediction_id:
                return
            
            await self.sio.leave_room(sid, prediction_id)
            
            if prediction_id in self.active_connections:
                self.active_connections[prediction_id].discard(sid)
            
            logger.info(f"Client {sid} unsubscribed from prediction {prediction_id}")
            
            await self.sio.emit('unsubscribed', {'prediction_id': prediction_id}, room=sid)
    
    async def emit_progress_update(
        self,
        prediction_id: str,
        data: Dict[str, Any]
    ):
        """Emit progress update to all subscribers"""
        logger.info(f"Emitting progress_update to room {prediction_id}: iteration {data.get('iteration')}/{data.get('total_iterations')}")
        await self.sio.emit('progress_update', data, room=prediction_id)
    
    async def emit_metrics_update(
        self,
        prediction_id: str,
        data: Dict[str, Any]
    ):
        """Emit metrics update to all subscribers"""
        logger.debug(f"Emitting metrics_update to room {prediction_id}")
        await self.sio.emit('metrics_update', data, room=prediction_id)
    
    async def emit_agent_update(
        self,
        prediction_id: str,
        data: Dict[str, Any]
    ):
        """Emit agent status update to all subscribers"""
        await self.sio.emit('agent_update', data, room=prediction_id)
    
    async def emit_event_log(
        self,
        prediction_id: str,
        data: Dict[str, Any]
    ):
        """Emit event log message to all subscribers"""
        await self.sio.emit('event_log', data, room=prediction_id)
    
    async def emit_status_change(
        self,
        prediction_id: str,
        data: Dict[str, Any]
    ):
        """Emit status change notification"""
        await self.sio.emit('status_change', data, room=prediction_id)
    
    async def emit_completion(
        self,
        prediction_id: str,
        data: Dict[str, Any]
    ):
        """Emit completion notification"""
        await self.sio.emit('prediction_complete', data, room=prediction_id)
    
    async def emit_error(
        self,
        prediction_id: str,
        data: Dict[str, Any]
    ):
        """Emit error notification"""
        await self.sio.emit('prediction_error', data, room=prediction_id)
    
    def get_subscriber_count(self, prediction_id: str) -> int:
        """Get number of active subscribers for a prediction"""
        return len(self.active_connections.get(prediction_id, set()))


# Global socket manager instance
socket_manager = SocketManager()


# Create the wrapped ASGI app for uvicorn
def get_socket_asgi_app():
    """Get Socket.IO wrapped ASGI app"""
    import socketio as sio_module
    from app.main import app
    
    wrapped_app = sio_module.ASGIApp(
        socket_manager.sio,
        other_asgi_app=app,
        socketio_path='socket.io'
    )
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("✓ Socket.IO wrapped around FastAPI app")
    
    return wrapped_app
