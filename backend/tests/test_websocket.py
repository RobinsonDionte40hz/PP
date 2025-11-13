"""
Tests for WebSocket functionality
"""
import pytest
import asyncio
from app.websocket.socket_manager import SocketManager
from app.websocket.events import (
    create_progress_event,
    create_metrics_event,
    create_agent_event,
    create_log_event,
    create_status_event,
    create_completion_event,
    create_error_event,
    EventTypes
)


class TestSocketManager:
    """Test WebSocket manager"""
    
    @pytest.fixture
    def socket_manager(self):
        """Create socket manager instance"""
        return SocketManager()
    
    def test_socket_manager_initialization(self, socket_manager):
        """Test socket manager initializes correctly"""
        assert socket_manager.sio is not None
        assert socket_manager.active_connections == {}
    
    def test_get_subscriber_count_empty(self, socket_manager):
        """Test subscriber count for non-existent prediction"""
        count = socket_manager.get_subscriber_count("pred_test")
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_emit_progress_update(self, socket_manager):
        """Test emitting progress update"""
        # This should not raise an error even with no subscribers
        await socket_manager.emit_progress_update(
            "pred_test",
            {"iteration": 100, "total": 1000, "progress": 10.0}
        )
    
    @pytest.mark.asyncio
    async def test_emit_metrics_update(self, socket_manager):
        """Test emitting metrics update"""
        await socket_manager.emit_metrics_update(
            "pred_test",
            {"energy": -100.0, "rmsd": 5.0}
        )
    
    @pytest.mark.asyncio
    async def test_emit_agent_update(self, socket_manager):
        """Test emitting agent update"""
        await socket_manager.emit_agent_update(
            "pred_test",
            {"agent_id": 1, "status": "exploring"}
        )
    
    @pytest.mark.asyncio
    async def test_emit_event_log(self, socket_manager):
        """Test emitting event log"""
        await socket_manager.emit_event_log(
            "pred_test",
            {"level": "info", "message": "Test message"}
        )
    
    @pytest.mark.asyncio
    async def test_emit_status_change(self, socket_manager):
        """Test emitting status change"""
        await socket_manager.emit_status_change(
            "pred_test",
            {"old_status": "running", "new_status": "completed"}
        )
    
    @pytest.mark.asyncio
    async def test_emit_completion(self, socket_manager):
        """Test emitting completion"""
        await socket_manager.emit_completion(
            "pred_test",
            {"final_energy": -150.0, "final_rmsd": 2.5}
        )
    
    @pytest.mark.asyncio
    async def test_emit_error(self, socket_manager):
        """Test emitting error"""
        await socket_manager.emit_error(
            "pred_test",
            {"error_message": "Test error", "error_type": "test"}
        )


class TestWebSocketEvents:
    """Test WebSocket event creation"""
    
    def test_create_progress_event(self):
        """Test creating progress event"""
        event = create_progress_event(
            prediction_id="pred_123",
            iteration=100,
            total_iterations=1000,
            progress_percentage=10.0
        )
        
        assert event["type"] == EventTypes.PROGRESS_UPDATE
        assert event["prediction_id"] == "pred_123"
        assert event["data"]["iteration"] == 100
        assert event["data"]["total_iterations"] == 1000
        assert event["data"]["progress_percentage"] == 10.0
        assert "timestamp" in event
    
    def test_create_metrics_event(self):
        """Test creating metrics event"""
        event = create_metrics_event(
            prediction_id="pred_123",
            energy=-100.0,
            rmsd=5.0,
            aggressiveness=9.0,
            consistency=0.7
        )
        
        assert event["type"] == EventTypes.METRICS_UPDATE
        assert event["prediction_id"] == "pred_123"
        assert event["data"]["energy"] == -100.0
        assert event["data"]["rmsd"] == 5.0
        assert event["data"]["aggressiveness"] == 9.0
        assert event["data"]["consistency"] == 0.7
    
    def test_create_metrics_event_with_extras(self):
        """Test creating metrics event with extra fields"""
        event = create_metrics_event(
            prediction_id="pred_123",
            energy=-100.0,
            rmsd=5.0,
            aggressiveness=9.0,
            consistency=0.7,
            memory_count=50,
            best_energy=-120.0
        )
        
        assert event["data"]["memory_count"] == 50
        assert event["data"]["best_energy"] == -120.0
    
    def test_create_agent_event(self):
        """Test creating agent event"""
        event = create_agent_event(
            prediction_id="pred_123",
            agent_id=5,
            status="exploring"
        )
        
        assert event["type"] == EventTypes.AGENT_UPDATE
        assert event["prediction_id"] == "pred_123"
        assert event["data"]["agent_id"] == 5
        assert event["data"]["status"] == "exploring"
    
    def test_create_log_event(self):
        """Test creating log event"""
        event = create_log_event(
            prediction_id="pred_123",
            level="info",
            message="Test message"
        )
        
        assert event["type"] == EventTypes.EVENT_LOG
        assert event["prediction_id"] == "pred_123"
        assert event["data"]["level"] == "info"
        assert event["data"]["message"] == "Test message"
    
    def test_create_status_event(self):
        """Test creating status change event"""
        event = create_status_event(
            prediction_id="pred_123",
            old_status="running",
            new_status="completed"
        )
        
        assert event["type"] == EventTypes.STATUS_CHANGE
        assert event["prediction_id"] == "pred_123"
        assert event["data"]["old_status"] == "running"
        assert event["data"]["new_status"] == "completed"
    
    def test_create_completion_event(self):
        """Test creating completion event"""
        final_metrics = {
            "best_energy": -150.0,
            "best_rmsd": 2.5,
            "conformations_explored": 10000
        }
        
        event = create_completion_event(
            prediction_id="pred_123",
            final_metrics=final_metrics
        )
        
        assert event["type"] == EventTypes.PREDICTION_COMPLETE
        assert event["prediction_id"] == "pred_123"
        assert event["data"] == final_metrics
    
    def test_create_error_event(self):
        """Test creating error event"""
        event = create_error_event(
            prediction_id="pred_123",
            error_message="Something went wrong",
            error_type="validation_error"
        )
        
        assert event["type"] == EventTypes.PREDICTION_ERROR
        assert event["prediction_id"] == "pred_123"
        assert event["data"]["error_message"] == "Something went wrong"
        assert event["data"]["error_type"] == "validation_error"
    
    def test_event_types_constants(self):
        """Test EventTypes constants"""
        assert EventTypes.PROGRESS_UPDATE == "progress_update"
        assert EventTypes.METRICS_UPDATE == "metrics_update"
        assert EventTypes.AGENT_UPDATE == "agent_update"
        assert EventTypes.EVENT_LOG == "event_log"
        assert EventTypes.STATUS_CHANGE == "status_change"
        assert EventTypes.PREDICTION_COMPLETE == "prediction_complete"
        assert EventTypes.PREDICTION_ERROR == "prediction_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
