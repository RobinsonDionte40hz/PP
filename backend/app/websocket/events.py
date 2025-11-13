"""
WebSocket event types and utilities
"""
from typing import Dict, Any
from datetime import datetime


class EventTypes:
    """WebSocket event type constants"""
    PROGRESS_UPDATE = "progress_update"
    METRICS_UPDATE = "metrics_update"
    AGENT_UPDATE = "agent_update"
    EVENT_LOG = "event_log"
    STATUS_CHANGE = "status_change"
    PREDICTION_COMPLETE = "prediction_complete"
    PREDICTION_ERROR = "prediction_error"


def create_progress_event(
    prediction_id: str,
    iteration: int,
    total_iterations: int,
    progress_percentage: float
) -> Dict[str, Any]:
    """Create a progress update event"""
    return {
        "type": EventTypes.PROGRESS_UPDATE,
        "prediction_id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "iteration": iteration,
            "total_iterations": total_iterations,
            "progress_percentage": progress_percentage
        }
    }


def create_metrics_event(
    prediction_id: str,
    energy: float,
    rmsd: float,
    aggressiveness: float,
    consistency: float,
    **kwargs
) -> Dict[str, Any]:
    """Create a metrics update event"""
    return {
        "type": EventTypes.METRICS_UPDATE,
        "prediction_id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "energy": energy,
            "rmsd": rmsd,
            "aggressiveness": aggressiveness,
            "consistency": consistency,
            **kwargs
        }
    }


def create_agent_event(
    prediction_id: str,
    agent_id: int,
    status: str,
    **kwargs
) -> Dict[str, Any]:
    """Create an agent status update event"""
    return {
        "type": EventTypes.AGENT_UPDATE,
        "prediction_id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "agent_id": agent_id,
            "status": status,
            **kwargs
        }
    }


def create_log_event(
    prediction_id: str,
    level: str,
    message: str,
    **kwargs
) -> Dict[str, Any]:
    """Create an event log entry"""
    return {
        "type": EventTypes.EVENT_LOG,
        "prediction_id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "level": level,
            "message": message,
            **kwargs
        }
    }


def create_status_event(
    prediction_id: str,
    old_status: str,
    new_status: str
) -> Dict[str, Any]:
    """Create a status change event"""
    return {
        "type": EventTypes.STATUS_CHANGE,
        "prediction_id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "old_status": old_status,
            "new_status": new_status
        }
    }


def create_completion_event(
    prediction_id: str,
    final_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a completion event"""
    return {
        "type": EventTypes.PREDICTION_COMPLETE,
        "prediction_id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": final_metrics
    }


def create_error_event(
    prediction_id: str,
    error_message: str,
    error_type: str = "unknown"
) -> Dict[str, Any]:
    """Create an error event"""
    return {
        "type": EventTypes.PREDICTION_ERROR,
        "prediction_id": prediction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "error_message": error_message,
            "error_type": error_type
        }
    }
