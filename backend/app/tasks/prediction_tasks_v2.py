"""
Celery task for running predictions using PredictionRunner.

This module provides Celery tasks that use the unified PredictionRunner
from ubf_protein. This ensures the website uses the SAME prediction logic
as the CLI (test_protein.py).

The old prediction_tasks.py is kept for reference but should be deprecated.
"""
from celery import Task
import sys
import os
import math
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Add backend root to path for celery_app import
backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from celery_app import celery_app
from app.services.prediction_service import prediction_service
from app.services.file_storage_service import FileStorageService
from app.services.work_session_service import work_session_service
from app.models.prediction import PredictionStatus
from app.schemas.prediction import PredictionUpdateSchema

# Add project root to path for ubf_protein imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Import unified PredictionRunner
from ubf_protein.prediction_runner import (
    PredictionRunner, 
    PredictionConfig, 
    ProgressUpdate,
    PredictionResults
)

logger = logging.getLogger(__name__)


def sanitize_metrics(metrics: dict) -> dict:
    """Convert Infinity and NaN values to None for JSON serialization"""
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isinf(value) or math.isnan(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = sanitize_metrics(value)
        else:
            sanitized[key] = value
    return sanitized


class PredictionTaskV2(Task):
    """Custom task class for prediction execution using PredictionRunner"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        prediction_id = kwargs.get('prediction_id')
        if prediction_id:
            prediction_service.update_prediction(
                prediction_id,
                PredictionUpdateSchema(
                    status=PredictionStatus.FAILED,
                    error_message=str(exc)
                )
            )
            logger.error(f"Prediction {prediction_id} failed: {exc}")


def create_websocket_progress_callback(prediction_id: str, total_iterations: int):
    """
    Create a progress callback that emits WebSocket updates.
    
    Args:
        prediction_id: The prediction ID for WebSocket room
        total_iterations: Total iterations for progress calculation
    
    Returns:
        Callback function compatible with PredictionRunner
    """
    import httpx
    
    backend_url = os.getenv('BACKEND_URL', 'http://backend:8000')
    
    def progress_callback(update: ProgressUpdate):
        """Emit progress update via WebSocket."""
        try:
            # Update database
            prediction_service.update_prediction(
                prediction_id,
                PredictionUpdateSchema(
                    current_iteration=update.iteration,
                    progress_percentage=update.progress_percentage,
                    metrics=sanitize_metrics({
                        "current_energy": update.current_energy,
                        "current_rmsd": update.current_rmsd,
                        "folding_rmsd": update.folding_rmsd,
                        "best_energy": update.best_energy,
                        "best_rmsd": update.best_rmsd,
                        "conformations_explored": update.conformations_explored,
                    })
                )
            )
            
            # Emit WebSocket event
            progress_payload = {
                'prediction_id': prediction_id,
                'iteration': update.iteration,
                'total_iterations': update.total_iterations,
                'progress_percentage': update.progress_percentage,
                'current_energy': update.current_energy,
                'current_rmsd': update.current_rmsd,
                'folding_rmsd': update.folding_rmsd,
                'best_energy': update.best_energy,
                'best_rmsd': update.best_rmsd,
                'conformations_explored': update.conformations_explored,
                'aggressiveness': update.aggressiveness,
                'consistency': update.consistency,
                'stage': update.stage,
                'message': update.message,
            }
            
            with httpx.Client() as client:
                response = client.post(
                    f'{backend_url}/api/ws/emit/progress',
                    json={
                        'prediction_id': prediction_id,
                        'data': progress_payload
                    },
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    logger.debug(f"WebSocket progress emitted: {update.iteration}/{update.total_iterations}")
                else:
                    logger.warning(f"WebSocket emission failed: {response.status_code}")
                    
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")
    
    return progress_callback


@celery_app.task(bind=True, base=PredictionTaskV2, name='tasks.run_prediction_v2')
def run_prediction_v2(self, prediction_id: str):
    """
    Execute protein structure prediction using unified PredictionRunner.
    
    This task uses the SAME prediction logic as test_protein.py,
    ensuring consistent results between CLI and web interface.
    
    Args:
        prediction_id: The prediction ID to execute
    
    Returns:
        Dict with prediction results
    """
    logger.info(f"Starting prediction task V2 for {prediction_id}")
    
    # Get prediction from database
    prediction = prediction_service.get_prediction(prediction_id)
    if not prediction:
        logger.error(f"Prediction {prediction_id} not found")
        return {"error": "Prediction not found"}
    
    try:
        # Update status to running
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(status=PredictionStatus.RUNNING)
        )
        
        # Extract configuration
        config = prediction.configuration
        sequence = prediction.sequence
        
        # Set up output directories
        file_storage = FileStorageService()
        
        if prediction.session_id:
            # Session-based storage
            session = work_session_service.get_session_by_id(prediction.session_id)
            if not session:
                raise ValueError(f"Session {prediction.session_id} not found")
            
            user_id = session.user_id
            prediction_dir = file_storage.get_prediction_directory(
                user_id, prediction.session_id, prediction_id
            )
        else:
            # Legacy storage
            prediction_dir = Path("./prediction_results") / prediction_id
        
        prediction_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = prediction_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {prediction_dir}")
        
        # Create PredictionConfig from web request config
        pred_config = PredictionConfig(
            sequence=sequence,
            native_pdb=config.get('native_pdb'),
            agents=config.get('agents', None),  # None = auto-configure
            iterations=config.get('iterations', None),  # None = auto-configure
            diversity=config.get('diversity', 'balanced'),
            qcpp_config=config.get('qcpp_config', 'default'),
            enable_refinement=config.get('enable_refinement', False),
            enable_mediators=config.get('enable_mediators', False),
            mediator_count=config.get('mediator_count', 3),
            target_geometry=config.get('target_geometry', 'auto'),
            enable_checkpointing=config.get('enable_checkpointing', True),
            checkpoint_interval=config.get('checkpoint_interval', 50),
            checkpoint_dir=str(checkpoint_dir),
            output_dir=str(prediction_dir),
            save_pdb=True,
            save_trajectory=True,
        )
        
        logger.info(f"Running prediction: seq_len={len(sequence)}, "
                   f"agents={pred_config.agents}, iter={pred_config.iterations}")
        
        # Create progress callback for WebSocket updates
        progress_callback = create_websocket_progress_callback(
            prediction_id, 
            pred_config.iterations
        )
        
        # Run prediction using unified PredictionRunner
        runner = PredictionRunner(pred_config)
        results = runner.run(progress_callback=progress_callback)
        
        # Convert results to metrics dict for database
        final_metrics = sanitize_metrics({
            "best_energy": results.best_energy,
            "best_rmsd": results.best_rmsd,
            "folding_rmsd": results.folding_rmsd,
            "final_energy": results.best_energy,
            "final_rmsd": results.best_rmsd,
            "current_energy": results.best_energy,
            "current_rmsd": results.best_rmsd,
            "conformations_explored": results.conformations_explored,
            "energy_change": results.energy_change,
            "convergence_rate": results.convergence_rate,
            "initial_energy": results.initial_energy,
            "unique_structures": results.unique_structures,
            "gdt_ts_score": results.gdt_ts_score,
            "tm_score": results.tm_score,
            "validation_quality": results.validation_quality,
            "qcpp_total_analyses": results.qcpp_total_analyses,
            "qcpp_cache_hit_rate": results.qcpp_cache_hit_rate,
            "qaap_alignment": results.qaap_alignment,
            "resonance_40hz": results.resonance_40hz,
            "water_shielding": results.water_shielding,
            "qcp_score": results.qcp_score,
            "refinement_applied": results.refinement_applied,
            "refinement_improvement_percent": results.refinement_improvement_percent,
        })
        
        # Update prediction as completed
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(
                status=PredictionStatus.COMPLETED,
                progress_percentage=100.0,
                result_path=str(prediction_dir),
                checkpoint_path=str(checkpoint_dir) if checkpoint_dir.exists() else None,
                metrics=final_metrics
            )
        )
        
        # Emit completion WebSocket event
        try:
            import httpx
            backend_url = os.getenv('BACKEND_URL', 'http://backend:8000')
            
            with httpx.Client() as client:
                client.post(
                    f'{backend_url}/api/ws/emit/complete',
                    json={
                        'prediction_id': prediction_id,
                        'data': {
                            'status': 'completed',
                            'best_energy': results.best_energy,
                            'best_rmsd': results.best_rmsd,
                            'gdt_ts_score': results.gdt_ts_score,
                            'tm_score': results.tm_score,
                            'validation_quality': results.validation_quality,
                            'total_time_seconds': results.total_time_seconds,
                        }
                    },
                    timeout=5.0
                )
        except Exception as e:
            logger.warning(f"Failed to emit completion event: {e}")
        
        logger.info(f"Prediction {prediction_id} completed successfully")
        logger.info(f"Results: RMSD={results.best_rmsd}, Energy={results.best_energy}, "
                   f"GDT-TS={results.gdt_ts_score}, Quality={results.validation_quality}")
        
        return {
            "status": "completed",
            "prediction_id": prediction_id,
            "best_rmsd": results.best_rmsd,
            "best_energy": results.best_energy,
            "gdt_ts_score": results.gdt_ts_score,
            "tm_score": results.tm_score,
            "validation_quality": results.validation_quality,
        }
        
    except Exception as e:
        logger.error(f"Prediction {prediction_id} failed: {e}", exc_info=True)
        
        # Update status to failed
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(
                status=PredictionStatus.FAILED,
                error_message=str(e)
            )
        )
        
        # Emit error WebSocket event
        try:
            import httpx
            backend_url = os.getenv('BACKEND_URL', 'http://backend:8000')
            
            with httpx.Client() as client:
                client.post(
                    f'{backend_url}/api/ws/emit/error',
                    json={
                        'prediction_id': prediction_id,
                        'data': {
                            'error': str(e),
                            'status': 'failed'
                        }
                    },
                    timeout=5.0
                )
        except Exception as ws_err:
            logger.warning(f"Failed to emit error event: {ws_err}")
        
        raise


@celery_app.task(name='tasks.pause_prediction_v2')
def pause_prediction_v2(prediction_id: str):
    """Pause a running prediction."""
    logger.info(f"Pausing prediction {prediction_id}")
    
    prediction_service.update_prediction(
        prediction_id,
        PredictionUpdateSchema(status=PredictionStatus.PAUSED)
    )
    
    return {"status": "paused", "prediction_id": prediction_id}


@celery_app.task(name='tasks.stop_prediction_v2')
def stop_prediction_v2(prediction_id: str):
    """Stop a running prediction."""
    logger.info(f"Stopping prediction {prediction_id}")
    
    prediction_service.update_prediction(
        prediction_id,
        PredictionUpdateSchema(status=PredictionStatus.CANCELLED)
    )
    
    return {"status": "cancelled", "prediction_id": prediction_id}
