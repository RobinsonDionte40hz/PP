"""
Celery task for running predictions using PredictionRunner.

This module provides Celery tasks that use the unified PredictionRunner
from ubf_protein.api. This ensures the website uses the SAME prediction logic
as the CLI (test_protein.py).

ARCHITECTURE NOTE:
- This module imports ONLY from ubf_protein.api (public interface)
- The sys.path manipulation below is temporary until ubf_protein is installed
  as a proper Python package (see docs/ARCHITECTURE_REFACTORING_PLAN.md Phase 2)
- Once installed as a package, the sys.path code can be removed

The old prediction_tasks.py is deprecated - do not use it.
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

# TEMPORARY: Add paths for ubf_protein imports
# TODO: Remove this once ubf_protein is installed as a package (Phase 2)
# In Docker: ubf_protein is at /packages/ubf_protein/ with PYTHONPATH=/packages
# Locally: ubf_protein is at ../../.. relative to this file
if os.path.exists('/packages/ubf_protein'):
    # Docker environment - new structure
    if '/packages' not in sys.path:
        sys.path.insert(0, '/packages')
elif os.path.exists('/ubf_protein'):
    # Docker environment - legacy structure (for backward compat)
    if '/' not in sys.path:
        sys.path.insert(0, '/')
else:
    # Local development
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import from public API (SOLID: Dependency Inversion Principle)
# External code should ONLY import from ubf_protein.api, not internal modules
from ubf_protein.api import (
    PredictionRunner, 
    PredictionConfig, 
    PredictionResults,
    ProgressUpdate,
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
            # Build metrics dict including secondary structure if available
            metrics_update = {
                "current_energy": update.current_energy,
                "current_rmsd": update.current_rmsd,
                "folding_rmsd": update.folding_rmsd,
                "best_energy": update.best_energy,
                "best_rmsd": update.best_rmsd,
                "conformations_explored": update.conformations_explored,
            }
            
            # Include secondary structure if present
            if update.secondary_structure:
                metrics_update["secondary_structure"] = update.secondary_structure
            
            # Update database with current progress AND total iterations
            # This ensures the frontend always knows the actual total iterations
            prediction_service.update_prediction(
                prediction_id,
                PredictionUpdateSchema(
                    current_iteration=update.iteration,
                    total_iterations=update.total_iterations,  # Update total too
                    progress_percentage=update.progress_percentage,
                    metrics=sanitize_metrics(metrics_update)
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
            
            # Include secondary structure in WebSocket payload if available
            if update.secondary_structure:
                progress_payload['secondary_structure'] = update.secondary_structure
            
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
        # Only pass parameters that PredictionConfig actually supports
        pred_config = PredictionConfig(
            sequence=sequence,
            native_pdb=config.get('native_pdb'),
            agents=config.get('agents', 10),
            iterations=config.get('iterations', 500),
            qcpp_config=config.get('qcpp_config', 'default'),
            enable_refinement=config.get('enable_refinement', False),
            enable_mediators=config.get('enable_mediators', False),
            output_dir=str(prediction_dir),
            checkpoint_interval=config.get('checkpoint_interval', 50),
        )
        
        # Store extra config for metrics tracking
        enable_hierarchical_folding = config.get('enable_hierarchical_folding', False)
        
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
            "hierarchical_folding_enabled": enable_hierarchical_folding,
            "hierarchical_folding_stats": results.hierarchical_folding_stats,
        })
        
        # Run aggregation screening if enabled
        if config.get('enable_screening', False):
            try:
                # Import from public API (SOLID: Dependency Inversion)
                from ubf_protein.api import AggregationScreener, ScreeningConfig
                
                screening_mode = config.get('screening_mode', 'balanced')
                screening_config_map = {
                    'fast': ScreeningConfig(window_size=5, threshold=0.6),
                    'balanced': ScreeningConfig(window_size=7, threshold=0.5),
                    'thorough': ScreeningConfig(window_size=9, threshold=0.4),
                }
                screening_config = screening_config_map.get(screening_mode, screening_config_map['balanced'])
                
                screener = AggregationScreener()
                screening_result = screener.screen(sequence, screening_config)
                
                # Add screening results to metrics
                final_metrics["screening"] = {
                    "aggregation_score": screening_result.aggregation_score,
                    "risk_level": screening_result.risk_level.value,
                    "risk_factors": screening_result.risk_factors,
                    "passes_screening": screening_result.passes_screening,
                    "energy_score": screening_result.energy_score,
                    "structure_score": screening_result.structure_score,
                    "hydrophobic_score": screening_result.hydrophobic_score,
                    "compactness_score": screening_result.compactness_score,
                    "secondary_structure_pct": screening_result.secondary_structure_pct,
                    "radius_of_gyration": screening_result.radius_of_gyration,
                }
                logger.info(f"Screening completed: risk={screening_result.risk_level.value}, "
                           f"score={screening_result.aggregation_score:.3f}")
            except Exception as screening_err:
                logger.warning(f"Screening failed (non-critical): {screening_err}")
                final_metrics["screening"] = {"error": str(screening_err)}
        
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
                    f'{backend_url}/api/ws/emit/completion',
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
