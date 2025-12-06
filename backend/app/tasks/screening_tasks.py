"""
Celery tasks for aggregation screening.

These tasks handle batch screening and screening campaigns asynchronously.
"""
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add backend root to path
backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from celery_app import celery_app

# Add project root for ubf_protein imports
if os.path.exists('/ubf_protein'):
    # Docker environment
    if '/' not in sys.path:
        sys.path.insert(0, '/')
else:
    # Local development
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


def get_screener(mode: str):
    """Get aggregation screener with appropriate config."""
    from ubf_protein.aggregation_screening import AggregationScreener, ScreeningConfig
    
    config_map = {
        'fast': ScreeningConfig.fast(),
        'balanced': ScreeningConfig.balanced(),
        'thorough': ScreeningConfig.thorough(),
    }
    
    config = config_map.get(mode, ScreeningConfig.balanced())
    return AggregationScreener(config)


def result_to_dict(result) -> Dict[str, Any]:
    """Convert AggregationMetrics to dict for JSON serialization."""
    return {
        'sequence': result.sequence,
        'sequence_length': result.sequence_length,
        'aggregation_score': result.aggregation_score,
        'energy_score': result.energy_score,
        'structure_score': result.structure_score,
        'hydrophobic_score': result.hydrophobic_score,
        'compactness_score': result.compactness_score,
        'risk_level': result.risk_level.value,
        'risk_factors': result.risk_factors,
        'passes_screening': result.passes_screening,
        'final_energy': result.final_energy,
        'secondary_structure_pct': result.secondary_structure_pct,
        'radius_of_gyration': result.radius_of_gyration,
        'screening_time_ms': result.screening_time_ms,
    }


@celery_app.task(bind=True, name='tasks.run_batch_screening')
def run_batch_screening(
    self,
    batch_id: str,
    sequences: List[str],
    mode: str = 'balanced',
    user_id: Optional[str] = None,
):
    """
    Run batch screening for multiple sequences.
    
    Args:
        batch_id: Unique batch identifier
        sequences: List of protein sequences
        mode: Screening mode ('fast', 'balanced', 'thorough')
        user_id: User who submitted the batch
    """
    logger.info(f"Starting batch screening {batch_id} with {len(sequences)} sequences")
    
    try:
        screener = get_screener(mode)
        results = []
        
        def progress_callback(current: int, total: int, result):
            """Update progress via WebSocket."""
            try:
                # Emit progress update
                from app.api.screening import _screening_batches
                if batch_id in _screening_batches:
                    batch = _screening_batches[batch_id]
                    batch.results.append(result_to_dict(result))
                    
                    # Update counts
                    if result.passes_screening:
                        _screening_batches[batch_id].sequences_passed += 1
                    else:
                        _screening_batches[batch_id].sequences_failed += 1
                    
                    # Update risk distribution
                    risk = result.risk_level.value
                    _screening_batches[batch_id].risk_summary[risk] = \
                        _screening_batches[batch_id].risk_summary.get(risk, 0) + 1
                    
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Run screening
        for i, seq in enumerate(sequences):
            result = screener.screen_sequence(seq)
            results.append(result)
            progress_callback(i + 1, len(sequences), result)
        
        # Sort by score (best first)
        results.sort(key=lambda x: -x.aggregation_score)
        
        # Final update
        from app.api.screening import _screening_batches
        if batch_id in _screening_batches:
            _screening_batches[batch_id].status = "completed"
            _screening_batches[batch_id].completed_at = datetime.utcnow()
            _screening_batches[batch_id].results = [result_to_dict(r) for r in results]
        
        logger.info(f"Batch screening {batch_id} completed: {len(results)} sequences screened")
        
        return {
            'batch_id': batch_id,
            'status': 'completed',
            'total_screened': len(results),
            'passed': sum(1 for r in results if r.passes_screening),
        }
        
    except Exception as e:
        logger.error(f"Batch screening {batch_id} failed: {e}", exc_info=True)
        
        from app.api.screening import _screening_batches
        if batch_id in _screening_batches:
            _screening_batches[batch_id].status = "failed"
        
        raise


@celery_app.task(bind=True, name='tasks.run_screening_campaign')
def run_screening_campaign(
    self,
    campaign_id: str,
    sequences: List[str],
    mode: str = 'balanced',
    min_score: float = 0.5,
    auto_predict: bool = False,
    user_id: Optional[str] = None,
):
    """
    Run a screening campaign with optional auto-prediction.
    
    Args:
        campaign_id: Unique campaign identifier
        sequences: List of protein sequences to screen
        mode: Screening mode
        min_score: Minimum aggregation score to pass
        auto_predict: Whether to auto-create predictions for passed sequences
        user_id: User who created the campaign
    """
    logger.info(f"Starting screening campaign {campaign_id} with {len(sequences)} sequences")
    
    try:
        from app.api.screening import _screening_campaigns
        
        screener = get_screener(mode)
        results = []
        passed_sequences = []
        
        for i, seq in enumerate(sequences):
            result = screener.screen_sequence(seq)
            result_dict = result_to_dict(result)
            results.append(result_dict)
            
            # Update campaign progress
            if campaign_id in _screening_campaigns:
                campaign = _screening_campaigns[campaign_id]
                campaign["screened_sequences"] = i + 1
                
                risk = result.risk_level.value
                campaign["risk_distribution"][risk] = campaign["risk_distribution"].get(risk, 0) + 1
                
                if result.passes_screening and result.aggregation_score >= min_score:
                    campaign["passed_count"] += 1
                    passed_sequences.append(seq)
                else:
                    campaign["failed_count"] += 1
        
        # Sort results
        results.sort(key=lambda x: -x["aggregation_score"])
        
        # Auto-create predictions if requested
        prediction_ids = []
        if auto_predict and passed_sequences:
            logger.info(f"Auto-creating predictions for {len(passed_sequences)} passed sequences")
            
            try:
                from app.services.prediction_service import prediction_service
                from app.schemas.prediction import PredictionCreateSchema, PredictionConfigurationSchema
                
                # Create predictions with fast settings (screening mode)
                config = PredictionConfigurationSchema(
                    iterations=500,  # Reduced for screening workflow
                    agents=5,
                    diversity="balanced",
                    enable_checkpointing=True,
                )
                
                for seq in passed_sequences[:50]:  # Limit to 50 predictions
                    try:
                        data = PredictionCreateSchema(sequence=seq, configuration=config)
                        prediction = prediction_service.create_prediction(data, user_id=user_id)
                        prediction_ids.append(prediction.id)
                        
                        # Queue prediction task
                        from app.tasks import run_prediction_v2
                        run_prediction_v2.delay(prediction.id)
                        
                    except Exception as pred_error:
                        logger.warning(f"Failed to create prediction for sequence: {pred_error}")
                
                logger.info(f"Created {len(prediction_ids)} predictions")
                
            except Exception as e:
                logger.error(f"Failed to auto-create predictions: {e}")
        
        # Update final campaign state
        if campaign_id in _screening_campaigns:
            campaign = _screening_campaigns[campaign_id]
            campaign["status"] = "completed"
            campaign["completed_at"] = datetime.utcnow()
            campaign["results"] = results
            campaign["prediction_ids"] = prediction_ids
        
        logger.info(f"Screening campaign {campaign_id} completed")
        
        return {
            'campaign_id': campaign_id,
            'status': 'completed',
            'total_screened': len(results),
            'passed': len(passed_sequences),
            'predictions_created': len(prediction_ids),
        }
        
    except Exception as e:
        logger.error(f"Screening campaign {campaign_id} failed: {e}", exc_info=True)
        
        from app.api.screening import _screening_campaigns
        if campaign_id in _screening_campaigns:
            _screening_campaigns[campaign_id]["status"] = "failed"
        
        raise
