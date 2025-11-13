"""
Celery task for running predictions
"""
from celery import Task
import sys
import os

# Add backend root to path for celery_app import
backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from celery_app import celery_app
from app.services.prediction_service import prediction_service
from app.models.prediction import PredictionStatus
from app.schemas.prediction import PredictionUpdateSchema
from app.websocket import socket_manager, create_progress_event, create_metrics_event, create_completion_event, create_error_event
import logging
import sys
import os
import json
from pathlib import Path

# Add project root to path for ubf_protein imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Import UBF system
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.adaptive_config import create_config_for_sequence

logger = logging.getLogger(__name__)


class PredictionTask(Task):
    """Custom task class for prediction execution"""
    
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


@celery_app.task(bind=True, base=PredictionTask, name='tasks.run_prediction')
def run_prediction(self, prediction_id: str):
    """
    Execute a protein structure prediction using UBF multi-agent system.
    
    This task runs the UBF protein prediction system and emits
    real-time updates via WebSocket.
    """
    logger.info(f"Starting prediction task for {prediction_id}")
    
    # Get prediction
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
        
        # Get configuration
        config = prediction.configuration
        sequence = prediction.sequence
        iterations = config.get('iterations', 1000)
        agents = config.get('agents', 10)
        diversity = config.get('diversity', 'balanced')
        native_pdb = config.get('native_pdb')
        enable_checkpointing = config.get('enable_checkpointing', True)
        checkpoint_interval = config.get('checkpoint_interval', 50)
        qcpp_config = config.get('qcpp_config', 'default')
        
        # Advanced module options
        enable_mediators = config.get('enable_mediators', False)
        mediator_count = config.get('mediator_count', 3)
        enable_refinement = config.get('enable_refinement', False)
        target_geometry = 'auto'  # Let QCPP analysis determine optimal geometry
        
        logger.info(f"Running prediction: seq_len={len(sequence)}, iter={iterations}, agents={agents}, diversity={diversity}")
        
        # Set up directories
        results_dir = Path("./prediction_results")
        results_dir.mkdir(exist_ok=True)
        prediction_dir = results_dir / prediction_id
        prediction_dir.mkdir(exist_ok=True)
        checkpoint_dir = prediction_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Get adaptive config based on sequence length
        adaptive_config = create_config_for_sequence(sequence)
        
        # Initialize QCPP integration if requested
        qcpp_integration = None
        if qcpp_config and qcpp_config != 'none':
            try:
                from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
                from ubf_protein.qcpp_config import (
                    get_default_config,
                    get_high_performance_config,
                    get_high_accuracy_config
                )
                
                config_map = {
                    'default': get_default_config,
                    'high_performance': get_high_performance_config,
                    'high_accuracy': get_high_accuracy_config
                }
                
                if qcpp_config in config_map:
                    qcpp_settings = config_map[qcpp_config]()
                    qcpp_integration = QCPPIntegrationAdapter(
                        protein_sequence=sequence,
                        config=qcpp_settings
                    )
                    logger.info(f"Initialized QCPP integration with {qcpp_config} config")
            except Exception as e:
                logger.warning(f"Failed to initialize QCPP integration: {e}")
        
        # Initialize multi-agent coordinator
        coordinator = MultiAgentCoordinator(
            protein_sequence=sequence,
            adaptive_config=adaptive_config,
            enable_checkpointing=enable_checkpointing,
            checkpoint_dir=str(checkpoint_dir),
            qcpp_integration=qcpp_integration,
            enable_mediators=enable_mediators,
            mediator_count=mediator_count,
            target_geometry=target_geometry
        )
        
        # Initialize agents with diversity profile
        coordinator.initialize_agents(count=agents, diversity_profile=diversity, native_structure=native_pdb)
        
        # Initialize mediators if enabled
        if enable_mediators:
            try:
                coordinator.initialize_mediators()
                logger.info(f"Initialized {mediator_count} mediator agents for pattern detection")
            except Exception as e:
                logger.warning(f"Failed to initialize mediator agents: {e}")
        
        logger.info(f"Initialized {agents} agents with {diversity} diversity profile")
        
        # Run exploration with progress tracking
        # We'll run in chunks to emit progress updates
        chunk_size = 50
        total_chunks = (iterations + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            chunk_iterations = min(chunk_size, iterations - (chunk_idx * chunk_size))
            
            # Run chunk
            results = coordinator.run_parallel_exploration(chunk_iterations)
            
            # Calculate progress
            completed_iterations = (chunk_idx + 1) * chunk_size
            if completed_iterations > iterations:
                completed_iterations = iterations
            progress = (completed_iterations / iterations) * 100
            
            # Get best metrics - MultiAgentCoordinator returns (conformation, energy, rmsd) tuple
            best_conf, best_energy, best_rmsd = coordinator.get_best_conformation()
            metrics = {
                "current_energy": best_energy,
                "current_rmsd": best_rmsd if best_rmsd is not None else None,
                "conformations_explored": results.total_conformations_explored,
                "best_energy": results.best_energy,
                "best_rmsd": results.best_rmsd
            }
            
            # Update prediction
            prediction_service.update_prediction(
                prediction_id,
                PredictionUpdateSchema(
                    current_iteration=completed_iterations,
                    progress_percentage=progress,
                    metrics=metrics
                )
            )
            
            # Emit WebSocket progress update
            try:
                import asyncio
                from app.websocket import socket_manager
                
                # Create event loop if needed (Celery worker context)
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Emit progress update
                loop.run_until_complete(
                    socket_manager.emit_progress_update(
                        prediction_id,
                        {
                            'prediction_id': prediction_id,
                            'iteration': completed_iterations,
                            'total_iterations': iterations,
                            'progress_percentage': progress,
                            'current_energy': best_energy,
                            'current_rmsd': best_rmsd,
                            'conformations_explored': results.total_conformations_explored
                        }
                    )
                )
                logger.info(f"✓ WebSocket progress update emitted for iteration {completed_iterations}/{iterations}")
            except Exception as e:
                logger.warning(f"Failed to emit WebSocket update: {e}", exc_info=True)
            
            logger.debug(f"Progress: {completed_iterations}/{iterations} ({progress:.1f}%)")
        
        # Get final results - 'results' variable already contains ExplorationResults from last chunk
        final_results = results
        best_conf, best_energy, best_rmsd = coordinator.get_best_conformation()
        
        # Apply quantum refinement if enabled
        refined_conf = best_conf
        refined_energy = best_energy
        refined_rmsd = best_rmsd
        refinement_applied = False
        
        if enable_refinement and native_pdb:
            try:
                from ubf_protein.quantum_refinement_engine import QuantumRefinementEngine
                from ubf_protein.rmsd_calculator import NativeStructureLoader
                
                logger.info(f"Applying quantum refinement with target geometry: {target_geometry}")
                
                # Load native structure
                native_loader = NativeStructureLoader()
                native_coords = native_loader.load_pdb_structure(native_pdb, sequence)
                
                # Initialize refinement engine
                refinement_engine = QuantumRefinementEngine(
                    protein_sequence=sequence,
                    qcpp_integration=qcpp_integration,
                    target_geometry=target_geometry
                )
                
                # Run two-stage refinement
                refined_conf = refinement_engine.refine_conformation(
                    best_conf,
                    native_coords,
                    max_iterations_stage1=100,
                    max_iterations_stage2=50
                )
                
                # Recalculate metrics
                from ubf_protein.energy_function import EnergyFunction
                from ubf_protein.rmsd_calculator import RMSDCalculator
                
                energy_fn = EnergyFunction(sequence)
                refined_energy = energy_fn.calculate_total_energy(refined_conf)
                
                rmsd_calc = RMSDCalculator()
                refined_rmsd = rmsd_calc.calculate_rmsd(refined_conf.atom_coordinates, native_coords)
                
                refinement_applied = True
                logger.info(f"Quantum refinement completed: RMSD {best_rmsd:.2f}Å -> {refined_rmsd:.2f}Å ({((best_rmsd - refined_rmsd) / best_rmsd * 100):.1f}% improvement)")
                
            except Exception as e:
                logger.warning(f"Quantum refinement failed: {e}. Using unrefined results.")
                refined_conf = best_conf
                refined_energy = best_energy
                refined_rmsd = best_rmsd
        
        # Save results
        result_file = prediction_dir / "results.json"
        results_data = {
            "prediction_id": prediction_id,
            "sequence": sequence,
            "iterations": iterations,
            "agents": agents,
            "diversity": diversity,
            "best_energy": refined_energy,
            "best_rmsd": refined_rmsd,
            "conformations_explored": final_results.total_conformations_explored,
            "runtime_seconds": final_results.total_runtime_seconds,
            "refinement_applied": refinement_applied,
            "original_metrics": {
                "energy": best_energy,
                "rmsd": best_rmsd
            } if refinement_applied else None,
            "best_conformation": {
                "energy": refined_energy,
                "rmsd_to_native": refined_rmsd,
                "coordinates": list(refined_conf.atom_coordinates)
            }
        }
        
        with open(result_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved results to {result_file}")
        
        # Save PDB structure if possible
        try:
            from ubf_protein.visualization import export_to_pdb
            pdb_file = prediction_dir / "structure.pdb"
            export_to_pdb(refined_conf, str(pdb_file), sequence)
            logger.info(f"Saved PDB structure to {pdb_file}")
        except Exception as e:
            logger.warning(f"Could not save PDB structure: {e}")
        
        # Mark as completed
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(
                status=PredictionStatus.COMPLETED,
                progress_percentage=100.0,
                result_path=str(prediction_dir),
                checkpoint_path=str(checkpoint_dir) if checkpoint_dir.exists() else None,
                metrics=results_data
            )
        )
        
        # Emit completion WebSocket event
        try:
            import asyncio
            from app.websocket import socket_manager
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.run_until_complete(
                socket_manager.emit_completion(
                    prediction_id,
                    {
                        'prediction_id': prediction_id,
                        'status': 'completed',
                        'best_energy': refined_energy,
                        'best_rmsd': refined_rmsd,
                        'refinement_applied': refinement_applied,
                        'result_path': str(prediction_dir)
                    }
                )
            )
        except Exception as e:
            logger.warning(f"Failed to emit completion WebSocket event: {e}")
        
        logger.info(f"Prediction {prediction_id} completed successfully")
        
        return {
            "prediction_id": prediction_id,
            "status": "completed",
            "iterations": iterations,
            "best_energy": final_results.best_energy,
            "best_rmsd": final_results.best_rmsd,
            "result_path": str(result_file)
        }
    
    except Exception as e:
        logger.error(f"Error in prediction task: {e}", exc_info=True)
        
        # Update prediction with error
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(
                status=PredictionStatus.FAILED,
                error_message=str(e)
            )
        )
        
        # Emit error WebSocket event
        try:
            import asyncio
            from app.websocket import socket_manager
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.run_until_complete(
                socket_manager.emit_error(
                    prediction_id,
                    {
                        'prediction_id': prediction_id,
                        'error': str(e),
                        'status': 'failed'
                    }
                )
            )
        except Exception as ws_error:
            logger.warning(f"Failed to emit error WebSocket event: {ws_error}")
        
        raise


@celery_app.task(name='tasks.run_campaign')
def run_campaign(campaign_id: str):
    """
    Execute a multi-protein campaign.
    
    Runs predictions for multiple proteins in phases with quality gates.
    """
    logger.info(f"Starting campaign task for {campaign_id}")
    
    # TODO: Implement campaign execution
    return {
        "campaign_id": campaign_id,
        "status": "not_implemented",
        "message": "Campaign execution not implemented yet"
    }


@celery_app.task(name='tasks.pause_prediction')
def pause_prediction(prediction_id: str):
    """Signal to pause a running prediction"""
    logger.info(f"Pausing prediction {prediction_id}")
    # TODO: Implement pause mechanism
    return {"status": "paused"}


@celery_app.task(name='tasks.stop_prediction')
def stop_prediction(prediction_id: str):
    """Signal to stop a running prediction"""
    logger.info(f"Stopping prediction {prediction_id}")
    # TODO: Implement stop mechanism
    return {"status": "stopped"}
