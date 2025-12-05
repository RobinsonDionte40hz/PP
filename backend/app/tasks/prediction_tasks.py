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
from app.services.file_storage_service import FileStorageService
from app.services.work_session_service import work_session_service
from app.models.prediction import PredictionStatus
from app.schemas.prediction import PredictionUpdateSchema
from app.websocket import socket_manager, create_progress_event, create_metrics_event, create_completion_event, create_error_event
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path for ubf_protein imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Import UBF system
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.adaptive_config import create_config_for_sequence
import math

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
        else:
            sanitized[key] = value
    return sanitized


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
        
        # Set up directories based on whether prediction is in a session
        file_storage = FileStorageService()
        
        if prediction.session_id:
            # Session-based storage: user_data/{user_id}/sessions/{session_id}/{prediction_id}/
            logger.info(f"Using session-based storage for prediction {prediction_id} in session {prediction.session_id}")
            
            # Get session to determine user_id
            session = work_session_service.get_session_by_id(prediction.session_id)
            if not session:
                raise ValueError(f"Session {prediction.session_id} not found for prediction {prediction_id}")
            
            user_id = session.user_id
            session_id = prediction.session_id
            
            # Get prediction directory from file storage service
            prediction_dir = file_storage.get_prediction_directory(user_id, session_id, prediction_id)
            prediction_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_dir = prediction_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            logger.info(f"Session-based path: {prediction_dir}")
        else:
            # Legacy storage for backward compatibility: ./prediction_results/{prediction_id}/
            logger.info(f"Using legacy storage for prediction {prediction_id} (no session)")
            results_dir = Path("./prediction_results")
            results_dir.mkdir(exist_ok=True)
            prediction_dir = results_dir / prediction_id
            prediction_dir.mkdir(exist_ok=True)
            checkpoint_dir = prediction_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            user_id = None  # No user association for legacy predictions
            session_id = None
            
            logger.info(f"Legacy path: {prediction_dir}")
        
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
        
        # Load native structure if PDB ID provided
        native_structure = None
        if native_pdb:
            try:
                from ubf_protein.rmsd_calculator import NativeStructureLoader
                native_loader = NativeStructureLoader()
                native_structure = native_loader.load_pdb_structure(native_pdb, sequence)
                logger.info(f"Loaded native structure from PDB ID: {native_pdb}")
            except Exception as e:
                logger.warning(f"Failed to load native structure {native_pdb}: {e}")
        
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
        
        # Initialize agents with diversity profile (pass loaded structure object, not PDB ID string)
        coordinator.initialize_agents(count=agents, diversity_profile=diversity, native_structure=native_structure)
        
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
            folding_rmsd = getattr(results, 'folding_rmsd', None)
            metrics = sanitize_metrics({
                "current_energy": best_energy,
                "current_rmsd": best_rmsd if best_rmsd is not None else None,
                "folding_rmsd": folding_rmsd,  # Always available - RMSD from initial state
                "conformations_explored": results.total_conformations_explored,
                "best_energy": results.best_energy,
                "best_rmsd": results.best_rmsd
            })
            
            # Update prediction
            prediction_service.update_prediction(
                prediction_id,
                PredictionUpdateSchema(
                    current_iteration=completed_iterations,
                    progress_percentage=progress,
                    metrics=metrics
                )
            )
            
            # Emit WebSocket progress update via HTTP (so it goes through actual Socket.IO server)
            try:
                import httpx
                
                # Get average aggressiveness and consistency from all agents
                agent_list = coordinator.get_agents()
                avg_aggressiveness = sum(agent.get_consciousness_state().get_frequency() for agent in agent_list) / len(agent_list) if agent_list else None
                avg_consistency = sum(agent.get_consciousness_state().get_coherence() for agent in agent_list) / len(agent_list) if agent_list else None
                
                progress_payload = {
                    'prediction_id': prediction_id,
                    'iteration': completed_iterations,
                    'total_iterations': iterations,
                    'progress_percentage': progress,
                    'current_energy': float(best_energy) if best_energy is not None else None,
                    'current_rmsd': float(best_rmsd) if best_rmsd is not None and best_rmsd != float('inf') else None,
                    'folding_rmsd': float(folding_rmsd) if folding_rmsd is not None else None,
                    'conformations_explored': results.total_conformations_explored,
                    'best_energy': float(results.best_energy) if results.best_energy is not None else None,
                    'best_rmsd': float(results.best_rmsd) if results.best_rmsd is not None and results.best_rmsd != float('inf') else None,
                    'aggressiveness': float(avg_aggressiveness) if avg_aggressiveness is not None else None,
                    'consistency': float(avg_consistency) if avg_consistency is not None else None,
                }
                
                logger.info(f"Attempting WebSocket emission for iteration {completed_iterations}/{iterations}...")
                
                # Call backend WebSocket emission endpoint using httpx (synchronous client)
                # Use 'backend' service name when running in Docker, 'localhost' for development
                backend_url = os.getenv('BACKEND_URL', 'http://backend:8000')
                with httpx.Client() as client:
                    response = client.post(
                        f'{backend_url}/api/ws/emit/progress',
                        json={
                            'prediction_id': prediction_id,
                            'data': progress_payload
                        },
                        timeout=5.0  # Increased timeout for localhost
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✓ WebSocket progress emitted: iteration {completed_iterations}/{iterations}, subscribers={result.get('subscribers', 0)}, energy={best_energy:.2f}")
                        
                        # Emit log event for significant energy improvements
                        if chunk_idx > 0 and results.best_energy is not None:
                            energy_improvement = (results.best_energy - initial_energy) if chunk_idx == 1 else None
                            if energy_improvement and energy_improvement < -10:  # Significant improvement
                                try:
                                    from datetime import datetime
                                    log_response = client.post(
                                        f'{backend_url}/api/ws/emit/log',
                                        json={
                                            'prediction_id': prediction_id,
                                            'data': {
                                                'level': 'success',
                                                'message': f'Energy improved by {abs(energy_improvement):.2f} kcal/mol (now {results.best_energy:.2f})',
                                                'timestamp': datetime.utcnow().isoformat()
                                            }
                                        },
                                        timeout=5.0
                                    )
                                except Exception as log_err:
                                    logger.debug(f"Failed to emit log event: {log_err}")
                    else:
                        logger.error(f"❌ Failed to emit WebSocket progress: {response.status_code} - {response.text}")
                    
            except httpx.TimeoutException as e:
                logger.error(f"❌ WebSocket emission timeout for iteration {completed_iterations}: {e}")
            except httpx.ConnectError as e:
                logger.error(f"❌ Cannot connect to backend for WebSocket emission: {e}")
            except Exception as e:
                logger.error(f"❌ Failed to emit WebSocket update for iteration {completed_iterations}: {e}", exc_info=True)
            
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
        
        # Calculate additional metrics from agent results
        initial_energy = None
        energy_change = None
        convergence_rate = None
        final_aggressiveness = None
        final_consistency = None
        unique_structures = None
        
        if final_results.agent_metrics and len(final_results.agent_metrics) > 0:
            # Get best and worst energies to calculate energy change
            all_best_energies = [m.best_energy_found for m in final_results.agent_metrics]
            if all_best_energies:
                initial_energy = max(all_best_energies)  # Worst (highest) energy as "initial"
                energy_change = refined_energy - initial_energy
                if initial_energy != 0:
                    convergence_rate = abs((initial_energy - refined_energy) / initial_energy * 100)
            
            # Count unique structures explored (approximation)
            unique_structures = sum(m.conformations_explored for m in final_results.agent_metrics)
        
        # Note: Aggressiveness and consistency are stored in the agent's consciousness state,
        # not in ExplorationMetrics. We don't have access to them here without storing them separately.
        # Setting to None for now - can be added to ExplorationMetrics in future.
        
        # Get quantum metrics if QCPP integration is enabled
        qaap_alignment = None
        resonance_40hz = None
        water_shielding = None
        qcp_score = None
        
        if qcpp_integration:
            try:
                # Get latest QCPP analysis for best conformation
                qcpp_result = qcpp_integration.analyze_conformation(refined_conf)
                if qcpp_result:
                    # Extract quantum metrics (these would come from QCPP analysis)
                    qaap_alignment = qcpp_result.get('qaap_score', None)  # QAAP alignment score
                    resonance_40hz = qcpp_result.get('resonance_score', None)  # 40 Hz resonance
                    water_shielding = qcpp_result.get('water_shielding', None)  # Water shielding time
                    qcp_score = qcpp_result.get('qcp_value', None)  # QCP score
            except Exception as e:
                logger.warning(f"Could not extract quantum metrics: {e}")
        
        # Save results with comprehensive metrics
        result_file = prediction_dir / "results.json"
        results_data = {
            "prediction_id": prediction_id,
            "sequence": sequence,
            "iterations": iterations,
            "agent_count": agents,
            "diversity": diversity,
            "best_energy": refined_energy,
            "best_rmsd": refined_rmsd,
            "final_energy": refined_energy,  # Alias for compatibility
            "final_rmsd": refined_rmsd,  # Alias for compatibility
            "conformations_explored": final_results.total_conformations_explored,
            "runtime_seconds": final_results.total_runtime_seconds,
            "refinement_applied": refinement_applied,
            "original_metrics": {
                "energy": best_energy,
                "rmsd": best_rmsd
            } if refinement_applied else None,
            "energy_change": energy_change,
            "convergence_rate": convergence_rate,
            "initial_energy": initial_energy,
            "final_aggressiveness": final_aggressiveness,
            "final_consistency": final_consistency,
            "unique_structures": unique_structures,
            "gdt_ts_score": final_results.best_gdt_ts,
            "tm_score": final_results.best_tm_score,
            "validation_quality": final_results.validation_quality,
            "qaap_alignment": qaap_alignment,
            "resonance_40hz": resonance_40hz,
            "water_shielding": water_shielding,
            "qcp_score": qcp_score,
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
        
        # Mark as completed with comprehensive metrics
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(
                status=PredictionStatus.COMPLETED,
                progress_percentage=100.0,
                result_path=str(prediction_dir),
                checkpoint_path=str(checkpoint_dir) if checkpoint_dir.exists() else None,
                metrics=sanitize_metrics({
                    "best_energy": refined_energy,
                    "best_rmsd": refined_rmsd,
                    "folding_rmsd": getattr(final_results, 'folding_rmsd', None),  # Always available
                    "final_energy": refined_energy,
                    "final_rmsd": refined_rmsd,
                    "current_energy": refined_energy,
                    "current_rmsd": refined_rmsd,
                    "conformations_explored": final_results.total_conformations_explored,
                    "energy_change": energy_change,
                    "convergence_rate": convergence_rate,
                    "initial_energy": initial_energy,
                    "final_aggressiveness": final_aggressiveness,
                    "final_consistency": final_consistency,
                    "unique_structures": unique_structures,
                    "gdt_ts_score": final_results.best_gdt_ts,
                    "tm_score": final_results.best_tm_score,
                    "validation_quality": final_results.validation_quality,
                    "qaap_alignment": qaap_alignment,
                    "resonance_40hz": resonance_40hz,
                    "water_shielding": water_shielding,
                    "qcp_score": qcp_score,
                    "refinement_applied": refinement_applied
                })
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
        
        # Update session last_active_at if this prediction is part of a session
        if prediction.session_id:
            try:
                work_session_service.update_session_activity(prediction.session_id)
                logger.info(f"Updated last_active_at for session {prediction.session_id}")
            except Exception as e:
                logger.warning(f"Failed to update session activity: {e}")
        
        logger.info(f"Prediction {prediction_id} completed successfully")
        
        return {
            "prediction_id": prediction_id,
            "status": "completed",
            "iterations": iterations,
            "best_energy": final_results.best_energy,
            "best_rmsd": final_results.best_rmsd,
            "folding_rmsd": getattr(final_results, 'folding_rmsd', None),
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
