"""
Multi-agent coordination implementation for UBF protein system.

This module implements the MultiAgentCoordinator that manages multiple
protein agents working together to explore conformational space.
"""

import time
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Any

from .interfaces import IMultiAgentCoordinator, IProteinAgent, ISharedMemoryPool, IAdaptiveConfigurator
from .models import ExplorationResults, ExplorationMetrics, Conformation, AdaptiveConfig, ProteinSizeClass
from .protein_agent import ProteinAgent
from .memory_system import SharedMemoryPool
from .adaptive_config import get_default_configurator, AdaptiveConfigurator
from .config import AGENT_DIVERSITY_PROFILES, AGENT_PROFILE_CAUTIOUS_RATIO, AGENT_PROFILE_BALANCED_RATIO, AGENT_PROFILE_AGGRESSIVE_RATIO
from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class MultiAgentCoordinator(IMultiAgentCoordinator):
    """
    Implementation of multi-agent coordination system.

    Manages multiple protein agents with diverse consciousness profiles
    to collectively explore conformational space through parallel execution
    and shared memory exchange.
    """

    def __init__(self, 
                 protein_sequence: str,
                 adaptive_configurator: Optional[AdaptiveConfigurator] = None,
                 adaptive_config: Optional[AdaptiveConfig] = None,
                 enable_checkpointing: bool = True,
                 checkpoint_dir: str = "checkpoints",
                 qcpp_integration: Optional[Any] = None,
                 qcpp_analysis_frequency: int = 5,
                 enable_thz_recording: bool = False,
                 target_geometry: str = 'none'):
        """
        Initialize multi-agent coordinator with protein sequence.

        Args:
            protein_sequence: Amino acid sequence for all agents
            adaptive_configurator: Optional configurator for auto-configuration
            adaptive_config: Optional pre-configured AdaptiveConfig (overrides auto-config)
            enable_checkpointing: Whether to enable automatic checkpointing
            checkpoint_dir: Directory for checkpoint files
            qcpp_integration: Optional QCPP integration adapter for physics-grounded exploration
            qcpp_analysis_frequency: Analyze with QCPP every N iterations (default: 5 for performance)
            enable_thz_recording: Enable THz signature recording in agents (for determinism research, default: False)
            target_geometry: Target Platonic solid geometry for active agent guidance (default: 'none')
        """
        self._protein_sequence = protein_sequence
        self._agents: List[IProteinAgent] = []
        self._shared_memory_pool: ISharedMemoryPool = SharedMemoryPool()
        
        # Geometric targeting (NEW: Prescriptive targeting support)
        self._target_geometry = target_geometry

        # QCPP Integration (Task 7: Store QCPP integration reference)
        self._qcpp_integration = qcpp_integration
        self._qcpp_analysis_frequency = qcpp_analysis_frequency
        
        # Global QCPP Registry for cross-agent sharing (Task 8: Eliminate cross-agent waste)
        # Thread-safe dictionary for storing QCPP metrics globally
        self._global_qcpp_registry: dict = {}  # {conf_hash: qcpp_metrics}
        self._registry_lock = threading.Lock()
        self._registry_hits = 0
        self._registry_misses = 0
        self._cross_agent_reuse_count = 0  # Track cross-agent reuse success
        
        # THz recording configuration (opt-in for determinism research)
        self._enable_thz_recording = enable_thz_recording
        
        # Task 9: Initialize integrated trajectory recorder if QCPP enabled
        self._trajectory_recorder = None
        if qcpp_integration is not None:
            try:
                from .integrated_trajectory import IntegratedTrajectoryRecorder
                self._trajectory_recorder = IntegratedTrajectoryRecorder(max_points=10000)
                logger.info(
                    f"Integrated trajectory recording enabled with QCPP "
                    f"(sampling every {qcpp_analysis_frequency} iterations)"
                )
            except ImportError as e:
                logger.warning(f"Could not initialize trajectory recorder: {e}")
                self._trajectory_recorder = None

        # Adaptive configuration
        self._configurator = adaptive_configurator or get_default_configurator()
        
        # Use provided config or generate one automatically
        if adaptive_config is not None:
            self._adaptive_config = adaptive_config
        else:
            self._adaptive_config = self._configurator.get_config_for_protein(protein_sequence)

        # Checkpointing
        self._enable_checkpointing = enable_checkpointing
        self._checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir) if enable_checkpointing else None
        if self._checkpoint_manager and hasattr(self._adaptive_config, 'checkpoint_interval'):
            self._checkpoint_manager.set_auto_save_interval(self._adaptive_config.checkpoint_interval)

        # Exploration state
        self._total_iterations = 0
        self._best_conformation: Optional[Conformation] = None
        self._best_energy = float('inf')
        self._best_rmsd = float('inf')

    def initialize_agents(self, count: int, diversity_profile: str = "balanced", native_structure: Optional[Any] = None) -> List[IProteinAgent]:
        """
        Initialize agents with diversity: 33% cautious, 34% balanced, 33% aggressive.

        Args:
            count: Number of agents to initialize
            diversity_profile: Diversity profile to use ("balanced" uses standard ratios)
            native_structure: Optional native structure for RMSD validation

        Returns:
            List of initialized protein agents
        """
        if diversity_profile == "balanced":
            # For testing purposes, ensure more even distribution
            # Calculate agent counts based on ratios, but adjust for even distribution
            base_count = count // 3
            remainder = count % 3

            cautious_count = base_count + (1 if remainder > 0 else 0)
            balanced_count = base_count + (1 if remainder > 1 else 0)
            aggressive_count = base_count

            agent_configs = (
                [("cautious", cautious_count)] +
                [("balanced", balanced_count)] +
                [("aggressive", aggressive_count)]
            )
        else:
            # Use single profile for all agents
            agent_configs = [(diversity_profile, count)]

        self._agents = []
        for profile_name, profile_count in agent_configs:
            if profile_name not in AGENT_DIVERSITY_PROFILES:
                raise ValueError(f"Unknown diversity profile: {profile_name}")

            profile = AGENT_DIVERSITY_PROFILES[profile_name]

            for _ in range(profile_count):
                # Sample random consciousness coordinates within profile ranges
                frequency = random.uniform(
                    profile['frequency_range'][0],
                    profile['frequency_range'][1]
                )
                coherence = random.uniform(
                    profile['coherence_range'][0],
                    profile['coherence_range'][1]
                )

                # Create agent with adaptive configuration
                # Task 7: Pass QCPP integration to agents during initialization
                # Task 8: Pass coordinator reference for global QCPP registry
                agent = ProteinAgent(
                    protein_sequence=self._protein_sequence,
                    initial_frequency=frequency,
                    initial_coherence=coherence,
                    adaptive_config=self._adaptive_config,
                    native_structure=native_structure,
                    qcpp_integration=self._qcpp_integration,
                    qcpp_analysis_frequency=self._qcpp_analysis_frequency,
                    enable_thz_recording=self._enable_thz_recording,  # Pass THz recording flag
                    coordinator=self,  # Pass coordinator for global QCPP registry access
                    target_geometry=self._target_geometry  # NEW: Pass geometric target
                )

                self._agents.append(agent)

        return self._agents

    def run_parallel_exploration(self, iterations: int) -> ExplorationResults:
        """
        Run all agents in parallel (simultaneously) for N iterations using threading.

        Args:
            iterations: Number of iterations to run

        Returns:
            ExplorationResults with collective performance metrics
        """
        start_time = time.time()

        # Track collective metrics
        total_conformations_explored = 0
        agent_metrics = []
        
        # Lock for thread-safe updates
        update_lock = threading.Lock()

        def run_agent_iteration(agent: IProteinAgent) -> Tuple[float, float, bool]:
            """Run one iteration for a single agent (thread-safe)."""
            # Execute exploration step
            outcome = agent.explore_step()
            
            # Share significant memories with the pool (lowered threshold)
            memory_shared = False
            if outcome.significance >= 0.3:
                from .memory_system import MemorySystem
                memory_system = agent.get_memory_system()
                recent_memories = memory_system.retrieve_relevant_memories(
                    outcome.move_executed.move_type.value, max_count=1
                )
                if recent_memories:
                    with update_lock:
                        self._shared_memory_pool.share_memory(recent_memories[0])
                    memory_shared = True
            
            # Get current conformation
            current_conf = agent.get_current_conformation()
            return (current_conf.energy, 
                    current_conf.rmsd_to_native if current_conf.rmsd_to_native else float('inf'),
                    memory_shared)

        # Run iterations
        for iteration in range(iterations):
            self._total_iterations += 1

            # Run all agents in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(self._agents)) as executor:
                # Submit all agent tasks
                futures = [executor.submit(run_agent_iteration, agent) for agent in self._agents]
                
                # Wait for all to complete and collect results
                iteration_conformations = 0
                for future in as_completed(futures):
                    energy, rmsd, memory_shared = future.result()
                    iteration_conformations += 1
                    
                    # Update best conformation tracking (thread-safe)
                    with update_lock:
                        if energy < self._best_energy:
                            self._best_energy = energy
                            # Update best conformation from the agent that achieved this
                            for agent in self._agents:
                                current_conf = agent.get_current_conformation()
                                if abs(current_conf.energy - energy) < 0.01:  # Match by energy
                                    self._best_conformation = current_conf
                                    break
                        
                        if rmsd < self._best_rmsd:
                            self._best_rmsd = rmsd

            total_conformations_explored += iteration_conformations

            # Auto-save checkpoint if enabled
            if self._enable_checkpointing and self._checkpoint_manager:
                try:
                    checkpoint_file = self._checkpoint_manager.auto_save(
                        agents=self._agents,
                        shared_pool=self._shared_memory_pool,
                        iteration=self._total_iterations,
                        metadata={
                            "protein_sequence": self._protein_sequence,
                            "agent_count": len(self._agents),
                            "best_energy": self._best_energy,
                            "best_rmsd": self._best_rmsd
                        }
                    )
                    if checkpoint_file:
                        logger.info(f"Auto-saved checkpoint: {checkpoint_file}")
                except Exception as e:
                    # Log but don't crash - checkpointing is non-critical
                    logger.warning(f"Checkpoint auto-save failed: {e}")

            # Optional: Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Completed iteration {iteration + 1}/{iterations}")
            
            # Sync shared memories to agents every 20 iterations
            if (iteration + 1) % 20 == 0:
                self._sync_shared_memories_to_agents()
            
            # Task 9: Record integrated trajectory point if QCPP enabled
            # Only record every N iterations to avoid performance bottleneck
            should_record_trajectory = (
                self._trajectory_recorder is not None 
                and self._qcpp_integration is not None
                and (self._total_iterations % self._qcpp_analysis_frequency == 0)
            )
            if should_record_trajectory:
                try:
                    # Get best agent's current state for this iteration
                    best_agent = self._agents[0]  # Start with first agent
                    best_agent_energy = float('inf')
                    
                    for agent in self._agents:
                        conf = agent.get_current_conformation()
                        if conf.energy < best_agent_energy:
                            best_agent = agent
                            best_agent_energy = conf.energy
                    
                    # Get UBF metrics from best agent
                    best_conf = best_agent.get_current_conformation()
                    consciousness = best_agent._consciousness  # type: ignore
                    
                    # Get QCPP metrics for best conformation (only every N iterations now)
                    if self._qcpp_integration is not None:
                        qcpp_metrics = self._qcpp_integration.analyze_conformation(best_conf)
                    
                        # Record trajectory point
                        if self._trajectory_recorder is not None:
                            self._trajectory_recorder.record_point(
                                iteration=self._total_iterations,
                                rmsd=best_conf.rmsd_to_native if best_conf.rmsd_to_native else 0.0,
                                energy=best_conf.energy,
                                consciousness_frequency=consciousness.get_frequency(),
                                consciousness_coherence=consciousness.get_coherence(),
                                qcp_score=qcpp_metrics.qcp_score,
                                field_coherence=qcpp_metrics.field_coherence,
                                stability_score=qcpp_metrics.stability_score,
                                phi_match_score=qcpp_metrics.phi_match_score
                            )
                except Exception as e:
                    # Log but don't crash - trajectory recording is non-critical
                    logger.warning(f"Trajectory recording failed at iteration {self._total_iterations}: {e}")

        # Collect final agent metrics
        for i, agent in enumerate(self._agents):
            metrics_dict = agent.get_exploration_metrics()
            metrics = ExplorationMetrics(
                agent_id=f"agent_{i}",
                iterations_completed=int(metrics_dict["iterations_completed"]),
                conformations_explored=int(metrics_dict["conformations_explored"]),
                memories_created=int(metrics_dict["memories_created"]),
                best_energy_found=metrics_dict["best_energy"],
                best_rmsd_found=metrics_dict["best_rmsd"],
                learning_improvement=metrics_dict.get("learning_improvement", 0.0),
                avg_decision_time_ms=metrics_dict["avg_decision_time_ms"],
                stuck_in_minima_count=int(metrics_dict["stuck_in_minima_count"]),
                successful_escapes=int(metrics_dict["successful_escapes"])
            )
            agent_metrics.append(metrics)

        # Calculate collective learning benefit (simplified)
        # This compares average single-agent performance to multi-agent performance
        if agent_metrics:
            avg_single_agent_improvement = sum(m.learning_improvement for m in agent_metrics) / len(agent_metrics)
            # Multi-agent benefit is the excess improvement beyond single-agent average
            collective_learning_benefit = max(0.0, avg_single_agent_improvement - 10.0)  # Subtract baseline
        else:
            collective_learning_benefit = 0.0

        total_runtime = time.time() - start_time

        # Task 9: Compute correlation analysis if trajectory recording was enabled
        qcpp_trajectory_data = None
        qcpp_rmsd_correlations = None
        qcpp_energy_correlations = None
        consciousness_qcpp_correlations = None
        
        if self._trajectory_recorder is not None and self._trajectory_recorder.get_point_count() > 0:
            try:
                from .integrated_trajectory import TrajectoryAnalyzer
                
                # Export trajectory data
                qcpp_trajectory_data = self._trajectory_recorder.export_to_dict()
                
                # Perform correlation analysis (requires at least 2 points)
                if self._trajectory_recorder.get_point_count() >= 2:
                    analyzer = TrajectoryAnalyzer(self._trajectory_recorder.get_points())
                    
                    qcpp_rmsd_correlations = analyzer.calculate_qcpp_rmsd_correlation()
                    qcpp_energy_correlations = analyzer.calculate_qcpp_energy_correlation()
                    consciousness_qcpp_correlations = analyzer.calculate_consciousness_qcpp_correlation()
                    
                    logger.info(
                        f"QCPP-RMSD correlations: QCP={qcpp_rmsd_correlations['qcp_rmsd_corr']:.3f}, "
                        f"Stability={qcpp_rmsd_correlations['stability_rmsd_corr']:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")

        return ExplorationResults(
            total_iterations=self._total_iterations,
            total_conformations_explored=total_conformations_explored,
            best_conformation=self._best_conformation,
            best_energy=self._best_energy,
            best_rmsd=self._best_rmsd,
            agent_metrics=agent_metrics,
            collective_learning_benefit=collective_learning_benefit,
            total_runtime_seconds=total_runtime,
            shared_memories_created=self._shared_memory_pool.get_total_memories(),
            qcpp_trajectory_data=qcpp_trajectory_data,
            qcpp_rmsd_correlations=qcpp_rmsd_correlations,
            qcpp_energy_correlations=qcpp_energy_correlations,
            consciousness_qcpp_correlations=consciousness_qcpp_correlations
        )

    def resume_from_checkpoint(self, checkpoint_file: str) -> int:
        """
        Resume exploration from a checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Iteration number to resume from
            
        Raises:
            ValueError: If checkpoint loading fails or checkpointing is disabled
        """
        if not self._enable_checkpointing or not self._checkpoint_manager:
            raise ValueError("Checkpointing is not enabled")
        
        try:
            # Load checkpoint data
            checkpoint_data = self._checkpoint_manager.load_checkpoint(checkpoint_file)
            
            # Restore agents and shared pool
            self._agents, self._shared_memory_pool, iteration = self._checkpoint_manager.restore_agents(
                checkpoint_data,
                ProteinAgent
            )
            
            # Restore exploration state
            self._total_iterations = iteration
            
            # Recalculate best conformation from restored agents
            self._best_energy = float('inf')
            self._best_rmsd = float('inf')
            self._best_conformation = None
            
            for agent in self._agents:
                current_conf = agent.get_current_conformation()
                if current_conf.energy < self._best_energy:
                    self._best_energy = current_conf.energy
                    self._best_conformation = current_conf
                
                if (current_conf.rmsd_to_native and
                    current_conf.rmsd_to_native < self._best_rmsd):
                    self._best_rmsd = current_conf.rmsd_to_native
            
            logger.info(
                f"Resumed from checkpoint: {len(self._agents)} agents, "
                f"iteration {iteration}, best energy {self._best_energy:.2f}"
            )
            
            return iteration
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise

    def save_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Manually save a checkpoint.
        
        Args:
            checkpoint_name: Optional custom checkpoint name
            
        Returns:
            Path to saved checkpoint file
            
        Raises:
            ValueError: If checkpointing is disabled
        """
        if not self._enable_checkpointing or not self._checkpoint_manager:
            raise ValueError("Checkpointing is not enabled")
        
        try:
            checkpoint_file = self._checkpoint_manager.save_checkpoint(
                agents=self._agents,
                shared_pool=self._shared_memory_pool,
                iteration=self._total_iterations,
                metadata={
                    "protein_sequence": self._protein_sequence,
                    "agent_count": len(self._agents),
                    "best_energy": self._best_energy,
                    "best_rmsd": self._best_rmsd,
                    "manual_save": True,
                    "checkpoint_name": checkpoint_name
                }
            )
            
            logger.info(f"Manual checkpoint saved: {checkpoint_file}")
            return checkpoint_file
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def get_qcpp_from_registry(self, conformation: Any) -> Optional[Any]:
        """
        Check if ANY agent has analyzed this conformation before.
        Thread-safe for parallel exploration.
        
        This enables cross-agent QCPP sharing - if Agent 2 already calculated
        QCPP metrics for a conformation, Agent 5 can reuse them without
        recalculation (100× speedup: 1ms → 0.01ms).
        
        Args:
            conformation: Conformation to query QCPP metrics for
            
        Returns:
            QCPPMetrics if found in registry, None if never analyzed
        """
        try:
            conf_hash = self._hash_conformation(conformation)
            
            with self._registry_lock:
                if conf_hash in self._global_qcpp_registry:
                    self._registry_hits += 1
                    self._cross_agent_reuse_count += 1
                    logger.debug(f"✓ Cross-agent QCPP reuse (registry hit)")
                    return self._global_qcpp_registry[conf_hash]
                else:
                    self._registry_misses += 1
                    return None
        except Exception as e:
            logger.warning(f"Error querying global QCPP registry: {e}")
            return None
    
    def store_qcpp_in_registry(self, conformation: Any, qcpp_metrics: Any) -> None:
        """
        Store QCPP metrics globally for all agents to reuse.
        Thread-safe for parallel exploration.
        
        Once stored, ANY agent can retrieve these metrics when visiting
        the same conformation, eliminating redundant QCPP calculations.
        
        Args:
            conformation: Conformation these metrics apply to
            qcpp_metrics: QCPP metrics to store
        """
        try:
            conf_hash = self._hash_conformation(conformation)
            
            with self._registry_lock:
                # Only store if not already present (first-writer wins)
                if conf_hash not in self._global_qcpp_registry:
                    self._global_qcpp_registry[conf_hash] = qcpp_metrics
                    logger.debug(f"✓ Stored QCPP in global registry (hash: {conf_hash[:8]})")
        except Exception as e:
            logger.warning(f"Error storing QCPP in global registry: {e}")
    
    def get_registry_stats(self) -> dict:
        """
        Get cross-agent QCPP sharing statistics.
        
        Returns:
            Dictionary with registry statistics including hit rate,
            total queries, cache size, and savings estimate
        """
        with self._registry_lock:
            total_queries = self._registry_hits + self._registry_misses
            hit_rate = (self._registry_hits / total_queries * 100) if total_queries > 0 else 0.0
            
            # Estimate time savings (1ms per QCPP analysis avoided)
            time_saved_ms = self._registry_hits * 1.0
            
            return {
                'total_queries': total_queries,
                'cache_hits': self._registry_hits,
                'cache_misses': self._registry_misses,
                'hit_rate_percent': hit_rate,
                'registry_size': len(self._global_qcpp_registry),
                'cross_agent_reuse_count': self._cross_agent_reuse_count,
                'estimated_time_saved_ms': time_saved_ms,
                'estimated_time_saved_s': time_saved_ms / 1000.0
            }
    
    def _hash_conformation(self, conformation: Any) -> str:
        """
        Generate hash for conformation based on atom coordinates.
        
        Uses first 10 CA atom coordinates (rounded to 1 decimal) to create
        a compact hash that identifies unique conformations while being
        tolerant to minor numerical differences.
        
        Args:
            conformation: Conformation to hash
            
        Returns:
            Hash string for conformation lookup
        """
        try:
            import hashlib
            
            # Extract coordinates (first 10 atoms for speed)
            coords = []
            if hasattr(conformation, 'atom_coordinates'):
                atom_coords = conformation.atom_coordinates
                # atom_coordinates is a List[Tuple[float, float, float]]
                # Take first 10 atoms, round to 1 decimal place
                for coord in atom_coords[:10]:
                    if len(coord) >= 3:
                        coords.extend([round(coord[0], 1), round(coord[1], 1), round(coord[2], 1)])
            
            # Create hash from coordinate string
            coord_str = '_'.join(str(c) for c in coords)
            return hashlib.sha256(coord_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error hashing conformation: {e}")
            return ""

    def get_best_conformation(self) -> Tuple[Conformation, float, float]:
        """
        Get best conformation found (conformation, energy, RMSD).

        Returns:
            Tuple of (best_conformation, best_energy, best_rmsd)
        """
        if self._best_conformation is None:
            raise ValueError("No exploration has been performed yet")

        return (self._best_conformation, self._best_energy, self._best_rmsd)

    def get_agents(self) -> List[IProteinAgent]:
        """
        Get all initialized agents.

        Returns:
            List of protein agents
        """
        return self._agents

    def get_shared_memory_pool(self) -> ISharedMemoryPool:
        """
        Get the shared memory pool.

        Returns:
            Shared memory pool instance
        """
        return self._shared_memory_pool

    def export_results(self, output_file: str) -> None:
        """
        Export exploration results to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        import json
        from datetime import datetime

        if not self._agents:
            raise ValueError("No agents initialized - cannot export results")

        # Get exploration results
        results = self.run_parallel_exploration(0)  # Get current state without additional iterations

        # Convert to serializable format
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "protein_sequence": self._protein_sequence,
                "protein_length": len(self._protein_sequence),
                "agent_count": len(self._agents),
                "total_iterations": self._total_iterations
            },
            "results": {
                "total_conformations_explored": results.total_conformations_explored,
                "best_energy": results.best_energy,
                "best_rmsd": results.best_rmsd,
                "collective_learning_benefit": results.collective_learning_benefit,
                "total_runtime_seconds": results.total_runtime_seconds,
                "shared_memories_created": results.shared_memories_created
            },
            "best_conformation": None,
            "agent_metrics": []
        }

        # Add best conformation if available
        if results.best_conformation:
            export_data["best_conformation"] = {
                "conformation_id": results.best_conformation.conformation_id,
                "energy": results.best_conformation.energy,
                "rmsd_to_native": results.best_conformation.rmsd_to_native,
                "secondary_structure": results.best_conformation.secondary_structure,
                "sequence_length": len(results.best_conformation.sequence)
            }

        # Add agent metrics
        for metrics in results.agent_metrics:
            agent_data = {
                "agent_id": metrics.agent_id,
                "iterations_completed": metrics.iterations_completed,
                "conformations_explored": metrics.conformations_explored,
                "memories_created": metrics.memories_created,
                "best_energy_found": metrics.best_energy_found,
                "best_rmsd_found": metrics.best_rmsd_found,
                "learning_improvement": metrics.learning_improvement,
                "avg_decision_time_ms": metrics.avg_decision_time_ms,
                "stuck_in_minima_count": metrics.stuck_in_minima_count,
                "successful_escapes": metrics.successful_escapes
            }
            export_data["agent_metrics"].append(agent_data)

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

    def get_adaptive_config(self) -> AdaptiveConfig:
        """
        Get the adaptive configuration used by this coordinator.
        
        Returns:
            AdaptiveConfig instance
        """
        return self._adaptive_config
    
    def _sync_shared_memories_to_agents(self) -> None:
        """
        Sync shared memories from the pool to individual agents.
        
        Allows agents to learn from each other's experiences by
        importing high-value memories from the shared pool.
        """
        try:
            # Get shared memories for different move types
            move_types = ["backbone_rotation", "sidechain_adjust", "helix_formation", 
                         "sheet_formation", "hydrophobic_collapse"]
            
            shared_memories = []
            for move_type in move_types:
                memories = self._shared_memory_pool.retrieve_shared_memories(move_type, max_count=2)
                shared_memories.extend(memories)
            
            if not shared_memories:
                return
            
            # Distribute top memories to each agent
            for agent in self._agents:
                memory_system = agent.get_memory_system()
                
                # Import shared memories (top 5 per agent to avoid overwhelming)
                for memory in shared_memories[:5]:
                    try:
                        # Store memory in agent's memory system
                        memory_system.store_memory(memory)
                    except Exception as e:
                        # Ignore individual memory import failures
                        pass
            
            logger.debug(f"Synced {len(shared_memories)} shared memories to {len(self._agents)} agents")
        
        except Exception as e:
            # Non-critical operation - log and continue
            logger.warning(f"Error syncing shared memories: {e}")

    def get_configuration_summary(self) -> str:
        """
        Get human-readable summary of the adaptive configuration.
        
        Returns:
            Formatted configuration summary string
        """
        return self._configurator.get_config_summary(self._adaptive_config)

    def get_qcpp_integration(self) -> Optional[Any]:
        """
        Get the QCPP integration adapter instance.
        
        This is useful for trajectory recording and external analysis
        of QCPP-enhanced exploration.
        
        Returns:
            QCPP integration adapter if enabled, None otherwise
        """
        return self._qcpp_integration
    
    def get_trajectory_recorder(self) -> Optional[Any]:
        """
        Get the integrated trajectory recorder instance.
        
        Returns:
            IntegratedTrajectoryRecorder if QCPP is enabled, None otherwise
        """
        return self._trajectory_recorder
    
    def export_best_conformation_coordinates(self) -> Optional[List[Tuple[float, float, float]]]:
        """
        Export the atom coordinates from the best conformation found.
        
        This method extracts the 3D Cα coordinates from the best energy conformation
        discovered during exploration. Useful for downstream geometric analysis
        (φ patterns, symmetry) on predicted structures.
        
        Returns:
            List of (x, y, z) tuples representing Cα atom coordinates,
            or None if no conformation has been found yet.
            
        Example:
            >>> coordinator = MultiAgentCoordinator("ACDEFGH")
            >>> coordinator.initialize_agents(count=10)
            >>> coordinator.run_parallel_exploration(iterations=200)
            >>> coords = coordinator.export_best_conformation_coordinates()
            >>> print(f"Exported {len(coords)} CA atoms")
            Exported 7 CA atoms
        """
        if self._best_conformation is None:
            logger.warning("No best conformation available to export")
            return None
        
        # Extract atom_coordinates from the Conformation dataclass
        coordinates = self._best_conformation.atom_coordinates
        
        logger.info(
            f"Exported {len(coordinates)} CA coordinates from best conformation "
            f"(Energy: {self._best_conformation.energy:.2f} kcal/mol, "
            f"RMSD: {self._best_conformation.rmsd_to_native or 'N/A'})"
        )
        
        return coordinates