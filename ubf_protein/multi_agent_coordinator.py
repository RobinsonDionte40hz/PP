"""
Multi-agent coordination implementation for UBF protein system.

This module implements the MultiAgentCoordinator that manages multiple
protein agents working together to explore conformational space.
"""

import time
import random
import logging
from typing import List, Tuple, Optional

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
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize multi-agent coordinator with protein sequence.

        Args:
            protein_sequence: Amino acid sequence for all agents
            adaptive_configurator: Optional configurator for auto-configuration
            adaptive_config: Optional pre-configured AdaptiveConfig (overrides auto-config)
            enable_checkpointing: Whether to enable automatic checkpointing
            checkpoint_dir: Directory for checkpoint files
        """
        self._protein_sequence = protein_sequence
        self._agents: List[IProteinAgent] = []
        self._shared_memory_pool: ISharedMemoryPool = SharedMemoryPool()

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

    def initialize_agents(self, count: int, diversity_profile: str = "balanced") -> List[IProteinAgent]:
        """
        Initialize agents with diversity: 33% cautious, 34% balanced, 33% aggressive.

        Args:
            count: Number of agents to initialize
            diversity_profile: Diversity profile to use ("balanced" uses standard ratios)

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
                agent = ProteinAgent(
                    protein_sequence=self._protein_sequence,
                    initial_frequency=frequency,
                    initial_coherence=coherence,
                    adaptive_config=self._adaptive_config
                )

                self._agents.append(agent)

        return self._agents

    def run_parallel_exploration(self, iterations: int) -> ExplorationResults:
        """
        Run all agents in parallel for N iterations.

        Args:
            iterations: Number of iterations to run

        Returns:
            ExplorationResults with collective performance metrics
        """
        start_time = time.time()

        # Track collective metrics
        total_conformations_explored = 0
        agent_metrics = []

        for iteration in range(iterations):
            self._total_iterations += 1

            # Run one iteration for each agent
            iteration_conformations = 0

            for agent in self._agents:
                # Execute exploration step
                outcome = agent.explore_step()
                iteration_conformations += 1

                # Share significant memories with the pool
                if outcome.significance >= 0.7:  # Same threshold as shared pool
                    from .memory_system import MemorySystem
                    memory_system = agent.get_memory_system()
                    # Get the most recent memory (should be the one from this outcome)
                    recent_memories = memory_system.retrieve_relevant_memories(
                        outcome.move_executed.move_type.value, max_count=1
                    )
                    if recent_memories:
                        self._shared_memory_pool.share_memory(recent_memories[0])

                # Update best conformation tracking
                current_conf = agent.get_current_conformation()
                if current_conf.energy < self._best_energy:
                    self._best_energy = current_conf.energy
                    self._best_conformation = current_conf

                if (current_conf.rmsd_to_native and
                    current_conf.rmsd_to_native < self._best_rmsd):
                    self._best_rmsd = current_conf.rmsd_to_native

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

        return ExplorationResults(
            total_iterations=self._total_iterations,
            total_conformations_explored=total_conformations_explored,
            best_conformation=self._best_conformation,
            best_energy=self._best_energy,
            best_rmsd=self._best_rmsd,
            agent_metrics=agent_metrics,
            collective_learning_benefit=collective_learning_benefit,
            total_runtime_seconds=total_runtime,
            shared_memories_created=self._shared_memory_pool.get_total_memories()
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
    
    def get_configuration_summary(self) -> str:
        """
        Get human-readable summary of the adaptive configuration.
        
        Returns:
            Formatted configuration summary string
        """
        return self._configurator.get_config_summary(self._adaptive_config)

        print(f"Results exported to {output_file}")