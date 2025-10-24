"""
Multi-agent coordination implementation for UBF protein system.

This module implements the MultiAgentCoordinator that manages multiple
protein agents working together to explore conformational space.
"""

import time
import random
from typing import List, Tuple, Optional

from .interfaces import IMultiAgentCoordinator, IProteinAgent, ISharedMemoryPool
from .models import ExplorationResults, ExplorationMetrics, Conformation, AdaptiveConfig, ProteinSizeClass
from .protein_agent import ProteinAgent
from .memory_system import SharedMemoryPool
from .config import AGENT_DIVERSITY_PROFILES, AGENT_PROFILE_CAUTIOUS_RATIO, AGENT_PROFILE_BALANCED_RATIO, AGENT_PROFILE_AGGRESSIVE_RATIO


class MultiAgentCoordinator(IMultiAgentCoordinator):
    """
    Implementation of multi-agent coordination system.

    Manages multiple protein agents with diverse consciousness profiles
    to collectively explore conformational space through parallel execution
    and shared memory exchange.
    """

    def __init__(self, protein_sequence: str):
        """
        Initialize multi-agent coordinator with protein sequence.

        Args:
            protein_sequence: Amino acid sequence for all agents
        """
        self._protein_sequence = protein_sequence
        self._agents: List[IProteinAgent] = []
        self._shared_memory_pool: ISharedMemoryPool = SharedMemoryPool()

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

                # Create adaptive config based on protein size and agent profile
                residue_count = len(self._protein_sequence)
                if residue_count < 50:
                    size_class = ProteinSizeClass.SMALL
                elif residue_count <= 150:
                    size_class = ProteinSizeClass.MEDIUM
                else:
                    size_class = ProteinSizeClass.LARGE

                adaptive_config = AdaptiveConfig(
                    size_class=size_class,
                    residue_count=residue_count,
                    initial_frequency_range=profile['frequency_range'],
                    initial_coherence_range=profile['coherence_range'],
                    stuck_detection_window=10,  # Default window
                    stuck_detection_threshold=5.0,  # Default threshold
                    memory_significance_threshold=0.3,
                    max_memories_per_agent=50,
                    convergence_energy_threshold=10.0,
                    convergence_rmsd_threshold=2.0,
                    max_iterations=1000,
                    checkpoint_interval=100
                )

                # Create agent with adaptive config and sampled coordinates
                agent = ProteinAgent(
                    protein_sequence=self._protein_sequence,
                    initial_frequency=frequency,
                    initial_coherence=coherence,
                    adaptive_config=adaptive_config
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
                learning_improvement=0.0,  # TODO: Calculate actual improvement
                avg_decision_time_ms=metrics_dict["avg_decision_time_ms"],
                stuck_in_minima_count=int(metrics_dict["stuck_in_minima_count"]),
                successful_escapes=int(metrics_dict["successful_escapes"])
            )
            agent_metrics.append(metrics)

        # Calculate collective learning benefit (simplified)
        # This would compare single-agent vs multi-agent performance
        collective_learning_benefit = 0.0  # TODO: Implement actual calculation

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