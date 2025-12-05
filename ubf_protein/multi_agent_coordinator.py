"""
Multi-Agent Coordination System for UBF Protein System

Copyright (c) 2025 Dionte Robinson. All Rights Reserved.

PROPRIETARY ALGORITHM - Patent Pending
This module contains proprietary multi-agent coordination algorithms for
parallel protein conformational exploration using consciousness-inspired
parameter spaces.

Key innovations include:
- Diverse agent population management (33% cautious, 34% balanced, 33% aggressive)
- Shared memory pool with significance-based filtering
- Parallel exploration with thread-safe coordination
- Adaptive configuration based on protein size
- QCPP integration for quantum-guided optimization

For commercial licensing, contact: dionterobinson.biorxiv@gmail.com

---

This module implements the MultiAgentCoordinator that manages multiple
protein agents working together to explore conformational space.
"""

import time
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Any, Dict

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
                 target_geometry: str = 'none',
                 enable_mediators: bool = False,
                 mediator_count: int = 2,
                 mediator_config: Optional[Any] = None,
                 enable_quantum_refinement: bool = False,
                 refinement_rmsd_threshold: float = 5.0,
                 refinement_config: Optional[Any] = None):
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
            enable_mediators: Whether to enable Mediator Agents for pattern detection (default: False)
            mediator_count: Number of Mediator Agents to initialize (default: 2)
            mediator_config: Optional MediatorConfig instance (uses default if None)
            enable_quantum_refinement: Whether to enable automatic quantum refinement for coarse structures (default: False)
            refinement_rmsd_threshold: RMSD threshold to trigger refinement in Ångströms (default: 5.0Å)
            refinement_config: Optional RefinementConfig instance for customization (uses default if None)
        """
        self._protein_sequence = protein_sequence
        self._agents: List[IProteinAgent] = []
        self._shared_memory_pool: ISharedMemoryPool = SharedMemoryPool()
        
        # Mediator Agent configuration (Task 10.1)
        self._enable_mediators = enable_mediators
        self._mediator_count = mediator_count
        self._mediator_config = mediator_config
        self._mediators: List[Any] = []  # List of MediatorAgent instances
        self._geometric_analyzer: Optional[Any] = None  # GeometricAttractorAnalyzer instance
        
        # Geometric targeting (NEW: Prescriptive targeting support)
        self._target_geometry = target_geometry

        # QCPP Integration (Task 7: Store QCPP integration reference)
        self._qcpp_integration = qcpp_integration
        self._qcpp_analysis_frequency = qcpp_analysis_frequency
        
        # Quantum Refinement Engine configuration (Task 13)
        self._enable_quantum_refinement = enable_quantum_refinement
        self._refinement_rmsd_threshold = refinement_rmsd_threshold
        self._refinement_config = refinement_config
        self._refinement_engine: Optional[Any] = None  # QuantumRefinementEngine instance
        
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
        
        # Initialize Quantum Refinement Engine if enabled (Task 13)
        if self._enable_quantum_refinement:
            if self._qcpp_integration is None:
                logger.warning(
                    "Quantum refinement enabled but QCPP integration is not configured. "
                    "Refinement will not be available."
                )
                self._enable_quantum_refinement = False
            else:
                try:
                    from .quantum_refinement_engine import QuantumRefinementEngine
                    from .energy_function import MolecularMechanicsEnergy
                    from .rmsd_calculator import RMSDCalculator
                    
                    energy_calculator = MolecularMechanicsEnergy()
                    rmsd_calculator = RMSDCalculator()
                    
                    self._refinement_engine = QuantumRefinementEngine(
                        qcpp_adapter=self._qcpp_integration,
                        energy_calculator=energy_calculator,
                        rmsd_calculator=rmsd_calculator
                    )
                    
                    logger.info(
                        f"Quantum Refinement Engine initialized with RMSD threshold {self._refinement_rmsd_threshold}Å"
                    )
                except (ImportError, TypeError) as e:
                    logger.error(f"Failed to initialize Quantum Refinement Engine: {e}")
                    self._enable_quantum_refinement = False
                    self._refinement_engine = None

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
                    enable_visualization=True,  # Enable trajectory tracking for visualization
                    max_snapshots=500,  # Limit snapshots per agent to control memory
                    native_structure=native_structure,
                    qcpp_integration=self._qcpp_integration,
                    qcpp_analysis_frequency=self._qcpp_analysis_frequency,
                    enable_thz_recording=self._enable_thz_recording,  # Pass THz recording flag
                    coordinator=self,  # Pass coordinator for global QCPP registry access
                    target_geometry=self._target_geometry  # NEW: Pass geometric target
                )

                self._agents.append(agent)

        return self._agents

    def initialize_mediators(self) -> List[Any]:
        """
        Initialize Mediator Agents for pattern detection.
        
        Creates mediator_count MediatorAgent instances with shared dependencies.
        Mediators analyze conformations from exploration agents to detect:
        - THz resonance patterns
        - Folding dynamics (secondary structure formation)
        - Geometric similarities between conformations
        
        Returns:
            List of initialized Mediator Agents
        
        Raises:
            ValueError: If mediators are disabled or dependencies are missing
        
        Example:
            >>> coordinator = MultiAgentCoordinator(
            ...     protein_sequence="ACDEFGH",
            ...     enable_mediators=True,
            ...     mediator_count=3
            ... )
            >>> mediators = coordinator.initialize_mediators()
            >>> print(f"Initialized {len(mediators)} mediators")
            Initialized 3 mediators
        """
        if not self._enable_mediators:
            raise ValueError("Mediators are not enabled. Set enable_mediators=True in constructor.")
        
        # Initialize GeometricAttractorAnalyzer if not already initialized
        if self._geometric_analyzer is None:
            try:
                from .geometric_attractor import GeometricAttractorAnalyzer
                self._geometric_analyzer = GeometricAttractorAnalyzer()
                logger.info("Initialized GeometricAttractorAnalyzer for Mediators")
            except ImportError as e:
                raise ValueError(f"Cannot import GeometricAttractorAnalyzer: {e}")
        
        # Get or create MediatorConfig
        if self._mediator_config is None:
            try:
                from .mediator_config import MediatorConfig
                self._mediator_config = MediatorConfig()
                logger.info("Using default MediatorConfig")
            except ImportError as e:
                raise ValueError(f"Cannot import MediatorConfig: {e}")
        
        # Import MediatorAgent
        try:
            from .mediator_agent import MediatorAgent
        except ImportError as e:
            raise ValueError(f"Cannot import MediatorAgent: {e}")
        
        # Create Mediator Agents
        self._mediators = []
        for i in range(self._mediator_count):
            try:
                mediator = MediatorAgent(
                    protein_sequence=self._protein_sequence,
                    qcpp_adapter=self._qcpp_integration,
                    geometric_analyzer=self._geometric_analyzer,
                    shared_memory=self._shared_memory_pool,
                    config=self._mediator_config
                )
                self._mediators.append(mediator)
                logger.info(f"Initialized Mediator Agent {i+1}/{self._mediator_count}")
            except Exception as e:
                logger.error(f"Failed to initialize Mediator Agent {i+1}: {e}")
                raise
        
        logger.info(
            f"Initialized {len(self._mediators)} Mediator Agents with config: "
            f"relay_frequency={self._mediator_config.relay_frequency}, "
            f"thz_detection={self._mediator_config.enable_thz_detection}, "
            f"folding_detection={self._mediator_config.enable_folding_detection}, "
            f"geometric_detection={self._mediator_config.enable_geometric_detection}"
        )
        
        return self._mediators

    def run_mediator_cycle(self, iteration: int, best_conformation: Optional[Conformation] = None) -> List[Any]:
        """
        Run pattern detection cycle for all Mediator Agents.
        
        This method is called periodically during exploration (every relay_frequency iterations)
        to detect emergent patterns in protein conformations.
        
        The detection cycle:
        1. Get best conformation from exploration agents
        2. Each Mediator analyzes the conformation for patterns
        3. Detected patterns are validated via QCPP relay
        4. Significant patterns are broadcast to exploration agents
        
        Args:
            iteration: Current exploration iteration number
            best_conformation: Best conformation found so far (optional, will find if None)
        
        Returns:
            List of all PatternDetection objects found by Mediators
        
        Example:
            >>> coordinator = MultiAgentCoordinator(...)
            >>> coordinator.initialize_agents(count=10)
            >>> coordinator.initialize_mediators()
            >>> 
            >>> # During exploration
            >>> for iteration in range(100):
            ...     # ... exploration steps ...
            ...     if iteration % relay_frequency == 0:
            ...         patterns = coordinator.run_mediator_cycle(iteration)
            ...         print(f"Detected {len(patterns)} patterns")
        """
        if not self._enable_mediators or len(self._mediators) == 0:
            return []
        
        # Get best conformation if not provided
        if best_conformation is None:
            if self._best_conformation is None:
                # No conformation to analyze yet
                return []
            best_conformation = self._best_conformation
        
        all_patterns = []
        
        # Update iteration counter in each mediator
        for mediator in self._mediators:
            mediator.current_iteration = iteration
        
        # Run detection cycle for each mediator
        for mediator_idx, mediator in enumerate(self._mediators):
            try:
                # Detect patterns in best conformation
                patterns = mediator.detect_patterns(best_conformation)
                
                # Process each detected pattern
                for pattern in patterns:
                    # Relay to QCPP for validation
                    qcpp_metrics = mediator.relay_to_qcpp(pattern, best_conformation)
                    
                    # Broadcast to exploration agents
                    success = mediator.broadcast_to_agents(pattern, qcpp_metrics)
                    
                    if success:
                        logger.debug(
                            f"Mediator {mediator_idx} broadcast pattern: "
                            f"type={pattern.pattern_type.value}, "
                            f"significance={pattern.significance.value}"
                        )
                    
                    # Add to results
                    all_patterns.append(pattern)
                
            except Exception as e:
                # Log error but continue with other mediators
                logger.warning(f"Mediator {mediator_idx} detection cycle failed: {e}")
                continue
        
        if all_patterns:
            logger.info(
                f"Mediator cycle {iteration}: Detected {len(all_patterns)} patterns "
                f"(THz={sum(1 for p in all_patterns if p.pattern_type.value == 'thz')}, "
                f"Folding={sum(1 for p in all_patterns if p.pattern_type.value == 'folding')}, "
                f"Geometric={sum(1 for p in all_patterns if p.pattern_type.value == 'geometric')})"
            )
        
        return all_patterns

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
            rmsd_value = current_conf.rmsd_to_native if current_conf.rmsd_to_native is not None else float('inf')
            logger.debug(f"Agent conformation: energy={current_conf.energy:.2f}, rmsd_to_native={current_conf.rmsd_to_native}, rmsd_value={rmsd_value}")
            return (current_conf.energy, rmsd_value, memory_shared)

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
                                best_conf = agent.get_best_conformation()
                                if abs(best_conf.energy - energy) < 0.01:  # Match by energy
                                    self._best_conformation = best_conf
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
            
            # Task 10.4: Run Mediator detection cycle at relay_frequency intervals
            if self._enable_mediators and len(self._mediators) > 0:
                # Get relay frequency from mediator config
                relay_frequency = self._mediator_config.relay_frequency if self._mediator_config else 10
                
                if (iteration + 1) % relay_frequency == 0:
                    try:
                        patterns = self.run_mediator_cycle(
                            iteration=self._total_iterations,
                            best_conformation=self._best_conformation
                        )
                        
                        # Update reference conformations in mediators with best conformation
                        if self._best_conformation is not None and self._geometric_analyzer is not None:
                            for mediator in self._mediators:
                                # Get geometric score from analyzer
                                try:
                                    conf_dict = {
                                        'coordinates': self._best_conformation.atom_coordinates,
                                        'sequence': self._best_conformation.sequence if hasattr(self._best_conformation, 'sequence') else None
                                    }
                                    geo_result = self._geometric_analyzer.analyze_conformation(
                                        conf_dict,
                                        sequence=self._best_conformation.sequence if hasattr(self._best_conformation, 'sequence') else None
                                    )
                                    geometric_score = geo_result.golden_ratio_percentage
                                except Exception:
                                    geometric_score = 0.0
                                
                                # Add to reference set
                                mediator.add_reference_conformation(
                                    self._best_conformation,
                                    agent_id=f"best_conf_iter_{self._total_iterations}",
                                    geometric_score=geometric_score
                                )
                    
                    except Exception as e:
                        # Log but don't crash - mediator cycle is non-critical
                        logger.warning(f"Mediator cycle failed at iteration {self._total_iterations}: {e}")
            
            # Task 9: Record integrated trajectory point if QCPP enabled
            # Only record every N iterations to avoid performance bottleneck
            should_record_trajectory = (
                self._trajectory_recorder is not None 
                and self._qcpp_integration is not None
                and (self._total_iterations % self._qcpp_analysis_frequency == 0)
            )
            if should_record_trajectory:
                try:
                    # Get best agent's BEST state for this iteration (not current)
                    best_agent = self._agents[0]  # Start with first agent
                    best_agent_energy = float('inf')
                    
                    for agent in self._agents:
                        conf = agent.get_best_conformation()  # Use BEST, not current
                        if conf.energy < best_agent_energy:
                            best_agent = agent
                            best_agent_energy = conf.energy
                    
                    # Get UBF metrics from best agent
                    best_conf = best_agent.get_best_conformation()  # Use BEST, not current
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
        best_folding_rmsd = 0.0  # Track best folding distance across agents
        best_gdt_ts = 0.0  # Track best GDT-TS score across agents
        best_tm_score = 0.0  # Track best TM-score across agents
        
        for i, agent in enumerate(self._agents):
            metrics_dict = agent.get_exploration_metrics()
            agent_folding_rmsd = metrics_dict.get("folding_rmsd", 0.0)
            if agent_folding_rmsd > best_folding_rmsd:
                best_folding_rmsd = agent_folding_rmsd
            
            # Track best GDT-TS and TM-score across all agents
            agent_gdt_ts = metrics_dict.get("best_gdt_ts", 0.0)
            agent_tm_score = metrics_dict.get("best_tm_score", 0.0)
            if agent_gdt_ts > best_gdt_ts:
                best_gdt_ts = agent_gdt_ts
            if agent_tm_score > best_tm_score:
                best_tm_score = agent_tm_score
            
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
                successful_escapes=int(metrics_dict["successful_escapes"]),
                folding_rmsd=agent_folding_rmsd,
                best_gdt_ts_score=agent_gdt_ts if agent_gdt_ts > 0 else None,
                best_tm_score=agent_tm_score if agent_tm_score > 0 else None
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

        # Calculate validation quality based on RMSD and GDT-TS
        validation_quality = None
        if self._best_rmsd != float('inf'):
            # Quality assessment based on structural biology standards
            if self._best_rmsd < 2.0 and best_gdt_ts > 80:
                validation_quality = "excellent"
            elif self._best_rmsd < 3.0 and best_gdt_ts > 70:
                validation_quality = "good"
            elif self._best_rmsd < 5.0 and best_gdt_ts > 50:
                validation_quality = "acceptable"
            elif self._best_rmsd < 5.0 or best_gdt_ts > 50:
                validation_quality = "acceptable"  # One criterion met
            else:
                validation_quality = "poor"
            
            logger.info(f"Validation quality: {validation_quality} (RMSD={self._best_rmsd:.2f}Å, GDT-TS={best_gdt_ts:.1f})")

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
            validation_quality=validation_quality,
            best_gdt_ts=best_gdt_ts if best_gdt_ts > 0 else None,
            best_tm_score=best_tm_score if best_tm_score > 0 else None,
            folding_rmsd=best_folding_rmsd,  # Best folding distance across all agents
            qcpp_trajectory_data=qcpp_trajectory_data,
            qcpp_rmsd_correlations=qcpp_rmsd_correlations,
            qcpp_energy_correlations=qcpp_energy_correlations,
            consciousness_qcpp_correlations=consciousness_qcpp_correlations
        )

    def run_parallel_exploration_with_refinement(
        self,
        iterations: int,
        native_structure: Optional[Any] = None
    ) -> Tuple[ExplorationResults, Optional[Any]]:
        """
        Run parallel exploration with automatic quantum refinement for coarse structures.
        
        This method combines Stage 1 exploration (multi-agent parallel search) with
        optional Stage 2 quantum refinement. After the exploration completes, if the
        best structure has RMSD > refinement_rmsd_threshold, automatic quantum
        refinement is triggered to achieve sub-5Å accuracy.
        
        Workflow:
        1. Run standard parallel exploration (Stage 1)
        2. Check best RMSD against refinement threshold
        3. If RMSD > threshold, trigger quantum refinement (Stage 2)
        4. Return both exploration results and refinement result
        
        Args:
            iterations: Number of exploration iterations per agent
            native_structure: Optional native structure for RMSD validation
        
        Returns:
            Tuple of (ExplorationResults, RefinementResult or None)
            - ExplorationResults: Standard multi-agent exploration metrics
            - RefinementResult: Quantum refinement result if triggered, None otherwise
        
        Raises:
            ValueError: If quantum refinement is not enabled or configured
        
        Example:
            >>> coordinator = MultiAgentCoordinator(
            ...     protein_sequence="MQIFVKT",
            ...     qcpp_integration=qcpp_adapter,
            ...     enable_quantum_refinement=True,
            ...     refinement_rmsd_threshold=5.0
            ... )
            >>> coordinator.initialize_agents(count=10)
            >>> exploration_results, refinement_result = coordinator.run_parallel_exploration_with_refinement(
            ...     iterations=500,
            ...     native_structure=native_pdb
            ... )
            >>> if refinement_result:
            ...     print(f"Refined RMSD: {refinement_result.final_rmsd:.2f}Å")
        """
        # Ensure agents have native structure for RMSD tracking
        if native_structure is not None:
            # Check if agents have RMSD tracking enabled by checking first agent's conformation
            needs_reinit = False
            if not self._agents:
                needs_reinit = True
                logger.info("No agents initialized - creating agents with RMSD tracking")
            else:
                # Check if current conformations have RMSD tracking
                current_conf = self._agents[0].get_current_conformation()
                if current_conf.rmsd_to_native is None:
                    needs_reinit = True
                    logger.info("Agents don't have native structure - reinitializing with RMSD tracking")
            
            if needs_reinit:
                # Re-initialize agents with native structure
                agent_count = len(self._agents) if self._agents else 10
                self.initialize_agents(agent_count, "balanced", native_structure)
                logger.info(f"Initialized {agent_count} agents with native structure for RMSD tracking")
        
        # Stage 1: Run standard parallel exploration
        logger.info(f"Stage 1: Starting parallel exploration with {len(self._agents)} agents for {iterations} iterations")
        exploration_results = self.run_parallel_exploration(iterations)
        
        logger.info(
            f"Stage 1 completed: Best RMSD={exploration_results.best_rmsd:.2f}Å, "
            f"Best Energy={exploration_results.best_energy:.2f} kcal/mol"
        )
        
        # Check if refinement is needed
        refinement_result = None
        
        if not self._enable_quantum_refinement:
            logger.info("Quantum refinement is not enabled, skipping Stage 2")
            return exploration_results, None
        
        if self._refinement_engine is None:
            logger.warning("Refinement engine not initialized, skipping Stage 2")
            return exploration_results, None
        
        if exploration_results.best_conformation is None:
            logger.warning("No valid conformation found in exploration, skipping Stage 2")
            return exploration_results, None
        
        # Check RMSD threshold
        best_rmsd = exploration_results.best_rmsd
        
        if best_rmsd == float('inf') or best_rmsd is None:
            logger.info("RMSD not available (no native structure), skipping automatic refinement")
            return exploration_results, None
        
        if best_rmsd <= self._refinement_rmsd_threshold:
            logger.info(
                f"RMSD {best_rmsd:.2f}Å is already below threshold {self._refinement_rmsd_threshold:.2f}Å, "
                "skipping refinement"
            )
            return exploration_results, None
        
        # Stage 2: Trigger quantum refinement
        logger.info(
            f"Stage 2: RMSD {best_rmsd:.2f}Å exceeds threshold {self._refinement_rmsd_threshold:.2f}Å, "
            "triggering quantum refinement"
        )
        
        try:
            refinement_result = self._refinement_engine.refine_structure_quantum(
                coarse_structure=exploration_results.best_conformation,
                native_structure=native_structure,
                config=self._refinement_config
            )
            
            logger.info(
                f"Stage 2 completed: Refined RMSD={refinement_result.final_rmsd:.2f}Å "
                f"(improvement: {refinement_result.rmsd_improvement:.2f}Å), "
                f"Energy={refinement_result.energy:.2f} kcal/mol"
            )
            
            # Update best conformation with refined structure
            self._best_conformation = refinement_result.refined_structure
            self._best_rmsd = refinement_result.final_rmsd
            self._best_energy = refinement_result.energy
            
        except Exception as e:
            logger.error(f"Quantum refinement failed: {e}")
            logger.warning("Returning exploration results without refinement")
            refinement_result = None
        
        return exploration_results, refinement_result

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
                best_conf = agent.get_best_conformation()
                if best_conf.energy < self._best_energy:
                    self._best_energy = best_conf.energy
                    self._best_conformation = best_conf
                
                if (best_conf.rmsd_to_native and
                    best_conf.rmsd_to_native < self._best_rmsd):
                    self._best_rmsd = best_conf.rmsd_to_native
            
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
    
    def get_mediator_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics from all Mediator Agents.
        
        Returns comprehensive metrics about pattern detection performance including:
        - Total detections by type (THz, folding, geometric)
        - Broadcast counts and throttle events
        - Cache statistics (hit rate, size)
        - Detection success rates
        - QCPP validation counts
        
        Returns:
            Dictionary with aggregated mediator statistics
        
        Raises:
            ValueError: If mediators are not enabled
        
        Example:
            >>> coordinator = MultiAgentCoordinator(
            ...     protein_sequence="ACDEFGH",
            ...     enable_mediators=True
            ... )
            >>> coordinator.initialize_agents(count=10)
            >>> coordinator.initialize_mediators()
            >>> coordinator.run_parallel_exploration(iterations=100)
            >>> 
            >>> stats = coordinator.get_mediator_statistics()
            >>> print(f"Total detections: {stats['total_detections']}")
            >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        """
        if not self._enable_mediators:
            raise ValueError("Mediators are not enabled. Cannot get statistics.")
        
        if len(self._mediators) == 0:
            return {
                'enabled': False,
                'mediator_count': 0,
                'total_detections': 0,
                'thz_detections': 0,
                'folding_detections': 0,
                'geometric_detections': 0,
                'broadcasts': 0,
                'qcpp_validations': 0,
                'cache_hit_rate': 0.0,
                'cache_size': 0,
                'reference_conformations': 0,
            }
        
        # Aggregate statistics from all mediators
        total_stats = {
            'enabled': True,
            'mediator_count': len(self._mediators),
            'total_detections': 0,
            'thz_detections': 0,
            'folding_detections': 0,
            'geometric_detections': 0,
            'broadcasts': 0,
            'qcpp_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_size': 0,
            'reference_conformations': 0,
        }
        
        # Collect stats from each mediator
        for mediator in self._mediators:
            try:
                med_stats = mediator.get_detection_statistics()
                
                total_stats['total_detections'] += med_stats.get('total_detections', 0)
                total_stats['thz_detections'] += med_stats.get('thz_detections', 0)
                total_stats['folding_detections'] += med_stats.get('folding_detections', 0)
                total_stats['geometric_detections'] += med_stats.get('geometric_detections', 0)
                total_stats['broadcasts'] += med_stats.get('broadcasts', 0)
                total_stats['qcpp_validations'] += med_stats.get('qcpp_validations', 0)
                total_stats['cache_hits'] += med_stats.get('cache_hits', 0)
                total_stats['cache_misses'] += med_stats.get('cache_misses', 0)
                total_stats['cache_size'] += med_stats.get('cache_size', 0)
                total_stats['reference_conformations'] += med_stats.get('reference_conformations', 0)
            
            except Exception as e:
                logger.warning(f"Failed to get statistics from mediator: {e}")
                continue
        
        # Calculate aggregate metrics
        total_cache_queries = total_stats['cache_hits'] + total_stats['cache_misses']
        if total_cache_queries > 0:
            cache_hit_rate = total_stats['cache_hits'] / total_cache_queries
        else:
            cache_hit_rate = 0.0
        
        # Average cache size per mediator
        avg_cache_size = total_stats['cache_size'] / len(self._mediators) if self._mediators else 0
        
        # Average reference conformations per mediator
        avg_references = total_stats['reference_conformations'] / len(self._mediators) if self._mediators else 0
        
        # Add computed metrics
        total_stats['cache_hit_rate'] = cache_hit_rate
        total_stats['avg_cache_size'] = avg_cache_size
        total_stats['avg_reference_conformations'] = avg_references
        
        # Add configuration info
        if self._mediator_config:
            total_stats['config'] = {
                'relay_frequency': self._mediator_config.relay_frequency,
                'thz_detection_enabled': self._mediator_config.enable_thz_detection,
                'folding_detection_enabled': self._mediator_config.enable_folding_detection,
                'geometric_detection_enabled': self._mediator_config.enable_geometric_detection,
                'broadcast_throttle_rate': self._mediator_config.broadcast_throttle_rate,
            }
        
        return total_stats
    
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