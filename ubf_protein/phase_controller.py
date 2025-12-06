"""
Phase Controller - Exploration Phase Management

This module implements the PhaseController that coordinates transitions between
exploration phases based on structural metrics rather than just iteration count.

Phases:
  1. FREE_EXPLORATION: All moves allowed, discover initial structure
  2. LOCAL_ANCHORING: Lock stable secondary structure, continue exploring
  3. TERTIARY_PACKING: Focus on bringing anchored elements together
  4. REFINEMENT: Fine-tune with minimal local moves only

The phase transitions are metric-driven:
  - FREE → LOCAL: When significant secondary structure is detected
  - LOCAL → TERTIARY: When anchoring reaches threshold percentage
  - TERTIARY → REFINEMENT: When energy/RMSD improvements plateau

Author: UBF Protein System
Date: December 5, 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ExplorationPhase(Enum):
    """Exploration phases with increasing constraint."""
    FREE_EXPLORATION = "free_exploration"
    LOCAL_ANCHORING = "local_anchoring"
    TERTIARY_PACKING = "tertiary_packing"
    REFINEMENT = "refinement"


@dataclass
class PhaseMetrics:
    """
    Metrics tracked for phase transition decisions.
    
    These metrics are updated each iteration and used to determine
    when to transition between phases.
    """
    # Iteration tracking
    current_iteration: int = 0
    iterations_in_phase: int = 0
    
    # Energy metrics
    current_energy: float = float('inf')
    best_energy: float = float('inf')
    energy_improvement_rate: float = 0.0  # Energy change per iteration (rolling avg)
    
    # RMSD metrics (if native structure available)
    current_rmsd: Optional[float] = None
    best_rmsd: Optional[float] = None
    rmsd_improvement_rate: float = 0.0
    
    # Secondary structure metrics
    helix_percentage: float = 0.0
    sheet_percentage: float = 0.0
    structured_percentage: float = 0.0  # helix + sheet
    
    # Anchoring metrics
    anchoring_percentage: float = 0.0
    soft_anchored: int = 0
    medium_anchored: int = 0
    hard_anchored: int = 0
    locked: int = 0
    
    # Stability metrics
    energy_stable_iterations: int = 0  # Iterations without significant improvement
    rmsd_stable_iterations: int = 0
    structure_stable_iterations: int = 0  # Iterations with stable SS


@dataclass
class PhaseControllerConfig:
    """Configuration for phase transitions.
    
    Note: iterations_in_phase counts update_metrics() calls, not raw iterations.
    If HierarchicalFoldingManager uses structure_update_frequency=5, then
    200 raw iterations = 40 update calls.
    """
    
    # Phase 1 → Phase 2 (Free → Local Anchoring)
    # Transition when secondary structure is detected
    min_structured_pct_for_anchoring: float = 15.0  # % helix+sheet to start anchoring
    min_iterations_free: int = 20  # Min update calls in free phase (was 50)
    
    # Phase 2 → Phase 3 (Local Anchoring → Tertiary Packing)
    # Transition when enough residues are anchored
    min_anchoring_pct_for_packing: float = 30.0  # % anchored to start packing
    min_medium_anchored_pct: float = 15.0  # % at medium+ strength
    min_iterations_anchoring: int = 40  # Min update calls in anchoring phase (was 100)
    
    # Phase 3 → Phase 4 (Tertiary Packing → Refinement)
    # Transition when improvements plateau
    min_anchoring_pct_for_refinement: float = 50.0  # % anchored to enter refinement
    energy_plateau_threshold: float = 0.5  # kcal/mol improvement threshold
    energy_plateau_iterations: int = 20  # Iterations without improvement (was 50)
    min_iterations_packing: int = 40  # Min update calls in packing phase (was 100)
    
    # Refinement settings
    refinement_move_scale: float = 0.2  # Scale factor for move magnitudes
    max_refinement_iterations: int = 200  # Cap on refinement
    
    # Rollback conditions (if things get worse)
    enable_phase_rollback: bool = True
    rollback_energy_threshold: float = 20.0  # Energy increase to trigger rollback
    rollback_rmsd_threshold: float = 5.0  # RMSD increase to trigger rollback


class PhaseController:
    """
    Controls exploration phase transitions based on structural metrics.
    
    The controller monitors key metrics and determines when the exploration
    should transition from one phase to the next. This enables adaptive
    exploration that responds to the protein's folding progress.
    
    Usage:
        >>> controller = PhaseController(sequence_length=76)
        >>> 
        >>> # During exploration loop
        >>> for iteration in range(max_iterations):
        ...     # Update metrics
        ...     controller.update_metrics(
        ...         iteration=iteration,
        ...         energy=current_energy,
        ...         rmsd=current_rmsd,
        ...         anchoring_pct=anchor_manager.get_anchoring_percentage()
        ...     )
        ...     
        ...     # Check for phase transition
        ...     if controller.should_transition():
        ...         controller.transition_to_next_phase()
        ...     
        ...     # Get current phase constraints
        ...     phase = controller.current_phase
        ...     move_scale = controller.get_move_scale()
    """
    
    def __init__(
        self,
        sequence_length: int,
        config: Optional[PhaseControllerConfig] = None
    ):
        """
        Initialize phase controller.
        
        Args:
            sequence_length: Number of residues in the protein
            config: Configuration settings (uses defaults if None)
        """
        self.sequence_length = sequence_length
        self.config = config or PhaseControllerConfig()
        
        # Current state
        self._current_phase = ExplorationPhase.FREE_EXPLORATION
        self._metrics = PhaseMetrics()
        
        # History for rate calculations
        self._energy_history: List[float] = []
        self._rmsd_history: List[float] = []
        self._structured_pct_history: List[float] = []
        self._history_window = 20  # Rolling window size
        
        # Phase transition history
        self._phase_transitions: List[Dict[str, Any]] = []
        self._phase_start_energy: float = float('inf')
        self._phase_start_rmsd: Optional[float] = None
        
        # Callbacks for phase transitions
        self._transition_callbacks: List[Callable[[ExplorationPhase, ExplorationPhase], None]] = []
        
        logger.info(f"PhaseController initialized for {sequence_length} residues")
    
    @property
    def current_phase(self) -> ExplorationPhase:
        """Get the current exploration phase."""
        return self._current_phase
    
    @property
    def metrics(self) -> PhaseMetrics:
        """Get current metrics."""
        return self._metrics
    
    def register_transition_callback(
        self,
        callback: Callable[[ExplorationPhase, ExplorationPhase], None]
    ) -> None:
        """
        Register a callback to be called on phase transitions.
        
        Args:
            callback: Function(old_phase, new_phase) called on transition
        """
        self._transition_callbacks.append(callback)
    
    def update_metrics(
        self,
        iteration: int,
        energy: float,
        rmsd: Optional[float] = None,
        helix_pct: float = 0.0,
        sheet_pct: float = 0.0,
        anchoring_pct: float = 0.0,
        anchor_distribution: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Update metrics with current state.
        
        Call this each iteration to track progress.
        
        Args:
            iteration: Current iteration number
            energy: Current best energy (kcal/mol)
            rmsd: Current RMSD to native (Å), if available
            helix_pct: Percentage of residues in helix
            sheet_pct: Percentage of residues in sheet
            anchoring_pct: Percentage of residues anchored
            anchor_distribution: Dict with anchor strength counts
        """
        m = self._metrics
        
        # Update iteration counters
        m.current_iteration = iteration
        m.iterations_in_phase += 1
        
        # Update energy metrics
        old_best = m.best_energy
        m.current_energy = energy
        if energy < m.best_energy:
            m.best_energy = energy
        
        # Track energy improvement rate
        self._energy_history.append(energy)
        if len(self._energy_history) > self._history_window:
            self._energy_history.pop(0)
        
        if len(self._energy_history) >= 2:
            m.energy_improvement_rate = (
                self._energy_history[0] - self._energy_history[-1]
            ) / len(self._energy_history)
        
        # Check energy stability
        if old_best - energy < self.config.energy_plateau_threshold:
            m.energy_stable_iterations += 1
        else:
            m.energy_stable_iterations = 0
        
        # Update RMSD metrics
        if rmsd is not None:
            old_best_rmsd = m.best_rmsd
            m.current_rmsd = rmsd
            if m.best_rmsd is None or rmsd < m.best_rmsd:
                m.best_rmsd = rmsd
            
            self._rmsd_history.append(rmsd)
            if len(self._rmsd_history) > self._history_window:
                self._rmsd_history.pop(0)
            
            if len(self._rmsd_history) >= 2:
                m.rmsd_improvement_rate = (
                    self._rmsd_history[0] - self._rmsd_history[-1]
                ) / len(self._rmsd_history)
            
            # Check RMSD stability
            if old_best_rmsd is not None and old_best_rmsd - rmsd < 0.1:
                m.rmsd_stable_iterations += 1
            else:
                m.rmsd_stable_iterations = 0
        
        # Update secondary structure metrics
        m.helix_percentage = helix_pct
        m.sheet_percentage = sheet_pct
        m.structured_percentage = helix_pct + sheet_pct
        
        self._structured_pct_history.append(m.structured_percentage)
        if len(self._structured_pct_history) > self._history_window:
            self._structured_pct_history.pop(0)
        
        # Check structure stability
        if len(self._structured_pct_history) >= 5:
            recent = self._structured_pct_history[-5:]
            variance = max(recent) - min(recent)
            if variance < 5.0:  # Less than 5% change
                m.structure_stable_iterations += 1
            else:
                m.structure_stable_iterations = 0
        
        # Update anchoring metrics
        m.anchoring_percentage = anchoring_pct
        
        if anchor_distribution:
            m.soft_anchored = anchor_distribution.get('soft', 0)
            m.medium_anchored = anchor_distribution.get('medium', 0)
            m.hard_anchored = anchor_distribution.get('hard', 0)
            m.locked = anchor_distribution.get('locked', 0)
    
    def should_transition(self) -> bool:
        """
        Check if we should transition to the next phase.
        
        Returns:
            True if transition conditions are met
        """
        m = self._metrics
        c = self.config
        
        if self._current_phase == ExplorationPhase.FREE_EXPLORATION:
            # Transition to LOCAL_ANCHORING when secondary structure detected
            return (
                m.iterations_in_phase >= c.min_iterations_free and
                m.structured_percentage >= c.min_structured_pct_for_anchoring and
                m.structure_stable_iterations >= 10
            )
        
        elif self._current_phase == ExplorationPhase.LOCAL_ANCHORING:
            # Transition to TERTIARY_PACKING when enough anchoring
            medium_plus = m.medium_anchored + m.hard_anchored + m.locked
            medium_plus_pct = (medium_plus / self.sequence_length) * 100.0
            
            return (
                m.iterations_in_phase >= c.min_iterations_anchoring and
                m.anchoring_percentage >= c.min_anchoring_pct_for_packing and
                medium_plus_pct >= c.min_medium_anchored_pct
            )
        
        elif self._current_phase == ExplorationPhase.TERTIARY_PACKING:
            # Transition to REFINEMENT when improvements plateau
            return (
                m.iterations_in_phase >= c.min_iterations_packing and
                m.anchoring_percentage >= c.min_anchoring_pct_for_refinement and
                m.energy_stable_iterations >= c.energy_plateau_iterations
            )
        
        elif self._current_phase == ExplorationPhase.REFINEMENT:
            # No further transitions
            return False
        
        return False
    
    def should_rollback(self) -> bool:
        """
        Check if we should roll back to a previous phase.
        
        This can happen if energy or RMSD gets significantly worse,
        indicating the phase transition was premature.
        
        Returns:
            True if rollback conditions are met
        """
        if not self.config.enable_phase_rollback:
            return False
        
        if self._current_phase == ExplorationPhase.FREE_EXPLORATION:
            return False  # Can't roll back from first phase
        
        m = self._metrics
        c = self.config
        
        # Check energy regression
        energy_regression = m.current_energy - self._phase_start_energy
        if energy_regression > c.rollback_energy_threshold:
            logger.warning(
                f"Energy regressed by {energy_regression:.1f} kcal/mol, "
                f"considering rollback"
            )
            return True
        
        # Check RMSD regression
        if m.current_rmsd is not None and self._phase_start_rmsd is not None:
            rmsd_regression = m.current_rmsd - self._phase_start_rmsd
            if rmsd_regression > c.rollback_rmsd_threshold:
                logger.warning(
                    f"RMSD regressed by {rmsd_regression:.1f}Å, "
                    f"considering rollback"
                )
                return True
        
        return False
    
    def transition_to_next_phase(self) -> ExplorationPhase:
        """
        Transition to the next exploration phase.
        
        Returns:
            The new phase
        """
        old_phase = self._current_phase
        
        phase_order = [
            ExplorationPhase.FREE_EXPLORATION,
            ExplorationPhase.LOCAL_ANCHORING,
            ExplorationPhase.TERTIARY_PACKING,
            ExplorationPhase.REFINEMENT,
        ]
        
        current_idx = phase_order.index(self._current_phase)
        if current_idx < len(phase_order) - 1:
            self._current_phase = phase_order[current_idx + 1]
        
        self._record_transition(old_phase, self._current_phase)
        
        return self._current_phase
    
    def rollback_phase(self) -> ExplorationPhase:
        """
        Roll back to the previous phase.
        
        Returns:
            The new (previous) phase
        """
        old_phase = self._current_phase
        
        phase_order = [
            ExplorationPhase.FREE_EXPLORATION,
            ExplorationPhase.LOCAL_ANCHORING,
            ExplorationPhase.TERTIARY_PACKING,
            ExplorationPhase.REFINEMENT,
        ]
        
        current_idx = phase_order.index(self._current_phase)
        if current_idx > 0:
            self._current_phase = phase_order[current_idx - 1]
        
        self._record_transition(old_phase, self._current_phase, is_rollback=True)
        
        return self._current_phase
    
    def _record_transition(
        self,
        old_phase: ExplorationPhase,
        new_phase: ExplorationPhase,
        is_rollback: bool = False
    ) -> None:
        """Record a phase transition."""
        m = self._metrics
        
        transition_record = {
            'iteration': m.current_iteration,
            'from_phase': old_phase.value,
            'to_phase': new_phase.value,
            'is_rollback': is_rollback,
            'metrics_at_transition': {
                'energy': m.current_energy,
                'best_energy': m.best_energy,
                'rmsd': m.current_rmsd,
                'best_rmsd': m.best_rmsd,
                'structured_pct': m.structured_percentage,
                'anchoring_pct': m.anchoring_percentage,
            },
            'iterations_in_old_phase': m.iterations_in_phase,
        }
        
        self._phase_transitions.append(transition_record)
        
        # Reset phase-specific counters
        m.iterations_in_phase = 0
        self._phase_start_energy = m.current_energy
        self._phase_start_rmsd = m.current_rmsd
        
        logger.info(
            f"Phase transition: {old_phase.value} → {new_phase.value} "
            f"{'(ROLLBACK)' if is_rollback else ''} "
            f"at iteration {m.current_iteration}"
        )
        
        # Call registered callbacks
        for callback in self._transition_callbacks:
            try:
                callback(old_phase, new_phase)
            except Exception as e:
                logger.warning(f"Transition callback failed: {e}")
    
    def get_move_scale(self) -> float:
        """
        Get the move magnitude scale factor for the current phase.
        
        Returns:
            Scale factor (0.0-1.0) to apply to move magnitudes
        """
        scales = {
            ExplorationPhase.FREE_EXPLORATION: 1.0,
            ExplorationPhase.LOCAL_ANCHORING: 0.8,
            ExplorationPhase.TERTIARY_PACKING: 0.5,
            ExplorationPhase.REFINEMENT: self.config.refinement_move_scale,
        }
        return scales[self._current_phase]
    
    def get_move_type_weights(self) -> Dict[str, float]:
        """
        Get weight adjustments for different move types in the current phase.
        
        Returns:
            Dict mapping move type names to weight multipliers
        """
        if self._current_phase == ExplorationPhase.FREE_EXPLORATION:
            # All moves equal
            return {
                'backbone_rotation': 1.0,
                'sidechain_adjust': 1.0,
                'helix_formation': 1.0,
                'sheet_formation': 1.0,
                'turn_formation': 1.0,
                'hydrophobic_collapse': 1.5,  # Slight boost for collapse
                'pivot_rotation': 1.0,
                'energy_minimization': 0.8,
            }
        
        elif self._current_phase == ExplorationPhase.LOCAL_ANCHORING:
            # Boost local structure formation
            return {
                'backbone_rotation': 1.0,
                'sidechain_adjust': 1.0,
                'helix_formation': 1.5,  # Boost
                'sheet_formation': 1.5,  # Boost
                'turn_formation': 1.2,
                'hydrophobic_collapse': 1.0,
                'pivot_rotation': 0.8,  # Reduce large moves
                'energy_minimization': 1.0,
            }
        
        elif self._current_phase == ExplorationPhase.TERTIARY_PACKING:
            # Focus on bringing structures together
            return {
                'backbone_rotation': 0.8,
                'sidechain_adjust': 1.2,
                'helix_formation': 0.5,  # Already formed
                'sheet_formation': 0.5,  # Already formed
                'turn_formation': 0.8,
                'hydrophobic_collapse': 2.0,  # Critical for packing
                'pivot_rotation': 1.5,  # Important for topology
                'energy_minimization': 1.2,
            }
        
        elif self._current_phase == ExplorationPhase.REFINEMENT:
            # Only small local adjustments
            return {
                'backbone_rotation': 0.5,
                'sidechain_adjust': 2.0,  # Main refinement move
                'helix_formation': 0.1,
                'sheet_formation': 0.1,
                'turn_formation': 0.3,
                'hydrophobic_collapse': 0.2,
                'pivot_rotation': 0.1,  # Avoid big changes
                'energy_minimization': 2.0,  # Main refinement move
            }
        
        return {}
    
    def is_in_refinement(self) -> bool:
        """Check if we're in the refinement phase."""
        return self._current_phase == ExplorationPhase.REFINEMENT
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """
        Get a summary of phase controller state.
        
        Returns:
            Dictionary with phase information
        """
        m = self._metrics
        
        return {
            'current_phase': self._current_phase.value,
            'iterations_in_phase': m.iterations_in_phase,
            'total_iterations': m.current_iteration,
            'metrics': {
                'current_energy': m.current_energy,
                'best_energy': m.best_energy,
                'current_rmsd': m.current_rmsd,
                'best_rmsd': m.best_rmsd,
                'structured_pct': m.structured_percentage,
                'anchoring_pct': m.anchoring_percentage,
                'energy_improvement_rate': m.energy_improvement_rate,
                'energy_stable_iterations': m.energy_stable_iterations,
            },
            'move_scale': self.get_move_scale(),
            'transition_history': self._phase_transitions,
        }
    
    def print_summary(self) -> None:
        """Print a human-readable summary."""
        summary = self.get_phase_summary()
        m = self._metrics
        
        print("\n" + "=" * 70)
        print("PHASE CONTROLLER STATUS")
        print("=" * 70)
        
        phase_emoji = {
            'free_exploration': '🔓',
            'local_anchoring': '🔗',
            'tertiary_packing': '📦',
            'refinement': '✨',
        }
        
        phase = summary['current_phase']
        print(f"\n{phase_emoji.get(phase, '❓')} Current Phase: {phase.upper()}")
        print(f"  Iterations in phase: {summary['iterations_in_phase']}")
        print(f"  Total iterations: {summary['total_iterations']}")
        print(f"  Move scale: {summary['move_scale']:.2f}")
        
        print(f"\n📊 Key Metrics:")
        metrics = summary['metrics']
        print(f"  Energy: {metrics['current_energy']:.1f} (best: {metrics['best_energy']:.1f})")
        if metrics['current_rmsd'] is not None:
            print(f"  RMSD: {metrics['current_rmsd']:.2f}Å (best: {metrics['best_rmsd']:.2f}Å)")
        print(f"  Structured: {metrics['structured_pct']:.1f}%")
        print(f"  Anchored: {metrics['anchoring_pct']:.1f}%")
        print(f"  Energy stable: {metrics['energy_stable_iterations']} iterations")
        
        if self._phase_transitions:
            print(f"\n📜 Transition History:")
            for t in self._phase_transitions:
                arrow = "⬅️" if t['is_rollback'] else "➡️"
                print(f"  {arrow} {t['from_phase']} → {t['to_phase']} @ iter {t['iteration']}")
        
        print("=" * 70 + "\n")


def create_phase_controller(
    sequence_length: int,
    fast_mode: bool = False
) -> PhaseController:
    """
    Factory function to create a phase controller.
    
    Args:
        sequence_length: Number of residues
        fast_mode: If True, use faster transition thresholds (for testing)
    
    Returns:
        Configured PhaseController
    """
    if fast_mode:
        config = PhaseControllerConfig(
            min_iterations_free=10,
            min_iterations_anchoring=20,
            min_iterations_packing=20,
            min_structured_pct_for_anchoring=10.0,
            min_anchoring_pct_for_packing=20.0,
            min_anchoring_pct_for_refinement=35.0,
            energy_plateau_iterations=10,
        )
    else:
        config = PhaseControllerConfig()
    
    return PhaseController(sequence_length, config)
