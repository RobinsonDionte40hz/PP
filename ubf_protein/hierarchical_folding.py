"""
Hierarchical Folding Integration Module

This module provides integration functions to wire the ResidueAnchorManager
and PhaseController into the existing protein exploration system.

Rather than heavily modifying existing code, this module provides:
1. A HierarchicalFoldingManager that coordinates anchoring and phases
2. Helper functions to integrate with ProteinAgent move execution
3. Secondary structure detection from conformations

Author: UBF Protein System
Date: December 5, 2025
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
import logging
import math

from .residue_anchor_manager import (
    ResidueAnchorManager, 
    AnchorManagerConfig,
    StructuralState,
    AnchorStrength,
    create_anchor_manager
)
from .phase_controller import (
    PhaseController,
    PhaseControllerConfig,
    ExplorationPhase,
    create_phase_controller
)

if TYPE_CHECKING:
    from .models import Conformation, ConformationalMove

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalFoldingConfig:
    """Configuration for hierarchical folding system."""
    
    # Enable/disable components
    enable_anchoring: bool = True
    enable_phase_control: bool = True
    
    # Anchoring aggressiveness
    aggressive_anchoring: bool = False
    
    # Phase control speed
    fast_phases: bool = False
    
    # Secondary structure detection
    helix_phi_range: Tuple[float, float] = (-80.0, -40.0)
    helix_psi_range: Tuple[float, float] = (-60.0, -20.0)
    sheet_phi_range: Tuple[float, float] = (-150.0, -100.0)
    sheet_psi_range: Tuple[float, float] = (100.0, 170.0)
    
    # Update frequency
    structure_update_frequency: int = 5  # Update every N iterations


class HierarchicalFoldingManager:
    """
    Coordinates hierarchical folding through anchoring and phase control.
    
    This manager integrates the ResidueAnchorManager and PhaseController
    to provide progressive search space confinement during exploration.
    
    Usage:
        >>> manager = HierarchicalFoldingManager(sequence_length=76)
        >>> 
        >>> # During exploration loop
        >>> for iteration in range(max_iterations):
        ...     # Update with current state
        ...     manager.update(
        ...         iteration=iteration,
        ...         conformation=current_conf,
        ...         energy=current_energy,
        ...         rmsd=current_rmsd
        ...     )
        ...     
        ...     # Check if move is allowed (respects anchoring)
        ...     if manager.is_move_allowed(move, proposed_angles):
        ...         execute_move()
        ...     
        ...     # Get phase-specific move weights
        ...     weights = manager.get_move_type_weights()
    """
    
    def __init__(
        self,
        sequence_length: int,
        config: Optional[HierarchicalFoldingConfig] = None
    ):
        """
        Initialize hierarchical folding manager.
        
        Args:
            sequence_length: Number of residues in the protein
            config: Configuration settings (uses defaults if None)
        """
        self.sequence_length = sequence_length
        self.config = config or HierarchicalFoldingConfig()
        
        # Initialize anchor manager
        if self.config.enable_anchoring:
            self.anchor_manager = create_anchor_manager(
                sequence_length=sequence_length,
                aggressive=self.config.aggressive_anchoring
            )
        else:
            self.anchor_manager = None
        
        # Initialize phase controller
        if self.config.enable_phase_control:
            self.phase_controller = create_phase_controller(
                sequence_length=sequence_length,
                fast_mode=self.config.fast_phases
            )
        else:
            self.phase_controller = None
        
        # Tracking
        self._last_update_iteration = -1
        self._structure_history: List[str] = []
        
        logger.info(
            f"HierarchicalFoldingManager initialized for {sequence_length} residues "
            f"(anchoring={self.config.enable_anchoring}, phases={self.config.enable_phase_control})"
        )
    
    def update(
        self,
        iteration: int,
        conformation: 'Conformation',
        energy: float,
        rmsd: Optional[float] = None
    ) -> None:
        """
        Update the hierarchical folding state.
        
        Call this each iteration to update anchoring and phase control.
        
        Args:
            iteration: Current iteration number
            conformation: Current best conformation
            energy: Current energy
            rmsd: Current RMSD to native (if available)
        """
        # Check if we should update (based on frequency)
        if (iteration - self._last_update_iteration) < self.config.structure_update_frequency:
            return
        
        self._last_update_iteration = iteration
        
        # Detect secondary structure from conformation
        ss_string, helix_pct, sheet_pct = self._detect_secondary_structure(conformation)
        
        # Update anchor manager
        if self.anchor_manager is not None:
            self.anchor_manager.set_iteration(iteration)
            
            # Update each residue's state based on detected structure
            phi_angles = conformation.phi_angles if hasattr(conformation, 'phi_angles') else None
            psi_angles = conformation.psi_angles if hasattr(conformation, 'psi_angles') else None
            
            if phi_angles and psi_angles:
                self.anchor_manager.update_from_secondary_structure(
                    ss_string=ss_string,
                    phi_angles=phi_angles,
                    psi_angles=psi_angles,
                    base_confidence=0.8
                )
            
            # Apply confidence decay periodically
            if iteration % 10 == 0:
                self.anchor_manager.decay_confidence()
        
        # Update phase controller
        if self.phase_controller is not None:
            anchor_pct = 0.0
            anchor_dist = None
            
            if self.anchor_manager is not None:
                anchor_pct = self.anchor_manager.get_anchoring_percentage()
                summary = self.anchor_manager.get_anchoring_summary()
                anchor_dist = summary['anchor_distribution']
            
            self.phase_controller.update_metrics(
                iteration=iteration,
                energy=energy,
                rmsd=rmsd,
                helix_pct=helix_pct,
                sheet_pct=sheet_pct,
                anchoring_pct=anchor_pct,
                anchor_distribution=anchor_dist
            )
            
            # Check for phase transition
            if self.phase_controller.should_transition():
                old_phase = self.phase_controller.current_phase
                new_phase = self.phase_controller.transition_to_next_phase()
                self._on_phase_transition(old_phase, new_phase)
            
            # Check for rollback
            elif self.phase_controller.should_rollback():
                old_phase = self.phase_controller.current_phase
                new_phase = self.phase_controller.rollback_phase()
                self._on_phase_transition(old_phase, new_phase, is_rollback=True)
    
    def _detect_secondary_structure(
        self, 
        conformation: 'Conformation'
    ) -> Tuple[str, float, float]:
        """
        Detect secondary structure from conformation's phi/psi angles.
        
        Returns:
            Tuple of (ss_string, helix_percentage, sheet_percentage)
        """
        # Try to use existing secondary structure if available
        if hasattr(conformation, 'secondary_structure') and conformation.secondary_structure:
            ss_list = conformation.secondary_structure
            if isinstance(ss_list, str):
                ss_string = ss_list
            else:
                ss_string = ''.join(ss_list)
            
            helix_count = ss_string.count('H') + ss_string.count('G') + ss_string.count('I')
            sheet_count = ss_string.count('E') + ss_string.count('B')
            total = len(ss_string) if ss_string else 1
            
            return ss_string, (helix_count / total) * 100, (sheet_count / total) * 100
        
        # Fall back to phi/psi based detection
        if not hasattr(conformation, 'phi_angles') or not hasattr(conformation, 'psi_angles'):
            return 'C' * self.sequence_length, 0.0, 0.0
        
        phi_angles = conformation.phi_angles
        psi_angles = conformation.psi_angles
        
        if not phi_angles or not psi_angles:
            return 'C' * self.sequence_length, 0.0, 0.0
        
        ss_chars = []
        helix_count = 0
        sheet_count = 0
        
        for i in range(min(len(phi_angles), len(psi_angles))):
            phi = phi_angles[i]
            psi = psi_angles[i]
            
            # Check helix region
            if (self.config.helix_phi_range[0] <= phi <= self.config.helix_phi_range[1] and
                self.config.helix_psi_range[0] <= psi <= self.config.helix_psi_range[1]):
                ss_chars.append('H')
                helix_count += 1
            # Check sheet region
            elif (self.config.sheet_phi_range[0] <= phi <= self.config.sheet_phi_range[1] and
                  self.config.sheet_psi_range[0] <= psi <= self.config.sheet_psi_range[1]):
                ss_chars.append('E')
                sheet_count += 1
            else:
                ss_chars.append('C')
        
        # Pad if needed
        while len(ss_chars) < self.sequence_length:
            ss_chars.append('C')
        
        ss_string = ''.join(ss_chars[:self.sequence_length])
        total = self.sequence_length if self.sequence_length > 0 else 1
        
        return ss_string, (helix_count / total) * 100, (sheet_count / total) * 100
    
    def _on_phase_transition(
        self,
        old_phase: ExplorationPhase,
        new_phase: ExplorationPhase,
        is_rollback: bool = False
    ) -> None:
        """Handle phase transition events."""
        action = "rolled back to" if is_rollback else "transitioned to"
        logger.info(f"Hierarchical folding {action} {new_phase.value}")
        
        # Log anchoring status at transition
        if self.anchor_manager is not None:
            summary = self.anchor_manager.get_anchoring_summary()
            logger.info(
                f"  Anchoring: {summary['anchoring_percentage']:.1f}% "
                f"(soft={summary['anchor_distribution']['soft']}, "
                f"medium={summary['anchor_distribution']['medium']}, "
                f"hard={summary['anchor_distribution']['hard']}, "
                f"locked={summary['anchor_distribution']['locked']})"
            )
    
    def is_move_allowed(
        self,
        move: 'ConformationalMove',
        proposed_phi: Optional[List[float]] = None,
        proposed_psi: Optional[List[float]] = None
    ) -> bool:
        """
        Check if a move is allowed by anchoring constraints.
        
        Args:
            move: The proposed move
            proposed_phi: Proposed phi angles after move
            proposed_psi: Proposed psi angles after move
        
        Returns:
            True if move is allowed
        """
        if self.anchor_manager is None:
            return True
        
        # If no angle information, allow the move but log
        if proposed_phi is None or proposed_psi is None:
            return True
        
        # Check each target residue
        for residue_idx in move.target_residues:
            if residue_idx >= len(proposed_phi) or residue_idx >= len(proposed_psi):
                continue
            
            if not self.anchor_manager.is_move_allowed(
                residue_index=residue_idx,
                new_phi=proposed_phi[residue_idx],
                new_psi=proposed_psi[residue_idx]
            ):
                logger.debug(
                    f"Move {move.move_id} blocked by anchor constraint "
                    f"on residue {residue_idx}"
                )
                return False
        
        return True
    
    def constrain_move_angles(
        self,
        move: 'ConformationalMove',
        proposed_phi: List[float],
        proposed_psi: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Constrain proposed angles to respect anchoring.
        
        Instead of rejecting moves, this adjusts angles to fit constraints.
        
        Args:
            move: The move being applied
            proposed_phi: Proposed phi angles
            proposed_psi: Proposed psi angles
        
        Returns:
            Tuple of (constrained_phi, constrained_psi)
        """
        if self.anchor_manager is None:
            return proposed_phi, proposed_psi
        
        new_phi = list(proposed_phi)
        new_psi = list(proposed_psi)
        
        for residue_idx in move.target_residues:
            if residue_idx >= len(new_phi) or residue_idx >= len(new_psi):
                continue
            
            constrained_phi, constrained_psi = self.anchor_manager.constrain_angles(
                residue_index=residue_idx,
                proposed_phi=new_phi[residue_idx],
                proposed_psi=new_psi[residue_idx]
            )
            
            new_phi[residue_idx] = constrained_phi
            new_psi[residue_idx] = constrained_psi
        
        return new_phi, new_psi
    
    def get_move_scale(self) -> float:
        """Get the current move magnitude scale factor."""
        if self.phase_controller is not None:
            return self.phase_controller.get_move_scale()
        return 1.0
    
    def get_move_type_weights(self) -> Dict[str, float]:
        """Get phase-specific move type weight adjustments."""
        if self.phase_controller is not None:
            return self.phase_controller.get_move_type_weights()
        return {}
    
    def get_current_phase(self) -> Optional[ExplorationPhase]:
        """Get the current exploration phase."""
        if self.phase_controller is not None:
            return self.phase_controller.current_phase
        return None
    
    def get_anchoring_percentage(self) -> float:
        """Get current percentage of anchored residues."""
        if self.anchor_manager is not None:
            return self.anchor_manager.get_anchoring_percentage()
        return 0.0
    
    def get_free_residues(self) -> List[int]:
        """Get list of residue indices that are free to move."""
        if self.anchor_manager is not None:
            return self.anchor_manager.get_free_residues()
        return list(range(self.sequence_length))
    
    def get_anchored_residues(self) -> List[int]:
        """Get list of residue indices that are anchored."""
        if self.anchor_manager is not None:
            return self.anchor_manager.get_anchored_residues()
        return []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of hierarchical folding state."""
        summary = {
            'sequence_length': self.sequence_length,
            'config': {
                'enable_anchoring': self.config.enable_anchoring,
                'enable_phase_control': self.config.enable_phase_control,
            }
        }
        
        if self.anchor_manager is not None:
            summary['anchoring'] = self.anchor_manager.get_anchoring_summary()
        
        if self.phase_controller is not None:
            summary['phase'] = self.phase_controller.get_phase_summary()
        
        return summary
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("HIERARCHICAL FOLDING STATUS")
        print("=" * 70)
        
        if self.phase_controller is not None:
            self.phase_controller.print_summary()
        
        if self.anchor_manager is not None:
            self.anchor_manager.print_summary()


def create_hierarchical_folding_manager(
    sequence_length: int,
    enable_anchoring: bool = True,
    enable_phases: bool = True,
    aggressive: bool = False,
    fast: bool = False
) -> HierarchicalFoldingManager:
    """
    Factory function to create a hierarchical folding manager.
    
    Args:
        sequence_length: Number of residues
        enable_anchoring: Enable residue anchoring
        enable_phases: Enable phase control
        aggressive: Use aggressive anchoring (faster locking)
        fast: Use fast phase transitions (for testing)
    
    Returns:
        Configured HierarchicalFoldingManager
    """
    config = HierarchicalFoldingConfig(
        enable_anchoring=enable_anchoring,
        enable_phase_control=enable_phases,
        aggressive_anchoring=aggressive,
        fast_phases=fast
    )
    
    return HierarchicalFoldingManager(sequence_length, config)


# Utility function to apply move scaling based on phase
def scale_move_magnitude(
    base_magnitude: float,
    phase_scale: float,
    move_type: str,
    move_weights: Dict[str, float]
) -> float:
    """
    Scale a move's magnitude based on current phase.
    
    Args:
        base_magnitude: Original move magnitude (e.g., angle change)
        phase_scale: Phase-based scale factor (0.0-1.0)
        move_type: Type of move (e.g., 'backbone_rotation')
        move_weights: Phase-specific move type weights
    
    Returns:
        Scaled magnitude
    """
    # Get type-specific weight
    type_weight = move_weights.get(move_type, 1.0)
    
    # Combine scales
    combined_scale = phase_scale * math.sqrt(type_weight)
    
    return base_magnitude * combined_scale
