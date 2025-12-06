"""
Residue Anchor Manager - Hierarchical Structural Anchoring System

This module implements progressive search space confinement by tracking
per-residue structural confidence and enforcing constraints on residues
that have formed stable secondary structure.

Key Concept:
  As exploration progresses, residues that consistently form stable
  secondary structure (helices, sheets) get "anchored" - their backbone
  angles are constrained to prevent regression while allowing the rest
  of the protein to continue exploring.

This mimics real protein folding where local structure forms first
(nanoseconds) and then those elements pack together (microseconds+).

Author: UBF Protein System
Date: December 5, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import math
import logging
import time

logger = logging.getLogger(__name__)


class StructuralState(Enum):
    """Structural state of a residue."""
    UNSTRUCTURED = "unstructured"  # Coil/loop - free to move
    HELIX = "helix"                # Alpha helix
    SHEET = "sheet"                # Beta sheet
    TURN = "turn"                  # Turn/bend


class AnchorStrength(Enum):
    """How tightly a residue is anchored."""
    FREE = "free"           # No constraints (±180°)
    SOFT = "soft"           # Light guidance (±30°)
    MEDIUM = "medium"       # Moderate constraint (±15°)
    HARD = "hard"           # Tight constraint (±5°)
    LOCKED = "locked"       # Minimal movement (±2°)


@dataclass
class ResidueAnchorState:
    """
    Tracking state for a single residue's anchoring.
    
    Attributes:
        residue_index: Index in the sequence (0-based)
        current_state: Current structural classification
        state_confidence: Confidence in the current state (0.0-1.0)
        consecutive_observations: How many iterations this state has been observed
        anchor_strength: Current constraint level
        target_phi: Target phi angle (degrees) if anchored
        target_psi: Target psi angle (degrees) if anchored
        last_update_iteration: When this was last updated
    """
    residue_index: int
    current_state: StructuralState = StructuralState.UNSTRUCTURED
    state_confidence: float = 0.0
    consecutive_observations: int = 0
    anchor_strength: AnchorStrength = AnchorStrength.FREE
    target_phi: Optional[float] = None
    target_psi: Optional[float] = None
    last_update_iteration: int = 0
    
    def get_allowed_deviation(self) -> float:
        """Get allowed angle deviation in degrees based on anchor strength."""
        deviations = {
            AnchorStrength.FREE: 180.0,
            AnchorStrength.SOFT: 30.0,
            AnchorStrength.MEDIUM: 15.0,
            AnchorStrength.HARD: 5.0,
            AnchorStrength.LOCKED: 2.0,
        }
        return deviations[self.anchor_strength]
    
    def is_anchored(self) -> bool:
        """Check if this residue has any anchoring constraint."""
        return self.anchor_strength != AnchorStrength.FREE


@dataclass
class AnchorManagerConfig:
    """Configuration for the anchor manager."""
    
    # Confidence thresholds for anchoring
    soft_anchor_confidence: float = 0.5      # Start soft anchoring
    medium_anchor_confidence: float = 0.7    # Upgrade to medium
    hard_anchor_confidence: float = 0.85     # Upgrade to hard
    locked_anchor_confidence: float = 0.95   # Lock the residue
    
    # Observation requirements
    min_observations_soft: int = 30          # Iterations before soft anchor
    min_observations_medium: int = 60        # Iterations before medium anchor
    min_observations_hard: int = 100         # Iterations before hard anchor
    min_observations_locked: int = 150       # Iterations before locking
    
    # Regression handling
    regression_penalty: float = 0.3          # Confidence drop on state change
    confidence_decay_rate: float = 0.01      # Decay per iteration if not reinforced
    
    # Ideal backbone angles for secondary structure (degrees)
    helix_phi: float = -60.0
    helix_psi: float = -45.0
    sheet_phi: float = -135.0
    sheet_psi: float = 135.0
    
    # Minimum segment length for anchoring
    min_helix_length: int = 4                # At least 4 residues for helix
    min_sheet_length: int = 3                # At least 3 residues for sheet


class ResidueAnchorManager:
    """
    Manages per-residue anchoring for hierarchical protein folding.
    
    This class tracks the structural state of each residue over time and
    progressively constrains residues that form stable secondary structure.
    
    The key insight is that real proteins fold hierarchically:
    1. Local secondary structure forms first (helices, sheets)
    2. These elements then pack together into tertiary structure
    
    By anchoring residues that have formed stable local structure, we:
    - Reduce the search space dimensionality
    - Prevent regression (losing good structure)
    - Focus exploration on unstructured regions and packing
    
    Usage:
        >>> manager = ResidueAnchorManager(sequence_length=76)
        >>> 
        >>> # During exploration, update with detected structure
        >>> manager.update_residue_state(10, StructuralState.HELIX, confidence=0.8)
        >>> 
        >>> # Check constraints before executing a move
        >>> if manager.is_move_allowed(residue_idx=10, new_phi=-65, new_psi=-50):
        ...     execute_move()
        >>> 
        >>> # Get summary of anchoring status
        >>> summary = manager.get_anchoring_summary()
    """
    
    def __init__(
        self,
        sequence_length: int,
        config: Optional[AnchorManagerConfig] = None
    ):
        """
        Initialize anchor manager for a protein sequence.
        
        Args:
            sequence_length: Number of residues in the protein
            config: Configuration settings (uses defaults if None)
        """
        self.sequence_length = sequence_length
        self.config = config or AnchorManagerConfig()
        
        # Per-residue tracking
        self._residue_states: Dict[int, ResidueAnchorState] = {
            i: ResidueAnchorState(residue_index=i)
            for i in range(sequence_length)
        }
        
        # Statistics
        self.total_updates = 0
        self.anchor_promotions = 0
        self.anchor_demotions = 0
        self.moves_constrained = 0
        self.moves_allowed = 0
        
        # Current iteration (updated externally)
        self._current_iteration = 0
        
        logger.info(f"ResidueAnchorManager initialized for {sequence_length} residues")
    
    def set_iteration(self, iteration: int) -> None:
        """Update the current iteration counter."""
        self._current_iteration = iteration
    
    def update_residue_state(
        self,
        residue_index: int,
        detected_state: StructuralState,
        confidence: float,
        phi_angle: Optional[float] = None,
        psi_angle: Optional[float] = None
    ) -> None:
        """
        Update the structural state observation for a residue.
        
        Call this each time secondary structure is detected/analyzed.
        The manager tracks consistency over time and anchors stable structure.
        
        Args:
            residue_index: Residue index (0-based)
            detected_state: The detected structural state
            confidence: Confidence in the detection (0.0-1.0)
            phi_angle: Observed phi angle (degrees), used as anchor target
            psi_angle: Observed psi angle (degrees), used as anchor target
        """
        if residue_index < 0 or residue_index >= self.sequence_length:
            logger.warning(f"Invalid residue index: {residue_index}")
            return
        
        self.total_updates += 1
        state = self._residue_states[residue_index]
        
        # Check if state is consistent with previous
        if detected_state == state.current_state:
            # Reinforce: increase confidence and observation count
            state.consecutive_observations += 1
            state.state_confidence = min(1.0, state.state_confidence + confidence * 0.1)
            
            # Update target angles if provided (exponential moving average)
            if phi_angle is not None and state.target_phi is not None:
                state.target_phi = 0.9 * state.target_phi + 0.1 * phi_angle
            elif phi_angle is not None:
                state.target_phi = phi_angle
                
            if psi_angle is not None and state.target_psi is not None:
                state.target_psi = 0.9 * state.target_psi + 0.1 * psi_angle
            elif psi_angle is not None:
                state.target_psi = psi_angle
        else:
            # State changed - apply regression penalty
            old_state = state.current_state
            state.current_state = detected_state
            state.state_confidence = max(0.0, confidence - self.config.regression_penalty)
            state.consecutive_observations = 1
            
            # Reset target angles for new state
            if detected_state == StructuralState.HELIX:
                state.target_phi = phi_angle or self.config.helix_phi
                state.target_psi = psi_angle or self.config.helix_psi
            elif detected_state == StructuralState.SHEET:
                state.target_phi = phi_angle or self.config.sheet_phi
                state.target_psi = psi_angle or self.config.sheet_psi
            else:
                state.target_phi = None
                state.target_psi = None
            
            # May need to demote anchor strength
            if state.anchor_strength != AnchorStrength.FREE:
                logger.info(
                    f"Residue {residue_index} state changed from {old_state.value} "
                    f"to {detected_state.value}, demoting anchor"
                )
                self._demote_anchor(residue_index)
        
        state.last_update_iteration = self._current_iteration
        
        # Check if we should promote anchor strength
        self._maybe_promote_anchor(residue_index)
    
    def update_from_secondary_structure(
        self,
        ss_string: str,
        phi_angles: Optional[List[float]] = None,
        psi_angles: Optional[List[float]] = None,
        base_confidence: float = 0.8
    ) -> None:
        """
        Batch update from a secondary structure string (e.g., from DSSP).
        
        Args:
            ss_string: Secondary structure string (H=helix, E=sheet, C/T/S/etc=other)
            phi_angles: List of phi angles for each residue
            psi_angles: List of psi angles for each residue
            base_confidence: Base confidence for detected structure
        """
        if len(ss_string) != self.sequence_length:
            logger.warning(
                f"SS string length {len(ss_string)} doesn't match "
                f"sequence length {self.sequence_length}"
            )
            return
        
        for i, ss_char in enumerate(ss_string):
            # Map DSSP codes to our states
            if ss_char in ['H', 'G', 'I']:  # Helix types
                state = StructuralState.HELIX
                confidence = base_confidence
            elif ss_char in ['E', 'B']:  # Sheet types
                state = StructuralState.SHEET
                confidence = base_confidence
            elif ss_char in ['T', 'S']:  # Turns
                state = StructuralState.TURN
                confidence = base_confidence * 0.7
            else:  # Coil/undefined
                state = StructuralState.UNSTRUCTURED
                confidence = base_confidence * 0.5
            
            phi = phi_angles[i] if phi_angles and i < len(phi_angles) else None
            psi = psi_angles[i] if psi_angles and i < len(psi_angles) else None
            
            self.update_residue_state(i, state, confidence, phi, psi)
    
    def _maybe_promote_anchor(self, residue_index: int) -> None:
        """Check if a residue should be promoted to a stronger anchor."""
        state = self._residue_states[residue_index]
        
        # Only anchor structured residues (helix/sheet)
        if state.current_state not in [StructuralState.HELIX, StructuralState.SHEET]:
            return
        
        # Check minimum segment length (don't anchor isolated residues)
        if not self._is_part_of_segment(residue_index):
            return
        
        old_strength = state.anchor_strength
        new_strength = old_strength
        
        # Determine appropriate anchor strength based on confidence and observations
        conf = state.state_confidence
        obs = state.consecutive_observations
        
        if (conf >= self.config.locked_anchor_confidence and 
            obs >= self.config.min_observations_locked):
            new_strength = AnchorStrength.LOCKED
        elif (conf >= self.config.hard_anchor_confidence and 
              obs >= self.config.min_observations_hard):
            new_strength = AnchorStrength.HARD
        elif (conf >= self.config.medium_anchor_confidence and 
              obs >= self.config.min_observations_medium):
            new_strength = AnchorStrength.MEDIUM
        elif (conf >= self.config.soft_anchor_confidence and 
              obs >= self.config.min_observations_soft):
            new_strength = AnchorStrength.SOFT
        
        # Only promote, never demote here (demotion happens on state change)
        if self._anchor_strength_value(new_strength) > self._anchor_strength_value(old_strength):
            state.anchor_strength = new_strength
            self.anchor_promotions += 1
            
            logger.info(
                f"Residue {residue_index} promoted to {new_strength.value} anchor "
                f"(state={state.current_state.value}, conf={conf:.2f}, obs={obs})"
            )
    
    def _demote_anchor(self, residue_index: int) -> None:
        """Demote a residue's anchor strength (called on state regression)."""
        state = self._residue_states[residue_index]
        
        # Step down one level
        demotion_map = {
            AnchorStrength.LOCKED: AnchorStrength.HARD,
            AnchorStrength.HARD: AnchorStrength.MEDIUM,
            AnchorStrength.MEDIUM: AnchorStrength.SOFT,
            AnchorStrength.SOFT: AnchorStrength.FREE,
            AnchorStrength.FREE: AnchorStrength.FREE,
        }
        
        old_strength = state.anchor_strength
        state.anchor_strength = demotion_map[old_strength]
        
        if state.anchor_strength != old_strength:
            self.anchor_demotions += 1
    
    def _anchor_strength_value(self, strength: AnchorStrength) -> int:
        """Convert anchor strength to numeric value for comparison."""
        values = {
            AnchorStrength.FREE: 0,
            AnchorStrength.SOFT: 1,
            AnchorStrength.MEDIUM: 2,
            AnchorStrength.HARD: 3,
            AnchorStrength.LOCKED: 4,
        }
        return values[strength]
    
    def _is_part_of_segment(self, residue_index: int) -> bool:
        """
        Check if a residue is part of a contiguous secondary structure segment.
        
        We only anchor residues that are part of a segment of minimum length
        to avoid anchoring isolated residues that may be noise.
        """
        state = self._residue_states[residue_index]
        target_state = state.current_state
        
        if target_state == StructuralState.HELIX:
            min_length = self.config.min_helix_length
        elif target_state == StructuralState.SHEET:
            min_length = self.config.min_sheet_length
        else:
            return False
        
        # Count contiguous residues with same state
        segment_size = 1
        
        # Look backwards
        for i in range(residue_index - 1, -1, -1):
            if self._residue_states[i].current_state == target_state:
                segment_size += 1
            else:
                break
        
        # Look forwards
        for i in range(residue_index + 1, self.sequence_length):
            if self._residue_states[i].current_state == target_state:
                segment_size += 1
            else:
                break
        
        return segment_size >= min_length
    
    def is_move_allowed(
        self,
        residue_index: int,
        new_phi: Optional[float] = None,
        new_psi: Optional[float] = None
    ) -> bool:
        """
        Check if a proposed backbone angle change is allowed by anchoring constraints.
        
        Args:
            residue_index: Residue to check
            new_phi: Proposed new phi angle (degrees)
            new_psi: Proposed new psi angle (degrees)
        
        Returns:
            True if the move is allowed, False if it violates constraints
        """
        if residue_index < 0 or residue_index >= self.sequence_length:
            return True  # Invalid index, let other validation catch it
        
        state = self._residue_states[residue_index]
        
        if not state.is_anchored():
            self.moves_allowed += 1
            return True
        
        allowed_deviation = state.get_allowed_deviation()
        
        # Check phi constraint
        if new_phi is not None and state.target_phi is not None:
            phi_diff = self._angle_difference(new_phi, state.target_phi)
            if phi_diff > allowed_deviation:
                self.moves_constrained += 1
                return False
        
        # Check psi constraint
        if new_psi is not None and state.target_psi is not None:
            psi_diff = self._angle_difference(new_psi, state.target_psi)
            if psi_diff > allowed_deviation:
                self.moves_constrained += 1
                return False
        
        self.moves_allowed += 1
        return True
    
    def constrain_angles(
        self,
        residue_index: int,
        proposed_phi: float,
        proposed_psi: float
    ) -> Tuple[float, float]:
        """
        Constrain proposed angles to be within anchor limits.
        
        Instead of rejecting moves, this adjusts them to fit within constraints.
        
        Args:
            residue_index: Residue index
            proposed_phi: Proposed phi angle (degrees)
            proposed_psi: Proposed psi angle (degrees)
        
        Returns:
            Tuple of (constrained_phi, constrained_psi)
        """
        if residue_index < 0 or residue_index >= self.sequence_length:
            return proposed_phi, proposed_psi
        
        state = self._residue_states[residue_index]
        
        if not state.is_anchored():
            return proposed_phi, proposed_psi
        
        allowed_deviation = state.get_allowed_deviation()
        
        # Constrain phi
        constrained_phi = proposed_phi
        if state.target_phi is not None:
            phi_diff = self._angle_difference(proposed_phi, state.target_phi)
            if phi_diff > allowed_deviation:
                # Clamp to allowed range
                constrained_phi = self._clamp_angle(
                    proposed_phi, state.target_phi, allowed_deviation
                )
        
        # Constrain psi
        constrained_psi = proposed_psi
        if state.target_psi is not None:
            psi_diff = self._angle_difference(proposed_psi, state.target_psi)
            if psi_diff > allowed_deviation:
                constrained_psi = self._clamp_angle(
                    proposed_psi, state.target_psi, allowed_deviation
                )
        
        return constrained_phi, constrained_psi
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate the absolute difference between two angles (handling wraparound)."""
        diff = abs(angle1 - angle2)
        if diff > 180.0:
            diff = 360.0 - diff
        return diff
    
    def _clamp_angle(self, proposed: float, target: float, max_deviation: float) -> float:
        """Clamp an angle to be within max_deviation of target."""
        diff = proposed - target
        
        # Handle wraparound
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0
        
        # Clamp
        if diff > max_deviation:
            clamped = target + max_deviation
        elif diff < -max_deviation:
            clamped = target - max_deviation
        else:
            clamped = proposed
        
        # Normalize to [-180, 180]
        while clamped > 180.0:
            clamped -= 360.0
        while clamped < -180.0:
            clamped += 360.0
        
        return clamped
    
    def get_anchored_residues(self) -> List[int]:
        """Get list of residue indices that are currently anchored."""
        return [
            i for i, state in self._residue_states.items()
            if state.is_anchored()
        ]
    
    def get_free_residues(self) -> List[int]:
        """Get list of residue indices that are free to move."""
        return [
            i for i, state in self._residue_states.items()
            if not state.is_anchored()
        ]
    
    def get_anchoring_percentage(self) -> float:
        """Get percentage of residues that are anchored."""
        anchored = len(self.get_anchored_residues())
        return (anchored / self.sequence_length) * 100.0 if self.sequence_length > 0 else 0.0
    
    def get_residue_state(self, residue_index: int) -> Optional[ResidueAnchorState]:
        """Get the anchor state for a specific residue."""
        return self._residue_states.get(residue_index)
    
    def decay_confidence(self) -> None:
        """
        Apply confidence decay to all residues.
        
        Call this periodically to allow anchors to weaken if not reinforced.
        """
        for state in self._residue_states.values():
            if state.state_confidence > 0:
                state.state_confidence = max(
                    0.0, 
                    state.state_confidence - self.config.confidence_decay_rate
                )
    
    def get_anchoring_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the anchoring state.
        
        Returns:
            Dictionary with anchoring statistics and details
        """
        anchor_counts = {strength: 0 for strength in AnchorStrength}
        state_counts = {state: 0 for state in StructuralState}
        
        for residue_state in self._residue_states.values():
            anchor_counts[residue_state.anchor_strength] += 1
            state_counts[residue_state.current_state] += 1
        
        return {
            'sequence_length': self.sequence_length,
            'current_iteration': self._current_iteration,
            'anchoring_percentage': self.get_anchoring_percentage(),
            'anchor_distribution': {
                strength.value: count for strength, count in anchor_counts.items()
            },
            'structural_state_distribution': {
                state.value: count for state, count in state_counts.items()
            },
            'statistics': {
                'total_updates': self.total_updates,
                'anchor_promotions': self.anchor_promotions,
                'anchor_demotions': self.anchor_demotions,
                'moves_constrained': self.moves_constrained,
                'moves_allowed': self.moves_allowed,
                'constraint_rate': (
                    self.moves_constrained / max(1, self.moves_constrained + self.moves_allowed)
                ) * 100.0
            },
            'anchored_segments': self._get_anchored_segments(),
        }
    
    def _get_anchored_segments(self) -> List[Dict[str, Any]]:
        """Get list of contiguous anchored segments."""
        segments = []
        current_segment = None
        
        for i in range(self.sequence_length):
            state = self._residue_states[i]
            
            if state.is_anchored():
                if current_segment is None:
                    current_segment = {
                        'start': i,
                        'end': i,
                        'state': state.current_state.value,
                        'min_strength': state.anchor_strength.value,
                    }
                else:
                    current_segment['end'] = i
                    # Track minimum strength in segment
                    if self._anchor_strength_value(state.anchor_strength) < \
                       self._anchor_strength_value(AnchorStrength[current_segment['min_strength'].upper()]):
                        current_segment['min_strength'] = state.anchor_strength.value
            else:
                if current_segment is not None:
                    current_segment['length'] = current_segment['end'] - current_segment['start'] + 1
                    segments.append(current_segment)
                    current_segment = None
        
        # Don't forget last segment
        if current_segment is not None:
            current_segment['length'] = current_segment['end'] - current_segment['start'] + 1
            segments.append(current_segment)
        
        return segments
    
    def print_summary(self) -> None:
        """Print a human-readable summary of anchoring state."""
        summary = self.get_anchoring_summary()
        
        print("\n" + "=" * 70)
        print("RESIDUE ANCHORING SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Overall Status:")
        print(f"  Sequence length: {summary['sequence_length']}")
        print(f"  Current iteration: {summary['current_iteration']}")
        print(f"  Anchoring percentage: {summary['anchoring_percentage']:.1f}%")
        
        print(f"\n🔒 Anchor Distribution:")
        for strength, count in summary['anchor_distribution'].items():
            pct = (count / self.sequence_length) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {strength:8s}: {count:3d} ({pct:5.1f}%) {bar}")
        
        print(f"\n🧬 Structural States:")
        for state, count in summary['structural_state_distribution'].items():
            pct = (count / self.sequence_length) * 100
            print(f"  {state:12s}: {count:3d} ({pct:5.1f}%)")
        
        print(f"\n📈 Statistics:")
        stats = summary['statistics']
        print(f"  Total updates: {stats['total_updates']}")
        print(f"  Anchor promotions: {stats['anchor_promotions']}")
        print(f"  Anchor demotions: {stats['anchor_demotions']}")
        print(f"  Moves constrained: {stats['moves_constrained']}")
        print(f"  Moves allowed: {stats['moves_allowed']}")
        print(f"  Constraint rate: {stats['constraint_rate']:.1f}%")
        
        if summary['anchored_segments']:
            print(f"\n🎯 Anchored Segments:")
            for seg in summary['anchored_segments']:
                print(f"  Residues {seg['start']}-{seg['end']} ({seg['length']} res): "
                      f"{seg['state']} [{seg['min_strength']}]")
        
        print("=" * 70 + "\n")


def create_anchor_manager(
    sequence_length: int,
    aggressive: bool = False
) -> ResidueAnchorManager:
    """
    Factory function to create an anchor manager with preset configurations.
    
    Args:
        sequence_length: Number of residues
        aggressive: If True, use more aggressive anchoring (faster locking)
    
    Returns:
        Configured ResidueAnchorManager
    """
    if aggressive:
        config = AnchorManagerConfig(
            soft_anchor_confidence=0.4,
            medium_anchor_confidence=0.6,
            hard_anchor_confidence=0.75,
            locked_anchor_confidence=0.9,
            min_observations_soft=20,
            min_observations_medium=40,
            min_observations_hard=70,
            min_observations_locked=100,
        )
    else:
        config = AnchorManagerConfig()  # Use defaults
    
    return ResidueAnchorManager(sequence_length, config)
