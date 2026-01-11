"""
Persistent Channel Targeting - Structural Blueprint Memory

Copyright (c) 2025 Dionte Robinson. All Rights Reserved.

This module implements PERSISTENT STRUCTURAL GOALS that guide exploration
across multiple iterations. Based on framework math, agents receive a
"blueprint" of target frequency channels for each residue.

Key Insight:
- Single-step channel guidance is too weak to overcome noise
- Agents need CUMULATIVE BIAS toward target structures
- This is what ML provides - persistent structural expectation
- We implement it using framework physics instead

The Blueprint:
1. Analyze sequence → predict target channels for each residue
2. Store as persistent memory that survives across iterations
3. Each move evaluation includes "blueprint alignment" score
4. Successfully reached targets reinforce the blueprint
5. Failed targets get re-evaluated

Framework Basis:
- Hydrophobic residues → low frequency (4-7 Hz) → core/sheet
- Polar residues → medium frequency (10 Hz) → helix/surface
- Charged residues → high frequency (12-15 Hz) → turns/loops
- Proline → very high frequency → breaks/turns
- Glycine → flexible → can be any
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2

# Amino acid properties for channel prediction
AMINO_ACID_PROPERTIES = {
    # Hydrophobic - tend toward low frequency (core, sheet)
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'size': 'large', 'preferred_channel': 7.0},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'size': 'medium', 'preferred_channel': 7.0},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'size': 'large', 'preferred_channel': 7.0},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'size': 'large', 'preferred_channel': 7.0},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'size': 'small', 'preferred_channel': 7.0},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'size': 'large', 'preferred_channel': 7.0},
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'size': 'small', 'preferred_channel': 10.0},
    
    # Polar - tend toward medium frequency (helix, surface)
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'size': 'large', 'preferred_channel': 10.0},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'size': 'large', 'preferred_channel': 10.0},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'size': 'medium', 'preferred_channel': 10.0},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'size': 'small', 'preferred_channel': 10.0},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'size': 'large', 'preferred_channel': 10.0},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'size': 'medium', 'preferred_channel': 10.0},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'size': 'large', 'preferred_channel': 10.0},
    
    # Charged - tend toward high frequency (surface, turns)
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'size': 'large', 'preferred_channel': 12.0},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'size': 'large', 'preferred_channel': 12.0},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'size': 'medium', 'preferred_channel': 12.0},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'size': 'large', 'preferred_channel': 12.0},
    
    # Special - high frequency (breaks, flexibility)
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'size': 'medium', 'preferred_channel': 15.0},  # Helix breaker
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'size': 'tiny', 'preferred_channel': 12.0},   # Flexible
}

# Channel to structure mapping
CHANNEL_TO_STRUCTURE = {
    4.0: {'ss': 'C', 'phi': -60.0, 'psi': 30.0, 'name': 'compact_coil'},
    7.0: {'ss': 'E', 'phi': -120.0, 'psi': 120.0, 'name': 'beta_sheet'},
    10.0: {'ss': 'H', 'phi': -60.0, 'psi': -45.0, 'name': 'alpha_helix'},
    12.0: {'ss': 'T', 'phi': -60.0, 'psi': 0.0, 'name': 'turn'},
    15.0: {'ss': 'C', 'phi': -80.0, 'psi': 150.0, 'name': 'extended'},
}


class TargetConfidence(Enum):
    """Confidence level in channel target"""
    LOW = 0.3       # Weak prediction
    MEDIUM = 0.5    # Moderate prediction
    HIGH = 0.7      # Strong prediction
    VERY_HIGH = 0.9 # Very confident


@dataclass
class ResidueTarget:
    """Target channel and structure for a single residue"""
    residue_idx: int
    amino_acid: str
    target_channel: float
    target_ss: str  # H, E, C, T
    target_phi: float
    target_psi: float
    confidence: float
    reason: str
    
    # Tracking
    times_reached: int = 0
    times_attempted: int = 0
    last_distance: float = 180.0  # Angular distance from target
    
    def success_rate(self) -> float:
        """Calculate success rate for this target"""
        if self.times_attempted == 0:
            return 0.0
        return self.times_reached / self.times_attempted
    
    def update_confidence(self, reached: bool):
        """Update confidence based on success/failure"""
        self.times_attempted += 1
        if reached:
            self.times_reached += 1
            # Increase confidence on success
            self.confidence = min(0.95, self.confidence + 0.05)
        else:
            # Decrease confidence on repeated failure
            if self.times_attempted > 5 and self.success_rate() < 0.2:
                self.confidence = max(0.2, self.confidence - 0.1)


@dataclass
class StructuralBlueprint:
    """
    Complete structural blueprint for a protein.
    
    This is the "memory" that persists across iterations,
    providing cumulative guidance toward target structure.
    """
    sequence: str
    residue_targets: List[ResidueTarget] = field(default_factory=list)
    
    # Pattern detection
    helix_regions: List[Tuple[int, int]] = field(default_factory=list)
    sheet_regions: List[Tuple[int, int]] = field(default_factory=list)
    
    # Progress tracking
    iteration_count: int = 0
    best_alignment_score: float = 0.0
    alignment_history: List[float] = field(default_factory=list)
    
    def get_target(self, residue_idx: int) -> Optional[ResidueTarget]:
        """Get target for a specific residue"""
        if residue_idx < len(self.residue_targets):
            return self.residue_targets[residue_idx]
        return None
    
    def get_target_angles(self, residue_idx: int) -> Tuple[float, float]:
        """Get target phi/psi for a residue"""
        target = self.get_target(residue_idx)
        if target:
            return target.target_phi, target.target_psi
        return -60.0, -45.0  # Default helix
    
    def get_region_type(self, residue_idx: int) -> str:
        """Determine if residue is in helix, sheet, or coil region"""
        for start, end in self.helix_regions:
            if start <= residue_idx <= end:
                return 'helix'
        for start, end in self.sheet_regions:
            if start <= residue_idx <= end:
                return 'sheet'
        return 'coil'


class BlueprintGenerator:
    """
    Generates structural blueprints from sequence using framework physics.
    
    This replaces ML's role of predicting structure from sequence,
    using the channel-energy correspondence from the framework.
    """
    
    def __init__(self):
        self.aa_properties = AMINO_ACID_PROPERTIES
        self.channel_structures = CHANNEL_TO_STRUCTURE
        self.phi = PHI
    
    def generate_blueprint(self, sequence: str) -> StructuralBlueprint:
        """
        Generate structural blueprint from amino acid sequence.
        
        Uses framework principles:
        1. Hydrophobic residues → low frequency → core/sheet
        2. Charged residues → high frequency → surface/turns
        3. Consecutive similar → nucleation sites
        4. Pattern detection for helix/sheet regions
        """
        blueprint = StructuralBlueprint(sequence=sequence)
        
        # Step 1: Assign initial channel targets based on amino acid properties
        for i, aa in enumerate(sequence):
            target = self._predict_residue_target(i, aa, sequence)
            blueprint.residue_targets.append(target)
        
        # Step 2: Detect potential helix regions (consecutive helix-favorable)
        blueprint.helix_regions = self._detect_helix_regions(blueprint)
        
        # Step 3: Detect potential sheet regions (alternating hydrophobic)
        blueprint.sheet_regions = self._detect_sheet_regions(sequence, blueprint)
        
        # Step 4: Refine targets based on detected regions
        self._refine_targets(blueprint)
        
        logger.info(
            f"Generated blueprint: {len(sequence)} residues, "
            f"{len(blueprint.helix_regions)} helix regions, "
            f"{len(blueprint.sheet_regions)} sheet regions"
        )
        
        return blueprint
    
    def _predict_residue_target(self, idx: int, aa: str, sequence: str) -> ResidueTarget:
        """Predict target channel for a single residue"""
        props = self.aa_properties.get(aa, {
            'hydrophobicity': 0, 'charge': 0, 'size': 'medium', 'preferred_channel': 10.0
        })
        
        # Base channel from amino acid preference
        base_channel = props['preferred_channel']
        
        # Adjust based on position (termini tend toward flexibility)
        n = len(sequence)
        if idx < 5 or idx >= n - 5:
            # Terminal regions - more flexible
            base_channel = max(12.0, base_channel)
            confidence = 0.4
            reason = "terminal_flexibility"
        elif props['charge'] != 0:
            # Charged residues - surface preference
            confidence = 0.6
            reason = "charge_surface"
        elif props['hydrophobicity'] > 2.0:
            # Strongly hydrophobic - core preference
            confidence = 0.7
            reason = "hydrophobic_core"
        else:
            confidence = 0.5
            reason = "sequence_based"
        
        # Get structure parameters for this channel
        struct_params = self.channel_structures.get(base_channel, {
            'ss': 'C', 'phi': -60.0, 'psi': 30.0
        })
        
        return ResidueTarget(
            residue_idx=idx,
            amino_acid=aa,
            target_channel=base_channel,
            target_ss=struct_params['ss'],
            target_phi=struct_params['phi'],
            target_psi=struct_params['psi'],
            confidence=confidence,
            reason=reason
        )
    
    def _detect_helix_regions(self, blueprint: StructuralBlueprint) -> List[Tuple[int, int]]:
        """
        Detect potential helix nucleation sites.
        
        Helix-favorable: A, L, E, M, Q, K (high helix propensity)
        Helix-breakers: P, G (break helix)
        Need 5+ consecutive high-propensity for nucleation.
        
        More conservative than before to avoid over-predicting helix.
        """
        helix_favorable = set('ALEMQK')  # Strong helix formers
        helix_moderate = set('RFWY')     # Moderate helix formers
        helix_breaker = set('PG')        # Break helices
        sheet_prefer = set('VIT')        # These prefer sheet over helix
        
        regions = []
        in_region = False
        start = 0
        score = 0
        window_scores = []  # Track scores in sliding window
        
        for i, target in enumerate(blueprint.residue_targets):
            aa = target.amino_acid
            
            # Score this residue
            if aa in helix_favorable:
                residue_score = 2
            elif aa in helix_moderate:
                residue_score = 1
            elif aa in helix_breaker:
                residue_score = -4  # Strong penalty
            elif aa in sheet_prefer:
                residue_score = -1  # Weak penalty
            else:
                residue_score = 0
            
            window_scores.append(residue_score)
            
            # Use sliding window of 5 residues
            if len(window_scores) > 5:
                window_scores.pop(0)
            
            window_sum = sum(window_scores)
            
            # More conservative threshold: need strong signal
            if not in_region and window_sum >= 6 and len(window_scores) >= 5:
                in_region = True
                start = i - len(window_scores) + 1
            elif in_region and window_sum < 2:
                in_region = False
                if i - start >= 5:  # Minimum length 5
                    regions.append((start, i - 1))
                window_scores = []
        
        # Handle region at end
        if in_region and len(blueprint.sequence) - start >= 5:
            regions.append((start, len(blueprint.sequence) - 1))
        
        return regions
    
    def _detect_sheet_regions(self, sequence: str, blueprint: StructuralBlueprint) -> List[Tuple[int, int]]:
        """
        Detect potential beta-sheet nucleation sites.
        
        Sheet-favorable: V, I, Y, F, W, T, C (high sheet propensity)
        Also look for alternating hydrophobic patterns (i, i+2).
        
        More aggressive sheet detection to balance helix bias.
        """
        sheet_favorable = set('VIYFWTC')
        sheet_strong = set('VIT')  # Very strong sheet preference
        
        regions = []
        window_scores = []
        in_region = False
        start = 0
        
        for i, aa in enumerate(sequence):
            # Score based on sheet propensity
            if aa in sheet_strong:
                score = 3
            elif aa in sheet_favorable:
                score = 2
            elif aa == 'P':  # Proline can be in sheets
                score = 0
            elif aa == 'G':  # Glycine flexible
                score = -1
            else:
                score = 0
            
            # Check for alternating pattern (classic sheet signature)
            # In sheets, hydrophobics often alternate: i, i+2
            if i >= 2:
                prev_aa = sequence[i-2]
                if aa in sheet_favorable and prev_aa in sheet_favorable:
                    score += 1  # Bonus for alternating pattern
            
            window_scores.append(score)
            if len(window_scores) > 4:
                window_scores.pop(0)
            
            window_sum = sum(window_scores)
            
            # Start sheet region when we see strong sheet signal
            if not in_region and window_sum >= 6 and len(window_scores) >= 3:
                in_region = True
                start = max(0, i - len(window_scores) + 1)
            elif in_region and window_sum < 2:
                if i - start >= 3:
                    regions.append((start, i - 1))
                in_region = False
                window_scores = []
        
        if in_region and len(sequence) - start >= 3:
            regions.append((start, len(sequence) - 1))
        
        return regions
    
    def _refine_targets(self, blueprint: StructuralBlueprint):
        """Refine targets based on detected secondary structure regions"""
        
        # Strengthen helix targets in helix regions
        for start, end in blueprint.helix_regions:
            for i in range(start, min(end + 1, len(blueprint.residue_targets))):
                target = blueprint.residue_targets[i]
                target.target_channel = 10.0
                target.target_ss = 'H'
                target.target_phi = -60.0
                target.target_psi = -45.0
                target.confidence = min(0.8, target.confidence + 0.2)
                target.reason = f"helix_region_{start}-{end}"
        
        # Strengthen sheet targets in sheet regions
        for start, end in blueprint.sheet_regions:
            for i in range(start, min(end + 1, len(blueprint.residue_targets))):
                target = blueprint.residue_targets[i]
                target.target_channel = 7.0
                target.target_ss = 'E'
                target.target_phi = -120.0
                target.target_psi = 120.0
                target.confidence = min(0.8, target.confidence + 0.2)
                target.reason = f"sheet_region_{start}-{end}"


class PersistentChannelMemory:
    """
    Memory system that maintains channel targets across iterations.
    
    This is the key innovation: instead of one-shot guidance,
    agents have persistent structural goals that accumulate
    influence over time.
    """
    
    def __init__(self, sequence: str):
        self.generator = BlueprintGenerator()
        self.blueprint = self.generator.generate_blueprint(sequence)
        
        # Tracking
        self.guidance_applications = 0
        self.successful_alignments = 0
        
    def get_move_bias(
        self,
        residue_idx: int,
        current_phi: float,
        current_psi: float,
        coherence: float
    ) -> Tuple[float, float, float]:
        """
        Get persistent bias toward target structure.
        
        Returns:
            (phi_delta, psi_delta, confidence)
            
        The bias ACCUMULATES - each call moves closer to target.
        """
        target = self.blueprint.get_target(residue_idx)
        if target is None:
            return 0.0, 0.0, 0.0
        
        # Calculate angular distance to target
        phi_diff = target.target_phi - current_phi
        psi_diff = target.target_psi - current_psi
        
        # Normalize to -180 to 180
        while phi_diff > 180: phi_diff -= 360
        while phi_diff < -180: phi_diff += 360
        while psi_diff > 180: psi_diff -= 360
        while psi_diff < -180: psi_diff += 360
        
        # Store distance for tracking
        target.last_distance = math.sqrt(phi_diff**2 + psi_diff**2)
        
        # Scale bias by confidence and coherence
        # Higher coherence = stronger bias toward target
        bias_strength = target.confidence * coherence * 0.5
        
        # Damped approach: move fraction of distance
        phi_delta = phi_diff * bias_strength
        psi_delta = psi_diff * bias_strength
        
        self.guidance_applications += 1
        
        return phi_delta, psi_delta, target.confidence
    
    def update_from_outcome(
        self,
        residue_idx: int,
        new_phi: float,
        new_psi: float,
        energy_improved: bool
    ):
        """
        Update target confidence based on outcome.
        
        If moving toward target improved energy, reinforce.
        If it made energy worse, reduce confidence.
        """
        target = self.blueprint.get_target(residue_idx)
        if target is None:
            return
        
        # Check if we reached target (within 30 degrees)
        phi_diff = abs(target.target_phi - new_phi)
        psi_diff = abs(target.target_psi - new_psi)
        
        if phi_diff > 180: phi_diff = 360 - phi_diff
        if psi_diff > 180: psi_diff = 360 - psi_diff
        
        reached = phi_diff < 30 and psi_diff < 30
        
        if reached and energy_improved:
            # Target was correct
            target.update_confidence(True)
            self.successful_alignments += 1
        elif reached and not energy_improved:
            # Reached target but energy got worse - might be wrong target
            target.update_confidence(False)
        # If not reached, don't update (still trying)
    
    def get_blueprint_alignment(
        self,
        phi_angles: List[float],
        psi_angles: List[float]
    ) -> float:
        """
        Calculate overall alignment with blueprint.
        
        Returns score 0-1 indicating how well current structure
        matches the predicted blueprint.
        """
        if not phi_angles or not psi_angles:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for i, target in enumerate(self.blueprint.residue_targets):
            if i >= len(phi_angles) or i >= len(psi_angles):
                break
            
            phi_diff = abs(target.target_phi - phi_angles[i])
            psi_diff = abs(target.target_psi - psi_angles[i])
            
            if phi_diff > 180: phi_diff = 360 - phi_diff
            if psi_diff > 180: psi_diff = 360 - psi_diff
            
            # Score: 1.0 at target, 0.0 at 90+ degrees away
            angle_dist = math.sqrt(phi_diff**2 + psi_diff**2)
            residue_score = max(0.0, 1.0 - angle_dist / 127.0)  # 127 = sqrt(90^2 + 90^2)
            
            # Weight by confidence
            total_score += residue_score * target.confidence
            total_weight += target.confidence
        
        if total_weight == 0:
            return 0.0
        
        alignment = total_score / total_weight
        
        # Track history
        self.blueprint.alignment_history.append(alignment)
        if alignment > self.blueprint.best_alignment_score:
            self.blueprint.best_alignment_score = alignment
        
        self.blueprint.iteration_count += 1
        
        return alignment
    
    def get_priority_residues(self, n: int = 5) -> List[int]:
        """
        Get residues that need most attention.
        
        Priority based on:
        1. High confidence targets that are far from goal
        2. Low success rate targets
        """
        priorities = []
        
        for target in self.blueprint.residue_targets:
            # Priority = confidence * (1 - success_rate) * distance
            priority = target.confidence * (1 - target.success_rate()) 
            if target.last_distance > 0:
                priority *= min(1.0, target.last_distance / 90.0)
            priorities.append((target.residue_idx, priority))
        
        # Sort by priority descending
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in priorities[:n]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'sequence_length': len(self.blueprint.sequence),
            'helix_regions': len(self.blueprint.helix_regions),
            'sheet_regions': len(self.blueprint.sheet_regions),
            'guidance_applications': self.guidance_applications,
            'successful_alignments': self.successful_alignments,
            'best_alignment': self.blueprint.best_alignment_score,
            'iteration_count': self.blueprint.iteration_count,
        }


# Convenience function for integration
def create_persistent_channel_memory(sequence: str) -> PersistentChannelMemory:
    """Create a persistent channel memory for a sequence."""
    return PersistentChannelMemory(sequence)
