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
# Using corrected Chou-Fasman propensities from literature
# P > 1.0 means favors that structure, P < 1.0 means disfavors
AMINO_ACID_PROPERTIES = {
    # Strong helix formers
    'A': {'hydrophobicity': 1.8, 'helix_prop': 1.42, 'sheet_prop': 0.83, 'turn_prop': 0.66},
    'E': {'hydrophobicity': -3.5, 'helix_prop': 1.51, 'sheet_prop': 0.37, 'turn_prop': 0.74},
    'L': {'hydrophobicity': 3.8, 'helix_prop': 1.21, 'sheet_prop': 1.30, 'turn_prop': 0.59},
    'M': {'hydrophobicity': 1.9, 'helix_prop': 1.45, 'sheet_prop': 1.05, 'turn_prop': 0.60},
    'K': {'hydrophobicity': -3.9, 'helix_prop': 1.16, 'sheet_prop': 0.74, 'turn_prop': 1.01},
    'R': {'hydrophobicity': -4.5, 'helix_prop': 0.98, 'sheet_prop': 0.93, 'turn_prop': 0.95},
    'Q': {'hydrophobicity': -3.5, 'helix_prop': 1.11, 'sheet_prop': 1.10, 'turn_prop': 0.98},
    'H': {'hydrophobicity': -3.2, 'helix_prop': 1.00, 'sheet_prop': 0.87, 'turn_prop': 0.95},
    
    # Strong sheet formers
    'V': {'hydrophobicity': 4.2, 'helix_prop': 1.06, 'sheet_prop': 1.70, 'turn_prop': 0.50},
    'I': {'hydrophobicity': 4.5, 'helix_prop': 1.08, 'sheet_prop': 1.60, 'turn_prop': 0.47},
    'Y': {'hydrophobicity': -1.3, 'helix_prop': 0.69, 'sheet_prop': 1.47, 'turn_prop': 1.14},
    'F': {'hydrophobicity': 2.8, 'helix_prop': 1.13, 'sheet_prop': 1.38, 'turn_prop': 0.60},
    'W': {'hydrophobicity': -0.9, 'helix_prop': 1.08, 'sheet_prop': 1.37, 'turn_prop': 0.96},
    'T': {'hydrophobicity': -0.7, 'helix_prop': 0.83, 'sheet_prop': 1.19, 'turn_prop': 0.96},
    'C': {'hydrophobicity': 2.5, 'helix_prop': 0.70, 'sheet_prop': 1.19, 'turn_prop': 1.19},
    
    # Turn/coil formers (helix/sheet breakers)
    'N': {'hydrophobicity': -3.5, 'helix_prop': 0.67, 'sheet_prop': 0.89, 'turn_prop': 1.56},
    'G': {'hydrophobicity': -0.4, 'helix_prop': 0.57, 'sheet_prop': 0.75, 'turn_prop': 1.56},
    'P': {'hydrophobicity': -1.6, 'helix_prop': 0.57, 'sheet_prop': 0.55, 'turn_prop': 1.52},
    'D': {'hydrophobicity': -3.5, 'helix_prop': 1.01, 'sheet_prop': 0.54, 'turn_prop': 1.46},
    'S': {'hydrophobicity': -0.8, 'helix_prop': 0.77, 'sheet_prop': 0.75, 'turn_prop': 1.43},
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
    
    # Global bias: 'helix', 'sheet', or 'neutral'
    global_bias: str = 'neutral'
    
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
    
    def _calculate_global_bias(self, sequence: str) -> str:
        """
        Calculate global structural bias from sequence composition.
        
        Uses patterns that strongly indicate helix vs sheet proteins:
        - Proline content (breaks helices)
        - Alternating hydrophobic pattern (sheets)
        - Charged residue clusters (helices - surface)
        
        Returns 'helix', 'sheet', or 'neutral'
        """
        n = len(sequence)
        if n < 10:
            return 'neutral'
        
        # Count Pro (helix breaker)
        pro_count = sequence.count('P')
        pro_fraction = pro_count / n
        
        # Count strong helix formers: E, A, L, M (high P_alpha, low P_beta)
        helix_strong = sum(1 for aa in sequence if aa in 'EALM')
        helix_frac = helix_strong / n
        
        # Count strong sheet formers: V, I, Y (high P_beta)
        sheet_strong = sum(1 for aa in sequence if aa in 'VIY')
        sheet_frac = sheet_strong / n
        
        # Alternating hydrophobic pattern (strong sheet indicator)
        alternating_count = 0
        hydrophobic = set('VILFY')
        for i in range(n - 4):
            if (sequence[i] in hydrophobic and 
                sequence[i+2] in hydrophobic and 
                sequence[i+4] in hydrophobic):
                alternating_count += 1
        alt_frac = alternating_count / (n - 4) if n > 4 else 0
        
        # Scoring
        helix_score = helix_frac * 2 - pro_fraction * 3
        sheet_score = sheet_frac * 2 + alt_frac * 5
        
        if sheet_score > helix_score + 0.1:
            return 'sheet'
        elif helix_score > sheet_score + 0.1:
            return 'helix'
        else:
            return 'neutral'
    
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
        
        # Calculate global structural bias
        blueprint.global_bias = self._calculate_global_bias(sequence)
        
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
            f"{len(blueprint.sheet_regions)} sheet regions, "
            f"bias={blueprint.global_bias}"
        )
        
        return blueprint
    
    def _predict_residue_target(self, idx: int, aa: str, sequence: str) -> ResidueTarget:
        """
        Predict target channel for a single residue using Chou-Fasman propensities.
        
        Uses actual secondary structure propensities rather than simple
        hydrophobicity mapping.
        """
        props = self.aa_properties.get(aa, {
            'hydrophobicity': 0, 'charge': 0, 'size': 'medium', 
            'helix_prop': 1.0, 'sheet_prop': 1.0, 'turn_prop': 1.0
        })
        
        helix_prop: float = float(props.get('helix_prop', 1.0))
        sheet_prop: float = float(props.get('sheet_prop', 1.0))
        turn_prop: float = float(props.get('turn_prop', 1.0))
        
        n = len(sequence)
        
        # Terminal regions favor turns/coils
        if idx < 4 or idx >= n - 4:
            base_channel = 12.0  # Turn
            target_ss = 'T'
            confidence = 0.5
            reason = "terminal"
        # Use propensities to determine most likely structure
        elif helix_prop > sheet_prop and helix_prop > turn_prop and helix_prop > 1.1:
            base_channel = 10.0  # Helix
            target_ss = 'H'
            confidence = min(0.8, 0.4 + (helix_prop - 1.0) * 0.5)
            reason = f"helix_prop_{helix_prop:.2f}"
        elif sheet_prop > helix_prop and sheet_prop > turn_prop and sheet_prop > 1.1:
            base_channel = 7.0   # Sheet
            target_ss = 'E'
            confidence = min(0.8, 0.4 + (sheet_prop - 1.0) * 0.5)
            reason = f"sheet_prop_{sheet_prop:.2f}"
        elif turn_prop > 1.2:
            base_channel = 12.0  # Turn
            target_ss = 'T'
            confidence = min(0.7, 0.4 + (turn_prop - 1.0) * 0.3)
            reason = f"turn_prop_{turn_prop:.2f}"
        else:
            # Ambiguous - use coil
            base_channel = 12.0
            target_ss = 'C'
            confidence = 0.4
            reason = "ambiguous"
        
        # Get structure parameters for this channel
        struct_params = self.channel_structures.get(base_channel, {
            'ss': 'C', 'phi': -60.0, 'psi': 30.0
        })
        
        return ResidueTarget(
            residue_idx=idx,
            amino_acid=aa,
            target_channel=base_channel,
            target_ss=target_ss,
            target_phi=struct_params['phi'],
            target_psi=struct_params['psi'],
            confidence=confidence,
            reason=reason
        )
    
    def _detect_helix_regions(self, blueprint: StructuralBlueprint) -> List[Tuple[int, int]]:
        """
        Detect helix regions using classic Chou-Fasman algorithm.
        
        Classic rules (not preference-based):
        1. Nucleation: 4+ residues out of 6 with P_alpha >= 1.03 (formers)
        2. Extension: Continue while running average P_alpha >= 1.0
        3. Termination: P_alpha < 1.0 for 4+ consecutive residues
        4. Proline breaks helix (except at N-terminus of helix)
        """
        sequence = blueprint.sequence
        n = len(sequence)
        
        HELIX_FORMER = 1.03   # P_alpha threshold for helix former
        HELIX_INDIF = 1.0     # Indifferent threshold
        HELIX_BREAKER = 0.7   # Strong helix breakers: P, G
        
        # Get helix propensities
        helix_scores = []
        for aa in sequence:
            props = self.aa_properties.get(aa, {'helix_prop': 1.0})
            helix_scores.append(props.get('helix_prop', 1.0))
        
        # Mark helix formers (P_alpha >= 1.03)
        is_former = [p >= HELIX_FORMER for p in helix_scores]
        
        # Find nucleation sites: 4+ formers in a window of 6
        nucleation_sites = []
        for i in range(n - 5):
            window = is_former[i:i+6]
            formers = sum(window)
            # Check for helix breakers (P anywhere, G at position 2+)
            has_breaker = 'P' in sequence[i:i+6]
            
            if formers >= 4 and not has_breaker:
                nucleation_sites.append(i)
        
        # Extend and merge regions
        regions = []
        used = set()
        
        for nuc in nucleation_sites:
            if nuc in used:
                continue
            
            start = nuc
            end = nuc + 5  # Initial nucleation is 6 residues
            
            # Extend left while helix-favorable
            while start > 0:
                if sequence[start-1] == 'P':
                    break
                # Extend if P_alpha >= 1.0
                if helix_scores[start-1] >= HELIX_INDIF:
                    start -= 1
                else:
                    break
            
            # Extend right while helix-favorable
            while end < n - 1:
                if sequence[end+1] == 'P':
                    break
                if helix_scores[end+1] >= HELIX_INDIF:
                    end += 1
                else:
                    break
            
            # Mark as used
            for i in range(start, end + 1):
                used.add(i)
            
            # Only keep if avg propensity > 1.0
            avg_prop = sum(helix_scores[start:end+1]) / (end - start + 1)
            if avg_prop >= HELIX_INDIF and end - start >= 3:
                regions.append((start, end))
        
        return self._merge_regions(regions)
    
    def _detect_sheet_regions(self, sequence: str, blueprint: StructuralBlueprint) -> List[Tuple[int, int]]:
        """
        Detect sheet regions using classic Chou-Fasman algorithm.
        
        Classic rules:
        1. Nucleation: 3+ residues out of 5 with P_beta >= 1.05 (formers)
        2. Extension: Continue while running average P_beta >= 1.0
        3. Termination: P_beta < 1.0 for 4+ consecutive residues
        4. Alternating hydrophobic pattern is strong sheet indicator
        """
        n = len(sequence)
        
        SHEET_FORMER = 1.05   # P_beta threshold for sheet former
        SHEET_INDIF = 1.0     # Indifferent threshold
        
        # Get sheet propensities
        sheet_scores = []
        for aa in sequence:
            props = self.aa_properties.get(aa, {'sheet_prop': 1.0})
            sheet_scores.append(props.get('sheet_prop', 1.0))
        
        # Mark sheet formers (P_beta >= 1.05)
        is_former = [p >= SHEET_FORMER for p in sheet_scores]
        
        # Find nucleation sites: 3+ formers in a window of 5
        nucleation_sites = []
        for i in range(n - 4):
            window = is_former[i:i+5]
            formers = sum(window)
            
            if formers >= 3:
                nucleation_sites.append(i)
        
        # Also detect alternating hydrophobic pattern (strong sheet indicator)
        # V, I, L, F, Y all have P_beta >= 1.2
        beta_sheet = set('VILFY')
        for i in range(n - 4):
            # i, i+2, i+4 all hydrophobic = strong alternating pattern
            if (sequence[i] in beta_sheet and 
                sequence[i+2] in beta_sheet and
                sequence[i+4] in beta_sheet):
                if i not in nucleation_sites:
                    nucleation_sites.append(i)
        
        nucleation_sites.sort()
        
        # Extend and merge
        regions = []
        used = set()
        
        for nuc in nucleation_sites:
            if nuc in used:
                continue
            
            start = nuc
            end = nuc + 4  # Initial nucleation is 5 residues
            
            # Extend left while sheet-favorable
            while start > 0:
                if sheet_scores[start-1] >= SHEET_INDIF:
                    start -= 1
                else:
                    break
            
            # Extend right while sheet-favorable  
            while end < n - 1:
                if sheet_scores[end+1] >= SHEET_INDIF:
                    end += 1
                else:
                    break
            
            for i in range(start, end + 1):
                used.add(i)
            
            # Only keep if avg propensity >= 1.0
            avg_prop = sum(sheet_scores[start:end+1]) / (end - start + 1)
            if avg_prop >= SHEET_INDIF and end - start >= 2:
                regions.append((start, end))
        
        return self._merge_regions(regions)
    
    def _merge_regions(self, regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent regions."""
        if not regions:
            return []
        
        regions = sorted(regions)
        merged = [regions[0]]
        
        for start, end in regions[1:]:
            last_start, last_end = merged[-1]
            # Merge if overlapping or adjacent (within 2 residues)
            if start <= last_end + 2:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged
    
    def _resolve_conflicts(self, blueprint: StructuralBlueprint):
        """
        Resolve conflicts between helix and sheet regions.
        
        Strategy: Look at the ENTIRE overlap region and compare:
        - Count of residues with helix_prop > 1.03 (helix formers)
        - Count of residues with sheet_prop > 1.05 (sheet formers)
        
        Whichever has more formers in the overlap wins.
        """
        sequence = blueprint.sequence
        n = len(sequence)
        
        if not blueprint.helix_regions and not blueprint.sheet_regions:
            return
        
        # Calculate per-residue propensity scores
        helix_scores = []
        sheet_scores = []
        for aa in sequence:
            props = self.aa_properties.get(aa, {'helix_prop': 1.0, 'sheet_prop': 1.0})
            helix_scores.append(props.get('helix_prop', 1.0))
            sheet_scores.append(props.get('sheet_prop', 1.0))
        
        # Mark initial claims with region info
        helix_regions_set = []  # (start, end, set of indices)
        for start, end in blueprint.helix_regions:
            helix_regions_set.append((start, end, set(range(start, end + 1))))
        
        sheet_regions_set = []
        for start, end in blueprint.sheet_regions:
            sheet_regions_set.append((start, end, set(range(start, end + 1))))
        
        # Build final assignments
        final_assignments = ['C'] * n
        
        # For each helix/sheet pair that overlaps, decide who wins the overlap
        resolved_helix = set()  # residues that helix wins
        resolved_sheet = set()  # residues that sheet wins
        
        # Get global bias (use attribute if set, otherwise neutral)
        global_bias = getattr(blueprint, 'global_bias', 'neutral')
        
        for h_start, h_end, h_indices in helix_regions_set:
            for s_start, s_end, s_indices in sheet_regions_set:
                overlap = h_indices & s_indices
                if not overlap:
                    continue
                
                # Count formers in the overlap region
                helix_formers = sum(1 for i in overlap if helix_scores[i] >= 1.03)
                sheet_formers = sum(1 for i in overlap if sheet_scores[i] >= 1.05)
                
                # Apply global bias for tie-breaking
                if helix_formers == sheet_formers:
                    if global_bias == 'sheet':
                        resolved_sheet.update(overlap)
                    else:
                        resolved_helix.update(overlap)  # helix wins ties by default
                elif helix_formers > sheet_formers:
                    resolved_helix.update(overlap)
                else:
                    resolved_sheet.update(overlap)
        
        # Now assign all residues
        for i in range(n):
            # Check if in any helix region
            in_helix = any(i in h_set for _, _, h_set in helix_regions_set)
            in_sheet = any(i in s_set for _, _, s_set in sheet_regions_set)
            
            if i in resolved_helix:
                final_assignments[i] = 'H'
            elif i in resolved_sheet:
                final_assignments[i] = 'E'
            elif in_helix and not in_sheet:
                final_assignments[i] = 'H'
            elif in_sheet and not in_helix:
                final_assignments[i] = 'E'
        
        # Rebuild regions from assignments
        new_helix = []
        new_sheet = []
        
        i = 0
        while i < n:
            if final_assignments[i] == 'H':
                start = i
                while i < n and final_assignments[i] == 'H':
                    i += 1
                if i - start >= 4:  # Min helix length
                    new_helix.append((start, i - 1))
            elif final_assignments[i] == 'E':
                start = i
                while i < n and final_assignments[i] == 'E':
                    i += 1
                if i - start >= 3:  # Min sheet length
                    new_sheet.append((start, i - 1))
            else:
                i += 1
        
        blueprint.helix_regions = new_helix
        blueprint.sheet_regions = new_sheet
    
    def _refine_targets(self, blueprint: StructuralBlueprint):
        """Refine targets based on detected secondary structure regions."""
        
        # First resolve conflicts between helix and sheet
        self._resolve_conflicts(blueprint)
        
        # Apply helix targets
        for start, end in blueprint.helix_regions:
            for i in range(start, min(end + 1, len(blueprint.residue_targets))):
                target = blueprint.residue_targets[i]
                target.target_channel = 10.0
                target.target_ss = 'H'
                target.target_phi = -60.0
                target.target_psi = -45.0
                target.confidence = min(0.8, target.confidence + 0.2)
                target.reason = f"helix_region_{start}-{end}"
        
        # Apply sheet targets
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
