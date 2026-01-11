"""
Channel-to-Action Mapper - Framework-Based Structural Guidance

Copyright (c) 2025 Dionte Robinson. All Rights Reserved.

This module bridges the gap between FREQUENCY CHANNEL identification
and STRUCTURAL ACTION. Based on the framework's math:

Key Insight from Theory Paper:
- Frequency channels correspond to specific energy types
- Impedance matching determines accessibility
- QCP formula tells us WHERE we are in channel space
- THIS MODULE tells us WHAT STRUCTURAL ACTION to take

Framework Equations Used:
1. QCP = 4.0 + 2^n * φ^l * m
   - n = structural hierarchy (coil=0, helix=1, sheet=2, phi-bend=3)
   - l = neighbor count (1-3)
   - m = hydrophobicity (-1 to 1)

2. Frequency-Structure Correspondence:
   - 4 Hz (K+, delta): Deep stable states -> compact core
   - 7 Hz (Mg2+, theta): Memory consolidation -> sheet formation
   - 10 Hz (Ca2+, alpha): Relaxed stability -> helix formation
   - 12 Hz (alpha peak): Structural refinement -> local optimization
   - 15 Hz (beta): Active exploration -> backbone rotation

3. φ-angle = 137.5° = 2π/φ (golden angle)
   - Optimal packing angle
   - Helix turn: 3.6 residues = 100° per residue (close to φ-based)

The Key Innovation:
- Current QCP tells you WHICH CHANNEL you're in
- Target QCP tells you WHICH CHANNEL to access
- Channel difference maps to SPECIFIC ANGLE CHANGES
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Golden ratio constants from framework
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618
PHI_ANGLE_RAD = 2 * math.pi / PHI  # ≈ 2.399 rad
PHI_ANGLE_DEG = PHI_ANGLE_RAD * 180 / math.pi  # ≈ 137.5°

# Ideal backbone angles for secondary structures
# From Ramachandran plot and phi-based analysis
IDEAL_ANGLES = {
    'helix': {'phi': -60.0, 'psi': -45.0},
    'sheet': {'phi': -120.0, 'psi': 120.0},
    'phi_turn': {'phi': -PHI_ANGLE_DEG, 'psi': PHI_ANGLE_DEG},  # Golden angle based
    'coil': {'phi': -60.0, 'psi': 30.0},  # Flexible
}

# Framework frequency-to-structure mapping
# From docs/THEORY_PAPER.md Table: Ion Channel Frequencies
FREQUENCY_STRUCTURE_MAP = {
    4.0: {
        'name': 'K_delta',
        'energy_type': 'ionic_slow',
        'structure_action': 'hydrophobic_collapse',
        'description': 'Deep stable states - compact core formation',
        'target_n': 0,  # Coil can become compact
        'phi_adjustment': 0.0,  # Maintain current
        'psi_adjustment': 0.0,
    },
    7.0: {
        'name': 'Mg_theta',
        'energy_type': 'ionic_slow', 
        'structure_action': 'sheet_formation',
        'description': 'Memory consolidation - beta sheet patterns',
        'target_n': 2,  # Sheet
        'phi_adjustment': -120.0,  # Sheet phi
        'psi_adjustment': 120.0,   # Sheet psi
    },
    10.0: {
        'name': 'Ca_alpha',
        'energy_type': 'ionic_medium',
        'structure_action': 'helix_formation',
        'description': 'Relaxed stability - alpha helix formation',
        'target_n': 1,  # Helix
        'phi_adjustment': -60.0,  # Helix phi
        'psi_adjustment': -45.0,  # Helix psi
    },
    12.0: {
        'name': 'alpha_peak',
        'energy_type': 'electrical',
        'structure_action': 'local_refinement',
        'description': 'Structural refinement - local optimization',
        'target_n': None,  # Keep current
        'phi_adjustment': 'refine',  # Small adjustments
        'psi_adjustment': 'refine',
    },
    15.0: {
        'name': 'beta_max',
        'energy_type': 'electrical',
        'structure_action': 'backbone_rotation',
        'description': 'Active exploration - backbone flexibility',
        'target_n': None,  # Explore
        'phi_adjustment': 'random',  # Large changes
        'psi_adjustment': 'random',
    },
}


class ChannelState(Enum):
    """Discrete channel states from frequency analysis"""
    K_DELTA = 4      # Deep stability
    MG_THETA = 7     # Memory/consolidation
    CA_ALPHA = 10    # Relaxed awareness
    ALPHA_PEAK = 12  # Peak stability
    BETA_MAX = 15    # Active exploration


@dataclass
class ChannelAction:
    """Structural action derived from channel analysis"""
    source_channel: float  # Current frequency channel
    target_channel: float  # Target frequency channel
    action_type: str       # What structural move to make
    target_residues: List[int]  # Which residues to modify
    phi_target: Optional[float]  # Target phi angle (or None for relative)
    psi_target: Optional[float]  # Target psi angle (or None for relative)
    phi_delta: float       # Change in phi (if relative)
    psi_delta: float       # Change in psi (if relative)
    confidence: float      # How confident in this action (0-1)
    rationale: str         # Human-readable explanation


@dataclass
class ResidueChannelState:
    """Channel state for a single residue"""
    residue_idx: int
    qcp_value: float
    structural_level: int  # n in QCP formula
    neighbor_level: int    # l in QCP formula
    hydrophobicity: float  # m in QCP formula
    inferred_channel: float  # Which frequency channel this maps to
    channel_name: str


class ChannelActionMapper:
    """
    Maps frequency channel states to structural actions.
    
    Core insight: QCP formula encodes structural hierarchy.
    By inverting this, we can determine WHAT CHANGES
    would move us to a different channel.
    
    QCP = 4.0 + 2^n * φ^l * m
    
    To change QCP:
    - Change n (structural type): helix/sheet/coil transitions
    - Change l (neighbors): compaction/expansion
    - Change m (environment): burial/exposure
    
    Each maps to specific backbone angle changes.
    """
    
    def __init__(self):
        self.phi = PHI
        self.base_energy = 4.0
        self.channel_map = FREQUENCY_STRUCTURE_MAP
        self.ideal_angles = IDEAL_ANGLES
        
        # Hydrophobicity scale (Kyte-Doolittle)
        self.hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
    
    def analyze_residue_channels(
        self,
        sequence: str,
        secondary_structure: List[str],
        neighbor_counts: List[int],
        phi_angles: List[float],
        psi_angles: List[float]
    ) -> List[ResidueChannelState]:
        """
        Analyze each residue's current channel state.
        
        Returns per-residue channel analysis showing which
        frequency channel each residue is currently in.
        """
        channel_states = []
        
        for i, aa in enumerate(sequence):
            # Calculate n (structural hierarchy)
            ss = secondary_structure[i] if i < len(secondary_structure) else 'C'
            if ss == 'H':
                n = 1  # Helix
            elif ss == 'E':
                n = 2  # Sheet
            elif ss == 'S':
                n = 3  # Special phi-bend
            else:
                n = 0  # Coil
            
            # Calculate l (neighbor scaling)
            neighbors = neighbor_counts[i] if i < len(neighbor_counts) else 0
            l = min(max(1, neighbors // 3), 3)
            
            # Calculate m (hydrophobicity)
            raw_hydro = self.hydrophobicity_scale.get(aa, 0)
            m = (raw_hydro + 4.5) / 9.0  # Normalize to 0-1
            m = m * 2 - 1  # Scale to -1 to 1
            
            # Calculate QCP
            qcp = self.base_energy + (2**n * (self.phi**l) * m)
            
            # Map QCP to channel frequency
            # Use log impedance matching from framework
            inferred_channel = self._qcp_to_channel(qcp, n)
            channel_name = self._get_channel_name(inferred_channel)
            
            channel_states.append(ResidueChannelState(
                residue_idx=i,
                qcp_value=qcp,
                structural_level=n,
                neighbor_level=l,
                hydrophobicity=m,
                inferred_channel=inferred_channel,
                channel_name=channel_name
            ))
        
        return channel_states
    
    def _qcp_to_channel(self, qcp: float, structural_level: int) -> float:
        """
        Map QCP value to nearest frequency channel.
        
        Uses impedance matching principle: channels couple
        when log(Z) values are close.
        """
        # Framework frequency channels
        channels = [4.0, 7.0, 10.0, 12.0, 15.0]
        
        # Structural level strongly influences channel
        # n=0 (coil) -> flexible, can be any
        # n=1 (helix) -> 10 Hz (Ca2+/alpha)
        # n=2 (sheet) -> 7 Hz (Mg2+/theta)
        # n=3 (phi-bend) -> 12 Hz (alpha peak)
        
        if structural_level == 1:
            return 10.0  # Helix -> alpha rhythm
        elif structural_level == 2:
            return 7.0   # Sheet -> theta rhythm
        elif structural_level == 3:
            return 12.0  # Phi-bend -> alpha peak
        else:
            # Coil: use QCP magnitude to determine
            # Low QCP -> stable (4 Hz), high QCP -> active (15 Hz)
            if qcp < 5.0:
                return 4.0
            elif qcp < 7.0:
                return 7.0
            elif qcp < 10.0:
                return 10.0
            elif qcp < 12.0:
                return 12.0
            else:
                return 15.0
    
    def _get_channel_name(self, frequency: float) -> str:
        """Get human-readable channel name."""
        channel_info = self.channel_map.get(frequency, {})
        return channel_info.get('name', f'{frequency}Hz')
    
    def compute_channel_actions(
        self,
        channel_states: List[ResidueChannelState],
        target_channels: Optional[Dict[int, float]] = None,
        energy_gradient: Optional[List[float]] = None,
        coherence: float = 0.5
    ) -> List[ChannelAction]:
        """
        Compute structural actions to reach target channels.
        
        This is the KEY FUNCTION that translates channel identification
        into specific structural moves.
        
        Args:
            channel_states: Current channel state per residue
            target_channels: Optional explicit targets {residue_idx: target_freq}
            energy_gradient: Optional per-residue energy gradient
            coherence: Current coherence level (affects action magnitude)
        
        Returns:
            List of ChannelAction objects with specific structural guidance
        """
        actions = []
        
        for state in channel_states:
            # Determine target channel
            if target_channels and state.residue_idx in target_channels:
                target_channel = target_channels[state.residue_idx]
            else:
                # Auto-determine target based on energy gradient
                target_channel = self._auto_target_channel(
                    state, energy_gradient, coherence
                )
            
            # Skip if already at target
            if abs(state.inferred_channel - target_channel) < 0.5:
                continue
            
            # Compute the action needed
            action = self._channel_transition_action(
                state, target_channel, coherence
            )
            
            if action:
                actions.append(action)
        
        return actions
    
    def _auto_target_channel(
        self,
        state: ResidueChannelState,
        energy_gradient: Optional[List[float]],
        coherence: float
    ) -> float:
        """
        Automatically determine target channel based on context.
        
        Uses framework principles:
        - High coherence -> seek stability (lower frequencies)
        - Low coherence -> explore (higher frequencies)
        - Negative energy gradient -> reinforce current structure
        - Positive energy gradient -> change structure
        """
        current = state.inferred_channel
        
        # Get energy gradient for this residue
        if energy_gradient and state.residue_idx < len(energy_gradient):
            grad = energy_gradient[state.residue_idx]
        else:
            grad = 0.0
        
        # Decision logic based on framework
        if coherence > 0.7:
            # High coherence: seek stability
            if current > 10.0:
                return 10.0  # Move toward alpha (stable)
            elif current < 7.0:
                return 7.0   # Move toward theta (memory consolidation)
            else:
                return current  # Maintain
        elif coherence < 0.4:
            # Low coherence: explore
            if grad > 0:  # Energy increasing = bad position
                return 15.0  # Escape via backbone rotation
            else:
                return 12.0  # Refine locally
        else:
            # Medium coherence: adaptive
            if state.structural_level == 0:  # Coil
                # Coil regions should form structure
                if state.hydrophobicity > 0.3:
                    return 7.0  # Hydrophobic -> sheet
                else:
                    return 10.0  # Polar -> helix
            else:
                return current  # Maintain structure
    
    def _channel_transition_action(
        self,
        state: ResidueChannelState,
        target_channel: float,
        coherence: float
    ) -> Optional[ChannelAction]:
        """
        Create specific action for channel transition.
        
        This is where frequency channels translate to backbone angles.
        """
        source = state.inferred_channel
        target_info = self.channel_map.get(target_channel, {})
        
        if not target_info:
            return None
        
        action_type = target_info.get('structure_action', 'unknown')
        target_n = target_info.get('target_n')
        
        # Determine angle targets/deltas
        phi_target = None
        psi_target = None
        phi_delta = 0.0
        psi_delta = 0.0
        
        phi_adj = target_info.get('phi_adjustment', 0.0)
        psi_adj = target_info.get('psi_adjustment', 0.0)
        
        if phi_adj == 'refine':
            # Small refinement based on golden angle
            phi_delta = PHI_ANGLE_DEG / 10 * (1 - coherence)  # ~13.75° max
            psi_delta = PHI_ANGLE_DEG / 10 * (1 - coherence)
        elif phi_adj == 'random':
            # Large exploration
            phi_delta = PHI_ANGLE_DEG * (1 - coherence)  # Up to 137.5°
            psi_delta = PHI_ANGLE_DEG * (1 - coherence)
        elif isinstance(phi_adj, (int, float)):
            # Absolute target
            phi_target = float(phi_adj)
            psi_target = float(psi_adj) if isinstance(psi_adj, (int, float)) else None
        
        # Confidence based on channel distance and coherence
        channel_distance = abs(target_channel - source)
        confidence = max(0.3, 1.0 - channel_distance / 15.0) * coherence
        
        # Rationale
        rationale = (
            f"Channel transition {source:.0f}Hz ({state.channel_name}) -> "
            f"{target_channel:.0f}Hz ({target_info.get('name', 'unknown')}): "
            f"{target_info.get('description', action_type)}"
        )
        
        return ChannelAction(
            source_channel=source,
            target_channel=target_channel,
            action_type=action_type,
            target_residues=[state.residue_idx],
            phi_target=phi_target,
            psi_target=psi_target,
            phi_delta=phi_delta,
            psi_delta=psi_delta,
            confidence=confidence,
            rationale=rationale
        )
    
    def get_collective_action(
        self,
        actions: List[ChannelAction],
        min_confidence: float = 0.3  # Lowered from 0.5 to allow more guidance
    ) -> Dict[str, Any]:
        """
        Aggregate individual actions into collective structural moves.
        
        Identifies patterns:
        - Consecutive helix_formation -> helix nucleation
        - Consecutive sheet_formation -> sheet nucleation
        - Scattered high-confidence -> distributed refinement
        """
        if not actions:
            return {'type': 'none', 'residues': [], 'confidence': 0.0}
        
        # Filter by confidence
        confident_actions = [a for a in actions if a.confidence >= min_confidence]
        
        if not confident_actions:
            return {'type': 'low_confidence', 'residues': [], 'confidence': 0.0}
        
        # Group by action type
        by_type: Dict[str, List[ChannelAction]] = {}
        for action in confident_actions:
            if action.action_type not in by_type:
                by_type[action.action_type] = []
            by_type[action.action_type].append(action)
        
        # Find dominant action type
        dominant_type = max(by_type.keys(), key=lambda t: len(by_type[t]))
        dominant_actions = by_type[dominant_type]
        
        # Check for consecutive residues (nucleation potential)
        residues = sorted([a.target_residues[0] for a in dominant_actions])
        consecutive_runs = self._find_consecutive_runs(residues)
        
        if consecutive_runs and max(len(r) for r in consecutive_runs) >= 4:
            # Found a run of 4+ -> nucleation site
            longest_run = max(consecutive_runs, key=len)
            return {
                'type': f'{dominant_type}_nucleation',
                'residues': longest_run,
                'confidence': sum(a.confidence for a in dominant_actions) / len(dominant_actions),
                'actions': dominant_actions,
                'phi_target': dominant_actions[0].phi_target,
                'psi_target': dominant_actions[0].psi_target,
            }
        else:
            # Distributed actions
            return {
                'type': f'{dominant_type}_distributed',
                'residues': residues,
                'confidence': sum(a.confidence for a in dominant_actions) / len(dominant_actions),
                'actions': dominant_actions,
            }
    
    def _find_consecutive_runs(self, indices: List[int]) -> List[List[int]]:
        """Find runs of consecutive integers."""
        if not indices:
            return []
        
        runs = []
        current_run = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_run.append(indices[i])
            else:
                if len(current_run) >= 2:
                    runs.append(current_run)
                current_run = [indices[i]]
        
        if len(current_run) >= 2:
            runs.append(current_run)
        
        return runs
    
    def apply_action_to_angles(
        self,
        action: ChannelAction,
        current_phi: float,
        current_psi: float
    ) -> Tuple[float, float]:
        """
        Apply channel action to get new backbone angles.
        
        Returns (new_phi, new_psi) after applying the action.
        """
        if action.phi_target is not None:
            # Move toward absolute target
            # Use damped approach: move 30% toward target
            new_phi = current_phi + 0.3 * (action.phi_target - current_phi)
            new_psi = current_psi + 0.3 * (action.psi_target - current_psi) if action.psi_target else current_psi
        else:
            # Apply delta (with sign based on direction)
            import random
            sign = 1 if random.random() > 0.5 else -1
            new_phi = current_phi + sign * action.phi_delta
            new_psi = current_psi + sign * action.psi_delta
        
        # Normalize to -180 to 180
        new_phi = ((new_phi + 180) % 360) - 180
        new_psi = ((new_psi + 180) % 360) - 180
        
        return new_phi, new_psi


# Convenience function for integration
def create_channel_guided_move(
    mapper: ChannelActionMapper,
    sequence: str,
    secondary_structure: List[str],
    neighbor_counts: List[int],
    phi_angles: List[float],
    psi_angles: List[float],
    coherence: float = 0.5,
    energy_gradient: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    High-level function to create channel-guided structural move.
    
    Returns a move specification that can be used by the move system.
    """
    # Analyze current channel states
    channel_states = mapper.analyze_residue_channels(
        sequence, secondary_structure, neighbor_counts, phi_angles, psi_angles
    )
    
    # Compute actions
    actions = mapper.compute_channel_actions(
        channel_states, 
        energy_gradient=energy_gradient,
        coherence=coherence
    )
    
    # Get collective action
    collective = mapper.get_collective_action(actions)
    
    # Compute new angles for target residues
    new_angles = {}
    if 'actions' in collective:
        for action in collective['actions']:
            for res_idx in action.target_residues:
                if res_idx < len(phi_angles) and res_idx < len(psi_angles):
                    new_phi, new_psi = mapper.apply_action_to_angles(
                        action, phi_angles[res_idx], psi_angles[res_idx]
                    )
                    new_angles[res_idx] = {'phi': new_phi, 'psi': new_psi}
    
    return {
        'move_type': collective['type'],
        'target_residues': collective['residues'],
        'confidence': collective['confidence'],
        'new_angles': new_angles,
        'channel_states': channel_states,
        'actions': actions,
    }
