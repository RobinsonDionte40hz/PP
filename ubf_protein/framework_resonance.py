"""
Framework Resonance Module - Resonance Lock Strength and Time Evolution

Implements the resonance quantization principles from the AstroFolds framework
for application to protein structure prediction:

1. Resonance Lock Strength: L = exp(-10 * error)
   - Measures how close two frequency-like values are to integer ratios
   - L = 1.0 = perfect lock, L < 0.5 = weak/no resonance

2. Logarithmic Impedance Matching: R = exp[-(log Z1 - log Z2)² / (2σ_log²)]
   - Scale-invariant coupling quality measure
   - σ_log = 1.5 provides ~3 orders of magnitude bandwidth

3. Time Evolution Maintenance: M(t) = 1 - (1 - e^(-t/τ)) * D_max
   - Resonance locks require time to establish
   - Protein folding builds up coherence over iterations

4. Golden Ratio Distance Matching: d = 3.8 * φ^n Angstroms
   - Optimal residue spacing follows golden ratio harmonics
   - n = 0,1,2,3,4 provides characteristic distances

References:
    Robinson, D. (2026). AstroFolds: Resonance-Based Orbital Stability Prediction.
    Robinson, D. (2026). Why Impedance Matching Must Be Logarithmic.
    Robinson, D. (2026). Computational Alchemy: Atomic Impedance.

Author: Dionte Robinson
Date: January 2026
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from functools import lru_cache

# =============================================================================
# CONSTANTS FROM FRAMEWORK
# =============================================================================

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895

# Golden angle in radians (2π/φ² ≈ 137.5°)
PHI_ANGLE_RAD = 2 * math.pi / (PHI ** 2)  # ≈ 2.399963229728653 rad ≈ 137.5°

# Logarithmic impedance bandwidth (from framework validation)
SIGMA_LOG = 1.5  # Systems couple if within ~3 orders of magnitude

# Resonance lock sensitivity (from AstroFolds)
RESONANCE_SENSITIVITY = 10.0  # Exponent coefficient for lock strength

# Time evolution parameters
DEFAULT_QUALITY_FACTOR = 150.0  # Q_sys for protein systems
DEFAULT_MAX_DECAY = 0.3  # Maximum decay without maintenance

# CA-CA equilibrium distance in Angstroms (protein backbone)
CA_CA_DISTANCE = 3.8

# Golden ratio harmonic distances (3.8 * φ^n Å)
# These are the "resonant" distances in protein structures
PHI_HARMONIC_DISTANCES = [
    CA_CA_DISTANCE * (PHI ** 0),  # 3.8 Å - adjacent residues
    CA_CA_DISTANCE * (PHI ** 1),  # 6.15 Å - i, i+2 typical
    CA_CA_DISTANCE * (PHI ** 2),  # 9.94 Å - i, i+3 helix turn
    CA_CA_DISTANCE * (PHI ** 3),  # 16.09 Å - secondary structure
    CA_CA_DISTANCE * (PHI ** 4),  # 26.02 Å - domain contacts
]


# =============================================================================
# RESONANCE LOCK STRENGTH
# =============================================================================

def find_best_integer_ratio(ratio: float, max_n: int = 12) -> Tuple[int, int, float]:
    """
    Find the best small integer ratio approximation to a given ratio.
    
    This implements the resonance quantization principle: stable configurations
    exhibit integer-ratio frequency relationships.
    
    Args:
        ratio: The ratio to approximate (e.g., f1/f2)
        max_n: Maximum integer to consider (default: 12)
    
    Returns:
        Tuple of (n1, n2, error) where:
        - n1:n2 is the best integer ratio
        - error is the relative error |ratio - n1/n2| / ratio
    """
    if ratio <= 0:
        return (1, 1, 1.0)
    
    best_n1, best_n2 = 1, 1
    best_error = abs(ratio - 1.0) / ratio
    
    for n1 in range(1, max_n + 1):
        for n2 in range(1, max_n + 1):
            approx = n1 / n2
            error = abs(ratio - approx) / ratio
            
            if error < best_error:
                best_error = error
                best_n1, best_n2 = n1, n2
    
    return (best_n1, best_n2, best_error)


def resonance_lock_strength(value1: float, value2: float, 
                            max_n: int = 12,
                            sensitivity: float = RESONANCE_SENSITIVITY) -> float:
    """
    Calculate resonance lock strength between two frequency-like values.
    
    From AstroFolds framework:
        L(f1, f2) = exp(-sensitivity × error)
    
    Properties:
        - L = 1.0 → Perfect integer ratio resonance
        - L > 0.8 → Strong resonance (typically stable)
        - L < 0.5 → Weak/no resonance (unstable)
    
    Args:
        value1, value2: Two frequency-like values (e.g., distances, energies)
        max_n: Maximum integer for ratio search (default: 12)
        sensitivity: Exponential decay rate (default: 10.0)
    
    Returns:
        Lock strength L in range [0, 1]
    """
    if value1 <= 0 or value2 <= 0:
        return 0.0
    
    ratio = value1 / value2
    _, _, error = find_best_integer_ratio(ratio, max_n)
    
    # Exponential lock strength (from AstroFolds Eq. 1)
    return math.exp(-sensitivity * error)


def pairwise_resonance_matrix(values: List[float], 
                               max_n: int = 12) -> List[List[float]]:
    """
    Calculate pairwise resonance lock strengths for a list of values.
    
    Useful for analyzing distance patterns in protein structures.
    
    Args:
        values: List of frequency-like values
        max_n: Maximum integer for ratio search
    
    Returns:
        NxN matrix of lock strengths
    """
    n = len(values)
    matrix = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                matrix[i][j] = resonance_lock_strength(values[i], values[j], max_n)
    
    return matrix


# =============================================================================
# LOGARITHMIC IMPEDANCE MATCHING
# =============================================================================

def log_impedance_match(z1: float, z2: float, 
                        sigma_log: float = SIGMA_LOG) -> float:
    """
    Calculate logarithmic impedance matching quality.
    
    From framework derivation:
        R(Z1, Z2) = exp[-(log Z1 - log Z2)² / (2σ_log²)]
    
    Key insight: This is scale-invariant and allows comparison across
    quantities spanning many orders of magnitude.
    
    Properties:
        - R = 1.0 → Perfect impedance match
        - R > 0.8 → Strong coupling possible
        - R < 0.5 → Weak/no coupling
    
    Args:
        z1, z2: Two impedance values
        sigma_log: Logarithmic bandwidth (default: 1.5)
    
    Returns:
        Match quality R in range [0, 1]
    """
    if z1 <= 0 or z2 <= 0:
        return 0.0
    
    log_diff = math.log10(z1) - math.log10(z2)
    return math.exp(-(log_diff ** 2) / (2 * sigma_log ** 2))


def combined_stability_score(lock_strengths: List[float],
                              impedance_matches: List[float]) -> float:
    """
    Calculate combined stability score from resonance and impedance.
    
    From AstroFolds:
        S_total = Σ L_ij × R_ij
    
    Args:
        lock_strengths: List of pairwise resonance lock strengths
        impedance_matches: List of pairwise impedance match qualities
    
    Returns:
        Total stability score
    """
    if len(lock_strengths) != len(impedance_matches):
        raise ValueError("Lock strengths and impedance matches must have same length")
    
    return sum(l * r for l, r in zip(lock_strengths, impedance_matches))


# =============================================================================
# TIME EVOLUTION AND MAINTENANCE
# =============================================================================

@dataclass
class ResonanceAccumulator:
    """
    Tracks resonance buildup over time (iterations).
    
    From framework: Resonance locks require time to establish.
    
    M(t) = 1 - (1 - e^(-t/τ_buildup)) × D_max
    
    Where:
        τ_buildup = Q_sys / ω
        Q_sys = quality factor (~100-200 for protein systems)
        ω = angular frequency (related to iteration rate)
    """
    quality_factor: float = DEFAULT_QUALITY_FACTOR
    max_decay: float = DEFAULT_MAX_DECAY
    
    # Internal state
    current_time: int = 0  # Iteration counter
    accumulated_strength: float = 0.0
    history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        pass  # history is now initialized by field(default_factory=list)
    
    def update(self, instantaneous_lock: float, iteration: int = 1) -> float:
        """
        Update accumulated resonance strength with new measurement.
        
        Args:
            instantaneous_lock: Current lock strength measurement [0, 1]
            iteration: Number of iterations since last update
        
        Returns:
            Current accumulated strength (maintenance factor)
        """
        self.current_time += iteration
        
        # Calculate buildup time constant
        # τ = Q / ω, where ω ∝ iteration rate
        tau_buildup = self.quality_factor / max(1.0, self.current_time ** 0.5)
        
        # Exponential accumulation
        decay_factor = math.exp(-iteration / tau_buildup)
        
        # Update accumulated strength (exponential moving average)
        self.accumulated_strength = (
            decay_factor * self.accumulated_strength + 
            (1 - decay_factor) * instantaneous_lock
        )
        
        # Calculate maintenance factor
        maintenance = 1 - (1 - math.exp(-self.current_time / tau_buildup)) * self.max_decay
        
        # Store in history
        self.history.append(self.accumulated_strength * maintenance)
        
        return self.accumulated_strength * maintenance
    
    def get_maintenance_factor(self) -> float:
        """Get current maintenance factor M(t)."""
        tau_buildup = self.quality_factor / max(1.0, self.current_time ** 0.5)
        return 1 - (1 - math.exp(-self.current_time / tau_buildup)) * self.max_decay
    
    def reset(self):
        """Reset accumulator state."""
        self.current_time = 0
        self.accumulated_strength = 0.0
        self.history = []


# =============================================================================
# GOLDEN RATIO DISTANCE MATCHING
# =============================================================================

def golden_distance_match(distance: float, 
                          base_distance: float = CA_CA_DISTANCE,
                          max_harmonic: int = 4,
                          tolerance_fraction: float = 0.15) -> Tuple[int, float]:
    """
    Check if a distance matches a golden ratio harmonic.
    
    From framework: Optimal distances follow d = base × φ^n
    
    Args:
        distance: The distance to check (Angstroms)
        base_distance: Base distance (default: 3.8 Å for CA-CA)
        max_harmonic: Maximum harmonic to check (default: 4)
        tolerance_fraction: Relative tolerance for matching (default: 15%)
    
    Returns:
        Tuple of (best_harmonic_n, match_quality)
        - best_harmonic_n: Which φ^n power matches best (-1 if none)
        - match_quality: Quality of match [0, 1]
    """
    if distance <= 0:
        return (-1, 0.0)
    
    best_n = -1
    best_quality = 0.0
    
    for n in range(max_harmonic + 1):
        target = base_distance * (PHI ** n)
        tolerance = target * tolerance_fraction
        
        if abs(distance - target) < tolerance:
            # Calculate quality as inverse of relative error
            relative_error = abs(distance - target) / target
            quality = math.exp(-10 * relative_error)  # Same formula as lock strength
            
            if quality > best_quality:
                best_n = n
                best_quality = quality
    
    return (best_n, best_quality)


def analyze_distance_pattern(distances: List[float]) -> Dict[str, Any]:
    """
    Analyze a set of distances for golden ratio patterns.
    
    This is useful for evaluating protein structure quality based
    on how well inter-residue distances follow φ-harmonics.
    
    Args:
        distances: List of distances to analyze
    
    Returns:
        Dictionary with analysis results:
        - phi_matches: Count of distances matching each harmonic
        - total_quality: Sum of match qualities
        - mean_quality: Average match quality
        - pattern_strength: Overall pattern strength [0, 1]
    """
    if not distances:
        return {
            'phi_matches': {},
            'total_quality': 0.0,
            'mean_quality': 0.0,
            'pattern_strength': 0.0
        }
    
    phi_matches = {n: 0 for n in range(-1, 5)}  # -1 = no match
    qualities = []
    
    for d in distances:
        n, quality = golden_distance_match(d)
        phi_matches[n] += 1
        qualities.append(quality)
    
    # Calculate pattern strength (fraction of distances matching some harmonic)
    matched = sum(count for n, count in phi_matches.items() if n >= 0)
    pattern_strength = matched / len(distances) if distances else 0.0
    
    return {
        'phi_matches': phi_matches,
        'total_quality': sum(qualities),
        'mean_quality': sum(qualities) / len(qualities) if qualities else 0.0,
        'pattern_strength': pattern_strength
    }


# =============================================================================
# RESONANCE-BASED ENERGY TERMS
# =============================================================================

def calculate_resonance_energy(distances: List[Tuple[int, int, float]],
                                sequence_length: int,
                                weight: float = 1.0) -> float:
    """
    Calculate energy contribution from resonance lock patterns.
    
    Distances that match golden ratio harmonics are rewarded (negative energy).
    Distances that don't match are neutral (zero energy).
    
    This guides the protein toward conformations with φ-harmonic spacing.
    
    Args:
        distances: List of (residue_i, residue_j, distance) tuples
        sequence_length: Total sequence length (for normalization)
        weight: Energy weight multiplier (default: 1.0)
    
    Returns:
        Resonance energy contribution (kcal/mol, negative = favorable)
    """
    if not distances:
        return 0.0
    
    total_energy = 0.0
    
    for i, j, d in distances:
        # Check for golden ratio match
        harmonic, quality = golden_distance_match(d)
        
        if harmonic >= 0 and quality > 0.5:
            # Reward φ-harmonic distances
            # Scale by sequence separation (longer-range contacts more valuable)
            seq_sep = abs(j - i)
            separation_bonus = min(1.0 + seq_sep / 20.0, 2.0)
            
            # Energy reward (negative = favorable)
            total_energy -= weight * quality * separation_bonus
    
    # Normalize by sequence length
    return total_energy / max(1, sequence_length / 10)


def calculate_impedance_coherence_energy(
    residue_impedances: List[float],
    contact_pairs: List[Tuple[int, int]],
    weight: float = 1.0
) -> float:
    """
    Calculate energy from impedance coherence between contacting residues.
    
    Contacts between similar-impedance residues are favorable (like with like).
    This drives hydrophobic core formation and charge clustering.
    
    Args:
        residue_impedances: Impedance value for each residue
        contact_pairs: List of (residue_i, residue_j) contact pairs
        weight: Energy weight multiplier
    
    Returns:
        Impedance coherence energy (negative = favorable)
    """
    if not contact_pairs or not residue_impedances:
        return 0.0
    
    total_energy = 0.0
    
    for i, j in contact_pairs:
        if i < len(residue_impedances) and j < len(residue_impedances):
            z_i = residue_impedances[i]
            z_j = residue_impedances[j]
            
            # Calculate impedance match
            match = log_impedance_match(z_i, z_j)
            
            # Good impedance match = favorable energy
            if match > 0.7:
                total_energy -= weight * match
    
    return total_energy / max(1, len(contact_pairs))


# =============================================================================
# STABILITY CLASSIFICATION (from AstroFolds)
# =============================================================================

def classify_stability(score: float) -> str:
    """
    Classify stability based on combined score.
    
    From AstroFolds validation:
        S > 2.5 → Highly Stable (> 1 Gyr orbital equivalent)
        2.0 < S < 2.5 → Stable (10 Myr - 1 Gyr)
        1.5 < S < 2.0 → Marginally Stable
        S < 1.5 → Unstable
    
    For proteins, we scale by sequence-dependent normalization.
    """
    if score > 2.5:
        return "HIGHLY_STABLE"
    elif score > 2.0:
        return "STABLE"
    elif score > 1.5:
        return "MARGINAL"
    else:
        return "UNSTABLE"


# =============================================================================
# PROTEIN-SPECIFIC APPLICATIONS
# =============================================================================

def analyze_protein_resonance(
    coordinates: List[Tuple[float, float, float]],
    sequence: str,
    contact_threshold: float = 8.0
) -> Dict[str, Any]:
    """
    Perform complete resonance analysis on a protein structure.
    
    This is the main entry point for framework-based structure analysis.
    
    Args:
        coordinates: List of CA coordinates [(x, y, z), ...]
        sequence: Amino acid sequence
        contact_threshold: Distance threshold for contacts (default: 8.0 Å)
    
    Returns:
        Comprehensive analysis dictionary
    """
    n = len(coordinates)
    
    if n < 2:
        return {'error': 'Need at least 2 residues'}
    
    # Calculate all pairwise distances
    distances = []
    contact_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(
                (coordinates[j][0] - coordinates[i][0])**2 +
                (coordinates[j][1] - coordinates[i][1])**2 +
                (coordinates[j][2] - coordinates[i][2])**2
            )
            distances.append((i, j, d))
            if d < contact_threshold:
                contact_pairs.append((i, j))
    
    # Analyze distance patterns
    dist_values = [d for _, _, d in distances]
    phi_analysis = analyze_distance_pattern(dist_values)
    
    # Calculate resonance lock matrix for sequential neighbors
    seq_distances = [d for i, j, d in distances if j == i + 1]
    if len(seq_distances) > 1:
        lock_matrix = pairwise_resonance_matrix(seq_distances)
        mean_lock = sum(sum(row) for row in lock_matrix) / (len(lock_matrix) ** 2)
    else:
        mean_lock = 0.0
    
    # Calculate resonance energy
    res_energy = calculate_resonance_energy(distances, n)
    
    # Calculate overall stability score
    stability_score = phi_analysis['mean_quality'] * 2.0 + mean_lock
    
    return {
        'n_residues': n,
        'n_contacts': len(contact_pairs),
        'phi_pattern_analysis': phi_analysis,
        'mean_sequential_lock': mean_lock,
        'resonance_energy': res_energy,
        'stability_score': stability_score,
        'stability_classification': classify_stability(stability_score),
        'phi_percentage': phi_analysis['pattern_strength'] * 100,
    }


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FRAMEWORK RESONANCE MODULE TEST")
    print("=" * 70)
    
    # Test resonance lock strength
    print("\n📊 RESONANCE LOCK STRENGTH TESTS:")
    test_ratios = [
        (1.0, 2.0, "1:2 (perfect 2:1)"),
        (1.0, 1.5, "2:3"),
        (1.0, 1.618, "1:φ (golden)"),
        (1.0, 1.732, "1:√3"),
        (1.0, 3.14159, "1:π"),
    ]
    for v1, v2, desc in test_ratios:
        L = resonance_lock_strength(v1, v2)
        n1, n2, err = find_best_integer_ratio(v2 / v1)
        print(f"  {desc}: L = {L:.4f} (best ratio: {n1}:{n2}, error: {err:.4f})")
    
    # Test golden distance matching
    print("\n📐 GOLDEN RATIO DISTANCE MATCHING:")
    for n in range(5):
        d = CA_CA_DISTANCE * (PHI ** n)
        print(f"  φ^{n}: {d:.2f} Å")
    
    print("\n  Testing distances:")
    test_distances = [3.8, 5.0, 6.15, 8.0, 10.0, 16.0, 26.0]
    for d in test_distances:
        n, q = golden_distance_match(d)
        match_str = f"φ^{n}" if n >= 0 else "no match"
        print(f"    {d:.1f} Å → {match_str} (quality: {q:.3f})")
    
    # Test logarithmic impedance matching
    print("\n⚡ LOGARITHMIC IMPEDANCE MATCHING:")
    test_pairs = [
        (2.64, 2.77, "Cu-Zn (enzyme cofactors)"),
        (2.44, 2.77, "Fe-Zn"),
        (1.15, 1.38, "Na-Li (ion mimicry)"),
        (6.96, 9.38, "C-O atoms"),
    ]
    for z1, z2, desc in test_pairs:
        R = log_impedance_match(z1, z2)
        print(f"  {desc}: R = {R:.4f}")
    
    print("\n" + "=" * 70)
