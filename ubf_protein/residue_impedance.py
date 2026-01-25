"""
Residue Impedance Calculator

Derives amino acid impedance (Z) from atomic composition using the
framework-validated atomic impedance formula:

    Z_atom = √(E_ion × χ) / r

Where:
    E_ion = First ionization energy (eV)
    χ = Pauling electronegativity
    r = Atomic radius (pm)

This module calculates Z_residue as the weighted average of side chain
atomic impedances, enabling impedance-based analysis of protein sequences.

Reference: Robinson, D. (2026). Computational Alchemy: Atomic Impedance
as a Unifying Principle for Chemical Behavior.

Author: Dionte Robinson
Date: January 2026
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


# =============================================================================
# ATOMIC IMPEDANCE DATA (from Computational Alchemy paper)
# =============================================================================

@dataclass(frozen=True)
class AtomicData:
    """Atomic properties for impedance calculation."""
    symbol: str
    ionization_energy_eV: float  # First ionization energy
    electronegativity: float     # Pauling scale
    radius_pm: float             # Covalent/atomic radius in picometers
    
    @property
    def impedance(self) -> float:
        """Calculate atomic impedance: Z = √(E_ion × χ) / r"""
        return math.sqrt(self.ionization_energy_eV * self.electronegativity) / (self.radius_pm / 100)


# Standard atomic data for biological elements
# Using radii consistent with Computational Alchemy paper
ATOMIC_DATA = {
    'H':  AtomicData('H',  13.598, 2.20, 53),   # Hydrogen (Z = 10.32)
    'C':  AtomicData('C',  11.260, 2.55, 77),   # Carbon (Z = 6.96)
    'N':  AtomicData('N',  14.534, 3.04, 75),   # Nitrogen (Z = 8.86)
    'O':  AtomicData('O',  13.618, 3.44, 73),   # Oxygen (Z = 9.38)
    'S':  AtomicData('S',  10.360, 2.58, 105),  # Sulfur (Z = 4.92) - KEY: closest to BRIDGE!
    'Se': AtomicData('Se', 9.752,  2.55, 120),  # Selenium (Z = 4.16) - also near BRIDGE
}

# Pre-calculated atomic impedances
ATOMIC_Z = {symbol: data.impedance for symbol, data in ATOMIC_DATA.items()}

# Note on interpretation:
# Biological amino acids are composed of C, N, O, H - all TAKERS (Z > 4)
# This is CORRECT - amino acids are the "TAKER" interface that coordinates
# BRIDGE metals (Fe, Cu, Zn with Z ≈ 2.5-3.0) for catalysis.
# 
# Key insight: Sulfur-containing residues (Cys, Met) have LOWER Z values
# because sulfur (Z ≈ 4.9) is much closer to the BRIDGE zone than C, N, O.
# This explains why Cys/Met are key metal-coordinating residues!


# =============================================================================
# AMINO ACID SIDE CHAIN DEFINITIONS
# =============================================================================

# Side chain composition (excluding backbone C, N, O, H)
# Format: {atom_symbol: count}
# Only heavy atoms + hydrogens that affect chemistry significantly

AMINO_ACID_SIDECHAINS = {
    # Non-polar, aliphatic
    'G': {},                                    # Glycine - no side chain
    'A': {'C': 1, 'H': 3},                     # Alanine - CH3
    'V': {'C': 3, 'H': 7},                     # Valine - CH(CH3)2
    'L': {'C': 4, 'H': 9},                     # Leucine - CH2CH(CH3)2
    'I': {'C': 4, 'H': 9},                     # Isoleucine - CH(CH3)CH2CH3
    'P': {'C': 3, 'H': 6},                     # Proline - cyclic (special)
    'M': {'C': 3, 'H': 7, 'S': 1},            # Methionine - CH2CH2SCH3
    
    # Aromatic
    'F': {'C': 7, 'H': 7},                     # Phenylalanine - CH2-phenyl
    'Y': {'C': 7, 'H': 7, 'O': 1},            # Tyrosine - CH2-phenol
    'W': {'C': 9, 'H': 8, 'N': 1},            # Tryptophan - CH2-indole
    
    # Polar, uncharged
    'S': {'C': 1, 'H': 3, 'O': 1},            # Serine - CH2OH
    'T': {'C': 2, 'H': 5, 'O': 1},            # Threonine - CH(OH)CH3
    'N': {'C': 2, 'H': 4, 'N': 1, 'O': 1},    # Asparagine - CH2CONH2
    'Q': {'C': 3, 'H': 6, 'N': 1, 'O': 1},    # Glutamine - CH2CH2CONH2
    'C': {'C': 1, 'H': 3, 'S': 1},            # Cysteine - CH2SH
    
    # Positively charged (at pH 7)
    'K': {'C': 4, 'H': 11, 'N': 1},           # Lysine - (CH2)4NH3+
    'R': {'C': 4, 'H': 11, 'N': 3},           # Arginine - (CH2)3NHC(NH2)2+
    'H': {'C': 4, 'H': 5, 'N': 2},            # Histidine - CH2-imidazole
    
    # Negatively charged (at pH 7)
    'D': {'C': 2, 'H': 3, 'N': 0, 'O': 2},    # Aspartate - CH2COO-
    'E': {'C': 3, 'H': 5, 'N': 0, 'O': 2},    # Glutamate - CH2CH2COO-
}

# Full amino acid names
AA_NAMES = {
    'G': 'Glycine', 'A': 'Alanine', 'V': 'Valine', 'L': 'Leucine',
    'I': 'Isoleucine', 'P': 'Proline', 'M': 'Methionine', 'F': 'Phenylalanine',
    'Y': 'Tyrosine', 'W': 'Tryptophan', 'S': 'Serine', 'T': 'Threonine',
    'N': 'Asparagine', 'Q': 'Glutamine', 'C': 'Cysteine', 'K': 'Lysine',
    'R': 'Arginine', 'H': 'Histidine', 'D': 'Aspartate', 'E': 'Glutamate',
}


# =============================================================================
# IMPEDANCE CATEGORIES (from Computational Alchemy framework)
# =============================================================================

class ImpedanceCategory(Enum):
    """Impedance-based functional categories."""
    GIVER = "GIVER"       # Z < 2.0 - electron donors
    BRIDGE = "BRIDGE"     # 2.0 ≤ Z ≤ 4.0 - catalytic sweet spot
    TAKER = "TAKER"       # Z > 4.0 - electron acceptors


def categorize_impedance(z: float) -> ImpedanceCategory:
    """Categorize impedance value."""
    if z < 2.0:
        return ImpedanceCategory.GIVER
    elif z <= 4.0:
        return ImpedanceCategory.BRIDGE
    else:
        return ImpedanceCategory.TAKER


def categorize_for_biology(z: float) -> str:
    """
    Categorize amino acid Z for biological interpretation.
    
    Since all amino acids are TAKERS (Z > 4), we use a finer scale:
    - SULFUR-RICH (Z ≈ 5-7): Best metal coordinators (Cys, Met)
    - MODERATE (Z ≈ 7-10): Good H-bond/ionic interactions
    - HIGH (Z > 10): Pure carbon chains, hydrophobic
    """
    if z < 7.0:
        return "SULFUR-RICH"  # Closest to BRIDGE zone
    elif z < 10.0:
        return "MODERATE"
    else:
        return "HIGH"


# =============================================================================
# RESIDUE IMPEDANCE CALCULATION
# =============================================================================

@dataclass
class ResidueImpedance:
    """Calculated impedance for an amino acid residue."""
    code: str
    name: str
    z_value: float
    category: ImpedanceCategory
    side_chain_atoms: Dict[str, int]
    atom_contributions: Dict[str, float]  # Weighted Z contribution per atom type
    
    def __str__(self) -> str:
        return f"{self.code} ({self.name}): Z = {self.z_value:.3f} [{self.category.value}]"


def calculate_residue_impedance(aa_code: str, 
                                 weighting: str = 'heavy') -> ResidueImpedance:
    """
    Calculate impedance for an amino acid from its side chain atomic composition.
    
    Args:
        aa_code: Single letter amino acid code
        weighting: 'heavy' (exclude H, default) or 'all' (include H)
    
    Returns:
        ResidueImpedance dataclass with calculated values
    
    Note: All amino acids are TAKERS (Z > 4) because they're made of C, N, O, H.
    The key insight is that SULFUR-containing residues (Cys, Met) have LOWER Z,
    making them better BRIDGE metal coordinators.
    """
    aa_code = aa_code.upper()
    
    if aa_code not in AMINO_ACID_SIDECHAINS:
        raise ValueError(f"Unknown amino acid code: {aa_code}")
    
    side_chain = AMINO_ACID_SIDECHAINS[aa_code]
    
    # Special case: Glycine has no side chain
    if not side_chain:
        # Use alpha carbon as reference
        return ResidueImpedance(
            code=aa_code,
            name=AA_NAMES.get(aa_code, 'Unknown'),
            z_value=ATOMIC_Z['C'],  # ~6.96, just carbon
            category=ImpedanceCategory.TAKER,
            side_chain_atoms={},
            atom_contributions={'Cα': ATOMIC_Z['C']}
        )
    
    # Calculate weighted average impedance (heavy atoms only by default)
    total_z = 0.0
    total_weight = 0.0
    contributions = {}
    
    for atom, count in side_chain.items():
        if atom not in ATOMIC_Z:
            continue
        
        # Skip hydrogens for heavy-atom weighting (default)
        if weighting == 'heavy' and atom == 'H':
            continue
            
        atom_z = ATOMIC_Z[atom]
        
        # Weight by atom count, but give extra weight to heteroatoms
        # (S, N, O are more chemically significant than C)
        if atom == 'S':
            weight = count * 2.0  # Sulfur is critical for metal coordination
        elif atom in ['N', 'O']:
            weight = count * 1.5  # Heteroatoms matter more
        else:
            weight = count * 1.0
        
        total_z += atom_z * weight
        total_weight += weight
        contributions[atom] = atom_z * count
    
    if total_weight == 0:
        z_value = ATOMIC_Z['C']  # Default to carbon
    else:
        z_value = total_z / total_weight
    
    return ResidueImpedance(
        code=aa_code,
        name=AA_NAMES.get(aa_code, 'Unknown'),
        z_value=z_value,
        category=categorize_impedance(z_value),
        side_chain_atoms=side_chain,
        atom_contributions=contributions
    )


def get_all_residue_impedances(weighting: str = 'average') -> Dict[str, ResidueImpedance]:
    """Calculate impedance for all standard amino acids."""
    return {
        aa: calculate_residue_impedance(aa, weighting)
        for aa in AMINO_ACID_SIDECHAINS.keys()
    }


# =============================================================================
# SEQUENCE ANALYSIS
# =============================================================================

@dataclass
class SequenceImpedanceAnalysis:
    """Complete impedance analysis for a protein sequence."""
    sequence: str
    length: int
    residue_impedances: List[ResidueImpedance]
    
    # Statistics
    mean_z: float
    std_z: float
    min_z: float
    max_z: float
    
    # Category distribution
    giver_count: int
    bridge_count: int
    taker_count: int
    
    # Metal coordination potential
    metal_coordination_residues: List[Tuple[int, str, float]]  # (position, aa, Z)
    
    # Sweet spot residues (Z ≈ 2.5-3.5)
    sweet_spot_residues: List[Tuple[int, str, float]]


def analyze_sequence_impedance(sequence: str) -> SequenceImpedanceAnalysis:
    """
    Perform complete impedance analysis on a protein sequence.
    
    Args:
        sequence: Amino acid sequence (1-letter codes)
    
    Returns:
        SequenceImpedanceAnalysis with full statistics
    """
    sequence = sequence.upper()
    residue_imps = []
    z_values = []
    
    giver_count = 0
    bridge_count = 0
    taker_count = 0
    
    metal_coord = []  # Potential metal coordination sites
    sweet_spot = []   # Catalytic sweet spot residues
    
    for i, aa in enumerate(sequence):
        if aa not in AMINO_ACID_SIDECHAINS:
            continue  # Skip unknown residues
            
        res_imp = calculate_residue_impedance(aa)
        residue_imps.append(res_imp)
        z_values.append(res_imp.z_value)
        
        # Count categories
        if res_imp.category == ImpedanceCategory.GIVER:
            giver_count += 1
        elif res_imp.category == ImpedanceCategory.BRIDGE:
            bridge_count += 1
        else:
            taker_count += 1
        
        # Check for metal coordination potential (Cys, His, Met, Asp, Glu)
        if aa in ['C', 'H', 'M', 'D', 'E']:
            metal_coord.append((i + 1, aa, res_imp.z_value))
        
        # Check for sweet spot (Z ≈ 2.5-3.5)
        if 2.5 <= res_imp.z_value <= 3.5:
            sweet_spot.append((i + 1, aa, res_imp.z_value))
    
    # Calculate statistics
    if z_values:
        mean_z = sum(z_values) / len(z_values)
        variance = sum((z - mean_z) ** 2 for z in z_values) / len(z_values)
        std_z = math.sqrt(variance)
        min_z = min(z_values)
        max_z = max(z_values)
    else:
        mean_z = std_z = min_z = max_z = 0.0
    
    return SequenceImpedanceAnalysis(
        sequence=sequence,
        length=len(sequence),
        residue_impedances=residue_imps,
        mean_z=mean_z,
        std_z=std_z,
        min_z=min_z,
        max_z=max_z,
        giver_count=giver_count,
        bridge_count=bridge_count,
        taker_count=taker_count,
        metal_coordination_residues=metal_coord,
        sweet_spot_residues=sweet_spot
    )


# =============================================================================
# IMPEDANCE MATCHING (from framework)
# =============================================================================

# Framework constant: logarithmic bandwidth
SIGMA_LOG = 1.5


def impedance_match_quality(z1: float, z2: float, sigma_log: float = SIGMA_LOG) -> float:
    """
    Calculate impedance matching quality using logarithmic formula.
    
    R(Z1, Z2) = exp[-(log Z1 - log Z2)² / (2σ_log²)]
    
    From the framework: σ_log = 1.5 means systems can couple efficiently
    if impedances are within ~3 orders of magnitude (factor of ~30×).
    
    Returns:
        R value from 0 (no coupling) to 1 (perfect match)
    """
    if z1 <= 0 or z2 <= 0:
        return 0.0
    
    log_diff = math.log10(z1) - math.log10(z2)
    return math.exp(-(log_diff ** 2) / (2 * sigma_log ** 2))


# Common cofactor/metal impedances for matching analysis
COFACTOR_Z = {
    'Zn2+': 2.77,   # Zinc - ubiquitous enzyme cofactor
    'Cu2+': 2.64,   # Copper - electron transfer
    'Cu+':  2.45,   # Reduced copper
    'Fe2+': 2.44,   # Iron (ferrous)
    'Fe3+': 2.89,   # Iron (ferric)
    'Mn2+': 2.31,   # Manganese
    'Co2+': 2.53,   # Cobalt (B12)
    'Ni2+': 2.56,   # Nickel
    'Mg2+': 1.82,   # Magnesium (borderline GIVER)
    'Ca2+': 1.54,   # Calcium (GIVER)
}


def analyze_metal_coordination(sequence: str) -> Dict[str, List[Tuple[int, str, float]]]:
    """
    Analyze which residues could coordinate which metals based on impedance matching.
    
    Returns dict mapping metal names to list of (position, residue, match_quality).
    """
    analysis = analyze_sequence_impedance(sequence)
    
    results = {}
    for metal, metal_z in COFACTOR_Z.items():
        matches = []
        for i, res_imp in enumerate(analysis.residue_impedances):
            # Only consider typical coordinating residues
            if res_imp.code in ['C', 'H', 'M', 'D', 'E', 'N', 'Q', 'S', 'T', 'Y']:
                quality = impedance_match_quality(res_imp.z_value, metal_z)
                if quality > 0.5:  # Significant match
                    matches.append((i + 1, res_imp.code, quality))
        
        if matches:
            results[metal] = sorted(matches, key=lambda x: -x[2])  # Sort by quality
    
    return results


# =============================================================================
# DISPLAY / REPORTING
# =============================================================================

def print_impedance_table():
    """Print a table of all amino acid impedances."""
    print("\n" + "=" * 75)
    print("AMINO ACID IMPEDANCE TABLE (Framework-Derived)")
    print("From: Robinson, D. (2026). Computational Alchemy")
    print("=" * 75)
    print(f"\n{'AA':<4} {'Name':<14} {'Z Value':<10} {'Bio-Category':<14} {'Metal Match':<12}")
    print("-" * 75)
    
    # Sort by Z value (lowest = best metal coordinator)
    all_imps = get_all_residue_impedances()
    sorted_aa = sorted(all_imps.items(), key=lambda x: x[1].z_value)
    
    for aa, imp in sorted_aa:
        bio_cat = categorize_for_biology(imp.z_value)
        
        # Calculate match quality to Zn2+ (Z=2.77) as reference
        zn_match = impedance_match_quality(imp.z_value, COFACTOR_Z['Zn2+'])
        match_str = f"R={zn_match:.2f}" if zn_match > 0.5 else "poor"
        
        # Highlight sulfur-containing
        marker = "⚡" if 'S' in imp.side_chain_atoms else "  "
        
        print(f"{aa:<4} {imp.name:<14} {imp.z_value:<10.2f} {bio_cat:<14} {match_str:<12} {marker}")
    
    print("-" * 75)
    print("\n📊 INTERPRETATION:")
    print("  • ALL amino acids are TAKERS (Z > 4) - made of C, N, O")
    print("  • SULFUR-RICH residues (⚡) have LOWEST Z - best metal coordinators")
    print("  • Catalytic metals (Fe, Cu, Zn) are BRIDGES (Z ≈ 2.5-3.0)")
    print("  • Lower Z = better impedance match to metal cofactors")
    print("\n📐 FRAMEWORK INSIGHT:")
    print("  Amino acids don't catalyze directly - they provide the")
    print("  IMPEDANCE-MATCHED INTERFACE to coordinate metal BRIDGES")
    print("=" * 75 + "\n")


def print_sequence_analysis(analysis: SequenceImpedanceAnalysis):
    """Print detailed sequence impedance analysis."""
    print("\n" + "=" * 75)
    print("SEQUENCE IMPEDANCE ANALYSIS")
    print("Framework: Robinson, D. (2026). Computational Alchemy")
    print("=" * 75)
    
    print(f"\nSequence: {analysis.sequence[:50]}{'...' if len(analysis.sequence) > 50 else ''}")
    print(f"Length: {analysis.length} residues")
    
    print(f"\n📊 IMPEDANCE STATISTICS:")
    print(f"  Mean Z:  {analysis.mean_z:.2f}")
    print(f"  Std Z:   {analysis.std_z:.2f}")
    print(f"  Range:   {analysis.min_z:.2f} - {analysis.max_z:.2f}")
    
    # Bio-category distribution
    sulfur_rich = sum(1 for imp in analysis.residue_impedances if imp.z_value < 7.0)
    moderate = sum(1 for imp in analysis.residue_impedances if 7.0 <= imp.z_value < 10.0)
    high = sum(1 for imp in analysis.residue_impedances if imp.z_value >= 10.0)
    total = len(analysis.residue_impedances)
    
    print(f"\n📁 BIO-CATEGORY DISTRIBUTION:")
    if total > 0:
        print(f"  SULFUR-RICH (Z < 7):   {sulfur_rich:3d} ({100*sulfur_rich/total:5.1f}%) - Best metal coordinators")
        print(f"  MODERATE (7-10):       {moderate:3d} ({100*moderate/total:5.1f}%) - H-bond/ionic")
        print(f"  HIGH (Z > 10):         {high:3d} ({100*high/total:5.1f}%) - Hydrophobic core")
    
    print(f"\n🔗 METAL COORDINATION POTENTIAL:")
    # Calculate average match to common metals
    for metal, metal_z in [('Zn2+', 2.77), ('Cu2+', 2.64), ('Fe2+', 2.44)]:
        matches = [imp for imp in analysis.residue_impedances 
                   if impedance_match_quality(imp.z_value, metal_z) > 0.7]
        if matches:
            best = min(matches, key=lambda x: abs(x.z_value - metal_z))
            print(f"  {metal}: {len(matches)} potential sites (best: {best.code} at pos)")
    
    print(f"\n🔧 LOWEST IMPEDANCE RESIDUES (Best Metal Coordinators):")
    sorted_imps = sorted(enumerate(analysis.residue_impedances), key=lambda x: x[1].z_value)
    for i, (pos, imp) in enumerate(sorted_imps[:8]):
        zn_match = impedance_match_quality(imp.z_value, 2.77)
        marker = "⚡" if 'S' in imp.side_chain_atoms else "  "
        print(f"  {marker} Position {pos+1:4d}: {imp.code} ({imp.name:<12}) Z = {imp.z_value:.2f}  R(Zn) = {zn_match:.2f}")
    
    print("\n" + "=" * 75)


def print_residue_profile(sequence: str):
    """Print per-residue impedance profile."""
    print("\n" + "=" * 70)
    print("RESIDUE IMPEDANCE PROFILE")
    print("=" * 70)
    print(f"\n{'Pos':<5} {'AA':<4} {'Z':<8} {'Category':<8} {'Bar':<30}")
    print("-" * 70)
    
    for i, aa in enumerate(sequence.upper()):
        if aa not in AMINO_ACID_SIDECHAINS:
            continue
        
        imp = calculate_residue_impedance(aa)
        
        # Visual bar (scale 0-10)
        bar_len = int(min(imp.z_value, 10) * 3)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        
        # Color coding via symbols
        if imp.category == ImpedanceCategory.GIVER:
            marker = "⬇"  # Low Z
        elif imp.category == ImpedanceCategory.BRIDGE:
            marker = "◆"  # Sweet spot
        else:
            marker = "⬆"  # High Z
        
        print(f"{i+1:<5} {aa:<4} {imp.z_value:<8.3f} {marker} {imp.category.value:<6} {bar}")
    
    print("-" * 70)
    print("Legend: ⬇ GIVER | ◆ BRIDGE (catalytic) | ⬆ TAKER")
    print("=" * 70 + "\n")


# =============================================================================
# STRUCTURAL CONSTRAINTS (for energy function integration)
# =============================================================================

@dataclass
class ImpedanceConstraint:
    """
    A single structural constraint derived from impedance analysis.
    
    Constraints represent predicted spatial relationships between residues
    based on their impedance properties (metal coordination, hydrophobic
    packing, salt bridges, etc.)
    """
    constraint_type: str           # 'metal_site', 'hydrophobic_core', 'salt_bridge', 'disulfide'
    residue_indices: List[int]     # 0-indexed positions in sequence
    target_distance: float         # Expected CA-CA distance in Angstroms
    tolerance: float               # Acceptable deviation from target
    strength: float                # Energy weight (higher = more important)
    description: str               # Human-readable description
    
    def __post_init__(self):
        """Validate constraint parameters."""
        if len(self.residue_indices) < 2:
            raise ValueError("Constraint must involve at least 2 residues")
        if self.target_distance <= 0:
            raise ValueError("Target distance must be positive")


@dataclass
class ImpedanceConstraintSet:
    """
    Complete set of impedance-derived constraints for a protein sequence.
    """
    sequence: str
    constraints: List[ImpedanceConstraint]
    
    # Statistics
    n_metal_sites: int = 0
    n_hydrophobic_cores: int = 0
    n_salt_bridges: int = 0
    n_disulfide_candidates: int = 0
    
    def __len__(self) -> int:
        return len(self.constraints)
    
    def by_type(self, constraint_type: str) -> List[ImpedanceConstraint]:
        """Get constraints of a specific type."""
        return [c for c in self.constraints if c.constraint_type == constraint_type]


def predict_structural_constraints(sequence: str, 
                                    include_metal_sites: bool = True,
                                    include_hydrophobic: bool = True,
                                    include_salt_bridges: bool = True,
                                    include_disulfides: bool = True) -> ImpedanceConstraintSet:
    """
    Predict structural constraints from sequence using impedance analysis.
    
    This function analyzes the sequence to identify residues that should be
    spatially close based on their impedance properties:
    
    1. Metal coordination sites: Cys/His clusters (low-Z residues that match metal BRIDGES)
    2. Hydrophobic core: Carbon-rich residues (high-Z) that should pack together
    3. Salt bridges: Oppositely charged residues (D/E with K/R)
    4. Disulfide candidates: Cysteine pairs
    
    Args:
        sequence: Amino acid sequence (1-letter codes)
        include_*: Flags to enable/disable specific constraint types
    
    Returns:
        ImpedanceConstraintSet with all predicted constraints
    """
    sequence = sequence.upper()
    constraints: List[ImpedanceConstraint] = []
    
    # Track counts
    n_metal = 0
    n_hydrophobic = 0
    n_salt = 0
    n_disulfide = 0
    
    # Get impedance analysis
    analysis = analyze_sequence_impedance(sequence)
    
    # ==========================================================================
    # 1. METAL COORDINATION SITES
    # ==========================================================================
    # Find clusters of low-Z coordinating residues (Cys, His, Met, Asp, Glu)
    # These residues should be spatially close to coordinate metal ions
    
    if include_metal_sites:
        # Metal-coordinating residues and their coordination distances
        metal_coordinators = {'C': 2.3, 'H': 2.2, 'M': 2.4, 'D': 2.1, 'E': 2.1}
        coord_positions = []
        
        for i, aa in enumerate(sequence):
            if aa in metal_coordinators:
                # Get impedance match quality to zinc (representative metal)
                imp = analysis.residue_impedances[i]
                match_quality = impedance_match_quality(imp.z_value, COFACTOR_Z['Zn2+'])
                if match_quality > 0.6:  # Good coordination potential
                    coord_positions.append((i, aa, match_quality))
        
        # Find clusters of 2-4 coordinating residues within sequence distance
        # Typical metal sites span 10-40 residues in sequence
        for i, (pos_i, aa_i, q_i) in enumerate(coord_positions):
            for j, (pos_j, aa_j, q_j) in enumerate(coord_positions[i+1:], i+1):
                seq_dist = abs(pos_j - pos_i)
                
                # Skip if too close in sequence (same secondary structure element)
                # or too far (unlikely to be same binding site)
                if seq_dist < 3 or seq_dist > 50:
                    continue
                
                # Cys-Cys or Cys-His pairs are strongest indicators
                is_strong_pair = (aa_i in 'CH' and aa_j in 'CH')
                
                # Calculate constraint strength from impedance match
                strength = (q_i + q_j) / 2.0 * (1.5 if is_strong_pair else 1.0)
                
                if strength > 0.5:
                    # Metal coordination geometry: 4-8 Å between coordinating atoms
                    # CA-CA distance is typically 6-10 Å for tetrahedral coordination
                    constraints.append(ImpedanceConstraint(
                        constraint_type='metal_site',
                        residue_indices=[pos_i, pos_j],
                        target_distance=8.0,  # CA-CA for metal coordination
                        tolerance=3.0,
                        strength=strength * 2.0,  # Metal sites are important
                        description=f"Metal coord: {aa_i}{pos_i+1}-{aa_j}{pos_j+1} (R={strength:.2f})"
                    ))
                    n_metal += 1
    
    # ==========================================================================
    # 2. HYDROPHOBIC CORE
    # ==========================================================================
    # High-Z (carbon-rich) residues should pack together in the protein core
    # This drives the hydrophobic collapse
    
    if include_hydrophobic:
        # Hydrophobic residues (aliphatic and aromatic)
        hydrophobic = set('VILMFYW')
        hydro_positions = [i for i, aa in enumerate(sequence) if aa in hydrophobic]
        
        # Create pairwise constraints for hydrophobic residues
        # that are separated in sequence but should be close in space
        for i, pos_i in enumerate(hydro_positions):
            for pos_j in hydro_positions[i+1:]:
                seq_dist = pos_j - pos_i
                
                # Only constrain if separated in sequence (will fold to be close)
                if seq_dist < 5:
                    continue
                
                # Closer pairs in sequence get stronger constraints
                # (more likely to actually contact)
                if seq_dist < 15:
                    strength = 0.8
                elif seq_dist < 30:
                    strength = 0.5
                else:
                    strength = 0.3
                
                # Only add significant constraints
                if strength > 0.4:
                    aa_i, aa_j = sequence[pos_i], sequence[pos_j]
                    constraints.append(ImpedanceConstraint(
                        constraint_type='hydrophobic_core',
                        residue_indices=[pos_i, pos_j],
                        target_distance=7.0,  # Typical core packing distance
                        tolerance=4.0,  # More tolerance for hydrophobic packing
                        strength=strength,
                        description=f"Hydrophobic: {aa_i}{pos_i+1}-{aa_j}{pos_j+1}"
                    ))
                    n_hydrophobic += 1
    
    # ==========================================================================
    # 3. SALT BRIDGES
    # ==========================================================================
    # Oppositely charged residues (K/R with D/E) form salt bridges
    # These provide electrostatic stabilization
    
    if include_salt_bridges:
        positive = set('KR')  # Positively charged
        negative = set('DE')  # Negatively charged
        
        pos_positions = [(i, sequence[i]) for i in range(len(sequence)) if sequence[i] in positive]
        neg_positions = [(i, sequence[i]) for i in range(len(sequence)) if sequence[i] in negative]
        
        for pos_i, aa_i in pos_positions:
            for pos_j, aa_j in neg_positions:
                seq_dist = abs(pos_j - pos_i)
                
                # Skip adjacent residues (no structural constraint)
                if seq_dist < 3:
                    continue
                
                # Local salt bridges (i, i+3 or i+4) are common in helices
                if 3 <= seq_dist <= 4:
                    strength = 1.2  # Strong - helix stabilization
                    target = 6.0
                elif seq_dist < 20:
                    strength = 0.7
                    target = 8.0
                else:
                    strength = 0.4
                    target = 10.0
                
                if strength > 0.5:
                    constraints.append(ImpedanceConstraint(
                        constraint_type='salt_bridge',
                        residue_indices=[pos_i, pos_j],
                        target_distance=target,
                        tolerance=3.0,
                        strength=strength,
                        description=f"Salt bridge: {aa_i}{pos_i+1}-{aa_j}{pos_j+1}"
                    ))
                    n_salt += 1
    
    # ==========================================================================
    # 4. DISULFIDE CANDIDATES
    # ==========================================================================
    # Cysteine pairs that could form disulfide bonds
    # These are very strong structural constraints
    
    if include_disulfides:
        cys_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
        
        for i, pos_i in enumerate(cys_positions):
            for pos_j in cys_positions[i+1:]:
                seq_dist = pos_j - pos_i
                
                # Disulfides need some sequence separation
                if seq_dist < 4:
                    continue
                
                # Typical disulfide: CA-CA distance ~5.5 Å
                constraints.append(ImpedanceConstraint(
                    constraint_type='disulfide',
                    residue_indices=[pos_i, pos_j],
                    target_distance=5.5,
                    tolerance=1.5,  # Tight constraint
                    strength=2.0,   # Strong - covalent bond
                    description=f"Disulfide candidate: C{pos_i+1}-C{pos_j+1}"
                ))
                n_disulfide += 1
    
    return ImpedanceConstraintSet(
        sequence=sequence,
        constraints=constraints,
        n_metal_sites=n_metal,
        n_hydrophobic_cores=n_hydrophobic,
        n_salt_bridges=n_salt,
        n_disulfide_candidates=n_disulfide
    )


def calculate_impedance_restraint_energy(
    constraints: ImpedanceConstraintSet,
    atom_coordinates: List[Tuple[float, float, float]],
    tolerance: float = 1.5,
    scale_factor: float = 5.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate energy penalty for violating impedance-derived constraints.
    
    Uses a flat-bottom harmonic potential:
    - Zero penalty within tolerance of target distance
    - Quadratic penalty outside tolerance
    
    This is called by energy_function.py during structure prediction.
    
    Args:
        constraints: ImpedanceConstraintSet from predict_structural_constraints()
        atom_coordinates: Current CA coordinates [(x,y,z), ...]
        tolerance: Additional tolerance in Angstroms (added to constraint tolerance)
        scale_factor: Energy scaling factor
    
    Returns:
        Tuple of (total_energy, details_dict)
    """
    import math
    
    if constraints is None or len(constraints) == 0:
        return 0.0, {'n_satisfied': 0, 'n_violated': 0, 'violations': []}
    
    total_energy = 0.0
    n_satisfied = 0
    n_violated = 0
    violations = []
    
    n_atoms = len(atom_coordinates)
    
    for constraint in constraints.constraints:
        # Get residue positions
        indices = constraint.residue_indices
        
        # Skip if indices out of range
        if any(idx >= n_atoms or idx < 0 for idx in indices):
            continue
        
        # Calculate pairwise distances for multi-residue constraints
        for i, idx_i in enumerate(indices):
            for idx_j in indices[i+1:]:
                coord_i = atom_coordinates[idx_i]
                coord_j = atom_coordinates[idx_j]
                
                # Euclidean distance
                dist = math.sqrt(
                    (coord_j[0] - coord_i[0])**2 +
                    (coord_j[1] - coord_i[1])**2 +
                    (coord_j[2] - coord_i[2])**2
                )
                
                # Flat-bottom harmonic potential
                target = constraint.target_distance
                tol = constraint.tolerance + tolerance
                
                if dist < target - tol:
                    # Too close - repulsive
                    deviation = (target - tol) - dist
                    energy = constraint.strength * scale_factor * deviation**2
                    n_violated += 1
                    violations.append({
                        'type': constraint.constraint_type,
                        'residues': indices,
                        'dist': dist,
                        'target': target,
                        'energy': energy
                    })
                elif dist > target + tol:
                    # Too far - attractive
                    deviation = dist - (target + tol)
                    energy = constraint.strength * scale_factor * deviation**2
                    n_violated += 1
                    violations.append({
                        'type': constraint.constraint_type,
                        'residues': indices,
                        'dist': dist,
                        'target': target,
                        'energy': energy
                    })
                else:
                    # Within tolerance - no penalty
                    energy = 0.0
                    n_satisfied += 1
                
                total_energy += energy
    
    details = {
        'n_satisfied': n_satisfied,
        'n_violated': n_violated,
        'n_total': n_satisfied + n_violated,
        'violations': violations[:10],  # Limit for performance
        'energy_by_type': {}
    }
    
    # Summarize energy by constraint type
    for ctype in ['metal_site', 'hydrophobic_core', 'salt_bridge', 'disulfide']:
        type_energy = sum(v['energy'] for v in violations if v['type'] == ctype)
        if type_energy > 0:
            details['energy_by_type'][ctype] = type_energy
    
    return total_energy, details


def print_constraint_analysis(constraints: ImpedanceConstraintSet):
    """Print detailed constraint analysis."""
    print("\n" + "=" * 75)
    print("IMPEDANCE-DERIVED STRUCTURAL CONSTRAINTS")
    print("Framework: Robinson, D. (2026). Computational Alchemy")
    print("=" * 75)
    
    print(f"\nSequence: {constraints.sequence[:50]}{'...' if len(constraints.sequence) > 50 else ''}")
    print(f"Length: {len(constraints.sequence)} residues")
    
    print(f"\n📊 CONSTRAINT SUMMARY:")
    print(f"  Total constraints: {len(constraints)}")
    print(f"  Metal sites:       {constraints.n_metal_sites}")
    print(f"  Hydrophobic core:  {constraints.n_hydrophobic_cores}")
    print(f"  Salt bridges:      {constraints.n_salt_bridges}")
    print(f"  Disulfide cands:   {constraints.n_disulfide_candidates}")
    
    # Print by type
    for ctype, symbol, color in [
        ('metal_site', '🔗', 'Metal Coordination Sites'),
        ('disulfide', '⚡', 'Disulfide Bond Candidates'),
        ('salt_bridge', '±', 'Salt Bridges'),
        ('hydrophobic_core', '●', 'Hydrophobic Core Packing'),
    ]:
        type_constraints = constraints.by_type(ctype)
        if type_constraints:
            print(f"\n{symbol} {color}:")
            # Sort by strength
            sorted_c = sorted(type_constraints, key=lambda c: -c.strength)
            for c in sorted_c[:8]:  # Show top 8
                indices_str = '-'.join(str(i+1) for i in c.residue_indices)
                print(f"    {c.description:<45} target={c.target_distance:.1f}Å ±{c.tolerance:.1f}")
    
    print("\n" + "=" * 75)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Print the reference table
    print_impedance_table()
    
    # Test sequence (lysozyme fragment)
    test_seq = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"
    
    analysis = analyze_sequence_impedance(test_seq)
    print_sequence_analysis(analysis)
    print_residue_profile(test_seq)
    
    # Metal coordination analysis
    print("\n🔬 METAL COORDINATION MATCHING:")
    metal_matches = analyze_metal_coordination(test_seq)
    for metal, matches in metal_matches.items():
        print(f"\n  {metal}:")
        for pos, aa, quality in matches[:5]:
            print(f"    Position {pos}: {aa} (R = {quality:.3f})")
