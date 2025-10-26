"""
Amino acid physical and chemical properties database.

This module provides comprehensive property data for all 20 standard amino acids,
used for side-chain field calculations and interaction modeling.

Properties include:
- Charge: Electrostatic charge at pH 7.0 (-1.0, 0.0, +1.0)
- Hydrophobicity: Kyte-Doolittle scale (-4.5 to +4.5, higher = more hydrophobic)
- Volume: Van der Waals volume in Ų (cubic Angstroms)

References:
- Kyte & Doolittle (1982) J. Mol. Biol. 157:105-132
- Zamyatnin (1972) Prog. Biophys. Mol. Biol. 24:107-123
"""

from typing import Dict, Tuple

# Type alias for amino acid properties
AminoAcidProperties = Dict[str, Tuple[float, float, float]]

# Properties: (charge, hydrophobicity, volume)
AMINO_ACID_PROPERTIES: AminoAcidProperties = {
    # Nonpolar, aliphatic
    'G': (0.0, -0.40, 60.1),   # Glycine - smallest, most flexible
    'A': (0.0,  1.80, 88.6),   # Alanine - small hydrophobic
    'V': (0.0,  4.20, 140.0),  # Valine - branched hydrophobic
    'L': (0.0,  3.80, 166.7),  # Leucine - hydrophobic
    'I': (0.0,  4.50, 166.7),  # Isoleucine - most hydrophobic aliphatic
    
    # Aromatic
    'F': (0.0,  2.80, 189.9),  # Phenylalanine - hydrophobic aromatic
    'Y': (0.0, -1.30, 193.6),  # Tyrosine - polar aromatic (OH group)
    'W': (0.0, -0.90, 237.6),  # Tryptophan - largest, amphipathic
    
    # Polar, uncharged
    'S': (0.0, -0.80, 89.0),   # Serine - small polar
    'T': (0.0, -0.70, 116.1),  # Threonine - polar with methyl
    'C': (0.0,  2.50, 108.5),  # Cysteine - can form disulfides
    'M': (0.0,  1.90, 162.9),  # Methionine - hydrophobic sulfur
    'N': (0.0, -3.50, 114.1),  # Asparagine - polar amide
    'Q': (0.0, -3.50, 143.8),  # Glutamine - polar amide
    
    # Positively charged (basic)
    'K': (+1.0, -3.90, 168.6), # Lysine - long positively charged
    'R': (+1.0, -4.50, 173.4), # Arginine - most basic, guanidinium
    'H': (+0.5, -3.20, 153.2), # Histidine - weakly positive at pH 7
    
    # Negatively charged (acidic)
    'D': (-1.0, -3.50, 111.1), # Aspartate - short negatively charged
    'E': (-1.0, -3.50, 138.4), # Glutamate - longer negatively charged
    
    # Special
    'P': (0.0, -1.60, 112.7),  # Proline - rigid, helix breaker
}

def get_property(amino_acid: str, property_name: str) -> float:
    """
    Get a specific property value for an amino acid.
    
    Args:
        amino_acid: Single-letter amino acid code (case-insensitive)
        property_name: Property to retrieve ('charge', 'hydrophobicity', 'volume')
        
    Returns:
        Property value as float
        
    Raises:
        ValueError: If amino acid or property name is invalid
        
    Example:
        >>> get_property('W', 'hydrophobicity')
        -0.9
        >>> get_property('K', 'charge')
        1.0
    """
    amino_acid = amino_acid.upper()
    
    if amino_acid not in AMINO_ACID_PROPERTIES:
        raise ValueError(f"Unknown amino acid: {amino_acid}")
    
    property_index = {
        'charge': 0,
        'hydrophobicity': 1,
        'volume': 2
    }
    
    if property_name not in property_index:
        raise ValueError(f"Unknown property: {property_name}. "
                        f"Must be one of: {list(property_index.keys())}")
    
    props = AMINO_ACID_PROPERTIES[amino_acid]
    return props[property_index[property_name]]

def get_all_properties(amino_acid: str) -> Tuple[float, float, float]:
    """
    Get all properties for an amino acid.
    
    Args:
        amino_acid: Single-letter amino acid code (case-insensitive)
        
    Returns:
        Tuple of (charge, hydrophobicity, volume)
        
    Raises:
        ValueError: If amino acid is invalid
        
    Example:
        >>> get_all_properties('W')
        (0.0, -0.9, 237.6)
    """
    amino_acid = amino_acid.upper()
    
    if amino_acid not in AMINO_ACID_PROPERTIES:
        raise ValueError(f"Unknown amino acid: {amino_acid}")
    
    return AMINO_ACID_PROPERTIES[amino_acid]

def is_hydrophobic(amino_acid: str, threshold: float = 0.0) -> bool:
    """
    Check if amino acid is hydrophobic based on Kyte-Doolittle scale.
    
    Args:
        amino_acid: Single-letter amino acid code (case-insensitive)
        threshold: Hydrophobicity threshold (default 0.0)
        
    Returns:
        True if hydrophobicity > threshold
        
    Example:
        >>> is_hydrophobic('L')  # Leucine is hydrophobic
        True
        >>> is_hydrophobic('D')  # Aspartate is hydrophilic
        False
    """
    return get_property(amino_acid, 'hydrophobicity') > threshold

def is_charged(amino_acid: str) -> bool:
    """
    Check if amino acid is charged at pH 7.0.
    
    Args:
        amino_acid: Single-letter amino acid code (case-insensitive)
        
    Returns:
        True if charge is non-zero
        
    Example:
        >>> is_charged('K')  # Lysine is positively charged
        True
        >>> is_charged('A')  # Alanine is neutral
        False
    """
    return get_property(amino_acid, 'charge') != 0.0

def is_positive(amino_acid: str) -> bool:
    """Check if amino acid is positively charged."""
    return get_property(amino_acid, 'charge') > 0.0

def is_negative(amino_acid: str) -> bool:
    """Check if amino acid is negatively charged."""
    return get_property(amino_acid, 'charge') < 0.0


# Amino acid groups for easy reference
HYDROPHOBIC_RESIDUES = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'}
POLAR_RESIDUES = {'S', 'T', 'C', 'N', 'Q', 'Y'}
POSITIVELY_CHARGED_RESIDUES = {'K', 'R', 'H'}
NEGATIVELY_CHARGED_RESIDUES = {'D', 'E'}
AROMATIC_RESIDUES = {'F', 'Y', 'W'}
SMALL_RESIDUES = {'G', 'A', 'S'}
