"""
Secondary Structure Calculation Utility

Calculates helix/sheet/coil assignment from CA coordinates using
distance-based geometry analysis. Based on SimpleQuantumDSSP logic
but optimized for real-time use.

This is used to provide secondary structure breakdown during live
monitoring and in results analysis.
"""
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SecondaryStructureResult:
    """Result of secondary structure calculation"""
    # Per-residue assignment: 'H' (helix), 'E' (sheet), 'C' (coil)
    assignments: List[str]
    
    # Counts
    helix_count: int
    sheet_count: int
    coil_count: int
    
    # Percentages
    helix_percent: float
    sheet_percent: float
    coil_percent: float
    
    # Segments (contiguous stretches)
    helix_segments: List[Tuple[int, int]]  # (start, end) indices
    sheet_segments: List[Tuple[int, int]]
    coil_segments: List[Tuple[int, int]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'assignments': ''.join(self.assignments),  # Compact string like "HHHCCCEEE"
            'helix_count': self.helix_count,
            'sheet_count': self.sheet_count,
            'coil_count': self.coil_count,
            'helix_percent': round(self.helix_percent, 1),
            'sheet_percent': round(self.sheet_percent, 1),
            'coil_percent': round(self.coil_percent, 1),
            'helix_segments': self.helix_segments,
            'sheet_segments': self.sheet_segments,
            'coil_segments': self.coil_segments,
            'total_residues': len(self.assignments),
        }


def calculate_distance(p1: Tuple[float, float, float], 
                       p2: Tuple[float, float, float]) -> float:
    """Calculate Euclidean distance between two 3D points"""
    return math.sqrt(
        (p1[0] - p2[0])**2 + 
        (p1[1] - p2[1])**2 + 
        (p1[2] - p2[2])**2
    )


def calculate_secondary_structure(
    coordinates: List[Tuple[float, float, float]],
    sequence: Optional[str] = None
) -> SecondaryStructureResult:
    """
    Calculate secondary structure from CA coordinates.
    
    Uses distance-based criteria:
    - Alpha helix: CA(i) to CA(i+4) distance ~5.5-6.5 Å
    - 3-10 helix: CA(i) to CA(i+3) distance ~5.0-6.0 Å  
    - Beta sheet: Extended conformation, CA(i) to CA(i+2) distance ~6.5-7.5 Å
    
    Args:
        coordinates: List of (x, y, z) CA coordinates
        sequence: Optional amino acid sequence (not currently used)
    
    Returns:
        SecondaryStructureResult with assignments and statistics
    """
    n_res = len(coordinates)
    
    if n_res < 4:
        # Too short for meaningful SS assignment
        assignments = ['C'] * n_res
        return _build_result(assignments)
    
    # Initialize all as coil
    ss = ['C'] * n_res
    
    # Calculate key distances
    # i to i+3 distances (for 3-10 helix)
    # i to i+4 distances (for alpha helix)
    # i to i+2 distances (for extended strand)
    
    # Pass 1: Identify alpha helices (i,i+4 pattern)
    for i in range(n_res - 4):
        dist_i4 = calculate_distance(coordinates[i], coordinates[i+4])
        
        # Alpha helix has CA(i)-CA(i+4) distance of ~5.5-6.5 Å
        if 5.3 <= dist_i4 <= 6.8:
            # Check additional helix criteria: i,i+3 distance
            dist_i3 = calculate_distance(coordinates[i], coordinates[i+3])
            if 4.5 <= dist_i3 <= 6.0:
                # Mark residues i through i+4 as helix
                for j in range(i, i+5):
                    if ss[j] == 'C':  # Don't override existing assignments
                        ss[j] = 'H'
    
    # Pass 2: Identify 3-10 helices (i,i+3 pattern) where not already helix
    for i in range(n_res - 3):
        if ss[i] == 'H':
            continue  # Already alpha helix
            
        dist_i3 = calculate_distance(coordinates[i], coordinates[i+3])
        
        # 3-10 helix has CA(i)-CA(i+3) distance of ~5.0-6.0 Å
        if 4.8 <= dist_i3 <= 5.8:
            # Mark as helix (using 'H' for simplicity)
            for j in range(i, i+4):
                if ss[j] == 'C':
                    ss[j] = 'H'
    
    # Pass 3: Identify extended strands (beta sheets)
    # Look for extended conformations where i,i+2 distance is large
    for i in range(n_res - 2):
        if ss[i] != 'C':
            continue  # Already assigned
            
        dist_i2 = calculate_distance(coordinates[i], coordinates[i+2])
        
        # Extended strand has CA(i)-CA(i+2) distance of ~6.5-7.5 Å
        if 6.2 <= dist_i2 <= 7.8:
            # Also check the adjacent i+1,i+3 distance for consistency
            if i + 3 < n_res:
                dist_i1_i3 = calculate_distance(coordinates[i+1], coordinates[i+3])
                if 6.0 <= dist_i1_i3 <= 7.8:
                    # Mark as extended strand
                    for j in range(i, min(i+3, n_res)):
                        if ss[j] == 'C':
                            ss[j] = 'E'
    
    # Pass 4: Clean up short segments (min 3 residues for helix, 2 for sheet)
    ss = _cleanup_short_segments(ss, min_helix=3, min_sheet=2)
    
    return _build_result(ss)


def _cleanup_short_segments(
    ss: List[str], 
    min_helix: int = 3, 
    min_sheet: int = 2
) -> List[str]:
    """Remove segments shorter than minimum length"""
    result = list(ss)
    n = len(result)
    
    i = 0
    while i < n:
        current = result[i]
        
        # Find end of current segment
        j = i
        while j < n and result[j] == current:
            j += 1
        
        segment_length = j - i
        
        # Check minimum lengths
        if current == 'H' and segment_length < min_helix:
            for k in range(i, j):
                result[k] = 'C'
        elif current == 'E' and segment_length < min_sheet:
            for k in range(i, j):
                result[k] = 'C'
        
        i = j
    
    return result


def _build_result(assignments: List[str]) -> SecondaryStructureResult:
    """Build SecondaryStructureResult from assignments list"""
    n = len(assignments)
    
    # Count residues
    helix_count = sum(1 for s in assignments if s == 'H')
    sheet_count = sum(1 for s in assignments if s == 'E')
    coil_count = sum(1 for s in assignments if s == 'C')
    
    # Calculate percentages
    helix_percent = (helix_count / n * 100) if n > 0 else 0
    sheet_percent = (sheet_count / n * 100) if n > 0 else 0
    coil_percent = (coil_count / n * 100) if n > 0 else 0
    
    # Find segments
    helix_segments = _find_segments(assignments, 'H')
    sheet_segments = _find_segments(assignments, 'E')
    coil_segments = _find_segments(assignments, 'C')
    
    return SecondaryStructureResult(
        assignments=assignments,
        helix_count=helix_count,
        sheet_count=sheet_count,
        coil_count=coil_count,
        helix_percent=helix_percent,
        sheet_percent=sheet_percent,
        coil_percent=coil_percent,
        helix_segments=helix_segments,
        sheet_segments=sheet_segments,
        coil_segments=coil_segments,
    )


def _find_segments(assignments: List[str], target: str) -> List[Tuple[int, int]]:
    """Find contiguous segments of a given type"""
    segments = []
    n = len(assignments)
    
    i = 0
    while i < n:
        if assignments[i] == target:
            start = i
            while i < n and assignments[i] == target:
                i += 1
            segments.append((start, i - 1))
        else:
            i += 1
    
    return segments


def ss_from_string(ss_string: str) -> SecondaryStructureResult:
    """
    Build SecondaryStructureResult from a pre-computed SS string.
    
    Args:
        ss_string: String like "HHHCCCEEE" where each char is H/E/C
    
    Returns:
        SecondaryStructureResult
    """
    assignments = list(ss_string.upper())
    return _build_result(assignments)


def estimate_ss_from_sequence(sequence: str) -> SecondaryStructureResult:
    """
    Estimate secondary structure propensities from sequence alone.
    
    This is a rough estimate based on amino acid propensities and is
    used when no structure is available yet (initial state).
    
    Propensities based on Chou-Fasman parameters:
    - High helix: A, E, L, M, Q, K, R
    - High sheet: V, I, Y, F, W, T
    - High coil/turn: G, P, S, N, D
    """
    # Simplified propensity scores
    HELIX_FAVORING = set('AELMQKR')
    SHEET_FAVORING = set('VIYFTW')
    COIL_FAVORING = set('GPSND')
    
    n = len(sequence)
    if n < 4:
        return _build_result(['C'] * n)
    
    # Score each position
    scores = []
    for aa in sequence.upper():
        if aa in HELIX_FAVORING:
            scores.append(('H', 1.0))
        elif aa in SHEET_FAVORING:
            scores.append(('E', 1.0))
        elif aa in COIL_FAVORING:
            scores.append(('C', 1.0))
        else:
            scores.append(('C', 0.5))  # Neutral
    
    # Smooth with sliding window to create segments
    assignments = []
    window = 4
    
    for i in range(n):
        # Get window around position
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        
        # Count propensities in window
        h_count = sum(1 for s, _ in scores[start:end] if s == 'H')
        e_count = sum(1 for s, _ in scores[start:end] if s == 'E')
        c_count = sum(1 for s, _ in scores[start:end] if s == 'C')
        
        # Assign based on majority in window
        if h_count > e_count and h_count > c_count:
            assignments.append('H')
        elif e_count > h_count and e_count > c_count:
            assignments.append('E')
        else:
            assignments.append('C')
    
    # Cleanup short segments
    assignments = _cleanup_short_segments(assignments, min_helix=4, min_sheet=3)
    
    return _build_result(assignments)
