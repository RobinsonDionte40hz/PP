"""
Disulfide Bond Detection Module

This module provides functionality for detecting disulfide bonds in proteins
from PDB files (SSBOND records) or predicting them from sequence (cysteine positions).

Classes:
    DisulfideDetector: Main class for disulfide bond detection

Example:
    >>> detector = DisulfideDetector()
    >>> bonds = detector.detect_from_pdb("path/to/1crn.pdb")
    >>> print(f"Found {len(bonds)} disulfide bonds")
    Found 3 disulfide bonds
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple
from .models import DisulfideBond


class DisulfideDetector:
    """
    Detector for disulfide bonds in protein structures.
    
    Provides two detection methods:
    1. PDB file parsing - Reads SSBOND records from PDB files
    2. Sequence prediction - Predicts likely bonds from cysteine positions
    
    The detector handles edge cases including:
    - PDB files with zero, one, or multiple disulfide bonds
    - Invalid or malformed PDB formats
    - Sequences with varying numbers of cysteines
    
    Attributes:
        default_distance: Default target CA-CA distance for disulfide bonds (3.8 Å)
        default_tolerance: Default tolerance for constraint satisfaction (1.0 Å)
    """
    
    def __init__(
        self, 
        default_distance: float = 3.8,
        default_tolerance: float = 1.0
    ):
        """
        Initialize disulfide detector with default parameters.
        
        Args:
            default_distance: Default target CA-CA distance in Angstroms
            default_tolerance: Default tolerance in Angstroms
        """
        self.default_distance = default_distance
        self.default_tolerance = default_tolerance
    
    def detect_from_pdb(
        self, 
        pdb_file: str,
        chain_id: Optional[str] = None
    ) -> List[DisulfideBond]:
        """
        Parse SSBOND records from PDB file to detect disulfide bonds.
        
        PDB SSBOND record format:
        SSBOND   1 CYS A    6    CYS A  127                          1555   1555  2.03
        Columns: 8-10 (serial), 16 (chain1), 18-21 (resSeq1), 30 (chain2), 32-35 (resSeq2)
        
        Args:
            pdb_file: Path to PDB file
            chain_id: Optional chain identifier to filter bonds (e.g., 'A')
                     If None, detects bonds from all chains
        
        Returns:
            List of DisulfideBond objects, empty list if none found
            
        Raises:
            FileNotFoundError: If PDB file doesn't exist
            ValueError: If PDB file is malformed and cannot be parsed
            
        Example:
            >>> detector = DisulfideDetector()
            >>> bonds = detector.detect_from_pdb("1crn.pdb", chain_id='A')
            >>> for bond in bonds:
            ...     print(bond)
            DisulfideBond(CYS3 ↔ CYS40, target=3.8±1.0Å)
        """
        pdb_path = Path(pdb_file)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        bonds = []
        
        try:
            with open(pdb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # SSBOND records start with 'SSBOND'
                    if not line.startswith('SSBOND'):
                        continue
                    
                    # Parse SSBOND record
                    bond = self._parse_ssbond_line(line, chain_id)
                    if bond is not None:
                        bonds.append(bond)
        
        except Exception as e:
            raise ValueError(f"Error parsing PDB file {pdb_file}: {str(e)}")
        
        return bonds
    
    def _parse_ssbond_line(
        self, 
        line: str,
        chain_id: Optional[str] = None
    ) -> Optional[DisulfideBond]:
        """
        Parse a single SSBOND line from PDB file.
        
        Args:
            line: SSBOND record line from PDB file
            chain_id: Optional chain filter
            
        Returns:
            DisulfideBond object or None if line doesn't match criteria
        """
        # PDB SSBOND format (fixed width):
        # SSBOND   1 CYS A    6    CYS A  127
        # Positions (1-indexed in PDB spec, 0-indexed in Python):
        # 15: chain1
        # 17-21: resSeq1 (residue number)
        # 29: chain2  
        # 31-35: resSeq2 (residue number)
        
        try:
            # Ensure line is long enough
            if len(line) < 35:
                return None
            
            # Extract chain identifiers
            chain1 = line[15:16].strip()
            chain2 = line[29:30].strip()
            
            # Filter by chain if specified
            if chain_id is not None:
                if chain1 != chain_id or chain2 != chain_id:
                    return None
            
            # Extract residue numbers (convert to 0-based indexing)
            res1_str = line[17:21].strip()
            res2_str = line[31:35].strip()
            
            if not res1_str or not res2_str:
                return None
            
            # Parse residue numbers (PDB is 1-indexed, convert to 0-indexed)
            residue_i = int(res1_str) - 1
            residue_j = int(res2_str) - 1
            
            # Create disulfide bond (order residues by index)
            if residue_i > residue_j:
                residue_i, residue_j = residue_j, residue_i
            
            return DisulfideBond(
                residue_i=residue_i,
                residue_j=residue_j,
                distance=self.default_distance,
                tolerance=self.default_tolerance
            )
        
        except (ValueError, IndexError) as e:
            # Skip malformed lines silently (common in PDB files)
            return None
    
    def predict_from_sequence(
        self, 
        sequence: str,
        max_sequence_distance: Optional[int] = None
    ) -> List[DisulfideBond]:
        """
        Predict likely disulfide bonds from protein sequence based on cysteine positions.
        
        This is a simple heuristic predictor that pairs cysteines based on sequence
        proximity. For more accurate predictions, use structure-based detection or
        specialized prediction algorithms.
        
        Strategy:
        1. Find all cysteine (C) positions in sequence
        2. Pair cysteines by sequence proximity (nearest unpaired partners)
        3. Skip if cysteines are too close in sequence (< 10 residues apart)
        
        Args:
            sequence: Protein sequence as single-letter amino acid string
            max_sequence_distance: Maximum sequence separation for pairing
                                  If None, no maximum limit (pairs all cysteines)
        
        Returns:
            List of predicted DisulfideBond objects, empty if < 2 cysteines
            
        Example:
            >>> detector = DisulfideDetector()
            >>> bonds = detector.predict_from_sequence("ACDEFGCKLMNPC")
            >>> print(f"Predicted {len(bonds)} disulfide bonds")
            Predicted 1 disulfide bonds
        """
        # Find all cysteine positions (0-indexed)
        cysteine_positions = [
            i for i, aa in enumerate(sequence.upper()) 
            if aa == 'C'
        ]
        
        # Need at least 2 cysteines to form a bond
        if len(cysteine_positions) < 2:
            return []
        
        bonds = []
        paired = set()
        
        # Simple pairing strategy: pair nearest unpaired cysteines
        # Sort by position to process in sequence order
        sorted_positions = sorted(cysteine_positions)
        
        i = 0
        while i < len(sorted_positions):
            pos_i = sorted_positions[i]
            
            # Skip if already paired
            if pos_i in paired:
                i += 1
                continue
            
            # Find nearest unpaired partner
            for j in range(i + 1, len(sorted_positions)):
                pos_j = sorted_positions[j]
                
                # Skip if already paired
                if pos_j in paired:
                    continue
                
                # Skip if too close in sequence (< 10 residues)
                if pos_j - pos_i < 10:
                    continue
                
                # Check max distance constraint if specified
                if max_sequence_distance is not None:
                    if pos_j - pos_i > max_sequence_distance:
                        continue
                
                # Create bond (already ordered)
                bond = DisulfideBond(
                    residue_i=pos_i,
                    residue_j=pos_j,
                    distance=self.default_distance,
                    tolerance=self.default_tolerance
                )
                bonds.append(bond)
                
                # Mark both as paired
                paired.add(pos_i)
                paired.add(pos_j)
                break
            
            i += 1
        
        return bonds
    
    def detect_with_fallback(
        self,
        sequence: str,
        pdb_file: Optional[str] = None,
        chain_id: Optional[str] = None
    ) -> Tuple[List[DisulfideBond], str]:
        """
        Detect disulfide bonds with fallback from PDB to sequence prediction.
        
        Attempts to detect from PDB file first. If PDB file is not available
        or contains no SSBOND records, falls back to sequence-based prediction.
        
        Args:
            sequence: Protein sequence
            pdb_file: Optional path to PDB file
            chain_id: Optional chain identifier
            
        Returns:
            Tuple of (bonds list, detection method)
            Detection method is 'pdb', 'sequence', or 'none'
            
        Example:
            >>> detector = DisulfideDetector()
            >>> bonds, method = detector.detect_with_fallback(
            ...     sequence="ACDEFGCKLMNPC",
            ...     pdb_file="structure.pdb"
            ... )
            >>> print(f"Detected {len(bonds)} bonds using {method}")
        """
        bonds = []
        method = 'none'
        
        # Try PDB detection first
        if pdb_file is not None:
            try:
                bonds = self.detect_from_pdb(pdb_file, chain_id)
                if bonds:
                    method = 'pdb'
                    return bonds, method
            except (FileNotFoundError, ValueError):
                # Fall through to sequence prediction
                pass
        
        # Fall back to sequence prediction
        if not bonds:
            bonds = self.predict_from_sequence(sequence)
            if bonds:
                method = 'sequence'
        
        return bonds, method
    
    def validate_bonds(
        self,
        bonds: List[DisulfideBond],
        sequence: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate that detected disulfide bonds are consistent with sequence.
        
        Checks:
        1. Residue indices are within sequence bounds
        2. Both residues are cysteines in the sequence
        3. Bonds don't overlap (same residue in multiple bonds)
        
        Args:
            bonds: List of disulfide bonds to validate
            sequence: Protein sequence
            
        Returns:
            Tuple of (is_valid, error_messages)
            
        Example:
            >>> detector = DisulfideDetector()
            >>> bonds = [DisulfideBond(5, 55)]
            >>> is_valid, errors = detector.validate_bonds(bonds, sequence)
            >>> if not is_valid:
            ...     print(f"Validation errors: {errors}")
        """
        errors = []
        seq_len = len(sequence)
        used_positions = set()
        
        for bond in bonds:
            # Check bounds
            if bond.residue_i >= seq_len:
                errors.append(
                    f"Bond {bond}: residue_i ({bond.residue_i}) exceeds sequence length ({seq_len})"
                )
            if bond.residue_j >= seq_len:
                errors.append(
                    f"Bond {bond}: residue_j ({bond.residue_j}) exceeds sequence length ({seq_len})"
                )
            
            # Check if residues are cysteines
            if bond.residue_i < seq_len:
                if sequence[bond.residue_i].upper() != 'C':
                    errors.append(
                        f"Bond {bond}: residue {bond.residue_i} is {sequence[bond.residue_i]}, not cysteine"
                    )
            
            if bond.residue_j < seq_len:
                if sequence[bond.residue_j].upper() != 'C':
                    errors.append(
                        f"Bond {bond}: residue {bond.residue_j} is {sequence[bond.residue_j]}, not cysteine"
                    )
            
            # Check for overlapping bonds
            if bond.residue_i in used_positions:
                errors.append(
                    f"Bond {bond}: residue {bond.residue_i} appears in multiple bonds"
                )
            if bond.residue_j in used_positions:
                errors.append(
                    f"Bond {bond}: residue {bond.residue_j} appears in multiple bonds"
                )
            
            used_positions.add(bond.residue_i)
            used_positions.add(bond.residue_j)
        
        return len(errors) == 0, errors
