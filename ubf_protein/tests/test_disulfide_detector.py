"""
Unit tests for DisulfideDetector and DisulfideBond models.

Tests cover:
- DisulfideBond data model validation
- PDB SSBOND record parsing (0, 1, 3+ bonds)
- Sequence-based disulfide prediction
- Invalid PDB format handling
- Bond constraint satisfaction checking
- Bond validation against sequence

Run with: pytest ubf_protein/tests/test_disulfide_detector.py -v
"""

import pytest
import tempfile
from pathlib import Path
from typing import List

# Import modules under test
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ubf_protein.models import DisulfideBond
from ubf_protein.disulfide_detector import DisulfideDetector


# ============================================================================
# DisulfideBond Model Tests
# ============================================================================

class TestDisulfideBondModel:
    """Test DisulfideBond immutable data model."""
    
    def test_basic_creation(self):
        """Test basic DisulfideBond creation with default parameters."""
        bond = DisulfideBond(residue_i=5, residue_j=55)
        
        assert bond.residue_i == 5
        assert bond.residue_j == 55
        assert bond.distance == 3.8
        assert bond.tolerance == 1.0
    
    def test_custom_parameters(self):
        """Test DisulfideBond with custom distance and tolerance."""
        bond = DisulfideBond(
            residue_i=10,
            residue_j=20,
            distance=4.0,
            tolerance=0.5
        )
        
        assert bond.distance == 4.0
        assert bond.tolerance == 0.5
    
    def test_immutability(self):
        """Test that DisulfideBond is immutable (frozen dataclass)."""
        bond = DisulfideBond(residue_i=5, residue_j=55)
        
        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
            bond.residue_i = 10
    
    def test_validation_negative_residue(self):
        """Test validation rejects negative residue indices."""
        with pytest.raises(ValueError, match="must be non-negative"):
            DisulfideBond(residue_i=-1, residue_j=55)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            DisulfideBond(residue_i=5, residue_j=-5)
    
    def test_validation_same_residue(self):
        """Test validation rejects bonds between same residue."""
        with pytest.raises(ValueError, match="must be different"):
            DisulfideBond(residue_i=5, residue_j=5)
    
    def test_validation_negative_distance(self):
        """Test validation rejects non-positive distance."""
        with pytest.raises(ValueError, match="distance must be positive"):
            DisulfideBond(residue_i=5, residue_j=55, distance=0.0)
        
        with pytest.raises(ValueError, match="distance must be positive"):
            DisulfideBond(residue_i=5, residue_j=55, distance=-1.0)
    
    def test_validation_negative_tolerance(self):
        """Test validation rejects negative tolerance."""
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            DisulfideBond(residue_i=5, residue_j=55, tolerance=-0.5)
    
    def test_is_satisfied_exact_distance(self):
        """Test is_satisfied returns True for exact target distance."""
        bond = DisulfideBond(residue_i=5, residue_j=55, distance=3.8, tolerance=1.0)
        
        assert bond.is_satisfied(3.8) is True
    
    def test_is_satisfied_within_tolerance(self):
        """Test is_satisfied returns True for distances within tolerance."""
        bond = DisulfideBond(residue_i=5, residue_j=55, distance=3.8, tolerance=1.0)
        
        # Just within upper bound
        assert bond.is_satisfied(4.8) is True
        # Just within lower bound
        assert bond.is_satisfied(2.8) is True
        # Slightly within
        assert bond.is_satisfied(4.0) is True
        assert bond.is_satisfied(3.5) is True
    
    def test_is_satisfied_outside_tolerance(self):
        """Test is_satisfied returns False for distances outside tolerance."""
        bond = DisulfideBond(residue_i=5, residue_j=55, distance=3.8, tolerance=1.0)
        
        # Just outside upper bound
        assert bond.is_satisfied(4.81) is False
        # Just outside lower bound
        assert bond.is_satisfied(2.79) is False
        # Far outside
        assert bond.is_satisfied(6.0) is False
        assert bond.is_satisfied(1.0) is False
    
    def test_get_violation_satisfied(self):
        """Test get_violation returns 0.0 for satisfied bonds."""
        bond = DisulfideBond(residue_i=5, residue_j=55, distance=3.8, tolerance=1.0)
        
        assert bond.get_violation(3.8) == 0.0
        assert bond.get_violation(4.5) == 0.0
        assert bond.get_violation(3.0) == 0.0
    
    def test_get_violation_unsatisfied(self):
        """Test get_violation returns correct deviation for violated bonds."""
        bond = DisulfideBond(residue_i=5, residue_j=55, distance=3.8, tolerance=1.0)
        
        # Distance 5.0: deviation = 1.2, tolerance = 1.0, violation = 0.2
        assert abs(bond.get_violation(5.0) - 0.2) < 1e-6
        
        # Distance 2.0: deviation = 1.8, tolerance = 1.0, violation = 0.8
        assert abs(bond.get_violation(2.0) - 0.8) < 1e-6
    
    def test_string_representation(self):
        """Test string representation for debugging."""
        bond = DisulfideBond(residue_i=5, residue_j=55)
        bond_str = str(bond)
        
        assert "CYS5" in bond_str
        assert "CYS55" in bond_str
        assert "3.8" in bond_str
        assert "1.0" in bond_str


# ============================================================================
# DisulfideDetector PDB Parsing Tests
# ============================================================================

class TestDisulfideDetectorPDB:
    """Test DisulfideDetector PDB file parsing."""
    
    def create_pdb_file(self, content: str) -> Path:
        """Helper to create temporary PDB file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    def test_detect_zero_bonds(self):
        """Test PDB file with no SSBOND records."""
        pdb_content = """
HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1       0.000   0.000   0.000
ATOM      2  CA  GLY A   2       3.800   0.000   0.000
END
"""
        pdb_file = self.create_pdb_file(pdb_content)
        
        try:
            detector = DisulfideDetector()
            bonds = detector.detect_from_pdb(str(pdb_file))
            
            assert len(bonds) == 0
            assert isinstance(bonds, list)
        finally:
            pdb_file.unlink()
    
    def test_detect_one_bond(self):
        """Test PDB file with one SSBOND record."""
        pdb_content = """
HEADER    TEST PROTEIN
SSBOND   1 CYS A    6    CYS A  127
ATOM      1  CA  CYS A   6       0.000   0.000   0.000
ATOM      2  CA  CYS A 127       3.800   0.000   0.000
END
"""
        pdb_file = self.create_pdb_file(pdb_content)
        
        try:
            detector = DisulfideDetector()
            bonds = detector.detect_from_pdb(str(pdb_file))
            
            assert len(bonds) == 1
            bond = bonds[0]
            assert bond.residue_i == 5  # 0-indexed (6-1)
            assert bond.residue_j == 126  # 0-indexed (127-1)
            assert bond.distance == 3.8
            assert bond.tolerance == 1.0
        finally:
            pdb_file.unlink()
    
    def test_detect_three_bonds(self):
        """Test PDB file with three SSBOND records (like Crambin)."""
        pdb_content = """
HEADER    CRAMBIN
SSBOND   1 CYS A    3    CYS A   40
SSBOND   2 CYS A    4    CYS A   32
SSBOND   3 CYS A   16    CYS A   26
ATOM      1  CA  CYS A   3       0.000   0.000   0.000
END
"""
        pdb_file = self.create_pdb_file(pdb_content)
        
        try:
            detector = DisulfideDetector()
            bonds = detector.detect_from_pdb(str(pdb_file))
            
            assert len(bonds) == 3
            
            # Check first bond
            assert bonds[0].residue_i == 2  # 0-indexed
            assert bonds[0].residue_j == 39
            
            # Check second bond
            assert bonds[1].residue_i == 3
            assert bonds[1].residue_j == 31
            
            # Check third bond
            assert bonds[2].residue_i == 15
            assert bonds[2].residue_j == 25
        finally:
            pdb_file.unlink()
    
    def test_detect_with_chain_filter(self):
        """Test chain filtering during detection."""
        pdb_content = """
HEADER    MULTI-CHAIN
SSBOND   1 CYS A    6    CYS A  127
SSBOND   2 CYS B   10    CYS B   50
SSBOND   3 CYS A   20    CYS A   80
END
"""
        pdb_file = self.create_pdb_file(pdb_content)
        
        try:
            detector = DisulfideDetector()
            
            # Detect only chain A bonds
            bonds_a = detector.detect_from_pdb(str(pdb_file), chain_id='A')
            assert len(bonds_a) == 2
            
            # Detect only chain B bonds
            bonds_b = detector.detect_from_pdb(str(pdb_file), chain_id='B')
            assert len(bonds_b) == 1
            
            # Detect all chains
            bonds_all = detector.detect_from_pdb(str(pdb_file))
            assert len(bonds_all) == 3
        finally:
            pdb_file.unlink()
    
    def test_file_not_found(self):
        """Test error handling for missing PDB file."""
        detector = DisulfideDetector()
        
        with pytest.raises(FileNotFoundError):
            detector.detect_from_pdb("nonexistent_file.pdb")
    
    def test_malformed_ssbond_line(self):
        """Test graceful handling of malformed SSBOND records."""
        pdb_content = """
HEADER    TEST
SSBOND   1 CYS A    6    CYS A  127
SSBOND   MALFORMED LINE
SSBOND   2 CYS A   20
SSBOND   3 CYS A   30    CYS A   90
END
"""
        pdb_file = self.create_pdb_file(pdb_content)
        
        try:
            detector = DisulfideDetector()
            bonds = detector.detect_from_pdb(str(pdb_file))
            
            # Should only detect valid bonds (1 and 3)
            assert len(bonds) == 2
        finally:
            pdb_file.unlink()
    
    def test_bonds_ordered_by_index(self):
        """Test that bonds are always ordered with residue_i < residue_j."""
        pdb_content = """
HEADER    TEST
SSBOND   1 CYS A  127    CYS A    6
END
"""
        pdb_file = self.create_pdb_file(pdb_content)
        
        try:
            detector = DisulfideDetector()
            bonds = detector.detect_from_pdb(str(pdb_file))
            
            assert len(bonds) == 1
            # Should be reordered so residue_i < residue_j
            assert bonds[0].residue_i == 5
            assert bonds[0].residue_j == 126
        finally:
            pdb_file.unlink()


# ============================================================================
# DisulfideDetector Sequence Prediction Tests
# ============================================================================

class TestDisulfideDetectorSequence:
    """Test DisulfideDetector sequence-based prediction."""
    
    def test_predict_zero_cysteines(self):
        """Test prediction with no cysteines in sequence."""
        detector = DisulfideDetector()
        bonds = detector.predict_from_sequence("ARNDQEGHILKMFPSTWYV")
        
        assert len(bonds) == 0
    
    def test_predict_one_cysteine(self):
        """Test prediction with only one cysteine."""
        detector = DisulfideDetector()
        bonds = detector.predict_from_sequence("ACDEFGH")
        
        assert len(bonds) == 0  # Need at least 2 to form a bond
    
    def test_predict_two_cysteines(self):
        """Test prediction with two cysteines."""
        detector = DisulfideDetector()
        # Cysteines at positions 1 and 15
        bonds = detector.predict_from_sequence("ACDEFGHIKLMNPQC")
        
        assert len(bonds) == 1
        bond = bonds[0]
        assert bond.residue_i == 1
        assert bond.residue_j == 14
    
    def test_predict_four_cysteines(self):
        """Test prediction with four cysteines (should form 2 bonds)."""
        detector = DisulfideDetector()
        # Cysteines at positions 0, 10, 20, 30
        sequence = "CDEFGHIKLMCDEFGHIKLMCDEFGHIKLMC"
        bonds = detector.predict_from_sequence(sequence)
        
        # Pairs nearest unpaired: (0,10) and (20,30)
        assert len(bonds) == 2
        assert bonds[0].residue_i == 0
        assert bonds[0].residue_j == 10
        assert bonds[1].residue_i == 20
        assert bonds[1].residue_j == 30
    
    def test_predict_six_cysteines(self):
        """Test prediction with six cysteines (like Crambin)."""
        detector = DisulfideDetector()
        # Cysteines at positions 2, 3, 15, 25, 37, 42
        # Create sequence with better spacing
        sequence = "AACCDEFGHIKLMNPCDEFGHIKLMCDEFGHIKLMNPCDEFGC"
        bonds = detector.predict_from_sequence(sequence)
        
        # With nearest pairing: (2,15), (3,25), (37,42) - only pairs 5 apart minimum
        # But position 37-42 is only 5 apart, so won't pair
        # Should get (2,15) and (3,25) only
        assert len(bonds) >= 2
    
    def test_predict_too_close_rejected(self):
        """Test that cysteines too close in sequence are not paired."""
        detector = DisulfideDetector()
        # Cysteines at positions 0, 5 (only 5 apart, < 10 minimum)
        bonds = detector.predict_from_sequence("CDEFGC")
        
        assert len(bonds) == 0  # Too close to form realistic bond
    
    def test_predict_max_sequence_distance(self):
        """Test max sequence distance constraint."""
        detector = DisulfideDetector()
        # Cysteines at positions 0 and 100
        sequence = "C" + "A" * 99 + "C"
        
        # Without constraint - should pair
        bonds_no_limit = detector.predict_from_sequence(sequence)
        assert len(bonds_no_limit) == 1
        
        # With constraint - should not pair
        bonds_limited = detector.predict_from_sequence(sequence, max_sequence_distance=50)
        assert len(bonds_limited) == 0
    
    def test_predict_case_insensitive(self):
        """Test that prediction works with lowercase sequence."""
        detector = DisulfideDetector()
        bonds = detector.predict_from_sequence("acdefghiklmnpqc")
        
        assert len(bonds) == 1
        assert bonds[0].residue_i == 1
        assert bonds[0].residue_j == 14


# ============================================================================
# DisulfideDetector Advanced Features Tests
# ============================================================================

class TestDisulfideDetectorAdvanced:
    """Test advanced DisulfideDetector features."""
    
    def test_detect_with_fallback_pdb_success(self):
        """Test fallback detection when PDB is available."""
        pdb_content = """
SSBOND   1 CYS A    6    CYS A  127
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            pdb_file = Path(f.name)
        
        try:
            detector = DisulfideDetector()
            sequence = "ACDEFGHIKLMNPQC"
            bonds, method = detector.detect_with_fallback(sequence, str(pdb_file))
            
            assert method == 'pdb'
            assert len(bonds) == 1
        finally:
            pdb_file.unlink()
    
    def test_detect_with_fallback_sequence(self):
        """Test fallback to sequence when PDB unavailable."""
        detector = DisulfideDetector()
        sequence = "ACDEFGHIKLMNPQC"
        bonds, method = detector.detect_with_fallback(sequence, pdb_file=None)
        
        assert method == 'sequence'
        assert len(bonds) == 1
    
    def test_detect_with_fallback_none(self):
        """Test fallback when no cysteines in sequence."""
        detector = DisulfideDetector()
        sequence = "ARNDQEGHILKMFPSTWYV"
        bonds, method = detector.detect_with_fallback(sequence)
        
        assert method == 'none'
        assert len(bonds) == 0
    
    def test_validate_bonds_success(self):
        """Test validation passes for valid bonds."""
        detector = DisulfideDetector()
        sequence = "ACDEFGHIKLMNPQC"
        bonds = [DisulfideBond(residue_i=1, residue_j=14)]
        
        is_valid, errors = detector.validate_bonds(bonds, sequence)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_bonds_out_of_bounds(self):
        """Test validation detects out-of-bounds indices."""
        detector = DisulfideDetector()
        sequence = "ACDEFGH"  # Length 7
        bonds = [DisulfideBond(residue_i=1, residue_j=10)]  # Index 10 out of bounds
        
        is_valid, errors = detector.validate_bonds(bonds, sequence)
        
        assert is_valid is False
        assert len(errors) >= 1
        assert "exceeds sequence length" in errors[0]
    
    def test_validate_bonds_not_cysteine(self):
        """Test validation detects non-cysteine residues."""
        detector = DisulfideDetector()
        sequence = "ACDEFGHIKLMNPQA"  # Position 14 is 'A', not 'C'
        bonds = [DisulfideBond(residue_i=1, residue_j=14)]
        
        is_valid, errors = detector.validate_bonds(bonds, sequence)
        
        assert is_valid is False
        assert any("not cysteine" in err for err in errors)
    
    def test_validate_bonds_overlapping(self):
        """Test validation detects overlapping bonds."""
        detector = DisulfideDetector()
        sequence = "CDEFGHIKLMCDEFGHIKLMC"
        bonds = [
            DisulfideBond(residue_i=0, residue_j=10),
            DisulfideBond(residue_i=0, residue_j=20)  # Residue 0 used twice
        ]
        
        is_valid, errors = detector.validate_bonds(bonds, sequence)
        
        assert is_valid is False
        assert any("multiple bonds" in err for err in errors)
    
    def test_custom_distance_tolerance(self):
        """Test detector with custom distance and tolerance."""
        detector = DisulfideDetector(default_distance=4.0, default_tolerance=0.5)
        
        sequence = "ACDEFGHIKLMNPQC"
        bonds = detector.predict_from_sequence(sequence)
        
        assert len(bonds) == 1
        assert bonds[0].distance == 4.0
        assert bonds[0].tolerance == 0.5


# ============================================================================
# Integration Tests
# ============================================================================

class TestDisulfideDetectorIntegration:
    """Integration tests for complete workflows."""
    
    def test_crambin_workflow(self):
        """Test complete workflow for Crambin-like protein."""
        # Crambin has 46 residues with 3 disulfide bonds
        detector = DisulfideDetector()
        
        # Actual Crambin sequence
        sequence = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
        
        # Predict bonds from sequence
        predicted_bonds = detector.predict_from_sequence(sequence)
        
        # Should predict some bonds (simple predictor won't match exact PDB bonds)
        assert len(predicted_bonds) >= 1
        
        # Validate predicted bonds  
        is_valid, errors = detector.validate_bonds(predicted_bonds, sequence)
        
        # Print errors for debugging if validation fails
        if not is_valid:
            print(f"\nValidation errors: {errors}")
            print(f"Predicted bonds: {predicted_bonds}")
        
        # All predicted bonds should be valid (cysteines in right places)
        assert is_valid is True, f"Validation failed: {errors}"
    
    def test_end_to_end_pdb_workflow(self):
        """Test end-to-end workflow: PDB → Detection → Validation."""
        pdb_content = """
HEADER    TEST PROTEIN
SSBOND   1 CYS A    6    CYS A  127
SSBOND   2 CYS A   20    CYS A   80
ATOM      1  CA  CYS A   6       0.000   0.000   0.000
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            pdb_file = Path(f.name)
        
        try:
            detector = DisulfideDetector()
            
            # Step 1: Detect from PDB
            bonds = detector.detect_from_pdb(str(pdb_file))
            assert len(bonds) == 2
            
            # Step 2: Create sequence with cysteines at correct positions
            sequence = "A" * 130
            sequence_list = list(sequence)
            sequence_list[5] = 'C'   # Position 6 (0-indexed)
            sequence_list[126] = 'C'  # Position 127
            sequence_list[19] = 'C'  # Position 20
            sequence_list[79] = 'C'  # Position 80
            sequence = ''.join(sequence_list)
            
            # Step 3: Validate bonds against sequence
            is_valid, errors = detector.validate_bonds(bonds, sequence)
            assert is_valid is True
            
            # Step 4: Check constraint satisfaction
            # Simulate distance checking
            for bond in bonds:
                assert bond.is_satisfied(3.8) is True
                assert bond.is_satisfied(5.0) is False
        finally:
            pdb_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
