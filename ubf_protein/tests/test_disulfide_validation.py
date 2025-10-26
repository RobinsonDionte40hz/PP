"""
Unit tests for disulfide bond validation in StructuralValidation.

Tests cover:
- Disulfide bond constraint validation
- Integration with existing validation system
- Performance requirements (<5ms)
- Edge cases and error handling

Run with: pytest ubf_protein/tests/test_disulfide_validation.py -v
"""

import pytest
import time
from typing import List, Tuple, Optional

# Import modules under test
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ubf_protein.models import Conformation, DisulfideBond
from ubf_protein.structural_validation import StructuralValidation, ValidationResult


def create_test_conformation(
    sequence: str = "ACDEFGHIKLMNPQRSTVWY",
    coordinates: Optional[List[Tuple[float, float, float]]] = None
) -> Conformation:
    """Helper to create test conformation."""
    if coordinates is None:
        # Create linear chain with 3.8 Å spacing
        coordinates = [(i * 3.8, 0.0, 0.0) for i in range(len(sequence))]
    
    return Conformation(
        conformation_id="test_conf",
        sequence=sequence,
        atom_coordinates=coordinates,
        energy=0.0,
        rmsd_to_native=0.0,
        secondary_structure=['C'] * len(sequence),
        phi_angles=[0.0] * len(sequence),
        psi_angles=[0.0] * len(sequence),
        available_move_types=[],
        structural_constraints={}
    )


# ============================================================================
# Basic Disulfide Validation Tests
# ============================================================================

class TestDisulfideValidationBasic:
    """Test basic disulfide bond validation functionality."""
    
    def test_validate_satisfied_bond(self):
        """Test validation passes for satisfied disulfide bond."""
        validator = StructuralValidation()
        
        # Create conformation with cysteines 3.8 Å apart (exact target)
        coords = [
            (0.0, 0.0, 0.0),   # Residue 0
            (3.8, 0.0, 0.0),   # Residue 1
            (7.6, 0.0, 0.0),   # Residue 2
            (11.4, 0.0, 0.0),  # Residue 3
            (15.2, 0.0, 0.0),  # Residue 4
            (19.0, 0.0, 0.0),  # Residue 5 - CYS
            (41.8, 0.0, 0.0),  # Residue 6
            (45.6, 0.0, 0.0),  # Residue 7
            (49.4, 0.0, 0.0),  # Residue 8
            (53.2, 0.0, 0.0),  # Residue 9
            (57.0, 0.0, 0.0),  # Residue 10 - CYS
        ]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        # Cysteines at positions 5 and 10 with exact 3.8 Å distance
        bonds = [DisulfideBond(residue_i=5, residue_j=10)]
        
        # Adjust coordinates so residues 5 and 10 are 3.8 Å apart
        coords[10] = (22.8, 0.0, 0.0)  # 3.8 Å from residue 5
        conf.atom_coordinates = coords
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is True
        assert len(violations) == 0
    
    def test_validate_bond_within_tolerance(self):
        """Test validation passes for bond within tolerance."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        # Bond with tolerance
        bonds = [DisulfideBond(residue_i=5, residue_j=10, distance=3.8, tolerance=1.0)]
        
        # Set distance to 4.5 Å (within 3.8 ± 1.0)
        coords[10] = (coords[5][0] + 4.5, 0.0, 0.0)
        conf.atom_coordinates = coords
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is True
        assert len(violations) == 0
    
    def test_validate_violated_bond_too_long(self):
        """Test validation fails for bond that's too long."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        bonds = [DisulfideBond(residue_i=5, residue_j=10, distance=3.8, tolerance=1.0)]
        
        # Set distance to 6.0 Å (exceeds 3.8 + 1.0 = 4.8 Å)
        coords[10] = (coords[5][0] + 6.0, 0.0, 0.0)
        conf.atom_coordinates = coords
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is False
        assert len(violations) == 1
        assert "CYS5-CYS10" in violations[0]
        assert "6.00" in violations[0]
        assert "violation" in violations[0].lower()
    
    def test_validate_violated_bond_too_short(self):
        """Test validation fails for bond that's too short."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        bonds = [DisulfideBond(residue_i=5, residue_j=10, distance=3.8, tolerance=1.0)]
        
        # Set distance to 2.0 Å (below 3.8 - 1.0 = 2.8 Å)
        coords[10] = (coords[5][0] + 2.0, 0.0, 0.0)
        conf.atom_coordinates = coords
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is False
        assert len(violations) == 1
        assert "CYS5-CYS10" in violations[0]
        assert "2.00" in violations[0]
    
    def test_validate_multiple_bonds_all_satisfied(self):
        """Test validation with multiple satisfied bonds."""
        validator = StructuralValidation()
        
        # Create sequence with 3 disulfide bonds (like Crambin)
        coords = [(i * 3.8, 0.0, 0.0) for i in range(15)]
        conf = create_test_conformation(
            sequence="ACCDEFGHIKLMNCC",
            coordinates=coords
        )
        
        bonds = [
            DisulfideBond(residue_i=1, residue_j=13),
            DisulfideBond(residue_i=2, residue_j=14),
        ]
        
        # Set distances to exact target
        coords[13] = (coords[1][0] + 3.8, 0.0, 0.0)
        coords[14] = (coords[2][0] + 3.8, 0.0, 0.0)
        conf.atom_coordinates = coords
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is True
        assert len(violations) == 0
    
    def test_validate_multiple_bonds_some_violated(self):
        """Test validation with some bonds satisfied and some violated."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(15)]
        conf = create_test_conformation(
            sequence="ACCDEFGHIKLMNCC",
            coordinates=coords
        )
        
        bonds = [
            DisulfideBond(residue_i=1, residue_j=13),
            DisulfideBond(residue_i=2, residue_j=14),
        ]
        
        # First bond satisfied, second violated
        coords[13] = (coords[1][0] + 3.8, 0.0, 0.0)  # Satisfied
        coords[14] = (coords[2][0] + 8.0, 0.0, 0.0)  # Violated (too long)
        conf.atom_coordinates = coords
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is False
        assert len(violations) == 1
        assert "CYS2-CYS14" in violations[0]
        assert "CYS1-CYS13" not in violations[0]


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestDisulfideValidationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_validate_empty_bond_list(self):
        """Test validation with empty bond list."""
        validator = StructuralValidation()
        conf = create_test_conformation()
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, [])
        
        assert is_valid is True
        assert len(violations) == 0
    
    def test_validate_bond_out_of_bounds(self):
        """Test validation detects out-of-bounds residue indices."""
        validator = StructuralValidation()
        conf = create_test_conformation(sequence="ACDEFGH")  # 7 residues
        
        # Bond with index 10 (out of bounds)
        bonds = [DisulfideBond(residue_i=5, residue_j=10)]
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is False
        assert len(violations) == 1
        assert "out of bounds" in violations[0].lower()
        assert "5-10" in violations[0]
    
    def test_validate_both_indices_out_of_bounds(self):
        """Test validation handles both indices out of bounds."""
        validator = StructuralValidation()
        conf = create_test_conformation(sequence="ACDEFGH")  # 7 residues
        
        bonds = [DisulfideBond(residue_i=10, residue_j=20)]
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is False
        assert len(violations) >= 1
    
    def test_validate_bond_with_custom_tolerance(self):
        """Test validation with custom tolerance."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        # Tight tolerance
        bonds = [DisulfideBond(residue_i=5, residue_j=10, distance=3.8, tolerance=0.2)]
        
        # Distance 4.2 Å (would pass with 1.0 tolerance, fails with 0.2)
        coords[10] = (coords[5][0] + 4.2, 0.0, 0.0)
        conf.atom_coordinates = coords
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is False
        assert len(violations) == 1
    
    def test_validate_zero_distance_bond(self):
        """Test validation handles zero distance (overlapping atoms)."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        bonds = [DisulfideBond(residue_i=5, residue_j=10)]
        
        # Set to same position (distance = 0)
        coords[10] = coords[5]
        conf.atom_coordinates = coords
        
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        
        assert is_valid is False
        assert len(violations) == 1


# ============================================================================
# Integration with Existing Validation
# ============================================================================

class TestDisulfideValidationIntegration:
    """Test integration with existing validation system."""
    
    def test_validate_conformation_with_disulfide_bonds(self):
        """Test full validate_conformation with disulfide bonds."""
        validator = StructuralValidation()
        
        # Create a properly folded loop structure
        # Make a hairpin where residues 5 and 10 can be close
        coords = [
            (0.0, 0.0, 0.0),    # 0
            (3.8, 0.0, 0.0),    # 1
            (7.6, 0.0, 0.0),    # 2
            (11.4, 0.0, 0.0),   # 3
            (15.2, 0.0, 0.0),   # 4
            (19.0, 0.0, 0.0),   # 5 - CYS
            (22.0, 2.0, 0.0),   # 6 - turn
            (23.5, 4.5, 0.0),   # 7 - top of turn
            (22.0, 7.0, 0.0),   # 8 - coming down
            (19.5, 7.8, 0.0),   # 9 - approaching 10
            (19.0, 3.8, 0.0),   # 10 - CYS (3.8 Å from residue 5)
        ]
        
        bonds = [DisulfideBond(residue_i=5, residue_j=10)]
        
        # Create conformation
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        # Call full validation with disulfide bonds
        result = validator.validate_conformation(conf, disulfide_bonds=bonds)
        
        # Debug if it fails
        if not result.is_valid:
            print(f"\nValidation issues: {result.issues}")
        
        assert result.is_valid is True
        assert len(result.issues) == 0
    
    def test_validate_conformation_disulfide_violation(self):
        """Test full validation detects disulfide violations."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        bonds = [DisulfideBond(residue_i=5, residue_j=10)]
        coords[10] = (coords[5][0] + 10.0, 0.0, 0.0)  # Violated
        conf.atom_coordinates = coords
        
        result = validator.validate_conformation(conf, disulfide_bonds=bonds)
        
        assert result.is_valid is False
        assert len(result.issues) >= 1
        assert any("CYS5-CYS10" in issue for issue in result.issues)
    
    def test_validate_conformation_without_disulfide_bonds(self):
        """Test backward compatibility - validation without disulfide bonds."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        # Call without disulfide_bonds parameter (backward compatible)
        result = validator.validate_conformation(conf)
        
        assert result.is_valid is True  # No disulfide checking
    
    def test_validate_conformation_combined_issues(self):
        """Test validation catches both standard and disulfide issues."""
        validator = StructuralValidation()
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence="ACDEFCGHIKC", coordinates=coords)
        
        # Create both a bond length issue and disulfide violation
        coords[1] = (10.0, 0.0, 0.0)  # Too far from residue 0 (bond length issue)
        
        bonds = [DisulfideBond(residue_i=5, residue_j=10)]
        coords[10] = (coords[5][0] + 10.0, 0.0, 0.0)  # Disulfide violation
        conf.atom_coordinates = coords
        
        result = validator.validate_conformation(conf, disulfide_bonds=bonds)
        
        assert result.is_valid is False
        assert len(result.issues) >= 2
        
        # Check both types of issues are reported
        has_bond_issue = any("Bond 0-1" in issue for issue in result.issues)
        has_disulfide_issue = any("CYS5-CYS10" in issue for issue in result.issues)
        
        assert has_bond_issue
        assert has_disulfide_issue


# ============================================================================
# Performance Tests
# ============================================================================

class TestDisulfideValidationPerformance:
    """Test performance requirements (<5ms per conformation)."""
    
    def test_performance_small_protein(self):
        """Test performance for small protein (~50 residues, 3 bonds)."""
        validator = StructuralValidation()
        
        # Crambin-like: 46 residues, 3 disulfide bonds
        sequence = "C" * 46
        coords = [(i * 3.8, 0.0, 0.0) for i in range(46)]
        conf = create_test_conformation(sequence=sequence, coordinates=coords)
        
        bonds = [
            DisulfideBond(residue_i=3, residue_j=40),
            DisulfideBond(residue_i=4, residue_j=32),
            DisulfideBond(residue_i=16, residue_j=26),
        ]
        
        # Warm-up
        validator.validate_disulfide_bonds(conf, bonds)
        
        # Measure performance
        start = time.perf_counter()
        for _ in range(100):
            validator.validate_disulfide_bonds(conf, bonds)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        
        assert avg_time_ms < 5.0, f"Performance target not met: {avg_time_ms:.3f}ms > 5ms"
    
    def test_performance_medium_protein(self):
        """Test performance for medium protein (~150 residues, 5 bonds)."""
        validator = StructuralValidation()
        
        sequence = "C" * 150
        coords = [(i * 3.8, 0.0, 0.0) for i in range(150)]
        conf = create_test_conformation(sequence=sequence, coordinates=coords)
        
        bonds = [
            DisulfideBond(residue_i=10, residue_j=50),
            DisulfideBond(residue_i=20, residue_j=60),
            DisulfideBond(residue_i=30, residue_j=70),
            DisulfideBond(residue_i=40, residue_j=80),
            DisulfideBond(residue_i=100, residue_j=140),
        ]
        
        # Warm-up
        validator.validate_disulfide_bonds(conf, bonds)
        
        # Measure performance
        start = time.perf_counter()
        for _ in range(100):
            validator.validate_disulfide_bonds(conf, bonds)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        
        assert avg_time_ms < 5.0, f"Performance target not met: {avg_time_ms:.3f}ms > 5ms"
    
    def test_performance_large_protein(self):
        """Test performance for large protein (~300 residues, 10 bonds)."""
        validator = StructuralValidation()
        
        sequence = "C" * 300
        coords = [(i * 3.8, 0.0, 0.0) for i in range(300)]
        conf = create_test_conformation(sequence=sequence, coordinates=coords)
        
        bonds = [
            DisulfideBond(residue_i=i * 30, residue_j=(i * 30) + 50)
            for i in range(10)
            if (i * 30) + 50 < 300
        ]
        
        # Warm-up
        validator.validate_disulfide_bonds(conf, bonds)
        
        # Measure performance
        start = time.perf_counter()
        for _ in range(100):
            validator.validate_disulfide_bonds(conf, bonds)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        
        assert avg_time_ms < 5.0, f"Performance target not met: {avg_time_ms:.3f}ms > 5ms"


# ============================================================================
# Real-World Workflow Tests
# ============================================================================

class TestDisulfideValidationWorkflows:
    """Test realistic protein folding workflows."""
    
    def test_crambin_workflow(self):
        """Test Crambin folding workflow with disulfide validation."""
        validator = StructuralValidation()
        
        # Crambin: 46 residues, 3 disulfide bonds
        sequence = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
        coords = [(i * 3.8, 0.0, 0.0) for i in range(46)]
        conf = create_test_conformation(sequence=sequence, coordinates=coords)
        
        # Real Crambin disulfide bonds (1-indexed in PDB, 0-indexed here)
        bonds = [
            DisulfideBond(residue_i=2, residue_j=39),   # CYS3-CYS40
            DisulfideBond(residue_i=3, residue_j=31),   # CYS4-CYS32
            DisulfideBond(residue_i=15, residue_j=25),  # CYS16-CYS26
        ]
        
        # Initial conformation - bonds not satisfied
        result = validator.validate_conformation(conf, disulfide_bonds=bonds)
        assert result.is_valid is False
        
        # Simulate folding - bring cysteines closer
        coords[39] = (coords[2][0] + 3.8, coords[2][1], coords[2][2])
        coords[31] = (coords[3][0] + 3.8, coords[3][1], coords[3][2])
        coords[25] = (coords[15][0] + 3.8, coords[15][1], coords[15][2])
        conf.atom_coordinates = coords
        
        # After folding - bonds satisfied
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        assert is_valid is True
        assert len(violations) == 0
    
    def test_iterative_folding_convergence(self):
        """Test iterative folding with disulfide bond feedback."""
        validator = StructuralValidation()
        
        sequence = "ACDEFCGHIKC"
        coords = [(i * 3.8, 0.0, 0.0) for i in range(11)]
        conf = create_test_conformation(sequence=sequence, coordinates=coords)
        
        bonds = [DisulfideBond(residue_i=5, residue_j=10)]
        
        # Initial distance: ~19 Å (too far)
        is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
        assert is_valid is False
        
        # Simulate iterative moves to bring cysteines closer
        distances = [10.0, 7.0, 5.0, 4.2, 3.8]
        
        for target_dist in distances:
            coords[10] = (coords[5][0] + target_dist, 0.0, 0.0)
            conf.atom_coordinates = coords
            
            is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
            
            if target_dist <= 4.8:  # Within tolerance (3.8 + 1.0)
                assert is_valid is True
            else:
                assert is_valid is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
