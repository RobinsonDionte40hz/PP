"""
Tests for disulfide-constrained move generation in MaplessMoveGenerator.

Tests Task 9: Integration of disulfide constraints into move generation system.
"""

import pytest
import math
from typing import List, Optional, Tuple

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from ubf_protein.models import DisulfideBond, Conformation, MoveType
from ubf_protein.mapless_moves import MaplessMoveGenerator


def create_test_conformation(sequence: str = "ACDEFGHIKLMNPQRSTVWYC",
                             atom_coordinates: Optional[List[Tuple[float, float, float]]] = None) -> Conformation:
    """Create a test conformation with specified coordinates."""
    if atom_coordinates is None:
        # Create default linear coordinates
        atom_coordinates = [(float(i), 0.0, 0.0) for i in range(len(sequence))]
    
    return Conformation(
        conformation_id="test_conf",
        sequence=sequence,
        atom_coordinates=atom_coordinates,
        energy=100.0,
        rmsd_to_native=5.0,
        secondary_structure=['C'] * len(sequence),
        phi_angles=[0.0] * len(sequence),
        psi_angles=[0.0] * len(sequence),
        available_move_types=[],
        structural_constraints={}
    )


class TestDisulfideMoveGeneration:
    """Tests for disulfide-constrained move generation."""

    def test_no_moves_for_satisfied_bonds(self):
        """Test that no moves are generated for satisfied disulfide bonds."""
        # Create conformation with cysteines at correct distance (3.8 Å)
        coords = [
            (0.0, 0.0, 0.0),   # Residue 0
            (3.8, 0.0, 0.0),   # Residue 1 - exactly at target distance
            (10.0, 0.0, 0.0),  # Residue 2
        ]
        conformation = create_test_conformation(sequence="CCC", atom_coordinates=coords)
        
        # Create bond between residues 0 and 1 (satisfied)
        bonds = [DisulfideBond(residue_i=0, residue_j=1, distance=3.8, tolerance=1.0)]
        
        # Generate moves
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        
        # Filter for disulfide constraint moves
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        # Should generate NO moves for satisfied bond
        assert len(disulfide_moves) == 0

    def test_moves_generated_for_violated_bonds(self):
        """Test that moves are generated for violated disulfide bonds."""
        # Create conformation with cysteines far apart (10.0 Å > 3.8 + 1.0)
        coords = [
            (0.0, 0.0, 0.0),   # Residue 0
            (10.0, 0.0, 0.0),  # Residue 1 - far apart (violation = 5.2 Å)
            (20.0, 0.0, 0.0),  # Residue 2
        ]
        conformation = create_test_conformation(sequence="CCC", atom_coordinates=coords)
        
        # Create bond between residues 0 and 1 (violated)
        bonds = [DisulfideBond(residue_i=0, residue_j=1, distance=3.8, tolerance=1.0)]
        
        # Generate moves
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        
        # Filter for disulfide constraint moves
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        # Should generate at least one move for violated bond
        assert len(disulfide_moves) >= 1
        
        # Check move properties
        move = disulfide_moves[0]
        assert move.move_type == MoveType.DISULFIDE_CONSTRAINT
        assert set(move.target_residues) == {0, 1}
        assert move.estimated_energy_change < 0  # Should be stabilizing

    def test_move_targets_both_residues(self):
        """Test that generated moves target both residues in the bond."""
        coords = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 10.0),  # Far apart in z-direction
        ]
        conformation = create_test_conformation(sequence="CC", atom_coordinates=coords)
        bonds = [DisulfideBond(residue_i=0, residue_j=1)]
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        assert len(disulfide_moves) >= 1
        move = disulfide_moves[0]
        assert len(move.target_residues) == 2
        assert 0 in move.target_residues
        assert 1 in move.target_residues

    def test_rmsd_change_is_step_size(self):
        """Test that RMSD change matches step size (0.5 Å)."""
        coords = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
        ]
        conformation = create_test_conformation(sequence="CC", atom_coordinates=coords)
        bonds = [DisulfideBond(residue_i=0, residue_j=1)]
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        assert len(disulfide_moves) >= 1
        move = disulfide_moves[0]
        assert move.estimated_rmsd_change == 0.5  # Step size

    def test_multiple_violated_bonds(self):
        """Test handling of multiple violated disulfide bonds."""
        # Create conformation with 4 cysteines, forming 2 bonds
        coords = [
            (0.0, 0.0, 0.0),   # Cys 0
            (10.0, 0.0, 0.0),  # Cys 1 - bond 0-1 violated
            (20.0, 0.0, 0.0),  # Cys 2
            (30.0, 0.0, 0.0),  # Cys 3 - bond 2-3 violated
        ]
        conformation = create_test_conformation(sequence="CCCC", atom_coordinates=coords)
        
        bonds = [
            DisulfideBond(residue_i=0, residue_j=1),
            DisulfideBond(residue_i=2, residue_j=3),
        ]
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        # Should generate moves for both violated bonds
        assert len(disulfide_moves) >= 2

    def test_mixed_satisfied_and_violated_bonds(self):
        """Test handling of mix of satisfied and violated bonds."""
        coords = [
            (0.0, 0.0, 0.0),   # Cys 0
            (3.8, 0.0, 0.0),   # Cys 1 - bond 0-1 satisfied
            (10.0, 0.0, 0.0),  # Cys 2
            (20.0, 0.0, 0.0),  # Cys 3 - bond 2-3 violated
        ]
        conformation = create_test_conformation(sequence="CCCC", atom_coordinates=coords)
        
        bonds = [
            DisulfideBond(residue_i=0, residue_j=1),  # Satisfied
            DisulfideBond(residue_i=2, residue_j=3),  # Violated
        ]
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        # Should generate move only for violated bond
        assert len(disulfide_moves) >= 1
        
        # Check that moves are only for violated bond (residues 2, 3)
        for move in disulfide_moves:
            assert set(move.target_residues) == {2, 3}

    def test_energy_change_scales_with_violation(self):
        """Test that energy change magnitude scales with violation size."""
        # Create two conformations with different violations
        coords_small = [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]  # Small violation
        coords_large = [(0.0, 0.0, 0.0), (15.0, 0.0, 0.0)]  # Large violation
        
        conf_small = create_test_conformation(sequence="CC", atom_coordinates=coords_small)
        conf_large = create_test_conformation(sequence="CC", atom_coordinates=coords_large)
        
        bonds = [DisulfideBond(residue_i=0, residue_j=1)]
        
        generator = MaplessMoveGenerator()
        
        moves_small = generator.generate_moves(conf_small, disulfide_bonds=bonds)
        moves_large = generator.generate_moves(conf_large, disulfide_bonds=bonds)
        
        disulfide_small = [m for m in moves_small if m.move_type == MoveType.DISULFIDE_CONSTRAINT][0]
        disulfide_large = [m for m in moves_large if m.move_type == MoveType.DISULFIDE_CONSTRAINT][0]
        
        # Larger violation should have more negative (stabilizing) energy change
        assert abs(disulfide_large.estimated_energy_change) > abs(disulfide_small.estimated_energy_change)

    def test_energy_barrier_scales_with_violation(self):
        """Test that energy barrier scales with violation magnitude."""
        coords_small = [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]
        coords_large = [(0.0, 0.0, 0.0), (15.0, 0.0, 0.0)]
        
        conf_small = create_test_conformation(sequence="CC", atom_coordinates=coords_small)
        conf_large = create_test_conformation(sequence="CC", atom_coordinates=coords_large)
        
        bonds = [DisulfideBond(residue_i=0, residue_j=1)]
        
        generator = MaplessMoveGenerator()
        
        moves_small = generator.generate_moves(conf_small, disulfide_bonds=bonds)
        moves_large = generator.generate_moves(conf_large, disulfide_bonds=bonds)
        
        disulfide_small = [m for m in moves_small if m.move_type == MoveType.DISULFIDE_CONSTRAINT][0]
        disulfide_large = [m for m in moves_large if m.move_type == MoveType.DISULFIDE_CONSTRAINT][0]
        
        # Larger violation should have higher energy barrier
        assert disulfide_large.energy_barrier > disulfide_small.energy_barrier

    def test_no_disulfide_bonds_provided(self):
        """Test that no disulfide moves are generated when bonds not provided."""
        conformation = create_test_conformation()
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=None)
        
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        assert len(disulfide_moves) == 0

    def test_empty_bonds_list(self):
        """Test that no disulfide moves are generated for empty bonds list."""
        conformation = create_test_conformation()
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=[])
        
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        assert len(disulfide_moves) == 0

    def test_out_of_bounds_residue_indices(self):
        """Test that bonds with out-of-bounds indices are skipped gracefully."""
        coords = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        conformation = create_test_conformation(sequence="CC", atom_coordinates=coords)
        
        # Bond with out-of-bounds index
        bonds = [DisulfideBond(residue_i=0, residue_j=10)]  # residue_j=10 out of bounds
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        # Should handle gracefully and generate no moves
        assert len(disulfide_moves) == 0

    def test_structural_feasibility_high(self):
        """Test that disulfide constraint moves have high structural feasibility."""
        coords = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        conformation = create_test_conformation(sequence="CC", atom_coordinates=coords)
        bonds = [DisulfideBond(residue_i=0, residue_j=1)]
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        assert len(disulfide_moves) >= 1
        move = disulfide_moves[0]
        assert move.structural_feasibility >= 0.8  # High feasibility

    def test_distance_calculation_3d(self):
        """Test that 3D Euclidean distance is calculated correctly."""
        # Create cysteines at known 3D distance
        # Distance = sqrt(3^2 + 4^2 + 0^2) = 5.0 Å
        coords = [
            (0.0, 0.0, 0.0),
            (3.0, 4.0, 0.0),  # 5.0 Å away
        ]
        conformation = create_test_conformation(sequence="CC", atom_coordinates=coords)
        bonds = [DisulfideBond(residue_i=0, residue_j=1, distance=3.8, tolerance=1.0)]
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        # Distance 5.0 > 3.8 + 1.0, so violation exists
        assert len(disulfide_moves) >= 1

    def test_move_id_uniqueness(self):
        """Test that move IDs are unique for different bonds."""
        coords = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
        ]
        conformation = create_test_conformation(sequence="CCCC", atom_coordinates=coords)
        
        bonds = [
            DisulfideBond(residue_i=0, residue_j=1),
            DisulfideBond(residue_i=2, residue_j=3),
        ]
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        move_ids = [m.move_id for m in disulfide_moves]
        # IDs should be unique (though random component means not 100% guaranteed)
        assert len(move_ids) == len(set(move_ids))

    def test_regular_moves_still_generated(self):
        """Test that regular moves are still generated alongside disulfide moves."""
        coords = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
        conformation = create_test_conformation(sequence="CCC", atom_coordinates=coords)
        bonds = [DisulfideBond(residue_i=0, residue_j=1)]
        
        generator = MaplessMoveGenerator()
        moves = generator.generate_moves(conformation, disulfide_bonds=bonds)
        
        # Should have both regular moves and disulfide moves
        regular_moves = [m for m in moves if m.move_type != MoveType.DISULFIDE_CONSTRAINT]
        disulfide_moves = [m for m in moves if m.move_type == MoveType.DISULFIDE_CONSTRAINT]
        
        assert len(regular_moves) > 0  # Regular moves present
        assert len(disulfide_moves) > 0  # Disulfide moves present


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
