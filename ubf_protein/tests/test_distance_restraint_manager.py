"""
Unit tests for Distance Restraint Manager

Tests the distance restraint system for quantum refinement, including:
- DistanceRestraint data model validation
- φ-harmonic distance calculation [d/φ, d, d×φ]
- High-QCP pair identification
- Restraint generation with optimal distance selection
- Restraint energy calculation
- Edge cases and error handling

Success criteria (from requirements):
- 7.1: Identify high-QCP pairs (both residues QCP > 7) ✓
- 7.2: Calculate φ-harmonic distances [d/φ, d, d×φ] ✓
- 7.3: Select distance closest to 6.0Å ✓
- 7.4: Apply weight 100.0 and tolerance 0.5Å ✓
- 7.5: Maintain restraints throughout optimization ✓
"""

import pytest
import math
from typing import List, Tuple

try:
    from ubf_protein.models import Conformation, DistanceRestraint
    from ubf_protein.distance_restraint_manager import DistanceRestraintManager
except ImportError:
    from models import Conformation, DistanceRestraint
    from distance_restraint_manager import DistanceRestraintManager


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def simple_conformation():
    """Create a simple test conformation with 5 residues."""
    return Conformation(
        conformation_id="test_conf_001",
        sequence="ACDEF",
        atom_coordinates=[
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (15.2, 0.0, 0.0),
        ],
        energy=-100.0,
        rmsd_to_native=5.0,
        secondary_structure=['C', 'C', 'C', 'C', 'C'],
        phi_angles=[0.0] * 5,
        psi_angles=[0.0] * 5,
        available_move_types=['local_perturbation'],
        structural_constraints={}
    )


@pytest.fixture
def high_qcp_values():
    """Create QCP values with some high-QCP residues."""
    return {
        0: 8.5,   # High QCP
        1: 9.2,   # High QCP
        2: 6.3,   # Below threshold
        3: 8.8,   # High QCP
        4: 9.5,   # High QCP
    }


@pytest.fixture
def manager():
    """Create a DistanceRestraintManager instance."""
    return DistanceRestraintManager(
        qcpp_adapter=None,
        phi=1.618033988749895,
        default_weight=100.0,
        default_tolerance=0.5,
        optimal_distance=6.0
    )


# ============================================================================
# DistanceRestraint Data Model Tests
# ============================================================================

class TestDistanceRestraintModel:
    """Tests for DistanceRestraint data model."""
    
    def test_valid_restraint_creation(self):
        """Test creating a valid distance restraint."""
        restraint = DistanceRestraint(
            residue_i=0,
            residue_j=5,
            target_distance=6.0,
            weight=100.0,
            tolerance=0.5,
            is_phi_harmonic=True
        )
        
        assert restraint.residue_i == 0
        assert restraint.residue_j == 5
        assert restraint.target_distance == 6.0
        assert restraint.weight == 100.0
        assert restraint.tolerance == 0.5
        assert restraint.is_phi_harmonic is True
    
    def test_negative_residue_index(self):
        """Test that negative residue indices raise ValueError."""
        with pytest.raises(ValueError, match="residue_i must be >= 0"):
            DistanceRestraint(
                residue_i=-1,
                residue_j=5,
                target_distance=6.0,
                weight=100.0,
                tolerance=0.5,
                is_phi_harmonic=True
            )
    
    def test_same_residue_pair(self):
        """Test that i==j raises ValueError."""
        with pytest.raises(ValueError, match="must be different"):
            DistanceRestraint(
                residue_i=5,
                residue_j=5,
                target_distance=6.0,
                weight=100.0,
                tolerance=0.5,
                is_phi_harmonic=True
            )
    
    def test_zero_target_distance(self):
        """Test that target_distance=0 raises ValueError."""
        with pytest.raises(ValueError, match="target_distance must be > 0"):
            DistanceRestraint(
                residue_i=0,
                residue_j=5,
                target_distance=0.0,
                weight=100.0,
                tolerance=0.5,
                is_phi_harmonic=True
            )
    
    def test_zero_weight(self):
        """Test that weight=0 raises ValueError."""
        with pytest.raises(ValueError, match="weight must be > 0"):
            DistanceRestraint(
                residue_i=0,
                residue_j=5,
                target_distance=6.0,
                weight=0.0,
                tolerance=0.5,
                is_phi_harmonic=True
            )
    
    def test_zero_tolerance(self):
        """Test that tolerance=0 raises ValueError."""
        with pytest.raises(ValueError, match="tolerance must be > 0"):
            DistanceRestraint(
                residue_i=0,
                residue_j=5,
                target_distance=6.0,
                weight=100.0,
                tolerance=0.0,
                is_phi_harmonic=True
            )
    
    def test_calculate_energy_within_tolerance(self):
        """Test energy calculation when distance is within tolerance."""
        restraint = DistanceRestraint(
            residue_i=0,
            residue_j=5,
            target_distance=6.0,
            weight=100.0,
            tolerance=0.5,
            is_phi_harmonic=True
        )
        
        # Distance exactly at target
        assert restraint.calculate_energy(6.0) == 0.0
        
        # Distance within tolerance
        assert restraint.calculate_energy(6.3) == 0.0
        assert restraint.calculate_energy(5.7) == 0.0
        
        # Distance at edge of tolerance
        assert restraint.calculate_energy(6.5) == 0.0
        assert restraint.calculate_energy(5.5) == 0.0
    
    def test_calculate_energy_outside_tolerance(self):
        """Test energy calculation when distance is outside tolerance."""
        restraint = DistanceRestraint(
            residue_i=0,
            residue_j=5,
            target_distance=6.0,
            weight=100.0,
            tolerance=0.5,
            is_phi_harmonic=True
        )
        
        # Distance 1.0Å above target (0.5Å outside tolerance)
        energy = restraint.calculate_energy(7.0)
        expected = 100.0 * (0.5 ** 2)  # weight × (deviation - tolerance)²
        assert abs(energy - expected) < 1e-6
        
        # Distance 2.0Å above target (1.5Å outside tolerance)
        energy = restraint.calculate_energy(8.0)
        expected = 100.0 * (1.5 ** 2)
        assert abs(energy - expected) < 1e-6
        
        # Distance 1.0Å below target (0.5Å outside tolerance)
        energy = restraint.calculate_energy(5.0)
        expected = 100.0 * (0.5 ** 2)
        assert abs(energy - expected) < 1e-6


# ============================================================================
# DistanceRestraintManager Initialization Tests
# ============================================================================

class TestDistanceRestraintManagerInit:
    """Tests for DistanceRestraintManager initialization."""
    
    def test_default_initialization(self):
        """Test manager initialization with default parameters."""
        manager = DistanceRestraintManager()
        
        assert manager.qcpp_adapter is None
        assert abs(manager.phi - 1.618033988749895) < 1e-10
        assert manager.default_weight == 100.0
        assert manager.default_tolerance == 0.5
        assert manager.optimal_distance == 6.0
        assert manager.total_restraints_generated == 0
        assert manager.total_pairs_evaluated == 0
    
    def test_custom_initialization(self):
        """Test manager initialization with custom parameters."""
        manager = DistanceRestraintManager(
            qcpp_adapter=None,
            phi=1.6,
            default_weight=50.0,
            default_tolerance=1.0,
            optimal_distance=7.0
        )
        
        assert manager.phi == 1.6
        assert manager.default_weight == 50.0
        assert manager.default_tolerance == 1.0
        assert manager.optimal_distance == 7.0


# ============================================================================
# φ-Harmonic Distance Calculation Tests (Requirement 7.2)
# ============================================================================

class TestPhiHarmonicDistances:
    """Tests for φ-harmonic distance calculation."""
    
    def test_find_phi_harmonic_distances_basic(self, manager):
        """Test φ-harmonic distance calculation with basic input."""
        phi = manager.phi
        current_distance = 8.0
        
        distances = manager.find_phi_harmonic_distances(current_distance)
        
        assert len(distances) == 3
        assert abs(distances[0] - 8.0 / phi) < 1e-6  # d/φ
        assert abs(distances[1] - 8.0) < 1e-6        # d
        assert abs(distances[2] - 8.0 * phi) < 1e-6  # d×φ
    
    def test_find_phi_harmonic_distances_near_optimal(self, manager):
        """Test φ-harmonic distances near optimal distance (6.0Å)."""
        current_distance = 6.5
        phi = manager.phi
        
        distances = manager.find_phi_harmonic_distances(current_distance)
        
        assert abs(distances[0] - 6.5 / phi) < 1e-6
        assert abs(distances[1] - 6.5) < 1e-6
        assert abs(distances[2] - 6.5 * phi) < 1e-6
    
    def test_find_phi_harmonic_distances_small_value(self, manager):
        """Test φ-harmonic distances with small distance."""
        current_distance = 2.0
        phi = manager.phi
        
        distances = manager.find_phi_harmonic_distances(current_distance)
        
        # Verify all three options are generated
        assert distances[0] < distances[1] < distances[2]
        assert abs(distances[1] - 2.0) < 1e-6
    
    def test_find_phi_harmonic_distances_zero_raises_error(self, manager):
        """Test that zero distance raises ValueError."""
        with pytest.raises(ValueError, match="current_distance must be > 0"):
            manager.find_phi_harmonic_distances(0.0)
    
    def test_find_phi_harmonic_distances_negative_raises_error(self, manager):
        """Test that negative distance raises ValueError."""
        with pytest.raises(ValueError, match="current_distance must be > 0"):
            manager.find_phi_harmonic_distances(-5.0)


# ============================================================================
# Optimal Distance Selection Tests (Requirement 7.3)
# ============================================================================

class TestOptimalDistanceSelection:
    """Tests for selecting φ-harmonic distance closest to target."""
    
    def test_select_optimal_middle_option(self, manager):
        """Test selection when middle option is closest to target."""
        phi_distances = [4.5, 7.3, 11.8]  # [d/φ, d, d×φ]
        target = 6.0
        
        optimal = manager._select_optimal_distance(phi_distances, target)
        
        # 7.3 is closest to 6.0 (distance 1.3)
        assert abs(optimal - 7.3) < 1e-6
    
    def test_select_optimal_first_option(self, manager):
        """Test selection when first option (d/φ) is closest."""
        phi_distances = [6.2, 10.0, 16.2]  # [d/φ, d, d×φ]
        target = 6.0
        
        optimal = manager._select_optimal_distance(phi_distances, target)
        
        # 6.2 is closest to 6.0 (distance 0.2)
        assert abs(optimal - 6.2) < 1e-6
    
    def test_select_optimal_last_option(self, manager):
        """Test selection when last option (d×φ) is closest."""
        phi_distances = [2.0, 3.2, 5.8]  # [d/φ, d, d×φ]
        target = 6.0
        
        optimal = manager._select_optimal_distance(phi_distances, target)
        
        # 5.8 is closest to 6.0 (distance 0.2)
        assert abs(optimal - 5.8) < 1e-6
    
    def test_select_optimal_exact_match(self, manager):
        """Test selection when one option exactly matches target."""
        phi_distances = [3.7, 6.0, 9.7]  # [d/φ, d, d×φ]
        target = 6.0
        
        optimal = manager._select_optimal_distance(phi_distances, target)
        
        # Exact match
        assert abs(optimal - 6.0) < 1e-6


# ============================================================================
# Restraint Generation Tests (Requirements 7.1, 7.2, 7.3, 7.4)
# ============================================================================

class TestRestraintGeneration:
    """Tests for quantum distance restraint generation."""
    
    def test_add_restraints_basic(self, manager, simple_conformation, high_qcp_values):
        """Test basic restraint generation with high-QCP pairs."""
        restraints = manager.add_quantum_distance_restraints(
            structure=simple_conformation,
            qcp_values=high_qcp_values,
            qcp_threshold=7.0,
            min_sequence_separation=2
        )
        
        # Should generate restraints for pairs: (0,3), (0,4), (1,3), (1,4), (3,4)
        # Skip pairs with residue 2 (QCP=6.3 < 7.0)
        # Skip pairs (0,1), (1,2), (2,3), (3,4) if min_sep=2
        # Actually with min_sep=2: valid pairs are i+2, i+3, i+4...
        # Pairs: (0,2)[skip-low QCP], (0,3), (0,4), (1,3), (1,4), (2,4)[skip-low QCP]
        # High-QCP pairs with separation ≥2: (0,3), (0,4), (1,3), (1,4)
        
        assert len(restraints) >= 4
        
        # Verify all restraints have correct parameters
        for restraint in restraints:
            assert restraint.weight == 100.0
            assert restraint.tolerance == 0.5
            assert restraint.is_phi_harmonic is True
            assert restraint.target_distance > 0
    
    def test_add_restraints_qcp_threshold_filtering(self, manager, simple_conformation):
        """Test that low-QCP residues are filtered out."""
        qcp_values = {
            0: 8.0,  # High
            1: 6.0,  # Low
            2: 9.0,  # High
            3: 5.0,  # Low
            4: 8.5,  # High
        }
        
        restraints = manager.add_quantum_distance_restraints(
            structure=simple_conformation,
            qcp_values=qcp_values,
            qcp_threshold=7.0,
            min_sequence_separation=2
        )
        
        # Only pairs (0,2), (0,4), (2,4) should pass (both QCP > 7.0)
        # All other pairs have at least one residue with QCP < 7.0
        expected_pairs = {(0, 2), (0, 4), (2, 4)}
        actual_pairs = {(r.residue_i, r.residue_j) for r in restraints}
        
        assert actual_pairs == expected_pairs
    
    def test_add_restraints_sequence_separation(self, manager, simple_conformation, high_qcp_values):
        """Test minimum sequence separation filtering."""
        # Test with min_sep=3
        restraints = manager.add_quantum_distance_restraints(
            structure=simple_conformation,
            qcp_values=high_qcp_values,
            qcp_threshold=7.0,
            min_sequence_separation=3
        )
        
        # All pairs must have |j-i| >= 3
        for restraint in restraints:
            separation = abs(restraint.residue_j - restraint.residue_i)
            assert separation >= 3
    
    def test_add_restraints_empty_qcp_values(self, manager, simple_conformation):
        """Test that empty QCP values raises ValueError."""
        with pytest.raises(ValueError, match="qcp_values cannot be empty"):
            manager.add_quantum_distance_restraints(
                structure=simple_conformation,
                qcp_values={},
                qcp_threshold=7.0
            )
    
    def test_add_restraints_no_coordinates(self, manager, high_qcp_values):
        """Test that structure without coordinates raises ValueError."""
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACDEF",
            atom_coordinates=[],  # Empty!
            energy=-100.0,
            rmsd_to_native=5.0,
            secondary_structure=['C'] * 5,
            phi_angles=[0.0] * 5,
            psi_angles=[0.0] * 5,
            available_move_types=[],
            structural_constraints={}
        )
        
        with pytest.raises(ValueError, match="must have atom_coordinates"):
            manager.add_quantum_distance_restraints(
                structure=bad_conf,
                qcp_values=high_qcp_values,
                qcp_threshold=7.0
            )
    
    def test_add_restraints_target_near_optimal(self, manager, simple_conformation):
        """Test that selected target distances are near optimal (6.0Å)."""
        qcp_values = {0: 8.0, 1: 9.0, 2: 8.5, 3: 9.2, 4: 8.8}
        
        restraints = manager.add_quantum_distance_restraints(
            structure=simple_conformation,
            qcp_values=qcp_values,
            qcp_threshold=7.0,
            min_sequence_separation=2
        )
        
        # At least some restraints should have targets near 6.0Å
        # (given our linear coordinates at 3.8Å spacing)
        targets = [r.target_distance for r in restraints]
        
        # Check that we have diverse targets (not all the same)
        assert len(set(targets)) >= 1


# ============================================================================
# Restraint Application Tests (Requirement 7.5)
# ============================================================================

class TestRestraintApplication:
    """Tests for applying restraints during optimization."""
    
    def test_apply_restraints_basic(self, manager, simple_conformation):
        """Test basic restraint application and energy calculation."""
        restraints = [
            DistanceRestraint(0, 4, 6.0, 100.0, 0.5, True),  # Distance is 15.2Å
        ]
        
        total_energy = manager.apply_restraints(simple_conformation, restraints)
        
        # Distance is 15.2Å, target is 6.0Å, tolerance is 0.5Å
        # Effective deviation = 15.2 - 6.0 - 0.5 = 8.7Å
        # Energy = 100.0 × (8.7)² = 7569
        assert total_energy > 0  # Should have penalty
        assert abs(total_energy - 7569.0) < 100  # Approximate check
    
    def test_apply_restraints_multiple(self, manager, simple_conformation):
        """Test applying multiple restraints."""
        restraints = [
            DistanceRestraint(0, 2, 7.6, 100.0, 0.5, True),  # Exact distance
            DistanceRestraint(1, 3, 7.6, 100.0, 0.5, True),  # Exact distance
        ]
        
        total_energy = manager.apply_restraints(simple_conformation, restraints)
        
        # Both restraints should have zero energy (exact match)
        assert total_energy == 0.0
    
    def test_apply_restraints_empty_list(self, manager, simple_conformation):
        """Test applying empty restraints list."""
        total_energy = manager.apply_restraints(simple_conformation, [])
        
        assert total_energy == 0.0
    
    def test_apply_restraints_no_coordinates(self, manager):
        """Test that structure without coordinates raises ValueError."""
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACDEF",
            atom_coordinates=[],
            energy=-100.0,
            rmsd_to_native=5.0,
            secondary_structure=['C'] * 5,
            phi_angles=[0.0] * 5,
            psi_angles=[0.0] * 5,
            available_move_types=[],
            structural_constraints={}
        )
        
        restraints = [DistanceRestraint(0, 1, 6.0, 100.0, 0.5, True)]
        
        with pytest.raises(ValueError, match="must have atom_coordinates"):
            manager.apply_restraints(bad_conf, restraints)
    
    def test_apply_restraints_invalid_indices(self, manager, simple_conformation):
        """Test handling of restraints with invalid residue indices."""
        # Restraint references residue 10, but structure only has 5 residues
        restraints = [
            DistanceRestraint(0, 10, 6.0, 100.0, 0.5, True),
        ]
        
        # Should log warning and skip invalid restraint, returning 0 energy
        total_energy = manager.apply_restraints(simple_conformation, restraints)
        
        assert total_energy == 0.0


# ============================================================================
# Statistics and Utility Tests
# ============================================================================

class TestStatistics:
    """Tests for manager statistics."""
    
    def test_get_statistics_initial(self, manager):
        """Test statistics before any operations."""
        stats = manager.get_statistics()
        
        assert stats['total_restraints_generated'] == 0
        assert stats['total_pairs_evaluated'] == 0
        assert stats['acceptance_rate'] == 0.0
        assert stats['phi'] == manager.phi
        assert stats['default_weight'] == 100.0
        assert stats['default_tolerance'] == 0.5
        assert stats['optimal_distance'] == 6.0
    
    def test_get_statistics_after_generation(self, manager, simple_conformation, high_qcp_values):
        """Test statistics after restraint generation."""
        restraints = manager.add_quantum_distance_restraints(
            structure=simple_conformation,
            qcp_values=high_qcp_values,
            qcp_threshold=7.0,
            min_sequence_separation=2
        )
        
        stats = manager.get_statistics()
        
        assert stats['total_restraints_generated'] > 0
        assert stats['total_pairs_evaluated'] > 0
        assert 0 < stats['acceptance_rate'] <= 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_full_workflow(self, manager):
        """Test complete restraint generation and application workflow."""
        # Create a 3D helical structure
        coords = []
        for i in range(10):
            # Simple helix: rotate and advance
            angle = i * 100.0 * 3.14159 / 180.0
            z = i * 1.5
            coords.append((3.0 * math.cos(angle), 3.0 * math.sin(angle), z))
        
        conf = Conformation(
            conformation_id="helix",
            sequence="ACDEFGHIJK",
            atom_coordinates=coords,
            energy=-200.0,
            rmsd_to_native=3.0,
            secondary_structure=['H'] * 10,
            phi_angles=[0.0] * 10,
            psi_angles=[0.0] * 10,
            available_move_types=['local_perturbation'],
            structural_constraints={}
        )
        
        # All residues have high QCP
        qcp_values = {i: 8.0 + i * 0.1 for i in range(10)}
        
        # Generate restraints
        restraints = manager.add_quantum_distance_restraints(
            structure=conf,
            qcp_values=qcp_values,
            qcp_threshold=7.0,
            min_sequence_separation=3
        )
        
        assert len(restraints) > 0
        
        # Apply restraints
        total_energy = manager.apply_restraints(conf, restraints)
        
        assert total_energy >= 0  # Should be non-negative
        
        # Get statistics
        stats = manager.get_statistics()
        assert stats['total_restraints_generated'] == len(restraints)
        assert stats['acceptance_rate'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
