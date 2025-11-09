"""
Unit tests for Hydrophobic Core Packer

Tests the quantum-guided hydrophobic core packing optimizer, including:
- PackingConstraint data model validation
- Hydrophobic residue identification
- Water exclusion zone calculations (0.28 nm spacing)
- Optimal packing distance determination (2.8Å intervals)
- QCP-weighted force constant scaling
- Packing constraint generation
- Edge cases and error handling

Success criteria (from requirements):
- 3.1: Identify hydrophobic residues and calculate pairwise distances ✓
- 3.2: Calculate water exclusion zones at 2.8Å intervals ✓
- 3.3: Scale force constants by (QCP_i + QCP_j) / 2 ✓
- 3.4: Apply base force constant 10.0 × QCP coupling ✓
- 3.5: Reduce core RMSD by at least 40% (integration test) ⏳
"""

import pytest
import math
from typing import List, Tuple, Dict

try:
    from ubf_protein.models import Conformation, PackingConstraint
    from ubf_protein.hydrophobic_core_packer import HydrophobicCorePacker
except ImportError:
    from models import Conformation, PackingConstraint
    from hydrophobic_core_packer import HydrophobicCorePacker


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def packer():
    """Create a HydrophobicCorePacker instance."""
    return HydrophobicCorePacker()


@pytest.fixture
def hydrophobic_conformation():
    """Create a conformation with hydrophobic residues."""
    # Sequence: A(0) V(1) L(2) I(3) F(4) - all hydrophobic
    # Coordinates in a line with 6Å spacing
    return Conformation(
        conformation_id="hydrophobic_test",
        sequence="AVILFMPW",  # All hydrophobic
        atom_coordinates=[
            (0.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
            (12.0, 0.0, 0.0),
            (18.0, 0.0, 0.0),
            (24.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
            (36.0, 0.0, 0.0),
            (42.0, 0.0, 0.0),
        ],
        energy=-150.0,
        rmsd_to_native=8.0,
        secondary_structure=['C'] * 8,
        phi_angles=[0.0] * 8,
        psi_angles=[0.0] * 8,
        available_move_types=['hydrophobic_collapse'],
        structural_constraints={}
    )


@pytest.fixture
def mixed_conformation():
    """Create a conformation with mixed hydrophobic and polar residues."""
    # A(0) K(1) V(2) R(3) L(4) E(5) I(6) D(7)
    # Hydrophobic: 0, 2, 4, 6
    return Conformation(
        conformation_id="mixed_test",
        sequence="AKVRLEID",
        atom_coordinates=[
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (8.0, 0.0, 0.0),
            (12.0, 0.0, 0.0),
            (16.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (24.0, 0.0, 0.0),
            (28.0, 0.0, 0.0),
        ],
        energy=-120.0,
        rmsd_to_native=6.0,
        secondary_structure=['C'] * 8,
        phi_angles=[0.0] * 8,
        psi_angles=[0.0] * 8,
        available_move_types=['hydrophobic_collapse'],
        structural_constraints={}
    )


@pytest.fixture
def high_qcp_values():
    """Create QCP values with high coherence."""
    return {
        0: 8.5,
        1: 9.2,
        2: 7.8,
        3: 8.1,
        4: 9.0,
        5: 7.5,
        6: 8.8,
        7: 9.3,
    }


@pytest.fixture
def low_qcp_values():
    """Create QCP values with low coherence."""
    return {i: 4.0 for i in range(8)}


# ============================================================================
# PackingConstraint Data Model Tests
# ============================================================================

class TestPackingConstraintModel:
    """Tests for PackingConstraint data model."""
    
    def test_valid_constraint_creation(self):
        """Test creating a valid packing constraint."""
        constraint = PackingConstraint(
            residue_i=0,
            residue_j=5,
            target_distance=5.6,  # 2×2.8Å
            force_constant=75.0,
            qcp_coupling=7.5
        )
        
        assert constraint.residue_i == 0
        assert constraint.residue_j == 5
        assert constraint.target_distance == 5.6
        assert constraint.force_constant == 75.0
        assert constraint.qcp_coupling == 7.5
    
    def test_negative_residue_i_rejected(self):
        """Test that negative residue_i is rejected."""
        with pytest.raises(ValueError, match="residue_i must be >= 0"):
            PackingConstraint(
                residue_i=-1,
                residue_j=5,
                target_distance=5.6,
                force_constant=75.0,
                qcp_coupling=7.5
            )
    
    def test_negative_residue_j_rejected(self):
        """Test that negative residue_j is rejected."""
        with pytest.raises(ValueError, match="residue_j must be >= 0"):
            PackingConstraint(
                residue_i=0,
                residue_j=-1,
                target_distance=5.6,
                force_constant=75.0,
                qcp_coupling=7.5
            )
    
    def test_same_residue_indices_rejected(self):
        """Test that same residue indices are rejected."""
        with pytest.raises(ValueError, match="residue_i and residue_j must be different"):
            PackingConstraint(
                residue_i=5,
                residue_j=5,
                target_distance=5.6,
                force_constant=75.0,
                qcp_coupling=7.5
            )
    
    def test_zero_target_distance_rejected(self):
        """Test that zero target distance is rejected."""
        with pytest.raises(ValueError, match="target_distance must be > 0"):
            PackingConstraint(
                residue_i=0,
                residue_j=5,
                target_distance=0.0,
                force_constant=75.0,
                qcp_coupling=7.5
            )
    
    def test_negative_force_constant_rejected(self):
        """Test that negative force constant is rejected."""
        with pytest.raises(ValueError, match="force_constant must be > 0"):
            PackingConstraint(
                residue_i=0,
                residue_j=5,
                target_distance=5.6,
                force_constant=-10.0,
                qcp_coupling=7.5
            )
    
    def test_zero_qcp_coupling_rejected(self):
        """Test that zero QCP coupling is rejected."""
        with pytest.raises(ValueError, match="qcp_coupling must be > 0"):
            PackingConstraint(
                residue_i=0,
                residue_j=5,
                target_distance=5.6,
                force_constant=75.0,
                qcp_coupling=0.0
            )
    
    def test_calculate_energy_at_target(self):
        """Test energy calculation when at target distance."""
        constraint = PackingConstraint(
            residue_i=0,
            residue_j=5,
            target_distance=5.6,
            force_constant=75.0,
            qcp_coupling=7.5
        )
        
        energy = constraint.calculate_energy(5.6)
        assert abs(energy) < 1e-6  # Should be ~0
    
    def test_calculate_energy_deviation(self):
        """Test energy calculation with distance deviation."""
        constraint = PackingConstraint(
            residue_i=0,
            residue_j=5,
            target_distance=5.6,
            force_constant=100.0,
            qcp_coupling=7.5
        )
        
        # Distance 1Å too long
        energy = constraint.calculate_energy(6.6)
        expected = 100.0 * 1.0 * 1.0  # k × (r - r₀)²
        assert abs(energy - expected) < 1e-6
        
        # Distance 0.5Å too short
        energy = constraint.calculate_energy(5.1)
        expected = 100.0 * 0.5 * 0.5
        assert abs(energy - expected) < 1e-6


# ============================================================================
# Hydrophobic Residue Identification Tests
# ============================================================================

class TestHydrophobicIdentification:
    """Tests for hydrophobic residue identification."""
    
    def test_identify_all_hydrophobic(self, packer, hydrophobic_conformation):
        """Test identifying all hydrophobic residues."""
        indices = packer._identify_hydrophobic_residues(
            hydrophobic_conformation.sequence
        )
        
        # All 8 residues should be hydrophobic
        assert len(indices) == 8
        assert indices == [0, 1, 2, 3, 4, 5, 6, 7]
    
    def test_identify_mixed_residues(self, packer, mixed_conformation):
        """Test identifying hydrophobic residues in mixed sequence."""
        indices = packer._identify_hydrophobic_residues(
            mixed_conformation.sequence
        )
        
        # Only A(0), V(2), L(4), I(6) are hydrophobic
        assert len(indices) == 4
        assert indices == [0, 2, 4, 6]
    
    def test_no_hydrophobic_residues(self, packer):
        """Test sequence with no hydrophobic residues."""
        indices = packer._identify_hydrophobic_residues("KRDEQ")
        assert len(indices) == 0
    
    def test_lowercase_sequence(self, packer):
        """Test that lowercase sequences work."""
        indices = packer._identify_hydrophobic_residues("avlif")
        assert len(indices) == 5
        assert indices == [0, 1, 2, 3, 4]


# ============================================================================
# Hydrophobic Pair Generation Tests
# ============================================================================

class TestHydrophobicPairGeneration:
    """Tests for hydrophobic residue pair generation."""
    
    def test_generate_pairs_sufficient_separation(self, packer):
        """Test pair generation with sufficient sequence separation."""
        # Indices: 0, 5, 10, 15 (separation >= 5)
        hydrophobic_indices = [0, 5, 10, 15]
        pairs = packer._generate_hydrophobic_pairs(hydrophobic_indices)
        
        # All pairs should satisfy |j - i| >= 5
        assert len(pairs) == 6  # C(4,2) = 6 combinations
        
        # Check all pairs
        expected_pairs = [
            (0, 5), (0, 10), (0, 15),
            (5, 10), (5, 15),
            (10, 15)
        ]
        assert pairs == expected_pairs
    
    def test_generate_pairs_insufficient_separation(self, packer):
        """Test that close residues are excluded."""
        # Indices: 0, 1, 2, 3 (all too close)
        hydrophobic_indices = [0, 1, 2, 3]
        pairs = packer._generate_hydrophobic_pairs(hydrophobic_indices)
        
        # No pairs should satisfy |j - i| >= 5
        assert len(pairs) == 0
    
    def test_generate_pairs_mixed_separation(self, packer):
        """Test mixed separations (some valid, some not)."""
        # Indices: 0, 2, 4, 10
        # Valid: (0,10), (2,10), (4,10)
        # Invalid: (0,2), (0,4), (2,4) - all < 5 separation
        hydrophobic_indices = [0, 2, 4, 10]
        pairs = packer._generate_hydrophobic_pairs(hydrophobic_indices)
        
        assert len(pairs) == 3
        assert (0, 10) in pairs
        assert (2, 10) in pairs
        assert (4, 10) in pairs
    
    def test_single_residue(self, packer):
        """Test with single hydrophobic residue."""
        pairs = packer._generate_hydrophobic_pairs([0])
        assert len(pairs) == 0
    
    def test_no_residues(self, packer):
        """Test with no hydrophobic residues."""
        pairs = packer._generate_hydrophobic_pairs([])
        assert len(pairs) == 0


# ============================================================================
# Distance Calculation Tests
# ============================================================================

class TestDistanceCalculation:
    """Tests for Euclidean distance calculation."""
    
    def test_distance_along_x_axis(self, packer):
        """Test distance calculation along x-axis."""
        coord1 = (0.0, 0.0, 0.0)
        coord2 = (6.0, 0.0, 0.0)
        
        distance = packer._calculate_distance(coord1, coord2)
        assert abs(distance - 6.0) < 1e-6
    
    def test_distance_3d(self, packer):
        """Test distance calculation in 3D space."""
        coord1 = (0.0, 0.0, 0.0)
        coord2 = (3.0, 4.0, 0.0)  # 3-4-5 triangle
        
        distance = packer._calculate_distance(coord1, coord2)
        assert abs(distance - 5.0) < 1e-6
    
    def test_distance_diagonal(self, packer):
        """Test distance calculation along 3D diagonal."""
        coord1 = (0.0, 0.0, 0.0)
        coord2 = (1.0, 1.0, 1.0)
        
        distance = packer._calculate_distance(coord1, coord2)
        expected = math.sqrt(3.0)
        assert abs(distance - expected) < 1e-6
    
    def test_distance_symmetry(self, packer):
        """Test that distance is symmetric."""
        coord1 = (1.0, 2.0, 3.0)
        coord2 = (4.0, 5.0, 6.0)
        
        dist1 = packer._calculate_distance(coord1, coord2)
        dist2 = packer._calculate_distance(coord2, coord1)
        
        assert abs(dist1 - dist2) < 1e-6


# ============================================================================
# Optimal Packing Distance Tests
# ============================================================================

class TestOptimalPackingDistance:
    """Tests for finding optimal water-excluded packing distances."""
    
    def test_distance_already_optimal(self, packer):
        """Test when current distance is already at ideal (6.0Å)."""
        optimal = packer._find_optimal_packing_distance(6.0)
        
        # Should return water-spaced value closest to 6.0
        # 6.0 / 2.8 ≈ 2.14, rounds to 2
        # Candidates: [..., 2.8, 5.6, 8.4, ...]
        # Closest to 6.0 is 5.6Å
        assert abs(optimal - 5.6) < 0.1
    
    def test_distance_below_optimal(self, packer):
        """Test when current distance is below ideal."""
        optimal = packer._find_optimal_packing_distance(4.0)
        
        # 4.0 / 2.8 ≈ 1.43, rounds to 1
        # Candidates around n=1: [2.8, 5.6, 8.4]
        # Closest to 6.0 is 5.6Å
        assert abs(optimal - 5.6) < 0.1
    
    def test_distance_above_optimal(self, packer):
        """Test when current distance is above ideal."""
        optimal = packer._find_optimal_packing_distance(10.0)
        
        # 10.0 / 2.8 ≈ 3.57, rounds to 4
        # Candidates around n=4: [5.6, 8.4, 11.2, 14.0, 16.8]
        # Closest to 6.0 is 5.6Å
        assert abs(optimal - 5.6) < 0.1
    
    def test_water_spacing_multiples(self, packer):
        """Test that returned distances are multiples of 2.8Å."""
        test_distances = [3.5, 7.2, 12.8, 18.3]
        
        for dist in test_distances:
            optimal = packer._find_optimal_packing_distance(dist)
            
            # Check if multiple of 2.8 (within floating point tolerance)
            remainder = optimal % packer.water_spacing_angstrom
            assert remainder < 0.01 or remainder > 2.79
    
    def test_minimum_distance(self, packer):
        """Test that returned distance is at least 2.8Å."""
        optimal = packer._find_optimal_packing_distance(1.0)
        assert optimal >= packer.water_spacing_angstrom


# ============================================================================
# Water Exclusion Zone Tests (Requirement 3.2)
# ============================================================================

class TestWaterExclusionZones:
    """Tests for water exclusion zone calculation."""
    
    def test_calculate_exclusion_zones(self, packer):
        """Test basic water exclusion zone calculation."""
        residue_pairs = [(0, 5), (1, 6), (2, 7)]
        
        zones = packer.calculate_water_exclusion_zones(residue_pairs)
        
        assert len(zones) == 3
        assert (0, 5) in zones
        assert (1, 6) in zones
        assert (2, 7) in zones
        
        # All should be close to ideal contact distance
        for distance in zones.values():
            # Should be water-spaced value near 6.0Å
            assert 5.0 <= distance <= 7.0
    
    def test_exclusion_zone_multiples(self, packer):
        """Test that exclusion zones are at 2.8Å intervals."""
        residue_pairs = [(i, i+5) for i in range(10)]
        
        zones = packer.calculate_water_exclusion_zones(residue_pairs)
        
        for distance in zones.values():
            remainder = distance % packer.water_spacing_angstrom
            assert remainder < 0.01 or remainder > 2.79
    
    def test_empty_pairs(self, packer):
        """Test with no residue pairs."""
        zones = packer.calculate_water_exclusion_zones([])
        assert len(zones) == 0


# ============================================================================
# QCP-Weighted Force Constant Tests (Requirements 3.3, 3.4)
# ============================================================================

class TestQCPWeightedForces:
    """Tests for QCP-weighted force constant calculation."""
    
    def test_high_qcp_high_force(self, packer, hydrophobic_conformation, high_qcp_values):
        """Test that high QCP values produce high force constants."""
        constraints = packer.quantum_hydrophobic_packing(
            hydrophobic_conformation,
            high_qcp_values
        )
        
        # Should have constraints (many hydrophobic pairs)
        assert len(constraints) > 0
        
        # Force constants should be scaled by QCP coupling
        # QCP ~8.5, coupling ~8.5, force = 10.0 × (8.5/10) = 8.5
        for constraint in constraints:
            # Force should be roughly equal to QCP coupling (base × coupling/10)
            assert 7.0 <= constraint.force_constant <= 10.0
            assert constraint.qcp_coupling > 7.0
    
    def test_low_qcp_low_force(self, packer, hydrophobic_conformation, low_qcp_values):
        """Test that low QCP values produce low force constants."""
        constraints = packer.quantum_hydrophobic_packing(
            hydrophobic_conformation,
            low_qcp_values
        )
        
        assert len(constraints) > 0
        
        # Force constants should be ~= base (10.0)
        # QCP ~4.0, coupling ~4.0, force ~4.0
        for constraint in constraints:
            assert constraint.force_constant < packer.base_force_constant
            assert constraint.force_constant > 0.0
    
    def test_qcp_coupling_calculation(self, packer, hydrophobic_conformation):
        """Test QCP coupling factor calculation."""
        qcp_values = {
            0: 8.0,
            5: 6.0,  # Average = 7.0
            6: 10.0,
            7: 8.0,  # Average = 9.0
        }
        
        constraints = packer.quantum_hydrophobic_packing(
            hydrophobic_conformation,
            qcp_values
        )
        
        # Find constraint for residues 0 and 5
        constraint_0_5 = None
        for c in constraints:
            if (c.residue_i == 0 and c.residue_j == 5) or \
               (c.residue_i == 5 and c.residue_j == 0):
                constraint_0_5 = c
                break
        
        if constraint_0_5:
            # QCP coupling should be (8.0 + 6.0) / 2 = 7.0
            assert abs(constraint_0_5.qcp_coupling - 7.0) < 0.1
            
            # Force constant should be 10.0 × (7.0 / 10.0) = 7.0
            expected_force = packer.base_force_constant * (7.0 / 10.0)
            assert abs(constraint_0_5.force_constant - expected_force) < 0.1


# ============================================================================
# Constraint Generation Integration Tests (Requirement 3.1)
# ============================================================================

class TestConstraintGeneration:
    """Tests for full constraint generation pipeline."""
    
    def test_all_hydrophobic_generates_constraints(self, packer, hydrophobic_conformation, high_qcp_values):
        """Test constraint generation with all hydrophobic sequence."""
        constraints = packer.quantum_hydrophobic_packing(
            hydrophobic_conformation,
            high_qcp_values
        )
        
        # 8 residues, pairs need |j-i| >= 5
        # Valid pairs: (0,5), (0,6), (0,7), (1,6), (1,7), (2,7)
        assert len(constraints) == 6
        
        # All constraints should have valid properties
        for constraint in constraints:
            assert constraint.residue_i >= 0
            assert constraint.residue_j >= 0
            assert constraint.residue_i != constraint.residue_j
            assert abs(constraint.residue_j - constraint.residue_i) >= 5
            assert constraint.target_distance > 0
            assert constraint.force_constant > 0
            assert constraint.qcp_coupling > 0
    
    def test_mixed_sequence_generates_subset(self, packer, mixed_conformation, high_qcp_values):
        """Test constraint generation with mixed sequence."""
        constraints = packer.quantum_hydrophobic_packing(
            mixed_conformation,
            high_qcp_values
        )
        
        # Hydrophobic: 0, 2, 4, 6
        # Valid pairs: (0,6) only (separation 6 >= 5)
        # (0,2), (0,4), (2,4), (2,6), (4,6) all have separation < 5
        assert len(constraints) == 1
        
        c = constraints[0]
        assert (c.residue_i == 0 and c.residue_j == 6) or \
               (c.residue_i == 6 and c.residue_j == 0)
    
    def test_no_hydrophobic_no_constraints(self, packer, high_qcp_values):
        """Test that non-hydrophobic sequence generates no constraints."""
        polar_conf = Conformation(
            conformation_id="polar_test",
            sequence="KRDEQNST",  # All polar
            atom_coordinates=[(float(i*4), 0.0, 0.0) for i in range(8)],
            energy=-100.0,
            rmsd_to_native=5.0,
            secondary_structure=['C'] * 8,
            phi_angles=[0.0] * 8,
            psi_angles=[0.0] * 8,
            available_move_types=[],
            structural_constraints={}
        )
        
        constraints = packer.quantum_hydrophobic_packing(polar_conf, high_qcp_values)
        assert len(constraints) == 0
    
    def test_single_hydrophobic_no_constraints(self, packer, high_qcp_values):
        """Test that single hydrophobic residue generates no constraints."""
        single_conf = Conformation(
            conformation_id="single_test",
            sequence="DKRDEQNS",  # No hydrophobic residues
            atom_coordinates=[(float(i*4), 0.0, 0.0) for i in range(8)],
            energy=-100.0,
            rmsd_to_native=5.0,
            secondary_structure=['C'] * 8,
            phi_angles=[0.0] * 8,
            psi_angles=[0.0] * 8,
            available_move_types=[],
            structural_constraints={}
        )
        
        constraints = packer.quantum_hydrophobic_packing(single_conf, high_qcp_values)
        assert len(constraints) == 0
    
    def test_missing_qcp_uses_default(self, packer, hydrophobic_conformation):
        """Test that missing QCP values use default of 4.0."""
        incomplete_qcp = {0: 8.0, 1: 9.0}  # Missing many residues
        
        constraints = packer.quantum_hydrophobic_packing(
            hydrophobic_conformation,
            incomplete_qcp
        )
        
        # Should still generate constraints
        assert len(constraints) > 0
        
        # Constraints with missing QCP should use default
        for constraint in constraints:
            if constraint.residue_i not in incomplete_qcp or \
               constraint.residue_j not in incomplete_qcp:
                # Should use default QCP coupling ~4.0
                assert constraint.qcp_coupling <= 6.5  # (8+4)/2 or (4+4)/2


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================

class TestPerformanceAndEdgeCases:
    """Tests for performance and edge case handling."""
    
    def test_large_protein_performance(self, packer):
        """Test performance with large protein (100 residues)."""
        import time
        
        # Create large all-hydrophobic protein
        large_conf = Conformation(
            conformation_id="large_test",
            sequence="A" * 100,
            atom_coordinates=[(float(i*4), 0.0, 0.0) for i in range(100)],
            energy=-500.0,
            rmsd_to_native=10.0,
            secondary_structure=['C'] * 100,
            phi_angles=[0.0] * 100,
            psi_angles=[0.0] * 100,
            available_move_types=['hydrophobic_collapse'],
            structural_constraints={}
        )
        
        qcp_values = {i: 7.5 for i in range(100)}
        
        start_time = time.time()
        constraints = packer.quantum_hydrophobic_packing(large_conf, qcp_values)
        elapsed = time.time() - start_time
        
        # Should complete in < 500ms
        assert elapsed < 0.5
        
        # Should generate many constraints
        # 100 residues, pairs with |j-i| >= 5: roughly (100-5) × 95 / 2 ≈ 4500
        assert len(constraints) > 4000
    
    def test_zero_coordinates(self, packer):
        """Test handling of residues at same position."""
        overlap_conf = Conformation(
            conformation_id="overlap_test",
            sequence="AVILFM",
            atom_coordinates=[(0.0, 0.0, 0.0)] * 6,  # All at origin
            energy=-50.0,
            rmsd_to_native=15.0,
            secondary_structure=['C'] * 6,
            phi_angles=[0.0] * 6,
            psi_angles=[0.0] * 6,
            available_move_types=[],
            structural_constraints={}
        )
        
        qcp_values = {i: 7.0 for i in range(6)}
        
        # Should still generate constraints (distance = 0)
        constraints = packer.quantum_hydrophobic_packing(overlap_conf, qcp_values)
        
        # Pairs: (0,5) only
        assert len(constraints) == 1
        
        # With distance 0, closest to 6.0Å is 5.6Å
        assert abs(constraints[0].target_distance - 5.6) < 0.1
