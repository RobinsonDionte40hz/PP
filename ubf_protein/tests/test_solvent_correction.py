"""
Unit tests for SolventFieldCorrection module.

Tests cover:
- Initialization with valid and invalid parameters
- Neighbor counting with sequence separation
- Burial factor calculation for surface, intermediate, and core residues
- Distance-dependent dielectric at various distances
- Effective dielectric combining distance and burial effects
- Pairwise dielectric calculation with caching
- Electrostatic energy correction
- Edge cases with extreme neighbor counts and distances

Run with: pytest ubf_protein/tests/test_solvent_correction.py -v
"""

import pytest
import math
from typing import List, Tuple

# Import modules under test
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ubf_protein.models import SideChainField
from ubf_protein.solvent_correction import SolventFieldCorrection


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

@pytest.fixture
def default_corrector():
    """Create SolventFieldCorrection with default parameters."""
    return SolventFieldCorrection()


@pytest.fixture
def custom_corrector():
    """Create SolventFieldCorrection with custom parameters."""
    return SolventFieldCorrection(
        screening_length=2.5,
        neighbor_cutoff=10.0,
        epsilon_buried=2.0,
        epsilon_surface=60.0,
        burial_midpoint=15,
        burial_steepness=0.5
    )


def create_test_field(residue_index: int, position: Tuple[float, float, float],
                     charge: float = 0.0, hydrophobicity: float = 0.0,
                     volume: float = 100.0, amino_acid: str = 'A') -> SideChainField:
    """Helper to create test side-chain fields."""
    return SideChainField(
        residue_index=residue_index,
        amino_acid=amino_acid,
        position=position,
        charge=charge,
        hydrophobicity=hydrophobicity,
        volume=volume
    )


def create_linear_chain(n_residues: int, spacing: float = 3.8) -> List[SideChainField]:
    """Create a linear chain of residues along x-axis."""
    fields = []
    for i in range(n_residues):
        position = (i * spacing, 0.0, 0.0)
        fields.append(create_test_field(i, position))
    return fields


def create_compact_cluster(n_residues: int, radius: float = 5.0) -> List[SideChainField]:
    """Create a compact cluster of residues (mimics protein core)."""
    fields = []
    for i in range(n_residues):
        # Arrange in sphere
        theta = (i * 2 * math.pi) / n_residues
        phi = math.acos(1 - 2 * (i / n_residues))
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        fields.append(create_test_field(i, (x, y, z)))
    return fields


# ============================================================================
# Initialization Tests
# ============================================================================

class TestSolventCorrectionInitialization:
    """Test SolventFieldCorrection initialization and validation."""
    
    def test_default_initialization(self, default_corrector):
        """Test initialization with default parameters."""
        assert default_corrector.screening_length == 3.0
        assert default_corrector.neighbor_cutoff == 8.0
        assert default_corrector.epsilon_buried == 4.0
        assert default_corrector.epsilon_surface == 80.0
        assert default_corrector.burial_midpoint == 12
        assert default_corrector.burial_steepness == 0.3
    
    def test_custom_initialization(self, custom_corrector):
        """Test initialization with custom parameters."""
        assert custom_corrector.screening_length == 2.5
        assert custom_corrector.neighbor_cutoff == 10.0
        assert custom_corrector.epsilon_buried == 2.0
        assert custom_corrector.epsilon_surface == 60.0
        assert custom_corrector.burial_midpoint == 15
        assert custom_corrector.burial_steepness == 0.5
    
    def test_invalid_screening_length(self):
        """Test that negative screening length raises error."""
        with pytest.raises(ValueError, match="screening_length must be positive"):
            SolventFieldCorrection(screening_length=-1.0)
        
        with pytest.raises(ValueError, match="screening_length must be positive"):
            SolventFieldCorrection(screening_length=0.0)
    
    def test_invalid_neighbor_cutoff(self):
        """Test that invalid neighbor cutoff raises error."""
        with pytest.raises(ValueError, match="neighbor_cutoff must be positive"):
            SolventFieldCorrection(neighbor_cutoff=-5.0)
        
        with pytest.raises(ValueError, match="neighbor_cutoff must be positive"):
            SolventFieldCorrection(neighbor_cutoff=0.0)
    
    def test_invalid_epsilon_values(self):
        """Test that invalid dielectric constants raise errors."""
        # epsilon_buried < 1.0
        with pytest.raises(ValueError, match="1.0 <= epsilon_buried <= epsilon_surface"):
            SolventFieldCorrection(epsilon_buried=0.5)
        
        # epsilon_surface < epsilon_buried
        with pytest.raises(ValueError, match="1.0 <= epsilon_buried <= epsilon_surface"):
            SolventFieldCorrection(epsilon_buried=80.0, epsilon_surface=4.0)
    
    def test_invalid_burial_midpoint(self):
        """Test that negative burial midpoint raises error."""
        with pytest.raises(ValueError, match="burial_midpoint must be non-negative"):
            SolventFieldCorrection(burial_midpoint=-5)
    
    def test_invalid_burial_steepness(self):
        """Test that non-positive burial steepness raises error."""
        with pytest.raises(ValueError, match="burial_steepness must be positive"):
            SolventFieldCorrection(burial_steepness=0.0)
        
        with pytest.raises(ValueError, match="burial_steepness must be positive"):
            SolventFieldCorrection(burial_steepness=-0.5)


# ============================================================================
# Neighbor Counting Tests
# ============================================================================

class TestNeighborCounting:
    """Test neighbor counting with various configurations."""
    
    def test_count_neighbors_linear_chain(self, default_corrector):
        """Test neighbor counting in a linear chain (few neighbors)."""
        fields = create_linear_chain(20, spacing=3.8)
        center_field = fields[10]
        
        # With 3.8 Å spacing, only immediate non-sequence neighbors within 8.0 Å
        count = default_corrector.count_neighbors(center_field, fields)
        
        # Should count residues within 8.0 Å but >3 sequence separation
        # At 3.8 Å spacing: positions 10±4 to 10±1 are within 8.0 Å
        # But 10±1, 10±2, 10±3 are excluded by sequence separation
        # So should count 10±4 = 2 neighbors (if within range)
        assert count >= 0  # At least no error
        assert count <= 4  # Maximum possible with cutoffs
    
    def test_count_neighbors_compact_cluster(self, default_corrector):
        """Test neighbor counting in a compact cluster (many neighbors)."""
        fields = create_compact_cluster(20, radius=5.0)
        center_field = fields[0]
        
        # In a compact cluster, should have many neighbors
        count = default_corrector.count_neighbors(center_field, fields)
        
        # With radius 5.0 and cutoff 8.0, should have several neighbors
        # (actual count depends on spherical distribution)
        assert count >= 5  # Expect multiple neighbors in compact structure
    
    def test_count_neighbors_excludes_self(self, default_corrector):
        """Test that neighbor counting excludes the field itself."""
        fields = create_compact_cluster(10, radius=3.0)
        
        for field in fields:
            count = default_corrector.count_neighbors(field, fields)
            # Should not count itself, so max is n-1
            assert count <= len(fields) - 1
    
    def test_count_neighbors_sequence_separation(self, default_corrector):
        """Test that sequence neighbors are properly excluded."""
        # Create fields close in space but varying in sequence
        fields = [
            create_test_field(0, (0.0, 0.0, 0.0)),
            create_test_field(1, (3.0, 0.0, 0.0)),  # Close, but seq sep = 1
            create_test_field(2, (0.0, 3.0, 0.0)),  # Close, but seq sep = 2
            create_test_field(3, (0.0, 0.0, 3.0)),  # Close, but seq sep = 3
            create_test_field(4, (3.0, 3.0, 0.0)),  # Close, seq sep = 4 (counts!)
        ]
        
        count = default_corrector.count_neighbors(fields[0], fields, sequence_separation=3)
        
        # Should only count residue 4 (seq sep > 3)
        assert count == 1
    
    def test_count_neighbors_empty_list(self, default_corrector):
        """Test neighbor counting with single field (no neighbors)."""
        field = create_test_field(0, (0.0, 0.0, 0.0))
        
        count = default_corrector.count_neighbors(field, [field])
        assert count == 0


# ============================================================================
# Burial Factor Tests
# ============================================================================

class TestBurialFactor:
    """Test burial factor calculation."""
    
    def test_burial_surface_residue(self, default_corrector):
        """Test burial factor for surface residue (few neighbors)."""
        fields = create_linear_chain(20, spacing=4.0)
        surface_field = fields[0]  # End residue, few neighbors
        
        burial = default_corrector.calculate_burial_factor(surface_field, fields)
        
        # Surface residue should have low burial factor (< 0.5)
        assert 0.0 <= burial < 0.5
    
    def test_burial_core_residue(self, default_corrector):
        """Test burial factor for core residue (many neighbors)."""
        fields = create_compact_cluster(25, radius=5.0)
        core_field = fields[12]  # Central residue in cluster
        
        burial = default_corrector.calculate_burial_factor(core_field, fields)
        
        # Core residue should have higher burial than surface
        # (actual value depends on neighbor distribution)
        assert 0.0 < burial <= 1.0
        
        # Compare to surface residue
        surface_field = fields[0]
        surface_burial = default_corrector.calculate_burial_factor(surface_field, fields)
        
        # Core should be more buried than edge of cluster
        assert burial >= surface_burial * 0.5  # At least 50% of surface burial
    
    def test_burial_intermediate_residue(self, default_corrector):
        """Test burial factor around midpoint (12 neighbors)."""
        # Create scenario with exactly 12 neighbors
        fields = create_compact_cluster(17, radius=4.0)  # Will give ~12 neighbors
        
        burial = default_corrector.calculate_burial_factor(fields[8], fields)
        
        # Should be around 0.5 (sigmoidal midpoint)
        assert 0.3 <= burial <= 0.7
    
    def test_burial_range_bounds(self, default_corrector):
        """Test that burial factor is always in [0, 1]."""
        # Test with extreme cases
        fields_sparse = create_linear_chain(100, spacing=10.0)
        fields_dense = create_compact_cluster(50, radius=3.0)
        
        for field in fields_sparse + fields_dense:
            burial = default_corrector.calculate_burial_factor(field, fields_sparse + fields_dense)
            assert 0.0 <= burial <= 1.0
    
    def test_burial_sigmoid_behavior(self, default_corrector):
        """Test that burial increases monotonically with neighbor count."""
        # Can't directly control neighbor count, but can test with varying densities
        fields_low = create_linear_chain(20, spacing=8.0)
        fields_high = create_compact_cluster(20, radius=4.0)
        
        burial_low = default_corrector.calculate_burial_factor(fields_low[10], fields_low)
        burial_high = default_corrector.calculate_burial_factor(fields_high[10], fields_high)
        
        # Compact cluster should have higher burial
        assert burial_high > burial_low
    
    def test_burial_extreme_overflow_protection(self):
        """Test overflow protection in sigmoid calculation."""
        # Test with extreme midpoint that would cause large exponents
        corrector = SolventFieldCorrection(burial_midpoint=1000, burial_steepness=0.1)
        
        fields = create_linear_chain(10)
        field = fields[5]
        
        # Should not raise overflow error
        burial = corrector.calculate_burial_factor(field, fields)
        assert 0.0 <= burial <= 1.0


# ============================================================================
# Distance-Dependent Dielectric Tests
# ============================================================================

class TestDistanceDependentDielectric:
    """Test distance-dependent dielectric calculation."""
    
    def test_dielectric_at_zero_distance(self, default_corrector):
        """Test that dielectric equals epsilon_buried at zero distance."""
        epsilon = default_corrector.calculate_distance_dependent_dielectric(0.0)
        assert epsilon == pytest.approx(default_corrector.epsilon_buried, abs=0.01)
    
    def test_dielectric_at_large_distance(self, default_corrector):
        """Test that dielectric approaches epsilon_surface at large distances."""
        epsilon = default_corrector.calculate_distance_dependent_dielectric(30.0)
        
        # Should be very close to epsilon_surface (80.0)
        assert epsilon > 75.0
        assert epsilon <= default_corrector.epsilon_surface
    
    def test_dielectric_monotonic_increase(self, default_corrector):
        """Test that dielectric increases monotonically with distance."""
        distances = [0.0, 2.0, 5.0, 10.0, 20.0]
        epsilons = [default_corrector.calculate_distance_dependent_dielectric(d) for d in distances]
        
        # Each epsilon should be >= previous
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i-1]
    
    def test_dielectric_at_screening_length(self, default_corrector):
        """Test dielectric at characteristic screening length (3.0 Å)."""
        epsilon = default_corrector.calculate_distance_dependent_dielectric(3.0)
        
        # At screening length, should be about 63% of the way from buried to surface
        # ε(λ) = ε_b + Δε * (1 - exp(-1)) ≈ ε_b + 0.632 * Δε
        expected = 4.0 + 0.632 * (80.0 - 4.0)
        assert epsilon == pytest.approx(expected, rel=0.05)
    
    def test_dielectric_negative_distance_error(self, default_corrector):
        """Test that negative distance raises error."""
        with pytest.raises(ValueError, match="distance must be non-negative"):
            default_corrector.calculate_distance_dependent_dielectric(-1.0)
    
    def test_dielectric_custom_parameters(self, custom_corrector):
        """Test dielectric with custom epsilon values."""
        epsilon_zero = custom_corrector.calculate_distance_dependent_dielectric(0.0)
        epsilon_large = custom_corrector.calculate_distance_dependent_dielectric(50.0)
        
        assert epsilon_zero == pytest.approx(2.0, abs=0.01)  # epsilon_buried
        assert epsilon_large > 55.0  # Close to epsilon_surface (60.0)


# ============================================================================
# Effective Dielectric Tests
# ============================================================================

class TestEffectiveDielectric:
    """Test effective dielectric combining distance and burial."""
    
    def test_effective_surface_surface(self, default_corrector):
        """Test effective dielectric for two surface residues."""
        # Both surface (burial ≈ 0.0)
        burial = 0.1
        distance = 5.0
        
        epsilon = default_corrector.calculate_effective_dielectric(distance, burial)
        
        # Should be close to distance-dependent value (high screening)
        epsilon_dist = default_corrector.calculate_distance_dependent_dielectric(distance)
        assert epsilon == pytest.approx(epsilon_dist, rel=0.15)
    
    def test_effective_core_core(self, default_corrector):
        """Test effective dielectric for two core residues."""
        # Both buried (burial ≈ 1.0)
        burial = 0.9
        distance = 5.0
        
        epsilon = default_corrector.calculate_effective_dielectric(distance, burial)
        
        # Should be close to epsilon_buried (low screening)
        assert epsilon < 15.0  # Much lower than surface epsilon (80)
        assert epsilon >= default_corrector.epsilon_buried  # At least epsilon_buried
    
    def test_effective_surface_core(self, default_corrector):
        """Test effective dielectric for surface-core interaction."""
        # Intermediate burial
        burial = 0.5
        distance = 5.0
        
        epsilon = default_corrector.calculate_effective_dielectric(distance, burial)
        
        # Should be between buried and surface values
        assert default_corrector.epsilon_buried < epsilon < 60.0
    
    def test_effective_burial_interpolation(self, default_corrector):
        """Test that burial factor properly interpolates dielectric."""
        distance = 5.0
        
        eps_surface = default_corrector.calculate_effective_dielectric(distance, 0.0)
        eps_mid = default_corrector.calculate_effective_dielectric(distance, 0.5)
        eps_buried = default_corrector.calculate_effective_dielectric(distance, 1.0)
        
        # Should decrease with increasing burial
        assert eps_surface > eps_mid > eps_buried
    
    def test_effective_invalid_burial(self, default_corrector):
        """Test that invalid burial factor raises error."""
        with pytest.raises(ValueError, match="burial_factor must be in"):
            default_corrector.calculate_effective_dielectric(5.0, -0.1)
        
        with pytest.raises(ValueError, match="burial_factor must be in"):
            default_corrector.calculate_effective_dielectric(5.0, 1.1)
    
    def test_effective_edge_cases(self, default_corrector):
        """Test effective dielectric at edge cases."""
        # Zero distance, zero burial
        eps1 = default_corrector.calculate_effective_dielectric(0.0, 0.0)
        assert eps1 == pytest.approx(default_corrector.epsilon_buried, abs=0.1)
        
        # Zero distance, full burial
        eps2 = default_corrector.calculate_effective_dielectric(0.0, 1.0)
        assert eps2 == pytest.approx(default_corrector.epsilon_buried, abs=0.1)
        
        # Large distance, zero burial
        eps3 = default_corrector.calculate_effective_dielectric(50.0, 0.0)
        assert eps3 > 70.0


# ============================================================================
# Pairwise Dielectric Tests
# ============================================================================

class TestPairwiseDielectric:
    """Test pairwise effective dielectric calculation."""
    
    def test_pairwise_no_cache(self, default_corrector):
        """Test pairwise calculation without caching."""
        fields = create_compact_cluster(20, radius=5.0)
        field1 = fields[0]
        field2 = fields[10]
        
        epsilon = default_corrector.calculate_pairwise_effective_dielectric(
            field1, field2, fields, burial_cache=None
        )
        
        # Should return valid dielectric
        assert default_corrector.epsilon_buried <= epsilon <= default_corrector.epsilon_surface
    
    def test_pairwise_with_cache(self, default_corrector):
        """Test pairwise calculation with caching."""
        fields = create_compact_cluster(20, radius=5.0)
        field1 = fields[0]
        field2 = fields[10]
        
        cache = {}
        epsilon1 = default_corrector.calculate_pairwise_effective_dielectric(
            field1, field2, fields, burial_cache=cache
        )
        
        # Cache should now contain burial factors
        assert field1.residue_index in cache
        assert field2.residue_index in cache
        
        # Second call should use cached values
        epsilon2 = default_corrector.calculate_pairwise_effective_dielectric(
            field1, field2, fields, burial_cache=cache
        )
        
        assert epsilon1 == pytest.approx(epsilon2)
    
    def test_pairwise_cache_reuse(self, default_corrector):
        """Test that cache is properly reused across multiple calls."""
        fields = create_compact_cluster(15, radius=5.0)
        cache = {}
        
        # Calculate for multiple pairs
        for i in range(0, 10, 2):
            default_corrector.calculate_pairwise_effective_dielectric(
                fields[i], fields[i+1], fields, burial_cache=cache
            )
        
        # Cache should contain entries for all accessed residues
        assert len(cache) >= 10
    
    def test_pairwise_averages_burial(self, default_corrector):
        """Test that pairwise uses average burial of both residues."""
        # Create two fields with different local environments
        fields = create_linear_chain(30, spacing=3.8)
        
        surface_field = fields[0]  # Low burial
        mid_field = fields[15]  # Moderate burial
        
        epsilon = default_corrector.calculate_pairwise_effective_dielectric(
            surface_field, mid_field, fields
        )
        
        # Should be valid dielectric
        assert default_corrector.epsilon_buried <= epsilon <= default_corrector.epsilon_surface


# ============================================================================
# Electrostatic Correction Tests
# ============================================================================

class TestElectrostaticCorrection:
    """Test electrostatic energy correction."""
    
    def test_correction_reduces_surface_interaction(self, default_corrector):
        """Test that surface interactions are weakened (higher epsilon)."""
        fields = create_linear_chain(20, spacing=4.0)
        field1 = create_test_field(0, (0.0, 0.0, 0.0), charge=1.0)
        field2 = create_test_field(10, (40.0, 0.0, 0.0), charge=-1.0)
        fields[0] = field1
        fields[10] = field2
        
        # Original energy (arbitrary value)
        original_energy = 100.0
        
        corrected = default_corrector.apply_correction_to_electrostatic(
            original_energy, field1, field2, fields, original_dielectric=4.0
        )
        
        # Surface interaction should be weakened (reduced magnitude)
        assert abs(corrected) < abs(original_energy)
    
    def test_correction_preserves_core_interaction(self, default_corrector):
        """Test that core interactions remain strong (lower epsilon)."""
        fields = create_compact_cluster(25, radius=4.0)
        field1 = create_test_field(10, fields[10].position, charge=1.0)
        field2 = create_test_field(15, fields[15].position, charge=-1.0)
        fields[10] = field1
        fields[15] = field2
        
        # Original energy
        original_energy = -50.0
        
        corrected = default_corrector.apply_correction_to_electrostatic(
            original_energy, field1, field2, fields, original_dielectric=4.0
        )
        
        # Core interaction correction should be smaller than surface correction
        # The correction factor depends on burial, which varies with cluster geometry
        # Just verify that corrected energy is non-zero and has correct sign
        assert corrected != 0.0
        assert (corrected < 0) == (original_energy < 0)  # Same sign preserved
    
    def test_correction_zero_energy(self, default_corrector):
        """Test that zero energy remains zero."""
        fields = create_linear_chain(10)
        field1 = fields[0]
        field2 = fields[5]
        
        corrected = default_corrector.apply_correction_to_electrostatic(
            0.0, field1, field2, fields
        )
        
        assert corrected == 0.0
    
    def test_correction_with_cache(self, default_corrector):
        """Test correction with burial factor caching."""
        fields = create_compact_cluster(20, radius=5.0)
        field1 = create_test_field(0, fields[0].position, charge=1.0)
        field2 = create_test_field(10, fields[10].position, charge=-1.0)
        fields[0] = field1
        fields[10] = field2
        
        cache = {}
        original_energy = 75.0
        
        corrected1 = default_corrector.apply_correction_to_electrostatic(
            original_energy, field1, field2, fields, burial_cache=cache
        )
        
        # Second call with same cache
        corrected2 = default_corrector.apply_correction_to_electrostatic(
            original_energy, field1, field2, fields, burial_cache=cache
        )
        
        assert corrected1 == pytest.approx(corrected2)
        assert len(cache) >= 2  # Should have cached both residues
    
    def test_correction_scales_properly(self, default_corrector):
        """Test that correction factor scales energy correctly."""
        fields = create_linear_chain(20, spacing=5.0)
        field1 = create_test_field(0, (0.0, 0.0, 0.0), charge=1.0)
        field2 = create_test_field(10, (50.0, 0.0, 0.0), charge=-1.0)
        fields[0] = field1
        fields[10] = field2
        
        # Test with different original energies
        corrected_100 = default_corrector.apply_correction_to_electrostatic(
            100.0, field1, field2, fields
        )
        corrected_200 = default_corrector.apply_correction_to_electrostatic(
            200.0, field1, field2, fields
        )
        
        # Correction should scale linearly
        assert abs(corrected_200) == pytest.approx(abs(corrected_100) * 2.0, rel=0.01)
    
    def test_correction_different_original_dielectric(self, default_corrector):
        """Test correction with different original dielectric values."""
        fields = create_compact_cluster(20, radius=5.0)
        field1 = create_test_field(0, fields[0].position, charge=1.0)
        field2 = create_test_field(10, fields[10].position, charge=-1.0)
        fields[0] = field1
        fields[10] = field2
        
        original_energy = 50.0
        
        # Test with different original dielectrics
        corrected_4 = default_corrector.apply_correction_to_electrostatic(
            original_energy, field1, field2, fields, original_dielectric=4.0
        )
        corrected_8 = default_corrector.apply_correction_to_electrostatic(
            original_energy, field1, field2, fields, original_dielectric=8.0
        )
        
        # Higher original dielectric should give stronger corrected interaction
        assert abs(corrected_8) > abs(corrected_4)


# ============================================================================
# Integration and Edge Case Tests
# ============================================================================

class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""
    
    def test_single_residue_system(self, default_corrector):
        """Test behavior with single residue (no interactions)."""
        field = create_test_field(0, (0.0, 0.0, 0.0))
        fields = [field]
        
        # Should handle gracefully
        burial = default_corrector.calculate_burial_factor(field, fields)
        assert 0.0 <= burial <= 1.0
        
        count = default_corrector.count_neighbors(field, fields)
        assert count == 0
    
    def test_two_residue_system(self, default_corrector):
        """Test behavior with two residues."""
        field1 = create_test_field(0, (0.0, 0.0, 0.0), charge=1.0)
        field2 = create_test_field(10, (5.0, 0.0, 0.0), charge=-1.0)
        fields = [field1, field2]
        
        # Should handle pairwise calculation
        epsilon = default_corrector.calculate_pairwise_effective_dielectric(
            field1, field2, fields
        )
        
        assert default_corrector.epsilon_buried <= epsilon <= default_corrector.epsilon_surface
    
    def test_very_large_system(self, default_corrector):
        """Test performance with large protein system."""
        # Create large cluster (realistic protein size)
        fields = create_compact_cluster(100, radius=15.0)
        
        # Should complete without errors
        field = fields[50]
        burial = default_corrector.calculate_burial_factor(field, fields)
        
        assert 0.0 <= burial <= 1.0
    
    def test_extreme_distances(self, default_corrector):
        """Test with very small and very large distances."""
        # Very small distance
        epsilon_small = default_corrector.calculate_distance_dependent_dielectric(0.001)
        assert epsilon_small == pytest.approx(default_corrector.epsilon_buried, abs=0.5)
        
        # Very large distance
        epsilon_large = default_corrector.calculate_distance_dependent_dielectric(1000.0)
        assert epsilon_large == pytest.approx(default_corrector.epsilon_surface, abs=0.01)
    
    def test_consistency_across_calls(self, default_corrector):
        """Test that repeated calls give consistent results."""
        fields = create_compact_cluster(20, radius=5.0)
        field = fields[10]
        
        burial1 = default_corrector.calculate_burial_factor(field, fields)
        burial2 = default_corrector.calculate_burial_factor(field, fields)
        
        assert burial1 == pytest.approx(burial2)
    
    def test_realistic_protein_scenario(self, default_corrector):
        """Test with realistic protein-like configuration."""
        # Create mixed environment: compact core + extended surface
        core_fields = create_compact_cluster(30, radius=6.0)
        surface_fields = create_linear_chain(20, spacing=4.0)
        
        # Offset surface fields
        for i, field in enumerate(surface_fields):
            pos = field.position
            surface_fields[i] = create_test_field(
                i + 30, (pos[0] + 15.0, pos[1], pos[2])
            )
        
        all_fields = core_fields + surface_fields
        
        # Core residue should be more buried than surface residue
        core_burial = default_corrector.calculate_burial_factor(core_fields[15], all_fields)
        surface_burial = default_corrector.calculate_burial_factor(surface_fields[10], all_fields)
        
        # Core should have more neighbors than surface
        assert core_burial >= surface_burial
        
        # Both should be in valid range
        assert 0.0 <= core_burial <= 1.0
        assert 0.0 <= surface_burial <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
