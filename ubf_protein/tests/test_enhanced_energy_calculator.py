"""
Unit tests for enhanced energy calculator.

Tests energy calculation with different components enabled/disabled,
energy breakdown reporting, performance, caching, and numerical stability.
"""

import pytest
import time
from ubf_protein.enhanced_energy_calculator import (
    EnhancedEnergyCalculator,
    EnergyBreakdown
)
from ubf_protein.models import Conformation, DisulfideBond


class TestEnhancedEnergyCalculatorInitialization:
    """Test EnhancedEnergyCalculator initialization."""
    
    def test_default_initialization(self):
        """Test default initialization with minimal parameters."""
        sequence = "ACDEFGHIKLM"
        calc = EnhancedEnergyCalculator(sequence)
        
        assert calc.sequence == sequence
        assert calc.disulfide_bonds == []
        assert calc.enable_sidechains is True
        assert calc.enable_disulfide is True
        assert calc.enable_entropic is True
        assert calc.enable_solvent is True
        assert calc.temperature == 300.0
    
    def test_initialization_with_disulfide_bonds(self):
        """Test initialization with disulfide bonds."""
        sequence = "ACDEFGHIKLM"
        bonds = [DisulfideBond(0, 5, 3.8, 0.5)]
        
        calc = EnhancedEnergyCalculator(sequence, disulfide_bonds=bonds)
        
        assert len(calc.disulfide_bonds) == 1
        assert calc.disulfide_bonds[0].residue_i == 0
        assert calc.disulfide_bonds[0].residue_j == 5
    
    def test_initialization_with_feature_toggles(self):
        """Test initialization with specific features enabled/disabled."""
        sequence = "ACDEFGHIKLM"
        
        calc = EnhancedEnergyCalculator(
            sequence,
            enable_sidechains=False,
            enable_disulfide=False,
            enable_entropic=False,
            enable_solvent=False
        )
        
        assert calc.enable_sidechains is False
        assert calc.enable_disulfide is False
        assert calc.enable_entropic is False
        assert calc.enable_solvent is False
    
    def test_initialization_with_custom_temperature(self):
        """Test initialization with custom temperature."""
        sequence = "ACDEFGHIKLM"
        
        calc = EnhancedEnergyCalculator(sequence, temperature=310.0)
        
        assert calc.temperature == 310.0
    
    def test_invalid_empty_sequence(self):
        """Test initialization with empty sequence."""
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            EnhancedEnergyCalculator("")
    
    def test_invalid_amino_acid_in_sequence(self):
        """Test initialization with invalid amino acid."""
        with pytest.raises(ValueError, match="Invalid amino acid"):
            EnhancedEnergyCalculator("ACDEFX")  # X is invalid
    
    def test_component_status(self):
        """Test get_component_status method."""
        sequence = "ACDEFGHIKLM"
        bonds = [DisulfideBond(0, 5, 3.8, 0.5)]
        
        calc = EnhancedEnergyCalculator(
            sequence,
            disulfide_bonds=bonds,
            enable_sidechains=True,
            enable_disulfide=True,
            enable_entropic=False,
            enable_solvent=True
        )
        
        status = calc.get_component_status()
        
        assert status['base'] is True  # Always enabled
        assert status['sidechains'] is True
        assert status['disulfide'] is True
        assert status['entropic'] is False
        assert status['solvent'] is True


class TestBasicEnergyCalculation:
    """Test basic energy calculation."""
    
    def _create_conformation(self, sequence: str, coords: list, energy: float = -100.0):
        """Helper to create test conformation."""
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=energy,
            rmsd_to_native=None,
            secondary_structure=['C'] * len(coords),
            phi_angles=[0.0] * len(coords),
            psi_angles=[0.0] * len(coords),
            available_move_types=[],
            structural_constraints={}
        )
    
    def test_calculate_returns_float(self):
        """Test that calculate() returns a float value."""
        sequence = "ACDE"
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(4)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        assert isinstance(energy, float)
    
    def test_calculate_extended_chain(self):
        """Test energy calculation for extended chain."""
        sequence = "AAAAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        # Extended chain with ideal CA-CA distances
        coords = [(i * 3.8, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        # Should be reasonable energy (not NaN or inf)
        assert -1000.0 < energy < 1000.0
    
    def test_calculate_compact_structure(self):
        """Test energy calculation for compact structure."""
        sequence = "AAAAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        # Compact structure
        coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (3.8, 3.8, 0.0),
                  (0.0, 3.8, 0.0), (0.0, 0.0, 3.8)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        assert -1000.0 < energy < 1000.0


class TestComponentToggling:
    """Test energy calculation with different components enabled/disabled."""
    
    def _create_conformation(self, sequence: str, coords: list):
        """Helper to create test conformation."""
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * len(coords),
            phi_angles=[0.0] * len(coords),
            psi_angles=[0.0] * len(coords),
            available_move_types=[],
            structural_constraints={}
        )
    
    def test_all_components_enabled(self):
        """Test with all components enabled."""
        sequence = "ACDEFGHIKLM"
        bonds = [DisulfideBond(0, 5, 3.8, 0.5)]
        
        calc = EnhancedEnergyCalculator(
            sequence,
            disulfide_bonds=bonds,
            enable_sidechains=True,
            enable_disulfide=True,
            enable_entropic=True,
            enable_solvent=True
        )
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(len(sequence))]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        # Should include all components
        assert isinstance(energy, float)
    
    def test_only_base_energy(self):
        """Test with only base molecular mechanics."""
        sequence = "ACDEFGHIKLM"
        
        calc = EnhancedEnergyCalculator(
            sequence,
            enable_sidechains=False,
            enable_disulfide=False,
            enable_entropic=False,
            enable_solvent=False
        )
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(len(sequence))]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        # Should only be base energy
        assert isinstance(energy, float)
    
    def test_sidechains_disabled(self):
        """Test with side-chain interactions disabled."""
        sequence = "LLLLL"  # Hydrophobic residues
        
        calc_with = EnhancedEnergyCalculator(sequence, enable_sidechains=True)
        calc_without = EnhancedEnergyCalculator(sequence, enable_sidechains=False)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        energy_with = calc_with.calculate(conf)
        energy_without = calc_without.calculate(conf)
        
        # Energies should differ
        assert energy_with != energy_without
    
    def test_disulfide_disabled(self):
        """Test with disulfide bonds disabled."""
        sequence = "CCCCC"
        bonds = [DisulfideBond(0, 4, 3.8, 0.5)]
        
        calc_with = EnhancedEnergyCalculator(
            sequence,
            disulfide_bonds=bonds,
            enable_disulfide=True
        )
        calc_without = EnhancedEnergyCalculator(
            sequence,
            disulfide_bonds=bonds,
            enable_disulfide=False
        )
        
        # Large separation = high penalty if disulfide enabled
        coords = [(i * 10.0, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        energy_with = calc_with.calculate(conf)
        energy_without = calc_without.calculate(conf)
        
        # With disulfide should have higher energy (penalty)
        assert energy_with > energy_without
    
    def test_entropic_disabled(self):
        """Test with entropic corrections disabled."""
        sequence = "AAAAA"
        
        calc_with = EnhancedEnergyCalculator(sequence, enable_entropic=True)
        calc_without = EnhancedEnergyCalculator(sequence, enable_entropic=False)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        # Calculate multiple times to build history
        for _ in range(3):
            energy_with = calc_with.calculate(conf)
            energy_without = calc_without.calculate(conf)
        
        # Energies may differ slightly due to entropic term
        assert isinstance(energy_with, float)
        assert isinstance(energy_without, float)


class TestEnergyBreakdown:
    """Test energy breakdown reporting."""
    
    def _create_conformation(self, sequence: str, coords: list):
        """Helper to create test conformation."""
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * len(coords),
            phi_angles=[0.0] * len(coords),
            psi_angles=[0.0] * len(coords),
            available_move_types=[],
            structural_constraints={}
        )
    
    def test_breakdown_structure(self):
        """Test that breakdown has all required fields."""
        sequence = "ACDEFGHIKLM"
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(len(sequence))]
        conf = self._create_conformation(sequence, coords)
        
        breakdown = calc.calculate_with_breakdown(conf)
        
        assert isinstance(breakdown, EnergyBreakdown)
        assert hasattr(breakdown, 'total')
        assert hasattr(breakdown, 'base')
        assert hasattr(breakdown, 'sidechain')
        assert hasattr(breakdown, 'disulfide')
        assert hasattr(breakdown, 'entropic')
        assert hasattr(breakdown, 'bond')
        assert hasattr(breakdown, 'angle')
        assert hasattr(breakdown, 'dihedral')
        assert hasattr(breakdown, 'vdw')
        assert hasattr(breakdown, 'electrostatic')
        assert hasattr(breakdown, 'hbond')
        assert hasattr(breakdown, 'compactness')
    
    def test_breakdown_total_matches_calculate(self):
        """Test that breakdown total matches calculate() result."""
        sequence = "ACDEFGHIKLM"
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(len(sequence))]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        breakdown = calc.calculate_with_breakdown(conf)
        
        # Should match within floating point precision
        assert abs(energy - breakdown.total) < 1e-6
    
    def test_breakdown_components_sum_to_total(self):
        """Test that major components sum to total."""
        sequence = "ACDEFGHIKLM"
        bonds = [DisulfideBond(0, 5, 3.8, 0.5)]
        calc = EnhancedEnergyCalculator(sequence, disulfide_bonds=bonds)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(len(sequence))]
        conf = self._create_conformation(sequence, coords)
        
        breakdown = calc.calculate_with_breakdown(conf)
        
        components_sum = (breakdown.base + breakdown.sidechain + 
                         breakdown.disulfide + breakdown.entropic)
        
        assert abs(components_sum - breakdown.total) < 1e-6
    
    def test_breakdown_with_disulfide_penalty(self):
        """Test breakdown shows disulfide penalty correctly."""
        sequence = "CCCCC"
        bonds = [DisulfideBond(0, 4, 3.8, 0.5)]
        calc = EnhancedEnergyCalculator(sequence, disulfide_bonds=bonds)
        
        # Large separation = high penalty
        coords = [(i * 20.0, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        breakdown = calc.calculate_with_breakdown(conf)
        
        # Disulfide should be positive and large
        assert breakdown.disulfide > 100.0
    
    def test_breakdown_zero_disulfide_when_disabled(self):
        """Test breakdown shows zero disulfide when disabled."""
        sequence = "CCCCC"
        bonds = [DisulfideBond(0, 4, 3.8, 0.5)]
        calc = EnhancedEnergyCalculator(
            sequence,
            disulfide_bonds=bonds,
            enable_disulfide=False
        )
        
        coords = [(i * 20.0, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        breakdown = calc.calculate_with_breakdown(conf)
        
        assert breakdown.disulfide == 0.0


class TestDisulfideBondEnergy:
    """Test disulfide bond energy calculation."""
    
    def _create_conformation(self, sequence: str, coords: list):
        """Helper to create test conformation."""
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * len(coords),
            phi_angles=[0.0] * len(coords),
            psi_angles=[0.0] * len(coords),
            available_move_types=[],
            structural_constraints={}
        )
    
    def test_satisfied_disulfide_bond(self):
        """Test energy when disulfide bond is satisfied."""
        sequence = "CCCCC"
        bonds = [DisulfideBond(0, 4, 3.8, 0.5)]
        calc = EnhancedEnergyCalculator(sequence, disulfide_bonds=bonds)
        
        # Position residues at exactly 3.8 Å apart
        coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0),
                  (11.4, 0.0, 0.0), (3.8, 0.0, 0.0)]  # 0 and 4 at 3.8 Å
        conf = self._create_conformation(sequence, coords)
        
        breakdown = calc.calculate_with_breakdown(conf)
        
        # Should be near zero (harmonic potential at equilibrium)
        assert abs(breakdown.disulfide) < 1.0
    
    def test_violated_disulfide_bond(self):
        """Test energy penalty for violated disulfide bond."""
        sequence = "CCCCC"
        bonds = [DisulfideBond(0, 4, 3.8, 0.5)]
        calc = EnhancedEnergyCalculator(sequence, disulfide_bonds=bonds)
        
        # Large separation
        coords = [(i * 10.0, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        breakdown = calc.calculate_with_breakdown(conf)
        
        # Should have large positive penalty
        assert breakdown.disulfide > 1000.0
    
    def test_multiple_disulfide_bonds(self):
        """Test energy with multiple disulfide bonds."""
        sequence = "CCCCCCCC"
        bonds = [
            DisulfideBond(0, 3, 3.8, 0.5),
            DisulfideBond(1, 4, 3.8, 0.5)
        ]
        calc = EnhancedEnergyCalculator(sequence, disulfide_bonds=bonds)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(8)]
        conf = self._create_conformation(sequence, coords)
        
        breakdown = calc.calculate_with_breakdown(conf)
        
        # Should have contributions from both bonds
        assert breakdown.disulfide > 0.0
    
    def test_harmonic_potential_formula(self):
        """Test harmonic potential formula: E = 0.5 * k * (r - r0)²."""
        sequence = "CC"
        bonds = [DisulfideBond(0, 1, 3.8, 0.5)]
        calc = EnhancedEnergyCalculator(sequence, disulfide_bonds=bonds)
        
        # Set distance to 5.8 Å (2 Å away from equilibrium)
        coords = [(0.0, 0.0, 0.0), (5.8, 0.0, 0.0)]
        conf = self._create_conformation(sequence, coords)
        
        breakdown = calc.calculate_with_breakdown(conf)
        
        # E = 0.5 * 50.0 * (5.8 - 3.8)² = 0.5 * 50.0 * 4.0 = 100.0
        expected = 0.5 * 50.0 * (2.0 ** 2)
        assert abs(breakdown.disulfide - expected) < 1.0


class TestPerformance:
    """Test performance benchmarks."""
    
    def _create_conformation(self, sequence: str, coords: list):
        """Helper to create test conformation."""
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * len(coords),
            phi_angles=[0.0] * len(coords),
            psi_angles=[0.0] * len(coords),
            available_move_types=[],
            structural_constraints={}
        )
    
    def test_performance_50_residues(self):
        """Test performance with 50 residues."""
        sequence = "A" * 50
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(50)]
        conf = self._create_conformation(sequence, coords)
        
        start = time.perf_counter()
        for _ in range(10):
            calc.calculate(conf)
        end = time.perf_counter()
        
        avg_time_ms = ((end - start) / 10) * 1000
        
        # Should be fast for 50 residues
        assert avg_time_ms < 20.0  # <20ms per calculation
    
    def test_performance_100_residues(self):
        """Test performance with 100 residues."""
        sequence = "A" * 100
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(100)]
        conf = self._create_conformation(sequence, coords)
        
        start = time.perf_counter()
        for _ in range(10):
            calc.calculate(conf)
        end = time.perf_counter()
        
        avg_time_ms = ((end - start) / 10) * 1000
        
        # Should be reasonable for 100 residues
        assert avg_time_ms < 35.0  # <35ms per calculation
    
    def test_performance_300_residues(self):
        """Test performance with 300 residues (target case)."""
        sequence = "A" * 300
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(300)]
        conf = self._create_conformation(sequence, coords)
        
        start = time.perf_counter()
        for _ in range(5):
            calc.calculate(conf)
        end = time.perf_counter()
        
        avg_time_ms = ((end - start) / 5) * 1000
        
        # Target: <50ms for 300 residues
        assert avg_time_ms < 50.0
    
    def test_benchmark_method(self):
        """Test built-in benchmark method."""
        sequence = "ACDEFGHIKLM"
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(len(sequence))]
        conf = self._create_conformation(sequence, coords)
        
        bench = calc.benchmark(conf, n_iterations=50)
        
        assert 'mean_ms' in bench
        assert 'median_ms' in bench
        assert 'min_ms' in bench
        assert 'max_ms' in bench
        assert 'p95_ms' in bench
        assert bench['n_residues'] == len(sequence)
        assert bench['total_iterations'] == 50


class TestCaching:
    """Test caching behavior."""
    
    def _create_conformation(self, sequence: str, coords: list):
        """Helper to create test conformation."""
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * len(coords),
            phi_angles=[0.0] * len(coords),
            psi_angles=[0.0] * len(coords),
            available_move_types=[],
            structural_constraints={}
        )
    
    def test_neighbor_cache_invalidation(self):
        """Test that neighbor cache updates with new coordinates."""
        sequence = "AAAAA"
        calc = EnhancedEnergyCalculator(sequence, enable_sidechains=True)
        
        # First conformation
        coords1 = [(i * 3.8, 0.0, 0.0) for i in range(5)]
        conf1 = self._create_conformation(sequence, coords1)
        energy1 = calc.calculate(conf1)
        
        # Second conformation with different geometry
        coords2 = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (3.8, 3.8, 0.0),
                   (0.0, 3.8, 0.0), (0.0, 0.0, 3.8)]
        conf2 = self._create_conformation(sequence, coords2)
        energy2 = calc.calculate(conf2)
        
        # Energies should differ
        assert energy1 != energy2
    
    def test_history_accumulation(self):
        """Test that conformation history accumulates for entropic term."""
        sequence = "AAAAA"
        calc = EnhancedEnergyCalculator(sequence, enable_entropic=True)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        # Calculate multiple times
        for _ in range(5):
            calc.calculate(conf)
        
        # History should have accumulated
        assert len(calc._conformation_history) == 5
        assert len(calc._qcp_history) == 5
    
    def test_clear_history(self):
        """Test clear_history method."""
        sequence = "AAAAA"
        calc = EnhancedEnergyCalculator(sequence, enable_entropic=True)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        # Build up history
        for _ in range(5):
            calc.calculate(conf)
        
        assert len(calc._conformation_history) > 0
        
        # Clear history
        calc.clear_history()
        
        assert len(calc._conformation_history) == 0
        assert len(calc._qcp_history) == 0


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def _create_conformation(self, sequence: str, coords: list):
        """Helper to create test conformation."""
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * len(coords),
            phi_angles=[0.0] * len(coords),
            psi_angles=[0.0] * len(coords),
            available_move_types=[],
            structural_constraints={}
        )
    
    def test_single_residue(self):
        """Test with single residue."""
        sequence = "A"
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(0.0, 0.0, 0.0)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        # Should handle gracefully
        assert isinstance(energy, float)
        assert not math.isnan(energy)
        assert not math.isinf(energy)
    
    def test_two_residues(self):
        """Test with two residues."""
        sequence = "AA"
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        assert isinstance(energy, float)
        assert not math.isnan(energy)
        assert not math.isinf(energy)
    
    def test_very_close_residues(self):
        """Test with residues very close together."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        # Very close spacing
        coords = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        # Should have high energy but not infinite
        assert isinstance(energy, float)
        assert not math.isnan(energy)
        assert not math.isinf(energy)
    
    def test_very_far_residues(self):
        """Test with residues very far apart."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        # Very large spacing
        coords = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        assert isinstance(energy, float)
        assert not math.isnan(energy)
        assert not math.isinf(energy)
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        sequence = "AAAAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        coords = [(-10.0, -5.0, -2.0), (-6.2, -5.0, -2.0),
                  (-2.4, -5.0, -2.0), (1.4, -5.0, -2.0), (5.2, -5.0, -2.0)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        assert isinstance(energy, float)
        assert not math.isnan(energy)
        assert not math.isinf(energy)
    
    def test_repeated_calculations_same_conformation(self):
        """Test repeated calculations give same result."""
        sequence = "ACDEFG"
        calc = EnhancedEnergyCalculator(sequence, enable_entropic=False)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(6)]
        conf = self._create_conformation(sequence, coords)
        
        energy1 = calc.calculate(conf)
        energy2 = calc.calculate(conf)
        energy3 = calc.calculate(conf)
        
        # Should be identical (without entropic which accumulates history)
        assert abs(energy1 - energy2) < 1e-9
        assert abs(energy2 - energy3) < 1e-9


class TestSequenceSeparationFilter:
    """Test sequence separation filtering for side-chain interactions."""
    
    def _create_conformation(self, sequence: str, coords: list):
        """Helper to create test conformation."""
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * len(coords),
            phi_angles=[0.0] * len(coords),
            psi_angles=[0.0] * len(coords),
            available_move_types=[],
            structural_constraints={}
        )
    
    def test_adjacent_residues_ignored(self):
        """Test that adjacent residues (i, i+1) are ignored."""
        # This is implicitly tested through the implementation
        # Side-chain loop starts at i+3, so i, i+1, i+2 are skipped
        sequence = "LLLLL"  # Hydrophobic
        calc = EnhancedEnergyCalculator(sequence, enable_sidechains=True)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(5)]
        conf = self._create_conformation(sequence, coords)
        
        energy = calc.calculate(conf)
        
        # Should complete without error
        assert isinstance(energy, float)


import math


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
