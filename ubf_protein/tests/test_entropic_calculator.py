"""
Unit tests for entropic calculator.

Tests coherence entropy, configurational entropy, temperature dependence,
and edge cases for the EntropicCalculator class.
"""

import pytest
import math
from ubf_protein.entropic_calculator import (
    EntropicCalculator,
    EntropicContributions
)
from ubf_protein.models import Conformation


class TestEntropicCalculatorInitialization:
    """Test EntropicCalculator initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        calc = EntropicCalculator()
        assert calc.temperature == 300.0
        assert calc.boltzmann_constant == 0.001987
        assert calc.max_variance == 10.0
        assert calc.window_size == 50
    
    def test_custom_temperature(self):
        """Test initialization with custom temperature."""
        calc = EntropicCalculator(temperature=310.0)
        assert calc.temperature == 310.0
    
    def test_custom_parameters(self):
        """Test initialization with all custom parameters."""
        calc = EntropicCalculator(
            temperature=320.0,
            boltzmann_constant=0.002,
            max_variance=15.0,
            window_size=100
        )
        assert calc.temperature == 320.0
        assert calc.boltzmann_constant == 0.002
        assert calc.max_variance == 15.0
        assert calc.window_size == 100
    
    def test_invalid_temperature(self):
        """Test initialization with invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            EntropicCalculator(temperature=0.0)
        
        with pytest.raises(ValueError, match="temperature must be positive"):
            EntropicCalculator(temperature=-100.0)
    
    def test_invalid_boltzmann_constant(self):
        """Test initialization with invalid Boltzmann constant."""
        with pytest.raises(ValueError, match="boltzmann_constant must be positive"):
            EntropicCalculator(boltzmann_constant=0.0)
        
        with pytest.raises(ValueError, match="boltzmann_constant must be positive"):
            EntropicCalculator(boltzmann_constant=-0.001)
    
    def test_invalid_max_variance(self):
        """Test initialization with invalid max variance."""
        with pytest.raises(ValueError, match="max_variance must be positive"):
            EntropicCalculator(max_variance=0.0)
    
    def test_invalid_conformation_window(self):
        """Test initialization with invalid conformation window."""
        with pytest.raises(ValueError, match="window_size must be"):
            EntropicCalculator(window_size=1)
        
        with pytest.raises(ValueError, match="window_size must be"):
            EntropicCalculator(window_size=0)


class TestQCPVarianceCalculation:
    """Test QCP variance calculation."""
    
    def test_uniform_qcp_values(self):
        """Test variance with uniform QCP values."""
        calc = EntropicCalculator()
        qcp_values = [4.0, 4.0, 4.0, 4.0, 4.0]
        
        variance = calc.calculate_qcp_variance(qcp_values)
        
        assert variance == 0.0
    
    def test_low_variance_qcp(self):
        """Test variance with low spread QCP values."""
        calc = EntropicCalculator()
        qcp_values = [4.0, 4.1, 3.9, 4.05, 3.95]
        
        variance = calc.calculate_qcp_variance(qcp_values)
        
        # Variance should be small but positive
        assert 0.0 < variance < 0.1
    
    def test_high_variance_qcp(self):
        """Test variance with high spread QCP values."""
        calc = EntropicCalculator()
        qcp_values = [2.0, 4.0, 6.0, 8.0, 10.0]
        
        variance = calc.calculate_qcp_variance(qcp_values)
        
        # Variance should be substantial
        assert variance > 5.0
    
    def test_variance_clamping(self):
        """Test variance is clamped to max_variance."""
        calc = EntropicCalculator(max_variance=5.0)
        qcp_values = [0.0, 5.0, 10.0, 15.0, 20.0]
        
        variance = calc.calculate_qcp_variance(qcp_values)
        
        # Should be clamped to max_variance
        assert variance == 5.0
    
    def test_single_qcp_value(self):
        """Test variance with single QCP value."""
        calc = EntropicCalculator()
        qcp_values = [4.0]
        
        variance = calc.calculate_qcp_variance(qcp_values)
        
        assert variance == 0.0
    
    def test_empty_qcp_values(self):
        """Test variance with empty QCP list."""
        calc = EntropicCalculator()
        
        with pytest.raises(ValueError, match="qcp_values cannot be empty"):
            calc.calculate_qcp_variance([])


class TestCoherenceEntropy:
    """Test coherence entropy calculation."""
    
    def test_zero_variance_entropy(self):
        """Test entropy with zero variance."""
        calc = EntropicCalculator()
        qcp_values = [4.0, 4.0, 4.0]
        
        entropy, free_energy = calc.calculate_coherence_entropy(qcp_values)
        
        # S = k_B * ln(1 + 0) = 0
        assert entropy == 0.0
        assert free_energy == 0.0
    
    def test_low_variance_entropy(self):
        """Test entropy with low variance."""
        calc = EntropicCalculator()
        qcp_values = [4.0, 4.2, 3.8, 4.1, 3.9]
        
        entropy, free_energy = calc.calculate_coherence_entropy(qcp_values)
        
        # Should be small positive value
        assert 0.0 < entropy < 0.001
        assert free_energy < 0.0  # Negative (favorable)
    
    def test_high_variance_entropy(self):
        """Test entropy with high variance."""
        calc = EntropicCalculator()
        qcp_values = [2.0, 4.0, 6.0, 8.0, 10.0]
        
        entropy, free_energy = calc.calculate_coherence_entropy(qcp_values)
        
        # Should be larger entropy
        assert entropy > 0.003
        assert free_energy < -0.5  # Significant negative contribution
    
    def test_entropy_formula(self):
        """Test entropy formula S = k_B * ln(1 + variance)."""
        calc = EntropicCalculator()
        qcp_values = [3.0, 5.0]  # Variance = 1.0
        
        entropy, free_energy = calc.calculate_coherence_entropy(qcp_values)
        
        # S = 0.001987 * ln(1 + 1.0) = 0.001987 * ln(2)
        expected = 0.001987 * math.log(2.0)
        assert abs(entropy - expected) < 1e-6
        assert abs(free_energy - (-300.0 * expected)) < 1e-6
    
    def test_entropy_free_energy(self):
        """Test free energy contribution from coherence entropy."""
        calc = EntropicCalculator(temperature=300.0)
        qcp_values = [3.0, 5.0]  # Known variance
        
        entropy, free_energy = calc.calculate_coherence_entropy(qcp_values)
        
        # Should be negative (favorable)
        assert free_energy < 0.0
        
        # Should match -T*S
        expected = -300.0 * entropy
        assert abs(free_energy - expected) < 1e-6


class TestConfigurationalEntropy:
    """Test configurational entropy calculation."""
    
    def _create_conformation(self, conf_id: str, coords: list, energy: float = -100.0):
        """Helper to create test conformation."""
        sequence = "A" * len(coords)
        return Conformation(
            conformation_id=conf_id,
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
    
    def test_identical_conformations(self):
        """Test entropy with identical conformations."""
        calc = EntropicCalculator()
        
        coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)]
        conformations = [
            self._create_conformation(f"c{i}", coords)
            for i in range(5)
        ]
        
        entropy, free_energy, rmsd = calc.calculate_configurational_entropy(conformations)
        
        # All RMSDs are 0, so avg_rmsd = 0, S = 0
        assert entropy == 0.0
        assert free_energy == 0.0
        assert rmsd == 0.0
    
    def test_low_diversity_conformations(self):
        """Test entropy with low structural diversity."""
        calc = EntropicCalculator()
        
        conformations = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0), (7.6, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 0.1, 0), (7.6, 0.2, 0)]),
            self._create_conformation("c3", [(0, 0, 0), (3.8, 0.05, 0), (7.6, 0.1, 0)])
        ]
        
        entropy, free_energy, rmsd = calc.calculate_configurational_entropy(conformations)
        
        # Small RMSDs, small entropy
        assert 0.0 < entropy < 0.0005
        assert free_energy < 0.0
        assert rmsd > 0.0
    
    def test_high_diversity_conformations(self):
        """Test entropy with high structural diversity."""
        calc = EntropicCalculator()
        
        conformations = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0), (7.6, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 2, 0), (7.6, 4, 0)]),
            self._create_conformation("c3", [(0, 0, 0), (3.8, -2, 0), (7.6, -4, 0)]),
            self._create_conformation("c4", [(0, 0, 0), (0, 3.8, 0), (0, 7.6, 0)])
        ]
        
        entropy, free_energy, rmsd = calc.calculate_configurational_entropy(conformations)
        
        # Large RMSDs, larger entropy
        assert entropy > 0.002
        assert free_energy < -0.5
        assert rmsd > 2.0
    
    def test_entropy_increases_with_diversity(self):
        """Test entropy increases as conformational diversity increases."""
        calc = EntropicCalculator()
        
        # Low diversity
        low_div = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 0.1, 0)])
        ]
        
        # High diversity
        high_div = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 2.0, 0)])
        ]
        
        entropy_low, _, _ = calc.calculate_configurational_entropy(low_div)
        entropy_high, _, _ = calc.calculate_configurational_entropy(high_div)
        
        assert entropy_high > entropy_low
    
    def test_conformation_window_limiting(self):
        """Test that only last N conformations are used."""
        calc = EntropicCalculator(window_size=3)
        
        # Create many conformations, last 3 are different
        conformations = [
            self._create_conformation(f"c{i}", [(0, 0, 0), (3.8, 0, 0)])
            for i in range(10)
        ]
        # Make last 3 diverse
        conformations[-3] = self._create_conformation("c_last3", [(0, 0, 0), (3.8, 2, 0)])
        conformations[-2] = self._create_conformation("c_last2", [(0, 0, 0), (3.8, -2, 0)])
        conformations[-1] = self._create_conformation("c_last1", [(0, 0, 0), (0, 3.8, 0)])
        
        entropy, _, _ = calc.calculate_configurational_entropy(conformations)
        
        # Should only use last 3 (diverse), so entropy should be substantial
        assert entropy > 0.002
    
    def test_insufficient_conformations(self):
        """Test with less than 2 conformations."""
        calc = EntropicCalculator()
        
        # Single conformation
        confs = [self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0)])]
        
        entropy, free_energy, rmsd = calc.calculate_configurational_entropy(confs)
        # Should return zeros for insufficient data
        assert entropy == 0.0
        assert free_energy == 0.0
        assert rmsd == 0.0
        
        # Empty list
        entropy2, free_energy2, rmsd2 = calc.calculate_configurational_entropy([])
        assert entropy2 == 0.0
        assert free_energy2 == 0.0
        assert rmsd2 == 0.0


class TestTemperatureDependence:
    """Test temperature dependence of entropic corrections."""
    
    def _create_conformation(self, conf_id: str, coords: list):
        """Helper to create test conformation."""
        sequence = "A" * len(coords)
        return Conformation(
            conformation_id=conf_id,
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
    
    def test_free_energy_scales_with_temperature(self):
        """Test that free energy ΔG = -T*S scales linearly with T."""
        qcp_values = [3.0, 5.0, 4.0, 6.0]
        
        calc_300 = EntropicCalculator(temperature=300.0)
        calc_600 = EntropicCalculator(temperature=600.0)
        
        entropy_300, free_energy_300 = calc_300.calculate_coherence_entropy(qcp_values)
        entropy_600, free_energy_600 = calc_600.calculate_coherence_entropy(qcp_values)
        
        # Entropy should be the same (temperature doesn't affect S)
        assert abs(entropy_300 - entropy_600) < 1e-9
        
        # Free energy should double
        assert abs(free_energy_600 / free_energy_300 - 2.0) < 1e-6
    
    def test_higher_temperature_larger_entropic_contribution(self):
        """Test that higher temperature increases entropic effects."""
        qcp_values = [3.0, 5.0, 4.0, 6.0]
        conformations = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 1, 0)])
        ]
        
        calc_low = EntropicCalculator(temperature=100.0)
        calc_high = EntropicCalculator(temperature=500.0)
        
        contrib_low = calc_low.calculate_entropic_contributions(qcp_values, conformations)
        contrib_high = calc_high.calculate_entropic_contributions(qcp_values, conformations)
        
        # Higher temperature = larger magnitude (more negative) free energy
        assert abs(contrib_high.total_entropic_energy) > abs(contrib_low.total_entropic_energy)
    
    def test_zero_entropy_zero_temperature_effect(self):
        """Test that zero entropy gives zero temperature effect."""
        qcp_values = [4.0, 4.0, 4.0]  # Zero variance
        
        calc = EntropicCalculator(temperature=300.0)
        entropy, free_energy = calc.calculate_coherence_entropy(qcp_values)
        
        assert entropy == 0.0
        assert free_energy == 0.0


class TestEntropicContributions:
    """Test combined entropic contributions calculation."""
    
    def _create_conformation(self, conf_id: str, coords: list):
        """Helper to create test conformation."""
        sequence = "A" * len(coords)
        return Conformation(
            conformation_id=conf_id,
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
    
    def test_contributions_structure(self):
        """Test that EntropicContributions has all required fields."""
        calc = EntropicCalculator()
        
        qcp_values = [4.0, 4.2, 3.8]
        conformations = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 0.5, 0)])
        ]
        
        contrib = calc.calculate_entropic_contributions(qcp_values, conformations)
        
        assert hasattr(contrib, 'qcp_variance')
        assert hasattr(contrib, 'coherence_entropy')
        assert hasattr(contrib, 'avg_pairwise_rmsd')
        assert hasattr(contrib, 'configurational_entropy')
        assert hasattr(contrib, 'total_entropic_energy')
    
    def test_total_energy_is_sum_of_terms(self):
        """Test that total energy equals -T*(S_coh + S_conf)."""
        calc = EntropicCalculator(temperature=300.0)
        
        qcp_values = [3.5, 4.5, 4.0, 4.2]
        conformations = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 1, 0)]),
            self._create_conformation("c3", [(0, 0, 0), (3.8, -0.5, 0)])
        ]
        
        contrib = calc.calculate_entropic_contributions(qcp_values, conformations)
        
        expected_total = -300.0 * (contrib.coherence_entropy + contrib.configurational_entropy)
        
        assert abs(contrib.total_entropic_energy - expected_total) < 1e-6
    
    def test_contributions_negative_free_energy(self):
        """Test that entropic contributions give negative (favorable) free energy."""
        calc = EntropicCalculator()
        
        qcp_values = [3.0, 5.0, 4.0, 6.0]
        conformations = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0), (7.6, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 1, 0), (7.6, 2, 0)]),
            self._create_conformation("c3", [(0, 0, 0), (3.8, -1, 0), (7.6, -2, 0)])
        ]
        
        contrib = calc.calculate_entropic_contributions(qcp_values, conformations)
        
        # Both entropy terms should be non-negative
        assert contrib.coherence_entropy >= 0.0
        assert contrib.configurational_entropy >= 0.0
        
        # Free energy should be non-positive (ΔG = -T*S)
        assert contrib.total_entropic_energy <= 0.0
    
    def test_zero_entropy_zero_contribution(self):
        """Test that zero entropy gives zero energetic contribution."""
        calc = EntropicCalculator()
        
        # Zero variance QCP
        qcp_values = [4.0, 4.0, 4.0]
        
        # Identical conformations
        coords = [(0, 0, 0), (3.8, 0, 0)]
        conformations = [
            self._create_conformation(f"c{i}", coords)
            for i in range(3)
        ]
        
        contrib = calc.calculate_entropic_contributions(qcp_values, conformations)
        
        assert contrib.coherence_entropy == 0.0
        assert contrib.configurational_entropy == 0.0
        assert contrib.total_entropic_energy == 0.0
    
    def test_contributions_realistic_values(self):
        """Test with realistic QCP and conformational data."""
        calc = EntropicCalculator(temperature=300.0)
        
        # Realistic QCP values with moderate variance
        qcp_values = [4.0, 4.2, 3.8, 4.5, 4.1, 3.9, 4.3]
        
        # Diverse conformations with realistic coordinates
        conformations = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0), (7.6, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 0.5, 0), (7.6, 1, 0)]),
            self._create_conformation("c3", [(0, 0, 0), (3.8, 1, 0), (7.6, 2, 0)])
        ]
        
        contrib = calc.calculate_entropic_contributions(qcp_values, conformations)
        
        # Check ranges are physically reasonable
        assert 0.0 < contrib.qcp_variance < 1.0
        assert 0.0 < contrib.coherence_entropy < 0.01
        assert 0.0 < contrib.avg_pairwise_rmsd < 5.0
        assert 0.0 < contrib.configurational_entropy < 0.01
        assert -5.0 < contrib.total_entropic_energy < 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def _create_conformation(self, conf_id: str, coords: list):
        """Helper to create test conformation."""
        sequence = "A" * len(coords)
        return Conformation(
            conformation_id=conf_id,
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
    
    def test_very_large_variance(self):
        """Test with variance exceeding max_variance."""
        calc = EntropicCalculator(max_variance=5.0)
        
        qcp_values = [0.0, 10.0, 20.0, 30.0, 40.0]
        
        variance = calc.calculate_qcp_variance(qcp_values)
        assert variance == 5.0  # Clamped
        
        entropy, free_energy = calc.calculate_coherence_entropy(qcp_values)
        # S = k_B * ln(1 + 5.0)
        expected = 0.001987 * math.log(6.0)
        assert abs(entropy - expected) < 1e-6
        assert free_energy < 0.0
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        calc = EntropicCalculator()
        
        conformations = [
            self._create_conformation("c1", [(-5, -3, -2), (-1, 0, 1)]),
            self._create_conformation("c2", [(-5, -3, -2), (-1, 0.5, 1)])
        ]
        
        entropy, free_energy, rmsd = calc.calculate_configurational_entropy(conformations)
        
        # Should work fine with negative coords
        assert entropy > 0.0
        assert free_energy < 0.0
    
    def test_very_small_rmsd(self):
        """Test with extremely small RMSD differences."""
        calc = EntropicCalculator()
        
        conformations = [
            self._create_conformation("c1", [(0, 0, 0), (3.8, 0, 0)]),
            self._create_conformation("c2", [(0, 0, 0), (3.8, 0.001, 0)])
        ]
        
        entropy, free_energy, rmsd = calc.calculate_configurational_entropy(conformations)
        
        # Should be very small but non-zero
        assert 0.0 < entropy < 0.0001
        assert free_energy < 0.0
    
    def test_mixed_amino_acid_conformations(self):
        """Test with conformations of different sizes."""
        calc = EntropicCalculator()
        
        # Different sized conformations
        conf1 = Conformation(
            "c1", "AAA",
            [(0, 0, 0), (3.8, 0, 0), (7.6, 0, 0)],
            -100, None, ['C']*3, [0]*3, [0]*3, [], {}
        )
        conf2 = Conformation(
            "c2", "AA",
            [(0, 0, 0), (3.8, 0, 0)],
            -95, None, ['C']*2, [0]*2, [0]*2, [], {}
        )
        
        # Should raise error due to incompatible sizes
        with pytest.raises(ValueError):
            calc.calculate_configurational_entropy([conf1, conf2])
    
    def test_extreme_temperature(self):
        """Test with extreme temperature values."""
        # Very low temperature
        calc_low = EntropicCalculator(temperature=10.0)
        qcp_values = [3.5, 4.5]
        
        entropy, free_energy_low = calc_low.calculate_coherence_entropy(qcp_values)
        
        assert abs(free_energy_low) < 0.1  # Small due to low T
        
        # Very high temperature
        calc_high = EntropicCalculator(temperature=1000.0)
        _, free_energy_high = calc_high.calculate_coherence_entropy(qcp_values)
        
        assert abs(free_energy_high) > abs(free_energy_low)


class TestNumericalStability:
    """Test numerical stability and precision."""
    
    def test_variance_numerical_precision(self):
        """Test variance calculation with very similar values."""
        calc = EntropicCalculator()
        
        # Values differ only in last decimal place
        qcp_values = [4.000001, 4.000002, 4.000003, 4.000004]
        
        variance = calc.calculate_qcp_variance(qcp_values)
        
        # Should be extremely small but positive
        assert 0.0 < variance < 1e-10
    
    def test_entropy_with_unit_variance(self):
        """Test entropy with variance = 1.0."""
        calc = EntropicCalculator()
        
        # Create QCP values with exactly variance = 1.0
        qcp_values = [3.0, 5.0]  # Mean=4, var=1
        
        entropy, free_energy = calc.calculate_coherence_entropy(qcp_values)
        
        # S = k_B * ln(2)
        expected = 0.001987 * math.log(2.0)
        assert abs(entropy - expected) < 1e-9
    
    def test_rmsd_calculation_precision(self):
        """Test RMSD calculation with high precision coordinates."""
        calc = EntropicCalculator()
        
        conformations = [
            Conformation(
                "c1", "AA",
                [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)],
                -100, None, ['C']*2, [0]*2, [0]*2, [], {}
            ),
            Conformation(
                "c2", "AA",
                [(0.0, 0.0, 0.0), (3.800001, 0.0, 0.0)],
                -100, None, ['C']*2, [0]*2, [0]*2, [], {}
            )
        ]
        
        entropy, free_energy, rmsd = calc.calculate_configurational_entropy(conformations)
        
        # Should handle tiny differences
        assert entropy >= 0.0
        assert free_energy <= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
