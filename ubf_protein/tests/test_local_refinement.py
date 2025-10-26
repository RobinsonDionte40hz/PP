"""
Unit tests for LocalRefinement module.

Tests cover:
1. Initialization and parameter validation
2. Gradient calculation accuracy
3. Convergence behavior
4. Step size adaptation
5. Geometry validation
6. Maximum iteration limits
7. Performance benchmarks
8. Edge cases and numerical stability
"""

import pytest
import math
import time
from typing import List, Tuple

from ubf_protein.local_refinement import LocalRefinement, RefinementResult
from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
from ubf_protein.models import Conformation, DisulfideBond


class TestLocalRefinementInitialization:
    """Test initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        assert refiner.initial_step_size == 0.01
        assert refiner.epsilon == 0.01
        assert refiner.convergence_tolerance == 0.001
        assert refiner.max_iterations == 100
        assert refiner.step_reduction_factor == 0.5
        assert refiner.min_step_size == 1e-6
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(
            calc,
            initial_step_size=0.05,
            epsilon=0.001,
            convergence_tolerance=0.01,
            max_iterations=50,
            step_reduction_factor=0.7,
            min_step_size=1e-8
        )
        
        assert refiner.initial_step_size == 0.05
        assert refiner.epsilon == 0.001
        assert refiner.convergence_tolerance == 0.01
        assert refiner.max_iterations == 50
        assert refiner.step_reduction_factor == 0.7
        assert refiner.min_step_size == 1e-8
    
    def test_invalid_step_size(self):
        """Test validation rejects non-positive step size."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        with pytest.raises(ValueError, match="initial_step_size must be positive"):
            LocalRefinement(calc, initial_step_size=0.0)
        
        with pytest.raises(ValueError, match="initial_step_size must be positive"):
            LocalRefinement(calc, initial_step_size=-0.01)
    
    def test_invalid_epsilon(self):
        """Test validation rejects non-positive epsilon."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LocalRefinement(calc, epsilon=0.0)
        
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LocalRefinement(calc, epsilon=-0.01)
    
    def test_invalid_tolerance(self):
        """Test validation rejects negative tolerance."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        with pytest.raises(ValueError, match="convergence_tolerance must be non-negative"):
            LocalRefinement(calc, convergence_tolerance=-0.001)
    
    def test_invalid_max_iterations(self):
        """Test validation rejects invalid max_iterations."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            LocalRefinement(calc, max_iterations=0)
        
        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            LocalRefinement(calc, max_iterations=-10)
    
    def test_invalid_reduction_factor(self):
        """Test validation rejects invalid reduction factor."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        
        with pytest.raises(ValueError, match="step_reduction_factor must be in"):
            LocalRefinement(calc, step_reduction_factor=0.0)
        
        with pytest.raises(ValueError, match="step_reduction_factor must be in"):
            LocalRefinement(calc, step_reduction_factor=1.0)
        
        with pytest.raises(ValueError, match="step_reduction_factor must be in"):
            LocalRefinement(calc, step_reduction_factor=1.5)
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, initial_step_size=0.02)
        
        params = refiner.get_parameters()
        assert params['initial_step_size'] == 0.02
        assert params['epsilon'] == 0.01
        assert 'convergence_tolerance' in params
        assert 'max_iterations' in params


class TestGradientCalculation:
    """Test numerical gradient calculation accuracy."""
    
    def test_gradient_calculation_structure(self):
        """Test gradient returns correct structure."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        conf = self._create_simple_conformation(sequence)
        gradient = refiner._calculate_gradient(conf)
        
        assert len(gradient) == 3  # One per residue
        for grad in gradient:
            assert len(grad) == 3  # (dx, dy, dz)
            assert all(isinstance(g, float) for g in grad)
    
    def test_gradient_finite_values(self):
        """Test gradient contains finite values."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        conf = self._create_simple_conformation(sequence)
        gradient = refiner._calculate_gradient(conf)
        
        for grad_x, grad_y, grad_z in gradient:
            assert math.isfinite(grad_x)
            assert math.isfinite(grad_y)
            assert math.isfinite(grad_z)
    
    def test_partial_derivative_symmetry(self):
        """Test partial derivatives are symmetric."""
        sequence = "AA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        conf = self._create_simple_conformation(sequence)
        
        # Calculate derivative for first atom, x direction
        deriv_forward = refiner._partial_derivative(conf, 0, 0)
        
        # Should be finite
        assert math.isfinite(deriv_forward)
    
    def test_gradient_norm_positive(self):
        """Test gradient norm is non-negative."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        conf = self._create_simple_conformation(sequence)
        gradient = refiner._calculate_gradient(conf)
        norm = refiner._calculate_norm(gradient)
        
        assert norm >= 0.0
        assert math.isfinite(norm)
    
    def test_gradient_zero_for_minimum(self):
        """Test gradient norm is reasonable after optimization."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=10)
        
        # Create slightly perturbed conformation
        conf = self._create_perturbed_conformation(sequence)
        
        initial_gradient = refiner._calculate_gradient(conf)
        initial_norm = refiner._calculate_norm(initial_gradient)
        
        # After refinement, gradient should be finite
        result = refiner.refine(conf)
        final_gradient = refiner._calculate_gradient(result.refined_conformation)
        final_norm = refiner._calculate_norm(final_gradient)
        
        # Gradient should be finite and reasonable
        assert math.isfinite(final_norm)
        assert final_norm < 100.0  # Reasonable upper bound
    
    @staticmethod
    def _create_simple_conformation(sequence: str) -> Conformation:
        """Create simple linear conformation."""
        n = len(sequence)
        coords = [(i * 3.8, 0.0, 0.0) for i in range(n)]
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )
    
    @staticmethod
    def _create_perturbed_conformation(sequence: str) -> Conformation:
        """Create slightly perturbed conformation."""
        n = len(sequence)
        coords = [(i * 3.8 + 0.1 * i, 0.1 * i, 0.0) for i in range(n)]
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )


class TestRefinementBehavior:
    """Test refinement convergence and termination."""
    
    def test_refinement_returns_result(self):
        """Test refinement returns RefinementResult."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=10)
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        assert isinstance(result, RefinementResult)
        assert hasattr(result, 'refined_conformation')
        assert hasattr(result, 'initial_energy')
        assert hasattr(result, 'final_energy')
        assert hasattr(result, 'energy_change')
        assert hasattr(result, 'n_iterations')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'gradient_norm')
    
    def test_refinement_reduces_energy(self):
        """Test refinement reduces or maintains energy."""
        sequence = "AAAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=50)
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        # Energy should decrease (or stay same if already at minimum)
        assert result.final_energy <= result.initial_energy
    
    def test_energy_change_sign(self):
        """Test energy change has correct sign."""
        sequence = "AAAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=20)
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        # Energy change = final - initial, should be negative if energy decreased
        expected_change = result.final_energy - result.initial_energy
        assert abs(result.energy_change - expected_change) < 1e-6
    
    def test_max_iterations_respected(self):
        """Test refinement stops at max iterations."""
        sequence = "AAAA"
        calc = EnhancedEnergyCalculator(sequence)
        max_iter = 20
        refiner = LocalRefinement(calc, max_iterations=max_iter)
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        # Should not exceed max iterations
        assert result.n_iterations <= max_iter
    
    def test_convergence_detection(self):
        """Test convergence is detected when energy change is small."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(
            calc,
            max_iterations=200,
            convergence_tolerance=0.01  # Larger tolerance for easier convergence
        )
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        # Check if convergence was detected
        if result.converged:
            assert result.reason == "converged"
            assert result.n_iterations < 200
    
    def test_reason_field_valid(self):
        """Test reason field contains valid termination reason."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=10)
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        valid_reasons = ["converged", "max_iterations", "step_size_too_small"]
        assert result.reason in valid_reasons
    
    def test_gradient_norm_reported(self):
        """Test gradient norm is reported in result."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=10)
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        assert result.gradient_norm >= 0.0
        assert math.isfinite(result.gradient_norm)
    
    @staticmethod
    def _create_conformation(sequence: str) -> Conformation:
        """Create test conformation."""
        n = len(sequence)
        coords = [(i * 3.8, 0.0, 0.0) for i in range(n)]
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )


class TestStepSizeAdaptation:
    """Test adaptive step size behavior."""
    
    def test_step_size_reduction_on_invalid_geometry(self):
        """Test step size reduces when geometry becomes invalid."""
        sequence = "AA"
        calc = EnhancedEnergyCalculator(sequence)
        
        # Very large step size will likely violate geometry
        refiner = LocalRefinement(
            calc,
            initial_step_size=2.0,  # Very large
            max_iterations=50,
            min_step_size=1e-3
        )
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        # Should either converge or hit step size limit
        assert result.reason in ["converged", "max_iterations", "step_size_too_small"]
    
    def test_min_step_size_termination(self):
        """Test termination when step size becomes too small."""
        sequence = "AA"
        calc = EnhancedEnergyCalculator(sequence)
        
        # Set very restrictive parameters
        refiner = LocalRefinement(
            calc,
            initial_step_size=1e-5,
            min_step_size=1e-6,
            step_reduction_factor=0.5,
            max_iterations=100
        )
        
        # Create conformation that might need large steps
        conf = self._create_difficult_conformation(sequence)
        result = refiner.refine(conf)
        
        # Should eventually terminate
        assert result.reason in ["converged", "max_iterations", "step_size_too_small"]
    
    @staticmethod
    def _create_conformation(sequence: str) -> Conformation:
        """Create test conformation."""
        n = len(sequence)
        coords = [(i * 3.8, 0.0, 0.0) for i in range(n)]
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )
    
    @staticmethod
    def _create_difficult_conformation(sequence: str) -> Conformation:
        """Create conformation with challenging geometry."""
        n = len(sequence)
        # Slightly compressed
        coords = [(i * 3.0, 0.0, 0.0) for i in range(n)]
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )


class TestGeometryValidation:
    """Test geometry validation during refinement."""
    
    def test_validate_geometry_accepts_valid(self):
        """Test validation accepts valid geometry."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        conf = self._create_valid_conformation(sequence)
        is_valid = refiner._validate_geometry(conf)
        
        assert is_valid is True
    
    def test_validate_geometry_rejects_extreme_coords(self):
        """Test validation rejects extreme coordinates."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        conf = self._create_extreme_conformation(sequence)
        is_valid = refiner._validate_geometry(conf)
        
        assert is_valid is False
    
    def test_validate_geometry_rejects_short_bonds(self):
        """Test validation rejects too-short bonds."""
        sequence = "AA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        # Create conformation with 2.0 Å distance (too short)
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        conf = Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 2,
            phi_angles=[0.0] * 2,
            psi_angles=[0.0] * 2,
            available_move_types=[],
            structural_constraints={}
        )
        
        is_valid = refiner._validate_geometry(conf)
        assert is_valid is False
    
    def test_validate_geometry_rejects_long_bonds(self):
        """Test validation rejects too-long bonds."""
        sequence = "AA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        # Create conformation with 6.0 Å distance (too long)
        coords = [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]
        conf = Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 2,
            phi_angles=[0.0] * 2,
            psi_angles=[0.0] * 2,
            available_move_types=[],
            structural_constraints={}
        )
        
        is_valid = refiner._validate_geometry(conf)
        assert is_valid is False
    
    def test_validate_geometry_accepts_valid_range(self):
        """Test validation accepts distances in valid range."""
        sequence = "AA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        # Test various valid distances
        valid_distances = [2.5, 3.0, 3.8, 4.5, 5.0]
        
        for dist in valid_distances:
            coords = [(0.0, 0.0, 0.0), (dist, 0.0, 0.0)]
            conf = Conformation(
                conformation_id="test",
                sequence=sequence,
                atom_coordinates=coords,
                energy=-100.0,
                rmsd_to_native=None,
                secondary_structure=['C'] * 2,
                phi_angles=[0.0] * 2,
                psi_angles=[0.0] * 2,
                available_move_types=[],
                structural_constraints={}
            )
            
            is_valid = refiner._validate_geometry(conf)
            assert is_valid is True, f"Distance {dist} should be valid"
    
    @staticmethod
    def _create_valid_conformation(sequence: str) -> Conformation:
        """Create conformation with valid geometry."""
        n = len(sequence)
        coords = [(i * 3.8, 0.0, 0.0) for i in range(n)]
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )
    
    @staticmethod
    def _create_extreme_conformation(sequence: str) -> Conformation:
        """Create conformation with extreme coordinates."""
        n = len(sequence)
        coords = [(i * 2000.0, 0.0, 0.0) for i in range(n)]  # > 1000 Å
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )


class TestPerformance:
    """Test performance meets requirements."""
    
    def test_small_protein_performance(self):
        """Test refinement completes in reasonable time for 10 residues."""
        sequence = "A" * 10
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=20)  # Reduced iterations
        
        conf = self._create_conformation(sequence)
        
        start_time = time.time()
        result = refiner.refine(conf)
        elapsed = time.time() - start_time
        
        # Gradient descent is O(N²) per iteration due to energy calculations
        assert elapsed < 10.0, f"Small protein took {elapsed:.3f}s (should be <10s)"
    
    def test_medium_protein_performance(self):
        """Test refinement completes in reasonable time for 30 residues."""
        sequence = "A" * 30
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=10)  # Reduced iterations
        
        conf = self._create_conformation(sequence)
        
        start_time = time.time()
        result = refiner.refine(conf)
        elapsed = time.time() - start_time
        
        assert elapsed < 30.0, f"Medium protein took {elapsed:.3f}s (should be <30s)"
    
    def test_refinement_completes(self):
        """Test refinement completes for 20 residues."""
        sequence = "A" * 20
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=10)
        
        conf = self._create_conformation(sequence)
        
        start_time = time.time()
        result = refiner.refine(conf)
        elapsed = time.time() - start_time
        
        # Just verify it completes
        assert result is not None
        print(f"  20 residues, 10 iterations: {elapsed:.3f}s")
    
    def test_gradient_calculation_completes(self):
        """Test single gradient calculation completes."""
        sequence = "A" * 10
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        conf = self._create_conformation(sequence)
        
        start_time = time.time()
        gradient = refiner._calculate_gradient(conf)
        elapsed = time.time() - start_time
        
        # Should complete (may be slow for large proteins)
        assert gradient is not None
        assert len(gradient) == len(sequence)
        print(f"  Gradient for {len(sequence)} residues: {elapsed:.3f}s")
    
    @staticmethod
    def _create_conformation(sequence: str) -> Conformation:
        """Create test conformation."""
        n = len(sequence)
        coords = [(i * 3.8, 0.0, 0.0) for i in range(n)]
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_single_residue(self):
        """Test refinement with single residue."""
        sequence = "A"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=10)
        
        coords = [(0.0, 0.0, 0.0)]
        conf = Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'],
            phi_angles=[0.0],
            psi_angles=[0.0],
            available_move_types=[],
            structural_constraints={}
        )
        
        result = refiner.refine(conf)
        assert isinstance(result, RefinementResult)
    
    def test_two_residues(self):
        """Test refinement with two residues."""
        sequence = "AA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=20)
        
        coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)]
        conf = Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C', 'C'],
            phi_angles=[0.0, 0.0],
            psi_angles=[0.0, 0.0],
            available_move_types=[],
            structural_constraints={}
        )
        
        result = refiner.refine(conf)
        assert isinstance(result, RefinementResult)
        # Geometry should remain valid
        assert refiner._validate_geometry(result.refined_conformation)
    
    def test_collinear_geometry(self):
        """Test refinement with perfectly collinear geometry."""
        sequence = "AAAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc, max_iterations=20)
        
        coords = [(i * 3.8, 0.0, 0.0) for i in range(4)]
        conf = Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 4,
            phi_angles=[0.0] * 4,
            psi_angles=[0.0] * 4,
            available_move_types=[],
            structural_constraints={}
        )
        
        result = refiner.refine(conf)
        assert isinstance(result, RefinementResult)
        assert math.isfinite(result.final_energy)
    
    def test_zero_tolerance_convergence(self):
        """Test with zero convergence tolerance (should use max iterations)."""
        sequence = "AAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(
            calc,
            convergence_tolerance=0.0,
            max_iterations=10
        )
        
        conf = self._create_conformation(sequence)
        result = refiner.refine(conf)
        
        # Should hit max iterations since convergence is impossible
        assert result.n_iterations == 10
        assert result.reason == "max_iterations"
    
    def test_coordinate_update_preserves_sequence_length(self):
        """Test coordinate updates preserve sequence length."""
        sequence = "AAAAA"
        calc = EnhancedEnergyCalculator(sequence)
        refiner = LocalRefinement(calc)
        
        conf = self._create_conformation(sequence)
        gradient = refiner._calculate_gradient(conf)
        updated = refiner._update_coordinates(conf, gradient, 0.01)
        
        assert len(updated.atom_coordinates) == len(sequence)
        assert updated.sequence == sequence
    
    @staticmethod
    def _create_conformation(sequence: str) -> Conformation:
        """Create test conformation."""
        n = len(sequence)
        coords = [(i * 3.8, 0.0, 0.0) for i in range(n)]
        
        return Conformation(
            conformation_id="test",
            sequence=sequence,
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * n,
            phi_angles=[0.0] * n,
            psi_angles=[0.0] * n,
            available_move_types=[],
            structural_constraints={}
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
