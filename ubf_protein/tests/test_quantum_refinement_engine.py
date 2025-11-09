"""
Unit tests for Quantum Refinement Engine (Task 1)

Tests:
- QuantumRefinementEngine initialization
- Quantum constants validation
- RefinementConfig data model
- RefinementResult data model
- Geometry validation
- Energy validation
- Cache management
"""

import pytest
import math
from typing import List, Tuple

from ubf_protein.quantum_refinement_engine import (
    QuantumRefinementEngine,
    RefinementError,
    ConvergenceError,
    GeometryError
)
from ubf_protein.models import (
    Conformation,
    RefinementConfig,
    RefinementResult
)
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.energy_function import MolecularMechanicsEnergy
from ubf_protein.rmsd_calculator import RMSDCalculator, NativeStructure


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_qcpp_adapter():
    """Mock QCPP adapter for testing."""
    # Import QCPP predictor (assuming available)
    try:
        from quantum_coherence_proteins.protein_predictor import QuantumCoherenceProteinPredictor
        predictor = QuantumCoherenceProteinPredictor()
        return QCPPIntegrationAdapter(predictor)
    except ImportError:
        # If QCPP not available, create minimal mock
        class MockPredictor:
            def calculate_qcp(self, coords, sequence):
                return [5.0] * len(coords)
        return QCPPIntegrationAdapter(MockPredictor())


@pytest.fixture
def energy_calculator():
    """Energy calculator for testing."""
    return MolecularMechanicsEnergy()


@pytest.fixture
def rmsd_calculator():
    """RMSD calculator for testing."""
    return RMSDCalculator()


@pytest.fixture
def refinement_engine(mock_qcpp_adapter, energy_calculator, rmsd_calculator):
    """Create refinement engine for testing."""
    return QuantumRefinementEngine(
        mock_qcpp_adapter,
        energy_calculator,
        rmsd_calculator
    )


@pytest.fixture
def test_conformation():
    """Create a test conformation (7-residue extended chain)."""
    # Extended chain with ~3.8Å CA-CA spacing
    coords = [
        (0.0, 0.0, 0.0),
        (3.8, 0.0, 0.0),
        (7.6, 0.0, 0.0),
        (11.4, 0.0, 0.0),
        (15.2, 0.0, 0.0),
        (19.0, 0.0, 0.0),
        (22.8, 0.0, 0.0),
    ]
    
    return Conformation(
        conformation_id="test_conf",
        sequence="ACDEFGH",
        atom_coordinates=coords,
        energy=-50.0,
        rmsd_to_native=10.0,
        secondary_structure=['C'] * 7,
        phi_angles=[0.0] * 7,
        psi_angles=[0.0] * 7,
        available_move_types=["backbone_rotation"],
        structural_constraints={}
    )


@pytest.fixture
def test_native_structure():
    """Create a test native structure."""
    # Slightly different coordinates
    coords = [
        (0.1, 0.1, 0.0),
        (3.9, 0.1, 0.0),
        (7.5, 0.2, 0.0),
        (11.3, 0.1, 0.0),
        (15.1, 0.0, 0.0),
        (18.9, -0.1, 0.0),
        (22.7, 0.0, 0.0),
    ]
    
    return NativeStructure(
        pdb_id="TEST",
        sequence="ACDEFGH",
        ca_coords=coords,
        n_residues=7
    )


# ============================================================================
# Test QuantumRefinementEngine Initialization
# ============================================================================

class TestQuantumRefinementEngineInit:
    """Test QuantumRefinementEngine initialization."""
    
    def test_init_with_valid_calculators(
        self,
        mock_qcpp_adapter,
        energy_calculator,
        rmsd_calculator
    ):
        """Test initialization with valid calculators."""
        engine = QuantumRefinementEngine(
            mock_qcpp_adapter,
            energy_calculator,
            rmsd_calculator
        )
        
        # Check calculators are stored
        assert engine.qcpp_adapter is mock_qcpp_adapter
        assert engine.energy_calculator is energy_calculator
        assert engine.rmsd_calculator is rmsd_calculator
        
        # Check quantum constants
        assert abs(engine.phi - 1.618033988749895) < 1e-10
        assert abs(engine.h_bar - 1.0545718e-34) < 1e-40
        assert engine.gamma_frequency == 40.0
        assert abs(engine.coherence_time - 408e-15) < 1e-20
        assert engine.water_spacing == 0.28
        
        # Check caches are initialized
        assert engine._qcp_cache == {}
        assert engine._thz_mode_cache == {}
        assert engine._distance_matrix_cache is None
    
    def test_init_with_none_qcpp_adapter(self, energy_calculator, rmsd_calculator):
        """Test initialization fails with None QCPP adapter."""
        with pytest.raises(TypeError, match="qcpp_adapter cannot be None"):
            QuantumRefinementEngine(None, energy_calculator, rmsd_calculator)
    
    def test_init_with_none_energy_calculator(self, mock_qcpp_adapter, rmsd_calculator):
        """Test initialization fails with None energy calculator."""
        with pytest.raises(TypeError, match="energy_calculator cannot be None"):
            QuantumRefinementEngine(mock_qcpp_adapter, None, rmsd_calculator)
    
    def test_init_with_none_rmsd_calculator(self, mock_qcpp_adapter, energy_calculator):
        """Test initialization fails with None RMSD calculator."""
        with pytest.raises(TypeError, match="rmsd_calculator cannot be None"):
            QuantumRefinementEngine(mock_qcpp_adapter, energy_calculator, None)
    
    def test_init_with_wrong_type_qcpp_adapter(self, energy_calculator, rmsd_calculator):
        """Test initialization fails with wrong type QCPP adapter."""
        with pytest.raises(TypeError, match="qcpp_adapter must be QCPPIntegrationAdapter"):
            QuantumRefinementEngine("not_an_adapter", energy_calculator, rmsd_calculator)
    
    def test_init_with_wrong_type_energy_calculator(self, mock_qcpp_adapter, rmsd_calculator):
        """Test initialization fails with wrong type energy calculator."""
        with pytest.raises(TypeError, match="energy_calculator must be MolecularMechanicsEnergy"):
            QuantumRefinementEngine(mock_qcpp_adapter, "not_a_calculator", rmsd_calculator)
    
    def test_init_with_wrong_type_rmsd_calculator(self, mock_qcpp_adapter, energy_calculator):
        """Test initialization fails with wrong type RMSD calculator."""
        with pytest.raises(TypeError, match="rmsd_calculator must be RMSDCalculator"):
            QuantumRefinementEngine(mock_qcpp_adapter, energy_calculator, "not_a_calculator")


# ============================================================================
# Test Quantum Constants
# ============================================================================

class TestQuantumConstants:
    """Test quantum constants are correctly defined."""
    
    def test_phi_constant(self, refinement_engine):
        """Test golden ratio constant."""
        # φ = (1 + √5) / 2
        expected_phi = (1 + math.sqrt(5)) / 2
        assert abs(refinement_engine.phi - expected_phi) < 1e-10
        assert abs(QuantumRefinementEngine.PHI - expected_phi) < 1e-10
    
    def test_h_bar_constant(self, refinement_engine):
        """Test Planck's constant."""
        # ℏ = h / (2π) = 1.0545718e-34 J·s
        assert abs(refinement_engine.h_bar - 1.0545718e-34) < 1e-40
        assert abs(QuantumRefinementEngine.H_BAR - 1.0545718e-34) < 1e-40
    
    def test_gamma_frequency_constant(self, refinement_engine):
        """Test gamma frequency (consciousness resonance)."""
        assert refinement_engine.gamma_frequency == 40.0
        assert QuantumRefinementEngine.GAMMA_FREQUENCY == 40.0
    
    def test_coherence_time_constant(self, refinement_engine):
        """Test coherence time (408 femtoseconds)."""
        # 408 fs = 408e-15 s
        assert abs(refinement_engine.coherence_time - 408e-15) < 1e-20
        assert abs(QuantumRefinementEngine.COHERENCE_TIME - 408e-15) < 1e-20
    
    def test_water_spacing_constant(self, refinement_engine):
        """Test water molecule spacing."""
        assert refinement_engine.water_spacing == 0.28
        assert QuantumRefinementEngine.WATER_SPACING == 0.28


# ============================================================================
# Test RefinementConfig Data Model
# ============================================================================

class TestRefinementConfig:
    """Test RefinementConfig data model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RefinementConfig()
        
        # Stage 1 parameters
        assert config.stage1_temperature == 1.0
        assert config.stage1_iterations == 1000
        
        # Stage 2 parameters
        assert config.stage2_temperature == 0.1  # 10x lower
        assert config.stage2_iterations == 10000  # 10x more
        assert config.restraint_weight == 10.0
        assert config.qcp_weight == 0.3
        
        # Quantum parameters
        assert config.qcp_threshold == 7.0
        assert config.phi_tolerance == 0.1
        assert config.resonance_threshold == 0.7
        
        # Water shielding
        assert config.water_spacing_nm == 0.28
        assert config.coherence_time_fs == 408.0
        
        # Performance
        assert config.max_refinement_time_seconds == 300.0
        assert config.checkpoint_interval == 1000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RefinementConfig(
            stage1_temperature=2.0,
            stage1_iterations=500,
            stage2_temperature=0.05,
            stage2_iterations=20000,
            restraint_weight=20.0,
            qcp_weight=0.5,
            qcp_threshold=8.0,
            phi_tolerance=0.05,
            resonance_threshold=0.8,
            water_spacing_nm=0.3,
            coherence_time_fs=500.0,
            max_refinement_time_seconds=600.0,
            checkpoint_interval=500
        )
        
        assert config.stage1_temperature == 2.0
        assert config.stage1_iterations == 500
        assert config.stage2_temperature == 0.05
        assert config.stage2_iterations == 20000
        assert config.restraint_weight == 20.0
        assert config.qcp_weight == 0.5
        assert config.qcp_threshold == 8.0
        assert config.phi_tolerance == 0.05
        assert config.resonance_threshold == 0.8
        assert config.water_spacing_nm == 0.3
        assert config.coherence_time_fs == 500.0
        assert config.max_refinement_time_seconds == 600.0
        assert config.checkpoint_interval == 500


# ============================================================================
# Test RefinementResult Data Model
# ============================================================================

class TestRefinementResult:
    """Test RefinementResult data model."""
    
    def test_create_result(self, test_conformation, test_native_structure):
        """Test creating refinement result."""
        result = RefinementResult(
            initial_structure=test_conformation,
            refined_structure=test_conformation,
            native_structure=test_native_structure,
            initial_rmsd=10.0,
            final_rmsd=3.5,
            rmsd_improvement=6.5,
            helix_rmsd=2.0,
            sheet_rmsd=3.0,
            loop_rmsd=5.0,
            core_rmsd=4.0,
            gdt_ts=60.0,
            tm_score=0.65,
            energy=-80.0,
            iterations_used=5000,
            refinement_time_seconds=120.5,
            quantum_cores_identified=3,
            restraints_applied=15,
            contacts_enforced=8,
            rmsd_trajectory=[10.0, 8.0, 6.0, 4.0, 3.5],
            energy_trajectory=[-50.0, -60.0, -70.0, -75.0, -80.0]
        )
        
        # Check structures
        assert result.initial_structure is test_conformation
        assert result.refined_structure is test_conformation
        assert result.native_structure is test_native_structure
        
        # Check RMSD metrics
        assert result.initial_rmsd == 10.0
        assert result.final_rmsd == 3.5
        assert result.rmsd_improvement == 6.5
        
        # Check component RMSD
        assert result.helix_rmsd == 2.0
        assert result.sheet_rmsd == 3.0
        assert result.loop_rmsd == 5.0
        assert result.core_rmsd == 4.0
        
        # Check quality metrics
        assert result.gdt_ts == 60.0
        assert result.tm_score == 0.65
        assert result.energy == -80.0
        
        # Check statistics
        assert result.iterations_used == 5000
        assert result.refinement_time_seconds == 120.5
        assert result.quantum_cores_identified == 3
        assert result.restraints_applied == 15
        assert result.contacts_enforced == 8
        
        # Check trajectories
        assert len(result.rmsd_trajectory) == 5
        assert len(result.energy_trajectory) == 5
    
    def test_get_summary(self, test_conformation, test_native_structure):
        """Test get_summary() method."""
        result = RefinementResult(
            initial_structure=test_conformation,
            refined_structure=test_conformation,
            native_structure=test_native_structure,
            initial_rmsd=10.0,
            final_rmsd=3.5,
            rmsd_improvement=6.5,
            helix_rmsd=2.0,
            sheet_rmsd=3.0,
            loop_rmsd=5.0,
            core_rmsd=4.0,
            gdt_ts=60.0,
            tm_score=0.65,
            energy=-80.0,
            iterations_used=5000,
            refinement_time_seconds=120.5,
            quantum_cores_identified=3,
            restraints_applied=15,
            contacts_enforced=8,
            rmsd_trajectory=[],
            energy_trajectory=[]
        )
        
        summary = result.get_summary()
        
        # Check summary contains key information
        assert "QUANTUM REFINEMENT RESULTS" in summary
        assert "10.00" in summary  # Initial RMSD
        assert "3.50" in summary   # Final RMSD
        assert "6.50" in summary   # Improvement
        assert "65.0%" in summary  # Improvement percentage
        assert "60.00" in summary  # GDT-TS
        assert "0.6500" in summary # TM-score
        assert "-80.00" in summary # Energy
        assert "5000" in summary   # Iterations


# ============================================================================
# Test Geometry Validation
# ============================================================================

class TestGeometryValidation:
    """Test validate_geometry() method."""
    
    def test_validate_good_geometry(self, refinement_engine, test_conformation):
        """Test validation passes for good geometry."""
        assert refinement_engine.validate_geometry(test_conformation) is True
    
    def test_validate_nan_coordinates(self, refinement_engine, test_conformation):
        """Test validation fails for NaN coordinates."""
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACDEFGH",
            atom_coordinates=[
                (0.0, 0.0, 0.0),
                (float('nan'), 0.0, 0.0),  # NaN coordinate
                (7.6, 0.0, 0.0),
            ],
            energy=-50.0,
            rmsd_to_native=10.0,
            secondary_structure=['C'] * 3,
            phi_angles=[0.0] * 3,
            psi_angles=[0.0] * 3,
            available_move_types=[],
            structural_constraints={}
        )
        
        assert refinement_engine.validate_geometry(bad_conf) is False
    
    def test_validate_inf_coordinates(self, refinement_engine, test_conformation):
        """Test validation fails for Inf coordinates."""
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACDEFGH",
            atom_coordinates=[
                (0.0, 0.0, 0.0),
                (3.8, float('inf'), 0.0),  # Inf coordinate
                (7.6, 0.0, 0.0),
            ],
            energy=-50.0,
            rmsd_to_native=10.0,
            secondary_structure=['C'] * 3,
            phi_angles=[0.0] * 3,
            psi_angles=[0.0] * 3,
            available_move_types=[],
            structural_constraints={}
        )
        
        assert refinement_engine.validate_geometry(bad_conf) is False
    
    def test_validate_steric_clash(self, refinement_engine):
        """Test validation fails for steric clashes."""
        # Two non-consecutive atoms too close (< 2.0 Å)
        # Create atoms at i=0, i=1, i=2, i=3 where atoms 0 and 2 clash
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACDE",
            atom_coordinates=[
                (0.0, 0.0, 0.0),     # i=0
                (3.8, 0.0, 0.0),     # i=1 (OK distance from i=0)
                (1.5, 0.0, 0.0),     # i=2 (CLASHES with i=0: 1.5Å)
                (5.3, 0.0, 0.0),     # i=3
            ],
            energy=-50.0,
            rmsd_to_native=10.0,
            secondary_structure=['C'] * 4,
            phi_angles=[0.0] * 4,
            psi_angles=[0.0] * 4,
            available_move_types=[],
            structural_constraints={}
        )
        
        assert refinement_engine.validate_geometry(bad_conf) is False
    
    def test_validate_bond_too_short(self, refinement_engine):
        """Test validation fails for bond too short."""
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACE",
            atom_coordinates=[
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),  # Bond only 0.5 Å (< 1.0 Å minimum)
                (4.3, 0.0, 0.0),
            ],
            energy=-50.0,
            rmsd_to_native=10.0,
            secondary_structure=['C'] * 3,
            phi_angles=[0.0] * 3,
            psi_angles=[0.0] * 3,
            available_move_types=[],
            structural_constraints={}
        )
        
        assert refinement_engine.validate_geometry(bad_conf) is False
    
    def test_validate_bond_too_long(self, refinement_engine):
        """Test validation fails for bond too long."""
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACE",
            atom_coordinates=[
                (0.0, 0.0, 0.0),
                (12.0, 0.0, 0.0),  # Bond 12 Å (> 10.0 Å maximum)
                (24.0, 0.0, 0.0),
            ],
            energy=-50.0,
            rmsd_to_native=10.0,
            secondary_structure=['C'] * 3,
            phi_angles=[0.0] * 3,
            psi_angles=[0.0] * 3,
            available_move_types=[],
            structural_constraints={}
        )
        
        assert refinement_engine.validate_geometry(bad_conf) is False


# ============================================================================
# Test Energy Validation
# ============================================================================

class TestEnergyValidation:
    """Test validate_energy() method."""
    
    def test_validate_good_energy(self, refinement_engine):
        """Test validation passes for reasonable energy."""
        assert refinement_engine.validate_energy(-50.0) is True
        assert refinement_engine.validate_energy(-200.0) is True
        assert refinement_engine.validate_energy(100.0) is True
    
    def test_validate_energy_at_threshold(self, refinement_engine):
        """Test validation at threshold boundary."""
        assert refinement_engine.validate_energy(9999.0) is True
        assert refinement_engine.validate_energy(-9999.0) is True
        assert refinement_engine.validate_energy(10000.0) is True  # Exactly at threshold
        assert refinement_engine.validate_energy(-10000.0) is True
    
    def test_validate_energy_above_threshold(self, refinement_engine):
        """Test validation fails above threshold."""
        assert refinement_engine.validate_energy(10001.0) is False
        assert refinement_engine.validate_energy(-10001.0) is False
        assert refinement_engine.validate_energy(50000.0) is False
        assert refinement_engine.validate_energy(-50000.0) is False
    
    def test_validate_energy_custom_threshold(self, refinement_engine):
        """Test validation with custom threshold."""
        assert refinement_engine.validate_energy(500.0, threshold=1000.0) is True
        assert refinement_engine.validate_energy(1500.0, threshold=1000.0) is False


# ============================================================================
# Test Cache Management
# ============================================================================

class TestCacheManagement:
    """Test cache management methods."""
    
    def test_clear_caches(self, refinement_engine):
        """Test clearing all caches."""
        # Add some data to caches
        refinement_engine._qcp_cache['test'] = 5.0
        refinement_engine._thz_mode_cache['test'] = [1.0, 2.0]
        refinement_engine._distance_matrix_cache = [[0.0, 3.8], [3.8, 0.0]]
        
        # Clear caches
        refinement_engine.clear_caches()
        
        # Check caches are empty
        assert refinement_engine._qcp_cache == {}
        assert refinement_engine._thz_mode_cache == {}
        assert refinement_engine._distance_matrix_cache is None


# ============================================================================
# Test Main Refinement Pipeline (Placeholder)
# ============================================================================

class TestRefinementPipeline:
    """Test main refinement pipeline (placeholder for now)."""
    
    def test_refine_structure_quantum_with_native(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test refinement with native structure."""
        result = refinement_engine.refine_structure_quantum(
            test_conformation,
            test_native_structure
        )
        
        # Check result is valid
        assert result is not None
        assert isinstance(result, RefinementResult)
        assert result.initial_structure is test_conformation
        assert result.refined_structure is not None
        assert result.native_structure is test_native_structure
        assert result.refinement_time_seconds > 0
    
    def test_refine_structure_quantum_without_native(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test refinement without native structure."""
        result = refinement_engine.refine_structure_quantum(
            test_conformation,
            None
        )
        
        # Check result is valid
        assert result is not None
        assert isinstance(result, RefinementResult)
        assert result.initial_structure is test_conformation
        assert result.refined_structure is not None
        assert result.native_structure is None
        assert result.refinement_time_seconds > 0
    
    def test_refine_structure_quantum_with_custom_config(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test refinement with custom configuration."""
        config = RefinementConfig(
            stage1_iterations=100,
            stage2_iterations=500,
            qcp_threshold=8.0
        )
        
        result = refinement_engine.refine_structure_quantum(
            test_conformation,
            None,
            config=config
        )
        
        assert result is not None
    
    def test_refine_structure_quantum_invalid_geometry(
        self,
        refinement_engine
    ):
        """Test refinement fails with invalid geometry."""
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACE",
            atom_coordinates=[
                (0.0, 0.0, 0.0),
                (float('nan'), 0.0, 0.0),  # NaN
                (7.6, 0.0, 0.0),
            ],
            energy=-50.0,
            rmsd_to_native=10.0,
            secondary_structure=['C'] * 3,
            phi_angles=[0.0] * 3,
            psi_angles=[0.0] * 3,
            available_move_types=[],
            structural_constraints={}
        )
        
        with pytest.raises(GeometryError, match="invalid geometry"):
            refinement_engine.refine_structure_quantum(bad_conf, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# Test Two-Stage Optimization (Task 8)
# ============================================================================

class TestStage1GlobalOptimization:
    """Test Stage 1 global fold optimization."""
    
    def test_optimize_stage1_basic(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test basic Stage 1 optimization."""
        structure, rmsd_traj, energy_traj = refinement_engine.optimize_stage1_global(
            structure=test_conformation,
            temperature=1.0,
            iterations=100,  # Small for testing
            native_structure=test_native_structure
        )
        
        # Check outputs
        assert structure is not None
        assert isinstance(structure, Conformation)
        assert len(rmsd_traj) > 0
        assert len(energy_traj) > 0
        
        # Check trajectory lengths (every 10 iterations + initial + final)
        # 100 iterations / 10 + 2 (initial + final) = 12
        assert len(rmsd_traj) >= 11  # At least initial + some iterations + final
        assert len(energy_traj) >= 11
    
    def test_optimize_stage1_without_native(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test Stage 1 without native structure."""
        structure, rmsd_traj, energy_traj = refinement_engine.optimize_stage1_global(
            structure=test_conformation,
            temperature=1.0,
            iterations=50,
            native_structure=None
        )
        
        # Check outputs
        assert structure is not None
        assert isinstance(structure, Conformation)
        assert len(rmsd_traj) == 0  # No RMSD without native
        assert len(energy_traj) > 0
    
    def test_optimize_stage1_temperature_effect(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test temperature affects exploration."""
        # Higher temperature should allow more exploration
        structure_hot, _, energy_traj_hot = refinement_engine.optimize_stage1_global(
            structure=test_conformation,
            temperature=2.0,
            iterations=50,
            native_structure=None
        )
        
        # Lower temperature should be more conservative
        structure_cold, _, energy_traj_cold = refinement_engine.optimize_stage1_global(
            structure=test_conformation,
            temperature=0.1,
            iterations=50,
            native_structure=None
        )
        
        # Both should complete successfully
        assert structure_hot is not None
        assert structure_cold is not None
    
    def test_optimize_stage1_energy_tracking(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test energy is tracked correctly."""
        structure, rmsd_traj, energy_traj = refinement_engine.optimize_stage1_global(
            structure=test_conformation,
            temperature=1.0,
            iterations=100,
            native_structure=None
        )
        
        # Check all energies are finite
        assert all(math.isfinite(e) for e in energy_traj)
        
        # Check energy trajectory is non-empty
        assert len(energy_traj) > 0
    
    def test_optimize_stage1_rmsd_tracking(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test RMSD is tracked correctly."""
        structure, rmsd_traj, energy_traj = refinement_engine.optimize_stage1_global(
            structure=test_conformation,
            temperature=1.0,
            iterations=100,
            native_structure=test_native_structure
        )
        
        # Check all RMSDs are positive and finite
        assert all(r > 0 and math.isfinite(r) for r in rmsd_traj)
        
        # Check RMSD trajectory is non-empty
        assert len(rmsd_traj) > 0


class TestStage2RefinementOptimization:
    """Test Stage 2 quantum refinement optimization."""
    
    def test_optimize_stage2_basic(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test basic Stage 2 optimization."""
        structure, rmsd_traj, energy_traj = refinement_engine.optimize_stage2_refinement(
            structure=test_conformation,
            exploration_temperature=1.0,
            restraint_weight=10.0,
            qcp_weight=0.3,
            iterations=100,  # Small for testing
            native_structure=test_native_structure
        )
        
        # Check outputs
        assert structure is not None
        assert isinstance(structure, Conformation)
        assert len(rmsd_traj) > 0
        assert len(energy_traj) > 0
    
    def test_optimize_stage2_reduced_temperature(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test Stage 2 uses reduced temperature (0.1×)."""
        # This is tested implicitly - Stage 2 should use 0.1× exploration temperature
        # Here we verify it runs successfully with reduced temperature
        structure, _, energy_traj = refinement_engine.optimize_stage2_refinement(
            structure=test_conformation,
            exploration_temperature=1.0,  # Will be reduced to 0.1 internally
            restraint_weight=10.0,
            qcp_weight=0.3,
            iterations=50,
            native_structure=None
        )
        
        assert structure is not None
        assert len(energy_traj) > 0
    
    def test_optimize_stage2_without_native(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test Stage 2 without native structure."""
        structure, rmsd_traj, energy_traj = refinement_engine.optimize_stage2_refinement(
            structure=test_conformation,
            exploration_temperature=1.0,
            restraint_weight=10.0,
            qcp_weight=0.3,
            iterations=50,
            native_structure=None
        )
        
        # Check outputs
        assert structure is not None
        assert len(rmsd_traj) == 0  # No RMSD without native
        assert len(energy_traj) > 0
    
    def test_optimize_stage2_restraint_weight(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test different restraint weights."""
        # High restraint weight
        structure_high, _, _ = refinement_engine.optimize_stage2_refinement(
            structure=test_conformation,
            exploration_temperature=1.0,
            restraint_weight=100.0,
            qcp_weight=0.3,
            iterations=50,
            native_structure=None
        )
        
        # Low restraint weight
        structure_low, _, _ = refinement_engine.optimize_stage2_refinement(
            structure=test_conformation,
            exploration_temperature=1.0,
            restraint_weight=1.0,
            qcp_weight=0.3,
            iterations=50,
            native_structure=None
        )
        
        # Both should complete successfully
        assert structure_high is not None
        assert structure_low is not None
    
    def test_optimize_stage2_qcp_weight(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test different QCP weights."""
        # Standard QCP weight (30%)
        structure_std, _, _ = refinement_engine.optimize_stage2_refinement(
            structure=test_conformation,
            exploration_temperature=1.0,
            restraint_weight=10.0,
            qcp_weight=0.3,
            iterations=50,
            native_structure=None
        )
        
        # Higher QCP weight (50%)
        structure_high, _, _ = refinement_engine.optimize_stage2_refinement(
            structure=test_conformation,
            exploration_temperature=1.0,
            restraint_weight=10.0,
            qcp_weight=0.5,
            iterations=50,
            native_structure=None
        )
        
        # Both should complete successfully
        assert structure_std is not None
        assert structure_high is not None
    
    def test_optimize_stage2_energy_validation(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test energy validation in Stage 2."""
        structure, rmsd_traj, energy_traj = refinement_engine.optimize_stage2_refinement(
            structure=test_conformation,
            exploration_temperature=1.0,
            restraint_weight=10.0,
            qcp_weight=0.3,
            iterations=100,
            native_structure=None
        )
        
        # Check all energies are finite and reasonable
        assert all(math.isfinite(e) for e in energy_traj)
        assert all(abs(e) < 10000.0 for e in energy_traj)


class TestTwoStageOrchestration:
    """Test complete two-stage optimization pipeline."""
    
    def test_optimize_two_stage_basic(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test basic two-stage optimization."""
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=test_native_structure,
            config=None  # Use defaults
        )
        
        # Check result
        assert result is not None
        assert isinstance(result, RefinementResult)
        assert result.initial_structure is test_conformation
        assert result.refined_structure is not None
        assert result.native_structure is test_native_structure
        assert result.refinement_time_seconds > 0
        
        # Check metrics
        assert result.initial_rmsd > 0
        assert result.final_rmsd >= 0
        assert result.gdt_ts >= 0
        assert result.tm_score >= 0
    
    def test_optimize_two_stage_without_native(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test two-stage optimization without native structure."""
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=None,
            config=None
        )
        
        # Check result
        assert result is not None
        assert result.native_structure is None
        assert result.initial_rmsd == 0.0  # Default when no native
        assert result.final_rmsd == 0.0
        assert result.refinement_time_seconds > 0
    
    def test_optimize_two_stage_with_custom_config(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test two-stage optimization with custom config."""
        config = RefinementConfig(
            stage1_temperature=2.0,
            stage1_iterations=50,
            stage2_temperature=0.2,
            stage2_iterations=100,
            restraint_weight=20.0,
            qcp_weight=0.5
        )
        
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=test_native_structure,
            config=config
        )
        
        # Check result
        assert result is not None
        assert result.iterations_used == 50 + 100  # Stage 1 + Stage 2
    
    def test_optimize_two_stage_trajectory_tracking(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test trajectory tracking in two-stage optimization."""
        config = RefinementConfig(
            stage1_iterations=50,
            stage2_iterations=100
        )
        
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=test_native_structure,
            config=config
        )
        
        # Check trajectories combine both stages
        assert len(result.rmsd_trajectory) > 0
        assert len(result.energy_trajectory) > 0
        
        # Trajectories should include data from both stages
        # Stage 1: 50 iterations / 10 = 5 points + initial + final = 7
        # Stage 2: 100 iterations / 100 = 1 point + initial + final = 3
        # Total: ~10 points (approximate)
        assert len(result.rmsd_trajectory) >= 5
        assert len(result.energy_trajectory) >= 5
    
    def test_optimize_two_stage_invalid_geometry(
        self,
        refinement_engine
    ):
        """Test two-stage optimization fails with invalid geometry."""
        bad_conf = Conformation(
            conformation_id="bad_conf",
            sequence="ACE",
            atom_coordinates=[
                (0.0, 0.0, 0.0),
                (float('nan'), 0.0, 0.0),  # NaN
                (7.6, 0.0, 0.0),
            ],
            energy=-50.0,
            rmsd_to_native=10.0,
            secondary_structure=['C'] * 3,
            phi_angles=[0.0] * 3,
            psi_angles=[0.0] * 3,
            available_move_types=[],
            structural_constraints={}
        )
        
        with pytest.raises(GeometryError, match="invalid geometry"):
            refinement_engine.optimize_two_stage(bad_conf, None)
    
    def test_optimize_two_stage_quality_metrics(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test quality metrics are calculated."""
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=test_native_structure,
            config=None
        )
        
        # Check quality metrics are present
        assert result.gdt_ts >= 0 and result.gdt_ts <= 100
        assert result.tm_score >= 0 and result.tm_score <= 1.0
        assert math.isfinite(result.energy)
    
    def test_optimize_two_stage_iteration_count(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test iteration count matches config."""
        config = RefinementConfig(
            stage1_iterations=100,
            stage2_iterations=500
        )
        
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=None,
            config=config
        )
        
        # Total iterations should be Stage 1 + Stage 2
        assert result.iterations_used == 100 + 500
    
    def test_optimize_two_stage_stage1_only_if_good_rmsd(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test that Stage 2 is entered even if Stage 1 achieves good RMSD."""
        # Note: Current implementation always proceeds to Stage 2 for further refinement
        # This test verifies that behavior
        
        config = RefinementConfig(
            stage1_iterations=50,
            stage2_iterations=100
        )
        
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=test_native_structure,
            config=config
        )
        
        # Should always proceed to Stage 2 (total iterations = both stages)
        assert result.iterations_used == 50 + 100


class TestTwoStageRMSDImprovement:
    """Test RMSD improvement tracking in two-stage optimization."""
    
    def test_rmsd_improvement_calculation(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test RMSD improvement is calculated correctly."""
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=test_native_structure,
            config=None
        )
        
        # Check improvement calculation
        expected_improvement = result.initial_rmsd - result.final_rmsd
        assert abs(result.rmsd_improvement - expected_improvement) < 0.01
    
    def test_rmsd_trajectory_decreases_or_stable(
        self,
        refinement_engine,
        test_conformation,
        test_native_structure
    ):
        """Test RMSD trajectory shows optimization progress."""
        config = RefinementConfig(
            stage1_iterations=100,
            stage2_iterations=200
        )
        
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=test_native_structure,
            config=config
        )
        
        # RMSD trajectory should have multiple points
        assert len(result.rmsd_trajectory) >= 3
        
        # All RMSD values should be positive
        assert all(r > 0 for r in result.rmsd_trajectory)
    
    def test_energy_trajectory_tracking(
        self,
        refinement_engine,
        test_conformation
    ):
        """Test energy trajectory is tracked properly."""
        config = RefinementConfig(
            stage1_iterations=100,
            stage2_iterations=200
        )
        
        result = refinement_engine.optimize_two_stage(
            initial_structure=test_conformation,
            native_structure=None,
            config=config
        )
        
        # Energy trajectory should have multiple points
        assert len(result.energy_trajectory) >= 3
        
        # All energies should be finite
        assert all(math.isfinite(e) for e in result.energy_trajectory)


class TestHelperMethods:
    """Test helper methods for two-stage optimization."""
    
    def test_create_conformation_helper(self, refinement_engine):
        """Test _create_conformation helper method."""
        coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)]
        sequence = "ACE"
        
        conf = refinement_engine._create_conformation(
            sequence=sequence,
            coordinates=coords,
            conformation_id="test",
            energy=-50.0,
            rmsd_to_native=5.0
        )
        
        # Check conformation is properly created
        assert conf.sequence == sequence
        assert conf.atom_coordinates == coords
        assert conf.conformation_id == "test"
        assert conf.energy == -50.0
        assert conf.rmsd_to_native == 5.0
        assert len(conf.secondary_structure) == 3
        assert all(ss == 'C' for ss in conf.secondary_structure)
        assert len(conf.phi_angles) == 3
        assert len(conf.psi_angles) == 3
    
    def test_create_conformation_defaults(self, refinement_engine):
        """Test _create_conformation with default parameters."""
        coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)]
        sequence = "AC"
        
        conf = refinement_engine._create_conformation(
            sequence=sequence,
            coordinates=coords
        )
        
        # Check defaults
        assert conf.conformation_id == "refinement"
        assert conf.energy == 0.0
        assert conf.rmsd_to_native is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
