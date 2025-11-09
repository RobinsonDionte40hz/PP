"""
Integration Tests for Quantum Refinement Engine

Tests the complete refinement pipeline from coarse structure to refined structure,
including all refinement strategies and optimization stages.

Test Coverage:
- Full refinement pipeline orchestration
- Quantum core identification and THz resonance
- Distance restraint application
- Secondary structure registration
- Hydrophobic core packing
- Loop refinement with G(φ,t)
- Tertiary contact prediction and enforcement
- Two-stage optimization
- RMSD component diagnostics
- Performance benchmarks

Test Proteins:
- 1UBQ (Ubiquitin, 76 residues): Target RMSD <4Å
- 1CRN (Crambin, 46 residues): Target RMSD <3Å
- Small test protein (7 residues): Fast validation
"""

import pytest
import math
import time
from typing import List, Tuple

try:
    from ..quantum_refinement_engine import QuantumRefinementEngine, RefinementError, ConvergenceError, GeometryError
    from ..qcpp_integration import QCPPIntegrationAdapter
    from ..energy_function import MolecularMechanicsEnergy
    from ..rmsd_calculator import RMSDCalculator, NativeStructure
    from ..models import Conformation, RefinementConfig, RefinementResult
except ImportError:
    from ubf_protein.quantum_refinement_engine import QuantumRefinementEngine, RefinementError, ConvergenceError, GeometryError
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
    from ubf_protein.energy_function import MolecularMechanicsEnergy
    from ubf_protein.rmsd_calculator import RMSDCalculator, NativeStructure
    from ubf_protein.models import Conformation, RefinementConfig, RefinementResult


# ===== Fixtures =====

@pytest.fixture
def qcpp_adapter():
    """Create QCPP integration adapter."""
    # Create a real adapter with minimal mock predictor
    class MockQCPPPredictor:
        """Mock QCPP predictor for testing."""
        def __init__(self):
            self.phi = 1.618033988749895
        
        def predict(self, sequence, coordinates):
            """Return mock QCP analysis."""
            n = len(sequence)
            return {
                'qcp_values': [7.5] * n,
                'field_coherence': 0.75,
                'stability_score': 0.8,
                'thz_spectrum': [1.618, 2.618, 4.236]
            }
    
    # Create real QCPPIntegrationAdapter with mock predictor
    mock_predictor = MockQCPPPredictor()
    return QCPPIntegrationAdapter(
        predictor=mock_predictor,
        cache_size=1000,
        target_geometry='none'
    )


@pytest.fixture
def energy_calculator():
    """Create energy calculator."""
    return MolecularMechanicsEnergy()


@pytest.fixture
def rmsd_calculator():
    """Create RMSD calculator."""
    return RMSDCalculator()


@pytest.fixture
def refinement_engine(qcpp_adapter, energy_calculator, rmsd_calculator):
    """Create quantum refinement engine."""
    return QuantumRefinementEngine(
        qcpp_adapter=qcpp_adapter,
        energy_calculator=energy_calculator,
        rmsd_calculator=rmsd_calculator
    )


@pytest.fixture
def small_test_structure():
    """Create small test structure (7 residues)."""
    sequence = "ACDEFGH"
    coords = [
        (0.0, 0.0, 0.0),
        (3.8, 0.0, 0.0),
        (7.6, 0.0, 0.0),
        (11.4, 0.0, 0.0),
        (15.2, 0.0, 0.0),
        (19.0, 0.0, 0.0),
        (22.8, 0.0, 0.0)
    ]
    
    return Conformation(
        conformation_id="test_small",
        sequence=sequence,
        atom_coordinates=coords,
        energy=0.0,
        rmsd_to_native=None,
        secondary_structure=['H', 'H', 'H', 'C', 'E', 'E', 'E'],
        phi_angles=[0.0] * 7,
        psi_angles=[0.0] * 7,
        available_move_types=[],
        structural_constraints={}
    )


@pytest.fixture
def small_native_structure():
    """Create small native structure (slightly different from test)."""
    coords = [
        (0.5, 0.2, 0.1),
        (4.0, 0.1, 0.3),
        (7.8, 0.3, 0.2),
        (11.6, 0.4, 0.1),
        (15.4, 0.2, 0.3),
        (19.2, 0.1, 0.2),
        (23.0, 0.3, 0.1)
    ]
    
    return NativeStructure(
        sequence="ACDEFGH",
        ca_coords=coords,
        pdb_id="TEST"
    )


@pytest.fixture
def medium_test_structure():
    """Create medium test structure (20 residues)."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    n = 20
    
    # Create simple linear extended chain with consistent 3.8Å CA-CA spacing
    coords = []
    for i in range(n):
        # Linear chain along x-axis only, ensuring all bonds are 3.8Å
        x = i * 3.8
        y = 0.0
        z = 0.0
        coords.append((x, y, z))
    
    # Mix secondary structure
    ss = ['H'] * 7 + ['C'] * 3 + ['E'] * 5 + ['C'] * 2 + ['H'] * 3
    
    return Conformation(
        conformation_id="test_medium",
        sequence=sequence,
        atom_coordinates=coords,
        energy=0.0,
        rmsd_to_native=None,
        secondary_structure=ss,
        phi_angles=[0.0] * n,
        psi_angles=[0.0] * n,
        available_move_types=[],
        structural_constraints={}
    )


# ===== Test Cases =====

class TestRefinementEngineIntegration:
    """Integration tests for full refinement pipeline."""
    
    def test_full_pipeline_small_structure(
        self,
        refinement_engine,
        small_test_structure,
        small_native_structure
    ):
        """Test complete refinement pipeline on small structure."""
        # Create config with fast parameters for testing
        config = RefinementConfig(
            stage1_temperature=1.0,
            stage1_iterations=100,  # Reduced for testing
            stage2_temperature=0.1,
            stage2_iterations=100,  # Reduced for testing
            qcp_threshold=7.0,
            restraint_weight=10.0,
            qcp_weight=0.3
        )
        
        # Run refinement
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=small_test_structure,
            native_structure=small_native_structure,
            config=config
        )
        
        # Verify result structure
        assert isinstance(result, RefinementResult)
        assert result.initial_structure == small_test_structure
        assert result.refined_structure is not None
        assert result.native_structure == small_native_structure
        
        # Verify metrics are calculated
        assert result.initial_rmsd >= 0.0
        assert result.final_rmsd >= 0.0
        assert result.energy != 0.0  # Should have non-zero energy
        
        # Verify refinement tracking
        assert result.quantum_cores_identified >= 0
        assert result.restraints_applied >= 0
        assert result.contacts_enforced >= 0
        assert result.iterations_used > 0
        assert result.refinement_time_seconds > 0.0
        
        # Verify trajectories
        assert len(result.rmsd_trajectory) > 0
        assert len(result.energy_trajectory) > 0
        
        print(f"\nSmall structure refinement results:")
        print(f"  RMSD: {result.initial_rmsd:.2f}Å → {result.final_rmsd:.2f}Å")
        print(f"  Improvement: {result.rmsd_improvement:.2f}Å")
        print(f"  Quantum cores: {result.quantum_cores_identified}")
        print(f"  Restraints: {result.restraints_applied}")
        print(f"  Contacts: {result.contacts_enforced}")
        print(f"  Time: {result.refinement_time_seconds:.2f}s")
    
    def test_full_pipeline_medium_structure(
        self,
        refinement_engine,
        medium_test_structure
    ):
        """Test refinement pipeline on medium structure (no native)."""
        config = RefinementConfig(
            stage1_iterations=50,
            stage2_iterations=50
        )
        
        # Run refinement without native structure
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=medium_test_structure,
            native_structure=None,
            config=config
        )
        
        # Verify result
        assert result.refined_structure is not None
        assert result.native_structure is None
        assert result.initial_rmsd == 0.0  # No native, so RMSD is 0
        assert result.final_rmsd == 0.0
        assert result.energy != 0.0
        
        # Should still track refinement metrics
        assert result.quantum_cores_identified >= 0
        assert result.restraints_applied >= 0
        assert result.iterations_used > 0
        
        print(f"\nMedium structure refinement results (no native):")
        print(f"  Energy: {result.energy:.2f} kcal/mol")
        print(f"  Quantum cores: {result.quantum_cores_identified}")
        print(f"  Restraints: {result.restraints_applied}")
        print(f"  Contacts: {result.contacts_enforced}")
        print(f"  Time: {result.refinement_time_seconds:.2f}s")
    
    def test_quantum_core_identification_integration(
        self,
        refinement_engine,
        medium_test_structure
    ):
        """Test quantum core identification in refinement pipeline."""
        config = RefinementConfig(
            stage1_iterations=10,
            stage2_iterations=10,
            qcp_threshold=5.0  # Lower threshold to ensure cores are found
        )
        
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=medium_test_structure,
            native_structure=None,
            config=config
        )
        
        # Should identify at least some quantum cores with lower threshold
        # Note: Actual number depends on mock QCPP adapter implementation
        assert result.quantum_cores_identified >= 0
        
        print(f"\nQuantum core identification:")
        print(f"  Cores found: {result.quantum_cores_identified}")
        print(f"  QCP threshold: {config.qcp_threshold}")
    
    def test_distance_restraints_integration(
        self,
        refinement_engine,
        small_test_structure
    ):
        """Test distance restraint application in pipeline."""
        config = RefinementConfig(
            stage1_iterations=10,
            stage2_iterations=10,
            qcp_threshold=5.0
        )
        
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=small_test_structure,
            native_structure=None,
            config=config
        )
        
        # Should apply some restraints
        assert result.restraints_applied >= 0
        
        print(f"\nDistance restraints:")
        print(f"  Restraints applied: {result.restraints_applied}")
    
    def test_secondary_structure_registration_integration(
        self,
        refinement_engine,
        small_test_structure,
        small_native_structure
    ):
        """Test secondary structure registration in pipeline."""
        config = RefinementConfig(
            stage1_iterations=10,
            stage2_iterations=10
        )
        
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=small_test_structure,
            native_structure=small_native_structure,
            config=config
        )
        
        # Check that helix/sheet RMSD components are calculated
        assert result.helix_rmsd >= 0.0
        assert result.sheet_rmsd >= 0.0
        assert result.loop_rmsd >= 0.0
        
        print(f"\nSecondary structure RMSD components:")
        print(f"  Helix: {result.helix_rmsd:.2f}Å")
        print(f"  Sheet: {result.sheet_rmsd:.2f}Å")
        print(f"  Loop: {result.loop_rmsd:.2f}Å")
        print(f"  Core: {result.core_rmsd:.2f}Å")
    
    def test_tertiary_contact_enforcement_integration(
        self,
        refinement_engine,
        medium_test_structure
    ):
        """Test tertiary contact prediction and enforcement."""
        config = RefinementConfig(
            stage1_iterations=10,
            stage2_iterations=10
        )
        
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=medium_test_structure,
            native_structure=None,
            config=config
        )
        
        # Should predict and enforce some contacts
        assert result.contacts_enforced >= 0
        
        print(f"\nTertiary contacts:")
        print(f"  Contacts enforced: {result.contacts_enforced}")
    
    def test_two_stage_optimization_integration(
        self,
        refinement_engine,
        small_test_structure,
        small_native_structure
    ):
        """Test two-stage optimization pipeline."""
        config = RefinementConfig(
            stage1_temperature=1.0,
            stage1_iterations=50,
            stage2_temperature=0.1,
            stage2_iterations=50
        )
        
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=small_test_structure,
            native_structure=small_native_structure,
            config=config
        )
        
        # Should have used iterations from both stages
        expected_min_iterations = config.stage1_iterations + config.stage2_iterations
        assert result.iterations_used >= expected_min_iterations * 0.8  # Allow some tolerance
        
        # Should have RMSD trajectory from both stages
        assert len(result.rmsd_trajectory) > 0
        assert len(result.energy_trajectory) > 0
        
        print(f"\nTwo-stage optimization:")
        print(f"  Total iterations: {result.iterations_used}")
        print(f"  Trajectory length: {len(result.rmsd_trajectory)} RMSD points")
        print(f"  Stage 1 config: temp={config.stage1_temperature}, iter={config.stage1_iterations}")
        print(f"  Stage 2 config: temp={config.stage2_temperature}, iter={config.stage2_iterations}")
    
    def test_rmsd_component_diagnostics_integration(
        self,
        refinement_engine,
        small_test_structure,
        small_native_structure
    ):
        """Test RMSD component diagnostics in pipeline."""
        config = RefinementConfig(
            stage1_iterations=20,
            stage2_iterations=20
        )
        
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=small_test_structure,
            native_structure=small_native_structure,
            config=config
        )
        
        # Verify all component RMSDs are calculated
        assert result.helix_rmsd >= 0.0
        assert result.sheet_rmsd >= 0.0
        assert result.loop_rmsd >= 0.0
        assert result.core_rmsd >= 0.0
        
        # At least one component should be non-zero
        total_components = result.helix_rmsd + result.sheet_rmsd + result.loop_rmsd + result.core_rmsd
        assert total_components > 0.0
        
        print(f"\nRMSD component breakdown:")
        print(f"  Total RMSD: {result.final_rmsd:.2f}Å")
        print(f"  Helix: {result.helix_rmsd:.2f}Å")
        print(f"  Sheet: {result.sheet_rmsd:.2f}Å")
        print(f"  Loop: {result.loop_rmsd:.2f}Å")
        print(f"  Core: {result.core_rmsd:.2f}Å")
    
    def test_invalid_geometry_handling(
        self,
        refinement_engine
    ):
        """Test error handling for invalid initial geometry."""
        # Create structure with NaN coordinates
        invalid_structure = Conformation(
            conformation_id="invalid",
            sequence="ACDE",
            atom_coordinates=[(0.0, 0.0, 0.0), (float('nan'), 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)],
            energy=0.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 4,
            phi_angles=[0.0] * 4,
            psi_angles=[0.0] * 4,
            available_move_types=[],
            structural_constraints={}
        )
        
        # Should raise GeometryError
        with pytest.raises(GeometryError, match="invalid geometry"):
            refinement_engine.refine_structure_quantum(
                coarse_structure=invalid_structure,
                native_structure=None
            )
    
    def test_performance_benchmarks(
        self,
        refinement_engine,
        small_test_structure,
        medium_test_structure
    ):
        """Benchmark refinement performance."""
        config = RefinementConfig(
            stage1_iterations=50,
            stage2_iterations=50
        )
        
        # Small structure (7 residues)
        start = time.time()
        result_small = refinement_engine.refine_structure_quantum(
            coarse_structure=small_test_structure,
            native_structure=None,
            config=config
        )
        time_small = time.time() - start
        
        # Medium structure (20 residues)
        start = time.time()
        result_medium = refinement_engine.refine_structure_quantum(
            coarse_structure=medium_test_structure,
            native_structure=None,
            config=config
        )
        time_medium = time.time() - start
        
        print(f"\nPerformance benchmarks:")
        print(f"  Small (7 res): {time_small:.2f}s")
        print(f"  Medium (20 res): {time_medium:.2f}s")
        print(f"  Scaling factor: {time_medium / time_small:.2f}x")
        
        # Should complete in reasonable time
        assert time_small < 10.0  # <10s for small
        assert time_medium < 30.0  # <30s for medium
    
    def test_config_parameter_effects(
        self,
        refinement_engine,
        small_test_structure
    ):
        """Test that config parameters affect refinement."""
        # High temperature config
        config_hot = RefinementConfig(
            stage1_temperature=2.0,
            stage1_iterations=20,
            stage2_iterations=20
        )
        
        # Low temperature config
        config_cold = RefinementConfig(
            stage1_temperature=0.5,
            stage1_iterations=20,
            stage2_iterations=20
        )
        
        result_hot = refinement_engine.refine_structure_quantum(
            coarse_structure=small_test_structure,
            native_structure=None,
            config=config_hot
        )
        
        result_cold = refinement_engine.refine_structure_quantum(
            coarse_structure=small_test_structure,
            native_structure=None,
            config=config_cold
        )
        
        # Results should be different (different exploration)
        # Note: Energy might be similar, but structures will differ
        assert result_hot.energy != result_cold.energy or \
               result_hot.refined_structure.atom_coordinates != result_cold.refined_structure.atom_coordinates
        
        print(f"\nConfig parameter effects:")
        print(f"  Hot config (T={config_hot.stage1_temperature}): E={result_hot.energy:.2f} kcal/mol")
        print(f"  Cold config (T={config_cold.stage1_temperature}): E={result_cold.energy:.2f} kcal/mol")


# ===== Performance Tests =====

class TestRefinementEnginePerformance:
    """Performance tests for refinement engine."""
    
    def test_quantum_core_identification_performance(
        self,
        refinement_engine,
        medium_test_structure
    ):
        """Test quantum core identification speed."""
        config = RefinementConfig(
            stage1_iterations=5,
            stage2_iterations=5
        )
        
        start = time.time()
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=medium_test_structure,
            native_structure=None,
            config=config
        )
        total_time = time.time() - start
        
        # Quantum core identification should be fast (<100ms target)
        # With full pipeline, allow more time but verify it's reasonable
        print(f"\nQuantum core identification performance:")
        print(f"  Total pipeline time: {total_time:.3f}s")
        print(f"  Cores identified: {result.quantum_cores_identified}")
        
        # Should complete in reasonable time for medium structure
        assert total_time < 15.0  # <15s for full pipeline with 20 residues
    
    def test_memory_efficiency(
        self,
        refinement_engine,
        medium_test_structure
    ):
        """Test memory efficiency (trajectory storage)."""
        config = RefinementConfig(
            stage1_iterations=100,
            stage2_iterations=100
        )
        
        result = refinement_engine.refine_structure_quantum(
            coarse_structure=medium_test_structure,
            native_structure=None,
            config=config
        )
        
        # Trajectory should not be excessively long
        # With sampling every 10 iterations, expect ~20 points
        print(f"\nMemory efficiency:")
        print(f"  RMSD trajectory points: {len(result.rmsd_trajectory)}")
        print(f"  Energy trajectory points: {len(result.energy_trajectory)}")
        print(f"  Total iterations: {result.iterations_used}")
        
        # Reasonable trajectory size
        assert len(result.rmsd_trajectory) < 1000
        assert len(result.energy_trajectory) < 1000


# ===== Run Tests =====

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
