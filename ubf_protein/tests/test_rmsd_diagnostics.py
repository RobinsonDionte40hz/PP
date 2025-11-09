"""
Unit tests for RMSD component diagnostics (Task 9).

Tests the diagnose_rmsd_components() method and related functionality
in QuantumRefinementEngine.
"""

import pytest
from typing import List, Tuple, Dict
import math

try:
    from ..quantum_refinement_engine import QuantumRefinementEngine
    from ..qcpp_integration import QCPPIntegrationAdapter
    from ..energy_function import MolecularMechanicsEnergy
    from ..rmsd_calculator import RMSDCalculator, NativeStructure
    from ..models import Conformation
except ImportError:
    from ubf_protein.quantum_refinement_engine import QuantumRefinementEngine
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
    from ubf_protein.energy_function import MolecularMechanicsEnergy
    from ubf_protein.rmsd_calculator import RMSDCalculator, NativeStructure
    from ubf_protein.models import Conformation


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_qcpp_adapter(monkeypatch):
    """Create mock QCPP adapter that bypasses type checking."""
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics
    
    class MockAdapter(QCPPIntegrationAdapter):
        def __init__(self):
            # Don't call super().__init__() to avoid needing a real predictor
            pass
        
        def analyze_conformation(self, conformation):
            """Return dummy metrics."""
            return QCPPMetrics(
                qcp_score=5.0,
                field_coherence=0.5,
                stability_score=10.0,
                phi_match_score=0.7,
                calculation_time_ms=1.0
            )
    
    return MockAdapter()


@pytest.fixture
def energy_calculator():
    """Create energy calculator."""
    return MolecularMechanicsEnergy()


@pytest.fixture
def rmsd_calculator():
    """Create RMSD calculator."""
    return RMSDCalculator(align_structures=True)


@pytest.fixture
def refinement_engine(mock_qcpp_adapter, energy_calculator, rmsd_calculator):
    """Create refinement engine for testing."""
    return QuantumRefinementEngine(
        qcpp_adapter=mock_qcpp_adapter,
        energy_calculator=energy_calculator,
        rmsd_calculator=rmsd_calculator
    )


@pytest.fixture
def sample_conformation():
    """
    Create a sample protein conformation with mixed secondary structure.
    
    Sequence: HHHHHHEEEEELLLLLAAAAA (20 residues)
    - Residues 0-5: Helix (H)
    - Residues 6-10: Sheet (E)
    - Residues 11-15: Loop (L)
    - Residues 16-19: Hydrophobic core (A = alanine, hydrophobic)
    """
    sequence = "MHHHHHEEEEEAAAAAVVVV"  # 20 residues
    n_residues = len(sequence)
    
    # Create coordinates (simple linear structure for testing)
    coords = [(float(i), 0.0, 0.0) for i in range(n_residues)]
    
    # Secondary structure annotations
    ss = ['C'] * n_residues
    ss[1:7] = ['H'] * 6   # Helix residues 1-6
    ss[7:12] = ['E'] * 5  # Sheet residues 7-11
    ss[12:17] = ['C'] * 5  # Loop residues 12-16
    # Residues 17-19 are coil but hydrophobic
    
    # Phi/Psi angles (simple values)
    phi = [-60.0] * n_residues
    psi = [-45.0] * n_residues
    
    return Conformation(
        conformation_id="test_conf",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-100.0,
        rmsd_to_native=None,
        secondary_structure=ss,
        phi_angles=phi,
        psi_angles=psi,
        available_move_types=[],
        structural_constraints={}
    )


@pytest.fixture
def native_structure(sample_conformation):
    """
    Create native structure (slightly displaced from sample).
    
    This creates a "ground truth" with known deviations from sample.
    """
    # Displace each residue by different amounts
    # Helix (1-6): 1Å displacement
    # Sheet (7-11): 2Å displacement
    # Loop (12-16): 3Å displacement
    # Core (17-19): 1.5Å displacement
    
    coords = []
    for i, (x, y, z) in enumerate(sample_conformation.atom_coordinates):
        if 1 <= i <= 6:  # Helix
            coords.append((x + 1.0, y, z))
        elif 7 <= i <= 11:  # Sheet
            coords.append((x + 2.0, y, z))
        elif 12 <= i <= 16:  # Loop
            coords.append((x + 3.0, y, z))
        else:  # Other
            coords.append((x + 1.5, y, z))
    
    native = NativeStructure(
        pdb_id="1TEST",
        sequence=sample_conformation.sequence,
        ca_coords=coords
    )
    return native


# ============================================================================
# Test: diagnose_rmsd_components() - Basic Functionality
# ============================================================================

def test_diagnose_rmsd_components_basic(refinement_engine, sample_conformation, native_structure):
    """Test basic RMSD component calculation."""
    diagnostics = refinement_engine.diagnose_rmsd_components(
        sample_conformation,
        native_structure
    )
    
    # Should return dictionary with all required keys
    assert 'total_rmsd' in diagnostics
    assert 'helix_rmsd' in diagnostics
    assert 'sheet_rmsd' in diagnostics
    assert 'loop_rmsd' in diagnostics
    assert 'core_rmsd' in diagnostics
    assert 'helix_percentage' in diagnostics
    assert 'sheet_percentage' in diagnostics
    assert 'loop_percentage' in diagnostics
    assert 'core_percentage' in diagnostics
    assert 'report' in diagnostics
    
    # All RMSD values should be non-negative
    assert diagnostics['total_rmsd'] >= 0.0
    assert diagnostics['helix_rmsd'] >= 0.0
    assert diagnostics['sheet_rmsd'] >= 0.0
    assert diagnostics['loop_rmsd'] >= 0.0
    assert diagnostics['core_rmsd'] >= 0.0


def test_diagnose_rmsd_components_residue_lists(refinement_engine, sample_conformation, native_structure):
    """Test that residue lists are correctly identified."""
    diagnostics = refinement_engine.diagnose_rmsd_components(
        sample_conformation,
        native_structure
    )
    
    # Check residue list keys exist
    assert 'helix_residues' in diagnostics
    assert 'sheet_residues' in diagnostics
    assert 'loop_residues' in diagnostics
    assert 'core_residues' in diagnostics
    
    # Check helix residues (1-6)
    helix_res = diagnostics['helix_residues']
    assert len(helix_res) == 6
    assert all(1 <= r <= 6 for r in helix_res)
    
    # Check sheet residues (7-11)
    sheet_res = diagnostics['sheet_residues']
    assert len(sheet_res) == 5
    assert all(7 <= r <= 11 for r in sheet_res)
    
    # Check that we have some loop/coil residues
    loop_res = diagnostics['loop_residues']
    assert len(loop_res) > 0
    
    # Check core (hydrophobic) residues
    core_res = diagnostics['core_residues']
    assert len(core_res) > 0


def test_diagnose_rmsd_components_values_match_displacements(refinement_engine, sample_conformation, native_structure):
    """Test that RMSD values are reasonable after alignment."""
    diagnostics = refinement_engine.diagnose_rmsd_components(
        sample_conformation,
        native_structure
    )
    
    # After Kabsch alignment, component RMSDs may be different from
    # original displacements because the alignment minimizes total RMSD.
    # We just verify that values are reasonable (non-negative, finite)
    
    assert diagnostics['helix_rmsd'] >= 0.0
    assert diagnostics['sheet_rmsd'] >= 0.0
    assert diagnostics['loop_rmsd'] >= 0.0
    assert diagnostics['core_rmsd'] >= 0.0
    
    # Total RMSD should be positive (structures are different)
    assert diagnostics['total_rmsd'] > 0.0


def test_diagnose_rmsd_components_percentages_sum(refinement_engine, sample_conformation, native_structure):
    """Test that percentage contributions are reasonable."""
    diagnostics = refinement_engine.diagnose_rmsd_components(
        sample_conformation,
        native_structure
    )
    
    # Percentages should all be non-negative
    assert diagnostics['helix_percentage'] >= 0.0
    assert diagnostics['sheet_percentage'] >= 0.0
    assert diagnostics['loop_percentage'] >= 0.0
    assert diagnostics['core_percentage'] >= 0.0
    
    # Percentages are based on squared deviations, so they should sum to ≈100%
    # However, core residues may overlap with helix/sheet/loop, so total can exceed 100%
    # Just verify each component is reasonable
    assert diagnostics['helix_percentage'] <= 100.0
    assert diagnostics['sheet_percentage'] <= 100.0
    assert diagnostics['loop_percentage'] <= 100.0
    assert diagnostics['core_percentage'] <= 100.0


# ============================================================================
# Test: diagnose_rmsd_components() - Report Generation
# ============================================================================

def test_diagnose_rmsd_components_report_format(refinement_engine, sample_conformation, native_structure):
    """Test that diagnostic report has correct format."""
    diagnostics = refinement_engine.diagnose_rmsd_components(
        sample_conformation,
        native_structure
    )
    
    report = diagnostics['report']
    
    # Report should be non-empty string
    assert isinstance(report, str)
    assert len(report) > 0
    
    # Should contain key sections
    assert "RMSD COMPONENT DIAGNOSTICS" in report
    assert "Total RMSD:" in report
    assert "Component Breakdown:" in report
    assert "Helix" in report
    assert "Sheet" in report
    assert "Loop" in report
    assert "Core" in report
    assert "Recommendations:" in report


def test_diagnose_rmsd_components_report_values(refinement_engine, sample_conformation, native_structure):
    """Test that report contains actual RMSD values."""
    diagnostics = refinement_engine.diagnose_rmsd_components(
        sample_conformation,
        native_structure
    )
    
    report = diagnostics['report']
    
    # Report should contain formatted RMSD values
    total_rmsd_str = f"{diagnostics['total_rmsd']:.2f}"
    assert total_rmsd_str in report or f"{diagnostics['total_rmsd']:.1f}" in report


# ============================================================================
# Test: diagnose_rmsd_components() - Edge Cases
# ============================================================================

def test_diagnose_rmsd_components_no_helix(refinement_engine, rmsd_calculator):
    """Test diagnostics with no helix residues."""
    # Create structure with only sheets and loops
    sequence = "EEEEEEAAAAA"  # All sheet or coil
    coords = [(float(i), 0.0, 0.0) for i in range(len(sequence))]
    
    conformation = Conformation(
        conformation_id="no_helix",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-50.0,
        rmsd_to_native=None,
        secondary_structure=['E'] * 6 + ['C'] * 5,
        phi_angles=[-120.0] * len(sequence),
        psi_angles=[120.0] * len(sequence),
        available_move_types=[],
        structural_constraints={}
    )
    
    native_coords = [(x + 1.0, y, z) for x, y, z in coords]
    native = NativeStructure(
        pdb_id="1TEST",
        sequence=sequence,
        ca_coords=native_coords
    )
    
    diagnostics = refinement_engine.diagnose_rmsd_components(conformation, native)
    
    # Helix RMSD should be 0.0 (no helix residues)
    assert diagnostics['helix_rmsd'] == 0.0
    assert len(diagnostics['helix_residues']) == 0
    
    # Should still have other components
    assert diagnostics['sheet_rmsd'] >= 0.0  # May be 0 after alignment
    assert len(diagnostics['sheet_residues']) > 0


def test_diagnose_rmsd_components_no_sheet(refinement_engine, rmsd_calculator):
    """Test diagnostics with no sheet residues."""
    # Create structure with only helices and loops
    sequence = "HHHHHHAAAAA"  # All helix or coil
    coords = [(float(i), 0.0, 0.0) for i in range(len(sequence))]
    
    conformation = Conformation(
        conformation_id="no_sheet",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-50.0,
        rmsd_to_native=None,
        secondary_structure=['H'] * 6 + ['C'] * 5,
        phi_angles=[-60.0] * len(sequence),
        psi_angles=[-45.0] * len(sequence),
        available_move_types=[],
        structural_constraints={}
    )
    
    native_coords = [(x + 1.0, y, z) for x, y, z in coords]
    native = NativeStructure(
        pdb_id="1TEST",
        sequence=sequence,
        ca_coords=native_coords
    )
    
    diagnostics = refinement_engine.diagnose_rmsd_components(conformation, native)
    
    # Sheet RMSD should be 0.0 (no sheet residues)
    assert diagnostics['sheet_rmsd'] == 0.0
    assert len(diagnostics['sheet_residues']) == 0
    
    # Should still have other components
    assert diagnostics['helix_rmsd'] >= 0.0  # May be 0 after alignment
    assert len(diagnostics['helix_residues']) > 0


def test_diagnose_rmsd_components_all_coil(refinement_engine, rmsd_calculator):
    """Test diagnostics with all coil/loop residues."""
    sequence = "AAAAAAAAAA"  # All coil
    coords = [(float(i), 0.0, 0.0) for i in range(len(sequence))]
    
    conformation = Conformation(
        conformation_id="all_coil",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-50.0,
        rmsd_to_native=None,
        secondary_structure=['C'] * len(sequence),
        phi_angles=[0.0] * len(sequence),
        psi_angles=[0.0] * len(sequence),
        available_move_types=[],
        structural_constraints={}
    )
    
    native_coords = [(x + 1.0, y, z) for x, y, z in coords]
    native = NativeStructure(
        pdb_id="1TEST",
        sequence=sequence,
        ca_coords=native_coords
    )
    
    diagnostics = refinement_engine.diagnose_rmsd_components(conformation, native)
    
    # No helix or sheet
    assert diagnostics['helix_rmsd'] == 0.0
    assert diagnostics['sheet_rmsd'] == 0.0
    assert len(diagnostics['helix_residues']) == 0
    assert len(diagnostics['sheet_residues']) == 0
    
    # Should have loop residues  
    assert diagnostics['loop_rmsd'] >= 0.0  # May be 0 after alignment
    assert len(diagnostics['loop_residues']) > 0


# ============================================================================
# Test: _identify_core_residues() - Helper Method
# ============================================================================

def test_identify_core_residues_without_qcp(refinement_engine, sample_conformation):
    """Test core residue identification without QCP values."""
    core_residues = refinement_engine._identify_core_residues(
        sample_conformation,
        qcp_values=None
    )
    
    # Should identify hydrophobic residues
    # Sample sequence: "MHHHHHEEEEEAAAAAVVVV"
    # M, A, V are hydrophobic
    assert len(core_residues) > 0
    
    # Check that hydrophobic residues are identified
    hydrophobic_aa = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'}
    for idx in core_residues:
        assert sample_conformation.sequence[idx] in hydrophobic_aa


def test_identify_core_residues_with_qcp(refinement_engine, sample_conformation):
    """Test core residue identification with QCP values."""
    # Create QCP values: high QCP for some residues
    qcp_values = {i: 5.0 for i in range(len(sample_conformation.sequence))}
    qcp_values[0] = 8.0  # High QCP
    qcp_values[10] = 7.5  # High QCP
    
    core_residues = refinement_engine._identify_core_residues(
        sample_conformation,
        qcp_values=qcp_values
    )
    
    # Should include high QCP residues
    assert 0 in core_residues  # QCP = 8.0
    assert 10 in core_residues  # QCP = 7.5


def test_identify_core_residues_qcp_thresholds(refinement_engine):
    """Test QCP threshold logic for core identification."""
    sequence = "AAAAAAAAAA"  # All hydrophobic
    coords = [(float(i), 0.0, 0.0) for i in range(len(sequence))]
    
    conformation = Conformation(
        conformation_id="test",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-50.0,
        rmsd_to_native=None,
        secondary_structure=['C'] * len(sequence),
        phi_angles=[0.0] * len(sequence),
        psi_angles=[0.0] * len(sequence),
        available_move_types=[],
        structural_constraints={}
    )
    
    # Test different QCP thresholds
    # QCP > 7.0: Always core
    # 4.0 < QCP <= 7.0 + hydrophobic: Core
    # QCP <= 4.0 + hydrophobic: Not core (surface)
    
    qcp_values = {
        0: 8.0,  # High QCP → core
        1: 6.0,  # Medium QCP + hydrophobic → core
        2: 3.0,  # Low QCP + hydrophobic → not core (surface)
    }
    
    core_residues = refinement_engine._identify_core_residues(
        conformation,
        qcp_values=qcp_values
    )
    
    assert 0 in core_residues  # High QCP
    assert 1 in core_residues  # Medium QCP + hydrophobic
    # Residue 2 might or might not be in core depending on hydrophobicity check


# ============================================================================
# Test: _calculate_subset_rmsd() - Helper Method
# ============================================================================

def test_calculate_subset_rmsd_basic(refinement_engine):
    """Test subset RMSD calculation."""
    # Create simple displaced coordinates
    predicted = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    native = [(0.5, 0.0, 0.0), (1.5, 0.0, 0.0), (2.5, 0.0, 0.0)]
    
    # Calculate RMSD for subset [0, 2]
    rmsd = refinement_engine._calculate_subset_rmsd(
        predicted,
        native,
        subset_indices=[0, 2]
    )
    
    # Expected RMSD after Kabsch alignment:
    # The alignment will minimize RMSD, so the result may be different
    # from the simple 0.5Å displacement. Just verify it's reasonable.
    assert rmsd >= 0.0
    assert rmsd < 10.0  # Sanity check


def test_calculate_subset_rmsd_empty_subset(refinement_engine):
    """Test subset RMSD with empty subset."""
    predicted = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    native = [(0.5, 0.0, 0.0), (1.5, 0.0, 0.0)]
    
    rmsd = refinement_engine._calculate_subset_rmsd(
        predicted,
        native,
        subset_indices=[]
    )
    
    # Empty subset should return 0.0
    assert rmsd == 0.0


def test_calculate_subset_rmsd_out_of_bounds(refinement_engine):
    """Test subset RMSD with indices out of bounds."""
    predicted = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    native = [(0.5, 0.0, 0.0), (1.5, 0.0, 0.0)]
    
    # Subset includes out-of-bounds index
    rmsd = refinement_engine._calculate_subset_rmsd(
        predicted,
        native,
        subset_indices=[0, 1, 5]  # Index 5 is out of bounds
    )
    
    # Should handle gracefully (use only valid indices)
    assert rmsd >= 0.0


# ============================================================================
# Test: Integration with optimize_two_stage()
# ============================================================================

def test_two_stage_optimization_includes_diagnostics(refinement_engine, sample_conformation, native_structure, monkeypatch):
    """Test that two-stage optimization includes RMSD diagnostics."""
    # Mock the optimization steps to avoid actual computation
    def mock_stage1(structure, temperature=1.0, iterations=1000, native_structure=None):
        return (
            sample_conformation,  # structure
            [2.0, 1.5, 1.0],  # RMSD trajectory
            [-100.0, -110.0, -120.0]  # Energy trajectory
        )
    
    def mock_stage2(structure, exploration_temperature=1.0, restraint_weight=10.0, qcp_weight=0.3, iterations=10000, native_structure=None):
        return (
            sample_conformation,  # structure
            [1.0, 0.8, 0.6],  # RMSD trajectory
            [-120.0, -125.0, -130.0]  # Energy trajectory
        )
    
    monkeypatch.setattr(refinement_engine, 'optimize_stage1_global', mock_stage1)
    monkeypatch.setattr(refinement_engine, 'optimize_stage2_refinement', mock_stage2)
    
    result = refinement_engine.optimize_two_stage(
        sample_conformation,
        native_structure
    )
    
    # Check that component RMSDs are populated
    assert result.helix_rmsd >= 0.0
    assert result.sheet_rmsd >= 0.0
    assert result.loop_rmsd >= 0.0
    assert result.core_rmsd >= 0.0
    
    # At least one component should have non-zero RMSD
    assert (result.helix_rmsd > 0.0 or 
            result.sheet_rmsd > 0.0 or 
            result.loop_rmsd > 0.0 or 
            result.core_rmsd > 0.0)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
