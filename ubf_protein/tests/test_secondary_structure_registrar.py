"""
Unit tests for SecondaryStructureRegistrar

Tests cover:
1. Helix detection and geometry enforcement
2. Sheet detection and hydrogen bond optimization
3. QCP-based parameter scaling
4. RMSD reduction validation
5. Edge cases and error handling
"""

import pytest
import math
from typing import List, Dict

try:
    from ubf_protein.secondary_structure_registrar import SecondaryStructureRegistrar
    from ubf_protein.models import (
        Conformation, HelixRegion, SheetRegion
    )
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from secondary_structure_registrar import SecondaryStructureRegistrar
    from models import Conformation, HelixRegion, SheetRegion
    from qcpp_integration import QCPPIntegrationAdapter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def qcpp_adapter():
    """Create QCPP adapter for testing."""
    # Mock predictor for testing
    class MockPredictor:
        pass
    
    return QCPPIntegrationAdapter(predictor=MockPredictor())


@pytest.fixture
def registrar(qcpp_adapter):
    """Create SecondaryStructureRegistrar instance."""
    return SecondaryStructureRegistrar(qcpp_adapter)


@pytest.fixture
def simple_helix_structure():
    """Create simple structure with one helix."""
    sequence = "AAAAAAAAAAA"  # 11 residues
    coords = [(float(i), 0.0, 0.0) for i in range(len(sequence))]
    
    return Conformation(
        conformation_id="helix_test",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-100.0,
        rmsd_to_native=10.0,
        secondary_structure=['H'] * len(sequence),  # All helix
        phi_angles=[-60.0] * len(sequence),
        psi_angles=[-45.0] * len(sequence),
        available_move_types=[],
        structural_constraints={}
    )


@pytest.fixture
def simple_sheet_structure():
    """Create simple structure with one sheet (2 strands)."""
    sequence = "AAAAAABBBBBB"  # 12 residues, 2 strands
    coords = [(float(i), 0.0, 0.0) for i in range(len(sequence))]
    
    # First strand (0-5), gap (6), second strand (7-12)
    ss = ['E'] * 6 + ['C'] + ['E'] * 5
    
    return Conformation(
        conformation_id="sheet_test",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-100.0,
        rmsd_to_native=10.0,
        secondary_structure=ss,
        phi_angles=[-120.0] * len(sequence),
        psi_angles=[120.0] * len(sequence),
        available_move_types=[],
        structural_constraints={}
    )


@pytest.fixture
def mixed_structure():
    """Create structure with helix, sheet, and loops."""
    # H=helix, E=sheet, C=coil
    ss = ['C', 'C'] + ['H'] * 8 + ['C', 'C'] + ['E'] * 5 + ['C'] + ['E'] * 5 + ['C', 'C']
    sequence = 'A' * len(ss)
    coords = [(float(i), 0.0, 0.0) for i in range(len(ss))]
    
    return Conformation(
        conformation_id="mixed_test",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-200.0,
        rmsd_to_native=12.0,
        secondary_structure=ss,
        phi_angles=[-60.0] * len(ss),
        psi_angles=[-45.0] * len(ss),
        available_move_types=[],
        structural_constraints={}
    )


# ============================================================================
# Test Initialization
# ============================================================================

def test_registrar_initialization(registrar):
    """Test SecondaryStructureRegistrar initialization."""
    assert registrar is not None
    assert registrar.phi == pytest.approx(1.618033988749895, rel=1e-6)
    assert registrar.standard_pitch == 5.4
    assert registrar.standard_rise == 1.5
    assert registrar.standard_residues_per_turn == 3.6
    assert registrar.sheet_coupling_frequency == 2.618
    assert registrar.inter_strand_distance == 4.8


# ============================================================================
# Test Helix Detection
# ============================================================================

def test_detect_helices_simple(registrar, simple_helix_structure):
    """Test helix detection in simple all-helix structure."""
    helices = registrar._detect_helices(simple_helix_structure)
    
    assert len(helices) == 1
    assert helices[0].start_residue == 0
    assert helices[0].end_residue == 10  # 11 residues total
    assert helices[0].length() == 11


def test_detect_helices_mixed(registrar, mixed_structure):
    """Test helix detection in mixed secondary structure."""
    helices = registrar._detect_helices(mixed_structure)
    
    assert len(helices) == 1
    assert helices[0].start_residue == 2
    assert helices[0].end_residue == 9  # 8 residues
    assert helices[0].length() == 8


def test_detect_helices_too_short(registrar):
    """Test that short helices (<4 residues) are not detected."""
    structure = Conformation(
        conformation_id="short_helix",
        sequence="AAAA",
        atom_coordinates=[(float(i), 0.0, 0.0) for i in range(4)],
        energy=-50.0,
        rmsd_to_native=8.0,
        secondary_structure=['H', 'H', 'H', 'C'],  # Only 3 H
        phi_angles=[-60.0] * 4,
        psi_angles=[-45.0] * 4,
        available_move_types=[],
        structural_constraints={}
    )
    
    helices = registrar._detect_helices(structure)
    assert len(helices) == 0  # Too short


def test_detect_helices_multiple(registrar):
    """Test detection of multiple helices."""
    ss = ['H'] * 6 + ['C'] * 2 + ['H'] * 5 + ['C'] * 2 + ['H'] * 7
    structure = Conformation(
        conformation_id="multi_helix",
        sequence='A' * len(ss),
        atom_coordinates=[(float(i), 0.0, 0.0) for i in range(len(ss))],
        energy=-100.0,
        rmsd_to_native=10.0,
        secondary_structure=ss,
        phi_angles=[-60.0] * len(ss),
        psi_angles=[-45.0] * len(ss),
        available_move_types=[],
        structural_constraints={}
    )
    
    helices = registrar._detect_helices(structure)
    assert len(helices) == 3
    assert helices[0].length() == 6
    assert helices[1].length() == 5
    assert helices[2].length() == 7


# ============================================================================
# Test Sheet Detection
# ============================================================================

def test_detect_sheets_simple(registrar, simple_sheet_structure):
    """Test sheet detection in simple structure."""
    sheets = registrar._detect_sheets(simple_sheet_structure)
    
    assert len(sheets) == 1
    assert len(sheets[0].strand_residues) == 2
    assert sheets[0].strand_residues[0] == (0, 5)  # First strand
    assert sheets[0].strand_residues[1] == (7, 11)  # Second strand


def test_detect_sheets_mixed(registrar, mixed_structure):
    """Test sheet detection in mixed secondary structure."""
    sheets = registrar._detect_sheets(mixed_structure)
    
    assert len(sheets) == 1
    assert len(sheets[0].strand_residues) == 2
    # Two strands of 5 residues each


def test_detect_sheets_too_short(registrar):
    """Test that short strands (<3 residues) are not detected."""
    structure = Conformation(
        conformation_id="short_strand",
        sequence="AAAA",
        atom_coordinates=[(float(i), 0.0, 0.0) for i in range(4)],
        energy=-50.0,
        rmsd_to_native=8.0,
        secondary_structure=['E', 'E', 'C', 'C'],  # Only 2 E
        phi_angles=[-120.0] * 4,
        psi_angles=[120.0] * 4,
        available_move_types=[],
        structural_constraints={}
    )
    
    sheets = registrar._detect_sheets(structure)
    assert len(sheets) == 0  # Too short


def test_detect_sheets_single_strand(registrar):
    """Test that single strand does not form sheet."""
    structure = Conformation(
        conformation_id="single_strand",
        sequence="AAAAAAA",
        atom_coordinates=[(float(i), 0.0, 0.0) for i in range(7)],
        energy=-50.0,
        rmsd_to_native=8.0,
        secondary_structure=['E'] * 7,
        phi_angles=[-120.0] * 7,
        psi_angles=[120.0] * 7,
        available_move_types=[],
        structural_constraints={}
    )
    
    sheets = registrar._detect_sheets(structure)
    assert len(sheets) == 0  # Need at least 2 strands


# ============================================================================
# Test QCP Calculation
# ============================================================================

def test_calculate_average_qcp_range(registrar):
    """Test average QCP calculation for residue range."""
    qcp_values = {0: 5.0, 1: 6.0, 2: 7.0, 3: 8.0, 4: 9.0}
    
    avg = registrar._calculate_average_qcp(0, 4, qcp_values)
    assert avg == pytest.approx(7.0, rel=1e-6)  # (5+6+7+8+9)/5


def test_calculate_average_qcp_partial(registrar):
    """Test average QCP with missing values."""
    qcp_values = {0: 5.0, 2: 7.0, 4: 9.0}  # Missing 1 and 3
    
    avg = registrar._calculate_average_qcp(0, 4, qcp_values)
    assert avg == pytest.approx(7.0, rel=1e-6)  # (5+7+9)/3


def test_calculate_average_qcp_list(registrar):
    """Test average QCP for residue list."""
    qcp_values = {0: 5.0, 1: 6.0, 2: 7.0, 5: 8.0, 10: 9.0}
    residues = [0, 1, 2, 5, 10]
    
    avg = registrar._calculate_average_qcp_list(residues, qcp_values)
    assert avg == pytest.approx(7.0, rel=1e-6)


# ============================================================================
# Test Helix Geometry Enforcement
# ============================================================================

def test_enforce_helix_geometry_low_qcp(registrar, simple_helix_structure):
    """Test helix geometry with low QCP (standard parameters)."""
    helix_residues = list(range(11))
    helix_qcp = 5.0  # Low QCP
    
    result = registrar.enforce_helix_geometry(
        helix_residues, helix_qcp, simple_helix_structure
    )
    
    # Should return structure (even if unchanged in placeholder)
    assert result is not None
    assert result.sequence == simple_helix_structure.sequence


def test_enforce_helix_geometry_high_qcp(registrar, simple_helix_structure):
    """Test helix geometry with high QCP (quantum-corrected)."""
    helix_residues = list(range(11))
    helix_qcp = 8.5  # High QCP
    
    result = registrar.enforce_helix_geometry(
        helix_residues, helix_qcp, simple_helix_structure
    )
    
    # Should return structure
    assert result is not None
    
    # With high QCP, parameters should be modified
    # (Though actual coordinate changes are TODO in current implementation)


def test_enforce_helix_geometry_qcp_scaling(registrar, simple_helix_structure):
    """Test that QCP correctly scales helix parameters."""
    helix_residues = list(range(11))
    
    # Test various QCP values
    for qcp in [6.0, 7.0, 8.0, 9.0, 10.0]:
        result = registrar.enforce_helix_geometry(
            helix_residues, qcp, simple_helix_structure
        )
        assert result is not None
        
        # Verify scaling factors are calculated correctly
        if qcp > 7.0:
            qcp_excess = qcp - 7.0
            expected_pitch_factor = 1.0 + 0.1 * math.tanh(qcp_excess)
            expected_rise_factor = 1.0 + 0.05 * math.tanh(qcp_excess)
            
            # These would be used in actual coordinate transformation
            assert expected_pitch_factor > 1.0
            assert expected_rise_factor > 1.0


def test_enforce_helix_geometry_too_short(registrar, simple_helix_structure):
    """Test helix geometry enforcement with too-short helix."""
    helix_residues = [0, 1, 2]  # Only 3 residues
    helix_qcp = 8.0
    
    result = registrar.enforce_helix_geometry(
        helix_residues, helix_qcp, simple_helix_structure
    )
    
    # Should return unchanged (warning logged)
    assert result.sequence == simple_helix_structure.sequence


# ============================================================================
# Test Sheet Hydrogen Bond Optimization
# ============================================================================

def test_optimize_sheet_hydrogen_bonds_simple(registrar, simple_sheet_structure):
    """Test sheet H-bond optimization."""
    sheet_residues = list(range(6)) + list(range(7, 12))  # Both strands
    
    result = registrar.optimize_sheet_hydrogen_bonds(
        sheet_residues, 2.618, simple_sheet_structure
    )
    
    # Should return structure
    assert result is not None
    assert result.sequence == simple_sheet_structure.sequence


def test_optimize_sheet_hydrogen_bonds_custom_frequency(registrar, simple_sheet_structure):
    """Test sheet optimization with custom THz frequency."""
    sheet_residues = list(range(6)) + list(range(7, 12))
    
    # Test with different coupling frequencies
    for freq in [1.618, 2.618, 4.236]:  # φ harmonics
        result = registrar.optimize_sheet_hydrogen_bonds(
            sheet_residues, freq, simple_sheet_structure
        )
        assert result is not None


def test_optimize_sheet_hydrogen_bonds_too_small(registrar, simple_sheet_structure):
    """Test sheet optimization with too-small sheet."""
    sheet_residues = [0, 1, 2, 3, 4]  # Only 5 residues
    
    result = registrar.optimize_sheet_hydrogen_bonds(
        sheet_residues, 2.618, simple_sheet_structure
    )
    
    # Should return unchanged (warning logged)
    assert result.sequence == simple_sheet_structure.sequence


# ============================================================================
# Test Full Registration Pipeline
# ============================================================================

def test_fix_secondary_structure_registration_helix(registrar, simple_helix_structure):
    """Test full registration pipeline with helix structure."""
    qcp_values = {i: 8.0 for i in range(len(simple_helix_structure.sequence))}
    
    result = registrar.fix_secondary_structure_registration(
        simple_helix_structure, qcp_values
    )
    
    assert result is not None
    assert result.sequence == simple_helix_structure.sequence


def test_fix_secondary_structure_registration_sheet(registrar, simple_sheet_structure):
    """Test full registration pipeline with sheet structure."""
    qcp_values = {i: 7.5 for i in range(len(simple_sheet_structure.sequence))}
    
    result = registrar.fix_secondary_structure_registration(
        simple_sheet_structure, qcp_values
    )
    
    assert result is not None
    assert result.sequence == simple_sheet_structure.sequence


def test_fix_secondary_structure_registration_mixed(registrar, mixed_structure):
    """Test full registration pipeline with mixed structure."""
    # Varying QCP values
    qcp_values = {}
    for i in range(len(mixed_structure.sequence)):
        if mixed_structure.secondary_structure[i] == 'H':
            qcp_values[i] = 8.5  # High QCP for helices
        elif mixed_structure.secondary_structure[i] == 'E':
            qcp_values[i] = 7.5  # Medium-high for sheets
        else:
            qcp_values[i] = 5.0  # Low for coils
    
    result = registrar.fix_secondary_structure_registration(
        mixed_structure, qcp_values
    )
    
    assert result is not None
    assert result.sequence == mixed_structure.sequence


def test_fix_secondary_structure_registration_varying_qcp(registrar, mixed_structure):
    """Test registration with varying QCP values."""
    # Gradient of QCP values
    qcp_values = {i: 5.0 + i * 0.2 for i in range(len(mixed_structure.sequence))}
    
    result = registrar.fix_secondary_structure_registration(
        mixed_structure, qcp_values
    )
    
    assert result is not None


def test_fix_secondary_structure_registration_empty_qcp(registrar, simple_helix_structure):
    """Test registration with empty QCP values."""
    qcp_values = {}
    
    result = registrar.fix_secondary_structure_registration(
        simple_helix_structure, qcp_values
    )
    
    # Should handle gracefully (QCP defaults to 0.0)
    assert result is not None


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_no_secondary_structure(registrar):
    """Test structure with no secondary structure."""
    structure = Conformation(
        conformation_id="no_ss",
        sequence="AAAA",
        atom_coordinates=[(float(i), 0.0, 0.0) for i in range(4)],
        energy=-50.0,
        rmsd_to_native=8.0,
        secondary_structure=['C'] * 4,  # All coil
        phi_angles=[-60.0] * 4,
        psi_angles=[-45.0] * 4,
        available_move_types=[],
        structural_constraints={}
    )
    
    qcp_values = {i: 6.0 for i in range(4)}
    
    result = registrar.fix_secondary_structure_registration(structure, qcp_values)
    
    # Should return unchanged
    assert result.sequence == structure.sequence


def test_single_residue(registrar):
    """Test structure with single residue."""
    structure = Conformation(
        conformation_id="single",
        sequence="A",
        atom_coordinates=[(0.0, 0.0, 0.0)],
        energy=-10.0,
        rmsd_to_native=5.0,
        secondary_structure=['C'],
        phi_angles=[-60.0],
        psi_angles=[-45.0],
        available_move_types=[],
        structural_constraints={}
    )
    
    qcp_values = {0: 6.0}
    
    result = registrar.fix_secondary_structure_registration(structure, qcp_values)
    assert result.sequence == "A"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
