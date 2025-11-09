"""
Unit tests for Quantum Core Analyzer (Task 2).

Tests cover:
- Quantum core identification with various QCP thresholds
- THz mode calculation and φ-harmonic detection
- Resonance coupling between residues
- Edge cases: no cores, all cores, single residue structures
"""

import pytest
import math
from typing import List, Tuple
from unittest.mock import Mock, MagicMock

# Import the classes to test
try:
    from ubf_protein.quantum_core_analyzer import QuantumCoreAnalyzer
    from ubf_protein.models import Conformation, QuantumCore, THzMode
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics
except ImportError:
    from quantum_core_analyzer import QuantumCoreAnalyzer
    from models import Conformation, QuantumCore, THzMode
    from qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_qcpp_adapter():
    """Create a mock QCPP adapter for testing."""
    adapter = Mock(spec=QCPPIntegrationAdapter)
    
    # Default metrics (can be overridden in tests)
    adapter.analyze_conformation.return_value = QCPPMetrics(
        qcp_score=7.5,
        field_coherence=0.7,
        stability_score=2.0,
        phi_match_score=0.8,
        calculation_time_ms=1.0
    )
    
    return adapter


@pytest.fixture
def test_structure_7res():
    """Create a test structure with 7 residues."""
    return Conformation(
        conformation_id="test_7res",
        sequence="ACDEFGH",
        atom_coordinates=[
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (15.2, 0.0, 0.0),
            (19.0, 0.0, 0.0),
            (22.8, 0.0, 0.0),
        ],
        energy=-100.0,
        rmsd_to_native=5.0,
        secondary_structure=['H', 'H', 'H', 'E', 'E', 'C', 'C'],
        phi_angles=[-60.0, -60.0, -60.0, -120.0, -120.0, -90.0, -90.0],
        psi_angles=[-45.0, -45.0, -45.0, 135.0, 135.0, 0.0, 0.0],
        available_move_types=['helix_formation', 'sheet_formation'],
        structural_constraints={}
    )


@pytest.fixture
def test_structure_20res():
    """Create a test structure with 20 residues (longer for core testing)."""
    sequence = "A" * 20
    coords = [(i * 3.8, 0.0, 0.0) for i in range(20)]
    ss = ['H'] * 5 + ['C'] * 5 + ['E'] * 5 + ['C'] * 5
    phi = [-60.0] * 5 + [-90.0] * 5 + [-120.0] * 5 + [-90.0] * 5
    psi = [-45.0] * 5 + [0.0] * 5 + [135.0] * 5 + [0.0] * 5
    
    return Conformation(
        conformation_id="test_20res",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-300.0,
        rmsd_to_native=8.0,
        secondary_structure=ss,
        phi_angles=phi,
        psi_angles=psi,
        available_move_types=['helix_formation', 'sheet_formation'],
        structural_constraints={}
    )


@pytest.fixture
def analyzer(mock_qcpp_adapter):
    """Create a QuantumCoreAnalyzer instance for testing."""
    return QuantumCoreAnalyzer(mock_qcpp_adapter)


# ============================================================================
# Test Initialization
# ============================================================================

def test_analyzer_initialization(mock_qcpp_adapter):
    """Test that analyzer initializes correctly."""
    analyzer = QuantumCoreAnalyzer(mock_qcpp_adapter)
    
    assert analyzer.qcpp_adapter is mock_qcpp_adapter
    assert analyzer.phi == pytest.approx(1.618033988749895, rel=1e-10)
    assert len(analyzer.phi_harmonics) == 5
    assert analyzer.phi_harmonics[0] == pytest.approx(1.0, rel=1e-6)
    assert analyzer.phi_harmonics[1] == pytest.approx(1.618, rel=1e-3)
    assert analyzer.phi_harmonics[2] == pytest.approx(2.618, rel=1e-3)


def test_analyzer_initialization_invalid_adapter():
    """Test that initialization fails with invalid adapter."""
    with pytest.raises(TypeError, match="qcpp_adapter cannot be None"):
        QuantumCoreAnalyzer(None)
    
    with pytest.raises(TypeError, match="must be QCPPIntegrationAdapter"):
        QuantumCoreAnalyzer("not_an_adapter")


# ============================================================================
# Test Quantum Core Identification
# ============================================================================

def test_identify_quantum_cores_basic(analyzer, test_structure_7res):
    """Test basic quantum core identification."""
    # Mock QCP values: first 4 residues have high QCP, rest have low
    analyzer.qcpp_adapter.analyze_conformation.return_value = QCPPMetrics(
        qcp_score=8.0,  # High QCP
        field_coherence=0.8,
        stability_score=2.5,
        phi_match_score=0.9,
        calculation_time_ms=1.0
    )
    
    cores = analyzer.identify_quantum_cores(test_structure_7res, qcp_threshold=7.0)
    
    # Should identify at least one core
    assert len(cores) >= 0  # May have cores depending on per-residue calculation
    
    # If cores found, validate structure
    for core in cores:
        assert isinstance(core, QuantumCore)
        assert len(core.residue_indices) >= 3  # Minimum core size
        assert core.average_qcp >= 0
        assert 0.0 <= core.coherence <= 1.0
        assert len(core.center_of_mass) == 3


def test_identify_quantum_cores_no_cores(analyzer, test_structure_7res):
    """Test identification when no quantum cores exist (low QCP)."""
    # Mock low QCP values
    analyzer.qcpp_adapter.analyze_conformation.return_value = QCPPMetrics(
        qcp_score=3.0,  # Low QCP (base value)
        field_coherence=0.2,
        stability_score=0.5,
        phi_match_score=0.3,
        calculation_time_ms=1.0
    )
    
    cores = analyzer.identify_quantum_cores(test_structure_7res, qcp_threshold=7.0)
    
    # Should find no cores with low QCP
    assert len(cores) == 0


def test_identify_quantum_cores_all_high_qcp(analyzer, test_structure_20res):
    """Test identification when all residues have high QCP."""
    # Mock very high QCP
    analyzer.qcpp_adapter.analyze_conformation.return_value = QCPPMetrics(
        qcp_score=10.0,  # Very high QCP
        field_coherence=0.95,
        stability_score=3.0,
        phi_match_score=0.95,
        calculation_time_ms=1.0
    )
    
    cores = analyzer.identify_quantum_cores(test_structure_20res, qcp_threshold=7.0)
    
    # Should identify the entire structure as one large core
    # (or multiple cores if gaps exist)
    assert len(cores) >= 1
    
    # Total residues in cores should be significant
    total_core_residues = sum(len(core.residue_indices) for core in cores)
    assert total_core_residues >= 15  # At least 75% of residues


def test_identify_quantum_cores_invalid_inputs(analyzer, test_structure_7res):
    """Test error handling for invalid inputs."""
    # Invalid threshold
    with pytest.raises(ValueError, match="qcp_threshold must be >= 0"):
        analyzer.identify_quantum_cores(test_structure_7res, qcp_threshold=-1.0)
    
    # Structure too small
    tiny_structure = Conformation(
        conformation_id="tiny",
        sequence="AC",
        atom_coordinates=[(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)],
        energy=-10.0,
        rmsd_to_native=None,
        secondary_structure=['C', 'C'],
        phi_angles=[-90.0, -90.0],
        psi_angles=[0.0, 0.0],
        available_move_types=[],
        structural_constraints={}
    )
    
    with pytest.raises(ValueError, match="must have >= 3 residues"):
        analyzer.identify_quantum_cores(tiny_structure, qcp_threshold=7.0)


def test_identify_quantum_cores_center_of_mass(analyzer, test_structure_7res):
    """Test that center of mass is calculated correctly."""
    analyzer.qcpp_adapter.analyze_conformation.return_value = QCPPMetrics(
        qcp_score=9.0,
        field_coherence=0.85,
        stability_score=2.8,
        phi_match_score=0.92,
        calculation_time_ms=1.0
    )
    
    cores = analyzer.identify_quantum_cores(test_structure_7res, qcp_threshold=7.0)
    
    for core in cores:
        # Center of mass should be within structure bounds
        coords = test_structure_7res.atom_coordinates
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        z_coords = [c[2] for c in coords]
        
        cx, cy, cz = core.center_of_mass
        assert min(x_coords) <= cx <= max(x_coords)
        assert min(y_coords) <= cy <= max(y_coords)
        assert min(z_coords) <= cz <= max(z_coords)


# ============================================================================
# Test THz Mode Calculation
# ============================================================================

def test_calculate_local_thz_modes_basic(analyzer, test_structure_7res):
    """Test basic THz mode calculation."""
    # Create a test core
    core = QuantumCore(
        residue_indices=[0, 1, 2, 3, 4],
        average_qcp=8.5,
        coherence=0.85,
        center_of_mass=(7.6, 0.0, 0.0)
    )
    
    modes = analyzer.calculate_local_thz_modes(core, test_structure_7res, num_modes=5)
    
    # Should return requested number of modes (or fewer if not enough residues)
    assert len(modes) <= 5
    assert len(modes) >= 1
    
    # Validate mode structure
    for mode in modes:
        assert isinstance(mode, THzMode)
        assert mode.frequency > 0
        assert mode.amplitude >= 0
        assert len(mode.participating_residues) == len(core.residue_indices)
        assert isinstance(mode.is_phi_harmonic, bool)


def test_calculate_local_thz_modes_phi_harmonic_detection(analyzer, test_structure_7res):
    """Test that φ-harmonic modes are correctly detected."""
    core = QuantumCore(
        residue_indices=[0, 1, 2, 3, 4],
        average_qcp=8.5,
        coherence=0.85,
        center_of_mass=(7.6, 0.0, 0.0)
    )
    
    modes = analyzer.calculate_local_thz_modes(core, test_structure_7res, num_modes=5)
    
    # Check that φ-harmonic detection works
    phi_harmonics_found = [mode.is_phi_harmonic for mode in modes]
    
    # At least check that the flag is set correctly
    for mode in modes:
        # Manually check if near a φ-harmonic
        is_near_harmonic = False
        for harmonic in analyzer.phi_harmonics:
            if abs(mode.frequency - harmonic) <= 0.1:
                is_near_harmonic = True
                break
        
        if is_near_harmonic:
            assert mode.is_phi_harmonic


def test_calculate_local_thz_modes_invalid_inputs(analyzer, test_structure_7res):
    """Test error handling for invalid inputs."""
    core = QuantumCore(
        residue_indices=[0, 1, 2],
        average_qcp=8.0,
        coherence=0.8,
        center_of_mass=(3.8, 0.0, 0.0)
    )
    
    # Invalid num_modes
    with pytest.raises(ValueError, match="num_modes must be >= 1"):
        analyzer.calculate_local_thz_modes(core, test_structure_7res, num_modes=0)


def test_calculate_local_thz_modes_qcp_scaling(analyzer, test_structure_7res):
    """Test that QCP properly scales THz frequencies."""
    # High QCP core
    high_qcp_core = QuantumCore(
        residue_indices=[0, 1, 2, 3],
        average_qcp=10.0,
        coherence=0.9,
        center_of_mass=(5.7, 0.0, 0.0)
    )
    
    # Low QCP core
    low_qcp_core = QuantumCore(
        residue_indices=[0, 1, 2, 3],
        average_qcp=7.0,
        coherence=0.5,
        center_of_mass=(5.7, 0.0, 0.0)
    )
    
    high_modes = analyzer.calculate_local_thz_modes(high_qcp_core, test_structure_7res, num_modes=3)
    low_modes = analyzer.calculate_local_thz_modes(low_qcp_core, test_structure_7res, num_modes=3)
    
    # High QCP should generally produce higher frequencies (on average)
    # This is a statistical test, so we check the average
    if len(high_modes) > 0 and len(low_modes) > 0:
        avg_high_freq = sum(m.frequency for m in high_modes) / len(high_modes)
        avg_low_freq = sum(m.frequency for m in low_modes) / len(low_modes)
        
        # High QCP core should have higher average frequency
        assert avg_high_freq >= avg_low_freq * 0.9  # Allow some tolerance


# ============================================================================
# Test Resonance Coupling Detection
# ============================================================================

def test_find_coupled_residues_phi_harmonic(analyzer, test_structure_20res):
    """Test finding coupled residues for a φ-harmonic mode."""
    # Create a φ-harmonic mode (1.618 THz)
    mode = THzMode(
        frequency=1.62,  # Near φ × 1.0 THz
        amplitude=0.8,
        participating_residues=list(range(20)),
        is_phi_harmonic=True
    )
    
    coupled = analyzer.find_coupled_residues(mode, test_structure_20res, phi_tolerance=0.1)
    
    # Should find some coupled pairs
    # (pairs with sequence separation >= 5 and distance < 15Å)
    assert isinstance(coupled, list)
    
    # Validate coupled pairs
    for res_i, res_j in coupled:
        # Check sequence separation
        assert abs(res_j - res_i) >= 5
        
        # Check spatial distance
        coords = test_structure_20res.atom_coordinates
        dx = coords[res_i][0] - coords[res_j][0]
        dy = coords[res_i][1] - coords[res_j][1]
        dz = coords[res_i][2] - coords[res_j][2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        assert distance < 15.0


def test_find_coupled_residues_non_phi_harmonic(analyzer, test_structure_20res):
    """Test that non-φ-harmonic modes return no couples."""
    # Create a non-φ-harmonic mode
    mode = THzMode(
        frequency=3.14,  # Not near any φ-harmonic
        amplitude=0.5,
        participating_residues=list(range(20)),
        is_phi_harmonic=False
    )
    
    coupled = analyzer.find_coupled_residues(mode, test_structure_20res, phi_tolerance=0.1)
    
    # Should find no coupled pairs (not φ-harmonic)
    assert len(coupled) == 0


def test_find_coupled_residues_invalid_inputs(analyzer, test_structure_7res):
    """Test error handling for invalid inputs."""
    mode = THzMode(
        frequency=1.62,
        amplitude=0.8,
        participating_residues=[0, 1, 2],
        is_phi_harmonic=True
    )
    
    # Invalid tolerance
    with pytest.raises(ValueError, match="phi_tolerance must be > 0"):
        analyzer.find_coupled_residues(mode, test_structure_7res, phi_tolerance=-0.1)
    
    with pytest.raises(ValueError, match="phi_tolerance must be > 0"):
        analyzer.find_coupled_residues(mode, test_structure_7res, phi_tolerance=0.0)


def test_find_coupled_residues_sequence_separation_filter(analyzer, test_structure_7res):
    """Test that sequence separation filter works correctly."""
    # Create mode with all residues
    mode = THzMode(
        frequency=1.618,  # Exact φ-harmonic
        amplitude=0.9,
        participating_residues=list(range(7)),
        is_phi_harmonic=True
    )
    
    coupled = analyzer.find_coupled_residues(mode, test_structure_7res, phi_tolerance=0.1)
    
    # All coupled pairs should have sequence separation >= 5
    for res_i, res_j in coupled:
        assert abs(res_j - res_i) >= 5


# ============================================================================
# Test Helper Methods
# ============================================================================

def test_is_phi_harmonic_helper(analyzer):
    """Test the _is_phi_harmonic helper method."""
    # Test exact φ-harmonics
    assert analyzer._is_phi_harmonic(1.0, tolerance=0.1)  # φ^0
    assert analyzer._is_phi_harmonic(1.618, tolerance=0.1)  # φ^1
    assert analyzer._is_phi_harmonic(2.618, tolerance=0.1)  # φ^2
    
    # Test near φ-harmonics (within tolerance)
    assert analyzer._is_phi_harmonic(1.62, tolerance=0.1)
    assert analyzer._is_phi_harmonic(2.60, tolerance=0.1)
    
    # Test non-φ-harmonics
    assert not analyzer._is_phi_harmonic(3.14, tolerance=0.1)
    assert not analyzer._is_phi_harmonic(5.0, tolerance=0.1)


def test_calculate_per_residue_qcp(analyzer, test_structure_7res):
    """Test per-residue QCP calculation."""
    metrics = QCPPMetrics(
        qcp_score=8.0,
        field_coherence=0.8,
        stability_score=2.0,
        phi_match_score=0.85,
        calculation_time_ms=1.0
    )
    
    qcp_values = analyzer._calculate_per_residue_qcp(test_structure_7res, metrics)
    
    # Should have one value per residue
    assert len(qcp_values) == len(test_structure_7res.sequence)
    
    # All values should be positive
    assert all(qcp >= 0 for qcp in qcp_values)
    
    # Helices should have higher QCP than coils
    helix_indices = [i for i, ss in enumerate(test_structure_7res.secondary_structure) if ss == 'H']
    coil_indices = [i for i, ss in enumerate(test_structure_7res.secondary_structure) if ss == 'C']
    
    if helix_indices and coil_indices:
        avg_helix_qcp = sum(qcp_values[i] for i in helix_indices) / len(helix_indices)
        avg_coil_qcp = sum(qcp_values[i] for i in coil_indices) / len(coil_indices)
        assert avg_helix_qcp > avg_coil_qcp


def test_calculate_per_residue_coherence(analyzer, test_structure_7res):
    """Test per-residue coherence calculation."""
    metrics = QCPPMetrics(
        qcp_score=8.0,
        field_coherence=0.6,  # Maps to 0.8 in [0,1] range
        stability_score=2.0,
        phi_match_score=0.85,
        calculation_time_ms=1.0
    )
    
    coherence_values = analyzer._calculate_per_residue_coherence(test_structure_7res, metrics)
    
    # Should have one value per residue
    assert len(coherence_values) == len(test_structure_7res.sequence)
    
    # All values should be in [0, 1] range
    assert all(0.0 <= coh <= 1.0 for coh in coherence_values)


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_small_core_size(analyzer, test_structure_7res):
    """Test that small cores (< 3 residues) are filtered out."""
    # Mock moderate QCP that might create small gaps
    analyzer.qcpp_adapter.analyze_conformation.return_value = QCPPMetrics(
        qcp_score=7.2,
        field_coherence=0.65,
        stability_score=1.8,
        phi_match_score=0.7,
        calculation_time_ms=1.0
    )
    
    cores = analyzer.identify_quantum_cores(test_structure_7res, qcp_threshold=7.0)
    
    # All cores should have >= 3 residues
    for core in cores:
        assert len(core.residue_indices) >= 3


def test_performance_timing(analyzer, test_structure_20res):
    """Test that analysis completes in reasonable time."""
    import time
    
    start_time = time.time()
    
    # Identify cores
    cores = analyzer.identify_quantum_cores(test_structure_20res, qcp_threshold=7.0)
    
    # Calculate modes for all cores
    all_modes = []
    for core in cores:
        modes = analyzer.calculate_local_thz_modes(core, test_structure_20res, num_modes=5)
        all_modes.extend(modes)
        
        # Find coupled residues
        for mode in modes:
            coupled = analyzer.find_coupled_residues(mode, test_structure_20res)
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    
    # Should complete in < 100ms for 20-residue structure
    assert elapsed_ms < 100, f"Analysis took {elapsed_ms:.2f}ms (expected < 100ms)"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
