import time

from ubf_protein.mediator_agent import MediatorAgent
from ubf_protein.mediator_config import MediatorConfig
from ubf_protein.models import Conformation
from ubf_protein.pattern_detection import PatternSignificance


def _make_mediator():
    # Minimal mediator for testing - qcpp_adapter/geometric_analyzer/shared_memory not used by folding detection
    return MediatorAgent(
        protein_sequence='A' * 50,
        qcpp_adapter=None,
        geometric_analyzer=None,
        shared_memory=None,
        config=MediatorConfig()
    )


def test_helix_classification_and_percentage():
    mediator = _make_mediator()

    # Create a conformation with 40 residues, first 20 satisfy helix criteria
    n = 40
    phi = [-60.0] * 20 + [180.0] * (n - 20)
    psi = [-40.0] * 20 + [0.0] * (n - 20)

    conf = Conformation(
        conformation_id='test1',
        sequence='A' * n,
        atom_coordinates=[(0.0, 0.0, 0.0)] * n,
        energy=0.0,
        rmsd_to_native=None,
        secondary_structure=['C'] * n,
        phi_angles=phi,
        psi_angles=psi,
        available_move_types=[],
        structural_constraints={}
    )

    pattern = mediator._detect_folding_dynamics(conf)
    assert pattern is not None
    assert pattern.folding_data.helix_percentage >= 49.0
    assert pattern.significance in (PatternSignificance.HIGH, PatternSignificance.MEDIUM)


def test_region_identification_for_helix_and_sheet():
    mediator = _make_mediator()

    # Build phi/psi such that residues 5-10 are helix (6 residues) and 20-22 are sheet (3 residues)
    n = 30
    phi = [180.0] * n
    psi = [0.0] * n

    # Helix region 5-10 -> indices 4..9 (0-based)
    for i in range(4, 10):
        phi[i] = -60.0
        psi[i] = -40.0

    # Sheet region 20-22 -> indices 19..21
    for i in range(19, 22):
        phi[i] = -120.0
        psi[i] = 140.0

    conf = Conformation(
        conformation_id='test2',
        sequence='A' * n,
        atom_coordinates=[(0.0, 0.0, 0.0)] * n,
        energy=0.0,
        rmsd_to_native=None,
        secondary_structure=['C'] * n,
        phi_angles=phi,
        psi_angles=psi,
        available_move_types=[],
        structural_constraints={}
    )

    pattern = mediator._detect_folding_dynamics(conf)
    assert pattern is not None
    fd = pattern.folding_data
    # Helix region found
    assert any((start <= 5 <= end) for start, end in fd.helix_regions)
    # Sheet region found
    assert any((start <= 20 <= end) for start, end in fd.sheet_regions)


def test_significance_high_when_large_helix_fraction():
    mediator = _make_mediator()

    n = 20
    phi = [-60.0] * 8 + [0.0] * (n - 8)
    psi = [-40.0] * 8 + [0.0] * (n - 8)

    conf = Conformation(
        conformation_id='test3',
        sequence='A' * n,
        atom_coordinates=[(0.0, 0.0, 0.0)] * n,
        energy=0.0,
        rmsd_to_native=None,
        secondary_structure=['C'] * n,
        phi_angles=phi,
        psi_angles=psi,
        available_move_types=[],
        structural_constraints={}
    )

    pattern = mediator._detect_folding_dynamics(conf)
    assert pattern is not None
    assert pattern.significance == PatternSignificance.HIGH
