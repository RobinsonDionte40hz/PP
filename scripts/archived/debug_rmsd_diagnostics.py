"""Debug script for RMSD diagnostics"""

from ubf_protein.quantum_refinement_engine import QuantumRefinementEngine
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics
from ubf_protein.energy_function import MolecularMechanicsEnergy
from ubf_protein.rmsd_calculator import RMSDCalculator, NativeStructure
from ubf_protein.models import Conformation

# Mock QCPP adapter
class MockAdapter(QCPPIntegrationAdapter):
    def __init__(self):
        pass
    
    def analyze_conformation(self, conformation):
        return QCPPMetrics(
            qcp_score=5.0,
            field_coherence=0.5,
            stability_score=10.0,
            phi_match_score=0.7,
            calculation_time_ms=1.0
        )

# Create engine
engine = QuantumRefinementEngine(
    qcpp_adapter=MockAdapter(),
    energy_calculator=MolecularMechanicsEnergy(),
    rmsd_calculator=RMSDCalculator(align_structures=True)
)

# Create sample conformation
sequence = "MHHHHHEEEEEAAAAAVVVV"
n_residues = len(sequence)
coords = [(float(i), 0.0, 0.0) for i in range(n_residues)]

ss = ['C'] * n_residues
ss[1:7] = ['H'] * 6   # Helix residues 1-6
ss[7:12] = ['E'] * 5  # Sheet residues 7-11
ss[12:17] = ['C'] * 5  # Loop residues 12-16

conformation = Conformation(
    conformation_id="test_conf",
    sequence=sequence,
    atom_coordinates=coords,
    energy=-100.0,
    rmsd_to_native=None,
    secondary_structure=ss,
    phi_angles=[-60.0] * n_residues,
    psi_angles=[-45.0] * n_residues,
    available_move_types=[],
    structural_constraints={}
)

# Create native structure with displacements
native_coords = []
for i, (x, y, z) in enumerate(coords):
    if 1 <= i <= 6:  # Helix
        native_coords.append((x + 1.0, y, z))
    elif 7 <= i <= 11:  # Sheet
        native_coords.append((x + 2.0, y, z))
    elif 12 <= i <= 16:  # Loop
        native_coords.append((x + 3.0, y, z))
    else:  # Other
        native_coords.append((x + 1.5, y, z))

native = NativeStructure(
    pdb_id="1TEST",
    sequence=sequence,
    ca_coords=native_coords
)

# Run diagnostics
print("Running diagnostics...")
diag = engine.diagnose_rmsd_components(conformation, native)

print("\nResults:")
print(f"Total RMSD: {diag['total_rmsd']:.3f} Å")
print(f"Helix residues: {diag['helix_residues']}")
print(f"Helix RMSD: {diag['helix_rmsd']:.3f} Å")
print(f"Sheet residues: {diag['sheet_residues']}")
print(f"Sheet RMSD: {diag['sheet_rmsd']:.3f} Å")
print(f"Loop residues: {diag['loop_residues']}")
print(f"Loop RMSD: {diag['loop_rmsd']:.3f} Å")
print(f"Core residues: {diag['core_residues']}")
print(f"Core RMSD: {diag['core_rmsd']:.3f} Å")

print("\n" + diag['report'])
