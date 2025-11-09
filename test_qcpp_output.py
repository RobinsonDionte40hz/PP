"""Quick test to check QCPP adapter output."""
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.models import Conformation

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

# Create adapter
adapter = QCPPIntegrationAdapter(
    predictor=MockQCPPPredictor(),
    cache_size=1000,
    target_geometry='none'
)

# Test analysis
conf = Conformation(
    conformation_id="test",
    sequence='ACDEFGH',
    atom_coordinates=[(i*3.8, 0.0, 0.0) for i in range(7)],
    energy=0.0,
    rmsd_to_native=None,
    secondary_structure=['C'] * 7,
    phi_angles=[0.0] * 7,
    psi_angles=[0.0] * 7,
    available_move_types=[],
    structural_constraints={}
)

result = adapter.analyze_conformation(conf)
print("Result:", result)
print("QCP values:", result.per_residue_qcp)
print("Are any negative?", any(v < 0 for v in result.per_residue_qcp))
