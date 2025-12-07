"""Test the latest improvements to the protein prediction system."""
import sys
import time
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, './ubf_protein')

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.rmsd_calculator import RMSDCalculator, NativeStructureLoader

# Ubiquitin sequence
SEQ = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
PDB_ID = '1UBQ'

# Load native structure
print(f"Loading native structure for {PDB_ID}...")
loader = NativeStructureLoader(cache_dir="./pdb_cache")
native_structure = loader.load_from_pdb_id(PDB_ID, ca_only=True)
native_coords = native_structure.ca_coords
print(f"  Loaded {len(native_coords)} CA atoms")

# Initialize coordinator with 5 agents for quick test
print("\nInitializing multi-agent coordinator...")
coordinator = MultiAgentCoordinator(
    protein_sequence=SEQ,
    qcpp_integration=None,
    enable_quantum_refinement=False
)
coordinator.initialize_agents(count=5, diversity_profile="balanced")
print(f"  Initialized 5 agents")

# Run exploration
print("\nRunning 200 iteration exploration...")
start_time = time.time()
results = coordinator.run_parallel_exploration(iterations=200)
elapsed = time.time() - start_time

print(f"\nResults:")
print(f"  Time: {elapsed:.1f}s")
print(f"  Best Energy: {results.best_energy:.2f} kcal/mol")

# Calculate RMSD
rmsd_calc = RMSDCalculator()
if results.best_conformation:
    predicted_coords = results.best_conformation.atom_coordinates
    rmsd = rmsd_calc.calculate_rmsd(predicted_coords, native_coords)
    print(f"  RMSD: {rmsd.rmsd:.2f}A")
    print(f"  Alignment: {rmsd.n_atoms} atoms aligned")
else:
    print("  No conformation found!")
