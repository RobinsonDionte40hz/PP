#!/usr/bin/env python3
"""
Debug script to test quantum refinement integration
"""
import sys
import logging
from pathlib import Path

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see attribute checks
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add ubf_protein to path
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

from src.protein_predictor import QuantumCoherenceProteinPredictor
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.rmsd_calculator import NativeStructureLoader
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import aa3, aa1

def test_refinement():
    """Test refinement with 1VII (Villin)"""
    
    pdb_id = "1VII"
    pdb_file = Path("pdb_cache/pdb1vii.ent")
    
    # Load sequence
    aa_map = dict(zip(aa3, aa1))
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))
    chain = list(structure.get_chains())[0]
    residues = list(chain.get_residues())
    
    sequence = ""
    for res in residues:
        if res.id[0] == ' ':
            resname = res.resname
            if resname in aa_map:
                sequence += aa_map[resname]
            else:
                sequence += 'X'
    
    print(f"Testing refinement with {pdb_id}")
    print(f"Sequence: {sequence} ({len(sequence)} residues)")
    
    # Initialize QCPP
    print("\n[1] Initializing QCPP predictor...")
    qcpp_predictor = QuantumCoherenceProteinPredictor()
    qcpp_adapter = QCPPIntegrationAdapter(qcpp_predictor, cache_size=10000)
    print("[OK] QCPP initialized")
    
    # Load native structure
    print("\n[2] Loading native structure...")
    loader = NativeStructureLoader(cache_dir="./pdb_cache")
    native_structure = loader.load_from_file(str(pdb_file), ca_only=True)
    print(f"[OK] Native structure loaded: {len(native_structure.ca_coords)} CA atoms")
    
    # Create coordinator with refinement enabled
    print("\n[3] Creating multi-agent coordinator with refinement enabled...")
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        qcpp_integration=qcpp_adapter,
        qcpp_analysis_frequency=20,
        enable_quantum_refinement=True,
        refinement_rmsd_threshold=5.0
    )
    
    print(f"[OK] Refinement enabled: {coordinator._enable_quantum_refinement}")
    print(f"[OK] Refinement engine initialized: {coordinator._refinement_engine is not None}")
    print(f"[OK] RMSD threshold: {coordinator._refinement_rmsd_threshold}A")
    
    # Initialize agents with native structure for RMSD tracking
    print("\n[4] Initializing agents...")
    num_agents = 20  # Scaled up from 5
    coordinator.initialize_agents(
        count=num_agents,
        diversity_profile="balanced",
        native_structure=native_structure  # Pass native structure for RMSD calculation
    )
    print(f"[OK] {len(coordinator._agents)} agents initialized with native structure")
    
    # Run exploration with refinement
    num_iterations = 100  # Scaled up from 20
    print(f"\n[5] Running exploration with refinement ({num_iterations} iterations, {num_agents} agents)...")
    print(f"    Total conformations to explore: {num_agents * num_iterations}")
    exploration_results, refinement_result = coordinator.run_parallel_exploration_with_refinement(
        iterations=num_iterations,
        native_structure=native_structure
    )
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    print(f"\nStage 1 (Exploration):")
    print(f"  Best Energy: {exploration_results.best_energy:.2f} kcal/mol")
    print(f"  Best RMSD: {exploration_results.best_rmsd:.2f}A")
    print(f"  Conformations explored: {len(coordinator._agents) * 20}")
    
    if refinement_result:
        print(f"\nStage 2 (Quantum Refinement):")
        print(f"  [OK] Refinement was triggered!")
        print(f"  Stage 1 RMSD: {refinement_result.stage1_rmsd:.2f}A")
        print(f"  Final RMSD: {refinement_result.final_rmsd:.2f}A")
        print(f"  Improvement: {refinement_result.rmsd_improvement:.2f}A ({refinement_result.rmsd_improvement/refinement_result.stage1_rmsd*100:.1f}%)")
        print(f"  Final Energy: {refinement_result.final_energy:.2f} kcal/mol")
        print(f"  GDT-TS: {refinement_result.gdt_ts:.1f}")
        print(f"  TM-score: {refinement_result.tm_score:.3f}")
    else:
        print(f"\n[WARNING] Refinement was NOT triggered")
        print(f"  Checking reasons:")
        print(f"    - Refinement enabled: {coordinator._enable_quantum_refinement}")
        print(f"    - Engine initialized: {coordinator._refinement_engine is not None}")
        print(f"    - Best conformation exists: {exploration_results.best_conformation is not None}")
        print(f"    - Best RMSD: {exploration_results.best_rmsd}")
        print(f"    - RMSD threshold: {coordinator._refinement_rmsd_threshold}A")
        if exploration_results.best_rmsd != float('inf'):
            print(f"    - Above threshold: {exploration_results.best_rmsd > coordinator._refinement_rmsd_threshold}")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    test_refinement()
