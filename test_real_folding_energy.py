"""
Test actual protein folding with disulfide bonds to verify total energy drops to <200 kcal/mol.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig
from ubf_protein.models import DisulfideBond


def test_real_folding_energy():
    """Run actual folding and track total energy evolution."""
    
    print("=" * 80)
    print("REAL FOLDING ENERGY TEST")
    print("=" * 80)
    print()
    
    # Small test protein with disulfide bond (Crambin-like sequence)
    sequence = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
    print(f"Protein: {len(sequence)} residues")
    print(f"Sequence: {sequence}")
    print(f"Cysteines at positions: {[i for i, aa in enumerate(sequence) if aa == 'C']}")
    print()
    
    # Identify disulfide bonds (using known Crambin pattern)
    disulfide_bonds = [
        DisulfideBond(residue_i=3, residue_j=40),   # CYS3-CYS40
        DisulfideBond(residue_i=4, residue_j=32),   # CYS4-CYS32
        DisulfideBond(residue_i=16, residue_j=26),  # CYS16-CYS26
    ]
    
    print(f"Disulfide bonds: {len(disulfide_bonds)}")
    for i, bond in enumerate(disulfide_bonds, 1):
        print(f"  Bond {i}: CYS{bond.residue_i} - CYS{bond.residue_j}")
    print()
    
    # Create enhanced physics config for small protein
    physics_config = EnhancedPhysicsConfig.for_small_protein(
        num_residues=len(sequence),
        disulfide_bonds=disulfide_bonds
    )
    
    print("Ramp Schedule:")
    if physics_config.disulfide_ramp_schedule:
        for iteration, k in physics_config.disulfide_ramp_schedule:
            print(f"  Iteration {iteration:4d}+: k = {k:5.1f} kcal/mol/Ų")
    print()
    
    # Create coordinator
    print("Initializing coordinator...")
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        physics_config=physics_config
    )
    
    print("Initializing 5 agents...")
    coordinator.initialize_agents(5, 'balanced')
    
    print("=" * 80)
    print("FOLDING EXPLORATION - TRACKING ENERGY")
    print("=" * 80)
    print()
    
    # Run exploration and track energy at checkpoints
    print("Running 600 iterations...")
    print()
    print(f"{'Iteration':<12} {'Best Total E':<15} {'Best RMSD':<12} Phase")
    print("-" * 60)
    
    # Run in chunks to track progress
    checkpoints = [50, 100, 150, 200, 300, 400, 500, 600]
    
    for checkpoint in checkpoints:
        # Run to checkpoint
        if checkpoint == 50:
            results = coordinator.run_parallel_exploration(50)
        else:
            # Get iterations since last checkpoint
            prev = checkpoints[checkpoints.index(checkpoint) - 1]
            iterations = checkpoint - prev
            results = coordinator.run_parallel_exploration(iterations)
        
        # Extract results
        best_energy = results.best_energy
        best_rmsd = results.best_rmsd
        
        # Determine phase
        if checkpoint < 200:
            phase = "Exploring (k=0.5)"
        elif checkpoint < 500:
            phase = "Constraining (k=2.0)"
        else:
            phase = "Refining (k=10.0)"
        
        print(f"{checkpoint:<12} {best_energy:>12.1f}   {best_rmsd:>10.2f}   {phase}")
    
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()
    
    print(f"Final Best Energy: {results.best_energy:.1f} kcal/mol")
    print(f"Final Best RMSD:   {results.best_rmsd:.2f} Å")
    print()
    
    # Check if we met the target
    if results.best_energy <= 200:
        print("✅ SUCCESS: Total energy under 200 kcal/mol target!")
    elif results.best_energy <= 300:
        print("⚠️  CLOSE: Total energy under 300 kcal/mol (may need more iterations)")
    else:
        print(f"❌ HIGH: Total energy still at {results.best_energy:.1f} kcal/mol")
        print("   This may indicate:")
        print("   - Need more exploration time (>600 iterations)")
        print("   - Base MM energy component still high from poor conformations")
        print("   - Need to tune move acceptance criteria")
    
    print()
    print("=" * 80)
    print()
    
    return results


if __name__ == "__main__":
    test_real_folding_energy()
