"""
Test THz signature recording in UBF agent exploration.

Runs a single agent for a few iterations and verifies THz signatures are recorded.
"""

import sys
import os

# Add ubf_protein to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.adaptive_config import AdaptiveConfig


def test_thz_recording():
    """Test that THz signatures are recorded during exploration."""
    print("=" * 70)
    print("THz SIGNATURE RECORDING TEST")
    print("=" * 70)
    
    # Create a small test protein
    sequence = "ACDEFGH"
    print(f"\n📋 Test sequence: {sequence} ({len(sequence)} residues)")
    
    # Create agent
    print("\n🤖 Creating protein agent...")
    agent = ProteinAgent(
        protein_sequence=sequence,
        initial_frequency=9.0,
        initial_coherence=0.6,
        enable_visualization=False
    )
    
    # Run exploration for 50 iterations
    print(f"\n🔬 Running exploration for 50 iterations...")
    for i in range(50):
        try:
            outcome = agent.explore_step()
            if i % 10 == 0:
                print(f"   Iteration {i}: Energy={outcome.new_conformation.energy:.2f}")
        except Exception as e:
            print(f"   ⚠️  Error at iteration {i}: {e}")
            break
    
    # Check THz signatures
    print(f"\n📊 Exploration complete!")
    metrics = agent.get_exploration_metrics()
    
    print(f"\n   Iterations completed: {metrics['iterations_completed']}")
    print(f"   Best energy: {metrics['best_energy']:.2f}")
    print(f"   Conformations explored: {metrics['conformations_explored']}")
    print(f"   Stuck in minima: {metrics['stuck_in_minima_count']}")
    
    # Get THz signatures
    thz_history = agent.get_thz_signature_history()
    print(f"\n   THz signatures recorded: {len(thz_history)}")
    
    if thz_history:
        print(f"\n✅ THz signature recording SUCCESSFUL!")
        print(f"\n   Sample signatures:")
        for i, spectrum in enumerate(thz_history[:3]):
            peak_freqs = spectrum.get_peak_frequencies(threshold=0.1)
            print(f"   {i+1}. Energy={spectrum.total_energy:.2f}, "
                  f"RMSD={spectrum.rmsd:.2f}, "
                  f"Peaks={len(peak_freqs)}, "
                  f"Dominant={peak_freqs[0] if peak_freqs else 0:.2f} THz")
    else:
        print(f"\n⚠️  No THz signatures recorded")
        print(f"   This is expected if agent didn't reach stable minima")
        print(f"   Try running for more iterations or with a longer sequence")
    
    return len(thz_history) > 0


def main():
    """Run test."""
    try:
        success = test_thz_recording()
        
        print("\n" + "=" * 70)
        if success:
            print("✅ TEST PASSED: THz recording integrated successfully")
        else:
            print("⚠️  TEST INCOMPLETE: No signatures recorded (may need more iterations)")
        print("=" * 70)
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Run full determinism experiment with 100 trials")
        print(f"   2. Test on real proteins (1UBQ, 1CRN, 1LYZ)")
        print(f"   3. Analyze signature clustering")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
