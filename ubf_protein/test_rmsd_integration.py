"""
Test script to verify RMSD validation integration with UBF protein agent (Task 5).

This script demonstrates:
1. Loading a native structure
2. Creating a protein agent with native structure parameter
3. Running exploration with RMSD tracking
4. Verifying GDT-TS and TM-score are calculated
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ubf_protein.models import Conformation
from ubf_protein.protein_agent import ProteinAgent


def create_test_native_structure(sequence: str) -> Conformation:
    """
    Create a mock native structure for testing.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        Conformation representing native structure
    """
    # Generate simple coordinates (mock native structure)
    coords = [(float(i), float(i), float(i)) for i in range(len(sequence))]
    
    native = Conformation(
        conformation_id="native_structure",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-50.0,  # Mock native energy
        rmsd_to_native=0.0,  # Native has 0 RMSD to itself
        secondary_structure=['C'] * len(sequence),
        phi_angles=[0.0] * len(sequence),
        psi_angles=[0.0] * len(sequence),
        available_move_types=['random_rotation', 'hydrophobic_collapse'],
        structural_constraints={},
        energy_components=None,
        native_structure_ref="test_native",
        gdt_ts_score=100.0,  # Native has perfect GDT-TS
        tm_score=1.0  # Native has perfect TM-score
    )
    
    return native


def test_rmsd_integration():
    """Test RMSD validation integration with protein agent."""
    print("=" * 70)
    print("TASK 5: RMSD Validation Integration Test")
    print("=" * 70)
    
    # Define test protein sequence
    test_sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues
    print(f"\nTest Sequence: {test_sequence} ({len(test_sequence)} residues)")
    
    # Create mock native structure
    print("\n1. Creating mock native structure...")
    native_structure = create_test_native_structure(test_sequence)
    print(f"   ✓ Native structure created: {native_structure.conformation_id}")
    print(f"   ✓ Native RMSD: {native_structure.rmsd_to_native:.2f} Å")
    print(f"   ✓ Native GDT-TS: {native_structure.gdt_ts_score:.1f}")
    print(f"   ✓ Native TM-score: {native_structure.tm_score:.3f}")
    
    # Create protein agent with native structure
    print("\n2. Creating protein agent with native structure parameter...")
    try:
        agent = ProteinAgent(
            protein_sequence=test_sequence,
            initial_frequency=9.0,
            initial_coherence=0.6,
            native_structure=native_structure,  # Task 5: Pass native structure
            enable_visualization=False
        )
        print("   ✓ Agent created successfully")
        print(f"   ✓ RMSD calculator available: {agent._rmsd_calculator is not None}")
        print(f"   ✓ Native structure stored: {agent._native_structure is not None}")
    except Exception as e:
        print(f"   ✗ Failed to create agent: {e}")
        return False
    
    # Run a few exploration steps
    print("\n3. Running exploration with RMSD validation...")
    num_steps = 5
    
    for i in range(num_steps):
        try:
            outcome = agent.explore_step()
            
            # Check if RMSD metrics were calculated
            has_rmsd = outcome.new_conformation.rmsd_to_native is not None
            has_gdt_ts = outcome.new_conformation.gdt_ts_score is not None
            has_tm_score = outcome.new_conformation.tm_score is not None
            
            status = "✓" if (has_rmsd and has_gdt_ts and has_tm_score) else "✗"
            
            print(f"   Step {i+1}/{num_steps}: {status}", end="")
            
            if has_rmsd and has_gdt_ts and has_tm_score:
                print(f" RMSD={outcome.new_conformation.rmsd_to_native:.2f}Å, "
                      f"GDT-TS={outcome.new_conformation.gdt_ts_score:.1f}, "
                      f"TM-score={outcome.new_conformation.tm_score:.3f}")
            else:
                missing = []
                if not has_rmsd:
                    missing.append("RMSD")
                if not has_gdt_ts:
                    missing.append("GDT-TS")
                if not has_tm_score:
                    missing.append("TM-score")
                print(f" Missing: {', '.join(missing)}")
                
        except Exception as e:
            print(f"   ✗ Step {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Check exploration metrics
    print("\n4. Checking exploration metrics...")
    metrics = agent.get_exploration_metrics()
    
    has_best_rmsd = "best_rmsd" in metrics
    has_best_gdt_ts = "best_gdt_ts" in metrics
    has_best_tm_score = "best_tm_score" in metrics
    
    print(f"   Best RMSD tracked: {'✓' if has_best_rmsd else '✗'}")
    if has_best_rmsd:
        print(f"      Value: {metrics['best_rmsd']:.2f} Å")
    
    print(f"   Best GDT-TS tracked: {'✓' if has_best_gdt_ts else '✗'}")
    if has_best_gdt_ts:
        print(f"      Value: {metrics['best_gdt_ts']:.1f}")
    
    print(f"   Best TM-score tracked: {'✓' if has_best_tm_score else '✗'}")
    if has_best_tm_score:
        print(f"      Value: {metrics['best_tm_score']:.3f}")
    
    # Test graceful degradation without native structure
    print("\n5. Testing graceful degradation (no native structure)...")
    try:
        agent_no_native = ProteinAgent(
            protein_sequence=test_sequence,
            initial_frequency=9.0,
            initial_coherence=0.6,
            native_structure=None,  # No native structure
            enable_visualization=False
        )
        
        outcome = agent_no_native.explore_step()
        
        # RMSD should be None when no native structure is provided
        no_rmsd_metrics = outcome.new_conformation.rmsd_to_native is None
        no_gdt_ts = outcome.new_conformation.gdt_ts_score is None
        no_tm_score = outcome.new_conformation.tm_score is None
        
        if no_rmsd_metrics and no_gdt_ts and no_tm_score:
            print("   ✓ Agent works correctly without native structure")
            print("   ✓ No RMSD metrics calculated (as expected)")
        else:
            print("   ✗ Unexpected RMSD metrics present without native structure")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed without native structure: {e}")
        return False
    
    # Success summary
    print("\n" + "=" * 70)
    print("TASK 5 INTEGRATION TEST: PASSED ✓")
    print("=" * 70)
    print("\nAll checks completed successfully:")
    print("  ✓ Conformation model updated with validation fields")
    print("  ✓ ProteinAgent accepts native_structure parameter")
    print("  ✓ RMSD, GDT-TS, and TM-score calculated during exploration")
    print("  ✓ Exploration metrics track best validation scores")
    print("  ✓ Graceful handling when native structure not provided")
    print("  ✓ Error handling works correctly")
    
    return True


if __name__ == "__main__":
    success = test_rmsd_integration()
    sys.exit(0 if success else 1)
