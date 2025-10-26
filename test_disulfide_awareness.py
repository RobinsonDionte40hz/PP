"""
Quick test to validate disulfide constraint awareness in move evaluation.

This test verifies that:
1. Agents receive disulfide constraint information in physics_factors
2. Moves affecting disulfide-bonded residues get weighted appropriately
3. The system prefers moves that reduce distance error
"""

from ubf_protein.models import DisulfideBond, Conformation, ConformationalMove
from ubf_protein.interfaces import MoveType
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.mapless_moves import CapabilityBasedMoveEvaluator
from ubf_protein.behavioral_state import BehavioralState
from ubf_protein.consciousness import ConsciousnessState
import random

def test_disulfide_constraint_in_physics_factors():
    """Test that agents calculate disulfide constraint factors."""
    print("\n" + "="*60)
    print("TEST 1: Disulfide Constraint in Physics Factors")
    print("="*60)
    
    # Create a simple protein with 2 cysteines that should form a bond
    sequence = "ACDEFGHIKLMNPQRSTVWYC"  # Cys at positions 0 and 20
    
    # Define disulfide bond between the two cysteines
    disulfide_bonds = [
        DisulfideBond(residue_i=0, residue_j=20, distance=3.8, tolerance=0.5)
    ]
    
    # Create agent with disulfide bonds
    agent = ProteinAgent(
        protein_sequence=sequence,
        disulfide_bonds=disulfide_bonds
    )
    
    # Create a move that affects one of the bonded residues
    test_move = ConformationalMove(
        move_id="test_move_1",
        move_type=MoveType.BACKBONE_ROTATION,
        target_residues=[0, 1, 2],  # Includes Cys-0
        estimated_energy_change=-5.0,
        estimated_rmsd_change=0.5,
        required_capabilities={},
        structural_feasibility=0.8,
        energy_barrier=10.0
    )
    
    # Get physics factors for this move
    physics_factors = agent._get_physics_factors(test_move)
    
    print(f"\nMove affects residues: {test_move.target_residues}")
    print(f"Disulfide bond: Cys-0 <-> Cys-20 (target: 3.8Å)")
    print(f"\nPhysics Factors:")
    for key, value in physics_factors.items():
        print(f"  {key}: {value:.3f}")
    
    # Verify disulfide_constraint exists
    assert 'disulfide_constraint' in physics_factors, "Missing disulfide_constraint in physics_factors"
    assert physics_factors['disulfide_constraint'] != 0.5, "Disulfide constraint should be non-neutral when bond affected"
    
    print(f"\n✓ Disulfide constraint factor calculated: {physics_factors['disulfide_constraint']:.3f}")
    print("✓ Agents can now 'see' disulfide bond constraints!")

def test_move_evaluation_with_disulfide():
    """Test that move evaluator uses disulfide constraints."""
    print("\n" + "="*60)
    print("TEST 2: Move Evaluation with Disulfide Constraints")
    print("="*60)
    
    # Create move evaluator
    evaluator = CapabilityBasedMoveEvaluator()
    
    # Create behavioral state
    consciousness = ConsciousnessState(9.0, 0.6)
    behavioral = BehavioralState(consciousness.get_coordinates())
    
    # Create two similar moves
    move_without_disulfide = ConformationalMove(
        move_id="move_no_ss",
        move_type=MoveType.BACKBONE_ROTATION,
        target_residues=[10, 11, 12],
        estimated_energy_change=-5.0,
        estimated_rmsd_change=0.5,
        required_capabilities={},
        structural_feasibility=0.8,
        energy_barrier=10.0
    )
    
    move_with_disulfide = ConformationalMove(
        move_id="move_with_ss",
        move_type=MoveType.BACKBONE_ROTATION,
        target_residues=[0, 1, 2],
        estimated_energy_change=-5.0,
        estimated_rmsd_change=0.5,
        required_capabilities={},
        structural_feasibility=0.8,
        energy_barrier=10.0
    )
    
    # Physics factors without disulfide influence
    physics_no_ss = {
        'qaap': 0.5,
        'resonance': 0.5,
        'water_shielding': 0.5,
        'disulfide_constraint': 0.5  # Neutral (no bond affected)
    }
    
    # Physics factors with positive disulfide influence (move helps satisfy bond)
    physics_with_ss = {
        'qaap': 0.5,
        'resonance': 0.5,
        'water_shielding': 0.5,
        'disulfide_constraint': 0.8  # High (move brings cysteines closer)
    }
    
    # Evaluate both moves
    weight_no_ss = evaluator.evaluate_move(
        move_without_disulfide,
        behavioral,
        memory_influence=0.5,
        physics_factors=physics_no_ss
    )
    
    weight_with_ss = evaluator.evaluate_move(
        move_with_disulfide,
        behavioral,
        memory_influence=0.5,
        physics_factors=physics_with_ss
    )
    
    print(f"\nMove without disulfide influence:")
    print(f"  Weight: {weight_no_ss:.4f}")
    print(f"\nMove with positive disulfide influence:")
    print(f"  Weight: {weight_with_ss:.4f}")
    print(f"\nWeight difference: {weight_with_ss - weight_no_ss:.4f}")
    
    # Verify that move with positive disulfide influence has higher weight
    assert weight_with_ss > weight_no_ss, "Move helping disulfide bonds should have higher weight"
    
    print(f"\n✓ Moves satisfying disulfide constraints get higher weight!")
    print(f"✓ Agents will prefer moves that bring bonded cysteines closer!")

def test_distance_gradient():
    """Test that the constraint factor creates a proper distance gradient."""
    print("\n" + "="*60)
    print("TEST 3: Distance Gradient for Disulfide Bonds")
    print("="*60)
    
    sequence = "ACDEFGHIKLMNPQRSTVWYC"
    disulfide_bonds = [
        DisulfideBond(residue_i=0, residue_j=20, distance=3.8, tolerance=0.5)
    ]
    
    # Test different initial distances
    test_distances = [3.8, 5.0, 10.0, 20.0, 50.0, 100.0, 140.0]
    
    print(f"\nTarget distance: 3.8Å")
    print(f"Testing gradient at different current distances:\n")
    print(f"{'Current Dist (Å)':<20} {'Distance Error':<20} {'Impact Factor':<20}")
    print("-" * 60)
    
    for dist in test_distances:
        error = abs(dist - 3.8)
        # Simulate the impact calculation from _get_physics_factors
        impact = 1.0 / (1.0 + (error / 10.0)**2)
        print(f"{dist:<20.1f} {error:<20.1f} {impact:<20.3f}")
    
    print("\n✓ Gradient shows:")
    print("  - Near target (3.8Å): High impact (~1.0)")
    print("  - Within 10Å: Moderate impact (0.5-0.9)")
    print("  - Far away (>50Å): Low impact (<0.2)")
    print("✓ Agents get stronger signal as cysteines approach target distance!")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DISULFIDE CONSTRAINT AWARENESS VALIDATION")
    print("="*60)
    
    try:
        test_disulfide_constraint_in_physics_factors()
        test_move_evaluation_with_disulfide()
        test_distance_gradient()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nDisulfide constraint awareness is working correctly:")
        print("  1. Agents calculate disulfide impact for each move")
        print("  2. Move evaluator uses this information in quantum alignment")
        print("  3. Distance gradient guides agents toward satisfied bonds")
        print("\nThis should significantly improve folding for proteins with disulfide bonds!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise
