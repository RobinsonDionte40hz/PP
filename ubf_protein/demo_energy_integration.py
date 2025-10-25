"""
Demo: Energy Function Integration with UBF Protein Agent

This script demonstrates how the molecular mechanics energy function
integrates with the protein agent during conformational exploration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein import config

def main():
    print("=" * 80)
    print("Energy Function Integration Demo")
    print("=" * 80)
    
    # Enable molecular mechanics energy calculation
    config.USE_MOLECULAR_MECHANICS_ENERGY = True
    
    # Create protein agent
    print("\n1. Initializing Protein Agent")
    print("-" * 80)
    agent = ProteinAgent(
        protein_sequence="ACDEFGHIKLMNPQRSTVWY",  # 20 residues
        initial_frequency=9.0,
        initial_coherence=0.6
    )
    
    print(f"   Sequence: {agent._protein_sequence}")
    print(f"   Energy Calculator: {type(agent._energy_calculator).__name__}")
    print(f"   Initial Energy: {agent._current_conformation.energy:.2f} kcal/mol")
    
    # Perform exploration steps
    print("\n2. Performing 10 Exploration Steps")
    print("-" * 80)
    
    for i in range(10):
        outcome = agent.explore_step()
        
        # Display outcome
        print(f"\nStep {i+1}:")
        print(f"   Move Type: {outcome.move_executed.move_type.value if outcome.move_executed else 'None'}")
        print(f"   Energy Change: {outcome.energy_change:+.2f} kcal/mol")
        print(f"   New Energy: {outcome.new_conformation.energy:.2f} kcal/mol")
        print(f"   Success: {outcome.success}")
        
        # Show energy components if available
        if outcome.new_conformation.energy_components:
            components = outcome.new_conformation.energy_components
            print(f"   Energy Components:")
            print(f"      Bond:          {components['bond_energy']:8.2f} kcal/mol")
            print(f"      Angle:         {components['angle_energy']:8.2f} kcal/mol")
            print(f"      Dihedral:      {components['dihedral_energy']:8.2f} kcal/mol")
            print(f"      VDW:           {components['vdw_energy']:8.2f} kcal/mol")
            print(f"      Electrostatic: {components['electrostatic_energy']:8.2f} kcal/mol")
            print(f"      H-bond:        {components['hbond_energy']:8.2f} kcal/mol")
            print(f"      Compactness:   {components['compactness_bonus']:8.2f} kcal/mol")
    
    # Show exploration metrics
    print("\n3. Final Exploration Metrics")
    print("-" * 80)
    metrics = agent.get_exploration_metrics()
    
    print(f"   Iterations Completed: {metrics['iterations_completed']}")
    print(f"   Conformations Explored: {metrics['conformations_explored']}")
    print(f"   Best Energy: {metrics['best_energy']:.2f} kcal/mol")
    print(f"   Memories Created: {metrics['memories_created']}")
    print(f"   Avg Decision Time: {metrics['avg_decision_time_ms']:.2f} ms")
    
    # Show consciousness state
    print("\n4. Final Consciousness State")
    print("-" * 80)
    consciousness = agent.get_consciousness_state()
    coords = consciousness.get_coordinates()
    
    print(f"   Frequency: {coords.frequency:.2f} Hz")
    print(f"   Coherence: {coords.coherence:.3f}")
    
    # Show behavioral state
    behavioral = agent.get_behavioral_state()
    behavior_data = behavioral.get_behavioral_data()
    
    print(f"\n5. Final Behavioral State")
    print("-" * 80)
    print(f"   Exploration Energy: {behavior_data.exploration_energy:.3f}")
    print(f"   Structural Focus: {behavior_data.structural_focus:.3f}")
    print(f"   Hydrophobic Drive: {behavior_data.hydrophobic_drive:.3f}")
    print(f"   Risk Tolerance: {behavior_data.risk_tolerance:.3f}")
    print(f"   Native State Ambition: {behavior_data.native_state_ambition:.3f}")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    # Compare with disabled energy calculator
    print("\n6. Comparison: Disabled Energy Calculator")
    print("-" * 80)
    config.USE_MOLECULAR_MECHANICS_ENERGY = False
    
    agent_simple = ProteinAgent(
        protein_sequence="ACDEFGHIKLMNPQRSTVWY",
        initial_frequency=9.0,
        initial_coherence=0.6
    )
    
    print(f"   Energy Calculator: {agent_simple._energy_calculator}")
    print(f"   Initial Energy: {agent_simple._current_conformation.energy:.2f} kJ/mol")
    print(f"   Note: Uses simplified estimation instead of molecular mechanics")
    
    outcome_simple = agent_simple.explore_step()
    print(f"\n   After 1 step:")
    print(f"   New Energy: {outcome_simple.new_conformation.energy:.2f} kJ/mol")
    print(f"   Energy Components: {outcome_simple.new_conformation.energy_components}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
