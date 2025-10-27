"""
Compare QCPP (Quantum Coherence Protein Predictor) and UBF (Universal Behavioral Framework) results for Ubiquitin
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_qcpp_results(protein_id="1UBQ"):
    """Load QCPP analysis results"""
    qcpp_file = Path(f"quantum_coherence_proteins/results/{protein_id}_analysis.json")
    with open(qcpp_file, 'r') as f:
        return json.load(f)

def load_ubf_results():
    """Load UBF multi-agent results"""
    ubf_file = Path("ubiquitin_parallel_2000iter.json")
    with open(ubf_file, 'r') as f:
        return json.load(f)

def compare_results():
    """Compare QCPP and UBF results"""
    print("="*80)
    print("QCPP vs UBF: Ubiquitin Comparison")
    print("="*80)
    
    # Load results
    qcpp = load_qcpp_results()
    ubf = load_ubf_results()
    
    print("\n" + "="*80)
    print("SYSTEM OVERVIEW")
    print("="*80)
    print(f"Protein: Ubiquitin (1UBQ)")
    print(f"Sequence Length: {len(ubf['metadata']['sequence'])}")
    print(f"Sequence: {ubf['metadata']['sequence']}")
    
    print("\n" + "="*80)
    print("QCPP RESULTS (Physics-Based Stability Prediction)")
    print("="*80)
    print(f"Approach: Quantum coherence + golden ratio patterns")
    print(f"Stability Score: {qcpp['stability_score']:.4f}")
    
    # Extract QCP values
    qcp_vals = [r['qcp'] for r in qcpp['qcp_values']]
    coherence_vals = [c['coherence'] for c in qcpp['coherence']]
    
    print(f"Average QCP Value: {np.mean(qcp_vals):.4f}")
    print(f"Average Coherence: {np.mean(coherence_vals):.4f}")
    print(f"Number of Residues: {len(qcp_vals)}")
    
    # THz spectrum info
    thz_spectrum = qcpp.get('thz_spectrum', [])
    if thz_spectrum:
        frequencies = [t['frequency'] for t in thz_spectrum]
        intensities = [t['intensity'] for t in thz_spectrum]
        print(f"THz Spectrum: {len(frequencies)} frequency points")
        if intensities:
            max_intensity_idx = np.argmax(np.abs(intensities))
            print(f"Peak THz Frequency: {frequencies[max_intensity_idx]:.2f} THz at intensity {intensities[max_intensity_idx]:.2f}")
    
    print("\n" + "="*80)
    print("UBF RESULTS (Consciousness-Based Conformational Exploration)")
    print("="*80)
    print(f"Approach: Autonomous agents with consciousness coordinates")
    print(f"Number of Agents: {ubf['metadata']['num_agents']}")
    print(f"Total Iterations: {ubf['metadata']['total_iterations']}")
    print(f"Runtime: {ubf['metadata']['total_runtime_seconds']:.2f} seconds")
    print(f"Best Energy: {ubf['best_results']['best_energy']:.4f} kcal/mol")
    print(f"Total Conformations Explored: {ubf['best_results']['total_conformations_explored']}")
    
    # Energy breakdown
    energy_components = ubf['energy_components']
    print(f"\nEnergy Components:")
    print(f"  Bond Energy: {energy_components['bond_energy']:.2f}")
    print(f"  Angle Energy: {energy_components['angle_energy']:.2f}")
    print(f"  Dihedral Energy: {energy_components['dihedral_energy']:.2f}")
    print(f"  VDW Energy: {energy_components['vdw_energy']:.2f}")
    print(f"  Electrostatic Energy: {energy_components['electrostatic_energy']:.2f}")
    
    # Agent performance
    agent_summary = ubf['agent_summary']
    print(f"\nAgent Performance:")
    print(f"  Avg Conformations/Agent: {agent_summary['avg_conformations_per_agent']:.1f}")
    print(f"  Avg Memories/Agent: {agent_summary['avg_memories_per_agent']:.1f}")
    print(f"  Avg Decision Time: {agent_summary['avg_decision_time_ms']:.2f} ms")
    print(f"  Total Stuck Events: {agent_summary['total_stuck_events']}")
    print(f"  Total Escapes: {agent_summary['total_escapes']}")
    
    # Collective learning
    collective = ubf['collective_learning']
    print(f"\nCollective Learning:")
    print(f"  Shared Memories: {collective['shared_memories_created']}")
    print(f"  Learning Benefit: {collective['collective_learning_benefit']:.4f}")
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES & COMPLEMENTARITY")
    print("="*80)
    print("\n1. APPROACH:")
    print("   QCPP: Physics-based stability prediction using quantum mechanics")
    print("   UBF:  Conformational space exploration using agent consciousness")
    
    print("\n2. OUTPUT:")
    print("   QCPP: Stability scores, coherence values, THz spectra")
    print("   UBF:  Energy landscapes, conformational trajectories, RMSD values")
    
    print("\n3. COMPUTATIONAL FOCUS:")
    print("   QCPP: Static analysis of native structure")
    print("   UBF:  Dynamic exploration of conformational space")
    
    print("\n4. VALIDATION:")
    print("   QCPP: Correlates with experimental stability (THz, melting temps)")
    print("   UBF:  Optimizes energy functions, explores low-energy states")
    
    print("\n5. COMPLEMENTARITY:")
    print("   - QCPP can validate UBF conformations for stability")
    print("   - UBF can explore conformations predicted stable by QCPP")
    print("   - QCPP quantum factors could guide UBF move evaluation")
    print("   - Both use 40 Hz resonance and phi-based patterns")
    
    print("\n" + "="*80)
    print("INTEGRATION OPPORTUNITIES")
    print("="*80)
    print("1. Use QCPP's QCP values in UBF's quantum factor calculation")
    print("2. Validate UBF's best conformations with QCPP stability scores")
    print("3. Combine consciousness-based (UBF) + quantum-based (QCPP) scoring")
    print("4. Use QCPP's THz spectra to identify resonant conformations in UBF")
    print("5. Apply QCPP coherence measures to UBF conformational ensembles")
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"QCPP Runtime: ~seconds (static analysis)")
    print(f"UBF Runtime: {ubf['metadata']['total_runtime_seconds']:.2f} seconds ({ubf['metadata']['num_agents']} agents × {ubf['metadata']['iterations_per_agent']} iterations)")
    print(f"UBF Avg Time/Iteration: {ubf['metadata']['avg_time_per_iteration_ms']:.2f} ms")
    print(f"UBF Avg Decision Time: {agent_summary['avg_decision_time_ms']:.2f} ms")
    
    # Create visualization
    create_comparison_plot(qcpp, ubf)
    
    print("\n" + "="*80)
    print("Comparison plot saved to: qcpp_ubf_comparison.png")
    print("="*80)

def create_comparison_plot(qcpp, ubf):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('QCPP vs UBF: Ubiquitin Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    qcp_vals = [r['qcp'] for r in qcpp['qcp_values']]
    coherence_vals = [c['coherence'] for c in qcpp['coherence']]
    
    # Plot 1: QCP values across residues
    ax1 = axes[0, 0]
    ax1.plot(range(len(qcp_vals)), qcp_vals, 'b-', linewidth=2)
    ax1.set_xlabel('Residue Index')
    ax1.set_ylabel('QCP Value')
    ax1.set_title('QCPP: Quantum Coherence Pattern')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Field coherence
    ax2 = axes[0, 1]
    ax2.plot(range(len(coherence_vals)), coherence_vals, 'g-', linewidth=2)
    ax2.set_xlabel('Residue Index')
    ax2.set_ylabel('Coherence')
    ax2.set_title('QCPP: Field Coherence')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: THz spectrum
    ax3 = axes[0, 2]
    thz = qcpp.get('thz_spectrum', [])
    if thz:
        frequencies = [t['frequency'] for t in thz]
        intensities = [t['intensity'] for t in thz]
        ax3.bar(frequencies, intensities, color='purple', alpha=0.7, width=0.1)
        ax3.set_xlabel('Frequency (THz)')
        ax3.set_ylabel('Intensity')
        ax3.set_title('QCPP: THz Spectrum')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: UBF Energy components
    ax4 = axes[1, 0]
    energy_comp = ubf['energy_components']
    components = ['Bond', 'Angle', 'Dihedral', 'VDW', 'Electro']
    values = [
        energy_comp['bond_energy'],
        energy_comp['angle_energy'],
        energy_comp['dihedral_energy'],
        energy_comp['vdw_energy'],
        energy_comp['electrostatic_energy']
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax4.bar(components, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Energy (kcal/mol)')
    ax4.set_title('UBF: Energy Components')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Agent performance
    ax5 = axes[1, 1]
    per_agent = ubf['per_agent_metrics']
    agent_ids = [f"A{i}" for i in range(len(per_agent))]
    best_energies = [agent['best_energy'] for agent in per_agent]
    ax5.bar(agent_ids, best_energies, color='teal', alpha=0.7)
    ax5.set_xlabel('Agent')
    ax5.set_ylabel('Best Energy (kcal/mol)')
    ax5.set_title('UBF: Per-Agent Best Energy')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Summary comparison
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = f"""
    QCPP SUMMARY
    ────────────
    Stability Score: {qcpp['stability_score']:.4f}
    Avg QCP: {np.mean([r['qcp'] for r in qcpp['qcp_values']]):.4f}
    Avg Coherence: {np.mean([c['coherence'] for c in qcpp['coherence']]):.4f}
    
    UBF SUMMARY
    ───────────
    Best Energy: {ubf['best_results']['best_energy']:.2f} kcal/mol
    Agents: {ubf['metadata']['num_agents']}
    Conformations: {ubf['best_results']['total_conformations_explored']}
    Runtime: {ubf['metadata']['total_runtime_seconds']:.2f}s
    
    COMPLEMENTARITY
    ──────────────
    • QCPP: Physics validation
    • UBF: Space exploration
    • Both: 40 Hz resonance
    • Integration: Quantum+Consciousness
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('qcpp_ubf_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot created successfully!")

if __name__ == "__main__":
    compare_results()
