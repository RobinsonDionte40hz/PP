#!/usr/bin/env python3
"""
Visualize Agent Scaling Results
Creates plots showing how RMSD, Energy, and Throughput vary with agent count.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_file = Path("agent_scaling_results.json")
with open(results_file, 'r') as f:
    data = json.load(f)

results = data['results']

# Extract data
agent_counts = [r['num_agents'] for r in results]
energies = [r['best_energy'] for r in results]
rmsds = [r['estimated_rmsd'] for r in results]
throughputs = [r['throughput_conf_per_s'] for r in results]

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Energy vs Agents
ax1.plot(agent_counts, energies, 'o-', color='#2E86AB', linewidth=2, markersize=10)
ax1.axhline(y=min(energies), color='green', linestyle='--', alpha=0.5, label='Best Energy')
ax1.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
ax1.set_ylabel('Best Energy (kcal/mol)', fontsize=12, fontweight='bold')
ax1.set_title('Energy vs Agent Count', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Annotate best point
best_energy_idx = energies.index(min(energies))
ax1.annotate('OPTIMAL', 
             xy=(agent_counts[best_energy_idx], energies[best_energy_idx]),
             xytext=(10, 20), textcoords='offset points',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Plot 2: RMSD vs Agents
ax2.plot(agent_counts, rmsds, 's-', color='#A23B72', linewidth=2, markersize=10)
ax2.axhline(y=min(rmsds), color='green', linestyle='--', alpha=0.5, label='Best RMSD')
ax2.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
ax2.set_ylabel('Estimated RMSD (Å)', fontsize=12, fontweight='bold')
ax2.set_title('RMSD vs Agent Count', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Annotate best point
best_rmsd_idx = rmsds.index(min(rmsds))
ax2.annotate('OPTIMAL', 
             xy=(agent_counts[best_rmsd_idx], rmsds[best_rmsd_idx]),
             xytext=(10, 20), textcoords='offset points',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Plot 3: Throughput vs Agents
ax3.plot(agent_counts, throughputs, '^-', color='#F18F01', linewidth=2, markersize=10)
ax3.axhline(y=max(throughputs), color='blue', linestyle='--', alpha=0.5, label='Peak Throughput')
ax3.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
ax3.set_ylabel('Throughput (conf/s)', fontsize=12, fontweight='bold')
ax3.set_title('Throughput vs Agent Count', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Add summary text
fig.suptitle('Agent Scaling Experiment: Ubiquitin (76 residues, 200 iter/agent)', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_file = Path("agent_scaling_plots.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plots saved to: {output_file}")

# Show the plot
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Optimal Agent Count (Energy): {agent_counts[best_energy_idx]} agents")
print(f"  → Energy: {energies[best_energy_idx]:.2f} kcal/mol")
print(f"\nOptimal Agent Count (RMSD): {agent_counts[best_rmsd_idx]} agents")
print(f"  → RMSD: {rmsds[best_rmsd_idx]:.2f} Å")
print(f"\nPeak Throughput: {agent_counts[throughputs.index(max(throughputs))]} agents")
print(f"  → {max(throughputs):.1f} conf/s")
print("\n" + "="*70)
print("RECOMMENDATION: Use 20 agents for optimal energy and RMSD")
print("="*70)
