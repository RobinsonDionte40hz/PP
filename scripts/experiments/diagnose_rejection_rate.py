"""
Diagnostic script to analyze the 98% move rejection rate.

This script runs a short exploration and tracks:
1. Why moves are rejected (validation failure, energy, etc.)
2. Bond length distribution
3. Move type success rates
4. Energy landscape characteristics
"""

import sys
from pathlib import Path
import json
from collections import Counter, defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.adaptive_config import AdaptiveConfigurator
from ubf_protein.models import Conformation
import math


def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two 3D coordinates."""
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    dz = coord1[2] - coord2[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def analyze_conformation(conformation: Conformation):
    """Analyze a conformation's bond lengths."""
    coords = conformation.atom_coordinates
    bond_lengths = []
    
    for i in range(len(coords) - 1):
        dist = calculate_distance(coords[i], coords[i + 1])
        bond_lengths.append(dist)
    
    return {
        'min': min(bond_lengths) if bond_lengths else 0,
        'max': max(bond_lengths) if bond_lengths else 0,
        'avg': sum(bond_lengths) / len(bond_lengths) if bond_lengths else 0,
        'violations_under_1': sum(1 for d in bond_lengths if d < 1.0),
        'violations_over_5': sum(1 for d in bond_lengths if d > 5.0),
        'total_bonds': len(bond_lengths)
    }


def run_diagnostic_exploration(sequence: str, iterations: int = 100):
    """Run exploration with detailed tracking."""
    
    print(f"="*70)
    print(f"REJECTION RATE DIAGNOSTIC")
    print(f"="*70)
    print(f"\nProtein Sequence: {sequence}")
    print(f"Length: {len(sequence)} residues")
    print(f"Iterations: {iterations}")
    
    # Create agent with adaptive configuration
    from ubf_protein.adaptive_config import create_config_for_sequence
    adaptive_config = create_config_for_sequence(sequence)
    
    agent = ProteinAgent(
        protein_sequence=sequence,
        initial_frequency=9.0,
        initial_coherence=0.6,
        adaptive_config=adaptive_config
    )
    
    # Track statistics
    stats = {
        'total_iterations': 0,
        'validation_failures': 0,
        'repair_attempts': 0,
        'repair_successes': 0,
        'energy_rejected': 0,
        'stuck_events': 0,
        'successful_moves': 0,
        'bond_violations': defaultdict(int),
        'conformation_stats': []
    }
    
    print(f"\n{'='*70}")
    print(f"Running exploration...")
    print(f"{'='*70}\n")
    
    # Track validation failures per iteration
    last_validation_failures = 0
    
    for i in range(iterations):
        # Get initial validation count
        validation_count_before = agent._validation_failures
        
        # Run one exploration step
        outcome = agent.explore_step()
        
        # Get updated validation count
        validation_count_after = agent._validation_failures
        
        # Track what happened
        stats['total_iterations'] += 1
        
        # Check if validation failed this iteration
        if validation_count_after > validation_count_before:
            stats['validation_failures'] += 1
        
        # Update repair statistics
        stats['repair_attempts'] = agent._repair_attempts
        stats['repair_successes'] = agent._repair_successes
        
        # Analyze current conformation
        conf_stats = analyze_conformation(outcome.new_conformation)
        stats['conformation_stats'].append(conf_stats)
        
        # Track bond violations
        if conf_stats['violations_over_5'] > 0:
            stats['bond_violations']['over_5'] += conf_stats['violations_over_5']
        if conf_stats['violations_under_1'] > 0:
            stats['bond_violations']['under_1'] += conf_stats['violations_under_1']
        
        # Progress update every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations} - "
                  f"Energy: {outcome.new_conformation.energy:.2f} kcal/mol, "
                  f"Validation failures so far: {stats['validation_failures']}")
    
    # Calculate final statistics
    total_validations = stats['validation_failures']
    total_repairs_attempted = stats['repair_attempts']
    total_repairs_successful = stats['repair_successes']
    
    # Bond length statistics
    all_bond_stats = stats['conformation_stats']
    avg_min_bond = sum(s['min'] for s in all_bond_stats) / len(all_bond_stats)
    avg_max_bond = sum(s['max'] for s in all_bond_stats) / len(all_bond_stats)
    avg_avg_bond = sum(s['avg'] for s in all_bond_stats) / len(all_bond_stats)
    
    total_violations_over_5 = stats['bond_violations']['over_5']
    total_violations_under_1 = stats['bond_violations']['under_1']
    
    # Print results
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC RESULTS")
    print(f"{'='*70}\n")
    
    print(f"📊 Validation Statistics:")
    print(f"  Total iterations: {stats['total_iterations']}")
    print(f"  Validation failures: {total_validations}")
    print(f"  Failure rate: {(total_validations/iterations)*100:.1f}%")
    print(f"  Repair attempts: {total_repairs_attempted}")
    print(f"  Successful repairs: {total_repairs_successful}")
    if total_repairs_attempted > 0:
        print(f"  Repair success rate: {(total_repairs_successful/total_repairs_attempted)*100:.1f}%")
    
    print(f"\n📏 Bond Length Statistics:")
    print(f"  Average minimum bond: {avg_min_bond:.2f} Å")
    print(f"  Average maximum bond: {avg_max_bond:.2f} Å")
    print(f"  Average mean bond: {avg_avg_bond:.2f} Å")
    print(f"  Ideal CA-CA distance: 3.8 Å")
    print(f"  Validation range: 1.0 - 5.0 Å")
    
    print(f"\n🚨 Constraint Violations:")
    print(f"  Bonds > 5.0 Å: {total_violations_over_5}")
    print(f"  Bonds < 1.0 Å: {total_violations_under_1}")
    print(f"  Total violations: {total_violations_over_5 + total_violations_under_1}")
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}\n")
    
    if total_validations > iterations * 0.9:
        print(f"❌ CRITICAL: >90% validation failure rate!")
        print(f"   This means nearly all moves violate structural constraints.")
        print(f"   Root cause: Move generation produces invalid bond lengths.")
    
    if avg_max_bond > 5.0:
        print(f"❌ PROBLEM: Average max bond ({avg_max_bond:.2f} Å) exceeds threshold (5.0 Å)")
        print(f"   Moves are generating bond lengths that are too long.")
        print(f"   Recommendation: Either relax MAX_BOND_LENGTH or constrain move sizes.")
    
    if total_violations_over_5 > 0:
        violations_per_conf = total_violations_over_5 / iterations
        print(f"⚠️  WARNING: Average {violations_per_conf:.1f} long bond violations per conformation")
        print(f"   {(violations_per_conf/(len(sequence)-1))*100:.1f}% of bonds violate max length")
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    if avg_max_bond > 5.0:
        print(f"1. IMMEDIATE FIX: Increase MAX_BOND_LENGTH in structural_validation.py")
        print(f"   Current: 5.0 Å")
        print(f"   Suggested: {avg_max_bond + 0.5:.1f} Å (observed max + safety margin)")
        print(f"   OR: Reduce move size scaling to keep bonds within 5.0 Å")
    
    if total_repairs_attempted > 0 and total_repairs_successful < total_repairs_attempted * 0.5:
        print(f"\n2. REPAIR SYSTEM: Repair success rate is {(total_repairs_successful/total_repairs_attempted)*100:.1f}%")
        print(f"   Consider improving repair algorithm or accepting minor violations")
    
    if total_validations > iterations * 0.9:
        print(f"\n3. MOVE GENERATION: Fundamental problem with how moves are generated")
        print(f"   Moves should maintain structural constraints by design")
        print(f"   Consider constraint-aware move generation")
    
    print(f"\n{'='*70}\n")
    
    # Save detailed results
    results_file = "diagnostic_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'sequence': sequence,
            'iterations': iterations,
            'statistics': {
                'validation_failure_rate': total_validations / iterations,
                'repair_success_rate': total_repairs_successful / max(1, total_repairs_attempted),
                'avg_min_bond': avg_min_bond,
                'avg_max_bond': avg_max_bond,
                'avg_avg_bond': avg_avg_bond,
                'violations_over_5': total_violations_over_5,
                'violations_under_1': total_violations_under_1
            }
        }, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    # Test with Ubiquitin sequence
    ubiquitin_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    
    run_diagnostic_exploration(ubiquitin_seq, iterations=100)
