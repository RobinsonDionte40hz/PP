#!/usr/bin/env python3
"""
Quick Start Example: QCPP-UBF Integration

This script provides a minimal example of running integrated QCPP-UBF exploration.
It's designed to run quickly (<30 seconds) to demonstrate the key features.

Run this to quickly verify the integration works on your system.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run a quick integrated exploration example."""
    print("=" * 70)
    print("QUICK START: QCPP-UBF Integrated Exploration")
    print("=" * 70)
    print()
    
    # Import components
    print("Importing components...")
    from ubf_protein.examples.integrated_exploration import (
        run_integrated_exploration,
        get_config_by_name
    )
    print("✓ Imports successful")
    print()
    
    # Configure for quick demo
    print("Configuration:")
    config = get_config_by_name('default')
    config.cache_size = 50  # Small cache for demo
    print(f"  • QCPP integration: {'ENABLED' if config.enabled else 'DISABLED'}")
    print(f"  • Cache size: {config.cache_size} conformations")
    print(f"  • Physics grounding: {'ENABLED' if config.enable_physics_grounding else 'DISABLED'}")
    print(f"  • Dynamic adjustment: {'ENABLED' if config.enable_dynamic_adjustment else 'DISABLED'}")
    print()
    
    # Run exploration
    print("Running exploration (this will take ~10-20 seconds)...")
    print("  • Sequence: ACDEFGH (7 residues)")
    print("  • Agents: 3")
    print("  • Iterations: 10 per agent")
    print()
    
    results = run_integrated_exploration(
        sequence="ACDEFGH",
        num_agents=3,
        iterations=10,
        diversity='balanced',
        qcpp_config=config,
        verbose=False  # Suppress detailed output
    )
    
    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best Energy:         {results['best_energy']:.2f} kcal/mol")
    print(f"Total Conformations: {results['total_conformations']}")
    print(f"Exploration Time:    {results['exploration_time_seconds']:.1f}s")
    print(f"Throughput:          {results['throughput_conformations_per_second']:.1f} conf/s")
    print()
    
    qcpp = results['qcpp_integration']
    print("QCPP Integration Statistics:")
    print(f"  • Total analyses:     {qcpp['total_analyses']}")
    print(f"  • Cache hits:         {qcpp['cache_hits']}")
    print(f"  • Cache hit rate:     {qcpp['cache_hit_rate']:.1f}%")
    print(f"  • Avg calc time:      {qcpp['avg_calculation_time_ms']:.2f}ms")
    print()
    
    print("=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print()
    print("The QCPP-UBF integration is working correctly!")
    print()
    print("Next steps:")
    print("  1. Try with a longer sequence for more realistic results")
    print("  2. Increase agents and iterations for better predictions")
    print("  3. Use --output to save results for analysis")
    print("  4. See README_INTEGRATED.md for full documentation")
    print()
    print("Example commands:")
    print()
    print("  # Full exploration with 10 agents")
    print("  python ubf_protein/examples/integrated_exploration.py \\")
    print("      --sequence ACDEFGHIKLMNPQRSTVWY \\")
    print("      --agents 10 \\")
    print("      --iterations 2000 \\")
    print("      --output results.json")
    print()
    print("  # High-performance mode")
    print("  python ubf_protein/examples/integrated_exploration.py \\")
    print("      --sequence ACDEFGH \\")
    print("      --config high_performance \\")
    print("      --cache-size 5000 \\")
    print("      --agents 20")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
