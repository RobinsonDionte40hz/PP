#!/usr/bin/env python3
"""
Simple benchmark test runner - validates CLI then runs benchmarks

This is a streamlined version that:
1. Tests one protein to ensure everything works
2. Provides simple output for validation
3. Can be extended to full benchmark suite
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ubf_protein.benchmark_collector import BenchmarkCollector


def test_single_protein():
    """Test a single small protein to verify everything works."""
    print("\n" + "="*70)
    print(" BENCHMARK SYSTEM TEST")
    print("="*70)
    print("\n🧪 Testing with 1VII (Villin headpiece, 36 residues)")
    print("This is a quick test to ensure the benchmark system works.\n")
    
    collector = BenchmarkCollector(output_dir="benchmark_results")
    
    # Run quick test with reduced iterations
    result = collector.run_protein(
        pdb_id="1VII",
        agents=5,
        iterations=50,  # Reduced for quick test
        enable_refinement=False,  # Disable for speed
        enable_mediators=False,   # Disable for speed
        qcpp_config="default"
    )
    
    if result and result.success:
        print("\n" + "="*70)
        print(" ✅ BENCHMARK SYSTEM TEST PASSED")
        print("="*70)
        print(f"\nThe system is working correctly!")
        print(f"  - Execution time: {result.execution_time_seconds:.1f}s")
        print(f"  - Energy: {result.best_energy:.2f} kcal/mol")
        if result.best_rmsd:
            print(f"  - RMSD: {result.best_rmsd:.2f} Å")
        print(f"\n📁 Results saved to: benchmark_results/individual/1VII_benchmark.json")
        print(f"\n✨ Ready to run full benchmark suite!")
        return True
    else:
        print("\n" + "="*70)
        print(" ❌ BENCHMARK SYSTEM TEST FAILED")
        print("="*70)
        if result:
            print(f"Error: {result.error_message}")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark testing for bioRxiv paper")
    parser.add_argument("--test", action="store_true", help="Run single protein test")
    parser.add_argument("--protein", help="Test specific protein")
    parser.add_argument("--small-batch", action="store_true", help="Test 5 small proteins")
    parser.add_argument("--fast-50", action="store_true", help="Fast 50-protein benchmark (1-2 hours)")
    parser.add_argument("--full", action="store_true", help="Run full 50-protein benchmark (high quality, 3-5 hours)")
    
    args = parser.parse_args()
    
    if args.test:
        # Quick system test
        success = test_single_protein()
        sys.exit(0 if success else 1)
        
    elif args.protein:
        # Single protein
        collector = BenchmarkCollector()
        collector.run_protein(args.protein)
        
    elif args.small_batch:
        # Small test batch
        print("\n🧪 Running small batch test (5 proteins)\n")
        collector = BenchmarkCollector()
        proteins = ["1L2Y", "1VII", "1CRN", "2MR9", "1ENH"]  # All small proteins
        collector.run_batch(proteins, iterations=100)
    
    elif args.fast_50:
        # Fast 50-protein benchmark (reduced settings for speed)
        print("\n🚀 Running FAST 50-protein benchmark")
        print("Optimized for limited computing resources (~30-60 minutes total)\n")
        print("Configuration:")
        print("  - 50 proteins: 25 small (<50 residues) + 25 medium (50-100 residues)")
        print("  - NO large proteins (>100 residues)")
        print("  - Agents: 5-10 (auto-scaled by size)")
        print("  - Iterations: 100 per agent")
        print("  - Refinement: DISABLED (saves time)")
        print("  - Mediators: DISABLED (saves time)")
        print("  - Results saved for bioRxiv paper\n")
        
        response = input("Continue with fast 50-protein benchmark? (yes/no): ")
        if response.lower() == 'yes':
            collector = BenchmarkCollector()
            proteins = collector.get_fast_50_protein_list()
            print(f"\nTesting {len(proteins)} proteins: {', '.join(proteins[:10])}...")
            
            # Use fast settings
            collector.run_batch(
                proteins,
                agents=None,  # Auto-scale but keep small
                iterations=100,  # Reduced for speed
                enable_refinement=False,  # Disable for speed
                enable_mediators=False,  # Disable for speed
                qcpp_config="default"
            )
        else:
            print("Cancelled.")
        
    elif args.full:
        # Full 50-protein benchmark
        print("\n🚀 Running FULL 50-protein benchmark")
        print("This will take several hours...\n")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() == 'yes':
            collector = BenchmarkCollector()
            proteins = collector.get_50_protein_list()
            collector.run_batch(proteins)
        else:
            print("Cancelled.")
    else:
        print("Please specify an option:")
        print("  --test          : Quick test with 1VII")
        print("  --protein 1UBQ  : Test specific protein")
        print("  --small-batch   : Test 5 small proteins")
        print("  --fast-50       : Fast 50-protein benchmark (recommended for paper)")
        print("  --full          : Full 50-protein high-quality benchmark")
        parser.print_help()


if __name__ == "__main__":
    main()
