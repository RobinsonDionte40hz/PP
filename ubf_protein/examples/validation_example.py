#!/usr/bin/env python3
"""
Validation Example for UBF Protein System

This script demonstrates how to use the validation suite to:
1. Validate predictions against known PDB structures
2. Run comprehensive test suites
3. Compare to baseline methods
4. Assess prediction quality

Examples include both quick tests and full validation runs.
"""

import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ubf_protein.validation_suite import ValidationSuite


def example_1_single_protein_validation():
    """
    Example 1: Validate a single protein with native structure.
    
    This is the most basic validation - predict structure for one protein
    and compare against its known native structure from PDB.
    """
    print("=" * 70)
    print("EXAMPLE 1: Single Protein Validation")
    print("=" * 70)
    print("\nThis example validates structure prediction for Crambin (1CRN),")
    print("a small 46-residue protein, against its known native structure.\n")
    
    # Initialize validation suite
    suite = ValidationSuite()
    
    # Validate single protein
    # Note: This will download PDB structure and run exploration
    print("Running validation (this may take 1-2 minutes)...\n")
    
    try:
        report = suite.validate_protein(
            pdb_id="1CRN",           # Crambin - small, fast-folding
            num_agents=5,            # Use 5 agents for faster demo
            iterations=100,          # 100 iterations per agent
            use_multi_agent=True
        )
        
        # Display results
        print(report.get_summary())
        
        # Demonstrate accessing metrics programmatically
        print("\nProgrammatic Access to Metrics:")
        print(f"  PDB ID: {report.pdb_id}")
        print(f"  RMSD: {report.best_rmsd:.2f} Å")
        print(f"  Energy: {report.best_energy:.2f} kcal/mol")
        print(f"  GDT-TS: {report.gdt_ts_score:.1f}")
        print(f"  TM-score: {report.tm_score:.3f}")
        print(f"  Quality: {report.assess_quality()}")
        print(f"  Successful: {report.is_successful()}")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        print("Note: This example requires internet connection to download PDB files.")
        return False
    
    return True


def example_2_quality_assessment():
    """
    Example 2: Understanding quality assessment criteria.
    
    Shows how different RMSD and GDT-TS values map to quality levels.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Quality Assessment Criteria")
    print("=" * 70)
    print("\nThe validation suite assesses prediction quality based on:")
    print("  - RMSD (Root Mean Square Deviation) in Ångströms")
    print("  - GDT-TS (Global Distance Test) score (0-100)")
    print("\nQuality Levels:\n")
    
    from ubf_protein.validation_suite import ValidationReport
    
    # Create example reports with different quality levels
    examples = [
        ("Excellent", 1.5, -80.0, 85.0, 0.92),
        ("Good", 2.5, -70.0, 72.0, 0.78),
        ("Acceptable", 4.0, -55.0, 55.0, 0.62),
        ("Poor", 7.0, -40.0, 35.0, 0.45),
    ]
    
    for name, rmsd, energy, gdt_ts, tm_score in examples:
        report = ValidationReport(
            pdb_id="EXAMPLE",
            sequence_length=50,
            best_rmsd=rmsd,
            best_energy=energy,
            gdt_ts_score=gdt_ts,
            tm_score=tm_score,
            runtime_seconds=10.0,
            conformations_explored=1000,
            num_agents=10,
            iterations_per_agent=100
        )
        
        quality = report.assess_quality()
        success = "✓" if report.is_successful() else "✗"
        
        print(f"{name:12s} RMSD={rmsd:.1f}Å, GDT-TS={gdt_ts:.0f}, TM={tm_score:.2f}")
        print(f"             Quality: {quality}, Successful: {success}\n")
    
    print("Success Criteria:")
    print("  ✓ RMSD < 5.0 Å")
    print("  ✓ Energy < 0 kcal/mol (thermodynamically stable)")
    print("  ✓ GDT-TS > 50 (correct fold)")
    
    return True


def example_3_test_suite():
    """
    Example 3: Run validation on multiple proteins.
    
    Demonstrates running the full test suite with multiple proteins
    and getting aggregated statistics.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Test Suite Validation")
    print("=" * 70)
    print("\nThis example runs validation on the configured test proteins.")
    print("Note: This is a longer-running example (5-10 minutes for all proteins).\n")
    
    # Initialize suite
    suite = ValidationSuite()
    
    print(f"Test proteins configured: {len(suite.test_proteins)}")
    if suite.test_proteins:
        print("\nAvailable proteins:")
        for protein in suite.test_proteins:
            print(f"  - {protein['pdb_id']}: {protein['name']} "
                  f"({protein['residues']} residues, {protein['difficulty']} difficulty)")
    
    # Ask user if they want to run full suite
    print("\nFull test suite will take several minutes.")
    print("For this demo, we'll show how to run it (but not execute).")
    
    print("\nTo run the full test suite, use:")
    print("  results = suite.run_test_suite(num_agents=10, iterations=500)")
    print("  print(results.get_summary())")
    
    print("\nThe results include:")
    print("  - Success rate (% of proteins with acceptable predictions)")
    print("  - Average RMSD across all proteins")
    print("  - Average GDT-TS score")
    print("  - Quality distribution (excellent/good/acceptable/poor)")
    
    return True


def example_4_baseline_comparison():
    """
    Example 4: Compare UBF to baseline methods.
    
    Shows how to run baseline comparisons to demonstrate that UBF
    outperforms random sampling and simple Monte Carlo.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Baseline Comparison")
    print("=" * 70)
    print("\nThis example compares UBF predictions to baseline methods:")
    print("  1. Random Sampling: Generate random conformations")
    print("  2. Monte Carlo: Simple Metropolis algorithm")
    print("\nNote: This requires PDB download and takes 2-3 minutes.\n")
    
    print("To run baseline comparison, use:")
    print("  suite = ValidationSuite()")
    print("  comparison = suite.compare_to_baseline('1CRN', num_samples=1000)")
    print("  improvements = comparison.get_improvement_summary()")
    print("\nExample output:")
    print("  UBF RMSD: 3.2 Å")
    print("  Random Sampling RMSD: 8.7 Å (UBF is 63% better)")
    print("  Monte Carlo RMSD: 5.1 Å (UBF is 37% better)")
    
    return True


def example_5_programmatic_usage():
    """
    Example 5: Programmatic usage in your own scripts.
    
    Shows how to integrate validation into your own analysis pipelines.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Programmatic Usage")
    print("=" * 70)
    print("\nIntegrate validation into your own scripts:\n")
    
    code_example = """
from ubf_protein.validation_suite import ValidationSuite

# Initialize
suite = ValidationSuite()

# Validate protein
report = suite.validate_protein(
    pdb_id="1UBQ",
    num_agents=10,
    iterations=1000
)

# Check quality
if report.is_successful():
    print(f"Success! Quality: {report.assess_quality()}")
    print(f"RMSD: {report.best_rmsd:.2f} Å")
    print(f"Energy: {report.best_energy:.2f} kcal/mol")
else:
    print(f"Poor prediction. RMSD: {report.best_rmsd:.2f} Å")

# Save results
suite.save_results(results, "validation_results.json")

# Run full test suite
results = suite.run_test_suite(num_agents=10, iterations=500)

# Analyze results
print(f"Success rate: {results.success_rate:.1f}%")
print(f"Average RMSD: {results.average_rmsd:.2f} Å")

for report in results.validation_reports:
    if report.assess_quality() == "excellent":
        print(f"Excellent prediction: {report.pdb_id}")
"""
    
    print(code_example)
    
    return True


def example_6_with_run_multi_agent():
    """
    Example 6: Using run_multi_agent.py with native structure validation.
    
    Shows command-line usage for validation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Command-Line Validation")
    print("=" * 70)
    print("\nYou can also use run_multi_agent.py directly with --native flag:\n")
    
    examples = [
        ("Basic validation", 
         "python run_multi_agent.py MQIFVKT --agents 10 --iterations 500 --native 1CRN"),
        
        ("With output file",
         "python run_multi_agent.py MQIFVKT --agents 15 --iterations 1000 --native 1UBQ --output ubiquitin_results.json"),
        
        ("Verbose mode",
         "python run_multi_agent.py MQIFVKT --agents 20 --iterations 500 --native 1VII --verbose"),
    ]
    
    for name, command in examples:
        print(f"{name}:")
        print(f"  {command}\n")
    
    print("The output will include:")
    print("  ✓ Energy components breakdown")
    print("  ✓ RMSD to native structure")
    print("  ✓ GDT-TS and TM-score")
    print("  ✓ Quality assessment (excellent/good/acceptable/poor)")
    print("  ✓ Energy validation (negative = folded)")
    
    return True


def main():
    """Run all validation examples."""
    print("\n" + "=" * 70)
    print("UBF PROTEIN SYSTEM - VALIDATION EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the validation capabilities of the")
    print("UBF (Universal Behavioral Framework) protein prediction system.\n")
    
    print("Available Examples:")
    print("  1. Single Protein Validation")
    print("  2. Quality Assessment Criteria")
    print("  3. Test Suite Validation")
    print("  4. Baseline Comparison")
    print("  5. Programmatic Usage")
    print("  6. Command-Line Validation")
    print("\n" + "=" * 70)
    
    # Run examples
    try:
        # Example 2 - Quick, no downloads
        example_2_quality_assessment()
        
        # Example 3 - Quick overview
        example_3_test_suite()
        
        # Example 4 - Overview only
        example_4_baseline_comparison()
        
        # Example 5 - Code examples
        example_5_programmatic_usage()
        
        # Example 6 - CLI usage
        example_6_with_run_multi_agent()
        
        # Example 1 - Actual validation (optional, requires download)
        print("\n" + "=" * 70)
        print("OPTIONAL: Run actual validation?")
        print("=" * 70)
        print("\nWould you like to run an actual validation on 1CRN?")
        print("This will download the PDB file and run exploration (~1-2 minutes).")
        
        response = input("\nRun validation? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            example_1_single_protein_validation()
        else:
            print("\nSkipping actual validation. See examples above for usage.")
        
        # Summary
        print("\n" + "=" * 70)
        print("EXAMPLES COMPLETE")
        print("=" * 70)
        print("\nFor more information:")
        print("  - See ubf_protein/validation_suite.py for API reference")
        print("  - See ubf_protein/validation_proteins.json for test protein list")
        print("  - Run 'python run_multi_agent.py --help' for CLI usage")
        print("\nHappy validating! 🎉")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
