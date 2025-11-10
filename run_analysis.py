#!/usr/bin/env python3
"""
QCPP Analysis Runner - Legacy QCPP-only Testing

NOTE: This is the LEGACY analysis script for QCPP-only predictions.
For comprehensive protein structure prediction with quantum refinement,
use the PRIMARY testing modules instead:

  PRIMARY MODULE: test_protein.py
    - Quantum Refinement Engine integration
    - Real RMSD calculations with CA-only alignment
    - QCPP-UBF multi-agent exploration
    - Production-ready validation

  SYSTEMATIC TESTING: systematic_protein_testing.py
    - Test 100+ proteins with varied configurations
    - Quantum refinement validation on all tests
    - Comprehensive robustness analysis

Usage:
  python run_analysis.py                    # Run legacy QCPP analysis (6 proteins)
  
Recommended instead:
  python test_protein.py --pdb 1UBQ         # Single protein with quantum refinement
  python systematic_protein_testing.py      # Multiple proteins systematically
"""

from src.qc_pipeline import QCProteinPipeline

# Initialize the pipeline
pipeline = QCProteinPipeline(data_dir="quantum_coherence_proteins")

# Use all six proteins we have experimental data for
test_proteins = [
    "1UBQ",  # Ubiquitin
    "1LYZ",  # Lysozyme
    "1TIM",  # TIM barrel
    "1PRN",  # Porin (beta barrel)
    "3SSI",  # Subtilisin inhibitor (highly stable)
    "2LZM"   # T4 Lysozyme
]

print(f"Running analysis on {len(test_proteins)} proteins: {', '.join(test_proteins)}")

# Run the analysis with real experimental data
results = pipeline.run_complete_analysis(
    pdb_ids=test_proteins,
    chain_id='A', 
    simulate_validation=False  # Use real experimental data instead of simulated
)

# Print summary of results
for protein_id, analysis in results["analysis_results"].items():
    if "error" not in analysis:
        print(f"\n{protein_id} Stability Score: {analysis['stability_score']}")
        print(f"Number of residues analyzed: {len(analysis['qcp_values'])}")
    else:
        print(f"\nError analyzing {protein_id}: {analysis['error']}")

# Run the comparison script
print("\nRunning comparison with experimental data...")
import compare_predictions
comparison_data = compare_predictions.compare_predictions_with_experimental()

print("\nAnalysis complete! Check the quantum_coherence_proteins directory for all results.")