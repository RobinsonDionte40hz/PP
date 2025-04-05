from qc_pipeline import QCProteinPipeline

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