"""
CLI Entry Point for Prediction Engine

This module provides command-line interface for protein structure prediction.
It wraps the public API for easy command-line usage.

Usage:
    predict-protein --sequence ACDEFGHIKLMNPQRSTVWY
    predict-protein --pdb 1UBQ
    predict-protein --help
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="predict-protein",
        description="EmergentFolds Protein Structure Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict structure from sequence
    predict-protein --sequence MQIFVKTLTGKTITLEVEPS
    
    # Predict from PDB ID (fetches native for RMSD)
    predict-protein --pdb 1UBQ
    
    # Quick test mode (faster, less accurate)
    predict-protein --sequence ACDEFGHIK --quick
    
    # Screen for aggregation
    predict-protein --sequence MQIFVKTLTGK --screen
    
    # Save results to file
    predict-protein --sequence MQIFVKT --output results/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sequence", "-s",
        type=str,
        help="Amino acid sequence to predict"
    )
    input_group.add_argument(
        "--pdb",
        type=str,
        help="PDB ID to fetch and predict (also used for RMSD validation)"
    )
    input_group.add_argument(
        "--fasta",
        type=str,
        help="Path to FASTA file"
    )
    
    # Configuration options
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=10,
        help="Number of exploration agents (default: 10)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=500,
        help="Maximum iterations (default: 500)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode (fewer agents/iterations)"
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Disable quantum refinement phase"
    )
    parser.add_argument(
        "--qcpp",
        choices=["none", "default", "high_performance", "high_accuracy"],
        default="default",
        help="QCPP configuration preset"
    )
    
    # Screening options
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Run aggregation screening instead of full prediction"
    )
    parser.add_argument(
        "--screen-mode",
        choices=["fast", "balanced", "thorough"],
        default="balanced",
        help="Screening mode (default: balanced)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--format",
        choices=["pdb", "json", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Import API (lazy import for faster --help)
    from ubf_protein.api import (
        PredictionRunner,
        PredictionConfig,
        AggregationScreener,
        ScreeningConfig,
        PDBExporter,
        JSONExporter,
    )
    
    # Get sequence
    sequence = None
    native_pdb = None
    
    if args.sequence:
        sequence = args.sequence.upper().strip()
    elif args.pdb:
        native_pdb = args.pdb.upper()
        # Fetch sequence from PDB
        try:
            from ubf_protein.rmsd_calculator import NativeStructureLoader
            loader = NativeStructureLoader()
            native_data = loader.load_native(native_pdb)
            sequence = native_data.get('sequence', '')
            print(f"Loaded {native_pdb}: {len(sequence)} residues")
        except Exception as e:
            print(f"Error loading PDB {native_pdb}: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.fasta:
        # Parse FASTA file
        try:
            with open(args.fasta) as f:
                lines = f.readlines()
            sequence = ''.join(
                line.strip() for line in lines 
                if not line.startswith('>')
            ).upper()
        except Exception as e:
            print(f"Error reading FASTA: {e}", file=sys.stderr)
            sys.exit(1)
    
    if not sequence:
        print("No sequence provided", file=sys.stderr)
        sys.exit(1)
    
    # Screening mode
    if args.screen:
        print(f"Screening sequence ({len(sequence)} residues)...")
        
        config_map = {
            "fast": ScreeningConfig(window_size=5, threshold=0.6),
            "balanced": ScreeningConfig(window_size=7, threshold=0.5),
            "thorough": ScreeningConfig(window_size=9, threshold=0.4),
        }
        config = config_map[args.screen_mode]
        
        screener = AggregationScreener()
        results = screener.screen(sequence, config)
        
        print(f"\n{'='*60}")
        print(f"SCREENING RESULTS")
        print(f"{'='*60}")
        print(f"Sequence length: {results.sequence_length}")
        print(f"Risk level: {results.risk_level.value.upper()}")
        print(f"Aggregation score: {results.aggregation_score:.3f}")
        print(f"Passes screening: {'YES' if results.passes_screening else 'NO'}")
        print(f"\nDetailed scores:")
        print(f"  Energy:      {results.energy_score:.3f}")
        print(f"  Structure:   {results.structure_score:.3f}")
        print(f"  Hydrophobic: {results.hydrophobic_score:.3f}")
        print(f"  Compactness: {results.compactness_score:.3f}")
        
        if results.risk_factors:
            print(f"\nRisk factors:")
            for factor in results.risk_factors:
                print(f"  - {factor}")
        
        # Save results
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "screening_results.json", "w") as f:
                json.dump(results.to_dict(), f, indent=2)
            print(f"\nResults saved to {output_dir}")
        
        sys.exit(0 if results.passes_screening else 1)
    
    # Prediction mode
    print(f"Predicting structure for {len(sequence)} residues...")
    
    # Configure
    if args.quick:
        agents = 4
        iterations = 100
        enable_refinement = False
        qcpp_config = "none"
    else:
        agents = args.agents
        iterations = args.iterations
        enable_refinement = not args.no_refinement
        qcpp_config = args.qcpp
    
    config = PredictionConfig(
        sequence=sequence,
        native_pdb=native_pdb,
        agents=agents,
        iterations=iterations,
        enable_refinement=enable_refinement,
        qcpp_config=qcpp_config,
    )
    
    # Progress callback
    def progress_callback(update):
        if args.verbose or update.iteration % 100 == 0:
            print(f"[{update.phase}] Iteration {update.iteration}/{update.total_iterations} "
                  f"({update.percentage:.1f}%) - {update.message}")
    
    # Run prediction
    runner = PredictionRunner(config)
    
    try:
        results = runner.run(progress_callback=progress_callback)
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Runtime: {results.runtime_seconds:.1f}s")
    
    if results.metrics:
        print(f"\nMetrics:")
        if results.metrics.rmsd is not None:
            print(f"  RMSD: {results.metrics.rmsd:.2f} Å")
        if results.metrics.energy_total is not None:
            print(f"  Energy: {results.metrics.energy_total:.2f} kcal/mol")
        if results.metrics.qcp_score is not None:
            print(f"  QCP Score: {results.metrics.qcp_score:.3f}")
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.format in ("pdb", "both"):
            pdb_exporter = PDBExporter()
            pdb_path = pdb_exporter.export(results, str(output_dir / "prediction.pdb"))
            print(f"\nPDB saved to {pdb_path}")
        
        if args.format in ("json", "both"):
            json_exporter = JSONExporter()
            json_path = json_exporter.export(results, str(output_dir / "prediction.json"))
            print(f"JSON saved to {json_path}")
    else:
        # Print PDB to stdout if no output specified
        if not args.verbose:
            print(f"\nPDB Output:")
            print(results.pdb_string[:500] + "..." if len(results.pdb_string) > 500 else results.pdb_string)
    
    print(f"\nPrediction complete!")


if __name__ == "__main__":
    main()
