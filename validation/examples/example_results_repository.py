"""
Example Usage: ResultsRepository

Demonstrates how to use ResultsRepository for centralized storage of validation results.

Examples include:
1. Basic result storage
2. Storing multiple results
3. Querying and filtering results
4. Saving predicted structures and logs
5. Generating statistics
6. Exporting to CSV
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from results_repository import ResultsRepository, TestRunMetadata


def example_1_basic_storage():
    """Example 1: Basic result storage."""
    print("\n" + "="*80)
    print("Example 1: Basic Result Storage")
    print("="*80)
    
    # Create repository
    repo = ResultsRepository(base_dir="example_results")
    
    # Create test metadata
    metadata = TestRunMetadata(
        pdb_id="1UBQ",
        timestamp=datetime.now().isoformat(),
        software_version="1.0.0",
        python_version="3.10.0",
        num_agents=10,
        iterations_per_agent=500,
        qcpp_enabled=True,
        random_seed=42,
        adaptive_config={"stuck_threshold": 10.0, "stuck_window": 30},
        execution_parameters={"temperature": 1.0, "diversity": "balanced"},
        warnings=["Minor convergence issue at iteration 250"],
        errors=[],
        execution_time_seconds=120.5,
        native_pdb_path="pdb_files/1UBQ.pdb",
        predicted_pdb_path="structures/1UBQ_predicted.pdb"
    )
    
    # Create validation metrics
    validation_metrics = {
        "rmsd": 2.5,
        "gdt_ts": 75.0,
        "tm_score": 0.65,
        "final_energy": -45.2
    }
    
    # Store result
    result_id = repo.store_result(
        pdb_id="1UBQ",
        validation_metrics=validation_metrics,
        metadata=metadata
    )
    
    print(f"✓ Stored result with ID: {result_id}")
    print(f"✓ Result saved to:")
    print(f"  - JSON database: {repo.database_file}")
    print(f"  - Markdown: {repo.markdown_file}")
    print(f"  - Metadata: {repo.metadata_dir}")


def example_2_multiple_results():
    """Example 2: Storing multiple results."""
    print("\n" + "="*80)
    print("Example 2: Storing Multiple Results")
    print("="*80)
    
    repo = ResultsRepository(base_dir="example_results")
    
    # Test proteins with different characteristics
    test_proteins = [
        {
            "pdb_id": "1CRN",
            "metrics": {"rmsd": 1.8, "gdt_ts": 85.0, "tm_score": 0.82, "final_energy": -52.3},
            "agents": 10,
            "iterations": 500,
            "qcpp": True,
            "seed": 100
        },
        {
            "pdb_id": "2MR9",
            "metrics": {"rmsd": 3.2, "gdt_ts": 68.0, "tm_score": 0.58, "final_energy": -38.5},
            "agents": 15,
            "iterations": 800,
            "qcpp": True,
            "seed": 101
        },
        {
            "pdb_id": "1VII",
            "metrics": {"rmsd": 4.5, "gdt_ts": 55.0, "tm_score": 0.52, "final_energy": -25.1},
            "agents": 10,
            "iterations": 500,
            "qcpp": False,
            "seed": 102
        }
    ]
    
    for protein in test_proteins:
        metadata = TestRunMetadata(
            pdb_id=protein["pdb_id"],
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=protein["agents"],
            iterations_per_agent=protein["iterations"],
            qcpp_enabled=protein["qcpp"],
            random_seed=protein["seed"],
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=150.0
        )
        
        result_id = repo.store_result(
            pdb_id=protein["pdb_id"],
            validation_metrics=protein["metrics"],
            metadata=metadata
        )
        
        print(f"✓ Stored {protein['pdb_id']}: RMSD={protein['metrics']['rmsd']:.2f}Å, GDT-TS={protein['metrics']['gdt_ts']:.1f}")
    
    print(f"\n✓ Total results stored: {len(test_proteins) + 1}")  # +1 from example 1


def example_3_query_and_filter():
    """Example 3: Querying and filtering results."""
    print("\n" + "="*80)
    print("Example 3: Querying and Filtering Results")
    print("="*80)
    
    repo = ResultsRepository(base_dir="example_results")
    
    # Get all results
    all_results = repo.get_all_results()
    print(f"\n✓ Total results: {len(all_results)}")
    
    # Query by RMSD range (good results only)
    good_results = repo.query_results({"max_rmsd": 3.0})
    print(f"\n✓ Good results (RMSD < 3.0Å): {len(good_results)}")
    for result in good_results:
        rmsd = result.validation_metrics.get('rmsd', 0)
        gdt = result.validation_metrics.get('gdt_ts', 0)
        print(f"  - {result.pdb_id}: RMSD={rmsd:.2f}Å, GDT-TS={gdt:.1f}")
    
    # Query by GDT-TS threshold (high quality)
    high_quality = repo.query_results({"min_gdt_ts": 70.0})
    print(f"\n✓ High quality results (GDT-TS ≥ 70): {len(high_quality)}")
    for result in high_quality:
        gdt = result.validation_metrics.get('gdt_ts', 0)
        print(f"  - {result.pdb_id}: GDT-TS={gdt:.1f}")
    
    # Query by QCPP enabled
    qcpp_results = repo.query_results({"qcpp_enabled": True})
    print(f"\n✓ Results with QCPP enabled: {len(qcpp_results)}")
    for result in qcpp_results:
        print(f"  - {result.pdb_id}")
    
    # Complex query: Good RMSD AND high GDT-TS AND QCPP enabled
    excellent_results = repo.query_results({
        "max_rmsd": 2.5,
        "min_gdt_ts": 75.0,
        "qcpp_enabled": True
    })
    print(f"\n✓ Excellent results (RMSD<2.5Å, GDT-TS≥75, QCPP): {len(excellent_results)}")
    for result in excellent_results:
        rmsd = result.validation_metrics.get('rmsd', 0)
        gdt = result.validation_metrics.get('gdt_ts', 0)
        print(f"  - {result.pdb_id}: RMSD={rmsd:.2f}Å, GDT-TS={gdt:.1f}")


def example_4_save_structures_and_logs():
    """Example 4: Saving predicted structures and execution logs."""
    print("\n" + "="*80)
    print("Example 4: Saving Predicted Structures and Logs")
    print("="*80)
    
    repo = ResultsRepository(base_dir="example_results")
    
    # Save predicted structure
    pdb_content = """REMARK   1 PREDICTED STRUCTURE BY UBF-QCPP
ATOM      1  N   MET A   1      10.000  20.000  30.000  1.00  0.00           N
ATOM      2  CA  MET A   1      11.000  21.000  31.000  1.00  0.00           C
ATOM      3  C   MET A   1      12.000  22.000  32.000  1.00  0.00           C
ATOM      4  O   MET A   1      13.000  23.000  33.000  1.00  0.00           O
END
"""
    
    structure_path = repo.save_predicted_structure(
        pdb_id="1UBQ",
        structure_content=pdb_content
    )
    print(f"✓ Saved predicted structure: {structure_path}")
    
    # Save execution log
    log_content = """2025-01-26 14:30:00 - INFO - Starting prediction for 1UBQ
2025-01-26 14:30:01 - INFO - Initialized 10 agents
2025-01-26 14:30:15 - INFO - Agent 1: Energy=-25.3 kcal/mol, RMSD=4.5Å
2025-01-26 14:31:00 - INFO - Agent 5: Energy=-45.2 kcal/mol, RMSD=2.5Å (best so far)
2025-01-26 14:32:30 - INFO - Prediction complete
2025-01-26 14:32:31 - INFO - Final RMSD: 2.5Å, GDT-TS: 75.0
"""
    
    log_path = repo.save_execution_log(
        pdb_id="1UBQ",
        log_content=log_content
    )
    print(f"✓ Saved execution log: {log_path}")


def example_5_statistics():
    """Example 5: Generating statistics."""
    print("\n" + "="*80)
    print("Example 5: Generating Statistics")
    print("="*80)
    
    repo = ResultsRepository(base_dir="example_results")
    
    # Get overall statistics
    stats = repo.get_statistics()
    
    print(f"\n✓ Summary Statistics:")
    print(f"  - Total results: {stats['total_results']}")
    print(f"  - Unique proteins: {stats['unique_proteins']}")
    print(f"  - Average RMSD: {stats['average_rmsd']:.2f} Å" if stats['average_rmsd'] else "  - Average RMSD: N/A")
    print(f"  - Average GDT-TS: {stats['average_gdt_ts']:.1f}" if stats['average_gdt_ts'] else "  - Average GDT-TS: N/A")
    print(f"  - Average TM-score: {stats['average_tm_score']:.3f}" if stats['average_tm_score'] else "  - Average TM-score: N/A")
    print(f"  - Average Energy: {stats['average_energy']:.2f} kcal/mol" if stats['average_energy'] else "  - Average Energy: N/A")
    
    print(f"\n✓ Metrics collected:")
    print(f"  - RMSD measurements: {stats['metrics_collected']['rmsd_count']}")
    print(f"  - GDT-TS measurements: {stats['metrics_collected']['gdt_ts_count']}")
    print(f"  - TM-score measurements: {stats['metrics_collected']['tm_score_count']}")
    print(f"  - Energy measurements: {stats['metrics_collected']['energy_count']}")


def example_6_csv_export():
    """Example 6: Exporting to CSV."""
    print("\n" + "="*80)
    print("Example 6: Exporting to CSV")
    print("="*80)
    
    repo = ResultsRepository(base_dir="example_results")
    
    # Export all results
    csv_path = "example_results/all_results.csv"
    repo.export_to_csv(csv_path)
    print(f"✓ Exported all results to: {csv_path}")
    
    # Export filtered results (good results only)
    csv_path_filtered = "example_results/good_results.csv"
    repo.export_to_csv(csv_path_filtered, filters={"max_rmsd": 3.0})
    print(f"✓ Exported good results (RMSD<3.0Å) to: {csv_path_filtered}")
    
    # Export QCPP-enabled results only
    csv_path_qcpp = "example_results/qcpp_results.csv"
    repo.export_to_csv(csv_path_qcpp, filters={"qcpp_enabled": True})
    print(f"✓ Exported QCPP results to: {csv_path_qcpp}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("ResultsRepository Usage Examples")
    print("="*80)
    
    # Run examples in sequence
    example_1_basic_storage()
    example_2_multiple_results()
    example_3_query_and_filter()
    example_4_save_structures_and_logs()
    example_5_statistics()
    example_6_csv_export()
    
    print("\n" + "="*80)
    print("All Examples Complete!")
    print("="*80)
    print("\nCheck the 'example_results' directory for generated files:")
    print("  - validation_database.json (machine-readable)")
    print("  - COMPREHENSIVE_TEST_RESULTS.md (human-readable)")
    print("  - structures/ (predicted PDB files)")
    print("  - logs/ (execution logs)")
    print("  - metadata/ (detailed metadata JSON)")
    print("  - *.csv (exported data)")


if __name__ == "__main__":
    main()
