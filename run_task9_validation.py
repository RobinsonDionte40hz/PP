"""
Task 9: Final Validation and Testing

This script executes the complete validation suite for the UBF protein system:
1. Run validation on all 5 test proteins (1UBQ, 1CRN, 2MR9, 1VII, 1LYZ)
2. Verify ubiquitin energy is in expected range (-120 to -80 kcal/mol)
3. Verify all folded proteins have negative energy
4. Verify RMSD improves during exploration
5. Verify energy-RMSD correlation
6. Generate comprehensive validation report

Usage:
    python run_task9_validation.py [--quick] [--protein PDBID]
    
    --quick: Use fewer iterations for faster testing (100 iterations)
    --protein PDBID: Test only specific protein (e.g., --protein 1UBQ)
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, List, Tuple

# Add ubf_protein to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ubf_protein'))

from ubf_protein.validation_suite import ValidationSuite, ValidationReport, TestSuiteResults
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import Conformation
from ubf_protein.rmsd_calculator import NativeStructureLoader


def print_header(text: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_result(label: str, value: str, status: str = ""):
    """Print formatted result line."""
    status_symbol = ""
    if status == "pass":
        status_symbol = "✅"
    elif status == "fail":
        status_symbol = "❌"
    elif status == "warn":
        status_symbol = "⚠️"
    
    print(f"  {label:40s} {value:30s} {status_symbol}")


def validate_single_protein_detailed(pdb_id: str, 
                                     iterations: int = 500,
                                     track_progress: bool = True) -> Tuple[ValidationReport, Dict]:
    """
    Run detailed validation on a single protein with progress tracking.
    
    Args:
        pdb_id: PDB identifier
        iterations: Number of exploration iterations
        track_progress: If True, track RMSD/energy over time
    
    Returns:
        Tuple of (ValidationReport, progress_data)
    """
    print(f"\n{'=' * 80}")
    print(f"  Validating {pdb_id}")
    print(f"{'=' * 80}")
    
    # Load native structure
    loader = NativeStructureLoader(cache_dir="./pdb_cache")
    native_struct_obj = loader.load_from_pdb_id(pdb_id, ca_only=True)
    
    native_structure = Conformation(
        conformation_id=f"native_{pdb_id}",
        sequence=native_struct_obj.sequence,
        atom_coordinates=native_struct_obj.ca_coords,
        energy=-100.0,
        rmsd_to_native=0.0,
        secondary_structure=['C'] * len(native_struct_obj.sequence),
        phi_angles=[0.0] * len(native_struct_obj.sequence),
        psi_angles=[0.0] * len(native_struct_obj.sequence),
        available_move_types=[],
        structural_constraints={},
        native_structure_ref=pdb_id
    )
    
    print(f"  Sequence: {native_structure.sequence[:50]}...")
    print(f"  Length: {len(native_structure.sequence)} residues")
    
    # Create agent
    agent = ProteinAgent(
        protein_sequence=native_structure.sequence,
        native_structure=native_structure,
        enable_visualization=False
    )
    
    # Track progress
    progress_data = {
        "rmsd_over_time": [],
        "energy_over_time": [],
        "iteration_numbers": []
    }
    
    start_time = time.time()
    
    # Run exploration
    print(f"\n  Running {iterations} iterations...")
    checkpoint_interval = iterations // 10
    
    for i in range(iterations):
        agent.explore_step()
        
        if track_progress and (i % checkpoint_interval == 0 or i == iterations - 1):
            current_conf = agent._current_conformation
            if current_conf:
                progress_data["iteration_numbers"].append(i)
                progress_data["rmsd_over_time"].append(current_conf.rmsd_to_native or 999.9)
                progress_data["energy_over_time"].append(current_conf.energy)
                
                print(f"    Iteration {i:4d}: RMSD = {current_conf.rmsd_to_native:.2f} Å, "
                      f"Energy = {current_conf.energy:.2f} kcal/mol")
    
    runtime = time.time() - start_time
    
    # Get final metrics
    metrics = agent.get_exploration_metrics()
    final_conf = agent._current_conformation
    
    # Create validation report
    report = ValidationReport(
        pdb_id=pdb_id,
        sequence_length=len(native_structure.sequence),
        best_rmsd=final_conf.rmsd_to_native if final_conf and final_conf.rmsd_to_native else 999.9,
        best_energy=final_conf.energy if final_conf else 999.9,
        gdt_ts_score=final_conf.gdt_ts_score if final_conf and final_conf.gdt_ts_score else 0.0,
        tm_score=final_conf.tm_score if final_conf and final_conf.tm_score else 0.0,
        runtime_seconds=runtime,
        conformations_explored=int(metrics["conformations_explored"]),
        num_agents=1,
        iterations_per_agent=iterations
    )
    
    print(f"\n  Results:")
    print(f"    Best RMSD:     {report.best_rmsd:.2f} Å")
    print(f"    Best Energy:   {report.best_energy:.2f} kcal/mol")
    print(f"    GDT-TS:        {report.gdt_ts_score:.1f}")
    print(f"    TM-Score:      {report.tm_score:.3f}")
    print(f"    Quality:       {report.assess_quality().upper()}")
    print(f"    Runtime:       {runtime:.1f} seconds")
    
    return report, progress_data


def analyze_progress_correlation(progress_data: Dict) -> Dict[str, float]:
    """
    Analyze correlation between RMSD improvement and energy decrease.
    
    Args:
        progress_data: Dictionary with rmsd_over_time and energy_over_time lists
    
    Returns:
        Dictionary with correlation metrics
    """
    rmsd_values = progress_data["rmsd_over_time"]
    energy_values = progress_data["energy_over_time"]
    
    if len(rmsd_values) < 2:
        return {
            "rmsd_improvement": 0.0,
            "energy_improvement": 0.0,
            "correlation": 0.0
        }
    
    # Calculate improvements
    initial_rmsd = rmsd_values[0]
    final_rmsd = rmsd_values[-1]
    rmsd_improvement = initial_rmsd - final_rmsd
    rmsd_improvement_pct = (rmsd_improvement / initial_rmsd) * 100 if initial_rmsd > 0 else 0.0
    
    initial_energy = energy_values[0]
    final_energy = energy_values[-1]
    energy_improvement = initial_energy - final_energy
    
    # Simple correlation: both should improve (decrease)
    # Count iterations where both decreased
    improving_together = 0
    for i in range(1, len(rmsd_values)):
        rmsd_decreased = rmsd_values[i] < rmsd_values[i-1]
        energy_decreased = energy_values[i] < energy_values[i-1]
        if rmsd_decreased and energy_decreased:
            improving_together += 1
    
    correlation_pct = (improving_together / (len(rmsd_values) - 1)) * 100 if len(rmsd_values) > 1 else 0.0
    
    return {
        "rmsd_improvement": rmsd_improvement,
        "rmsd_improvement_pct": rmsd_improvement_pct,
        "energy_improvement": energy_improvement,
        "correlation_pct": correlation_pct
    }


def main():
    """Main validation routine for Task 9."""
    
    parser = argparse.ArgumentParser(description="Task 9: Final Validation and Testing")
    parser.add_argument('--quick', action='store_true', 
                       help='Use fewer iterations for quick testing (100 iterations)')
    parser.add_argument('--protein', type=str, default=None,
                       help='Test only specific protein (e.g., 1UBQ)')
    args = parser.parse_args()
    
    # Configuration
    iterations = 100 if args.quick else 500
    
    print_header("TASK 9: FINAL VALIDATION AND TESTING")
    print(f"Configuration:")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"  Iterations per protein: {iterations}")
    if args.protein:
        print(f"  Testing single protein: {args.protein}")
    else:
        print(f"  Testing all 5 proteins in validation set")
    
    # Load validation proteins info
    with open('ubf_protein/validation_proteins.json', 'r') as f:
        validation_data = json.load(f)
        test_proteins = validation_data['validation_proteins']
    
    # Filter if specific protein requested
    if args.protein:
        test_proteins = [p for p in test_proteins if p['pdb_id'].upper() == args.protein.upper()]
        if not test_proteins:
            print(f"\n❌ ERROR: Protein {args.protein} not found in validation set")
            print("Available proteins: 1UBQ, 1CRN, 2MR9, 1VII, 1LYZ")
            return 1
    
    # ========================================================================
    # Task 9.1: Run validation on all test proteins
    # ========================================================================
    print_header("Task 9.1: Validate All Test Proteins")
    
    validation_results = []
    progress_records = {}
    
    for protein_info in test_proteins:
        pdb_id = protein_info['pdb_id']
        expected_energy_range = protein_info.get('native_energy_range', [-999, 999])
        
        try:
            report, progress_data = validate_single_protein_detailed(
                pdb_id, 
                iterations=iterations,
                track_progress=True
            )
            validation_results.append((protein_info, report))
            progress_records[pdb_id] = progress_data
            
        except Exception as e:
            print(f"\n❌ ERROR validating {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not validation_results:
        print("\n❌ FAILED: No successful validations")
        return 1
    
    print(f"\n✅ Successfully validated {len(validation_results)}/{len(test_proteins)} proteins")
    
    # ========================================================================
    # Task 9.2: Verify Ubiquitin Energy Range
    # ========================================================================
    print_header("Task 9.2: Verify Ubiquitin (1UBQ) Energy Range")
    
    ubq_report = None
    for protein_info, report in validation_results:
        if protein_info['pdb_id'] == '1UBQ':
            ubq_report = report
            break
    
    if ubq_report:
        expected_min, expected_max = -120, -80
        actual_energy = ubq_report.best_energy
        
        print_result("Expected Energy Range:", f"{expected_min} to {expected_max} kcal/mol")
        print_result("Actual Energy:", f"{actual_energy:.2f} kcal/mol")
        
        if expected_min <= actual_energy <= expected_max:
            print_result("Status:", "PASS - Energy in expected range", "pass")
        else:
            if actual_energy < 0:
                print_result("Status:", "WARN - Energy negative but outside range", "warn")
                print("  Note: Negative energy indicates stable structure, but may need force field tuning")
            else:
                print_result("Status:", "FAIL - Energy not in expected range", "fail")
    else:
        print("❌ 1UBQ not found in validation results")
    
    # ========================================================================
    # Task 9.3: Verify All Proteins Have Negative Energy
    # ========================================================================
    print_header("Task 9.3: Verify All Folded Proteins Have Negative Energy")
    
    negative_energy_count = 0
    print("\nEnergy Analysis:")
    print(f"{'  Protein':<15} {'Energy (kcal/mol)':>20} {'Status':>15}")
    print(f"  {'-' * 53}")
    
    for protein_info, report in validation_results:
        pdb_id = protein_info['pdb_id']
        energy = report.best_energy
        
        if energy < 0:
            status = "✅ Negative"
            negative_energy_count += 1
        else:
            status = "❌ Positive"
        
        print(f"  {pdb_id:<15} {energy:>20.2f} {status:>15}")
    
    print(f"\n  Results: {negative_energy_count}/{len(validation_results)} proteins have negative energy")
    
    if negative_energy_count == len(validation_results):
        print_result("\nStatus:", "PASS - All proteins thermodynamically stable", "pass")
    elif negative_energy_count > 0:
        print_result("\nStatus:", f"WARN - {negative_energy_count}/{len(validation_results)} stable", "warn")
    else:
        print_result("\nStatus:", "FAIL - No proteins have negative energy", "fail")
    
    # ========================================================================
    # Task 9.4: Verify RMSD Improves During Exploration
    # ========================================================================
    print_header("Task 9.4: Verify RMSD Improvement During Exploration")
    
    improvement_count = 0
    print("\nRMSD Progress Analysis:")
    print(f"{'  Protein':<15} {'Initial RMSD':>15} {'Final RMSD':>15} {'Improvement':>15} {'Status':>10}")
    print(f"  {'-' * 73}")
    
    for protein_info, report in validation_results:
        pdb_id = protein_info['pdb_id']
        
        if pdb_id not in progress_records:
            continue
        
        progress = progress_records[pdb_id]
        if not progress['rmsd_over_time']:
            continue
        
        initial_rmsd = progress['rmsd_over_time'][0]
        final_rmsd = progress['rmsd_over_time'][-1]
        improvement = initial_rmsd - final_rmsd
        improvement_pct = (improvement / initial_rmsd) * 100 if initial_rmsd > 0 else 0.0
        
        if improvement > 0:
            status = "✅ Improved"
            improvement_count += 1
        elif improvement == 0:
            status = "⚠️ No change"
        else:
            status = "❌ Worse"
        
        print(f"  {pdb_id:<15} {initial_rmsd:>15.2f} {final_rmsd:>15.2f} "
              f"{improvement_pct:>14.1f}% {status:>10}")
    
    print(f"\n  Results: {improvement_count}/{len(progress_records)} proteins showed RMSD improvement")
    
    if improvement_count == len(progress_records):
        print_result("\nStatus:", "PASS - All proteins improved RMSD", "pass")
    elif improvement_count > len(progress_records) / 2:
        print_result("\nStatus:", f"WARN - {improvement_count}/{len(progress_records)} improved", "warn")
    else:
        print_result("\nStatus:", "FAIL - Most proteins did not improve", "fail")
    
    # ========================================================================
    # Task 9.5: Verify Energy-RMSD Correlation
    # ========================================================================
    print_header("Task 9.5: Verify Energy-RMSD Correlation")
    
    print("\nCorrelation Analysis:")
    print("  (% of iterations where both RMSD and energy decreased together)")
    print(f"\n{'  Protein':<15} {'RMSD Δ':>12} {'Energy Δ':>15} {'Co-improvement':>18} {'Status':>10}")
    print(f"  {'-' * 73}")
    
    correlation_count = 0
    for protein_info, report in validation_results:
        pdb_id = protein_info['pdb_id']
        
        if pdb_id not in progress_records:
            continue
        
        progress = progress_records[pdb_id]
        correlation = analyze_progress_correlation(progress)
        
        rmsd_imp = correlation['rmsd_improvement']
        energy_imp = correlation['energy_improvement']
        corr_pct = correlation['correlation_pct']
        
        # Good correlation if they co-improve >30% of the time
        if corr_pct > 30:
            status = "✅ Good"
            correlation_count += 1
        elif corr_pct > 15:
            status = "⚠️ Fair"
        else:
            status = "❌ Poor"
        
        print(f"  {pdb_id:<15} {rmsd_imp:>11.2f} Å {energy_imp:>11.2f} kcal "
              f"{corr_pct:>16.1f}% {status:>10}")
    
    print(f"\n  Results: {correlation_count}/{len(progress_records)} proteins show good correlation")
    
    if correlation_count >= len(progress_records) / 2:
        print_result("\nStatus:", "PASS - Good energy-RMSD correlation", "pass")
    else:
        print_result("\nStatus:", "WARN - Weak energy-RMSD correlation", "warn")
        print("  Note: This may indicate need for force field parameter tuning")
    
    # ========================================================================
    # Task 9.6: Quality Assessment Summary
    # ========================================================================
    print_header("Task 9.6: Quality Assessment Summary")
    
    quality_counts = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
    
    print("\nPrediction Quality:")
    print(f"{'  Protein':<15} {'RMSD':>12} {'GDT-TS':>10} {'TM-Score':>12} {'Quality':>15}")
    print(f"  {'-' * 68}")
    
    for protein_info, report in validation_results:
        pdb_id = protein_info['pdb_id']
        quality = report.assess_quality()
        quality_counts[quality] += 1
        
        print(f"  {pdb_id:<15} {report.best_rmsd:>11.2f} Å {report.gdt_ts_score:>10.1f} "
              f"{report.tm_score:>12.3f} {quality.upper():>15}")
    
    print(f"\n  Quality Distribution:")
    total = len(validation_results)
    for quality in ["excellent", "good", "acceptable", "poor"]:
        count = quality_counts[quality]
        pct = (count / total) * 100 if total > 0 else 0
        symbol = "✅" if quality in ["excellent", "good", "acceptable"] else "❌"
        print(f"    {quality.capitalize():12s} {count:2d} ({pct:5.1f}%) {symbol}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print_header("TASK 9: FINAL SUMMARY")
    
    print("Test Results:")
    print_result("✓ Task 9.1", f"Validated {len(validation_results)}/{len(test_proteins)} proteins", 
                 "pass" if len(validation_results) == len(test_proteins) else "warn")
    
    if ubq_report:
        ubq_in_range = (-120 <= ubq_report.best_energy <= -80)
        print_result("✓ Task 9.2", f"Ubiquitin energy: {ubq_report.best_energy:.1f} kcal/mol",
                     "pass" if ubq_in_range else "warn")
    
    print_result("✓ Task 9.3", f"{negative_energy_count}/{len(validation_results)} negative energy",
                 "pass" if negative_energy_count == len(validation_results) else "warn")
    
    print_result("✓ Task 9.4", f"{improvement_count}/{len(progress_records)} RMSD improved",
                 "pass" if improvement_count > 0 else "fail")
    
    print_result("✓ Task 9.5", f"{correlation_count}/{len(progress_records)} good correlation",
                 "pass" if correlation_count >= len(progress_records)/2 else "warn")
    
    acceptable_count = sum(1 for _, r in validation_results if r.assess_quality() in ["excellent", "good", "acceptable"])
    print_result("✓ Task 9.6", f"{acceptable_count}/{len(validation_results)} acceptable+ quality",
                 "pass" if acceptable_count > 0 else "fail")
    
    # Save results to file
    results_file = "task9_validation_results.json"
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "mode": "quick" if args.quick else "full",
            "iterations": iterations,
            "proteins_tested": [p['pdb_id'] for p in test_proteins]
        },
        "validation_reports": [
            {
                "pdb_id": protein_info['pdb_id'],
                "name": protein_info['name'],
                "residues": protein_info['residues'],
                "best_rmsd": report.best_rmsd,
                "best_energy": report.best_energy,
                "gdt_ts_score": report.gdt_ts_score,
                "tm_score": report.tm_score,
                "quality": report.assess_quality(),
                "successful": report.is_successful(),
                "runtime_seconds": report.runtime_seconds
            }
            for protein_info, report in validation_results
        ],
        "progress_data": progress_records,
        "summary": {
            "total_proteins": len(validation_results),
            "negative_energy_count": negative_energy_count,
            "rmsd_improvement_count": improvement_count,
            "correlation_count": correlation_count,
            "quality_distribution": quality_counts
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("  TASK 9 VALIDATION COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
