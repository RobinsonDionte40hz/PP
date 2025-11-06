#!/usr/bin/env python3
"""
Monitor inverse scaling investigation progress.

Checks the results directory for completed investigations and reports status.
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_progress():
    """Monitor investigation progress."""
    results_dir = Path("results/inverse_scaling")
    
    if not results_dir.exists():
        print("⚠️  Results directory not found")
        return
    
    # Expected proteins
    expected = ['1VII', '1CRN', '1UBQ', '1LYZ', '1MBN']
    
    print(f"\n{'='*70}")
    print(f"INVERSE SCALING INVESTIGATION - PROGRESS MONITOR")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nExpected proteins: {len(expected)}")
    print(f"{'='*70}\n")
    
    completed = []
    in_progress = []
    pending = []
    
    for protein_id in expected:
        result_file = results_dir / f"{protein_id}_investigation.json"
        
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                # Check if complete
                if 'results' in data:
                    completed.append(protein_id)
                    results = data['results']
                    print(f"✓ {protein_id}: COMPLETE")
                    print(f"    Size: {results['protein_size']} residues")
                    print(f"    Best Energy: {results['best_energy']:.2f} kcal/mol")
                    print(f"    Iterations: {results['iterations_completed']}")
                    print(f"    Time: {results['exploration_time_seconds']:.1f}s")
                else:
                    in_progress.append(protein_id)
                    print(f"⏳ {protein_id}: IN PROGRESS")
            except Exception as e:
                in_progress.append(protein_id)
                print(f"⏳ {protein_id}: IN PROGRESS (parsing error: {e})")
        else:
            pending.append(protein_id)
            print(f"⏸️  {protein_id}: PENDING")
        
        print()
    
    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Completed: {len(completed)}/{len(expected)} ({len(completed)/len(expected)*100:.0f}%)")
    print(f"In Progress: {len(in_progress)}")
    print(f"Pending: {len(pending)}")
    
    if completed:
        print(f"\nCompleted: {', '.join(completed)}")
    if in_progress:
        print(f"In Progress: {', '.join(in_progress)}")
    if pending:
        print(f"Pending: {', '.join(pending)}")
    
    # Estimate time remaining
    if completed:
        total_time = 0
        for protein_id in completed:
            result_file = results_dir / f"{protein_id}_investigation.json"
            with open(result_file) as f:
                data = json.load(f)
                total_time += data['results']['exploration_time_seconds']
        
        avg_time = total_time / len(completed)
        remaining = len(pending) + len(in_progress)
        
        if remaining > 0:
            est_time = avg_time * remaining
            print(f"\n📊 Estimated time remaining: {est_time/3600:.1f} hours")
            print(f"   (Based on {avg_time/60:.1f} min average per protein)")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    monitor_progress()
