#!/usr/bin/env python3
"""
Batch Protein Testing Script

Tests multiple proteins sequentially and generates a comparison report.
Perfect for validating the system across diverse proteins.

Usage:
  python batch_test_proteins.py                    # Test all available proteins
  python batch_test_proteins.py --quick            # Quick test on small proteins
  python batch_test_proteins.py --ids 1UBQ 1CRN   # Test specific proteins
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Import the main testing function
from test_protein import (
    download_pdb, 
    load_sequence_from_pdb, 
    run_protein_test,
    AVAILABLE_PROTEINS
)


def run_batch_test(pdb_ids: List[str], output_file: Optional[str] = None) -> List[Dict]:
    """Run tests on multiple proteins and collect results."""
    
    print("\n" + "="*70)
    print("BATCH PROTEIN TESTING")
    print("="*70)
    print(f"\nTesting {len(pdb_ids)} proteins: {', '.join(pdb_ids)}")
    print(f"This will take approximately {len(pdb_ids) * 15 / 60:.1f} minutes\n")
    
    results = []
    start_time = time.time()
    
    for i, pdb_id in enumerate(pdb_ids, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(pdb_ids)}: {pdb_id}")
        print(f"{'='*70}")
        
        try:
            # Download PDB
            pdb_file = download_pdb(pdb_id)
            if not pdb_file or not pdb_file.exists():
                print(f"❌ Failed to get PDB file for {pdb_id}")
                results.append({
                    'pdb_id': pdb_id,
                    'status': 'failed',
                    'error': 'PDB download failed'
                })
                continue
            
            # Extract sequence
            sequence = load_sequence_from_pdb(pdb_file)
            
            # Run test
            result = run_protein_test(
                sequence=sequence,
                pdb_file=pdb_file,
                pdb_id=pdb_id
            )
            
            result['status'] = 'success'
            results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error testing {pdb_id}: {e}")
            results.append({
                'pdb_id': pdb_id,
                'status': 'failed',
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Generate comparison report
    print("\n" + "="*70)
    print("BATCH TEST COMPARISON")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"  - Total Proteins: {len(pdb_ids)}")
    print(f"  - Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"  - Failed: {sum(1 for r in results if r.get('status') == 'failed')}")
    print(f"  - Total Time: {total_time / 60:.1f} minutes")
    
    # Results table
    print(f"\n{'PDB':<6} {'Name':<15} {'Residues':<10} {'Energy':<15} {'RMSD':<12} {'Quality':<10}")
    print("-" * 70)
    
    for result in results:
        if result.get('status') == 'success':
            pdb_id = result['protein_info']['pdb_id']
            seq_len = result['protein_info']['sequence_length']
            energy = result['exploration_results']['best_energy']
            rmsd = result['exploration_results']['estimated_rmsd']
            quality = result['exploration_results']['rmsd_quality']
            
            name = AVAILABLE_PROTEINS.get(pdb_id, {}).get('name', 'Unknown')
            
            print(f"{pdb_id:<6} {name:<15} {seq_len:<10} {energy:<15.2f} {rmsd:<12.2f} {quality:<10}")
        else:
            pdb_id = result['pdb_id']
            print(f"{pdb_id:<6} {'FAILED':<15} {'-':<10} {'-':<15} {'-':<12} {'-':<10}")
    
    print("-" * 70)
    
    # Energy distribution
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        energies = [r['exploration_results']['best_energy'] for r in successful]
        rmsds = [r['exploration_results']['estimated_rmsd'] for r in successful]
        
        print(f"\n📈 Statistics:")
        print(f"  Energy Range: {min(energies):.2f} to {max(energies):.2f} kcal/mol")
        print(f"  Avg Energy: {sum(energies) / len(energies):.2f} kcal/mol")
        print(f"  RMSD Range: {min(rmsds):.2f} to {max(rmsds):.2f} Å")
        print(f"  Avg RMSD: {sum(rmsds) / len(rmsds):.2f} Å")
        
        # Quality breakdown
        quality_counts = {}
        for r in successful:
            quality = r['exploration_results']['rmsd_quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        print(f"\n🎯 Quality Distribution:")
        for quality, count in sorted(quality_counts.items()):
            pct = (count / len(successful)) * 100
            print(f"  {quality}: {count} ({pct:.1f}%)")
        
        # Best performer
        best_result = min(successful, key=lambda r: r['exploration_results']['best_energy'])
        best_pdb = best_result['protein_info']['pdb_id']
        best_energy = best_result['exploration_results']['best_energy']
        
        print(f"\n🏆 Best Performer: {best_pdb}")
        print(f"  Energy: {best_energy:.2f} kcal/mol")
        print(f"  RMSD: {best_result['exploration_results']['estimated_rmsd']:.2f} Å")
        
        # RMSE validation
        rmse_results = [r for r in successful if r.get('rmse_validation') is not None]
        if rmse_results:
            print(f"\n🔬 RMSE Validation ({len(rmse_results)} proteins):")
            for r in rmse_results:
                pdb = r['protein_info']['pdb_id']
                rmse = r['rmse_validation']
                print(f"  {pdb}: Temp={rmse['temperature_rmse']:.2f}°C, "
                      f"ΔG={rmse['dg_rmse']:.2f} kcal/mol ({rmse['quality']})")
    
    print("\n" + "="*70)
    
    # Save combined results
    if output_file is None:
        output_file = f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    combined_output = {
        'batch_info': {
            'proteins_tested': len(pdb_ids),
            'successful': sum(1 for r in results if r.get('status') == 'success'),
            'failed': sum(1 for r in results if r.get('status') == 'failed'),
            'total_time_minutes': total_time / 60,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(combined_output, f, indent=2)
    
    print(f"✓ Combined results saved to: {output_path}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch test multiple proteins and generate comparison report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_test_proteins.py                      # Test all available
  python batch_test_proteins.py --quick              # Test small proteins only
  python batch_test_proteins.py --ids 1UBQ 1CRN     # Test specific proteins
  python batch_test_proteins.py --output results.json  # Custom output file
        """
    )
    
    parser.add_argument('--ids', nargs='+', help='Specific PDB IDs to test')
    parser.add_argument('--quick', action='store_true', help='Test only small proteins (<50 residues)')
    parser.add_argument('--medium', action='store_true', help='Test only medium proteins (50-100 residues)')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Determine which proteins to test
    if args.ids:
        pdb_ids = [pid.upper() for pid in args.ids]
        print(f"Testing specific proteins: {pdb_ids}")
    elif args.quick:
        pdb_ids = [pid for pid, info in AVAILABLE_PROTEINS.items() if info['residues'] < 50]
        print(f"Testing small proteins: {pdb_ids}")
    elif args.medium:
        pdb_ids = [pid for pid, info in AVAILABLE_PROTEINS.items() if 50 <= info['residues'] < 100]
        print(f"Testing medium proteins: {pdb_ids}")
    else:
        pdb_ids = list(AVAILABLE_PROTEINS.keys())
        print("Testing all available proteins")
    
    if not pdb_ids:
        print("❌ No proteins to test")
        sys.exit(1)
    
    # Run batch test
    run_batch_test(pdb_ids, args.output)


if __name__ == "__main__":
    main()
