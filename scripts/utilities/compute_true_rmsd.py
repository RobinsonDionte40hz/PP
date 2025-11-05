#!/usr/bin/env python3
"""
Compute True RMSD using Kabsch Alignment
=========================================

Calculate true RMSD values for all 20 proteins by performing optimal 
superposition (Kabsch/SVD alignment) between predicted and native structures.

Updates the phi_reanalysis_results.json with accurate RMSD values.

Usage:
    python compute_true_rmsd.py
    python compute_true_rmsd.py --output updated_results.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
import numpy as np

try:
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.PDBList import PDBList
except ImportError:
    print("ERROR: BioPython required. Install: pip install biopython numpy")
    sys.exit(1)


def download_pdb(pdb_id: str, cache_dir: str = "pdb_cache") -> Path:
    """Download PDB file from RCSB if not cached"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    # Check multiple possible file names
    possible_files = [
        cache_path / f"{pdb_id.upper()}.pdb",
        cache_path / f"{pdb_id.lower()}.pdb",
        cache_path / f"pdb{pdb_id.lower()}.ent"
    ]
    
    for pdb_file in possible_files:
        if pdb_file.exists():
            print(f"  ✓ Using cached: {pdb_file}")
            return pdb_file
    
    # Download if not found
    print(f"  Downloading {pdb_id} from RCSB...")
    pdbl = PDBList()
    try:
        downloaded = pdbl.retrieve_pdb_file(
            pdb_id.lower(),
            file_format='pdb',
            pdir=str(cache_path)
        )
        
        # PDBList downloads as pdb{id}.ent - rename to {ID}.pdb
        downloaded_path = Path(downloaded)
        if downloaded_path.exists():
            target_path = cache_path / f"{pdb_id.upper()}.pdb"
            if not target_path.exists():
                downloaded_path.rename(target_path)
            return target_path
        else:
            raise FileNotFoundError(f"Download failed: {downloaded}")
            
    except Exception as e:
        print(f"  ERROR downloading {pdb_id}: {e}")
        return None


def load_ca_coords(pdb_path: Path) -> np.ndarray:
    """Load CA coordinates from PDB file"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
        
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
        
        return np.array(coords) if coords else None
            
    except Exception as e:
        print(f"  ERROR loading {pdb_path}: {e}")
        return None


def calculate_true_rmsd(predicted: np.ndarray, native: np.ndarray) -> float:
    """
    Calculate RMSD after optimal superposition using Kabsch algorithm.
    
    Steps:
    1. Center both structures at origin
    2. Calculate covariance matrix H = P^T @ N
    3. Perform SVD: H = U Σ V^T
    4. Calculate rotation matrix: R = V @ U^T
    5. Apply rotation to predicted structure
    6. Calculate RMSD between aligned structures
    
    Args:
        predicted: Predicted coordinates (N x 3)
        native: Native coordinates (N x 3)
    
    Returns:
        RMSD in Angstroms after optimal alignment
    """
    if len(predicted) != len(native):
        min_len = min(len(predicted), len(native))
        print(f"  WARNING: Coordinate mismatch ({len(predicted)} vs {len(native)}), using first {min_len} residues")
        predicted = predicted[:min_len]
        native = native[:min_len]
    
    if len(predicted) == 0:
        return float('inf')
    
    # Step 1: Center both structures
    pred_centroid = np.mean(predicted, axis=0)
    nat_centroid = np.mean(native, axis=0)
    pred_centered = predicted - pred_centroid
    nat_centered = native - nat_centroid
    
    # Step 2: Calculate covariance matrix
    H = pred_centered.T @ nat_centered
    
    # Step 3: Perform SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Step 4: Calculate rotation matrix
    R = Vt.T @ U.T
    
    # Check for reflection (det(R) should be +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Step 5: Apply rotation
    pred_rotated = pred_centered @ R
    
    # Step 6: Calculate RMSD
    diff = pred_rotated - nat_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


def compute_all_rmsd_values(
    results_file: str = "phi_reanalysis_results.json",
    predicted_dir: str = "results/predicted_structures",
    output_file: str = None
) -> dict:
    """
    Compute true RMSD for all proteins in results file.
    
    Args:
        results_file: Path to phi_reanalysis_results.json
        predicted_dir: Directory containing predicted PDB files
        output_file: Optional output file (defaults to results_file)
    
    Returns:
        Updated results dictionary with true RMSD values
    """
    # Load existing results
    print(f"Loading results from {results_file}...")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    print(f"Found {len(results)} proteins to process\n")
    
    predicted_path = Path(predicted_dir)
    if not predicted_path.exists():
        print(f"ERROR: Predicted structures directory not found: {predicted_path}")
        return data
    
    # Process each protein
    success_count = 0
    failure_count = 0
    
    for i, result in enumerate(results, 1):
        pdb_id = result['pdb_id']
        print(f"\n[{i}/{len(results)}] Processing {pdb_id}...")
        
        # Load native structure
        native_pdb = download_pdb(pdb_id)
        if native_pdb is None:
            print(f"  ✗ Failed to load native structure")
            failure_count += 1
            result['true_rmsd'] = None
            result['true_rmsd_error'] = "Failed to load native structure"
            continue
        
        native_coords = load_ca_coords(native_pdb)
        if native_coords is None:
            print(f"  ✗ Failed to parse native coordinates")
            failure_count += 1
            result['true_rmsd'] = None
            result['true_rmsd_error'] = "Failed to parse native coordinates"
            continue
        
        # Load predicted structure
        predicted_pdb = predicted_path / f"{pdb_id}_predicted.pdb"
        if not predicted_pdb.exists():
            print(f"  ✗ Predicted structure not found: {predicted_pdb}")
            failure_count += 1
            result['true_rmsd'] = None
            result['true_rmsd_error'] = f"Predicted structure not found"
            continue
        
        predicted_coords = load_ca_coords(predicted_pdb)
        if predicted_coords is None:
            print(f"  ✗ Failed to parse predicted coordinates")
            failure_count += 1
            result['true_rmsd'] = None
            result['true_rmsd_error'] = "Failed to parse predicted coordinates"
            continue
        
        # Calculate true RMSD
        try:
            true_rmsd = calculate_true_rmsd(predicted_coords, native_coords)
            result['true_rmsd'] = float(round(true_rmsd, 3))
            result['original_rmsd'] = float(result['predicted_rmsd'])
            result['rmsd_difference'] = float(round(abs(true_rmsd - result['predicted_rmsd']), 3))
            result['true_rmsd_error'] = None
            
            print(f"  ✓ True RMSD: {true_rmsd:.3f} Å")
            print(f"    Original: {result['predicted_rmsd']:.3f} Å")
            print(f"    Difference: {result['rmsd_difference']:.3f} Å")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error calculating RMSD: {e}")
            failure_count += 1
            result['true_rmsd'] = None
            result['true_rmsd_error'] = str(e)
    
    # Update metadata
    data['true_rmsd_computed'] = True
    data['true_rmsd_success_count'] = success_count
    data['true_rmsd_failure_count'] = failure_count
    
    # Generate summary statistics
    true_rmsds = [r['true_rmsd'] for r in results if r.get('true_rmsd') is not None]
    if true_rmsds:
        data['true_rmsd_statistics'] = {
            'mean': round(np.mean(true_rmsds), 3),
            'std': round(np.std(true_rmsds), 3),
            'min': round(np.min(true_rmsds), 3),
            'max': round(np.max(true_rmsds), 3),
            'median': round(np.median(true_rmsds), 3)
        }
        
        # Separate by protein type
        ordered_rmsds = [r['true_rmsd'] for r in results 
                        if r.get('true_rmsd') is not None and r['protein_type'] == 'ordered']
        disordered_rmsds = [r['true_rmsd'] for r in results 
                           if r.get('true_rmsd') is not None and r['protein_type'] == 'disordered']
        
        if ordered_rmsds:
            data['ordered_true_rmsd'] = {
                'mean': round(np.mean(ordered_rmsds), 3),
                'std': round(np.std(ordered_rmsds), 3),
                'n': len(ordered_rmsds)
            }
        
        if disordered_rmsds:
            data['disordered_true_rmsd'] = {
                'mean': round(np.mean(disordered_rmsds), 3),
                'std': round(np.std(disordered_rmsds), 3),
                'n': len(disordered_rmsds)
            }
    
    # Save updated results
    output_path = output_file if output_file else results_file
    print(f"\n\nSaving updated results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TRUE RMSD CALCULATION SUMMARY")
    print("="*60)
    print(f"Total proteins: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    
    if true_rmsds:
        print(f"\nOverall Statistics:")
        print(f"  Mean RMSD: {data['true_rmsd_statistics']['mean']:.3f} ± {data['true_rmsd_statistics']['std']:.3f} Å")
        print(f"  Range: {data['true_rmsd_statistics']['min']:.3f} - {data['true_rmsd_statistics']['max']:.3f} Å")
        print(f"  Median: {data['true_rmsd_statistics']['median']:.3f} Å")
        
        if 'ordered_true_rmsd' in data:
            print(f"\nOrdered proteins:")
            print(f"  Mean RMSD: {data['ordered_true_rmsd']['mean']:.3f} ± {data['ordered_true_rmsd']['std']:.3f} Å (N={data['ordered_true_rmsd']['n']})")
        
        if 'disordered_true_rmsd' in data:
            print(f"\nDisordered proteins:")
            print(f"  Mean RMSD: {data['disordered_true_rmsd']['mean']:.3f} ± {data['disordered_true_rmsd']['std']:.3f} Å (N={data['disordered_true_rmsd']['n']})")
    
    print("="*60 + "\n")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Compute true RMSD using Kabsch alignment for all proteins"
    )
    parser.add_argument(
        '--results',
        default='phi_reanalysis_results.json',
        help='Input results file (default: phi_reanalysis_results.json)'
    )
    parser.add_argument(
        '--predicted-dir',
        default='results/predicted_structures',
        help='Directory with predicted PDB files (default: results/predicted_structures)'
    )
    parser.add_argument(
        '--output',
        help='Output file (default: overwrite input file)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.results).exists():
        print(f"ERROR: Results file not found: {args.results}")
        sys.exit(1)
    
    compute_all_rmsd_values(
        results_file=args.results,
        predicted_dir=args.predicted_dir,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
