#!/usr/bin/env python3
"""
Analyze protein structure to find best matching Platonic solid geometry.

Usage:
    python analyze_protein_geometry.py --pdb 1TIM
    python analyze_protein_geometry.py --pdb 1UBQ
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser

# Add ubf_protein to path
sys.path.insert(0, str(Path(__file__).parent))

from ubf_protein.geometric_scoring import analyze_all_geometries


def load_protein_structure(pdb_file: str):
    """Load CA coordinates from PDB file."""
    parser = PDBParser(QUIET=True)
    
    # Handle both .pdb and .ent files
    if not Path(pdb_file).exists():
        # Try pdb_cache directory
        cache_path = Path("pdb_cache") / f"pdb{pdb_file.lower()}.ent"
        if cache_path.exists():
            pdb_file = str(cache_path)
        else:
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    structure = parser.get_structure("protein", pdb_file)
    
    # Extract CA coordinates from first model, first chain
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):
                    coords.append(residue['CA'].get_coord())
            break  # Only first chain
        break  # Only first model
    
    return [np.array(coord) for coord in coords]


def print_geometry_analysis(pdb_id: str, results: dict):
    """Pretty print geometry analysis results."""
    print(f"\n{'='*70}")
    print(f"GEOMETRIC ANALYSIS: {pdb_id}")
    print(f"{'='*70}\n")
    
    print("📊 Correlation Strength by Platonic Solid:\n")
    
    # Print ranked geometries with visual bars
    for i, (geometry, correlation) in enumerate(results['ranked_geometries']):
        bar_length = int(correlation / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        marker = '⭐' if i == 0 else '  '
        
        print(f"{marker} {geometry.capitalize():15} {correlation:6.2f}% {bar}")
    
    print(f"\n{'='*70}")
    print(f"✅ BEST MATCH: {results['best_match'].upper()}")
    print(f"   Correlation: {results['best_correlation']:.2f}%")
    print(f"{'='*70}\n")
    
    # Print individual correlations
    print("Detailed Correlations:")
    for geometry, correlation in sorted(results['correlations'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {geometry.capitalize():15} → {correlation:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze protein geometric similarity to Platonic solids')
    parser.add_argument('--pdb', required=True, help='PDB ID (e.g., 1TIM, 1UBQ)')
    args = parser.parse_args()
    
    pdb_id = args.pdb.upper()
    
    print(f"\n🔍 Loading protein structure: {pdb_id}...")
    
    try:
        coords = load_protein_structure(pdb_id)
        print(f"✓ Loaded {len(coords)} CA atoms")
        
        print(f"\n🧮 Analyzing geometric correlations...")
        results = analyze_all_geometries(coords)
        
        print_geometry_analysis(pdb_id, results)
        
        # Recommendation
        if results['best_correlation'] > 60:
            print(f"💡 RECOMMENDATION: Use --target-geometry {results['best_match']} for guided exploration")
        elif results['best_correlation'] > 40:
            print(f"💡 RECOMMENDATION: Moderate match with {results['best_match']}, may provide some guidance")
        else:
            print(f"⚠️  LOW CORRELATION: This protein doesn't strongly match any Platonic solid")
            print(f"   Consider using --target-geometry none for unguided exploration")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Make sure PDB file exists in:")
        print(f"   - pdb_cache/pdb{pdb_id.lower()}.ent")
        print(f"   - Current directory as {pdb_id}.pdb")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error analyzing protein: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
