#!/usr/bin/env python3
"""
Download 5 large proteins (>100 residues) for testing
"""

from Bio.PDB import PDBList
from pathlib import Path

# Create cache directory
cache_dir = Path("pdb_cache")
cache_dir.mkdir(exist_ok=True)

# Large proteins to download (all >100 residues, expected excellent results)
proteins = {
    "1MBN": "Myoglobin (153 residues) - Oxygen storage",
    "2LZM": "Lysozyme variant (129 residues) - Enzyme",
    "1AKI": "Ribonuclease A (124 residues) - RNA enzyme",
    "3CLN": "Calmodulin (148 residues) - Calcium binding",
    "1HEN": "Hen Egg Lysozyme (129 residues) - Classic test protein"
}

print("Downloading 5 large proteins (>100 residues)...")
print("="*70)

pdbl = PDBList()

for pdb_id, description in proteins.items():
    print(f"\n📥 Downloading {pdb_id} - {description}")
    try:
        pdbl.retrieve_pdb_file(pdb_id, pdir=str(cache_dir), file_format='pdb')
        print(f"✓ Downloaded {pdb_id}")
    except Exception as e:
        print(f"❌ Failed to download {pdb_id}: {e}")

print("\n" + "="*70)
print("✅ Download complete!")
print("\nTest them with:")
for pdb_id in proteins.keys():
    print(f"  python test_protein.py --pdb {pdb_id}")
