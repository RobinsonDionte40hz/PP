#!/usr/bin/env python3
"""
Quick Geometric Hypothesis Validation
======================================

Simple script to test the three critical questions:
1. Do PREDICTED structures show φ patterns? (vs native PDB)
2. What are TRUE RMSD values? (validate claims)
3. Brief comparison ordered vs IDP

Usage:
    python quick_validation_test.py --pdb 1VII
    python quick_validation_test.py --pdb 1CD3 --iterations 300
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

try:
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.Superimposer import Superimposer
    from Bio.PDB.PDBList import PDBList
except ImportError:
    print("ERROR: BioPython required. Install: pip install biopython numpy")
    sys.exit(1)

try:
    from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
except ImportError as e:
    print(f"ERROR: Cannot import UBF system: {e}")
    sys.exit(1)

try:
    from test_geometric_attractors import GoldenRatioAnalyzer, SymmetryAnalyzer, ProteinStructure as RealProteinStructure
    HAVE_REAL_ANALYZERS = True
except ImportError:
    print("WARNING: Using fallback geometric analysis")
    HAVE_REAL_ANALYZERS = False
    
    class GoldenRatioAnalyzer:
        def analyze(self, coords):
            phi = 1.618033988749
            distances = []
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    d = np.linalg.norm(coords[i] - coords[j])
                    distances.append(d)
            
            matches = 0
            for d in distances:
                for n in range(-3, 4):
                    target = 3.8 * (phi ** n)
                    if abs(d - target) / target < 0.05:
                        matches += 1
                        break
            
            return {
                'phi_percentage': (matches / len(distances) * 100) if distances else 0,
                'total_matches': matches,
                'total_distances': len(distances)
            }
    
    class SymmetryAnalyzer:
        def analyze(self, coords):
            if len(coords) < 3:
                return {'rotational_symmetry': 0.0, 'local_symmetry': 0.0}
            
            center = np.mean(coords, axis=0)
            distances = [np.linalg.norm(c - center) for c in coords]
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            rot_sym = max(0.0, min(1.0, 1.0 - (std_dist / avg_dist if avg_dist > 0 else 0.0)))
            
            local_sym = []
            for i in range(len(coords)):
                neighbors = []
                for j in range(len(coords)):
                    if i != j:
                        d = np.linalg.norm(coords[i] - coords[j])
                        if d < 10.0:
                            neighbors.append(d)
                if neighbors and np.mean(neighbors) > 0:
                    local_sym.append(1.0 - np.std(neighbors) / np.mean(neighbors))
            
            return {
                'rotational_symmetry': rot_sym,
                'local_symmetry': np.mean(local_sym) if local_sym else 0.0
            }
    
    class ProteinStructure:
        def __init__(self, coords):
            self.coords = coords


def load_native_pdb(pdb_id: str, pdb_dir: str = "pdb_cache") -> np.ndarray:
    """Load CA coordinates from native PDB"""
    Path(pdb_dir).mkdir(exist_ok=True)
    pdb_path = Path(pdb_dir) / f"{pdb_id.lower()}.pdb"
    
    if not pdb_path.exists():
        print(f"  Downloading {pdb_id}...")
        try:
            pdbl = PDBList()
            pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')
            downloaded = Path(pdb_dir) / f"pdb{pdb_id.lower()}.ent"
            if downloaded.exists():
                downloaded.rename(pdb_path)
        except Exception as e:
            print(f"  ERROR downloading: {e}")
            return None
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(pdb_path))
        
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                        coords.append(residue['CA'].get_coord())
        
        return np.array(coords) if coords else None
            
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return None


def calculate_true_rmsd(predicted: np.ndarray, native: np.ndarray) -> float:
    """Calculate RMSD after optimal superposition"""
    if len(predicted) != len(native):
        min_len = min(len(predicted), len(native))
        predicted = predicted[:min_len]
        native = native[:min_len]
    
    # Simple RMSD without BioPython (fallback)
    # Center both structures
    pred_centered = predicted - np.mean(predicted, axis=0)
    nat_centered = native - np.mean(native, axis=0)
    
    # Calculate rotation matrix using SVD
    H = pred_centered.T @ nat_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Apply rotation
    pred_rotated = pred_centered @ R
    
    # Calculate RMSD
    diff = pred_rotated - nat_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


def analyze_geometric_patterns(coords: np.ndarray, label: str, pdb_name: str = "unknown"):
    """Analyze φ and symmetry patterns"""
    print(f"\n  Analyzing {label}...")
    
    if HAVE_REAL_ANALYZERS:
        # Create temporary PDB file for analysis
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            # Write simple PDB format
            for i, coord in enumerate(coords):
                f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C\n")
            temp_pdb = f.name
        
        try:
            # Use real analyzers
            protein_struct = RealProteinStructure(
                name=pdb_name,
                pdb_file=Path(temp_pdb),
                sequence='A' * len(coords),
                num_residues=len(coords),
                rmsd=0.0,
                energy=0.0
            )
            
            phi_analyzer = GoldenRatioAnalyzer()
            sym_analyzer = SymmetryAnalyzer()
            
            phi_result = phi_analyzer.analyze_structure(protein_struct)
            sym_result = sym_analyzer.analyze_structure(protein_struct)
            
            phi_pct = phi_result.golden_ratio_percentage if hasattr(phi_result, 'golden_ratio_percentage') else 0.0
            rot_sym = sym_result.rotational_symmetry if hasattr(sym_result, 'rotational_symmetry') else 0.0
            local_sym = sym_result.local_symmetry if hasattr(sym_result, 'local_symmetry') else 0.0
            
        finally:
            # Cleanup temp file
            import os
            try:
                os.unlink(temp_pdb)
            except:
                pass
    else:
        # Use fallback analyzers
        phi_analyzer = GoldenRatioAnalyzer()
        sym_analyzer = SymmetryAnalyzer()
        
        phi_result = phi_analyzer.analyze(coords)
        sym_result = sym_analyzer.analyze(coords)
        
        phi_pct = phi_result.get('phi_percentage', 0.0)
        rot_sym = sym_result.get('rotational_symmetry', 0.0)
        local_sym = sym_result.get('local_symmetry', 0.0)
    
    print(f"    φ patterns: {phi_pct:.2f}%")
    print(f"    Rotational symmetry: {rot_sym:.3f}")
    print(f"    Local symmetry: {local_sym:.3f}")
    
    return {
        'phi_percentage': float(phi_pct),
        'rotational_symmetry': float(rot_sym),
        'local_symmetry': float(local_sym)
    }


def run_prediction(pdb_id: str, iterations: int = 500, agents: int = 10):
    """Run UBF prediction"""
    print(f"\n  Running prediction: {agents} agents × {iterations} iterations...")
    
    # Get native structure for sequence length
    native_coords = load_native_pdb(pdb_id)
    if native_coords is None:
        raise ValueError(f"Cannot load {pdb_id}")
    
    n_res = len(native_coords)
    # Use generic sequence (real implementation would extract from PDB)
    sequence = 'A' * n_res
    
    # Run coordinator
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        adaptive_config=None  # Auto-configure
    )
    
    # Initialize agents
    coordinator.initialize_agents(count=agents, diversity_profile="balanced")
    
    start = time.time()
    final_confs = coordinator.run_parallel_exploration(iterations=iterations)
    runtime = time.time() - start
    
    # Get best conformation
    best_tuple = coordinator.get_best_conformation()
    best_conf, best_energy, best_rmsd = best_tuple
    
    # Extract coordinates - Conformation has atom_coordinates field
    if hasattr(best_conf, 'atom_coordinates'):
        predicted_coords = np.array(best_conf.atom_coordinates)
    else:
        raise ValueError("Cannot extract coordinates from conformation")
    
    print(f"  Prediction complete: {runtime:.1f}s, Energy: {best_energy:.1f} kcal/mol")
    print(f"  Extracted {len(predicted_coords)} CA coordinates")
    
    return predicted_coords, best_energy, runtime


def main():
    parser = argparse.ArgumentParser(description='Quick Geometric Hypothesis Validation')
    parser.add_argument('--pdb', required=True, help='PDB ID (e.g., 1VII, 1CD3)')
    parser.add_argument('--iterations', type=int, default=500, help='Iterations per agent')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"QUICK VALIDATION TEST: {args.pdb}")
    print("="*80)
    
    results = {
        'protein_id': args.pdb,
        'iterations': args.iterations,
        'agents': args.agents
    }
    
    try:
        # Step 1: Load and analyze NATIVE structure
        print("\n[1/3] NATIVE PDB STRUCTURE")
        print("-"*80)
        native_coords = load_native_pdb(args.pdb)
        if native_coords is None:
            raise ValueError("Cannot load native structure")
        
        print(f"  Loaded {len(native_coords)} CA atoms")
        native_results = analyze_geometric_patterns(native_coords, "NATIVE PDB", args.pdb)
        results['native'] = native_results
        
        # Step 2: Run prediction and analyze PREDICTED structure
        print("\n[2/3] PREDICTED STRUCTURE")
        print("-"*80)
        predicted_coords, energy, runtime = run_prediction(
            args.pdb, args.iterations, args.agents
        )
        
        predicted_results = analyze_geometric_patterns(predicted_coords, "PREDICTED", args.pdb)
        results['predicted'] = predicted_results
        results['predicted']['energy'] = energy
        results['predicted']['runtime_seconds'] = runtime
        
        # Step 3: Calculate TRUE RMSD
        print("\n[3/3] TRUE RMSD CALCULATION")
        print("-"*80)
        rmsd = calculate_true_rmsd(predicted_coords, native_coords)
        print(f"  TRUE RMSD: {rmsd:.2f} Å")
        results['rmsd_true'] = float(rmsd)
        
        # Quality assessment
        if rmsd < 5.0:
            quality = "GOOD"
        elif rmsd < 8.0:
            quality = "FAIR"
        else:
            quality = "POOR"
        print(f"  Quality: {quality}")
        results['quality'] = quality
        
        # Comparison analysis
        print("\n" + "="*80)
        print("COMPARISON ANALYSIS")
        print("="*80)
        
        phi_diff = predicted_results['phi_percentage'] - native_results['phi_percentage']
        sym_diff = predicted_results['rotational_symmetry'] - native_results['rotational_symmetry']
        
        print(f"\n{'Metric':<25} {'Native':<12} {'Predicted':<12} {'Difference':<12}")
        print("-"*65)
        print(f"{'φ patterns (%)':<25} {native_results['phi_percentage']:>10.2f}  {predicted_results['phi_percentage']:>10.2f}  {phi_diff:>+10.2f}")
        print(f"{'Rotational symmetry':<25} {native_results['rotational_symmetry']:>10.3f}  {predicted_results['rotational_symmetry']:>10.3f}  {sym_diff:>+10.3f}")
        print(f"{'Local symmetry':<25} {native_results['local_symmetry']:>10.3f}  {predicted_results['local_symmetry']:>10.3f}  {predicted_results['local_symmetry'] - native_results['local_symmetry']:>+10.3f}")
        print(f"{'TRUE RMSD (Å)':<25} {0.0:>10.2f}  {rmsd:>10.2f}  {rmsd:>+10.2f}")
        
        print("\n" + "-"*80)
        print("FINDINGS:")
        print("-"*80)
        
        findings = []
        
        # Finding 1: PDB contamination test
        if abs(phi_diff) < 2.0:
            findings.append(f"⚠️  POTENTIAL CONTAMINATION: Δφ = {phi_diff:+.1f}% (too similar to native)")
            findings.append("    → Previous analysis may have used native PDB structures")
        else:
            findings.append(f"✓  CLEAN ANALYSIS: Δφ = {phi_diff:+.1f}% (analyzing predictions correctly)")
        
        # Finding 2: φ pattern presence
        if predicted_results['phi_percentage'] > 10.0:
            findings.append(f"✓  φ PATTERNS PRESENT: {predicted_results['phi_percentage']:.1f}% in prediction")
            if predicted_results['phi_percentage'] > native_results['phi_percentage']:
                findings.append("    → System ENHANCES φ patterns (algorithm bias?)")
            else:
                findings.append("    → System PRESERVES φ patterns")
        else:
            findings.append(f"❌  LOW φ PATTERNS: {predicted_results['phi_percentage']:.1f}% in prediction")
        
        # Finding 3: RMSD quality
        if rmsd < 5.0:
            findings.append(f"✓  EXCELLENT PREDICTION: RMSD = {rmsd:.2f} Å")
        elif rmsd < 8.0:
            findings.append(f"⚠️  MODERATE PREDICTION: RMSD = {rmsd:.2f} Å")
        else:
            findings.append(f"❌  POOR PREDICTION: RMSD = {rmsd:.2f} Å")
        
        # Finding 4: Symmetry
        if predicted_results['rotational_symmetry'] > 0.9:
            findings.append(f"✓  HIGH SYMMETRY: {predicted_results['rotational_symmetry']:.3f} (geometric order)")
        else:
            findings.append(f"⚠️  MODERATE SYMMETRY: {predicted_results['rotational_symmetry']:.3f}")
        
        for finding in findings:
            print(finding)
        
        results['findings'] = findings
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {output_path}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
