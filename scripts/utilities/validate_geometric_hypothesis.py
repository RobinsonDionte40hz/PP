#!/usr/bin/env python3
"""
Geometric Hypothesis Validation Script
======================================

Tests three critical questions:
1. Do PREDICTED structures show φ patterns? (vs native PDB contamination)
2. Do patterns vanish without QCPP? (algorithm bias vs real physics)
3. What are TRUE RMSD values? (validate quality claims)

Usage:
    python validate_geometric_hypothesis.py --proteins 1VII 1CD3 1F0R --mode full
    python validate_geometric_hypothesis.py --proteins 1UBQ --mode predicted_only
    python validate_geometric_hypothesis.py --proteins 1ROP --mode ablation
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, asdict

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

try:
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.Superimposer import Superimposer
    from Bio.PDB.PDBList import PDBList
    import numpy as np
except ImportError:
    print("ERROR: BioPython required. Install: pip install biopython")
    sys.exit(1)

# Import UBF system
try:
    from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
    from ubf_protein.models import Conformation
except ImportError as e:
    print(f"ERROR: Cannot import UBF system: {e}")
    print("Make sure ubf_protein/ is in the path")
    sys.exit(1)

# Import geometric analysis
try:
    from test_geometric_attractors import (
        GoldenRatioAnalyzer,
        SymmetryAnalyzer,
        ProteinStructure
    )
except ImportError:
    print("WARNING: test_geometric_attractors.py not found - creating minimal implementation")
    
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
            rot_sym = 1.0 - (std_dist / avg_dist if avg_dist > 0 else 0.0)
            
            local_sym = []
            for i in range(len(coords)):
                neighbors = []
                for j in range(len(coords)):
                    if i != j:
                        d = np.linalg.norm(coords[i] - coords[j])
                        if d < 10.0:
                            neighbors.append(d)
                if neighbors:
                    local_sym.append(1.0 - np.std(neighbors) / np.mean(neighbors))
            
            return {
                'rotational_symmetry': max(0.0, min(1.0, rot_sym)),
                'local_symmetry': np.mean(local_sym) if local_sym else 0.0
            }
    
    class ProteinStructure:
        def __init__(self, coords):
            self.coords = coords


@dataclass
class ValidationResult:
    """Results from one validation experiment"""
    protein_id: str
    mode: str  # 'native', 'predicted', 'ablation', 'random'
    
    # Geometric metrics
    phi_percentage: float
    rotational_symmetry: float
    local_symmetry: float
    
    # Quality metrics
    rmsd_estimated: float
    rmsd_true: Optional[float]
    energy: float
    
    # Metadata
    residues: int
    iterations: int
    runtime_seconds: float
    qcpp_enabled: bool


class NoPhysicsIntegration(IPhysicsIntegration):
    """Dummy physics integration for ablation tests"""
    
    def calculate_stability(self, conformation: Conformation) -> float:
        return 0.0
    
    def get_40hz_resonance_factor(self, conformation: Conformation) -> float:
        return 1.0
    
    def get_water_shielding_factor(self, conformation: Conformation) -> float:
        return 1.0


def load_native_pdb(pdb_id: str, pdb_dir: str = "pdb_cache") -> Optional[np.ndarray]:
    """Load CA coordinates from native PDB structure"""
    pdb_path = Path(pdb_dir) / f"{pdb_id.lower()}.pdb"
    
    if not pdb_path.exists():
        # Try downloading
        try:
            from Bio.PDB import PDBList
            pdbl = PDBList()
            pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')
            # PDBList saves as pdb{id}.ent
            downloaded = Path(pdb_dir) / f"pdb{pdb_id.lower()}.ent"
            if downloaded.exists():
                downloaded.rename(pdb_path)
        except Exception as e:
            print(f"  WARNING: Cannot download {pdb_id}: {e}")
            return None
    
    if not pdb_path.exists():
        print(f"  WARNING: PDB file not found: {pdb_path}")
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
        
        if coords:
            return np.array(coords)
        else:
            print(f"  WARNING: No CA atoms found in {pdb_id}")
            return None
            
    except Exception as e:
        print(f"  ERROR loading {pdb_id}: {e}")
        return None


def conformation_to_coords(conf: Conformation) -> np.ndarray:
    """Convert UBF Conformation to CA coordinate array"""
    coords = []
    for residue in conf.residues:
        if residue.ca_coord is not None:
            coords.append(residue.ca_coord)
    return np.array(coords)


def calculate_true_rmsd(predicted_coords: np.ndarray, native_coords: np.ndarray) -> float:
    """Calculate true RMSD after optimal superposition"""
    if len(predicted_coords) != len(native_coords):
        print(f"  WARNING: Coordinate mismatch - predicted {len(predicted_coords)}, native {len(native_coords)}")
        # Truncate to shorter length
        min_len = min(len(predicted_coords), len(native_coords))
        predicted_coords = predicted_coords[:min_len]
        native_coords = native_coords[:min_len]
    
    # Create dummy structures for BioPython Superimposer
    # (It needs Atom objects, but we can work around this)
    superimposer = Superimposer()
    
    try:
        # Set coordinates
        superimposer.set_atoms(
            [DummyAtom(c) for c in native_coords],
            [DummyAtom(c) for c in predicted_coords]
        )
        
        return superimposer.rms
        
    except Exception as e:
        print(f"  WARNING: Superimposition failed: {e}")
        # Fallback to simple RMSD
        diff = predicted_coords - native_coords
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


class DummyAtom:
    """Minimal atom object for Superimposer"""
    def __init__(self, coord):
        self.coord = np.array(coord)
    
    def get_coord(self):
        return self.coord


def analyze_native_structure(pdb_id: str) -> ValidationResult:
    """Analyze native PDB structure (baseline)"""
    print(f"\n[1/4] Analyzing NATIVE PDB structure: {pdb_id}")
    start_time = time.time()
    
    coords = load_native_pdb(pdb_id)
    if coords is None:
        raise ValueError(f"Cannot load native structure for {pdb_id}")
    
    print(f"  Loaded {len(coords)} CA atoms")
    
    # Geometric analysis
    phi_analyzer = GoldenRatioAnalyzer()
    sym_analyzer = SymmetryAnalyzer()
    
    protein_struct = ProteinStructure(coords)
    phi_result = phi_analyzer.analyze(protein_struct)
    sym_result = sym_analyzer.analyze(protein_struct)
    
    runtime = time.time() - start_time
    
    result = ValidationResult(
        protein_id=pdb_id,
        mode='native',
        phi_percentage=phi_result.get('phi_percentage', 0.0),
        rotational_symmetry=sym_result.get('rotational_symmetry', 0.0),
        local_symmetry=sym_result.get('local_symmetry', 0.0),
        rmsd_estimated=0.0,  # Native structure
        rmsd_true=0.0,
        energy=0.0,  # Not applicable
        residues=len(coords),
        iterations=0,
        runtime_seconds=runtime,
        qcpp_enabled=False
    )
    
    print(f"  ✓ Native: φ={result.phi_percentage:.2f}%, Symmetry={result.rotational_symmetry:.3f}")
    return result


def analyze_predicted_structure(pdb_id: str, sequence: str, iterations: int = 500,
                                agents: int = 10, enable_qcpp: bool = True) -> ValidationResult:
    """Run prediction and analyze PREDICTED structure"""
    mode = 'predicted' if enable_qcpp else 'ablation'
    print(f"\n[2/4] Analyzing PREDICTED structure: {pdb_id} (QCPP={'ON' if enable_qcpp else 'OFF'})")
    start_time = time.time()
    
    # Configure coordinator
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        num_agents=agents,
        diversity_mode='balanced'
    )
    
    # Ablation: disable QCPP if requested
    if not enable_qcpp:
        print("  🔧 ABLATION MODE: Disabling QCPP physics integration")
        for agent in coordinator.agents:
            agent._physics_integration = NoPhysicsIntegration()
    
    # Run exploration
    print(f"  Running {agents} agents × {iterations} iterations...")
    final_confs = coordinator.run_parallel_exploration(iterations=iterations)
    
    # Get best conformation
    best_conf = coordinator.get_best_conformation()
    if best_conf is None:
        raise ValueError("No valid conformation produced")
    
    print(f"  Best energy: {best_conf.energy:.2f} kcal/mol")
    
    # Extract coordinates
    predicted_coords = conformation_to_coords(best_conf)
    print(f"  Extracted {len(predicted_coords)} CA coordinates")
    
    # Calculate TRUE RMSD
    native_coords = load_native_pdb(pdb_id)
    rmsd_true = None
    if native_coords is not None:
        rmsd_true = calculate_true_rmsd(predicted_coords, native_coords)
        print(f"  TRUE RMSD: {rmsd_true:.2f} Å")
    
    # Geometric analysis ON PREDICTED STRUCTURE
    phi_analyzer = GoldenRatioAnalyzer()
    sym_analyzer = SymmetryAnalyzer()
    
    protein_struct = ProteinStructure(predicted_coords)
    phi_result = phi_analyzer.analyze(protein_struct)
    sym_result = sym_analyzer.analyze(protein_struct)
    
    runtime = time.time() - start_time
    
    result = ValidationResult(
        protein_id=pdb_id,
        mode=mode,
        phi_percentage=phi_result.get('phi_percentage', 0.0),
        rotational_symmetry=sym_result.get('rotational_symmetry', 0.0),
        local_symmetry=sym_result.get('local_symmetry', 0.0),
        rmsd_estimated=3.0,  # Placeholder - system doesn't estimate
        rmsd_true=rmsd_true,
        energy=best_conf.energy,
        residues=len(predicted_coords),
        iterations=iterations,
        runtime_seconds=runtime,
        qcpp_enabled=enable_qcpp
    )
    
    print(f"  ✓ Predicted: φ={result.phi_percentage:.2f}%, Symmetry={result.rotational_symmetry:.3f}, RMSD={rmsd_true:.2f}Å")
    return result


def analyze_random_walk(pdb_id: str, sequence: str, steps: int = 1000) -> ValidationResult:
    """Generate random walk structure (no physics, pure random)"""
    print(f"\n[3/4] Analyzing RANDOM WALK baseline: {pdb_id}")
    start_time = time.time()
    
    # Generate random structure
    # Start from extended chain, apply random perturbations
    n_res = len(sequence)
    
    # Extended chain: 3.8 Å spacing along x-axis
    coords = np.zeros((n_res, 3))
    for i in range(n_res):
        coords[i] = [i * 3.8, 0.0, 0.0]
    
    # Apply random walk
    np.random.seed(42)  # Reproducible
    for step in range(steps):
        idx = np.random.randint(0, n_res)
        coords[idx] += np.random.randn(3) * 0.5  # Small random displacement
    
    print(f"  Generated random structure with {n_res} residues")
    
    # Calculate RMSD vs native
    native_coords = load_native_pdb(pdb_id)
    rmsd_true = None
    if native_coords is not None:
        rmsd_true = calculate_true_rmsd(coords, native_coords)
        print(f"  TRUE RMSD: {rmsd_true:.2f} Å")
    
    # Geometric analysis
    phi_analyzer = GoldenRatioAnalyzer()
    sym_analyzer = SymmetryAnalyzer()
    
    protein_struct = ProteinStructure(coords)
    phi_result = phi_analyzer.analyze(protein_struct)
    sym_result = sym_analyzer.analyze(protein_struct)
    
    runtime = time.time() - start_time
    
    result = ValidationResult(
        protein_id=pdb_id,
        mode='random',
        phi_percentage=phi_result.get('phi_percentage', 0.0),
        rotational_symmetry=sym_result.get('rotational_symmetry', 0.0),
        local_symmetry=sym_result.get('local_symmetry', 0.0),
        rmsd_estimated=999.0,
        rmsd_true=rmsd_true,
        energy=999.0,
        residues=n_res,
        iterations=steps,
        runtime_seconds=runtime,
        qcpp_enabled=False
    )
    
    print(f"  ✓ Random: φ={result.phi_percentage:.2f}%, Symmetry={result.rotational_symmetry:.3f}")
    return result


def get_protein_sequence(pdb_id: str) -> str:
    """Extract sequence from PDB file"""
    coords = load_native_pdb(pdb_id)
    if coords is None:
        raise ValueError(f"Cannot load {pdb_id} to extract sequence")
    
    # For validation purposes, use a generic sequence
    # (Real implementation would parse SEQRES records)
    # Using alanine as placeholder
    n_res = len(coords)
    return 'A' * n_res


def compare_results(results: List[ValidationResult]) -> Dict:
    """Generate comparison analysis"""
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    
    comparison = {
        'protein_id': results[0].protein_id,
        'tests': []
    }
    
    # Table header
    print(f"\n{'Mode':<12} {'φ %':<8} {'Symmetry':<10} {'RMSD':<8} {'Energy':<10} {'QCPP':<6}")
    print("-" * 60)
    
    for r in results:
        rmsd_str = f"{r.rmsd_true:.2f}Å" if r.rmsd_true else "N/A"
        energy_str = f"{r.energy:.1f}" if r.energy != 0.0 and r.energy != 999.0 else "N/A"
        qcpp_str = "ON" if r.qcpp_enabled else "OFF"
        
        print(f"{r.mode:<12} {r.phi_percentage:>6.2f}  {r.rotational_symmetry:>8.3f}  "
              f"{rmsd_str:>6}  {energy_str:>8}  {qcpp_str:<6}")
        
        comparison['tests'].append(asdict(r))
    
    # Analysis
    print("\n" + "-"*80)
    print("KEY FINDINGS:")
    print("-"*80)
    
    native = next((r for r in results if r.mode == 'native'), None)
    predicted = next((r for r in results if r.mode == 'predicted'), None)
    ablation = next((r for r in results if r.mode == 'ablation'), None)
    random = next((r for r in results if r.mode == 'random'), None)
    
    findings = []
    
    # Finding 1: Native vs Predicted φ difference
    if native and predicted:
        phi_diff = predicted.phi_percentage - native.phi_percentage
        if abs(phi_diff) < 2.0:
            findings.append(f"⚠️  CONTAMINATION: Predicted φ ({predicted.phi_percentage:.1f}%) ≈ Native φ ({native.phi_percentage:.1f}%)")
            findings.append(f"    → Δφ = {phi_diff:+.1f}% (TOO SIMILAR - likely PDB contamination)")
        else:
            findings.append(f"✓  CLEAN: Predicted φ ({predicted.phi_percentage:.1f}%) ≠ Native φ ({native.phi_percentage:.1f}%)")
            findings.append(f"    → Δφ = {phi_diff:+.1f}% (analyzing predictions correctly)")
    
    # Finding 2: Ablation test
    if predicted and ablation:
        phi_drop = predicted.phi_percentage - ablation.phi_percentage
        if phi_drop > 3.0:
            findings.append(f"✓  PHYSICS MATTERS: φ drops {phi_drop:.1f}% without QCPP")
            findings.append(f"    → QCPP integration discovers/enforces geometric patterns")
        else:
            findings.append(f"⚠️  ALGORITHM BIAS: φ only drops {phi_drop:.1f}% without QCPP")
            findings.append(f"    → Patterns come from energy function, not QCPP")
    
    # Finding 3: Random baseline
    if predicted and random:
        phi_enrichment = predicted.phi_percentage / random.phi_percentage if random.phi_percentage > 0 else 999
        if phi_enrichment > 1.5:
            findings.append(f"✓  ABOVE RANDOM: Predicted φ is {phi_enrichment:.1f}x random baseline")
            findings.append(f"    → System creates geometric order (not just PDB artifact)")
        else:
            findings.append(f"⚠️  NEAR RANDOM: Predicted φ only {phi_enrichment:.1f}x random baseline")
            findings.append(f"    → Geometric patterns may be statistical artifact")
    
    # Finding 4: RMSD validation
    if predicted and predicted.rmsd_true:
        if predicted.rmsd_true < 5.0:
            findings.append(f"✓  GOOD PREDICTION: True RMSD = {predicted.rmsd_true:.2f} Å")
        elif predicted.rmsd_true < 8.0:
            findings.append(f"⚠️  FAIR PREDICTION: True RMSD = {predicted.rmsd_true:.2f} Å")
        else:
            findings.append(f"❌  POOR PREDICTION: True RMSD = {predicted.rmsd_true:.2f} Å")
    
    for finding in findings:
        print(finding)
    
    comparison['findings'] = findings
    
    print("\n" + "="*80)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Validate Geometric Attractor Hypothesis')
    parser.add_argument('--proteins', nargs='+', required=True,
                       help='PDB IDs to test (e.g., 1VII 1CD3 1F0R)')
    parser.add_argument('--mode', default='full',
                       choices=['full', 'predicted_only', 'ablation', 'quick'],
                       help='Test mode: full (all 4 tests), predicted_only, ablation (no QCPP), quick (native+predicted)')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Iterations per agent (default: 500)')
    parser.add_argument('--agents', type=int, default=10,
                       help='Number of agents (default: 10)')
    parser.add_argument('--output', default='validation_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GEOMETRIC HYPOTHESIS VALIDATION")
    print("="*80)
    print(f"Proteins: {', '.join(args.proteins)}")
    print(f"Mode: {args.mode}")
    print(f"Agents: {args.agents}, Iterations: {args.iterations}")
    print("="*80)
    
    all_results = {}
    
    for pdb_id in args.proteins:
        print(f"\n\n{'#'*80}")
        print(f"# TESTING: {pdb_id}")
        print(f"{'#'*80}")
        
        try:
            # Get sequence
            sequence = get_protein_sequence(pdb_id)
            print(f"Sequence length: {len(sequence)} residues")
            
            results = []
            
            # Test 1: Native PDB (baseline)
            if args.mode in ['full', 'quick']:
                results.append(analyze_native_structure(pdb_id))
            
            # Test 2: Predicted structure (with QCPP)
            if args.mode in ['full', 'predicted_only', 'quick']:
                results.append(analyze_predicted_structure(
                    pdb_id, sequence, args.iterations, args.agents, enable_qcpp=True
                ))
            
            # Test 3: Ablation (without QCPP)
            if args.mode in ['full', 'ablation']:
                results.append(analyze_predicted_structure(
                    pdb_id, sequence, args.iterations, args.agents, enable_qcpp=False
                ))
            
            # Test 4: Random walk baseline
            if args.mode == 'full':
                results.append(analyze_random_walk(pdb_id, sequence))
            
            # Compare
            comparison = compare_results(results)
            all_results[pdb_id] = comparison
            
        except Exception as e:
            print(f"\n❌ ERROR testing {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Summary
    print("\nSUMMARY ACROSS ALL PROTEINS:")
    print("-"*80)
    
    for pdb_id, comparison in all_results.items():
        print(f"\n{pdb_id}:")
        for finding in comparison.get('findings', []):
            print(f"  {finding}")


if __name__ == '__main__':
    main()
