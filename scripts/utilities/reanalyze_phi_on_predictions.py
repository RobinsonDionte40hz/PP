"""
Re-analyze φ patterns on PREDICTED structures vs NATIVE structures.

This script addresses the critical methodological flaw identified in validation tests:
Previous analysis calculated φ patterns from native PDB structures, not predictions.
This contaminated results with crystallographic order regardless of prediction quality.

CRITICAL TESTS:
1. Native vs Predicted φ: Compare φ in native PDB vs predicted conformation
2. Correlation Analysis: Does predicted φ correlate with RMSD quality?
3. Delta Check: Small Δφ despite poor RMSD indicates algorithm bias
4. IDP Behavior: Do IDPs show different predicted φ than ordered proteins?

Expected Outcomes:
- If algorithm bias: All predictions show ~14% φ regardless of RMSD
- If real attractors: Good predictions (RMSD<5Å) show high φ, poor show low φ
- If contamination only: Native φ matches quality, predicted φ is random
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
from dataclasses import dataclass

from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue

# Import geometric analyzers
try:
    import sys
    sys.path.insert(0, '.')
    from test_geometric_attractors import (
        GoldenRatioAnalyzer,
        SymmetryAnalyzer,
        ProteinStructure as RealProteinStructure
    )
    HAVE_REAL_ANALYZERS = True
    print("[OK] Successfully imported geometric analyzers")
except ImportError as e:
    HAVE_REAL_ANALYZERS = False
    print(f"WARNING: Could not import real analyzers: {e}")
    print("Will use fallback dummy values")


@dataclass
class PhiAnalysisResult:
    """Results from φ pattern analysis"""
    pdb_id: str
    protein_type: str  # "ordered" or "disordered"
    residues: int
    
    # Prediction quality
    predicted_rmsd: float
    predicted_energy: float
    
    # Native structure φ patterns
    native_phi_percent: float
    native_rot_symmetry: float
    native_local_symmetry: float
    
    # Predicted structure φ patterns  
    predicted_phi_percent: float
    predicted_rot_symmetry: float
    predicted_local_symmetry: float
    
    # Delta metrics (KEY for bias detection)
    delta_phi: float  # predicted - native
    delta_rot_symmetry: float
    delta_local_symmetry: float
    
    # Interpretation flags
    potential_contamination: bool  # |Δφ| < 2% despite poor RMSD
    algorithm_enhancement: bool  # predicted φ > native φ
    quality_correlation: bool  # Good RMSD → high φ, Poor RMSD → low φ


def create_pdb_from_coordinates(coords: List[Tuple[float, float, float]], 
                                sequence: str,
                                output_file: str) -> str:
    """
    Create a PDB file from CA coordinates.
    
    Args:
        coords: List of (x, y, z) CA atom coordinates
        sequence: Amino acid sequence
        output_file: Path to output PDB file
        
    Returns:
        Path to created PDB file
    """
    structure = Structure.Structure("predicted")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    structure.add(model)
    model.add(chain)
    
    # Create residues with CA atoms
    for i, ((x, y, z), aa) in enumerate(zip(coords, sequence)):
        # Create residue
        residue = Residue.Residue((' ', i+1, ' '), aa, ' ')
        
        # Create CA atom
        from Bio.PDB.Atom import Atom
        import numpy as np
        atom = Atom(
            name='CA',
            coord=np.array([x, y, z], dtype=float),
            bfactor=0.0,
            occupancy=1.0,
            altloc=' ',
            fullname=' CA ',
            serial_number=i+1,
            element='C'
        )
        
        residue.add(atom)
        chain.add(residue)
    
    # Write to PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)
    
    return output_file


def analyze_geometric_patterns(pdb_file: str, pdb_id: str, sequence: str, 
                               num_residues: int) -> Dict[str, float]:
    """
    Analyze geometric patterns (φ, symmetry) from a PDB file.
    
    Args:
        pdb_file: Path to PDB file
        pdb_id: PDB identifier
        sequence: Amino acid sequence
        num_residues: Number of residues
        
    Returns:
        Dictionary with geometric metrics
    """
    if not HAVE_REAL_ANALYZERS:
        # Fallback: return dummy values
        return {
            'phi_percent': 0.0,
            'rotational_symmetry': 0.0,
            'local_symmetry': 0.0
        }
    
    try:
        # Parse PDB file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        
        # Extract CA coordinates
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_coords.append(residue['CA'].get_coord())
        
        if len(ca_coords) == 0:
            print(f"WARNING: No CA atoms found in {pdb_file}")
            return {
                'phi_percent': 0.0,
                'rotational_symmetry': 0.0,
                'local_symmetry': 0.0
            }
        
        # Create ProteinStructure object for analyzers
        from pathlib import Path
        protein_struct = RealProteinStructure(
            name=pdb_id,
            pdb_file=Path(pdb_file),
            sequence=sequence,
            num_residues=num_residues,
            rmsd=0.0,  # Not used by geometric analyzers
            energy=0.0  # Not used by geometric analyzers
        )
        
        # Run φ pattern analysis
        phi_analyzer = GoldenRatioAnalyzer()
        phi_results = phi_analyzer.analyze_structure(protein_struct)
        
        # Run symmetry analysis
        sym_analyzer = SymmetryAnalyzer()
        sym_results = sym_analyzer.analyze_structure(protein_struct)
        
        return {
            'phi_percent': phi_results.golden_ratio_percentage,
            'rotational_symmetry': sym_results.rotational_symmetry,
            'local_symmetry': sym_results.local_symmetry
        }
        
    except Exception as e:
        print(f"ERROR analyzing {pdb_file}: {e}")
        return {
            'phi_percent': 0.0,
            'rotational_symmetry': 0.0,
            'local_symmetry': 0.0
        }


def load_result_file(result_path: str) -> Optional[Dict]:
    """Load and parse a test result JSON file."""
    try:
        with open(result_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR loading {result_path}: {e}")
        return None


def extract_predicted_coordinates(result_data: Dict) -> Optional[List[Tuple[float, float, float]]]:
    """
    Extract predicted CA coordinates from result JSON.
    
    The result format stores the best conformation's coordinates.
    """
    try:
        # Check multiple possible locations for coordinates
        if 'best_conformation' in result_data:
            best_conf = result_data['best_conformation']
            if 'atom_coordinates' in best_conf:
                coords = best_conf['atom_coordinates']
                # Convert to tuples if needed
                return [tuple(c) if isinstance(c, list) else c for c in coords]
        
        # Alternative: check agents
        if 'agents' in result_data and len(result_data['agents']) > 0:
            # Find agent with best energy
            best_agent = min(result_data['agents'], 
                           key=lambda a: a.get('final_energy', float('inf')))
            if 'current_conformation' in best_agent:
                conf = best_agent['current_conformation']
                if 'atom_coordinates' in conf:
                    coords = conf['atom_coordinates']
                    return [tuple(c) if isinstance(c, list) else c for c in coords]
        
        return None
        
    except Exception as e:
        print(f"ERROR extracting coordinates: {e}")
        return None


def get_native_pdb_path(pdb_id: str) -> Optional[str]:
    """Get path to native PDB file."""
    # Check common locations
    possible_paths = [
        f"quantum_coherence_proteins/pdb_files/pdb{pdb_id.lower()}.ent",
        f"pdb_cache/{pdb_id.upper()}.pdb",
        f"data/pdbs/{pdb_id.upper()}.pdb"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def classify_protein_type(pdb_id: str) -> str:
    """Classify protein as ordered or disordered based on research dataset."""
    # Disordered proteins from challenge suite
    disordered = ['1LMB', '1BPI', '2CI2', '2KJ3', '1BTA', '1RIS', '1MVF', '1CD3', '1F0R']
    
    return "disordered" if pdb_id.upper() in disordered else "ordered"


def analyze_single_protein(pdb_id: str, result_file: str) -> Optional[PhiAnalysisResult]:
    """
    Analyze φ patterns for a single protein (native vs predicted).
    
    Args:
        pdb_id: PDB identifier (e.g., "1VII")
        result_file: Path to test result JSON
        
    Returns:
        PhiAnalysisResult or None if analysis fails
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {pdb_id}...")
    print(f"{'='*80}")
    
    # Load result file
    result_data = load_result_file(result_file)
    if result_data is None:
        return None
    
    # Extract metadata
    sequence = result_data.get('protein_sequence', '')
    num_residues = len(sequence)
    protein_type = classify_protein_type(pdb_id)
    
    # Extract predicted metrics
    predicted_rmsd = result_data.get('best_rmsd', 0.0)
    predicted_energy = result_data.get('best_energy', 0.0)
    
    print(f"  Type: {protein_type}")
    print(f"  Residues: {num_residues}")
    print(f"  Predicted RMSD: {predicted_rmsd:.2f} Å")
    print(f"  Predicted Energy: {predicted_energy:.2f} kcal/mol")
    
    # Get native PDB path
    native_pdb = get_native_pdb_path(pdb_id)
    if native_pdb is None:
        print(f"  WARNING: Native PDB not found, skipping")
        return None
    
    print(f"  Native PDB: {native_pdb}")
    
    # Analyze native structure
    print(f"  Analyzing NATIVE structure...")
    native_metrics = analyze_geometric_patterns(native_pdb, pdb_id, sequence, num_residues)
    print(f"    φ patterns: {native_metrics['phi_percent']:.2f}%")
    print(f"    Rotational symmetry: {native_metrics['rotational_symmetry']:.3f}")
    print(f"    Local symmetry: {native_metrics['local_symmetry']:.3f}")
    
    # Extract predicted coordinates
    predicted_coords = extract_predicted_coordinates(result_data)
    if predicted_coords is None:
        print(f"  WARNING: Could not extract predicted coordinates, skipping")
        return None
    
    print(f"  Extracted {len(predicted_coords)} predicted CA coordinates")
    
    # Create temporary PDB for predicted structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        tmp_pdb_path = tmp.name
    
    try:
        create_pdb_from_coordinates(predicted_coords, sequence, tmp_pdb_path)
        
        # Analyze predicted structure
        print(f"  Analyzing PREDICTED structure...")
        predicted_metrics = analyze_geometric_patterns(tmp_pdb_path, f"{pdb_id}_pred", 
                                                      sequence, num_residues)
        print(f"    φ patterns: {predicted_metrics['phi_percent']:.2f}%")
        print(f"    Rotational symmetry: {predicted_metrics['rotational_symmetry']:.3f}")
        print(f"    Local symmetry: {predicted_metrics['local_symmetry']:.3f}")
        
        # Calculate deltas
        delta_phi = predicted_metrics['phi_percent'] - native_metrics['phi_percent']
        delta_rot = predicted_metrics['rotational_symmetry'] - native_metrics['rotational_symmetry']
        delta_local = predicted_metrics['local_symmetry'] - native_metrics['local_symmetry']
        
        print(f"\n  DELTAS (Predicted - Native):")
        print(f"    Δφ: {delta_phi:+.2f}%")
        print(f"    Δ Rotational Symmetry: {delta_rot:+.3f}")
        print(f"    Δ Local Symmetry: {delta_local:+.3f}")
        
        # Interpretation
        potential_contamination = (abs(delta_phi) < 2.0 and predicted_rmsd > 5.0)
        algorithm_enhancement = (predicted_metrics['phi_percent'] > native_metrics['phi_percent'])
        
        # Quality correlation: Good RMSD (< 5Å) should have high φ (>12%)
        quality_correlation = (
            (predicted_rmsd < 5.0 and predicted_metrics['phi_percent'] > 12.0) or
            (predicted_rmsd >= 5.0 and predicted_metrics['phi_percent'] < 12.0)
        )
        
        print(f"\n  INTERPRETATION:")
        if potential_contamination:
            print(f"    [!] POTENTIAL CONTAMINATION: |Delta-phi| < 2% despite RMSD {predicted_rmsd:.2f}A")
        if algorithm_enhancement:
            print(f"    [+] ALGORITHM ENHANCEMENT: Predicted phi > Native phi")
        if quality_correlation:
            print(f"    [+] QUALITY CORRELATION: phi matches prediction quality")
        else:
            print(f"    [-] NO QUALITY CORRELATION: phi doesn't match prediction quality")
        
        # Create result object
        result = PhiAnalysisResult(
            pdb_id=pdb_id,
            protein_type=protein_type,
            residues=num_residues,
            predicted_rmsd=predicted_rmsd,
            predicted_energy=predicted_energy,
            native_phi_percent=native_metrics['phi_percent'],
            native_rot_symmetry=native_metrics['rotational_symmetry'],
            native_local_symmetry=native_metrics['local_symmetry'],
            predicted_phi_percent=predicted_metrics['phi_percent'],
            predicted_rot_symmetry=predicted_metrics['rotational_symmetry'],
            predicted_local_symmetry=predicted_metrics['local_symmetry'],
            delta_phi=delta_phi,
            delta_rot_symmetry=delta_rot,
            delta_local_symmetry=delta_local,
            potential_contamination=potential_contamination,
            algorithm_enhancement=algorithm_enhancement,
            quality_correlation=quality_correlation
        )
        
        return result
        
    finally:
        # Clean up temporary PDB
        try:
            os.unlink(tmp_pdb_path)
        except:
            pass


def analyze_all_proteins() -> List[PhiAnalysisResult]:
    """Analyze all 20 proteins from the research dataset."""
    
    # 20 proteins from research report
    proteins = [
        # Ordered proteins
        '1VII', '1CRN', '1GB1', '1ROP', '1PGB', '1UTG', '1HIV', '3SSI', '1CHO', '1MBN',
        # Disordered/challenge suite
        '1LMB', '1BPI', '2CI2', '1UBQ', '2KJ3', '1BTA', '1RIS', '1MVF', '1CD3', '1F0R'
    ]
    
    results = []
    results_dir = Path("results/test_results")
    
    for pdb_id in proteins:
        result_file = results_dir / f"test_{pdb_id}_results.json"
        
        if not result_file.exists():
            print(f"\nWARNING: Result file not found for {pdb_id}: {result_file}")
            continue
        
        result = analyze_single_protein(pdb_id, str(result_file))
        if result:
            results.append(result)
    
    return results


def generate_summary_report(results: List[PhiAnalysisResult]) -> str:
    """Generate comprehensive summary report of φ re-analysis."""
    
    report = []
    report.append("\n" + "="*100)
    report.append("φ PATTERN RE-ANALYSIS: NATIVE vs PREDICTED STRUCTURES")
    report.append("="*100)
    report.append("")
    report.append(f"Total proteins analyzed: {len(results)}")
    report.append("")
    
    # Separate by protein type
    ordered = [r for r in results if r.protein_type == "ordered"]
    disordered = [r for r in results if r.protein_type == "disordered"]
    
    report.append(f"Ordered proteins: {len(ordered)}")
    report.append(f"Disordered proteins: {len(disordered)}")
    report.append("")
    
    # Summary table
    report.append("="*100)
    report.append("SUMMARY TABLE")
    report.append("="*100)
    report.append(f"{'PDB':<6} {'Type':<12} {'Res':<5} {'RMSD':<8} {'Native φ':<10} {'Pred φ':<10} {'Δφ':<8} {'Interp':<30}")
    report.append("-"*100)
    
    for r in sorted(results, key=lambda x: x.predicted_rmsd):
        interp = []
        if r.potential_contamination:
            interp.append("CONTAM")
        if r.algorithm_enhancement:
            interp.append("ENHANCED")
        if r.quality_correlation:
            interp.append("CORREL")
        
        report.append(
            f"{r.pdb_id:<6} {r.protein_type:<12} {r.residues:<5} {r.predicted_rmsd:<8.2f} "
            f"{r.native_phi_percent:<10.2f} {r.predicted_phi_percent:<10.2f} "
            f"{r.delta_phi:<+8.2f} {','.join(interp):<30}"
        )
    
    report.append("")
    
    # Statistical analysis
    report.append("="*100)
    report.append("STATISTICAL ANALYSIS")
    report.append("="*100)
    report.append("")
    
    # Calculate means
    mean_native_phi = sum(r.native_phi_percent for r in results) / len(results)
    mean_pred_phi = sum(r.predicted_phi_percent for r in results) / len(results)
    mean_delta_phi = sum(r.delta_phi for r in results) / len(results)
    
    report.append(f"Mean Native φ: {mean_native_phi:.2f}%")
    report.append(f"Mean Predicted φ: {mean_pred_phi:.2f}%")
    report.append(f"Mean Δφ: {mean_delta_phi:+.2f}%")
    report.append("")
    
    # Separate by quality
    good_preds = [r for r in results if r.predicted_rmsd < 5.0]
    poor_preds = [r for r in results if r.predicted_rmsd >= 5.0]
    
    if good_preds:
        mean_good_phi = sum(r.predicted_phi_percent for r in good_preds) / len(good_preds)
        report.append(f"Good predictions (RMSD<5Å, n={len(good_preds)}): Mean predicted φ = {mean_good_phi:.2f}%")
    
    if poor_preds:
        mean_poor_phi = sum(r.predicted_phi_percent for r in poor_preds) / len(poor_preds)
        report.append(f"Poor predictions (RMSD≥5Å, n={len(poor_preds)}): Mean predicted φ = {mean_poor_phi:.2f}%")
    
    report.append("")
    
    # Ordered vs Disordered
    if ordered:
        mean_ordered_pred_phi = sum(r.predicted_phi_percent for r in ordered) / len(ordered)
        report.append(f"Ordered proteins (n={len(ordered)}): Mean predicted φ = {mean_ordered_pred_phi:.2f}%")
    
    if disordered:
        mean_disordered_pred_phi = sum(r.predicted_phi_percent for r in disordered) / len(disordered)
        report.append(f"Disordered proteins (n={len(disordered)}): Mean predicted φ = {mean_disordered_pred_phi:.2f}%")
    
    report.append("")
    
    # Key findings
    report.append("="*100)
    report.append("KEY FINDINGS")
    report.append("="*100)
    report.append("")
    
    contamination_count = sum(1 for r in results if r.potential_contamination)
    enhancement_count = sum(1 for r in results if r.algorithm_enhancement)
    correlation_count = sum(1 for r in results if r.quality_correlation)
    
    report.append(f"1. Potential contamination detected: {contamination_count}/{len(results)} proteins ({contamination_count/len(results)*100:.1f}%)")
    report.append(f"   -> |Delta-phi| < 2% despite RMSD > 5A suggests previous analysis used native structures")
    report.append("")
    
    report.append(f"2. Algorithm enhancement: {enhancement_count}/{len(results)} proteins ({enhancement_count/len(results)*100:.1f}%)")
    report.append(f"   -> Predicted phi > Native phi indicates system creates geometric patterns")
    report.append("")
    
    report.append(f"3. Quality correlation: {correlation_count}/{len(results)} proteins ({correlation_count/len(results)*100:.1f}%)")
    report.append(f"   -> phi matches prediction quality suggests real geometric attractors")
    report.append("")
    
    # Interpretation
    report.append("="*100)
    report.append("INTERPRETATION")
    report.append("="*100)
    report.append("")
    
    if contamination_count > len(results) * 0.5:
        report.append("[!] HIGH CONTAMINATION RATE:")
        report.append("   Previous analysis likely calculated phi from NATIVE structures, not predictions.")
        report.append("   This invalidates geometric attractor hypothesis from original research.")
        report.append("")
    
    if enhancement_count > len(results) * 0.7:
        report.append("[+] ALGORITHM BIAS CONFIRMED:")
        report.append("   System systematically enhances phi patterns in predictions.")
        report.append("   This could reflect: a) energy function design, b) physics constraints,")
        report.append("   or c) multi-agent convergence to symmetric minima.")
        report.append("")
    
    if correlation_count > len(results) * 0.6:
        report.append("[+] GEOMETRIC ATTRACTORS PARTIALLY SUPPORTED:")
        report.append("   Predicted phi correlates with quality, suggesting real physical patterns.")
        report.append("   However, algorithm enhancement indicates computational artifacts.")
        report.append("")
    else:
        report.append("[-] NO QUALITY CORRELATION:")
        report.append("   Predicted phi doesn't match prediction quality.")
        report.append("   This strongly suggests algorithm bias, not physical attractors.")
        report.append("")
    
    # Ordered vs Disordered comparison
    if ordered and disordered:
        diff = abs(mean_ordered_pred_phi - mean_disordered_pred_phi)
        report.append(f"Ordered vs Disordered Delta-phi: {diff:.2f}%")
        if diff < 2.0:
            report.append("   -> IDENTICAL patterns suggest universal algorithm bias")
        else:
            report.append("   -> DIFFERENT patterns support physical attractors hypothesis")
        report.append("")
    
    return "\n".join(report)


def main():
    """Main execution function."""
    
    print("\n" + "="*100)
    print("φ PATTERN RE-ANALYSIS ON PREDICTED STRUCTURES")
    print("="*100)
    print("\nThis analysis addresses the critical methodological flaw:")
    print("Previous research calculated φ patterns from NATIVE PDB structures.")
    print("We now re-analyze φ from PREDICTED conformations to detect:")
    print("  1. Algorithm bias (all predictions show ~14% φ)")
    print("  2. Real attractors (φ correlates with RMSD quality)")
    print("  3. Contamination (small Δφ despite poor RMSD)")
    print("")
    
    if not HAVE_REAL_ANALYZERS:
        print("ERROR: Could not import geometric analyzers from test_geometric_attractors.py")
        print("Please ensure test_geometric_attractors.py is in the same directory.")
        return 1
    
    # Run analysis on all 20 proteins
    results = analyze_all_proteins()
    
    if len(results) == 0:
        print("\nERROR: No proteins were successfully analyzed")
        return 1
    
    # Generate summary report
    report = generate_summary_report(results)
    print(report)
    
    # Save detailed results to JSON
    output_file = "phi_reanalysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(
            {
                'summary': {
                    'total_proteins': len(results),
                    'mean_native_phi': sum(r.native_phi_percent for r in results) / len(results),
                    'mean_predicted_phi': sum(r.predicted_phi_percent for r in results) / len(results),
                    'mean_delta_phi': sum(r.delta_phi for r in results) / len(results),
                    'contamination_rate': sum(1 for r in results if r.potential_contamination) / len(results),
                    'enhancement_rate': sum(1 for r in results if r.algorithm_enhancement) / len(results),
                    'correlation_rate': sum(1 for r in results if r.quality_correlation) / len(results)
                },
                'results': [
                    {
                        'pdb_id': r.pdb_id,
                        'protein_type': r.protein_type,
                        'residues': r.residues,
                        'predicted_rmsd': r.predicted_rmsd,
                        'predicted_energy': r.predicted_energy,
                        'native_phi_percent': r.native_phi_percent,
                        'native_rot_symmetry': r.native_rot_symmetry,
                        'native_local_symmetry': r.native_local_symmetry,
                        'predicted_phi_percent': r.predicted_phi_percent,
                        'predicted_rot_symmetry': r.predicted_rot_symmetry,
                        'predicted_local_symmetry': r.predicted_local_symmetry,
                        'delta_phi': r.delta_phi,
                        'delta_rot_symmetry': r.delta_rot_symmetry,
                        'delta_local_symmetry': r.delta_local_symmetry,
                        'potential_contamination': r.potential_contamination,
                        'algorithm_enhancement': r.algorithm_enhancement,
                        'quality_correlation': r.quality_correlation
                    }
                    for r in results
                ]
            },
            f,
            indent=2
        )
    
    print(f"\n[OK] Detailed results saved to: {output_file}")
    print(f"[OK] Analysis complete for {len(results)}/20 proteins")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
