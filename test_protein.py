#!/usr/bin/env python3
"""
Universal Protein Test - CLI for Protein Structure Prediction

Uses PredictionRunner as the SINGLE SOURCE OF TRUTH for predictions.
This ensures CLI tests use the exact same code path as the web interface.

Key Features:
  - Uses PredictionRunner (same as website backend)
  - Quantum Refinement Engine (two-stage optimization)
  - Real RMSD calculations with Kabsch alignment
  - QCPP-UBF integration for quantum-guided exploration
  - Geometric attractor analysis (golden ratio patterns)
  - Mediator agents for pattern detection and information relay

Usage:
  python test_protein.py --pdb 1UBQ                    # Test Ubiquitin
  python test_protein.py --pdb 1CRN --enable-refinement # Explicit quantum refinement
  python test_protein.py --sequence ACDEFGHIKL          # Test custom sequence
  python test_protein.py --list                         # Show available proteins
  python test_protein.py --quick                        # Quick test on small protein

Performance Notes:
  - THz recording is DISABLED by default in main exploration (saves ~0.75s)
  - Quantum refinement adds ~20-40s for comprehensive validation
  - Real RMSD requires native PDB structure for comparison
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import unified PredictionRunner - THE SINGLE SOURCE OF TRUTH
from ubf_protein.prediction_runner import (
    PredictionRunner,
    PredictionConfig,
    PredictionResults,
    ProgressUpdate,
    get_optimal_settings,
    get_quick_test_settings
)

# Import utilities for PDB handling
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import aa3, aa1

# Create amino acid mapping dictionary
AA1_TO_AA3 = dict(zip(aa1, aa3))


def discover_pdb_files() -> dict:
    """Automatically discover PDB files in workspace folders."""
    discovered = {}
    
    # Search locations
    search_paths = [
        Path("pdb_cache"),
        Path("quantum_coherence_proteins/pdb_files"),
        Path("pdb_files")
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        # Find all PDB files
        for pdb_file in search_path.glob("*.pdb"):
            pdb_id = pdb_file.stem.upper()
            if pdb_id not in discovered:
                discovered[pdb_id] = {
                    "name": pdb_id,
                    "path": str(pdb_file),
                    "description": f"Found in {search_path}"
                }
        
        # Find all .ent files (BioPython format)
        for ent_file in search_path.glob("pdb*.ent"):
            pdb_id = ent_file.stem[3:].upper()  # Remove 'pdb' prefix
            if pdb_id not in discovered:
                discovered[pdb_id] = {
                    "name": pdb_id,
                    "path": str(ent_file),
                    "description": f"Found in {search_path}"
                }
    
    return discovered


# Predefined test proteins with experimental data (for reference)
KNOWN_PROTEINS = {
    "1UBQ": {"name": "Ubiquitin", "residues": 76, "description": "Small regulatory protein"},
    "1CRN": {"name": "Crambin", "residues": 46, "description": "Plant seed protein"},
    "1LYZ": {"name": "Lysozyme", "residues": 129, "description": "Enzyme, breaks bacterial cell walls"},
    "1VII": {"name": "Villin", "residues": 35, "description": "Actin-binding protein headpiece"},
    "2MR9": {"name": "BBL", "residues": 47, "description": "Three-helix bundle protein"},
    "5HLQ": {"name": "Myoglobin", "residues": 153, "description": "Oxygen storage protein"},
    "1MBN": {"name": "Myoglobin", "residues": 153, "description": "Oxygen storage protein"},
    "2LZM": {"name": "Lysozyme T4", "residues": 164, "description": "T4 phage lysozyme variant"},
    "1AKI": {"name": "Ribonuclease A", "residues": 124, "description": "RNA degradation enzyme"},
    "3CLN": {"name": "Calmodulin", "residues": 148, "description": "Calcium-binding protein"},
    "1HEN": {"name": "Hen Egg Lysozyme", "residues": 129, "description": "Classic test protein"},
    "1TIM": {"name": "Triose Phosphate Isomerase", "residues": 247, "description": "Glycolysis enzyme"},
    "1PRN": {"name": "Proteinase A", "residues": 290, "description": "Serine protease"},
    "3SSI": {"name": "SSI Inhibitor", "residues": 113, "description": "Protease inhibitor"},
}

# Combine discovered files with known proteins
AVAILABLE_PROTEINS = discover_pdb_files()


def download_pdb(pdb_id: str) -> Optional[Path]:
    """Download PDB file if not cached."""
    cache_dir = Path("pdb_cache")
    cache_dir.mkdir(exist_ok=True)
    
    pdb_file = cache_dir / f"pdb{pdb_id.lower()}.ent"
    
    if pdb_file.exists():
        print(f"✓ Using cached PDB file: {pdb_file}")
        return pdb_file
    
    print(f"📥 Downloading PDB {pdb_id}...")
    try:
        from Bio.PDB.PDBList import PDBList
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, pdir=str(cache_dir), file_format='pdb')
        print(f"✓ Downloaded to: {pdb_file}")
        return pdb_file
    except Exception as e:
        print(f"❌ Failed to download PDB {pdb_id}: {e}")
        return None


def load_sequence_from_pdb(pdb_file: Path) -> str:
    """Extract amino acid sequence from PDB file."""
    aa_map = dict(zip(aa3, aa1))
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))
    
    if structure is None:
        raise ValueError(f"Failed to parse PDB structure from {pdb_file}")
    
    chain = list(structure.get_chains())[0]
    residues = list(chain.get_residues())
    
    sequence = ""
    for res in residues:
        if res.id[0] == ' ':
            resname = res.resname
            if resname in aa_map:
                sequence += aa_map[resname]
            else:
                sequence += 'X'
    
    return sequence


def load_experimental_data(pdb_id: str) -> Optional[dict]:
    """Load experimental data if available."""
    exp_file = Path("data/experimental_stability.csv")
    if not exp_file.exists():
        return None
    
    try:
        df = pd.read_csv(exp_file)
        protein_data = df[df['PDB_ID'] == pdb_id.upper()]
        if protein_data.empty:
            return None
        
        return {
            'temperature': float(protein_data['Melting_Temperature_C'].values[0]),
            'deltaG': float(protein_data['DeltaG_kcal_mol'].values[0])
        }
    except Exception as e:
        print(f"⚠️  Could not load experimental data: {e}")
        return None


def create_cli_progress_callback():
    """Create a progress callback for CLI output."""
    last_update_time = [time.time()]
    
    def callback(update: ProgressUpdate):
        # Only print every 2 seconds to avoid spam
        current_time = time.time()
        if current_time - last_update_time[0] < 2.0 and update.progress_percentage < 100:
            return
        last_update_time[0] = current_time
        
        # Format progress bar
        bar_width = 30
        filled = int(bar_width * update.progress_percentage / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Build status line
        status = f"\r[{bar}] {update.progress_percentage:5.1f}%"
        
        if update.stage == "exploration":
            status += f" | Energy: {update.best_energy:8.1f}"
            if update.best_rmsd is not None and update.best_rmsd != float('inf'):
                status += f" | RMSD: {update.best_rmsd:5.2f}Å"
            status += f" | Conf: {update.conformations_explored:,}"
        elif update.stage == "refinement":
            status += " | 🔬 Quantum Refinement..."
        elif update.stage == "analysis":
            status += " | 📊 Analyzing..."
        elif update.stage == "complete":
            status += " | ✅ Complete!"
        
        if update.message and update.stage in ["initialization", "complete"]:
            print(f"\n{update.message}")
        else:
            print(status, end='', flush=True)
        
        if update.progress_percentage >= 100:
            print()  # New line at end
    
    return callback


def run_protein_test(sequence: str, pdb_file: Optional[Path] = None, pdb_id: Optional[str] = None, 
                     custom_agents: Optional[int] = None, custom_iterations: Optional[int] = None,
                     target_geometry: str = 'none', enable_mediators: bool = False, 
                     mediator_count: int = 2, enable_refinement: bool = False,
                     enable_hierarchical: bool = False) -> dict:
    """
    Run complete protein test using unified PredictionRunner.
    
    This function uses the SAME code path as the website backend,
    ensuring consistent results between CLI and web interface.
    """
    
    print("\n" + "="*70)
    print("QCPP-UBF PROTEIN STRUCTURE PREDICTION")
    print("Using PredictionRunner (production code path)")
    print("="*70)
    
    # Get optimal settings for display
    settings = get_optimal_settings(len(sequence))
    num_agents = custom_agents or settings['agents']
    iterations = custom_iterations or settings['iterations']
    
    print(f"\n📊 Test Configuration:")
    print(f"  - Sequence Length: {len(sequence)} residues")
    print(f"  - Protein Category: {settings['category']}")
    print(f"  - Agents: {num_agents}")
    print(f"  - Iterations: {iterations} per agent")
    print(f"  - Total Conformations: {num_agents * iterations:,}")
    if pdb_id:
        print(f"  - PDB ID: {pdb_id.upper()}")
    if target_geometry != 'none':
        print(f"  - 🎯 Geometric Target: {target_geometry.capitalize()}")
    if enable_mediators:
        print(f"  - 🔍 Mediator Agents: {mediator_count} agents")
    if enable_refinement:
        print(f"  - ⚛️ Quantum Refinement: ENABLED")
    if enable_hierarchical:
        print(f"  - 🔗 Hierarchical Folding: ENABLED")
    
    # Load experimental data if available
    exp_data = None
    if pdb_id:
        exp_data = load_experimental_data(pdb_id)
        if exp_data:
            print(f"\n🔬 Experimental Data Available:")
            print(f"  - Melting Temperature: {exp_data['temperature']:.1f} °C")
            print(f"  - ΔG Unfolding: {exp_data['deltaG']:.2f} kcal/mol")
    
    # Create PredictionConfig - THE KEY CHANGE
    config = PredictionConfig(
        sequence=sequence,
        native_pdb=pdb_id,
        pdb_file_path=str(pdb_file) if pdb_file else None,
        agents=num_agents,
        iterations=iterations,
        diversity="balanced",
        qcpp_config="default",
        qcpp_frequency=20,
        cache_size=10000,
        enable_refinement=enable_refinement,
        enable_mediators=enable_mediators,
        mediator_count=mediator_count,
        target_geometry=target_geometry,
        enable_hierarchical_folding=enable_hierarchical,
        enable_checkpointing=False,  # Disable for CLI tests
        output_dir=None,  # We'll handle output manually
        save_pdb=False,
        save_trajectory=False
    )
    
    print(f"\n🚀 Starting prediction (PredictionRunner)...")
    start_time = time.time()
    
    # Create runner and execute - SAME AS WEBSITE
    runner = PredictionRunner(config)
    results = runner.run(progress_callback=create_cli_progress_callback())
    
    total_time = time.time() - start_time
    
    # Print comprehensive results
    print_results_summary(results, exp_data, pdb_id)
    
    # Save results to organized directory
    output = save_results(results, pdb_id, exp_data, config)
    
    return output


def print_results_summary(results: PredictionResults, exp_data: Optional[dict], pdb_id: Optional[str]):
    """Print comprehensive results summary."""
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Structural exploration
    print(f"\n🔬 STRUCTURAL EXPLORATION:")
    print(f"  - Best Energy: {results.best_energy:.2f} kcal/mol")
    
    if results.best_rmsd is not None and results.best_rmsd != float('inf'):
        quality = results.validation_quality or "N/A"
        print(f"  - RMSD: {results.best_rmsd:.2f} Å ({quality.upper()})")
    else:
        print(f"  - RMSD: Not available (no native structure)")
    
    if results.gdt_ts_score is not None:
        print(f"  - GDT-TS: {results.gdt_ts_score:.1f}")
    if results.tm_score is not None:
        print(f"  - TM-score: {results.tm_score:.3f}")
    
    print(f"  - Conformations: {results.conformations_explored:,}")
    print(f"  - Time: {results.exploration_time_seconds:.1f}s")
    print(f"  - Throughput: {results.throughput_conf_per_sec:.1f} conf/s")
    
    # QCPP Integration
    print(f"\n📊 QCPP INTEGRATION:")
    print(f"  - Total Analyses: {results.qcpp_total_analyses:,}")
    print(f"  - Cache Hit Rate: {results.qcpp_cache_hit_rate:.1f}%")
    print(f"  - Avg Analysis Time: {results.qcpp_avg_time_ms:.2f}ms")
    
    # Quantum refinement
    if results.refinement_applied:
        print(f"\n⚛️ QUANTUM REFINEMENT:")
        print(f"  - Initial RMSD: {results.refinement_initial_rmsd:.2f} Å")
        print(f"  - Final RMSD: {results.refinement_final_rmsd:.2f} Å")
        improvement = results.refinement_improvement_percent or 0
        print(f"  - RMSD Improvement: {improvement:.1f}%")
        if results.refinement_time_seconds:
            print(f"  - Refinement Time: {results.refinement_time_seconds:.1f}s")
    
    # Geometric analysis
    if results.geometric_analysis:
        geo = results.geometric_analysis
        print(f"\n🔬 GEOMETRIC ATTRACTOR ANALYSIS:")
        
        if 'golden_ratio_percentage' in geo:
            phi_pct = geo['golden_ratio_percentage']
            print(f"  - Golden Ratio (φ) Patterns: {phi_pct:.1f}%")
        
        if 'symmetry_metrics' in geo:
            sym = geo['symmetry_metrics']
            print(f"  - Rotational Symmetry: {sym.get('rotational', 0):.3f}")
        
        if 'platonic_similarities' in geo:
            plat = geo['platonic_similarities']
            print(f"  - Icosahedron Similarity: {plat.get('icosahedron', 0):.3f}")
            print(f"  - Dodecahedron Similarity: {plat.get('dodecahedron', 0):.3f}")
        
        # Interpretation
        phi_pct = geo.get('golden_ratio_percentage', 0)
        icosa = geo.get('platonic_similarities', {}).get('icosahedron', 0)
        dodeca = geo.get('platonic_similarities', {}).get('dodecahedron', 0)
        
        if phi_pct > 15 or icosa > 0.6 or dodeca > 0.6:
            print(f"  ✨ HYPOTHESIS SUPPORT: Strong geometric optimization detected!")
        elif phi_pct > 10:
            print(f"  ⚡ HYPOTHESIS SUPPORT: Moderate geometric patterns present")
    
    # Mediator statistics
    if results.mediator_stats:
        ms = results.mediator_stats
        print(f"\n🔍 MEDIATOR AGENT ANALYSIS:")
        print(f"  - Active Mediators: {ms.get('mediator_count', 0)}")
        print(f"  - Total Patterns Detected: {ms.get('total_detections', 0)}")
    
    # Hierarchical folding statistics
    if results.hierarchical_folding_stats:
        hf = results.hierarchical_folding_stats
        print(f"\n🔗 HIERARCHICAL FOLDING:")
        if 'anchoring' in hf:
            anchor_pct = hf['anchoring'].get('anchoring_percentage', 0)
            print(f"  - Anchoring: {anchor_pct:.1f}%")
    
    print(f"\n" + "="*70)
    
    # Overall assessment
    if results.best_rmsd is not None and results.best_rmsd != float('inf'):
        if results.best_rmsd < 4.0:
            print("✅ TEST SUCCESSFUL!")
            print("   Structure prediction shows promising results")
        elif results.best_rmsd < 8.0:
            print("⚠️  Results show moderate accuracy")
            print("   Consider: longer iterations, quantum refinement")
        else:
            print("⚠️  Results show room for improvement")
            print("   Consider: longer iterations, more agents, quantum refinement")
    else:
        if results.best_energy < 0:
            print("✅ Energy minimization successful (no native structure for RMSD)")
        else:
            print("⚠️  Exploration may need more iterations")
    
    print("="*70)


def save_results(results: PredictionResults, pdb_id: Optional[str], 
                 exp_data: Optional[dict], config: PredictionConfig) -> dict:
    """Save results to JSON file."""
    
    results_dir = Path("results/test_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine RMSD quality
    rmsd_quality = "N/A"
    if results.best_rmsd is not None and results.best_rmsd != float('inf'):
        if results.best_rmsd < 2.0:
            rmsd_quality = "EXCELLENT"
        elif results.best_rmsd < 4.0:
            rmsd_quality = "GOOD"
        elif results.best_rmsd < 6.0:
            rmsd_quality = "FAIR"
        else:
            rmsd_quality = "NEEDS IMPROVEMENT"
    
    # Build output structure (compatible with previous format)
    output = {
        'protein_info': {
            'pdb_id': pdb_id,
            'sequence_length': results.sequence_length,
            'category': get_optimal_settings(results.sequence_length)['category']
        },
        'test_config': {
            'num_agents': config.agents,
            'iterations_per_agent': config.iterations,
            'total_conformations': (config.agents or 0) * (config.iterations or 0),
            'mediators_enabled': config.enable_mediators,
            'mediator_count': config.mediator_count if config.enable_mediators else 0,
            'quantum_refinement_enabled': config.enable_refinement,
            'hierarchical_folding_enabled': config.enable_hierarchical_folding,
            'target_geometry': config.target_geometry,
            'using_prediction_runner': True  # Mark as using unified code path
        },
        'exploration_results': {
            'best_energy': results.best_energy,
            'estimated_rmsd': results.best_rmsd,
            'rmsd_quality': rmsd_quality,
            'exploration_time_s': results.exploration_time_seconds,
            'throughput_conf_per_s': results.throughput_conf_per_sec
        },
        'rmsd_validation': {
            'rmsd': results.best_rmsd,
            'gdt_ts': results.gdt_ts_score,
            'tm_score': results.tm_score,
            'n_atoms': results.sequence_length,
            'aligned': True if results.best_rmsd else False,
            'calculation_method': results.rmsd_calculation_method
        } if results.best_rmsd else None,
        'quantum_refinement': {
            'initial_rmsd': results.refinement_initial_rmsd,
            'final_rmsd': results.refinement_final_rmsd,
            'rmsd_improvement': (results.refinement_initial_rmsd - results.refinement_final_rmsd) 
                               if results.refinement_initial_rmsd and results.refinement_final_rmsd else None,
            'improvement_percent': results.refinement_improvement_percent,
            'refinement_time_seconds': results.refinement_time_seconds
        } if results.refinement_applied else None,
        'qcpp_integration': {
            'total_analyses': results.qcpp_total_analyses,
            'cache_hit_rate': results.qcpp_cache_hit_rate,
            'avg_calculation_time_ms': results.qcpp_avg_time_ms
        },
        'geometric_attractor_analysis': results.geometric_analysis,
        'mediator_statistics': results.mediator_stats,
        'hierarchical_folding_statistics': results.hierarchical_folding_stats,
        'prediction_runner_version': '2.0',  # Mark version
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = results_dir / f"test_{pdb_id or 'custom'}_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Also save PDB structure if we have coordinates
    if results.best_conformation_coords:
        pdb_dir = Path("results/predicted_structures")
        pdb_dir.mkdir(parents=True, exist_ok=True)
        
        pdb_filename = f"{pdb_id or 'custom'}_predicted.pdb"
        if config.target_geometry != 'none':
            pdb_filename = f"{pdb_id or 'custom'}_predicted_{config.target_geometry}.pdb"
        
        pdb_path = pdb_dir / pdb_filename
        save_conformation_as_pdb(
            sequence=results.sequence,
            coordinates=results.best_conformation_coords,
            energy=results.best_energy,
            output_file=pdb_path,
            pdb_id=pdb_id
        )
    
    return output


def save_conformation_as_pdb(sequence: str, coordinates: list, energy: float, 
                              output_file: Path, pdb_id: Optional[str] = None):
    """Save a conformation to PDB format."""
    with open(output_file, 'w') as f:
        f.write("HEADER    PROTEIN STRUCTURE PREDICTION\n")
        if pdb_id:
            f.write(f"TITLE     UBF-QCPP PREDICTION FOR {pdb_id.upper()}\n")
        else:
            f.write(f"TITLE     UBF-QCPP PREDICTION\n")
        f.write(f"REMARK    SEQUENCE: {sequence}\n")
        f.write(f"REMARK    ENERGY: {energy:.2f} kcal/mol\n")
        f.write(f"REMARK    METHOD: UBF with QCPP integration (PredictionRunner)\n")
        f.write(f"REMARK    DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        for i, (aa_letter, coord) in enumerate(zip(sequence, coordinates), 1):
            x, y, z = coord
            aa_3letter = AA1_TO_AA3.get(aa_letter, 'UNK')
            f.write(f"ATOM  {i:5d}  CA  {aa_3letter:3s} A{i:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        
        f.write("END\n")
    
    print(f"✓ Structure saved to: {output_file}")


def list_available_proteins():
    """Display available test proteins."""
    print("\n" + "="*70)
    print("AVAILABLE PDB FILES")
    print("="*70)
    
    if not AVAILABLE_PROTEINS:
        print("\n⚠️  No PDB files found in workspace")
        print("   Files will be auto-downloaded when you test a protein")
    else:
        print(f"\nFound {len(AVAILABLE_PROTEINS)} PDB files in your workspace:\n")
        
        for pdb_id, info in sorted(AVAILABLE_PROTEINS.items()):
            if pdb_id in KNOWN_PROTEINS:
                known = KNOWN_PROTEINS[pdb_id]
                print(f"{pdb_id} - {known['name']}")
                print(f"  Residues: {known['residues']}")
                print(f"  Description: {known['description']}")
            else:
                print(f"{pdb_id}")
                print(f"  Location: {info.get('path', 'cached')}")
            
            print(f"  Command: python test_protein.py --pdb {pdb_id}")
            print()
    
    print("="*70)
    print("You can also test any PDB ID - it will be auto-downloaded!")
    print("Example: python test_protein.py --pdb 1ABC")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test protein structure prediction with QCPP-UBF integration (uses PredictionRunner)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_protein.py --pdb 1UBQ                    # Test Ubiquitin
  python test_protein.py --pdb 1CRN                    # Test Crambin
  python test_protein.py --sequence ACDEFGHIKL          # Custom sequence
  python test_protein.py --list                         # Show available proteins
  python test_protein.py --quick                        # Quick test (small protein)
  python test_protein.py --pdb 1UBQ --agents 30        # Custom agent count
  python test_protein.py --pdb 1UBQ --enable-mediators # Enable pattern detection
  python test_protein.py --pdb 1UBQ --enable-refinement # Enable quantum refinement
  python test_protein.py --pdb 1UBQ --enable-hierarchical # Enable hierarchical folding

NOTE: This CLI uses PredictionRunner, the SAME code path as the website backend.
        """
    )
    
    parser.add_argument('--pdb', type=str, help='PDB ID to test (e.g., 1UBQ)')
    parser.add_argument('--sequence', type=str, help='Custom amino acid sequence')
    parser.add_argument('--agents', type=int, help='Number of agents (optional, auto-configured)')
    parser.add_argument('--iterations', type=int, help='Iterations per agent (optional, auto-configured)')
    parser.add_argument('--target-geometry', 
                        choices=['none', 'octahedron', 'icosahedron', 'dodecahedron', 'tetrahedron', 'cube'],
                        default='none',
                        help='Target Platonic solid geometry for active agent guidance (default: none)')
    parser.add_argument('--enable-mediators', action='store_true',
                        help='Enable Mediator Agents for pattern detection and information relay')
    parser.add_argument('--mediator-count', type=int, default=2,
                        help='Number of Mediator Agents to deploy (default: 2)')
    parser.add_argument('--enable-refinement', action='store_true',
                        help='Enable quantum refinement for two-stage optimization')
    parser.add_argument('--enable-hierarchical', action='store_true',
                        help='Enable hierarchical folding with progressive search confinement')
    parser.add_argument('--list', action='store_true', help='List available test proteins')
    parser.add_argument('--quick', action='store_true', help='Quick test on Villin (35 residues)')
    
    args = parser.parse_args()
    
    # Show available proteins
    if args.list:
        list_available_proteins()
        return
    
    # Quick test
    if args.quick:
        print("🚀 Quick Test Mode: Using Villin (1VII, 35 residues, reduced iterations)")
        args.pdb = '1VII'
        quick_settings = get_quick_test_settings(35)
        if not args.agents:
            args.agents = quick_settings['agents']
        if not args.iterations:
            args.iterations = quick_settings['iterations']
    
    # Validate input
    if not args.pdb and not args.sequence:
        parser.print_help()
        print("\n❌ Error: Provide either --pdb or --sequence")
        print("   Example: python test_protein.py --pdb 1UBQ")
        print("   Or use: python test_protein.py --list")
        sys.exit(1)
    
    # Test with PDB ID
    if args.pdb:
        pdb_id = args.pdb.upper()
        
        print(f"\n🧬 Testing protein: {pdb_id}")
        if pdb_id in KNOWN_PROTEINS:
            info = KNOWN_PROTEINS[pdb_id]
            print(f"   Name: {info['name']}")
            print(f"   Size: {info['residues']} residues")
            print(f"   Description: {info['description']}")
        elif pdb_id in AVAILABLE_PROTEINS:
            print(f"   PDB file found: {AVAILABLE_PROTEINS[pdb_id].get('path', 'cached')}")
        
        # Download PDB
        pdb_file = download_pdb(pdb_id)
        if not pdb_file or not pdb_file.exists():
            print("❌ Failed to get PDB file")
            sys.exit(1)
        
        # Extract sequence
        sequence = load_sequence_from_pdb(pdb_file)
        print(f"✓ Loaded sequence: {len(sequence)} residues")
        
        # Run test
        run_protein_test(
            sequence=sequence,
            pdb_file=pdb_file,
            pdb_id=pdb_id,
            custom_agents=args.agents,
            custom_iterations=args.iterations,
            target_geometry=args.target_geometry,
            enable_mediators=args.enable_mediators,
            mediator_count=args.mediator_count,
            enable_refinement=args.enable_refinement,
            enable_hierarchical=args.enable_hierarchical
        )
    
    # Test with custom sequence
    elif args.sequence:
        sequence = args.sequence.upper()
        print(f"\n🧬 Testing custom sequence: {len(sequence)} residues")
        
        run_protein_test(
            sequence=sequence,
            custom_agents=args.agents,
            custom_iterations=args.iterations,
            target_geometry=args.target_geometry,
            enable_mediators=args.enable_mediators,
            mediator_count=args.mediator_count,
            enable_refinement=args.enable_refinement,
            enable_hierarchical=args.enable_hierarchical
        )


if __name__ == "__main__":
    main()
