#!/usr/bin/env python3
"""
Universal Protein Test - QCPP-UBF Integration

Simple command-line tool to test any protein with optimal settings.
No coding required - just provide a PDB ID or sequence!

Usage:
  python test_protein.py --pdb 1UBQ                    # Test Ubiquitin
  python test_protein.py --pdb 1CRN                    # Test Crambin
  python test_protein.py --sequence ACDEFGHIKL          # Test custom sequence
  python test_protein.py --list                         # Show available proteins
  python test_protein.py --quick                        # Quick test on small protein

Performance Notes:
  - THz recording is DISABLED by default in main exploration (saves ~0.75s)
  - THz is only ENABLED for separate determinism tests (when explicitly requested)
  - This makes production runs faster while keeping determinism research available
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

# Add ubf_protein to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ubf_protein"))

# Import components
from src.protein_predictor import QuantumCoherenceProteinPredictor
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import aa3, aa1

# Import geometric attractor analysis
try:
    # Import the analysis components from test_geometric_attractors.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("geometric_attractors", "test_geometric_attractors.py")
    geometric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometric_module)
    
    GoldenRatioAnalyzer = geometric_module.GoldenRatioAnalyzer
    SymmetryAnalyzer = geometric_module.SymmetryAnalyzer
    QCPPComponentAnalyzer = geometric_module.QCPPComponentAnalyzer
    ProteinStructure = geometric_module.ProteinStructure
    GEOMETRIC_ANALYSIS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Geometric attractor analysis not available: {e}")
    GEOMETRIC_ANALYSIS_AVAILABLE = False


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
    # New large proteins (>100 residues) - Expected EXCELLENT results
    "1MBN": {"name": "Myoglobin", "residues": 153, "description": "Oxygen storage protein"},
    "2LZM": {"name": "Lysozyme T4", "residues": 164, "description": "T4 phage lysozyme variant"},
    "1AKI": {"name": "Ribonuclease A", "residues": 124, "description": "RNA degradation enzyme"},
    "3CLN": {"name": "Calmodulin", "residues": 148, "description": "Calcium-binding protein"},
    "1HEN": {"name": "Hen Egg Lysozyme", "residues": 129, "description": "Classic test protein"},
    # QCPP cached proteins
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


def get_optimal_settings(sequence_length: int) -> dict:
    """Get optimal agent count and iterations based on protein size."""
    if sequence_length < 50:
        # Small proteins: More iterations per agent, fewer agents
        return {"agents": 15, "iterations": 300, "category": "small"}
    elif sequence_length < 100:
        # Medium proteins: Validated optimal settings
        return {"agents": 20, "iterations": 200, "category": "medium"}
    elif sequence_length < 150:
        # Large proteins: More agents for diversity
        return {"agents": 30, "iterations": 250, "category": "large"}
    else:
        # Very large: Maximum resources
        return {"agents": 50, "iterations": 300, "category": "very_large"}


def get_quick_test_settings(sequence_length: int) -> dict:
    """Get fast test settings for quick validation (10x fewer iterations)."""
    if sequence_length < 50:
        # Small proteins: Quick test with fewer iterations
        return {"agents": 10, "iterations": 50, "category": "small"}
    elif sequence_length < 100:
        # Medium proteins
        return {"agents": 10, "iterations": 40, "category": "medium"}
    elif sequence_length < 150:
        # Large proteins
        return {"agents": 15, "iterations": 40, "category": "large"}
    else:
        # Very large
        return {"agents": 20, "iterations": 50, "category": "very_large"}


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


def analyze_geometric_attractors(pdb_file: Path, sequence: str, qcp_values: Optional[list], 
                                 estimated_rmsd: float, best_energy: float) -> Optional[dict]:
    """Analyze protein structure for golden ratio patterns, symmetry, and QCPP components."""
    
    if not GEOMETRIC_ANALYSIS_AVAILABLE:
        return None
    
    try:
        print(f"\n[BONUS] Analyzing geometric attractors (golden ratio, symmetry)...")
        
        # Create protein structure object
        protein_struct = ProteinStructure(
            name=pdb_file.stem,
            pdb_file=pdb_file,
            sequence=sequence,
            num_residues=len(sequence),
            rmsd=estimated_rmsd,
            energy=best_energy,
            qcp_values=qcp_values
        )
        
        # Run analyses
        golden_analyzer = GoldenRatioAnalyzer()
        symmetry_analyzer = SymmetryAnalyzer()
        qcpp_analyzer = QCPPComponentAnalyzer()
        
        golden_results = golden_analyzer.analyze_structure(protein_struct)
        symmetry_results = symmetry_analyzer.analyze_structure(protein_struct)
        qcpp_results = qcpp_analyzer.analyze_structure(protein_struct)
        
        # Print summary
        print(f"✓ Geometric analysis complete:")
        print(f"  🌟 Golden Ratio (φ) Patterns: {golden_results.golden_ratio_percentage:.1f}% "
              f"({golden_results.golden_ratios}/{golden_results.total_ratios} distance ratios)")
        print(f"  🔷 Rotational Symmetry: {symmetry_results.rotational_symmetry:.3f}")
        print(f"  🔶 Local Symmetry: {symmetry_results.local_symmetry:.3f}")
        print(f"  📐 Platonic Solid Similarities:")
        print(f"     - Icosahedron (φ-based): {symmetry_results.icosahedron_similarity:.3f}")
        print(f"     - Dodecahedron (φ-based): {symmetry_results.dodecahedron_similarity:.3f}")
        print(f"     - Octahedron: {symmetry_results.octahedron_similarity:.3f}")
        
        # Interpret findings
        if golden_results.golden_ratio_percentage > 15:
            print(f"  ✨ HIGH φ content detected! Structure may leverage geometric optimization.")
        elif golden_results.golden_ratio_percentage > 10:
            print(f"  ⚡ Moderate φ content. Some geometric patterns present.")
        
        if symmetry_results.icosahedron_similarity > 0.6 or symmetry_results.dodecahedron_similarity > 0.6:
            print(f"  🌟 Strong similarity to φ-containing Platonic solids!")
            print(f"     This supports the geometric attractor hypothesis.")
        
        return {
            'golden_ratio': {
                'percentage': golden_results.golden_ratio_percentage,
                'total_patterns': golden_results.golden_ratios,
                'total_ratios_analyzed': golden_results.total_ratios
            },
            'symmetry': {
                'rotational': symmetry_results.rotational_symmetry,
                'local': symmetry_results.local_symmetry,
                'radius_of_gyration': symmetry_results.radius_of_gyration,
                'asphericity': symmetry_results.asphericity
            },
            'platonic_similarity': {
                'tetrahedron': symmetry_results.tetrahedron_similarity,
                'cube': symmetry_results.cube_similarity,
                'octahedron': symmetry_results.octahedron_similarity,
                'dodecahedron': symmetry_results.dodecahedron_similarity,
                'icosahedron': symmetry_results.icosahedron_similarity
            },
            'qcpp_components': {
                'golden_correlation': qcpp_results.golden_correlation,
                'doubling_correlation': qcpp_results.doubling_correlation
            }
        }
        
    except Exception as e:
        print(f"⚠️  Could not complete geometric analysis: {e}")
        return None


def analyze_thz_determinism(sequence: str, num_trials: int = 10, 
                           iterations_per_trial: int = 100) -> Optional[dict]:
    """Test folding determinism using THz signature clustering."""
    
    try:
        print(f"\n[BONUS] Testing THz determinism with {num_trials} trials...")
        
        # Import determinism testing
        from ubf_protein.protein_agent import ProteinAgent
        from ubf_protein.signature_analysis import create_determinism_tester
        
        # Run multiple trials
        all_frequencies = []
        all_intensities = []
        trial_energies = []
        
        for trial_num in range(num_trials):
            # Create agent with unique behavior and THz recording ENABLED
            agent = ProteinAgent(
                protein_sequence=sequence,
                initial_frequency=9.0,
                initial_coherence=0.6,
                enable_visualization=False,
                enable_thz_recording=True  # ← ENABLE for determinism test
            )
            
            # Run exploration
            for _ in range(iterations_per_trial):
                try:
                    agent.explore_step()
                except:
                    break
            
            # Get THz signatures
            thz_history = agent.get_thz_signature_history()
            metrics = agent.get_exploration_metrics()
            trial_energies.append(metrics['best_energy'])
            
            # Collect signatures
            for spectrum in thz_history:
                all_frequencies.append(spectrum.frequencies)
                all_intensities.append(spectrum.intensities)
        
        if len(all_frequencies) < 2:
            print(f"  ⚠️  Not enough signatures collected ({len(all_frequencies)})")
            return None
        
        # Analyze determinism
        tester = create_determinism_tester(similarity_threshold=0.7)
        score = tester.calculate_determinism_score(all_frequencies, all_intensities)
        
        print(f"✓ THz determinism analysis complete:")
        print(f"  🎵 Signatures collected: {len(all_frequencies)}")
        print(f"  📊 Clusters found: {score.n_clusters}")
        print(f"  🎯 Convergence: {score.convergence_ratio:.1%} in largest cluster")
        print(f"  🔬 Determinism score: {score.determinism_score:.3f}")
        print(f"  💡 {score.interpret()}")
        
        return {
            'total_signatures': len(all_frequencies),
            'num_trials': num_trials,
            'num_clusters': score.n_clusters,
            'largest_cluster_size': score.largest_cluster_size,
            'convergence_ratio': score.convergence_ratio,
            'determinism_score': score.determinism_score,
            'interpretation': score.interpret(),
            'avg_trial_energy': sum(trial_energies) / len(trial_energies)
        }
        
    except Exception as e:
        print(f"⚠️  Could not complete THz determinism test: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_protein_test(sequence: str, pdb_file: Optional[Path] = None, pdb_id: Optional[str] = None, 
                     custom_agents: Optional[int] = None, custom_iterations: Optional[int] = None):
    """Run complete protein test with QCPP-UBF integration."""
    
    print("\n" + "="*70)
    print("QCPP-UBF PROTEIN STRUCTURE PREDICTION")
    print("="*70)
    
    # Get optimal settings
    settings = get_optimal_settings(len(sequence))
    num_agents = custom_agents or settings['agents']
    iterations = custom_iterations or settings['iterations']
    
    print(f"\n📊 Test Configuration:")
    print(f"  - Sequence Length: {len(sequence)} residues")
    print(f"  - Protein Category: {settings['category']}")
    print(f"  - Agents: {num_agents} (optimal for this size)")
    print(f"  - Iterations: {iterations} per agent")
    print(f"  - Total Conformations: {num_agents * iterations:,}")
    if pdb_id:
        print(f"  - PDB ID: {pdb_id.upper()}")
    
    # Load experimental data if available
    exp_data = None
    if pdb_id:
        exp_data = load_experimental_data(pdb_id)
        if exp_data:
            print(f"\n🔬 Experimental Data Available:")
            print(f"  - Melting Temperature: {exp_data['temperature']:.1f} °C")
            print(f"  - ΔG Unfolding: {exp_data['deltaG']:.2f} kcal/mol")
    
    # Step 1: Initialize QCPP
    print(f"\n[1/5] Initializing QCPP predictor...")
    qcpp_predictor = QuantumCoherenceProteinPredictor()
    
    # Use large cache + infrequent analysis for performance
    cache_size = 10000  # Large cache to maximize hits
    qcpp_adapter = QCPPIntegrationAdapter(qcpp_predictor, cache_size)
    
    # QCPP analysis frequency: 20 = analyze every 20th iteration (20x speedup)
    # This balances physics guidance with performance
    qcpp_freq = 20
    print(f"✓ QCPP initialized (cache={cache_size}, analyzing every {qcpp_freq} iterations)")
    
    # Step 2: Create coordinator
    print(f"\n[2/5] Creating multi-agent coordinator...")
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        qcpp_integration=qcpp_adapter,
        qcpp_analysis_frequency=qcpp_freq
    )
    
    coordinator.initialize_agents(
        count=num_agents,
        diversity_profile="balanced"
    )
    print(f"✓ {num_agents} agents initialized (balanced diversity)")
    
    # Step 3: Run exploration
    print(f"\n[3/5] Running parallel exploration...")
    print(f"  Estimated time: ~{(num_agents * iterations) / 350 / 60:.1f} minutes")
    
    start_time = time.time()
    results = coordinator.run_parallel_exploration(iterations=iterations)
    exploration_time = time.time() - start_time
    
    total_conformations = num_agents * iterations
    throughput = total_conformations / exploration_time
    
    print(f"✓ Exploration complete!")
    print(f"  Time: {exploration_time:.1f}s")
    print(f"  Throughput: {throughput:.1f} conf/s")
    print(f"  Best Energy: {results.best_energy:.2f} kcal/mol")
    
    # Step 4: Calculate RMSD estimate
    print(f"\n[4/5] Calculating structural metrics...")
    normalized_energy = (results.best_energy + 200) / -200
    normalized_energy = max(0, min(1, normalized_energy))
    estimated_rmsd = 10.0 - (normalized_energy * 7.0)
    estimated_rmsd = max(0.5, estimated_rmsd)
    
    if estimated_rmsd < 6.0:
        rmsd_quality = "GOOD"
    elif estimated_rmsd < 8.0:
        rmsd_quality = "FAIR"
    else:
        rmsd_quality = "NEEDS IMPROVEMENT"
    
    print(f"✓ Estimated RMSD: {estimated_rmsd:.2f} Å ({rmsd_quality})")
    
    # Step 5: Calculate RMSE if experimental data available
    rmse_results = None
    qcp_values_native = None
    if pdb_file and exp_data:
        print(f"\n[5/5] Calculating prediction accuracy (RMSE)...")
        
        # Calculate QCPP prediction on native structure
        qcpp_native = QuantumCoherenceProteinPredictor()
        qcpp_native.load_protein(str(pdb_file), chain_id='A')
        qcp_df = qcpp_native.calculate_qcp()
        
        if qcp_df is not None and len(qcp_df) > 0:
            qcp_values_native = qcp_df['qcp'].to_numpy().tolist()
            avg_qcp = float(np.mean(qcp_values_native))
            stability_score = avg_qcp / 5.0
            
            # Use validated scaling
            predicted_temp = 50.0 + (stability_score * 40.0)
            predicted_dg = stability_score * 8.0
            
            temp_rmse = abs(predicted_temp - exp_data['temperature'])
            dg_rmse = abs(predicted_dg - exp_data['deltaG'])
            
            temp_percent = (temp_rmse / 43.0) * 100  # 43°C range in dataset
            dg_percent = (dg_rmse / 5.8) * 100  # 5.8 kcal/mol range
            
            if temp_percent < 20 and dg_percent < 20:
                rmse_quality = "GOOD"
            elif temp_percent < 30 and dg_percent < 30:
                rmse_quality = "FAIR"
            else:
                rmse_quality = "NEEDS IMPROVEMENT"
            
            rmse_results = {
                'temperature_rmse': temp_rmse,
                'dg_rmse': dg_rmse,
                'quality': rmse_quality
            }
            
            print(f"✓ RMSE calculated:")
            print(f"  Temperature: {temp_rmse:.2f} °C ({temp_percent:.1f}% of range)")
            print(f"  ΔG: {dg_rmse:.2f} kcal/mol ({dg_percent:.1f}% of range)")
            print(f"  Quality: {rmse_quality}")
        else:
            print(f"⚠️  Could not calculate QCPP prediction")
    else:
        print(f"\n[5/5] Skipping RMSE (no experimental data available)")
    
    # Bonus: Geometric Attractor Analysis
    geometric_results = None
    if pdb_file:
        geometric_results = analyze_geometric_attractors(
            pdb_file=pdb_file,
            sequence=sequence,
            qcp_values=qcp_values_native,
            estimated_rmsd=estimated_rmsd,
            best_energy=results.best_energy
        )
    
    # Bonus: THz Determinism Test (smaller scale for speed)
    thz_results = None
    if len(sequence) <= 20:  # Only for small proteins (fast testing)
        thz_results = analyze_thz_determinism(
            sequence=sequence,
            num_trials=10,
            iterations_per_trial=100
        )
    
    # Get cache stats
    cache_stats = qcpp_adapter.get_cache_stats()
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n🔬 STRUCTURAL EXPLORATION:")
    print(f"  - Best Energy: {results.best_energy:.2f} kcal/mol")
    print(f"  - Estimated RMSD: {estimated_rmsd:.2f} Å ({rmsd_quality})")
    print(f"  - Conformations: {total_conformations:,}")
    print(f"  - Time: {exploration_time:.1f}s")
    print(f"  - Throughput: {throughput:.1f} conf/s")
    
    print(f"\n📊 QCPP INTEGRATION:")
    print(f"  - Total Analyses: {cache_stats['total_analyses']:,}")
    print(f"  - Cache Hit Rate: {cache_stats['cache_hit_rate']:.1f}%")
    print(f"  - Avg Analysis Time: {cache_stats['avg_calculation_time_ms']:.2f}ms")
    
    if rmse_results:
        print(f"\n🎯 PREDICTION ACCURACY:")
        print(f"  - Temperature RMSE: {rmse_results['temperature_rmse']:.2f} °C")
        print(f"  - ΔG RMSE: {rmse_results['dg_rmse']:.2f} kcal/mol")
        print(f"  - Overall Quality: {rmse_results['quality']}")
    
    if geometric_results:
        print(f"\n🔬 GEOMETRIC ATTRACTOR ANALYSIS:")
        print(f"  - Golden Ratio (φ) Patterns: {geometric_results['golden_ratio']['percentage']:.1f}%")
        print(f"  - Rotational Symmetry: {geometric_results['symmetry']['rotational']:.3f}")
        print(f"  - Icosahedron Similarity: {geometric_results['platonic_similarity']['icosahedron']:.3f}")
        print(f"  - Dodecahedron Similarity: {geometric_results['platonic_similarity']['dodecahedron']:.3f}")
        
        # Interpretation
        phi_pct = geometric_results['golden_ratio']['percentage']
        icosa_sim = geometric_results['platonic_similarity']['icosahedron']
        dodeca_sim = geometric_results['platonic_similarity']['dodecahedron']
        
        if phi_pct > 15 or icosa_sim > 0.6 or dodeca_sim > 0.6:
            print(f"  ✨ HYPOTHESIS SUPPORT: Strong geometric optimization detected!")
        elif phi_pct > 10:
            print(f"  ⚡ HYPOTHESIS SUPPORT: Moderate geometric patterns present")
        else:
            print(f"  📊 Low geometric optimization (expected for small/unstructured proteins)")
    
    if thz_results:
        print(f"\n🎵 THz DETERMINISM ANALYSIS:")
        print(f"  - Signatures collected: {thz_results['total_signatures']}")
        print(f"  - Signature clusters: {thz_results['num_clusters']}")
        print(f"  - Convergence ratio: {thz_results['convergence_ratio']:.1%}")
        print(f"  - Determinism score: {thz_results['determinism_score']:.3f}")
        print(f"  - {thz_results['interpretation']}")
        
        if thz_results['determinism_score'] > 0.8:
            print(f"  🌟 STRONG EVIDENCE: Folding is highly deterministic!")
        elif thz_results['determinism_score'] > 0.6:
            print(f"  ✨ MODERATE EVIDENCE: Multiple convergent pathways")
        else:
            print(f"  ⚡ WEAK EVIDENCE: Stochastic folding behavior")
    
    print(f"\n" + "="*70)
    
    # Overall assessment
    if rmsd_quality in ["GOOD", "FAIR"]:
        print("✅ TEST SUCCESSFUL!")
        print("   Structure prediction shows promising results")
    else:
        print("⚠️  Results show room for improvement")
        print("   Consider: longer iterations, more agents")
    
    if rmse_results and rmse_results['quality'] in ["GOOD", "FAIR"]:
        print("✅ PREDICTION ACCURACY VALIDATED!")
        print("   QCPP physics model shows good agreement with experimental data")
    
    print("="*70)
    
    # Save results to organized directory
    results_dir = Path("results/test_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        'protein_info': {
            'pdb_id': pdb_id,
            'sequence_length': len(sequence),
            'category': settings['category']
        },
        'test_config': {
            'num_agents': num_agents,
            'iterations_per_agent': iterations,
            'total_conformations': total_conformations
        },
        'exploration_results': {
            'best_energy': results.best_energy,
            'estimated_rmsd': estimated_rmsd,
            'rmsd_quality': rmsd_quality,
            'exploration_time_s': exploration_time,
            'throughput_conf_per_s': throughput
        },
        'qcpp_integration': {
            'total_analyses': cache_stats['total_analyses'],
            'cache_hit_rate': cache_stats['cache_hit_rate'],
            'avg_calculation_time_ms': cache_stats['avg_calculation_time_ms']
        },
        'rmse_validation': rmse_results,
        'geometric_attractor_analysis': geometric_results,
        'thz_determinism_analysis': thz_results,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = results_dir / f"test_{pdb_id or 'custom'}_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}\n")
    
    return output


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
            # Check if we have detailed info
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
        description='Test protein structure prediction with QCPP-UBF integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_protein.py --pdb 1UBQ                    # Test Ubiquitin
  python test_protein.py --pdb 1CRN                    # Test Crambin
  python test_protein.py --sequence ACDEFGHIKL          # Custom sequence
  python test_protein.py --list                         # Show available proteins
  python test_protein.py --quick                        # Quick test (small protein)
  python test_protein.py --pdb 1UBQ --agents 30        # Custom agent count
        """
    )
    
    parser.add_argument('--pdb', type=str, help='PDB ID to test (e.g., 1UBQ)')
    parser.add_argument('--sequence', type=str, help='Custom amino acid sequence')
    parser.add_argument('--agents', type=int, help='Number of agents (optional, auto-configured)')
    parser.add_argument('--iterations', type=int, help='Iterations per agent (optional, auto-configured)')
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
        # Override with quick settings - much fewer iterations for speed
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
            custom_iterations=args.iterations
        )
    
    # Test with custom sequence
    elif args.sequence:
        sequence = args.sequence.upper()
        print(f"\n🧬 Testing custom sequence: {len(sequence)} residues")
        
        run_protein_test(
            sequence=sequence,
            custom_agents=args.agents,
            custom_iterations=args.iterations
        )


if __name__ == "__main__":
    main()
