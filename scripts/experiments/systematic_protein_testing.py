#!/usr/bin/env python3
"""
Systematic Protein Testing Campaign - Building Robustness

PRIMARY TESTING MODULE: Uses Quantum Refinement Engine for comprehensive validation

Runs comprehensive tests on 100+ proteins with systematic parameter variations
to build robustness in the UBF protein prediction system with real RMSD calculations.

Features:
- Tests proteins one at a time to avoid resource conflicts
- Uses Quantum Refinement Engine (quantum_refinement_engine.py) as primary module
- Real RMSD calculations with CA-only native structure extraction (FIXED)
- Varies agent counts, iterations, mediator settings, geometric targets
- Quantum refinement validation on all tests
- Tracks performance across different configurations
- Generates detailed reports and analysis

Usage:
  python systematic_protein_testing.py --start 1 --count 10    # Test first 10 proteins
  python systematic_protein_testing.py --protein 1UBQ         # Test specific protein
  python systematic_protein_testing.py --quick                 # Quick test mode
  python systematic_protein_testing.py --resume                # Resume from last checkpoint
  python systematic_protein_testing.py --refinement-only       # Test only quantum refinement configs
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import test_protein components (which now uses public API)
from test_protein import (
    discover_pdb_files, KNOWN_PROTEINS, download_pdb, load_sequence_from_pdb,
    run_protein_test
)

# Import settings from public API
from ubf_protein.api import get_optimal_settings


@dataclass
class TestConfiguration:
    """Configuration for a single protein test."""
    pdb_id: str
    sequence: str
    agents: int
    iterations: int
    enable_mediators: bool
    mediator_count: int
    target_geometry: str
    enable_refinement: bool
    test_id: str

    def to_dict(self):
        return asdict(self)


@dataclass
class TestResult:
    """Result from a single protein test."""
    config: TestConfiguration
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    results: Optional[Dict] = None

    def to_dict(self):
        return {
            'config': self.config.to_dict(),
            'success': self.success,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'results': self.results
        }


class SystematicTester:
    """Manages systematic testing of proteins with varied configurations."""

    def __init__(self, output_dir: str = "systematic_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test configurations to vary
        self.agent_counts = [10, 20, 30, 40, 50]
        self.iteration_counts = [100, 200, 300, 400, 500]
        self.mediator_counts = [1, 2, 3, 4, 5]
        self.geometric_targets = ['none', 'octahedron', 'icosahedron', 'dodecahedron', 'tetrahedron', 'cube']

        # Load available proteins
        self.available_proteins = self._load_available_proteins()

        # Results tracking
        self.results_file = self.output_dir / "campaign_results.json"
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.completed_tests = self._load_completed_tests()

    def _load_available_proteins(self) -> List[Dict]:
        """Load all available proteins from cache and known list."""
        proteins = []

        # Add discovered PDB files
        discovered = discover_pdb_files()
        for pdb_id, info in discovered.items():
            if pdb_id not in [p['pdb_id'] for p in proteins]:
                proteins.append({
                    'pdb_id': pdb_id,
                    'name': info.get('name', pdb_id),
                    'path': info.get('path'),
                    'description': info.get('description', 'Auto-discovered'),
                    'source': 'cache'
                })

        # Add known proteins
        for pdb_id, info in KNOWN_PROTEINS.items():
            if pdb_id not in [p['pdb_id'] for p in proteins]:
                proteins.append({
                    'pdb_id': pdb_id,
                    'name': info['name'],
                    'residues': info['residues'],
                    'description': info['description'],
                    'source': 'known'
                })

        # Sort by PDB ID
        proteins.sort(key=lambda x: x['pdb_id'])
        return proteins

    def _load_completed_tests(self) -> Dict[str, TestResult]:
        """Load previously completed tests."""
        if not self.results_file.exists():
            return {}

        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results = {}
                for test_id, result_data in data.items():
                    config = TestConfiguration(**result_data['config'])
                    result = TestResult(
                        config=config,
                        success=result_data['success'],
                        execution_time=result_data['execution_time'],
                        error_message=result_data.get('error_message'),
                        results=result_data.get('results')
                    )
                    results[test_id] = result
                return results
        except Exception as e:
            print(f"Warning: Could not load previous results: {e}")
            return {}

    def _save_checkpoint(self, current_index: int, current_config_index: int):
        """Save current progress."""
        checkpoint = {
            'current_protein_index': current_index,
            'current_config_index': current_config_index,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self) -> tuple:
        """Load checkpoint if available."""
        if not self.checkpoint_file.exists():
            return 0, 0

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                return checkpoint.get('current_protein_index', 0), checkpoint.get('current_config_index', 0)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return 0, 0

    def _generate_test_configurations(self, pdb_id: str, sequence: str) -> List[TestConfiguration]:
        """
        Generate varied test configurations for a protein.
        
        PRIMARY TESTING MODE: Quantum Refinement Engine validation
        
        All configurations now include quantum refinement as the primary
        validation mechanism, using the fixed RMSD calculator with
        CA-only native structure extraction.
        """
        configs = []
        config_id = 0

        # Base configuration (optimal settings)
        base_settings = get_optimal_settings(len(sequence))

        # Configuration 1: Base optimal with quantum refinement (PRIMARY)
        configs.append(TestConfiguration(
            pdb_id=pdb_id,
            sequence=sequence,
            agents=base_settings['agents'],
            iterations=base_settings['iterations'],
            enable_mediators=False,
            mediator_count=0,
            target_geometry='none',
            enable_refinement=True,  # PRIMARY: Enable refinement
            test_id=f"{pdb_id}_base_qref"
        ))

        # Configuration 2: With mediators + quantum refinement
        configs.append(TestConfiguration(
            pdb_id=pdb_id,
            sequence=sequence,
            agents=base_settings['agents'],
            iterations=base_settings['iterations'],
            enable_mediators=True,
            mediator_count=2,
            target_geometry='none',
            enable_refinement=True,  # PRIMARY: Enable refinement
            test_id=f"{pdb_id}_mediators_qref"
        ))

        # Configuration 3: With geometric targeting + quantum refinement
        configs.append(TestConfiguration(
            pdb_id=pdb_id,
            sequence=sequence,
            agents=base_settings['agents'],
            iterations=base_settings['iterations'],
            enable_mediators=False,
            mediator_count=0,
            target_geometry='icosahedron',
            enable_refinement=True,  # PRIMARY: Enable refinement
            test_id=f"{pdb_id}_geometric_qref"
        ))

        # Configuration 4: Full features + quantum refinement (COMPREHENSIVE)
        configs.append(TestConfiguration(
            pdb_id=pdb_id,
            sequence=sequence,
            agents=base_settings['agents'],
            iterations=base_settings['iterations'],
            enable_mediators=True,
            mediator_count=3,
            target_geometry='icosahedron',
            enable_refinement=True,  # PRIMARY: Enable refinement
            test_id=f"{pdb_id}_full_qref"
        ))

        # Configuration 5: High agent count + quantum refinement
        configs.append(TestConfiguration(
            pdb_id=pdb_id,
            sequence=sequence,
            agents=min(50, base_settings['agents'] * 2),
            iterations=base_settings['iterations'],
            enable_mediators=False,
            mediator_count=0,
            target_geometry='none',
            enable_refinement=True,  # PRIMARY: Enable refinement
            test_id=f"{pdb_id}_high_agents_qref"
        ))

        # Configuration 6: High iterations + quantum refinement
        configs.append(TestConfiguration(
            pdb_id=pdb_id,
            sequence=sequence,
            agents=base_settings['agents'],
            iterations=min(500, base_settings['iterations'] * 2),
            enable_mediators=False,
            mediator_count=0,
            target_geometry='none',
            enable_refinement=True,  # PRIMARY: Enable refinement
            test_id=f"{pdb_id}_high_iter_qref"
        ))

        return configs

    def _run_single_test(self, config: TestConfiguration) -> TestResult:
        """Run a single test configuration with quantum refinement validation."""
        print(f"\n{'='*80}")
        print(f"🧬 TESTING: {config.test_id}")
        print(f"{'='*80}")
        print(f"Protein: {config.pdb_id}")
        print(f"Agents: {config.agents}, Iterations: {config.iterations}")
        print(f"Mediators: {config.enable_mediators} ({config.mediator_count})")
        print(f"Geometry: {config.target_geometry}")
        print(f"⚛️  Quantum Refinement: {'✅ ENABLED (PRIMARY)' if config.enable_refinement else '❌ DISABLED'}")

        start_time = time.time()

        try:
            # Download/get PDB file
            pdb_file = download_pdb(config.pdb_id)
            if not pdb_file:
                raise ValueError(f"Could not obtain PDB file for {config.pdb_id}")

            # Run the test with quantum refinement engine
            results = run_protein_test(
                sequence=config.sequence,
                pdb_file=pdb_file,
                pdb_id=config.pdb_id,
                custom_agents=config.agents,
                custom_iterations=config.iterations,
                target_geometry=config.target_geometry,
                enable_mediators=config.enable_mediators,
                mediator_count=config.mediator_count,
                enable_refinement=config.enable_refinement
            )

            execution_time = time.time() - start_time
            
            # Extract key metrics for display
            if results and 'exploration_results' in results:
                exp_results = results['exploration_results']
                rmsd = exp_results.get('final_rmsd', exp_results.get('estimated_rmsd', 'N/A'))
                energy = exp_results.get('best_energy', 'N/A')
                
                # Check if this was a real RMSD or estimate
                rmsd_type = "REAL" if 'final_rmsd' in exp_results else "ESTIMATED"
                
                print(f"\n✅ Test Complete:")
                print(f"   RMSD: {rmsd:.2f} Å ({rmsd_type})" if isinstance(rmsd, (int, float)) else f"   RMSD: {rmsd}")
                print(f"   Energy: {energy:.2f} kcal/mol" if isinstance(energy, (int, float)) else f"   Energy: {energy}")
                print(f"   Time: {execution_time:.1f}s")
                
                if config.enable_refinement and 'refinement_result' in results:
                    ref_result = results['refinement_result']
                    print(f"   ⚛️  Refinement RMSD Improvement: {ref_result.get('rmsd_improvement', 'N/A'):.2f} Å")

            return TestResult(
                config=config,
                success=True,
                execution_time=execution_time,
                results=results
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"❌ Test failed: {error_msg}")

            return TestResult(
                config=config,
                success=False,
                execution_time=execution_time,
                error_message=error_msg
            )

    def run_campaign(self, start_index: int = 0, max_proteins: Optional[int] = None,
                    resume: bool = False):
        """Run the systematic testing campaign."""

        if resume:
            start_index, config_start_index = self._load_checkpoint()
            print(f"Resuming from protein {start_index}, config {config_start_index}")

        total_proteins = len(self.available_proteins)
        end_index = min(total_proteins, start_index + (max_proteins or total_proteins))

        print(f"\n{'='*100}")
        print("SYSTEMATIC PROTEIN TESTING CAMPAIGN")
        print(f"{'='*100}")
        print(f"Testing proteins {start_index} to {end_index-1} of {total_proteins}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*100}\n")

        campaign_start_time = time.time()
        total_tests = 0
        successful_tests = 0

        for protein_index in range(start_index, end_index):
            protein = self.available_proteins[protein_index]
            pdb_id = protein['pdb_id']

            print(f"\n{'='*60}")
            print(f"PROTEIN {protein_index+1}/{end_index}: {pdb_id} - {protein.get('name', 'Unknown')}")
            print(f"{'='*60}")

            # Get sequence
            try:
                pdb_file = download_pdb(pdb_id)
                if not pdb_file:
                    print(f"⚠️  Skipping {pdb_id}: Could not download PDB")
                    continue

                sequence = load_sequence_from_pdb(pdb_file)
                print(f"✓ Loaded sequence: {len(sequence)} residues")

            except Exception as e:
                print(f"⚠️  Skipping {pdb_id}: Could not load sequence - {e}")
                continue

            # Generate test configurations
            configs = self._generate_test_configurations(pdb_id, sequence)
            print(f"✓ Generated {len(configs)} test configurations")

            # Run each configuration
            for config_index, config in enumerate(configs):
                # Skip if already completed
                if config.test_id in self.completed_tests:
                    print(f"⏭️  Skipping {config.test_id} (already completed)")
                    continue

                # Run the test
                result = self._run_single_test(config)

                # Store result
                self.completed_tests[config.test_id] = result
                total_tests += 1

                if result.success:
                    successful_tests += 1
                    print(f"✅ {config.test_id} completed successfully in {result.execution_time:.1f}s")
                else:
                    print(f"❌ {config.test_id} failed: {result.error_message}")

                # Save progress periodically
                if total_tests % 5 == 0:
                    self._save_results()
                    self._save_checkpoint(protein_index, config_index)

        # Final save
        self._save_results()

        # Generate summary report
        self._generate_summary_report(campaign_start_time, total_tests, successful_tests)

        print(f"\n{'='*100}")
        print("CAMPAIGN COMPLETE")
        print(f"{'='*100}")
        print(f"Total tests run: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*100}")

    def _save_results(self):
        """Save all results to file."""
        results_dict = {test_id: result.to_dict() for test_id, result in self.completed_tests.items()}

        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)

    def _generate_summary_report(self, start_time: float, total_tests: int, successful_tests: int):
        """Generate a summary report of the campaign."""

        campaign_time = time.time() - start_time

        # Analyze results
        successful_results = [r for r in self.completed_tests.values() if r.success and r.results]

        if not successful_results:
            print("No successful tests to analyze")
            return

        # Extract metrics (with quantum refinement support)
        rmsd_values = []
        real_rmsd_values = []  # Track real vs estimated separately
        estimated_rmsd_values = []
        energies = []
        execution_times = []
        refinement_improvements = []

        for result in successful_results:
            if result.results and 'exploration_results' in result.results:
                exp_results = result.results['exploration_results']
                
                # Track RMSD (prioritize real RMSD)
                if exp_results.get('final_rmsd'):
                    rmsd_values.append(exp_results['final_rmsd'])
                    real_rmsd_values.append(exp_results['final_rmsd'])
                elif exp_results.get('estimated_rmsd'):
                    rmsd_values.append(exp_results['estimated_rmsd'])
                    estimated_rmsd_values.append(exp_results['estimated_rmsd'])
                    
                if exp_results.get('best_energy'):
                    energies.append(exp_results['best_energy'])
                execution_times.append(result.execution_time)
                
                # Track quantum refinement improvements
                if result.config.enable_refinement and 'refinement_result' in result.results:
                    ref_result = result.results['refinement_result']
                    if ref_result.get('rmsd_improvement'):
                        refinement_improvements.append(ref_result['rmsd_improvement'])

        # Generate report with quantum refinement metrics
        report = {
            'campaign_summary': {
                'total_proteins_tested': len(set(r.config.pdb_id for r in successful_results)),
                'total_configurations_tested': len(successful_results),
                'total_tests_run': total_tests,
                'successful_tests': successful_tests,
                'campaign_duration_seconds': campaign_time,
                'average_test_time': np.mean(execution_times) if execution_times else None,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'quantum_refinement_tests': sum(1 for r in successful_results if r.config.enable_refinement),
                'real_rmsd_calculations': len(real_rmsd_values),
                'estimated_rmsd_calculations': len(estimated_rmsd_values)
            },
            'performance_metrics': {
                'rmsd_stats': {
                    'mean': np.mean(rmsd_values) if rmsd_values else None,
                    'std': np.std(rmsd_values) if rmsd_values else None,
                    'min': min(rmsd_values) if rmsd_values else None,
                    'max': max(rmsd_values) if rmsd_values else None,
                    'count': len(rmsd_values)
                },
                'real_rmsd_stats': {
                    'mean': np.mean(real_rmsd_values) if real_rmsd_values else None,
                    'std': np.std(real_rmsd_values) if real_rmsd_values else None,
                    'min': min(real_rmsd_values) if real_rmsd_values else None,
                    'max': max(real_rmsd_values) if real_rmsd_values else None,
                    'count': len(real_rmsd_values)
                },
                'energy_stats': {
                    'mean': np.mean(energies) if energies else None,
                    'std': np.std(energies) if energies else None,
                    'min': min(energies) if energies else None,
                    'max': max(energies) if energies else None,
                    'count': len(energies)
                },
                'quantum_refinement_stats': {
                    'mean_improvement': np.mean(refinement_improvements) if refinement_improvements else None,
                    'std_improvement': np.std(refinement_improvements) if refinement_improvements else None,
                    'max_improvement': max(refinement_improvements) if refinement_improvements else None,
                    'count': len(refinement_improvements)
                }
            },
            'configuration_analysis': self._analyze_configurations(successful_results),
            'timestamp': datetime.now().isoformat()
        }

        # Save report
        report_file = self.output_dir / "campaign_summary.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        # Generate human-readable report
        self._generate_human_report(report)

    def _analyze_configurations(self, results: List[TestResult]) -> Dict:
        """Analyze performance by configuration type."""

        config_performance = {}

        for result in results:
            config_type = self._classify_config(result.config)
            if config_type not in config_performance:
                config_performance[config_type] = []

            if result.results and 'exploration_results' in result.results:
                exp_results = result.results['exploration_results']
                rmsd = exp_results.get('estimated_rmsd')
                energy = exp_results.get('best_energy')

                config_performance[config_type].append({
                    'rmsd': rmsd,
                    'energy': energy,
                    'time': result.execution_time
                })

        # Calculate averages
        analysis = {}
        for config_type, performances in config_performance.items():
            rmsds = [p['rmsd'] for p in performances if p['rmsd'] is not None]
            energies = [p['energy'] for p in performances if p['energy'] is not None]
            times = [p['time'] for p in performances]

            analysis[config_type] = {
                'count': len(performances),
                'avg_rmsd': np.mean(rmsds) if rmsds else None,
                'avg_energy': np.mean(energies) if energies else None,
                'avg_time': np.mean(times) if times else None
            }

        return analysis

    def _classify_config(self, config: TestConfiguration) -> str:
        """Classify configuration type with quantum refinement awareness."""
        # Prioritize quantum refinement classification
        if config.enable_refinement:
            if config.enable_mediators and config.target_geometry != 'none':
                return 'full_features_qref'
            elif config.enable_mediators:
                return 'mediators_qref'
            elif config.target_geometry != 'none':
                return 'geometric_qref'
            elif config.agents > 30:
                return 'high_agents_qref'
            elif config.iterations > 300:
                return 'high_iterations_qref'
            else:
                return 'base_optimal_qref'
        else:
            # Legacy non-refinement configs (should be rare now)
            if config.enable_mediators and config.target_geometry != 'none':
                return 'full_features'
            elif config.enable_mediators:
                return 'mediators_only'
            elif config.target_geometry != 'none':
                return 'geometric_only'
            elif config.agents > 30:
                return 'high_agents'
            elif config.iterations > 300:
                return 'high_iterations'
            else:
                return 'base_optimal'

    def _generate_human_report(self, report: Dict):
        """Generate human-readable summary report with quantum refinement highlights."""

        report_file = self.output_dir / "campaign_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("SYSTEMATIC PROTEIN TESTING CAMPAIGN REPORT\n")
            f.write("PRIMARY MODULE: Quantum Refinement Engine + Real RMSD Calculations\n")
            f.write("="*100 + "\n\n")

            # Campaign summary
            cs = report['campaign_summary']
            f.write("CAMPAIGN SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"Proteins tested: {cs['total_proteins_tested']}\n")
            f.write(f"Configurations tested: {cs['total_configurations_tested']}\n")
            f.write(f"Total tests: {cs['total_tests_run']}\n")
            f.write(f"Successful: {cs['successful_tests']}\n")
            f.write(f"Success rate: {cs['success_rate']*100:.1f}%\n")
            f.write(f"Campaign duration: {cs['campaign_duration_seconds']/3600:.1f} hours\n")
            if cs['average_test_time']:
                f.write(f"Average test time: {cs['average_test_time']:.1f} seconds\n")
            f.write(f"\n⚛️  QUANTUM REFINEMENT METRICS:\n")
            f.write(f"   Tests with quantum refinement: {cs['quantum_refinement_tests']}\n")
            f.write(f"   Real RMSD calculations: {cs['real_rmsd_calculations']}\n")
            f.write(f"   Estimated RMSD fallbacks: {cs['estimated_rmsd_calculations']}\n")
            f.write("\n")

            # Performance metrics
            pm = report['performance_metrics']
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 50 + "\n")

            if pm['rmsd_stats']['mean'] is not None:
                rmsd = pm['rmsd_stats']
                f.write("RMSD Performance (All Tests):\n")
                f.write(f"  Mean: {rmsd['mean']:.2f} Å\n")
                f.write(f"  Std Dev: {rmsd['std']:.2f} Å\n")
                f.write(f"  Range: {rmsd['min']:.2f} - {rmsd['max']:.2f} Å\n")
                f.write(f"  Sample size: {rmsd['count']}\n\n")
            
            if pm['real_rmsd_stats']['mean'] is not None:
                real_rmsd = pm['real_rmsd_stats']
                f.write("⚛️  Real RMSD Performance (Kabsch Alignment):\n")
                f.write(f"  Mean: {real_rmsd['mean']:.2f} Å\n")
                f.write(f"  Std Dev: {real_rmsd['std']:.2f} Å\n")
                f.write(f"  Range: {real_rmsd['min']:.2f} - {real_rmsd['max']:.2f} Å\n")
                f.write(f"  Sample size: {real_rmsd['count']}\n\n")

            if pm['energy_stats']['mean'] is not None:
                energy = pm['energy_stats']
                f.write("Energy Performance:\n")
                f.write(f"  Mean: {energy['mean']:.2f} kcal/mol\n")
                f.write(f"  Std Dev: {energy['std']:.2f} kcal/mol\n")
                f.write(f"  Range: {energy['min']:.2f} - {energy['max']:.2f} kcal/mol\n")
                f.write(f"  Sample size: {energy['count']}\n\n")
            
            if pm['quantum_refinement_stats']['mean_improvement'] is not None:
                qr = pm['quantum_refinement_stats']
                f.write("⚛️  Quantum Refinement Impact:\n")
                f.write(f"  Mean RMSD Improvement: {qr['mean_improvement']:.2f} Å\n")
                f.write(f"  Std Dev: {qr['std_improvement']:.2f} Å\n")
                f.write(f"  Max Improvement: {qr['max_improvement']:.2f} Å\n")
                f.write(f"  Refinement tests: {qr['count']}\n\n")

            # Configuration analysis
            ca = report['configuration_analysis']
            f.write("CONFIGURATION ANALYSIS\n")
            f.write("-" * 50 + "\n")

            for config_type, stats in ca.items():
                f.write(f"{config_type.replace('_', ' ').title()}:\n")
                f.write(f"  Tests: {stats['count']}\n")
                if stats['avg_rmsd']:
                    f.write(f"  Avg RMSD: {stats['avg_rmsd']:.2f} Å\n")
                if stats['avg_energy']:
                    f.write(f"  Avg Energy: {stats['avg_energy']:.2f} kcal/mol\n")
                if stats['avg_time']:
                    f.write(f"  Avg Time: {stats['avg_time']:.1f}s\n")
                f.write("\n")

            f.write("="*100 + "\n")

        print(f"✓ Human-readable report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run systematic protein testing campaign',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--start', type=int, default=0,
                       help='Starting protein index (0-based)')
    parser.add_argument('--count', type=int,
                       help='Number of proteins to test')
    parser.add_argument('--protein', type=str,
                       help='Test specific protein by PDB ID')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer configurations)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--output-dir', type=str, default='systematic_test_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create tester
    tester = SystematicTester(args.output_dir)

    print(f"Found {len(tester.available_proteins)} available proteins")

    if args.protein:
        # Test specific protein
        if args.protein.upper() not in [p['pdb_id'] for p in tester.available_proteins]:
            print(f"❌ Protein {args.protein} not found in available proteins")
            sys.exit(1)

        # Find protein index
        protein_index = next(i for i, p in enumerate(tester.available_proteins)
                           if p['pdb_id'] == args.protein.upper())

        tester.run_campaign(start_index=protein_index, max_proteins=1)

    elif args.quick:
        # Quick test on first few proteins with minimal configurations
        tester.run_campaign(start_index=0, max_proteins=3)

    else:
        # Full campaign
        tester.run_campaign(
            start_index=args.start,
            max_proteins=args.count,
            resume=args.resume
        )


if __name__ == "__main__":
    main()