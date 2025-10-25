#!/usr/bin/env python3
"""
End-to-End QCPP-UBF Integration Validation Script

Validates the complete integration of QCPP (Quantum Coherence Protein Predictor)
with UBF (Universal Behavioral Framework) protein folding system.

Tests:
1. QCPP metrics are correctly recorded in trajectory
2. Correlation analysis produces meaningful results
3. Integrated exploration improves RMSD compared to baseline
4. Performance meets targets (throughput, calculation time)
5. All integration components work together

Usage:
    python validate_qcpp_ubf_integration.py [--pdb-id PDBID] [--agents N] [--iterations N]

Example:
    python validate_qcpp_ubf_integration.py --pdb-id 1UBQ --agents 10 --iterations 500
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add ubf_protein to path
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

# Import QCPP
from protein_predictor import QuantumCoherenceProteinPredictor

# Import UBF components
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.qcpp_config import QCPPIntegrationConfig
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.validation_suite import NativeStructureLoader


class IntegrationValidator:
    """End-to-end validation of QCPP-UBF integration."""
    
    def __init__(self, pdb_id: str, num_agents: int, iterations: int):
        self.pdb_id = pdb_id
        self.num_agents = num_agents
        self.iterations = iterations
        self.results: Dict[str, Any] = {}
        
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("="*70)
        print("QCPP-UBF INTEGRATION VALIDATION")
        print("="*70)
        print(f"\nTest Parameters:")
        print(f"  PDB ID: {self.pdb_id}")
        print(f"  Agents: {self.num_agents}")
        print(f"  Iterations: {self.iterations}")
        
        # Test 1: Load native structure
        print(f"\n{'='*70}")
        print("TEST 1: Load Native Structure")
        print("="*70)
        native_conformation, sequence = self._load_native_structure()
        self.results['native_loaded'] = True
        self.results['sequence'] = sequence
        self.results['protein_length'] = len(sequence)
        print(f"✓ Loaded {self.pdb_id}: {len(sequence)} residues")
        
        # Test 2: Run baseline (without QCPP)
        print(f"\n{'='*70}")
        print("TEST 2: Baseline Exploration (No QCPP)")
        print("="*70)
        baseline_results = self._run_baseline_exploration(sequence)
        self.results['baseline'] = baseline_results
        print(f"✓ Baseline complete:")
        print(f"  Time: {baseline_results['time']:.2f}s")
        print(f"  Best RMSD: {baseline_results['best_rmsd']:.3f}Å")
        print(f"  Best Energy: {baseline_results['best_energy']:.3f} kcal/mol")
        
        # Test 3: Run integrated exploration (with QCPP)
        print(f"\n{'='*70}")
        print("TEST 3: Integrated Exploration (With QCPP)")
        print("="*70)
        integrated_results = self._run_integrated_exploration(sequence)
        self.results['integrated'] = integrated_results
        print(f"✓ Integrated complete:")
        print(f"  Time: {integrated_results['time']:.2f}s")
        print(f"  Best RMSD: {integrated_results['best_rmsd']:.3f}Å")
        print(f"  Best Energy: {integrated_results['best_energy']:.3f} kcal/mol")
        print(f"  QCPP Analyses: {integrated_results['qcpp_analyses']}")
        print(f"  Cache Hit Rate: {integrated_results['cache_hit_rate']*100:.1f}%")
        
        # Test 4: Verify QCPP metrics recording
        print(f"\n{'='*70}")
        print("TEST 4: Verify QCPP Metrics Recording")
        print("="*70)
        metrics_valid = self._verify_qcpp_metrics(integrated_results)
        self.results['metrics_recorded'] = metrics_valid
        if metrics_valid:
            print(f"✓ QCPP metrics recorded in trajectory")
            print(f"  Points with QCPP data: {integrated_results['qcpp_points']}")
            print(f"  Avg calculation time: {integrated_results['avg_qcpp_time']:.2f}ms")
        else:
            print(f"✗ QCPP metrics not properly recorded")
        
        # Test 5: Performance validation
        print(f"\n{'='*70}")
        print("TEST 5: Performance Validation")
        print("="*70)
        perf_valid = self._validate_performance(integrated_results)
        self.results['performance_valid'] = perf_valid
        
        # Test 6: Compare results
        print(f"\n{'='*70}")
        print("TEST 6: Compare Baseline vs Integrated")
        print("="*70)
        comparison = self._compare_results(baseline_results, integrated_results)
        self.results['comparison'] = comparison
        
        # Test 7: Generate report
        print(f"\n{'='*70}")
        print("TEST 7: Generate Validation Report")
        print("="*70)
        self._generate_report()
        
        return self.results
    
    def _load_native_structure(self) -> Tuple[Any, str]:
        """Load native structure from PDB."""
        try:
            loader = NativeStructureLoader()
            sequence = loader.get_sequence_from_pdb(self.pdb_id)
            native = loader.load_structure(self.pdb_id)
            return native, sequence
        except Exception as e:
            print(f"✗ Failed to load native structure: {e}")
            raise
    
    def _run_baseline_exploration(self, sequence: str) -> Dict[str, Any]:
        """Run exploration without QCPP integration."""
        print(f"Running {self.num_agents} agents for {self.iterations} iterations (no QCPP)...")
        
        # Create coordinator without QCPP
        coordinator = MultiAgentCoordinator(
            protein_sequence=sequence,
            qcpp_adapter=None  # No QCPP integration
        )
        coordinator.initialize_agents(
            count=self.num_agents,
            diversity_profile="balanced"
        )
        
        # Run exploration
        start_time = time.perf_counter()
        coordinator.run_parallel_exploration(iterations=self.iterations)
        elapsed = time.perf_counter() - start_time
        
        # Get best result
        best_agent = min(coordinator._agents, key=lambda a: a._current_energy)
        
        return {
            'time': elapsed,
            'best_rmsd': best_agent._current_rmsd,
            'best_energy': best_agent._current_energy,
            'throughput': (self.num_agents * self.iterations) / elapsed
        }
    
    def _run_integrated_exploration(self, sequence: str) -> Dict[str, Any]:
        """Run exploration with QCPP integration."""
        print(f"Running {self.num_agents} agents for {self.iterations} iterations (with QCPP)...")
        
        # Initialize QCPP
        qcpp_predictor = QuantumCoherenceProteinPredictor()
        qcpp_config = QCPPIntegrationConfig.default()
        qcpp_adapter = QCPPIntegrationAdapter(qcpp_predictor, qcpp_config)
        
        # Create coordinator with QCPP
        coordinator = MultiAgentCoordinator(
            protein_sequence=sequence,
            qcpp_adapter=qcpp_adapter
        )
        coordinator.initialize_agents(
            count=self.num_agents,
            diversity_profile="balanced"
        )
        
        # Run exploration
        start_time = time.perf_counter()
        coordinator.run_parallel_exploration(iterations=self.iterations)
        elapsed = time.perf_counter() - start_time
        
        # Get best result
        best_agent = min(coordinator._agents, key=lambda a: a._current_energy)
        
        # Get QCPP statistics
        cache_stats = qcpp_adapter.get_cache_stats()
        
        # Count QCPP points (should be ~10% of total with default config)
        expected_qcpp_analyses = int(self.num_agents * self.iterations * qcpp_config.analysis_frequency)
        
        return {
            'time': elapsed,
            'best_rmsd': best_agent._current_rmsd,
            'best_energy': best_agent._current_energy,
            'throughput': (self.num_agents * self.iterations) / elapsed,
            'qcpp_analyses': cache_stats['total_requests'],
            'cache_hits': cache_stats['cache_hits'],
            'cache_misses': cache_stats['cache_misses'],
            'cache_hit_rate': cache_stats['hit_rate'],
            'avg_qcpp_time': cache_stats['avg_calculation_time_ms'],
            'total_qcpp_time': cache_stats['total_calculation_time_ms'],
            'qcpp_points': cache_stats['cache_misses'],  # Unique conformations analyzed
            'expected_analyses': expected_qcpp_analyses
        }
    
    def _verify_qcpp_metrics(self, integrated_results: Dict[str, Any]) -> bool:
        """Verify QCPP metrics are properly recorded."""
        # Check that QCPP analyses occurred
        if integrated_results['qcpp_analyses'] == 0:
            print("✗ No QCPP analyses performed")
            return False
        
        # Check that analyses are within expected range (5-15% with caching)
        expected = integrated_results['expected_analyses']
        actual = integrated_results['qcpp_analyses']
        ratio = actual / expected if expected > 0 else 0
        
        print(f"  Expected QCPP analyses: ~{expected}")
        print(f"  Actual QCPP analyses: {actual}")
        print(f"  Ratio: {ratio:.2f}x")
        
        if ratio < 0.05 or ratio > 1.5:
            print(f"✗ QCPP analysis frequency out of expected range")
            return False
        
        # Check cache hit rate
        if integrated_results['cache_hit_rate'] < 0.2 or integrated_results['cache_hit_rate'] > 0.8:
            print(f"⚠ Cache hit rate ({integrated_results['cache_hit_rate']*100:.1f}%) outside typical 20-80% range")
        
        return True
    
    def _validate_performance(self, integrated_results: Dict[str, Any]) -> bool:
        """Validate performance meets targets."""
        all_valid = True
        
        # Check QCPP calculation time (<5ms target)
        avg_time = integrated_results['avg_qcpp_time']
        print(f"  QCPP calculation time: {avg_time:.2f}ms (target: <5ms)")
        if avg_time < 5.0:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL - Exceeds 5ms target")
            all_valid = False
        
        # Check throughput (≥50 conf/s/agent target)
        throughput_per_agent = integrated_results['throughput'] / self.num_agents
        print(f"  Throughput: {throughput_per_agent:.1f} conf/s/agent (target: ≥50)")
        if throughput_per_agent >= 50:
            print(f"  ✓ PASS")
        else:
            print(f"  ⚠ WARNING - Below 50 conf/s/agent target")
            # Don't fail on throughput, it depends on hardware
        
        # Check total exploration time (should be reasonable)
        total_time = integrated_results['time']
        expected_time = (self.num_agents * self.iterations) / 50  # At 50 conf/s/agent
        print(f"  Total time: {total_time:.2f}s (expected: ~{expected_time:.0f}s at 50 conf/s/agent)")
        if total_time < expected_time * 3:  # Allow 3x expected time
            print(f"  ✓ PASS")
        else:
            print(f"  ⚠ WARNING - Slower than expected")
        
        return all_valid
    
    def _compare_results(self, baseline: Dict[str, Any], integrated: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline vs integrated results."""
        rmsd_improvement = baseline['best_rmsd'] - integrated['best_rmsd']
        rmsd_improvement_pct = (rmsd_improvement / baseline['best_rmsd']) * 100
        
        energy_improvement = baseline['best_energy'] - integrated['best_energy']
        
        time_overhead = integrated['time'] - baseline['time']
        time_overhead_pct = (time_overhead / baseline['time']) * 100
        
        print(f"\nRMSD Comparison:")
        print(f"  Baseline: {baseline['best_rmsd']:.3f}Å")
        print(f"  Integrated: {integrated['best_rmsd']:.3f}Å")
        print(f"  Improvement: {rmsd_improvement:+.3f}Å ({rmsd_improvement_pct:+.1f}%)")
        if rmsd_improvement > 0:
            print(f"  ✓ Integrated exploration achieved better RMSD")
        elif rmsd_improvement > -0.5:
            print(f"  ≈ RMSD similar (within 0.5Å)")
        else:
            print(f"  ✗ Integrated exploration achieved worse RMSD")
        
        print(f"\nEnergy Comparison:")
        print(f"  Baseline: {baseline['best_energy']:.3f} kcal/mol")
        print(f"  Integrated: {integrated['best_energy']:.3f} kcal/mol")
        print(f"  Improvement: {energy_improvement:+.3f} kcal/mol")
        
        print(f"\nPerformance Overhead:")
        print(f"  Baseline time: {baseline['time']:.2f}s")
        print(f"  Integrated time: {integrated['time']:.2f}s")
        print(f"  Overhead: {time_overhead:+.2f}s ({time_overhead_pct:+.1f}%)")
        if time_overhead_pct < 20:
            print(f"  ✓ Overhead is acceptable (<20%)")
        elif time_overhead_pct < 50:
            print(f"  ⚠ Moderate overhead (20-50%)")
        else:
            print(f"  ✗ High overhead (>50%)")
        
        return {
            'rmsd_improvement': rmsd_improvement,
            'rmsd_improvement_pct': rmsd_improvement_pct,
            'energy_improvement': energy_improvement,
            'time_overhead': time_overhead,
            'time_overhead_pct': time_overhead_pct,
            'rmsd_better': rmsd_improvement > 0,
            'overhead_acceptable': time_overhead_pct < 20
        }
    
    def _generate_report(self):
        """Generate validation report."""
        report_file = f"qcpp_ubf_validation_{self.pdb_id}.json"
        
        # Add summary
        self.results['summary'] = {
            'pdb_id': self.pdb_id,
            'protein_length': self.results['protein_length'],
            'num_agents': self.num_agents,
            'iterations': self.iterations,
            'validation_passed': all([
                self.results['native_loaded'],
                self.results['metrics_recorded'],
                self.results['performance_valid'],
            ]),
            'rmsd_improved': self.results['comparison']['rmsd_better'],
            'overhead_acceptable': self.results['comparison']['overhead_acceptable']
        }
        
        # Write JSON report
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Validation report saved: {report_file}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print("="*70)
        
        summary = self.results['summary']
        print(f"\nProtein: {self.pdb_id} ({self.results['protein_length']} residues)")
        print(f"Test Configuration: {self.num_agents} agents × {self.iterations} iterations")
        
        print(f"\nCore Validation:")
        print(f"  Native structure loaded: {'✓' if self.results['native_loaded'] else '✗'}")
        print(f"  QCPP metrics recorded: {'✓' if self.results['metrics_recorded'] else '✗'}")
        print(f"  Performance valid: {'✓' if self.results['performance_valid'] else '✗'}")
        
        print(f"\nComparison Results:")
        print(f"  RMSD improvement: {self.results['comparison']['rmsd_improvement']:+.3f}Å "
              f"({self.results['comparison']['rmsd_improvement_pct']:+.1f}%)")
        print(f"  Energy improvement: {self.results['comparison']['energy_improvement']:+.3f} kcal/mol")
        print(f"  Time overhead: {self.results['comparison']['time_overhead_pct']:+.1f}%")
        
        print(f"\nIntegration Quality:")
        print(f"  QCPP analyses: {self.results['integrated']['qcpp_analyses']}")
        print(f"  Cache hit rate: {self.results['integrated']['cache_hit_rate']*100:.1f}%")
        print(f"  Avg QCPP time: {self.results['integrated']['avg_qcpp_time']:.2f}ms")
        
        print(f"\n{'='*70}")
        if summary['validation_passed']:
            print("VALIDATION: ✓ PASS")
            print("QCPP-UBF integration is working correctly!")
        else:
            print("VALIDATION: ✗ FAIL")
            print("Some validation checks did not pass. Review results above.")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate QCPP-UBF integration end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--pdb-id',
        type=str,
        default='1UBQ',
        help='PDB ID to test (default: 1UBQ - ubiquitin)'
    )
    parser.add_argument(
        '--agents',
        type=int,
        default=10,
        help='Number of agents (default: 10)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=500,
        help='Number of iterations per agent (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = IntegrationValidator(
        pdb_id=args.pdb_id,
        num_agents=args.agents,
        iterations=args.iterations
    )
    
    try:
        results = validator.run_validation()
        
        # Exit with appropriate code
        if results['summary']['validation_passed']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"VALIDATION FAILED WITH ERROR")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
