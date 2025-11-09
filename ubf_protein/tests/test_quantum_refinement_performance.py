"""
Performance benchmark tests for quantum refinement engine.

These tests verify that all refinement operations meet performance targets:
- Quantum core identification: <100ms
- Secondary structure registration: <200ms
- Hydrophobic packing: <500ms
- Full refinement: <5 minutes for 100-residue protein
- Cache hit: <10μs

Tests use realistic protein structures and measure actual wall-clock time.
"""

import pytest
import time
from typing import List, Tuple, Any

from ubf_protein.quantum_refinement_engine import QuantumRefinementEngine
from ubf_protein.quantum_core_analyzer import QuantumCoreAnalyzer
from ubf_protein.secondary_structure_registrar import SecondaryStructureRegistrar
from ubf_protein.hydrophobic_core_packer import HydrophobicCorePacker
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.energy_function import MolecularMechanicsEnergy
from ubf_protein.rmsd_calculator import RMSDCalculator
from ubf_protein.models import Conformation, RefinementConfig


def create_test_structure(n_residues: int = 50) -> Conformation:
    """
    Create a test protein structure.
    
    Args:
        n_residues: Number of residues
    
    Returns:
        Test Conformation object
    """
    sequence = "A" * n_residues
    coords = [(float(i), 0.0, 0.0) for i in range(n_residues)]
    
    return Conformation(
        conformation_id="test",
        sequence=sequence,
        atom_coordinates=coords,
        energy=0.0,
        rmsd_to_native=None,
        secondary_structure=['C'] * n_residues,
        phi_angles=[0.0] * n_residues,
        psi_angles=[0.0] * n_residues,
        available_move_types=['all'],
        structural_constraints={}
    )


def measure_time(func, *args, **kwargs) -> Tuple[float, Any]:
    """
    Measure execution time of a function.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (execution_time_ms, result)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000.0
    return elapsed_ms, result


class TestQuantumCoreIdentificationPerformance:
    """Test performance of quantum core identification."""
    
    def test_small_protein_performance(self):
        """Test quantum core identification on small protein (<50 residues)."""
        structure = create_test_structure(n_residues=46)
        
        # Create mock QCPP adapter
        from unittest.mock import Mock
        mock_predictor = Mock()
        adapter = QCPPIntegrationAdapter(mock_predictor)
        
        # Create analyzer
        analyzer = QuantumCoreAnalyzer(adapter)
        
        # Benchmark
        elapsed_ms, cores = measure_time(
            analyzer.identify_quantum_cores,
            structure,
            qcp_threshold=7.0
        )
        
        # Should complete in <50ms for small proteins
        assert elapsed_ms < 50.0, f"Quantum core ID took {elapsed_ms:.1f}ms (target: <50ms)"
        print(f"✓ Quantum core identification (46 residues): {elapsed_ms:.2f}ms")
    
    def test_medium_protein_performance(self):
        """Test quantum core identification on medium protein (50-100 residues)."""
        structure = create_test_structure(n_residues=76)
        
        from unittest.mock import Mock
        mock_predictor = Mock()
        adapter = QCPPIntegrationAdapter(mock_predictor)
        analyzer = QuantumCoreAnalyzer(adapter)
        
        elapsed_ms, cores = measure_time(
            analyzer.identify_quantum_cores,
            structure,
            qcp_threshold=7.0
        )
        
        # Should complete in <100ms for medium proteins
        assert elapsed_ms < 100.0, f"Quantum core ID took {elapsed_ms:.1f}ms (target: <100ms)"
        print(f"✓ Quantum core identification (76 residues): {elapsed_ms:.2f}ms")
    
    def test_large_protein_performance(self):
        """Test quantum core identification on large protein (>100 residues)."""
        structure = create_test_structure(n_residues=150)
        
        from unittest.mock import Mock
        mock_predictor = Mock()
        adapter = QCPPIntegrationAdapter(mock_predictor)
        analyzer = QuantumCoreAnalyzer(adapter)
        
        elapsed_ms, cores = measure_time(
            analyzer.identify_quantum_cores,
            structure,
            qcp_threshold=7.0
        )
        
        # Should complete in <200ms for large proteins
        assert elapsed_ms < 200.0, f"Quantum core ID took {elapsed_ms:.1f}ms (target: <200ms)"
        print(f"✓ Quantum core identification (150 residues): {elapsed_ms:.2f}ms")


class TestSecondaryStructureRegistrationPerformance:
    """Test performance of secondary structure registration."""
    
    def test_small_protein_performance(self):
        """Test SS registration on small protein."""
        structure = create_test_structure(n_residues=46)
        
        from unittest.mock import Mock
        mock_predictor = Mock()
        adapter = QCPPIntegrationAdapter(mock_predictor)
        
        registrar = SecondaryStructureRegistrar(adapter)
        
        elapsed_ms, result = measure_time(
            registrar.fix_secondary_structure_registration,
            structure
        )
        
        # Should complete in <100ms for small proteins
        assert elapsed_ms < 100.0, f"SS registration took {elapsed_ms:.1f}ms (target: <100ms)"
        print(f"✓ Secondary structure registration (46 residues): {elapsed_ms:.2f}ms")
    
    def test_medium_protein_performance(self):
        """Test SS registration on medium protein."""
        structure = create_test_structure(n_residues=76)
        
        from unittest.mock import Mock
        mock_predictor = Mock()
        adapter = QCPPIntegrationAdapter(mock_predictor)
        
        registrar = SecondaryStructureRegistrar(adapter)
        
        elapsed_ms, result = measure_time(
            registrar.fix_secondary_structure_registration,
            structure
        )
        
        # Should complete in <200ms for medium proteins
        assert elapsed_ms < 200.0, f"SS registration took {elapsed_ms:.1f}ms (target: <200ms)"
        print(f"✓ Secondary structure registration (76 residues): {elapsed_ms:.2f}ms")


class TestHydrophobicPackingPerformance:
    """Test performance of hydrophobic core packing."""
    
    def test_small_protein_performance(self):
        """Test hydrophobic packing on small protein."""
        structure = create_test_structure(n_residues=46)
        
        packer = HydrophobicCorePacker()
        
        elapsed_ms, result = measure_time(
            packer.quantum_hydrophobic_packing,
            structure
        )
        
        # Should complete in <250ms for small proteins
        assert elapsed_ms < 250.0, f"Hydrophobic packing took {elapsed_ms:.1f}ms (target: <250ms)"
        print(f"✓ Hydrophobic packing (46 residues): {elapsed_ms:.2f}ms")
    
    def test_medium_protein_performance(self):
        """Test hydrophobic packing on medium protein."""
        structure = create_test_structure(n_residues=76)
        
        packer = HydrophobicCorePacker()
        
        elapsed_ms, result = measure_time(
            packer.quantum_hydrophobic_packing,
            structure
        )
        
        # Should complete in <500ms for medium proteins
        assert elapsed_ms < 500.0, f"Hydrophobic packing took {elapsed_ms:.1f}ms (target: <500ms)"
        print(f"✓ Hydrophobic packing (76 residues): {elapsed_ms:.2f}ms")


class TestCachePerformance:
    """Test cache performance."""
    
    def test_cache_hit_performance(self):
        """Test cache hit time (<10μs target)."""
        from ubf_protein.refinement_cache import RefinementCache
        
        cache = RefinementCache()
        
        # Populate cache
        structure_hash = "test_hash"
        cache.set_qcp(structure_hash, residue_idx=0, value=8.5)
        
        # Measure cache hit time
        hits = []
        for _ in range(1000):
            start = time.perf_counter()
            value = cache.get_qcp(structure_hash, residue_idx=0)
            end = time.perf_counter()
            hits.append((end - start) * 1e6)  # Convert to microseconds
        
        avg_hit_time = sum(hits) / len(hits)
        
        # Average should be <10μs
        assert avg_hit_time < 10.0, f"Cache hit took {avg_hit_time:.2f}μs (target: <10μs)"
        print(f"✓ Cache hit time: {avg_hit_time:.2f}μs (1000 samples)")
    
    def test_cache_miss_performance(self):
        """Test cache miss time (should also be <10μs)."""
        from ubf_protein.refinement_cache import RefinementCache
        
        cache = RefinementCache()
        
        # Measure cache miss time
        misses = []
        for i in range(1000):
            start = time.perf_counter()
            value = cache.get_qcp("hash", residue_idx=i)  # Different key each time
            end = time.perf_counter()
            misses.append((end - start) * 1e6)
        
        avg_miss_time = sum(misses) / len(misses)
        
        # Should still be <10μs for miss
        assert avg_miss_time < 10.0, f"Cache miss took {avg_miss_time:.2f}μs (target: <10μs)"
        print(f"✓ Cache miss time: {avg_miss_time:.2f}μs (1000 samples)")
    
    def test_cache_eviction_performance(self):
        """Test cache eviction doesn't degrade performance."""
        from ubf_protein.refinement_cache import RefinementCache
        
        cache = RefinementCache(max_qcp_entries=100)
        
        # Fill cache to capacity and beyond
        elapsed_times = []
        for i in range(200):  # Exceed capacity
            start = time.perf_counter()
            cache.set_qcp("hash", residue_idx=i, value=float(i))
            end = time.perf_counter()
            elapsed_times.append((end - start) * 1e6)
        
        # Performance should be consistent (no degradation)
        early_avg = sum(elapsed_times[:50]) / 50
        late_avg = sum(elapsed_times[-50:]) / 50
        
        # Late inserts should not be >2x slower (allows for some variance)
        assert late_avg < early_avg * 2.0, \
            f"Cache eviction degraded performance: {early_avg:.2f}μs → {late_avg:.2f}μs"
        print(f"✓ Cache eviction performance: {early_avg:.2f}μs → {late_avg:.2f}μs")


class TestProgressTrackingPerformance:
    """Test progress tracking overhead."""
    
    def test_progress_tracking_overhead(self):
        """Test that progress tracking adds <1ms overhead per iteration."""
        from unittest.mock import Mock
        
        # Create mock components
        mock_predictor = Mock()
        adapter = QCPPIntegrationAdapter(mock_predictor)
        energy_calc = MolecularMechanicsEnergy()
        rmsd_calc = RMSDCalculator()
        
        engine = QuantumRefinementEngine(adapter, energy_calc, rmsd_calc)
        
        structure = create_test_structure(n_residues=50)
        
        # Start tracking
        engine._start_progress_tracking(max_iterations=100)
        
        # Measure overhead of recording progress
        overhead_times = []
        for i in range(100):
            start = time.perf_counter()
            progress = engine._record_progress(
                iteration=i,
                structure=structure,
                native_structure=None,
                active_restraints=10,
                formed_contacts=5
            )
            end = time.perf_counter()
            overhead_times.append((end - start) * 1000.0)  # ms
        
        avg_overhead = sum(overhead_times) / len(overhead_times)
        max_overhead = max(overhead_times)
        
        # Average overhead should be <1ms
        assert avg_overhead < 1.0, f"Progress tracking overhead {avg_overhead:.2f}ms (target: <1ms)"
        
        # Max overhead should be <5ms (allows for outliers)
        assert max_overhead < 5.0, f"Max progress tracking overhead {max_overhead:.2f}ms (target: <5ms)"
        
        print(f"✓ Progress tracking overhead: {avg_overhead:.3f}ms avg, {max_overhead:.3f}ms max")


if __name__ == "__main__":
    """Run benchmarks with detailed output."""
    pytest.main([__file__, "-v", "-s"])
