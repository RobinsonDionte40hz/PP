#!/usr/bin/env python3
"""
Comprehensive Benchmark for Protein Predictions and Aggregation Screening

This module provides extensive benchmarking with COMPLETE JSON output capturing
ALL metrics, timings, configurations, and results.

Features:
- Full prediction benchmarks with RMSD validation
- Aggregation screening benchmarks
- Performance metrics (latency, throughput, memory)
- Comparative analysis across configurations
- Complete JSON export with all data fields

Usage:
  python comprehensive_benchmark.py                     # Run all benchmarks
  python comprehensive_benchmark.py --quick             # Quick benchmark
  python comprehensive_benchmark.py --predictions-only  # Only prediction tests
  python comprehensive_benchmark.py --screening-only    # Only screening tests
  python comprehensive_benchmark.py --output results.json  # Custom output file
"""

import sys
import json
import time
import argparse
import platform
import tracemalloc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

# Import from public API (SOLID: Dependency Inversion)
from ubf_protein.api import (
    PredictionRunner,
    PredictionConfig,
    PredictionResults,
    AggregationScreener,
    ScreeningConfig,
    AggregationRisk,
    get_optimal_settings,
    get_quick_test_settings,
)

# Import test utilities
from test_protein import (
    discover_pdb_files,
    KNOWN_PROTEINS,
    download_pdb,
    load_sequence_from_pdb
)


# ============================================================================
# Test Protein Sets
# ============================================================================

# Small test proteins for quick benchmarks (optimal performance range)
QUICK_TEST_PROTEINS = [
    {"pdb": "1CRN", "name": "Crambin", "residues": 46, "fold": "alpha+beta"},
    {"pdb": "1VII", "name": "Villin Headpiece", "residues": 36, "fold": "alpha"},
]

# Standard benchmark proteins - Focus on small proteins where physics-based methods excel
# Note: Physics-based folding (without ML) works best on proteins <60 residues
STANDARD_TEST_PROTEINS = [
    # Small proteins (<50 residues) - PRIMARY BENCHMARK SET
    {"pdb": "1VII", "name": "Villin Headpiece", "residues": 36, "fold": "alpha", "category": "small"},
    {"pdb": "1CRN", "name": "Crambin", "residues": 46, "fold": "alpha+beta", "category": "small"},
    {"pdb": "1PIN", "name": "Pin1 WW Domain", "residues": 34, "fold": "beta", "category": "small"},
    {"pdb": "1L2Y", "name": "Trp-cage", "residues": 20, "fold": "alpha", "category": "mini"},
    # Medium proteins (50-80 residues) - EXTENDED BENCHMARK
    {"pdb": "1GB1", "name": "Protein G B1", "residues": 56, "fold": "alpha+beta", "category": "medium"},
    {"pdb": "1BPI", "name": "BPTI", "residues": 58, "fold": "alpha+beta", "category": "medium"},
    {"pdb": "2CI2", "name": "CI2 Inhibitor", "residues": 64, "fold": "alpha+beta", "category": "medium"},
    {"pdb": "1UBQ", "name": "Ubiquitin", "residues": 76, "fold": "alpha+beta", "category": "medium"},
]

# Comprehensive protein set - includes challenging larger proteins for completeness
# Larger proteins >100 residues are included for comparison but expected to have higher RMSD
COMPREHENSIVE_TEST_PROTEINS = STANDARD_TEST_PROTEINS + [
    {"pdb": "1ROP", "name": "Rop Protein", "residues": 63, "fold": "alpha", "category": "medium"},
    {"pdb": "3SSI", "name": "Subtilisin Inhibitor", "residues": 107, "fold": "alpha+beta", "category": "large"},
    {"pdb": "1LYZ", "name": "Lysozyme", "residues": 129, "fold": "alpha+beta", "category": "large"},
]

# Scientific context for benchmark interpretation
BENCHMARK_CONTEXT = """
BENCHMARK INTERPRETATION GUIDE
==============================
This benchmark evaluates EmergentFolds, a physics-based protein structure prediction
system using quantum coherence principles and multi-agent exploration.

Performance expectations by protein size:
- Mini proteins (<25 residues): RMSD 2-4Å expected (excellent)
- Small proteins (25-50 residues): RMSD 4-7Å expected (good)
- Medium proteins (50-80 residues): RMSD 6-12Å expected (acceptable)
- Large proteins (>100 residues): RMSD 10-20+Å expected (challenging)

Key differentiators from ML methods (AlphaFold/ESMFold):
- No training data required - works on novel sequences
- Physics-based energy function provides interpretable results
- Real-time exploration visualization
- Aggregation risk screening capability
- Faster inference for small proteins

Metrics explained:
- RMSD: Root-mean-square deviation from native structure (lower is better)
- GDT-TS: Global Distance Test Total Score (higher is better, 0-100)
- TM-score: Template Modeling score (higher is better, 0-1)
- Energy: Molecular mechanics energy in kcal/mol (lower is more stable)
"""

# Test sequences for screening (various aggregation propensities)
SCREENING_TEST_SEQUENCES = [
    {
        "name": "stable_alpha",
        "sequence": "AEEEKKKKEEEEKKKKEEEEKKKK",  # Charged, should fold
        "expected_risk": "low",
    },
    {
        "name": "hydrophobic_stretch",
        "sequence": "VVVVVVVVVVVVVVVVVVVVVVVV",  # Aggregation-prone
        "expected_risk": "critical",
    },
    {
        "name": "balanced",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",  # Natural composition
        "expected_risk": "moderate",
    },
    {
        "name": "charged_only",
        "sequence": "RKDERKDERKDERKDERKDE",  # All charged
        "expected_risk": "low",
    },
    {
        "name": "aromatic_rich",
        "sequence": "FWYFWYFWYFWYFWYFWYFW",  # Aromatic-heavy
        "expected_risk": "high",
    },
    {
        "name": "glycine_rich",
        "sequence": "GGGGSGGGGSGGGGSGGGGG",  # Flexible, disordered
        "expected_risk": "moderate",
    },
    {
        "name": "proline_helix_breaker",
        "sequence": "AAAAPAAAAPAAAAPAAAAAP",  # Proline disrupts helix
        "expected_risk": "moderate",
    },
    {
        "name": "cysteine_rich",
        "sequence": "CRCRCRCRCRCRCRCRCRCR",  # Disulfide potential
        "expected_risk": "moderate",
    },
]


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class SystemInfo:
    """System and environment information."""
    python_version: str
    python_implementation: str
    platform: str
    processor: str
    timestamp: str
    benchmark_version: str = "1.0.0"
    
    @classmethod
    def capture(cls) -> 'SystemInfo':
        return cls(
            python_version=sys.version,
            python_implementation=platform.python_implementation(),
            platform=platform.platform(),
            processor=platform.processor(),
            timestamp=datetime.now().isoformat(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics from benchmarking."""
    move_evaluation_latency_ms: Optional[float] = None
    memory_retrieval_us: Optional[float] = None
    agent_memory_mb: Optional[float] = None
    throughput_conf_per_sec: Optional[float] = None
    
    # Targets for pass/fail
    latency_target_ms: float = 2.0
    retrieval_target_us: float = 10.0
    memory_target_mb: float = 50.0
    throughput_target: float = 5000.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['pass_fail'] = {
            'move_evaluation_latency': (
                self.move_evaluation_latency_ms is not None and 
                self.move_evaluation_latency_ms < self.latency_target_ms
            ),
            'memory_retrieval': (
                self.memory_retrieval_us is not None and 
                self.memory_retrieval_us < self.retrieval_target_us
            ),
            'agent_memory': (
                self.agent_memory_mb is not None and 
                self.agent_memory_mb < self.memory_target_mb
            ),
            'throughput': (
                self.throughput_conf_per_sec is not None and 
                self.throughput_conf_per_sec >= self.throughput_target
            ),
        }
        return result


@dataclass
class PredictionBenchmarkResult:
    """Complete result from a single prediction benchmark."""
    # Protein info
    pdb_id: str
    protein_name: str
    sequence: str
    sequence_length: int
    fold_type: Optional[str] = None
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Core results
    success: bool = False
    error_message: Optional[str] = None
    
    # Energy metrics
    initial_energy: Optional[float] = None
    final_energy: Optional[float] = None
    energy_improvement: Optional[float] = None
    
    # RMSD metrics
    best_rmsd: Optional[float] = None
    folding_rmsd: Optional[float] = None
    gdt_ts_score: Optional[float] = None
    tm_score: Optional[float] = None
    
    # Quality assessment
    quality_assessment: Optional[str] = None
    
    # Exploration statistics
    conformations_explored: int = 0
    unique_structures: Optional[int] = None
    
    # Timing
    exploration_time_seconds: float = 0.0
    refinement_time_seconds: Optional[float] = None
    total_time_seconds: float = 0.0
    throughput_conf_per_sec: float = 0.0
    
    # Refinement (if enabled)
    refinement_applied: bool = False
    refinement_initial_rmsd: Optional[float] = None
    refinement_final_rmsd: Optional[float] = None
    refinement_improvement_percent: Optional[float] = None
    
    # QCPP metrics
    qcpp_enabled: bool = False
    qcpp_total_analyses: int = 0
    qcpp_cache_hit_rate: float = 0.0
    qcpp_avg_time_ms: float = 0.0
    qcp_score: Optional[float] = None
    qaap_alignment: Optional[float] = None
    resonance_40hz: Optional[float] = None
    
    # Geometric analysis
    geometric_analysis: Optional[Dict[str, Any]] = None
    
    # Mediator stats
    mediator_stats: Optional[Dict[str, Any]] = None
    
    # Agent behavior
    final_aggressiveness: Optional[float] = None
    final_consistency: Optional[float] = None
    
    # Coordinates (optional, for detailed analysis)
    best_coords_included: bool = False
    best_coords: Optional[List[List[float]]] = None
    
    def to_dict(self, include_coords: bool = False) -> Dict[str, Any]:
        result = asdict(self)
        if not include_coords:
            result.pop('best_coords', None)
        return result


@dataclass
class ScreeningBenchmarkResult:
    """Complete result from a single screening benchmark."""
    # Sequence info
    sequence_name: str
    sequence: str
    sequence_length: int
    expected_risk: Optional[str] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Core results
    success: bool = False
    error_message: Optional[str] = None
    
    # Risk assessment
    aggregation_score: float = 0.0
    risk_level: str = "unknown"
    risk_factors: List[str] = field(default_factory=list)
    passes_screening: bool = False
    
    # Individual scores (0-1, higher = better)
    energy_score: float = 0.0
    structure_score: float = 0.0
    hydrophobic_score: float = 0.0
    convergence_score: float = 0.0
    compactness_score: float = 0.0
    
    # Raw metrics
    final_energy: float = 0.0
    secondary_structure_pct: float = 0.0
    hydrophobic_clustering: float = 0.0
    radius_of_gyration: float = 0.0
    convergence_iterations: int = 0
    
    # Conformations found
    num_unique_conformations: int = 0
    conformations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    screening_time_ms: float = 0.0
    iterations_used: int = 0
    
    # Validation
    risk_prediction_correct: Optional[bool] = None
    
    def to_dict(self, include_conformations: bool = False) -> Dict[str, Any]:
        result = asdict(self)
        if not include_conformations:
            result['conformations'] = []
        return result


@dataclass
class ComprehensiveBenchmarkResults:
    """Complete benchmark results with all data."""
    # Metadata
    system_info: SystemInfo
    benchmark_mode: str  # 'quick', 'standard', 'comprehensive'
    
    # Performance metrics
    performance_metrics: PerformanceMetrics
    
    # Prediction benchmarks
    prediction_results: List[PredictionBenchmarkResult] = field(default_factory=list)
    prediction_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Screening benchmarks
    screening_results: List[ScreeningBenchmarkResult] = field(default_factory=list)
    screening_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Overall statistics
    total_runtime_seconds: float = 0.0
    total_proteins_tested: int = 0
    total_sequences_screened: int = 0
    
    def to_dict(self, include_coords: bool = False, include_conformations: bool = False) -> Dict[str, Any]:
        return {
            'benchmark_context': BENCHMARK_CONTEXT,
            'system_info': self.system_info.to_dict(),
            'benchmark_mode': self.benchmark_mode,
            'performance_metrics': self.performance_metrics.to_dict(),
            'prediction_results': [r.to_dict(include_coords) for r in self.prediction_results],
            'prediction_summary': self.prediction_summary,
            'screening_results': [r.to_dict(include_conformations) for r in self.screening_results],
            'screening_summary': self.screening_summary,
            'total_runtime_seconds': self.total_runtime_seconds,
            'total_proteins_tested': self.total_proteins_tested,
            'total_sequences_screened': self.total_sequences_screened,
        }


# ============================================================================
# Benchmark Runner
# ============================================================================

class ComprehensiveBenchmark:
    """
    Comprehensive benchmark runner for predictions and screening.
    
    Captures ALL data in structured JSON output.
    """
    
    def __init__(
        self,
        mode: str = "standard",
        output_dir: str = "benchmark_results",
        include_coords: bool = False,
        include_conformations: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            mode: 'quick', 'standard', or 'comprehensive'
            output_dir: Directory for output files
            include_coords: Include 3D coordinates in JSON output
            include_conformations: Include all conformations in screening results
            verbose: Print progress messages
        """
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_coords = include_coords
        self.include_conformations = include_conformations
        self.verbose = verbose
        
        # Select protein sets based on mode
        if mode == "quick":
            self.test_proteins = QUICK_TEST_PROTEINS
            self.screening_sequences = SCREENING_TEST_SEQUENCES[:4]
        elif mode == "comprehensive":
            self.test_proteins = COMPREHENSIVE_TEST_PROTEINS
            self.screening_sequences = SCREENING_TEST_SEQUENCES
        else:
            self.test_proteins = STANDARD_TEST_PROTEINS
            self.screening_sequences = SCREENING_TEST_SEQUENCES
        
        # Results storage
        self.results = ComprehensiveBenchmarkResults(
            system_info=SystemInfo.capture(),
            benchmark_mode=mode,
            performance_metrics=PerformanceMetrics(),
        )
    
    def log(self, message: str):
        """Print message if verbose mode."""
        if self.verbose:
            print(message)
    
    # -------------------------------------------------------------------------
    # Performance Benchmarks
    # -------------------------------------------------------------------------
    
    def benchmark_performance(self) -> PerformanceMetrics:
        """Run all performance benchmarks."""
        self.log("\n" + "=" * 60)
        self.log("PERFORMANCE BENCHMARKS")
        self.log("=" * 60)
        
        metrics = PerformanceMetrics()
        
        # Move evaluation latency
        metrics.move_evaluation_latency_ms = self._benchmark_move_latency()
        
        # Memory retrieval
        metrics.memory_retrieval_us = self._benchmark_memory_retrieval()
        
        # Agent memory footprint
        metrics.agent_memory_mb = self._benchmark_memory_footprint()
        
        # Multi-agent throughput
        metrics.throughput_conf_per_sec = self._benchmark_throughput()
        
        self.results.performance_metrics = metrics
        return metrics
    
    def _benchmark_move_latency(self, iterations: int = 500) -> float:
        """Benchmark move evaluation latency."""
        self.log(f"\n--- Move Evaluation Latency (target: <2ms) ---")
        
        from ubf_protein.models import AdaptiveConfig, ProteinSizeClass
        from ubf_protein.protein_agent import ProteinAgent
        
        config = AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=30,
            initial_frequency_range=(3.0, 15.0),
            initial_coherence_range=(0.2, 1.0),
            stuck_detection_window=10,
            stuck_detection_threshold=5.0,
            memory_significance_threshold=0.3,
            max_memories_per_agent=50,
            convergence_energy_threshold=10.0,
            convergence_rmsd_threshold=2.0,
            max_iterations=1000,
            checkpoint_interval=100
        )
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
            adaptive_config=config
        )
        
        # Warm-up
        for _ in range(10):
            agent.explore_step()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            agent.explore_step()
        elapsed = time.perf_counter() - start
        
        latency_ms = (elapsed * 1000) / iterations
        status = "✅ PASS" if latency_ms < 2.0 else "❌ FAIL"
        self.log(f"Average latency: {latency_ms:.3f}ms {status}")
        
        return latency_ms
    
    def _benchmark_memory_retrieval(self, iterations: int = 5000) -> float:
        """Benchmark memory retrieval performance."""
        self.log(f"\n--- Memory Retrieval (target: <10μs) ---")
        
        from ubf_protein.memory_system import MemorySystem
        from ubf_protein.models import ConformationalMemory, ConsciousnessCoordinates, BehavioralStateData
        
        memory_system = MemorySystem()
        
        # Populate with memories
        for i in range(50):
            memory = ConformationalMemory(
                memory_id=f"mem_{i}",
                move_type="backbone_rotation",
                significance=0.5 + (i % 5) / 10.0,
                energy_change=-10.0 * (i % 3),
                rmsd_change=-0.5,
                success=True,
                timestamp=1000 + i,
                consciousness_state=ConsciousnessCoordinates(8.0, 0.6, 1000),
                behavioral_state=BehavioralStateData(0.5, 0.6, 0.5, 0.4, 0.6, 0.8, 1000)
            )
            memory_system.store_memory(memory)
        
        # Warm-up
        for _ in range(100):
            memory_system.retrieve_relevant_memories("backbone_rotation")
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            memory_system.retrieve_relevant_memories("backbone_rotation")
        elapsed = time.perf_counter() - start
        
        retrieval_us = (elapsed * 1_000_000) / iterations
        status = "✅ PASS" if retrieval_us < 10.0 else "❌ FAIL"
        self.log(f"Average retrieval: {retrieval_us:.3f}μs {status}")
        
        return retrieval_us
    
    def _benchmark_memory_footprint(self) -> float:
        """Benchmark agent memory footprint."""
        self.log(f"\n--- Agent Memory Footprint (target: <50MB) ---")
        
        from ubf_protein.models import AdaptiveConfig, ProteinSizeClass
        from ubf_protein.protein_agent import ProteinAgent
        
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        
        config = AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=30,
            initial_frequency_range=(3.0, 15.0),
            initial_coherence_range=(0.2, 1.0),
            stuck_detection_window=10,
            stuck_detection_threshold=5.0,
            memory_significance_threshold=0.3,
            max_memories_per_agent=50,
            convergence_energy_threshold=10.0,
            convergence_rmsd_threshold=2.0,
            max_iterations=1000,
            checkpoint_interval=100
        )
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
            adaptive_config=config
        )
        
        # Fill memory
        for _ in range(100):
            agent.explore_step()
        
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_bytes = sum(stat.size_diff for stat in stats)
        total_mb = total_bytes / (1024 * 1024)
        
        status = "✅ PASS" if total_mb < 50.0 else "❌ FAIL"
        self.log(f"Memory footprint: {total_mb:.2f}MB {status}")
        
        return total_mb
    
    def _benchmark_throughput(self, num_agents: int = 10, iterations: int = 50) -> float:
        """Benchmark multi-agent throughput."""
        self.log(f"\n--- Multi-Agent Throughput (target: 5000 conf/sec) ---")
        
        from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
        
        coordinator = MultiAgentCoordinator("ACDEFGHIKLMNPQRSTVWYACDEFGHIKL")
        coordinator.initialize_agents(num_agents, "balanced")
        
        start = time.perf_counter()
        results = coordinator.run_parallel_exploration(iterations)
        elapsed = time.perf_counter() - start
        
        total_conf = results.total_conformations_explored
        throughput = total_conf / elapsed
        
        status = "✅ PASS" if throughput >= 5000 else "❌ FAIL"
        self.log(f"Throughput: {throughput:.0f} conf/sec {status}")
        
        return throughput
    
    # -------------------------------------------------------------------------
    # Prediction Benchmarks
    # -------------------------------------------------------------------------
    
    def benchmark_predictions(self) -> List[PredictionBenchmarkResult]:
        """Run prediction benchmarks on test proteins."""
        self.log("\n" + "=" * 60)
        self.log("PREDICTION BENCHMARKS")
        self.log("=" * 60)
        
        results = []
        
        for protein in self.test_proteins:
            result = self._run_single_prediction(protein)
            results.append(result)
            self.results.prediction_results.append(result)
        
        # Calculate summary
        self.results.prediction_summary = self._calculate_prediction_summary(results)
        self.results.total_proteins_tested = len(results)
        
        return results
    
    def _run_single_prediction(self, protein: Dict[str, Any]) -> PredictionBenchmarkResult:
        """Run a single prediction benchmark."""
        pdb_id = protein["pdb"]
        name = protein.get("name", pdb_id)
        fold = protein.get("fold", "unknown")
        
        self.log(f"\n--- Testing {pdb_id}: {name} ---")
        
        result = PredictionBenchmarkResult(
            pdb_id=pdb_id,
            protein_name=name,
            sequence="",
            sequence_length=protein.get("residues", 0),
            fold_type=fold,
        )
        
        try:
            # Get sequence from PDB
            pdb_path = self._get_pdb_path(pdb_id)
            if not pdb_path:
                raise ValueError(f"Could not find/download PDB for {pdb_id}")
            
            sequence = load_sequence_from_pdb(pdb_path)
            result.sequence = sequence
            result.sequence_length = len(sequence)
            
            # Configure prediction based on mode and protein size
            seq_len = len(sequence)
            category = protein.get("category", "medium")
            
            if self.mode == "quick":
                settings = get_quick_test_settings(seq_len)
                enable_refinement = False
                enable_hierarchical = False
            else:
                settings = get_optimal_settings(seq_len)
                enable_refinement = True
                # Enable hierarchical folding for medium+ proteins (50+ residues)
                enable_hierarchical = (seq_len >= 50)
            
            # Scale iterations based on protein size for better coverage
            base_iterations = settings['iterations']
            if category == "mini":
                iterations = max(100, base_iterations // 2)
            elif category == "small":
                iterations = base_iterations
            elif category == "medium":
                iterations = int(base_iterations * 1.5)
            else:  # large
                iterations = int(base_iterations * 2.0)
            
            config = PredictionConfig(
                sequence=sequence,
                native_pdb=pdb_id,
                pdb_file_path=pdb_path,
                agents=settings['agents'],
                iterations=iterations,
                qcpp_config="default",
                enable_refinement=enable_refinement,
                enable_mediators=False,
                enable_hierarchical_folding=enable_hierarchical,
            )
            
            result.config = {
                'agents': config.agents,
                'iterations': config.iterations,
                'qcpp_config': config.qcpp_config,
                'enable_refinement': config.enable_refinement,
                'enable_mediators': config.enable_mediators,
                'enable_hierarchical_folding': config.enable_hierarchical_folding,
                'diversity': config.diversity,
                'category': category,
            }
            
            # Run prediction
            runner = PredictionRunner(config)
            
            start_time = time.perf_counter()
            pred_results = runner.run()
            total_time = time.perf_counter() - start_time
            
            # Extract all metrics
            result.success = True
            result.total_time_seconds = total_time
            result.exploration_time_seconds = pred_results.exploration_time_seconds
            
            # Energy metrics
            result.final_energy = pred_results.best_energy
            result.initial_energy = pred_results.initial_energy
            if result.initial_energy and result.final_energy:
                result.energy_improvement = result.initial_energy - result.final_energy
            
            # RMSD metrics
            result.best_rmsd = pred_results.best_rmsd
            result.folding_rmsd = pred_results.folding_rmsd
            result.gdt_ts_score = pred_results.gdt_ts_score
            result.tm_score = pred_results.tm_score
            result.quality_assessment = pred_results.assess_quality()
            
            # Exploration stats
            result.conformations_explored = pred_results.conformations_explored
            result.unique_structures = pred_results.unique_structures
            result.throughput_conf_per_sec = pred_results.throughput_conf_per_sec
            
            # Refinement
            result.refinement_applied = pred_results.refinement_applied
            result.refinement_initial_rmsd = pred_results.refinement_initial_rmsd
            result.refinement_final_rmsd = pred_results.refinement_final_rmsd
            result.refinement_improvement_percent = pred_results.refinement_improvement_percent
            result.refinement_time_seconds = pred_results.refinement_time_seconds
            
            # QCPP metrics
            result.qcpp_enabled = (config.qcpp_config != "none")
            result.qcpp_total_analyses = pred_results.qcpp_total_analyses
            result.qcpp_cache_hit_rate = pred_results.qcpp_cache_hit_rate
            result.qcpp_avg_time_ms = pred_results.qcpp_avg_time_ms
            result.qcp_score = pred_results.qcp_score
            result.qaap_alignment = pred_results.qaap_alignment
            result.resonance_40hz = pred_results.resonance_40hz
            
            # Geometric analysis
            result.geometric_analysis = pred_results.geometric_analysis
            
            # Mediator stats
            result.mediator_stats = pred_results.mediator_stats
            
            # Agent behavior
            result.final_aggressiveness = pred_results.final_aggressiveness
            result.final_consistency = pred_results.final_consistency
            
            # Coordinates
            if self.include_coords and pred_results.best_conformation_coords:
                result.best_coords_included = True
                result.best_coords = [list(c) for c in pred_results.best_conformation_coords]
            
            self.log(f"  ✅ Success: Energy={result.final_energy:.1f}, RMSD={result.best_rmsd or 'N/A'}, Time={total_time:.1f}s")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.log(f"  ❌ Failed: {e}")
        
        return result
    
    def _get_pdb_path(self, pdb_id: str) -> Optional[str]:
        """Get path to PDB file, downloading if necessary."""
        # Check pdb_cache
        cache_dir = Path("pdb_cache")
        
        # Try various naming conventions
        patterns = [
            cache_dir / f"{pdb_id.lower()}.pdb",
            cache_dir / f"{pdb_id.upper()}.pdb",
            cache_dir / f"pdb{pdb_id.lower()}.ent",
        ]
        
        for pattern in patterns:
            if pattern.exists():
                return str(pattern)
        
        # Try to download
        try:
            path = download_pdb(pdb_id.upper())
            return path
        except Exception:
            return None
    
    def _calculate_prediction_summary(self, results: List[PredictionBenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics for predictions with category breakdown."""
        successful = [r for r in results if r.success]
        
        if not successful:
            return {'total': len(results), 'successful': 0, 'failed': len(results)}
        
        # Collect metrics
        energies = [r.final_energy for r in successful if r.final_energy is not None]
        rmsds = [r.best_rmsd for r in successful if r.best_rmsd is not None]
        times = [r.total_time_seconds for r in successful]
        throughputs = [r.throughput_conf_per_sec for r in successful if r.throughput_conf_per_sec > 0]
        
        # Quality distribution
        quality_counts = {}
        for r in successful:
            q = r.quality_assessment or "unknown"
            quality_counts[q] = quality_counts.get(q, 0) + 1
        
        # Category-based breakdown (scientifically important)
        category_stats = {}
        for r in successful:
            cat = r.config.get('category', 'unknown') if r.config else 'unknown'
            if cat not in category_stats:
                category_stats[cat] = {'count': 0, 'rmsds': [], 'times': [], 'energies': []}
            category_stats[cat]['count'] += 1
            if r.best_rmsd is not None:
                category_stats[cat]['rmsds'].append(r.best_rmsd)
            if r.total_time_seconds:
                category_stats[cat]['times'].append(r.total_time_seconds)
            if r.final_energy is not None:
                category_stats[cat]['energies'].append(r.final_energy)
        
        # Calculate per-category averages
        for cat, stats in category_stats.items():
            stats['mean_rmsd'] = sum(stats['rmsds']) / len(stats['rmsds']) if stats['rmsds'] else None
            stats['mean_time'] = sum(stats['times']) / len(stats['times']) if stats['times'] else None
            stats['mean_energy'] = sum(stats['energies']) / len(stats['energies']) if stats['energies'] else None
            # Clean up raw lists for JSON
            del stats['rmsds']
            del stats['times']
            del stats['energies']
        
        return {
            'total': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'energy_stats': {
                'mean': sum(energies) / len(energies) if energies else None,
                'min': min(energies) if energies else None,
                'max': max(energies) if energies else None,
            },
            'rmsd_stats': {
                'mean': sum(rmsds) / len(rmsds) if rmsds else None,
                'min': min(rmsds) if rmsds else None,
                'max': max(rmsds) if rmsds else None,
                'below_2A': sum(1 for r in rmsds if r < 2.0) if rmsds else 0,
                'below_4A': sum(1 for r in rmsds if r < 4.0) if rmsds else 0,
                'below_6A': sum(1 for r in rmsds if r < 6.0) if rmsds else 0,
                'below_8A': sum(1 for r in rmsds if r < 8.0) if rmsds else 0,
            },
            'timing_stats': {
                'mean_seconds': sum(times) / len(times) if times else None,
                'min_seconds': min(times) if times else None,
                'max_seconds': max(times) if times else None,
                'total_seconds': sum(times),
            },
            'throughput_stats': {
                'mean_conf_per_sec': sum(throughputs) / len(throughputs) if throughputs else None,
            },
            'quality_distribution': quality_counts,
            'category_breakdown': category_stats,
        }
    
    # -------------------------------------------------------------------------
    # Screening Benchmarks
    # -------------------------------------------------------------------------
    
    def benchmark_screening(self) -> List[ScreeningBenchmarkResult]:
        """Run screening benchmarks on test sequences."""
        self.log("\n" + "=" * 60)
        self.log("AGGREGATION SCREENING BENCHMARKS")
        self.log("=" * 60)
        
        results = []
        
        # Configure screener based on mode
        if self.mode == "quick":
            config = ScreeningConfig.fast()
        elif self.mode == "comprehensive":
            config = ScreeningConfig.thorough()
        else:
            config = ScreeningConfig.balanced()
        
        screener = AggregationScreener(config)
        
        for seq_info in self.screening_sequences:
            result = self._run_single_screening(screener, seq_info, config)
            results.append(result)
            self.results.screening_results.append(result)
        
        # Calculate summary
        self.results.screening_summary = self._calculate_screening_summary(results)
        self.results.total_sequences_screened = len(results)
        
        return results
    
    def _run_single_screening(
        self,
        screener: AggregationScreener,
        seq_info: Dict[str, Any],
        config: ScreeningConfig
    ) -> ScreeningBenchmarkResult:
        """Run a single screening benchmark."""
        name = seq_info["name"]
        sequence = seq_info["sequence"]
        expected = seq_info.get("expected_risk")
        
        self.log(f"\n--- Screening: {name} ({len(sequence)} residues) ---")
        
        result = ScreeningBenchmarkResult(
            sequence_name=name,
            sequence=sequence,
            sequence_length=len(sequence),
            expected_risk=expected,
            config={
                'iterations': config.iterations,
                'agents': config.agents,
                'enable_qcpp': config.enable_qcpp,
            },
        )
        
        try:
            # Run screening
            metrics = screener.screen_sequence(
                sequence,
                include_conformations=self.include_conformations,
            )
            
            result.success = True
            
            # Risk assessment
            result.aggregation_score = metrics.aggregation_score
            result.risk_level = metrics.risk_level.value
            result.risk_factors = metrics.risk_factors
            result.passes_screening = metrics.passes_screening
            
            # Individual scores
            result.energy_score = metrics.energy_score
            result.structure_score = metrics.structure_score
            result.hydrophobic_score = metrics.hydrophobic_score
            result.convergence_score = metrics.convergence_score
            result.compactness_score = metrics.compactness_score
            
            # Raw metrics
            result.final_energy = metrics.final_energy
            result.secondary_structure_pct = metrics.secondary_structure_pct
            result.hydrophobic_clustering = metrics.hydrophobic_clustering
            result.radius_of_gyration = metrics.radius_of_gyration
            result.convergence_iterations = metrics.convergence_iterations
            
            # Conformations
            result.num_unique_conformations = metrics.num_unique_conformations
            if self.include_conformations and metrics.conformations:
                result.conformations = [c.to_dict() for c in metrics.conformations]
            
            # Timing
            result.screening_time_ms = metrics.screening_time_ms
            result.iterations_used = metrics.iterations_used
            
            # Validate prediction
            if expected:
                result.risk_prediction_correct = (result.risk_level == expected)
            
            status = "✅" if result.passes_screening else "⚠️"
            self.log(f"  {status} Risk: {result.risk_level}, Score: {result.aggregation_score:.2f}, Time: {result.screening_time_ms:.0f}ms")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.log(f"  ❌ Failed: {e}")
        
        return result
    
    def _calculate_screening_summary(self, results: List[ScreeningBenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics for screening."""
        successful = [r for r in results if r.success]
        
        if not successful:
            return {'total': len(results), 'successful': 0, 'failed': len(results)}
        
        # Risk distribution
        risk_counts = {}
        for r in successful:
            risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1
        
        # Timing
        times = [r.screening_time_ms for r in successful]
        
        # Prediction accuracy
        with_expected = [r for r in successful if r.expected_risk is not None]
        correct = sum(1 for r in with_expected if r.risk_prediction_correct)
        
        return {
            'total': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'risk_distribution': risk_counts,
            'passing_screening': sum(1 for r in successful if r.passes_screening),
            'timing_stats': {
                'mean_ms': sum(times) / len(times) if times else None,
                'min_ms': min(times) if times else None,
                'max_ms': max(times) if times else None,
                'total_ms': sum(times),
            },
            'prediction_accuracy': {
                'total_with_expected': len(with_expected),
                'correct': correct,
                'accuracy_pct': (correct / len(with_expected) * 100) if with_expected else None,
            },
        }
    
    # -------------------------------------------------------------------------
    # Main Run Methods
    # -------------------------------------------------------------------------
    
    def run_all(self) -> ComprehensiveBenchmarkResults:
        """Run all benchmarks."""
        start_time = time.perf_counter()
        
        self.log("\n" + "=" * 70)
        self.log("COMPREHENSIVE PROTEIN PREDICTION & SCREENING BENCHMARK")
        self.log(f"Mode: {self.mode.upper()}")
        self.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)
        
        # Performance benchmarks
        self.benchmark_performance()
        
        # Prediction benchmarks
        self.benchmark_predictions()
        
        # Screening benchmarks
        self.benchmark_screening()
        
        # Total runtime
        self.results.total_runtime_seconds = time.perf_counter() - start_time
        
        self._print_final_summary()
        
        return self.results
    
    def run_predictions_only(self) -> ComprehensiveBenchmarkResults:
        """Run only prediction benchmarks."""
        start_time = time.perf_counter()
        
        self.log("\n" + "=" * 70)
        self.log("PREDICTION BENCHMARK")
        self.log(f"Mode: {self.mode.upper()}")
        self.log("=" * 70)
        
        self.benchmark_predictions()
        self.results.total_runtime_seconds = time.perf_counter() - start_time
        
        return self.results
    
    def run_screening_only(self) -> ComprehensiveBenchmarkResults:
        """Run only screening benchmarks."""
        start_time = time.perf_counter()
        
        self.log("\n" + "=" * 70)
        self.log("SCREENING BENCHMARK")
        self.log(f"Mode: {self.mode.upper()}")
        self.log("=" * 70)
        
        self.benchmark_screening()
        self.results.total_runtime_seconds = time.perf_counter() - start_time
        
        return self.results
    
    def run_performance_only(self) -> ComprehensiveBenchmarkResults:
        """Run only performance benchmarks."""
        start_time = time.perf_counter()
        
        self.benchmark_performance()
        self.results.total_runtime_seconds = time.perf_counter() - start_time
        
        return self.results
    
    def _print_final_summary(self):
        """Print final benchmark summary."""
        self.log("\n" + "=" * 70)
        self.log("BENCHMARK COMPLETE - EmergentFolds Performance Profile")
        self.log("=" * 70)
        
        self.log(f"\nTotal Runtime: {self.results.total_runtime_seconds:.1f} seconds")
        
        # Performance summary
        perf = self.results.performance_metrics
        if perf.move_evaluation_latency_ms:
            self.log("\n📊 Performance Metrics:")
            self.log(f"  Move Latency: {perf.move_evaluation_latency_ms:.3f}ms (target: <2ms)")
            self.log(f"  Memory Retrieval: {perf.memory_retrieval_us:.3f}μs (target: <10μs)")
            self.log(f"  Agent Memory: {perf.agent_memory_mb:.2f}MB (target: <50MB)")
            self.log(f"  Throughput: {perf.throughput_conf_per_sec:.0f} conf/sec (target: 5000)")
        
        # Prediction summary with category breakdown
        pred = self.results.prediction_summary
        if pred:
            self.log(f"\n🧬 Prediction Results: {pred.get('successful', 0)}/{pred.get('total', 0)} successful")
            
            # Overall stats
            rmsd_stats = pred.get('rmsd_stats', {})
            if rmsd_stats.get('mean'):
                self.log(f"  Overall Mean RMSD: {rmsd_stats['mean']:.2f}Å")
                self.log(f"  Best RMSD: {rmsd_stats.get('min', 'N/A'):.2f}Å")
                self.log(f"  Proteins <4Å RMSD: {rmsd_stats.get('below_4A', 0)}")
                self.log(f"  Proteins <6Å RMSD: {rmsd_stats.get('below_6A', 0)}")
                self.log(f"  Proteins <8Å RMSD: {rmsd_stats.get('below_8A', 0)}")
            
            # Category breakdown (key for scientific presentation)
            cat_breakdown = pred.get('category_breakdown', {})
            if cat_breakdown:
                self.log("\n  📈 Performance by Protein Size:")
                for cat in ['mini', 'small', 'medium', 'large']:
                    if cat in cat_breakdown:
                        stats = cat_breakdown[cat]
                        rmsd = stats.get('mean_rmsd')
                        time_s = stats.get('mean_time')
                        if rmsd is not None:
                            self.log(f"    {cat.capitalize():8s}: RMSD={rmsd:.2f}Å, Time={time_s:.1f}s (n={stats['count']})")
            
            if pred.get('timing_stats', {}).get('mean_seconds'):
                self.log(f"\n  Mean Prediction Time: {pred['timing_stats']['mean_seconds']:.1f}s")
        
        # Screening summary
        screen = self.results.screening_summary
        if screen and screen.get('successful', 0) > 0:
            self.log(f"\n🔬 Screening Results: {screen.get('successful', 0)}/{screen.get('total', 0)} successful")
            if screen.get('timing_stats', {}).get('mean_ms'):
                self.log(f"  Mean Screening Time: {screen['timing_stats']['mean_ms']:.0f}ms")
            if screen.get('prediction_accuracy', {}).get('accuracy_pct'):
                self.log(f"  Risk Prediction Accuracy: {screen['prediction_accuracy']['accuracy_pct']:.0f}%")
            risk_dist = screen.get('risk_distribution', {})
            if risk_dist:
                self.log(f"  Risk Distribution: {risk_dist}")
        
        # Scientific context
        self.log("\n" + "-" * 70)
        self.log("Note: Physics-based methods (without ML) typically achieve:")
        self.log("  - Mini proteins (<25 res): 2-4Å RMSD")
        self.log("  - Small proteins (25-50 res): 4-7Å RMSD") 
        self.log("  - Medium proteins (50-80 res): 6-12Å RMSD")
        self.log("For comparison: AlphaFold achieves <2Å on most proteins using ML.")
        self.log("-" * 70)
            if screen.get('prediction_accuracy', {}).get('accuracy_pct'):
                self.log(f"  Prediction Accuracy: {screen['prediction_accuracy']['accuracy_pct']:.0f}%")
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{self.mode}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(
                self.results.to_dict(
                    include_coords=self.include_coords,
                    include_conformations=self.include_conformations
                ),
                f,
                indent=2,
                default=str  # Handle any non-serializable objects
            )
        
        self.log(f"\nResults saved to: {output_path}")
        return str(output_path)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Protein Prediction & Screening Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks (standard mode)
  python comprehensive_benchmark.py

  # Quick benchmark (fewer proteins, faster settings)
  python comprehensive_benchmark.py --quick

  # Comprehensive benchmark (more proteins, thorough settings)
  python comprehensive_benchmark.py --comprehensive

  # Only prediction benchmarks
  python comprehensive_benchmark.py --predictions-only

  # Only screening benchmarks
  python comprehensive_benchmark.py --screening-only

  # Only performance benchmarks
  python comprehensive_benchmark.py --performance-only

  # Custom output file
  python comprehensive_benchmark.py --output my_results.json

  # Include coordinates in output
  python comprehensive_benchmark.py --include-coords

  # Silent mode (no console output)
  python comprehensive_benchmark.py --quiet
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick', action='store_true',
                           help='Quick benchmark with fewer proteins and faster settings')
    mode_group.add_argument('--comprehensive', action='store_true',
                           help='Comprehensive benchmark with more proteins and thorough settings')
    
    # Benchmark selection
    bench_group = parser.add_mutually_exclusive_group()
    bench_group.add_argument('--predictions-only', action='store_true',
                            help='Run only prediction benchmarks')
    bench_group.add_argument('--screening-only', action='store_true',
                            help='Run only screening benchmarks')
    bench_group.add_argument('--performance-only', action='store_true',
                            help='Run only performance benchmarks')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON filename')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory (default: benchmark_results)')
    parser.add_argument('--include-coords', action='store_true',
                       help='Include 3D coordinates in JSON output')
    parser.add_argument('--include-conformations', action='store_true',
                       help='Include all conformations in screening output')
    
    # Other options
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.quick:
        mode = "quick"
    elif args.comprehensive:
        mode = "comprehensive"
    else:
        mode = "standard"
    
    # Create benchmark runner
    benchmark = ComprehensiveBenchmark(
        mode=mode,
        output_dir=args.output_dir,
        include_coords=args.include_coords,
        include_conformations=args.include_conformations,
        verbose=not args.quiet,
    )
    
    # Run selected benchmarks
    if args.predictions_only:
        benchmark.run_predictions_only()
    elif args.screening_only:
        benchmark.run_screening_only()
    elif args.performance_only:
        benchmark.run_performance_only()
    else:
        benchmark.run_all()
    
    # Save results
    output_file = benchmark.save_results(args.output)
    
    if not args.quiet:
        print(f"\n✅ Benchmark complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()
