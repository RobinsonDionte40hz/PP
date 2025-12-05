"""
PredictionRunner - Unified Protein Structure Prediction Module

This module extracts the core prediction logic from test_protein.py into a
reusable class that can be called from both CLI (test_protein.py) and
web interface (Celery tasks).

This is the SINGLE SOURCE OF TRUTH for protein structure prediction.
All prediction code paths should use this module.

Key Features:
- Full QCPP+UBF integration
- Quantum Refinement Engine (two-stage optimization)
- Real RMSD calculations with Kabsch alignment
- Geometric attractor analysis
- Mediator agent coordination
- Progress callbacks for real-time updates
- Comprehensive result reporting

Usage:
    from ubf_protein.prediction_runner import PredictionRunner, PredictionConfig
    
    config = PredictionConfig(
        sequence="ACDEFGH...",
        native_pdb="1UBQ",
        enable_refinement=True
    )
    
    runner = PredictionRunner(config)
    results = runner.run(progress_callback=my_callback)
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Callable, List, Tuple

# Setup logging
logger = logging.getLogger(__name__)

# Import core components
try:
    from .multi_agent_coordinator import MultiAgentCoordinator
    from .qcpp_integration import QCPPIntegrationAdapter
    from .rmsd_calculator import RMSDCalculator, NativeStructureLoader
    from .geometric_attractor import GeometricAttractorAnalyzer
    from .adaptive_config import create_config_for_sequence
except ImportError:
    # Handle direct execution
    from multi_agent_coordinator import MultiAgentCoordinator
    from qcpp_integration import QCPPIntegrationAdapter
    from rmsd_calculator import RMSDCalculator, NativeStructureLoader
    from geometric_attractor import GeometricAttractorAnalyzer
    from adaptive_config import create_config_for_sequence

# Import QCPP predictor
try:
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.protein_predictor import QuantumCoherenceProteinPredictor
    HAS_QCPP = True
except ImportError:
    HAS_QCPP = False
    logger.warning("QCPP predictor not available")


@dataclass
class PredictionConfig:
    """Configuration for protein structure prediction."""
    
    # Required
    sequence: str
    
    # Native structure (for RMSD validation)
    native_pdb: Optional[str] = None
    pdb_file_path: Optional[str] = None
    
    # Agent configuration (None = auto-configure based on sequence length)
    agents: Optional[int] = None
    iterations: Optional[int] = None
    diversity: str = "balanced"
    
    # QCPP configuration
    qcpp_config: str = "default"  # 'default', 'high_performance', 'high_accuracy', 'none'
    qcpp_frequency: int = 20  # Analyze every N iterations
    cache_size: int = 10000
    
    # Advanced features
    enable_refinement: bool = False
    enable_mediators: bool = False
    mediator_count: int = 2
    target_geometry: str = "none"  # 'none', 'octahedron', 'icosahedron', 'dodecahedron', etc.
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 50
    
    # Output
    output_dir: Optional[str] = None
    save_pdb: bool = True
    save_trajectory: bool = True
    
    def __post_init__(self):
        """Validate and auto-configure settings."""
        self.sequence = self.sequence.upper()
        
        # Auto-configure agents and iterations based on sequence length
        if self.agents is None or self.iterations is None:
            optimal = get_optimal_settings(len(self.sequence))
            if self.agents is None:
                self.agents = optimal['agents']
            if self.iterations is None:
                self.iterations = optimal['iterations']


@dataclass
class ProgressUpdate:
    """Progress update for callbacks."""
    iteration: int
    total_iterations: int
    progress_percentage: float
    current_energy: float
    current_rmsd: Optional[float]
    folding_rmsd: Optional[float]
    best_energy: float
    best_rmsd: Optional[float]
    conformations_explored: int
    aggressiveness: Optional[float] = None
    consistency: Optional[float] = None
    stage: str = "exploration"  # 'exploration', 'refinement', 'analysis'
    message: Optional[str] = None


@dataclass
class PredictionResults:
    """Complete results from a prediction run."""
    
    # Basic info
    prediction_id: str
    sequence: str
    sequence_length: int
    
    # Configuration used
    config: Dict[str, Any]
    
    # Exploration results
    best_energy: float
    best_rmsd: Optional[float]
    folding_rmsd: Optional[float]
    conformations_explored: int
    exploration_time_seconds: float
    throughput_conf_per_sec: float
    
    # Validation metrics (if native structure available)
    gdt_ts_score: Optional[float] = None
    tm_score: Optional[float] = None
    validation_quality: Optional[str] = None
    rmsd_calculation_method: str = "energy_estimate"
    
    # Quantum refinement (if enabled)
    refinement_applied: bool = False
    refinement_initial_rmsd: Optional[float] = None
    refinement_final_rmsd: Optional[float] = None
    refinement_improvement_percent: Optional[float] = None
    refinement_time_seconds: Optional[float] = None
    
    # QCPP metrics
    qcpp_total_analyses: int = 0
    qcpp_cache_hit_rate: float = 0.0
    qcpp_avg_time_ms: float = 0.0
    qaap_alignment: Optional[float] = None
    resonance_40hz: Optional[float] = None
    water_shielding: Optional[float] = None
    qcp_score: Optional[float] = None
    
    # Geometric analysis
    geometric_analysis: Optional[Dict[str, Any]] = None
    
    # Mediator statistics
    mediator_stats: Optional[Dict[str, Any]] = None
    
    # Best conformation data
    best_conformation_coords: Optional[List[Tuple[float, float, float]]] = None
    
    # Timing
    total_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Energy components
    energy_change: Optional[float] = None
    initial_energy: Optional[float] = None
    convergence_rate: Optional[float] = None
    
    # Agent metrics
    unique_structures: Optional[int] = None
    final_aggressiveness: Optional[float] = None
    final_consistency: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def assess_quality(self) -> str:
        """Assess prediction quality based on metrics."""
        if self.best_rmsd is None:
            return "unknown"
        
        if self.best_rmsd < 2.0 and (self.gdt_ts_score or 0) > 80:
            return "excellent"
        elif self.best_rmsd < 4.0 and (self.gdt_ts_score or 0) > 65:
            return "good"
        elif self.best_rmsd < 6.0 and (self.gdt_ts_score or 0) > 50:
            return "acceptable"
        else:
            return "poor"


def get_optimal_settings(sequence_length: int) -> Dict[str, Any]:
    """
    Get optimal agent count and iterations based on protein size.
    
    These settings are tuned from extensive testing to balance
    exploration quality with runtime.
    """
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


def get_quick_test_settings(sequence_length: int) -> Dict[str, Any]:
    """Get fast test settings for quick validation (10x fewer iterations)."""
    if sequence_length < 50:
        return {"agents": 10, "iterations": 50, "category": "small"}
    elif sequence_length < 100:
        return {"agents": 10, "iterations": 40, "category": "medium"}
    elif sequence_length < 150:
        return {"agents": 15, "iterations": 40, "category": "large"}
    else:
        return {"agents": 20, "iterations": 50, "category": "very_large"}


# Progress callback type
ProgressCallback = Callable[[ProgressUpdate], None]


class PredictionRunner:
    """
    Unified protein structure prediction runner.
    
    This class encapsulates the complete prediction workflow including:
    - QCPP integration and quantum physics calculations
    - Multi-agent conformational exploration
    - Real RMSD validation against native structures
    - Quantum refinement (two-stage optimization)
    - Geometric attractor analysis
    - Mediator agent coordination
    
    Example:
        config = PredictionConfig(
            sequence="MQIFVKTLTGKTITLEVEPS...",
            native_pdb="1UBQ",
            enable_refinement=True
        )
        
        runner = PredictionRunner(config)
        
        def on_progress(update: ProgressUpdate):
            print(f"Progress: {update.progress_percentage:.1f}%")
        
        results = runner.run(progress_callback=on_progress)
        print(f"Best RMSD: {results.best_rmsd:.2f} Å")
    """
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize prediction runner.
        
        Args:
            config: PredictionConfig with all settings
        """
        self.config = config
        self.prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Will be initialized in run()
        self.qcpp_predictor = None
        self.qcpp_adapter = None
        self.coordinator = None
        self.native_structure = None
        self.rmsd_calculator = None
        
    def run(self, progress_callback: Optional[ProgressCallback] = None) -> PredictionResults:
        """
        Run complete protein structure prediction.
        
        Args:
            progress_callback: Optional callback for progress updates.
                              Called with ProgressUpdate objects during execution.
        
        Returns:
            PredictionResults with all metrics and data
        """
        start_time = time.time()
        
        logger.info(f"Starting prediction {self.prediction_id}")
        logger.info(f"Sequence length: {len(self.config.sequence)}")
        logger.info(f"Agents: {self.config.agents}, Iterations: {self.config.iterations}")
        
        # Step 1: Initialize QCPP
        self._emit_progress(progress_callback, 0, 0, "initialization", "Initializing QCPP predictor...")
        self._init_qcpp()
        
        # Step 2: Load native structure (if available)
        self._emit_progress(progress_callback, 0, 0, "initialization", "Loading native structure...")
        self._load_native_structure()
        
        # Step 3: Create coordinator
        self._emit_progress(progress_callback, 0, 0, "initialization", "Creating multi-agent coordinator...")
        self._init_coordinator()
        
        # Step 4: Run exploration with progress tracking
        exploration_results, exploration_time = self._run_exploration(progress_callback)
        
        # Step 5: Apply quantum refinement (if enabled)
        refinement_result = None
        if self.config.enable_refinement and self.native_structure:
            self._emit_progress(progress_callback, 100, self.config.iterations, "refinement", 
                              "Applying quantum refinement...")
            refinement_result = self._run_refinement(exploration_results)
        
        # Step 6: Calculate final metrics
        self._emit_progress(progress_callback, 100, self.config.iterations, "analysis", 
                          "Calculating structural metrics...")
        
        # Get best conformation
        best_conf, best_energy, best_rmsd = self.coordinator.get_best_conformation()
        
        # Use refined results if available
        if refinement_result:
            final_rmsd = refinement_result.final_rmsd
            final_energy = refinement_result.energy
        else:
            final_rmsd = best_rmsd
            final_energy = best_energy
        
        # Calculate real RMSD if native structure available
        rmsd_result = None
        gdt_ts = None
        tm_score = None
        rmsd_method = "energy_estimate"
        
        if self.native_structure and best_conf:
            rmsd_result = self._calculate_rmsd(best_conf)
            if rmsd_result:
                final_rmsd = rmsd_result.rmsd
                gdt_ts = rmsd_result.gdt_ts
                tm_score = rmsd_result.tm_score
                rmsd_method = "kabsch"
        
        # Step 7: Geometric analysis
        self._emit_progress(progress_callback, 100, self.config.iterations, "analysis", 
                          "Running geometric attractor analysis...")
        geometric_results = self._run_geometric_analysis(best_conf)
        
        # Step 8: Get QCPP cache stats
        cache_stats = self.qcpp_adapter.get_cache_stats() if self.qcpp_adapter else {}
        
        # Step 9: Get mediator statistics
        mediator_stats = None
        if self.config.enable_mediators:
            try:
                mediator_stats = self.coordinator.get_mediator_statistics()
            except Exception as e:
                logger.warning(f"Could not get mediator stats: {e}")
        
        # Calculate additional metrics
        total_conformations = self.config.agents * self.config.iterations
        throughput = total_conformations / exploration_time if exploration_time > 0 else 0
        
        # Energy metrics
        energy_change = None
        initial_energy = None
        convergence_rate = None
        unique_structures = None
        
        if exploration_results.agent_metrics:
            all_best_energies = [m.best_energy_found for m in exploration_results.agent_metrics]
            if all_best_energies:
                initial_energy = max(all_best_energies)
                energy_change = final_energy - initial_energy
                if initial_energy != 0:
                    convergence_rate = abs((initial_energy - final_energy) / initial_energy * 100)
            unique_structures = sum(m.conformations_explored for m in exploration_results.agent_metrics)
        
        # Determine validation quality
        validation_quality = None
        if final_rmsd is not None and final_rmsd != float('inf'):
            if final_rmsd < 2.0:
                validation_quality = "excellent"
            elif final_rmsd < 4.0:
                validation_quality = "good"
            elif final_rmsd < 6.0:
                validation_quality = "acceptable"
            else:
                validation_quality = "poor"
        
        total_time = time.time() - start_time
        
        # Build results
        results = PredictionResults(
            prediction_id=self.prediction_id,
            sequence=self.config.sequence,
            sequence_length=len(self.config.sequence),
            config=asdict(self.config),
            
            # Exploration
            best_energy=final_energy,
            best_rmsd=final_rmsd if final_rmsd != float('inf') else None,
            folding_rmsd=getattr(exploration_results, 'folding_rmsd', None),
            conformations_explored=exploration_results.total_conformations_explored,
            exploration_time_seconds=exploration_time,
            throughput_conf_per_sec=throughput,
            
            # Validation
            gdt_ts_score=gdt_ts,
            tm_score=tm_score,
            validation_quality=validation_quality,
            rmsd_calculation_method=rmsd_method,
            
            # Refinement
            refinement_applied=refinement_result is not None,
            refinement_initial_rmsd=refinement_result.initial_rmsd if refinement_result else None,
            refinement_final_rmsd=refinement_result.final_rmsd if refinement_result else None,
            refinement_improvement_percent=(
                (refinement_result.rmsd_improvement / refinement_result.initial_rmsd * 100)
                if refinement_result and refinement_result.initial_rmsd > 0 else None
            ),
            refinement_time_seconds=refinement_result.refinement_time_seconds if refinement_result else None,
            
            # QCPP
            qcpp_total_analyses=cache_stats.get('total_analyses', 0),
            qcpp_cache_hit_rate=cache_stats.get('cache_hit_rate', 0.0),
            qcpp_avg_time_ms=cache_stats.get('avg_calculation_time_ms', 0.0),
            
            # Geometric
            geometric_analysis=geometric_results,
            
            # Mediator
            mediator_stats=mediator_stats,
            
            # Best conformation
            best_conformation_coords=list(best_conf.atom_coordinates) if best_conf else None,
            
            # Timing
            total_time_seconds=total_time,
            
            # Energy metrics
            energy_change=energy_change,
            initial_energy=initial_energy,
            convergence_rate=convergence_rate,
            unique_structures=unique_structures,
        )
        
        # Save outputs if configured
        if self.config.output_dir:
            self._save_outputs(results, best_conf)
        
        self._emit_progress(progress_callback, 100, self.config.iterations, "complete", 
                          f"Prediction complete! RMSD: {final_rmsd:.2f}Å" if final_rmsd else "Prediction complete!")
        
        logger.info(f"Prediction {self.prediction_id} complete in {total_time:.1f}s")
        
        return results
    
    def _emit_progress(self, callback: Optional[ProgressCallback], 
                       progress_pct: float, iteration: int, 
                       stage: str, message: str):
        """Emit progress update if callback provided."""
        if callback:
            try:
                update = ProgressUpdate(
                    iteration=iteration,
                    total_iterations=self.config.iterations,
                    progress_percentage=progress_pct,
                    current_energy=0.0,
                    current_rmsd=None,
                    folding_rmsd=None,
                    best_energy=0.0,
                    best_rmsd=None,
                    conformations_explored=0,
                    stage=stage,
                    message=message
                )
                callback(update)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def _init_qcpp(self):
        """Initialize QCPP predictor and adapter."""
        if self.config.qcpp_config == 'none' or not HAS_QCPP:
            logger.info("QCPP disabled or not available")
            return
        
        try:
            self.qcpp_predictor = QuantumCoherenceProteinPredictor()
            self.qcpp_adapter = QCPPIntegrationAdapter(
                self.qcpp_predictor,
                self.config.cache_size,
                target_geometry=self.config.target_geometry
            )
            logger.info(f"QCPP initialized (cache={self.config.cache_size})")
        except Exception as e:
            logger.warning(f"Failed to initialize QCPP: {e}")
    
    def _load_native_structure(self):
        """Load native structure for RMSD validation."""
        if not self.config.native_pdb and not self.config.pdb_file_path:
            logger.info("No native structure specified")
            return
        
        try:
            loader = NativeStructureLoader(cache_dir="./pdb_cache")
            
            if self.config.pdb_file_path and Path(self.config.pdb_file_path).exists():
                self.native_structure = loader.load_from_file(
                    self.config.pdb_file_path, ca_only=True
                )
                logger.info(f"Loaded native from file: {self.config.pdb_file_path}")
            elif self.config.native_pdb:
                self.native_structure = loader.load_from_pdb_id(
                    self.config.native_pdb, ca_only=True
                )
                logger.info(f"Loaded native from PDB: {self.config.native_pdb}")
            
            self.rmsd_calculator = RMSDCalculator(align_structures=True)
            
        except Exception as e:
            logger.warning(f"Failed to load native structure: {e}")
    
    def _init_coordinator(self):
        """Initialize multi-agent coordinator."""
        # Setup checkpoint directory
        checkpoint_dir = self.config.checkpoint_dir
        if not checkpoint_dir and self.config.output_dir:
            checkpoint_dir = str(Path(self.config.output_dir) / "checkpoints")
        elif not checkpoint_dir and self.config.enable_checkpointing:
            # Use default checkpoint directory
            checkpoint_dir = str(Path.cwd() / "checkpoints")
        
        self.coordinator = MultiAgentCoordinator(
            protein_sequence=self.config.sequence,
            qcpp_integration=self.qcpp_adapter,
            qcpp_analysis_frequency=self.config.qcpp_frequency,
            target_geometry=self.config.target_geometry,
            enable_mediators=self.config.enable_mediators,
            mediator_count=self.config.mediator_count,
            enable_checkpointing=self.config.enable_checkpointing,
            checkpoint_dir=checkpoint_dir,
        )
        
        # Initialize agents
        self.coordinator.initialize_agents(
            count=self.config.agents,
            diversity_profile=self.config.diversity,
            native_structure=self.native_structure
        )
        logger.info(f"Initialized {self.config.agents} agents ({self.config.diversity} diversity)")
        
        # Initialize mediators if enabled
        if self.config.enable_mediators:
            try:
                self.coordinator.initialize_mediators()
                logger.info(f"Initialized {self.config.mediator_count} mediator agents")
            except Exception as e:
                logger.warning(f"Failed to initialize mediators: {e}")
    
    def _run_exploration(self, callback: Optional[ProgressCallback]) -> Tuple[Any, float]:
        """Run multi-agent exploration with progress tracking."""
        start_time = time.time()
        
        chunk_size = 50
        total_chunks = (self.config.iterations + chunk_size - 1) // chunk_size
        
        results = None
        
        for chunk_idx in range(total_chunks):
            chunk_iterations = min(chunk_size, self.config.iterations - (chunk_idx * chunk_size))
            
            # Run chunk
            results = self.coordinator.run_parallel_exploration(chunk_iterations)
            
            # Calculate progress
            completed = min((chunk_idx + 1) * chunk_size, self.config.iterations)
            progress = (completed / self.config.iterations) * 100
            
            # Get current metrics
            best_conf, best_energy, best_rmsd = self.coordinator.get_best_conformation()
            folding_rmsd = getattr(results, 'folding_rmsd', None)
            
            # Get agent states for aggressiveness/consistency
            avg_aggressiveness = None
            avg_consistency = None
            try:
                agents = self.coordinator.get_agents()
                if agents:
                    avg_aggressiveness = sum(
                        a.get_consciousness_state().get_frequency() for a in agents
                    ) / len(agents)
                    avg_consistency = sum(
                        a.get_consciousness_state().get_coherence() for a in agents
                    ) / len(agents)
            except Exception:
                pass
            
            # Emit progress
            if callback:
                try:
                    update = ProgressUpdate(
                        iteration=completed,
                        total_iterations=self.config.iterations,
                        progress_percentage=progress,
                        current_energy=best_energy,
                        current_rmsd=best_rmsd if best_rmsd != float('inf') else None,
                        folding_rmsd=folding_rmsd,
                        best_energy=results.best_energy,
                        best_rmsd=results.best_rmsd if results.best_rmsd != float('inf') else None,
                        conformations_explored=results.total_conformations_explored,
                        aggressiveness=avg_aggressiveness,
                        consistency=avg_consistency,
                        stage="exploration",
                        message=f"Iteration {completed}/{self.config.iterations}"
                    )
                    callback(update)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
            
            logger.debug(f"Progress: {completed}/{self.config.iterations} ({progress:.1f}%)")
        
        exploration_time = time.time() - start_time
        return results, exploration_time
    
    def _run_refinement(self, exploration_results: Any) -> Optional[Any]:
        """Run quantum refinement on best conformation."""
        try:
            from .quantum_refinement_engine import QuantumRefinementEngine
            
            best_conf, _, _ = self.coordinator.get_best_conformation()
            if not best_conf:
                logger.warning("No best conformation for refinement")
                return None
            
            engine = QuantumRefinementEngine(
                protein_sequence=self.config.sequence,
                qcpp_integration=self.qcpp_adapter,
                target_geometry=self.config.target_geometry
            )
            
            result = engine.refine_structure_quantum(
                best_conf,
                native_structure=self.native_structure,
                max_iterations=150
            )
            
            logger.info(f"Refinement: RMSD {result.initial_rmsd:.2f}Å -> {result.final_rmsd:.2f}Å "
                       f"({result.rmsd_improvement / result.initial_rmsd * 100:.1f}% improvement)")
            
            return result
            
        except Exception as e:
            logger.warning(f"Quantum refinement failed: {e}")
            return None
    
    def _calculate_rmsd(self, conformation: Any) -> Optional[Any]:
        """Calculate real RMSD against native structure."""
        if not self.native_structure or not self.rmsd_calculator:
            return None
        
        try:
            predicted_coords = conformation.atom_coordinates
            native_coords = self.native_structure.ca_coords
            
            if len(predicted_coords) != len(native_coords):
                logger.warning(f"Length mismatch: predicted={len(predicted_coords)}, "
                             f"native={len(native_coords)}")
                return None
            
            result = self.rmsd_calculator.calculate_rmsd(
                predicted_coords=predicted_coords,
                native_coords=native_coords,
                calculate_metrics=True
            )
            
            logger.info(f"Real RMSD: {result.rmsd:.2f}Å, GDT-TS: {result.gdt_ts:.1f}, "
                       f"TM-score: {result.tm_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.warning(f"RMSD calculation failed: {e}")
            return None
    
    def _run_geometric_analysis(self, conformation: Any) -> Optional[Dict[str, Any]]:
        """Run geometric attractor analysis on best conformation."""
        if not conformation:
            return None
        
        try:
            analyzer = GeometricAttractorAnalyzer(
                cache_size=1000,
                cache_ttl=3600.0,
                phi_tolerance=0.05,
                neighbor_window=10
            )
            
            conformation_data = {
                'coordinates': conformation.atom_coordinates
            }
            
            result = analyzer.analyze_conformation(conformation_data, sequence=self.config.sequence)
            
            logger.info(f"Geometric analysis: φ={result.golden_ratio_percentage:.1f}%, "
                       f"icosahedron={result.icosahedron_similarity:.3f}")
            
            return result.to_dict()
            
        except Exception as e:
            logger.warning(f"Geometric analysis failed: {e}")
            return None
    
    def _save_outputs(self, results: PredictionResults, conformation: Any):
        """Save prediction outputs to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results JSON
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved results to {results_file}")
        
        # Save PDB structure
        if self.config.save_pdb and conformation:
            pdb_file = output_dir / "structure.pdb"
            self._save_pdb(conformation, pdb_file)
        
        # Save trajectory
        if self.config.save_trajectory:
            trajectory_file = output_dir / "trajectory.json"
            self._save_trajectory(trajectory_file)
    
    def _save_pdb(self, conformation: Any, output_file: Path):
        """Save conformation as PDB file."""
        try:
            # Map 1-letter to 3-letter amino acid codes
            AA1_TO_AA3 = {
                'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
            }
            
            with open(output_file, 'w') as f:
                f.write("HEADER    PROTEIN STRUCTURE PREDICTION\n")
                f.write(f"TITLE     UBF-QCPP PREDICTION {self.prediction_id}\n")
                f.write(f"REMARK    SEQUENCE LENGTH: {len(self.config.sequence)}\n")
                f.write(f"REMARK    METHOD: UBF with QCPP integration\n")
                f.write(f"REMARK    DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                
                for i, (aa_letter, coord) in enumerate(zip(self.config.sequence, 
                                                           conformation.atom_coordinates), 1):
                    x, y, z = coord
                    aa_3letter = AA1_TO_AA3.get(aa_letter, 'UNK')
                    f.write(f"ATOM  {i:5d}  CA  {aa_3letter:3s} A{i:4d}    "
                           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                
                f.write("END\n")
            
            logger.info(f"Saved PDB to {output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save PDB: {e}")
    
    def _save_trajectory(self, output_file: Path):
        """Save exploration trajectory."""
        try:
            trajectory_data = []
            agents = self.coordinator.get_agents()
            
            for agent in agents:
                agent_id = agent.get_agent_id()
                snapshots = agent.get_trajectory_snapshots()
                
                for snapshot in snapshots:
                    trajectory_data.append({
                        'iteration': snapshot.iteration,
                        'agent_id': agent_id,
                        'energy': snapshot.conformation.energy,
                        'rmsd': snapshot.conformation.rmsd_to_native,
                        'aggressiveness': snapshot.consciousness_state.frequency,
                        'consistency': snapshot.consciousness_state.coherence,
                        'timestamp': snapshot.timestamp
                    })
            
            trajectory_data.sort(key=lambda x: (x['iteration'], x['agent_id']))
            
            with open(output_file, 'w') as f:
                json.dump({
                    'prediction_id': self.prediction_id,
                    'total_points': len(trajectory_data),
                    'agent_count': len(agents),
                    'trajectory': trajectory_data
                }, f, indent=2)
            
            logger.info(f"Saved trajectory to {output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")


# Convenience function for simple usage
def run_prediction(
    sequence: str,
    native_pdb: Optional[str] = None,
    enable_refinement: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    **kwargs
) -> PredictionResults:
    """
    Convenience function to run a prediction with minimal configuration.
    
    Args:
        sequence: Amino acid sequence
        native_pdb: Optional PDB ID for validation
        enable_refinement: Enable quantum refinement
        progress_callback: Optional progress callback
        **kwargs: Additional PredictionConfig parameters
    
    Returns:
        PredictionResults
    """
    config = PredictionConfig(
        sequence=sequence,
        native_pdb=native_pdb,
        enable_refinement=enable_refinement,
        **kwargs
    )
    
    runner = PredictionRunner(config)
    return runner.run(progress_callback=progress_callback)
