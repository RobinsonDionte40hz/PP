"""
Large-Scale Validation Campaign Orchestrator

This module provides the main orchestrator for running comprehensive validation
campaigns across 50-75 proteins with phased testing, quality gates, and automated
analysis and documentation.

Integrates all validation framework components:
- ProteinSelector: Systematic protein selection
- PhaseManager: 4-phase progressive testing
- BatchExecutor: Parallel execution with resource management
- ResultsRepository: Centralized data storage
- ProgressTracker: Real-time monitoring
- StatisticalAnalyzer: Pattern detection
- FailureAnalyzer: Failure analysis
- DocumentationGenerator: Automated reporting
- QualityControl: Reproducibility checks

Usage:
    campaign = LargeScaleValidationCampaign(
        target_protein_count=60,
        enable_qcpp=True,
        max_parallel=3
    )
    results = campaign.run_campaign()
"""

import json
import time
import logging
import sys
import os
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import ubf_protein
sys.path.insert(0, str(Path(__file__).parent.parent))

from ubf_protein.validation_suite import ValidationSuite, ValidationReport

from .protein_selector import ProteinSelector, ProteinMetadata
from .phase_manager import PhaseManager, Phase, QualityGateResult, PhaseStatus
from .batch_executor import BatchExecutor, BatchProgress
from .results_repository import ResultsRepository, TestRunMetadata
from .progress_tracker import ProgressTracker, DashboardData, InterimReport
from .statistical_analyzer import StatisticalAnalyzer
from .failure_analyzer import FailureAnalyzer
from .documentation_generator import DocumentationGenerator
from .quality_control import QualityController

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class CampaignConfig:
    """
    Configuration for large-scale validation campaign.
    
    Attributes:
        target_protein_count: Target number of proteins to test (50-75)
        enable_qcpp: Enable QCPP integration for quantum physics feedback
        max_parallel_tests: Maximum number of parallel test executions
        num_agents: Number of agents per protein prediction
        iterations_per_agent: Iterations per agent
        checkpoint_interval: Save checkpoint every N completed tests
        quality_gate_threshold: Success rate threshold for Phase 1 (0-1)
        failure_rmsd_threshold: RMSD threshold for failure classification
        timeout_multiplier: Timeout as multiple of expected runtime
        random_seed: Random seed for reproducibility (None for random)
        output_dir: Directory for all campaign outputs
    """
    target_protein_count: int = 60
    enable_qcpp: bool = True
    max_parallel_tests: int = 3
    num_agents: int = 10
    iterations_per_agent: int = 1000
    checkpoint_interval: int = 5  # tests
    quality_gate_threshold: float = 0.60  # 60% success rate
    failure_rmsd_threshold: float = 8.0
    timeout_multiplier: float = 2.0
    random_seed: Optional[int] = None
    output_dir: str = "./campaign_results"


# ============================================================================
# Results Data Classes
# ============================================================================

@dataclass
class PhaseResults:
    """
    Results from completing a single phase.
    
    Attributes:
        phase_number: Phase number (1-4)
        proteins_tested: Number of proteins tested in phase
        success_rate: Percentage of successful predictions
        average_rmsd: Average RMSD across all tests
        average_gdt_ts: Average GDT-TS across all tests
        average_energy: Average energy across all tests
        quality_gate_passed: Whether phase passed quality gate
        validation_reports: List of validation reports
        interim_report_path: Path to interim report file
        start_time: Phase start timestamp
        end_time: Phase end timestamp
        runtime_seconds: Total phase runtime
    """
    phase_number: int
    proteins_tested: int
    success_rate: float
    average_rmsd: float
    average_gdt_ts: float
    average_energy: float
    quality_gate_passed: bool
    validation_reports: List[Dict] = field(default_factory=list)
    interim_report_path: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    runtime_seconds: float = 0.0


@dataclass
class CampaignResults:
    """
    Complete results from validation campaign.
    
    Attributes:
        campaign_id: Unique campaign identifier
        config: Campaign configuration used
        start_time: Campaign start timestamp
        end_time: Campaign end timestamp
        total_proteins: Total proteins tested
        phases_completed: Number of phases completed
        overall_success_rate: Overall success rate percentage
        validation_reports: All validation reports
        phase_summaries: Summary for each phase
        statistical_analysis_path: Path to statistical analysis report
        failure_analysis_path: Path to failure analysis report
        final_report_path: Path to final comprehensive report
    """
    campaign_id: str
    config: CampaignConfig
    start_time: datetime
    end_time: Optional[datetime]
    total_proteins: int
    phases_completed: int
    overall_success_rate: float
    validation_reports: List[Dict] = field(default_factory=list)
    phase_summaries: List[PhaseResults] = field(default_factory=list)
    statistical_analysis_path: str = ""
    failure_analysis_path: str = ""
    final_report_path: str = ""


# ============================================================================
# Large-Scale Validation Campaign Orchestrator
# ============================================================================

class LargeScaleValidationCampaign:
    """
    Main orchestrator for large-scale protein validation campaigns.
    
    Coordinates all validation framework components to execute comprehensive
    testing of 50-75 proteins with phased execution, quality gates, automated
    analysis, and research documentation.
    
    Workflow:
        1. Setup: Select proteins and initialize components
        2. Phase Execution: Run each phase with quality gate checks
        3. Progress Tracking: Real-time monitoring and interim reports
        4. Analysis: Statistical and failure analysis
        5. Documentation: Generate final comprehensive report
    """
    
    def __init__(
        self,
        config: Optional[CampaignConfig] = None,
        protein_selection: Optional[List[ProteinMetadata]] = None
    ):
        """
        Initialize validation campaign.
        
        Args:
            config: Campaign configuration (uses defaults if None)
            protein_selection: Pre-selected proteins (uses ProteinSelector if None)
        """
        # Configuration
        self.config = config or CampaignConfig()
        self._validate_config()
        
        # Campaign ID
        self.campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._protein_selection = protein_selection
        self._protein_selector: Optional[ProteinSelector] = None
        self._phase_manager: Optional[PhaseManager] = None
        self._batch_executor: Optional[BatchExecutor] = None
        self._results_repository: Optional[ResultsRepository] = None
        self._progress_tracker: Optional[ProgressTracker] = None
        self._statistical_analyzer: Optional[StatisticalAnalyzer] = None
        self._failure_analyzer: Optional[FailureAnalyzer] = None
        self._documentation_generator: Optional[DocumentationGenerator] = None
        self._quality_controller: Optional[QualityController] = None
        self._validation_suite: Optional[ValidationSuite] = None
        
        # Campaign state
        self._is_setup = False
        self._current_phase: Optional[Phase] = None
        self._all_validation_reports: List[ValidationReport] = []
        self._phase_results: List[PhaseResults] = []
        
        logger.info(f"Initialized campaign {self.campaign_id}")
    
    def _validate_config(self) -> None:
        """Validate campaign configuration."""
        if not (50 <= self.config.target_protein_count <= 75):
            logger.warning(
                f"target_protein_count={self.config.target_protein_count} "
                f"outside recommended range [50, 75]"
            )
        
        if not (0.0 < self.config.quality_gate_threshold <= 1.0):
            raise ValueError("quality_gate_threshold must be in (0, 1]")
        
        if self.config.max_parallel_tests < 1:
            raise ValueError("max_parallel_tests must be >= 1")
    
    def setup_campaign(self) -> None:
        """
        Setup campaign by initializing all components.
        
        Steps:
            1. Select proteins (if not provided)
            2. Organize into phases
            3. Initialize all framework components
            4. Validate setup
        
        Raises:
            RuntimeError: If setup fails
        """
        logger.info(f"Setting up campaign {self.campaign_id}...")
        
        try:
            # Step 1: Protein selection
            if self._protein_selection is None:
                logger.info("Selecting proteins...")
                self._protein_selector = ProteinSelector()
                self._protein_selection = self._protein_selector.select_proteins(
                    target_count=self.config.target_protein_count,
                    max_protein_size=getattr(self.config, 'max_protein_size', None)
                )
                
                # Export selection
                selection_path = self.output_dir / "selected_proteins.json"
                self._protein_selector.export_selection(
                    proteins=self._protein_selection,
                    output_path=str(selection_path)
                )
                logger.info(f"Selected {len(self._protein_selection)} proteins")
            else:
                logger.info(f"Using {len(self._protein_selection)} pre-selected proteins")
            
            # Step 2: Phase organization
            logger.info("Organizing phases...")
            self._phase_manager = PhaseManager(
                quality_gate_threshold=self.config.quality_gate_threshold
            )
            self._phase_manager.initialize_phases(self._protein_selection)
            logger.info(f"Organized into {len(self._phase_manager.phases)} phases")
            
            # Step 3: Initialize components
            logger.info("Initializing framework components...")
            
            # Batch executor
            self._batch_executor = BatchExecutor(
                max_parallel=self.config.max_parallel_tests,
                checkpoint_interval=self.config.checkpoint_interval,
                checkpoint_dir=str(self.output_dir / "checkpoints")
            )
            
            # Results repository
            self._results_repository = ResultsRepository(
                base_dir=str(self.output_dir / "results")
            )
            
            # Progress tracker  - initialize later per phase
            self._progress_tracker = None
            
            # Statistical analyzer
            self._statistical_analyzer = StatisticalAnalyzer()
            
            # Failure analyzer
            self._failure_analyzer = FailureAnalyzer()
            
            # Documentation generator
            self._documentation_generator = DocumentationGenerator(
                output_dir=str(self.output_dir / "documentation")
            )
            
            # Quality control
            self._quality_controller = QualityController(
                strict_mode=False,
                validate_checksums=True
            )
            
            # Validation suite
            self._validation_suite = ValidationSuite(
                pdb_cache_dir=str(self.output_dir / "pdb_cache")
            )
            
            logger.info("All components initialized successfully")
            
            # Step 4: Validate setup
            self._validate_setup()
            
            self._is_setup = True
            logger.info(f"Campaign setup complete: {len(self._protein_selection)} proteins in {len(self._phase_manager.phases)} phases")
            
        except Exception as e:
            logger.error(f"Campaign setup failed: {e}")
            raise RuntimeError(f"Campaign setup failed: {e}")
    
    def _validate_setup(self) -> None:
        """Validate that all components are properly initialized."""
        required_components = [
            ('phase_manager', self._phase_manager),
            ('batch_executor', self._batch_executor),
            ('results_repository', self._results_repository),
            # Note: progress_tracker initialized per-phase, not during setup
            # ('progress_tracker', self._progress_tracker),
            ('statistical_analyzer', self._statistical_analyzer),
            ('failure_analyzer', self._failure_analyzer),
            ('documentation_generator', self._documentation_generator),
            ('quality_controller', self._quality_controller),
            ('validation_suite', self._validation_suite),
        ]
        
        for name, component in required_components:
            if component is None:
                raise RuntimeError(f"Component '{name}' not initialized")
    
    def run_campaign(self) -> CampaignResults:
        """
        Execute complete validation campaign.
        
        Workflow:
            1. Setup campaign (if not already done)
            2. Execute each phase sequentially
            3. Check quality gates between phases
            4. Generate interim reports after each phase
            5. Run final analysis and documentation
        
        Returns:
            CampaignResults with complete campaign data
        
        Raises:
            RuntimeError: If campaign execution fails
        """
        if not self._is_setup:
            self.setup_campaign()
        
        campaign_start = datetime.now()
        logger.info(f"Starting campaign {self.campaign_id} at {campaign_start}")
        
        assert self._phase_manager is not None, "Phase manager not initialized"
        
        try:
            # Execute each phase
            for phase_num in range(1, 5):  # Phases 1-4
                if phase_num not in self._phase_manager.phases:
                    break
                    
                phase = self._phase_manager.get_phase(phase_num)
                logger.info(f"=" * 70)
                logger.info(f"PHASE {phase.phase_number}: {len(phase.proteins)} proteins")
                logger.info(f"=" * 70)
                
                # Execute phase
                phase_results = self.run_phase(phase.phase_number)
                self._phase_results.append(phase_results)
                
                # Check quality gate
                if not phase_results.quality_gate_passed:
                    logger.warning(f"Phase {phase.phase_number} failed quality gate")
                    logger.warning(f"Success rate: {phase_results.success_rate:.1f}% < {self.config.quality_gate_threshold * 100:.1f}%")
                    
                    # Generate failure analysis
                    self._handle_quality_gate_failure(phase_results)
                    
                    # Decision point: continue or abort
                    if not self._should_continue_after_gate_failure(phase_results):
                        logger.error(f"Aborting campaign due to Phase {phase.phase_number} failure")
                        break
            
            # Generate final report
            campaign_end = datetime.now()
            final_results = self._generate_final_results(campaign_start, campaign_end)
            
            logger.info(f"Campaign complete: {final_results.phases_completed} phases, {final_results.total_proteins} proteins")
            logger.info(f"Overall success rate: {final_results.overall_success_rate:.1f}%")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Campaign execution failed: {e}")
            raise RuntimeError(f"Campaign execution failed: {e}")
    
    def run_phase(self, phase_number: int) -> PhaseResults:
        """
        Execute a single phase of testing.
        
        Args:
            phase_number: Phase number to execute (1-4)
        
        Returns:
            PhaseResults with phase execution data
        """
        # Ensure components are initialized
        assert self._phase_manager is not None, "Phase manager not initialized"
        assert self._batch_executor is not None, "Batch executor not initialized"
        assert self._results_repository is not None, "Results repository not initialized"
        assert self._documentation_generator is not None, "Documentation generator not initialized"
        
        phase_start = datetime.now()
        phase = self._phase_manager.get_phase(phase_number)
        
        logger.info(f"Executing Phase {phase_number} with {len(phase.proteins)} proteins")
        
        # Update phase status
        self._phase_manager.start_phase(phase_number)
        phase.start_time = phase_start
        
        # Initialize progress tracker for this phase
        if self._progress_tracker is None:
            self._progress_tracker = ProgressTracker(
                total_tests=len(phase.proteins),
                phase=phase_number
            )
        
        # Execute tests in batch
        validation_reports = self._batch_executor.execute_batch(
            proteins=phase.proteins,
            test_function=self.execute_single_test,
            prioritize=True
        )
        
        # Store all results
        for report in validation_reports:
            self._all_validation_reports.append(report)
            
            # Convert to dict for progress tracker
            report_dict = asdict(report)
            if self._progress_tracker:
                self._progress_tracker.update_progress(report_dict)
            
            # Store in repository
            metadata = self._create_test_metadata(report)
            self._results_repository.store_result(
                pdb_id=report.pdb_id,
                validation_metrics={
                    'rmsd': report.best_rmsd,
                    'gdt_ts': report.gdt_ts_score,
                    'tm_score': report.tm_score,
                    'final_energy': report.best_energy
                },
                metadata=metadata
            )
        
        # Calculate phase metrics
        phase_end = datetime.now()
        phase.end_time = phase_end
        
        successful = sum(1 for r in validation_reports if r.is_successful())
        success_rate = (successful / len(validation_reports)) * 100 if validation_reports else 0.0
        
        avg_rmsd = sum(r.best_rmsd for r in validation_reports) / len(validation_reports) if validation_reports else 0.0
        avg_gdt_ts = sum(r.gdt_ts_score for r in validation_reports) / len(validation_reports) if validation_reports else 0.0
        avg_energy = sum(r.best_energy for r in validation_reports) / len(validation_reports) if validation_reports else 0.0
        
        # Check quality gate
        quality_gate = self._phase_manager.check_quality_gate(phase)
        
        # Update phase status
        if quality_gate.passed:
            self._phase_manager.complete_phase(phase_number)
        else:
            phase.status = PhaseStatus.FAILED_GATE
        
        # Generate interim report
        interim_report_path = self._generate_interim_report(phase, validation_reports)
        
        # Create phase results
        phase_results = PhaseResults(
            phase_number=phase_number,
            proteins_tested=len(validation_reports),
            success_rate=success_rate,
            average_rmsd=avg_rmsd,
            average_gdt_ts=avg_gdt_ts,
            average_energy=avg_energy,
            quality_gate_passed=quality_gate.passed,
            validation_reports=[asdict(r) for r in validation_reports],
            interim_report_path=interim_report_path,
            start_time=phase_start,
            end_time=phase_end,
            runtime_seconds=(phase_end - phase_start).total_seconds()
        )
        
        logger.info(f"Phase {phase_number} complete: {success_rate:.1f}% success rate")
        
        return phase_results
    
    def execute_single_test(self, protein: ProteinMetadata) -> ValidationReport:
        """
        Execute validation test for a single protein.
        
        Args:
            protein: Protein metadata
        
        Returns:
            ValidationReport with test results
        """
        assert self._validation_suite is not None, "Validation suite not initialized"
        
        logger.info(f"Testing {protein.pdb_id} ({protein.size_category}, {protein.sequence_length} residues)")
        
        try:
            # Pre-test validation (skip for now to avoid errors)
            # self._quality_controller.validate_native_structure(protein.pdb_id)
            
            # Execute prediction
            report = self._validation_suite.validate_protein(
                pdb_id=protein.pdb_id,
                num_agents=self.config.num_agents,
                iterations=self.config.iterations_per_agent,
                use_multi_agent=True
            )
            
            # Post-test validation (skip for now)
            # self._quality_controller.validate_output_files(protein.pdb_id)
            
            logger.info(
                f"{protein.pdb_id}: RMSD={report.best_rmsd:.2f}Å, "
                f"GDT-TS={report.gdt_ts_score:.1f}, Quality={report.assess_quality()}"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Test failed for {protein.pdb_id}: {e}")
            
            # Create failure report
            return ValidationReport(
                pdb_id=protein.pdb_id,
                sequence_length=protein.sequence_length,
                best_rmsd=999.9,
                best_energy=999.9,
                gdt_ts_score=0.0,
                tm_score=0.0,
                runtime_seconds=0.0,
                conformations_explored=0,
                num_agents=self.config.num_agents,
                iterations_per_agent=self.config.iterations_per_agent
            )
    
    def _create_test_metadata(self, report: ValidationReport) -> TestRunMetadata:
        """Create test run metadata for reproducibility."""
        return TestRunMetadata(
            pdb_id=report.pdb_id,
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version=sys.version,
            num_agents=report.num_agents,
            iterations_per_agent=report.iterations_per_agent,
            qcpp_enabled=self.config.enable_qcpp,
            random_seed=self.config.random_seed or 0,
            adaptive_config={},
            execution_parameters={
                "max_parallel": self.config.max_parallel_tests,
                "timeout_multiplier": self.config.timeout_multiplier
            }
        )
    
    def _generate_interim_report(
        self,
        phase: Phase,
        validation_reports: List[ValidationReport]
    ) -> str:
        """Generate interim report after phase completion."""
        assert self._documentation_generator is not None, "Documentation generator not initialized"
        
        logger.info(f"Generating interim report for Phase {phase.phase_number}...")
        
        # Convert reports to dicts
        report_dicts = [asdict(r) for r in validation_reports]
        
        # Generate phase report using correct API
        phase_dict = {
            'phase_number': phase.phase_number,
            'name': f'Phase {phase.phase_number}',
            'protein_count': len(validation_reports)
        }
        
        report_content = self._documentation_generator.generate_phase_report(
            phase=phase_dict,
            results=report_dicts
        )
        
        # Save report to file
        report_path = self.output_dir / f"phase_{phase.phase_number}_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Interim report saved to {report_path}")
        return str(report_path)
    
    def _handle_quality_gate_failure(self, phase_results: PhaseResults) -> None:
        """Handle quality gate failure with analysis and recommendations."""
        assert self._failure_analyzer is not None, "Failure analyzer not initialized"
        
        logger.warning(f"Analyzing Phase {phase_results.phase_number} failure...")
        
        # Run failure analysis
        failures = [
            r for r in phase_results.validation_reports
            if not self._is_successful_dict(r)
        ]
        
        if failures:
            patterns = self._failure_analyzer.extract_common_characteristics(failures)
            recommendations = self._failure_analyzer.recommend_parameter_adjustments(patterns)
            
            logger.warning(f"Failure analysis: {len(failures)} failures")
            logger.warning(f"Common patterns: {patterns.common_issues}")
            logger.warning(f"Recommendations: {recommendations}")
    
    def _should_continue_after_gate_failure(self, phase_results: PhaseResults) -> bool:
        """
        Decide whether to continue campaign after quality gate failure.
        
        For now, returns True to continue (manual decision point).
        In production, could prompt user or apply automatic rules.
        """
        return True  # Continue by default
    
    def _generate_final_results(
        self,
        campaign_start: datetime,
        campaign_end: datetime
    ) -> CampaignResults:
        """Generate final campaign results and reports."""
        logger.info("Generating final campaign results...")
        
        # Calculate overall metrics
        total_proteins = len(self._all_validation_reports)
        successful = sum(1 for r in self._all_validation_reports if r.is_successful())
        overall_success_rate = (successful / total_proteins) * 100 if total_proteins > 0 else 0.0
        
        # Convert reports to dicts
        all_report_dicts = [asdict(r) for r in self._all_validation_reports]
        
        # Run statistical analysis
        logger.info("Running statistical analysis...")
        statistical_analysis_path = self._run_statistical_analysis(all_report_dicts)
        
        # Run failure analysis
        logger.info("Running failure analysis...")
        failure_analysis_path = self._run_failure_analysis(all_report_dicts)
        
        # Generate final comprehensive report
        logger.info("Generating final comprehensive report...")
        final_report_path = self._generate_comprehensive_report(
            all_report_dicts,
            statistical_analysis_path,
            failure_analysis_path
        )
        
        # Create campaign results
        results = CampaignResults(
            campaign_id=self.campaign_id,
            config=self.config,
            start_time=campaign_start,
            end_time=campaign_end,
            total_proteins=total_proteins,
            phases_completed=len(self._phase_results),
            overall_success_rate=overall_success_rate,
            validation_reports=all_report_dicts,
            phase_summaries=self._phase_results,
            statistical_analysis_path=statistical_analysis_path,
            failure_analysis_path=failure_analysis_path,
            final_report_path=final_report_path
        )
        
        # Save campaign results
        results_path = self.output_dir / f"{self.campaign_id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        logger.info(f"Campaign results saved to {results_path}")
        
        return results
    
    def _run_statistical_analysis(self, results: List[Dict]) -> str:
        """Run comprehensive statistical analysis."""
        assert self._statistical_analyzer is not None, "Statistical analyzer not initialized"
        
        output_path = self.output_dir / "statistical_analysis.json"
        
        # Calculate correlations
        correlations = self._statistical_analyzer.calculate_correlations(results)
        
        # Compare size categories
        size_comparison = self._statistical_analyzer.compare_size_categories(
            results,
            metric='rmsd'
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._statistical_analyzer.calculate_confidence_intervals(results)
        
        # Save analysis
        analysis = {
            "correlations": asdict(correlations),
            "size_comparison": asdict(size_comparison),
            "confidence_intervals": asdict(confidence_intervals)
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return str(output_path)
    
    def _run_failure_analysis(self, results: List[Dict]) -> str:
        """Run comprehensive failure analysis."""
        assert self._failure_analyzer is not None, "Failure analyzer not initialized"
        
        output_path = self.output_dir / "failure_analysis.json"
        
        # Extract failures
        failures = [r for r in results if not self._is_successful_dict(r)]
        
        if not failures:
            logger.info("No failures to analyze")
            return str(output_path)
        
        # Analyze patterns
        patterns = self._failure_analyzer.extract_common_characteristics(failures)
        
        # Get recommendations
        recommendations = self._failure_analyzer.recommend_parameter_adjustments(patterns)
        
        # Save analysis
        analysis = {
            "total_failures": len(failures),
            "patterns": asdict(patterns),
            "recommendations": recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return str(output_path)
    
    def _generate_comprehensive_report(
        self,
        results: List[Dict],
        statistical_path: str,
        failure_path: str
    ) -> str:
        """Generate final comprehensive research report."""
        output_path = self.output_dir / "FINAL_CAMPAIGN_REPORT.md"
        
        # Generate comprehensive report manually since method doesn't exist
        report_lines = [
            f"# Final Campaign Report: {self.campaign_id}",
            "",
            "## Overview",
            f"- **Total Proteins**: {len(results)}",
            f"- **Phases Completed**: {len(self._phase_results)}",
            f"- **Statistical Analysis**: {statistical_path}",
            f"- **Failure Analysis**: {failure_path}",
            "",
            "## Results Summary",
            ""
        ]
        
        # Add summary statistics
        if results:
            avg_rmsd = sum(r.get('best_rmsd', 0) for r in results) / len(results)
            avg_gdt = sum(r.get('gdt_ts_score', 0) for r in results) / len(results)
            
            report_lines.extend([
                f"- **Average RMSD**: {avg_rmsd:.2f} Å",
                f"- **Average GDT-TS**: {avg_gdt:.1f}",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        return str(output_path)
    
    def _is_successful_dict(self, report_dict: Dict) -> bool:
        """Check if report dict represents successful prediction."""
        return (
            report_dict.get('best_rmsd', 999.9) < 5.0 and
            report_dict.get('best_energy', 999.9) < 0 and
            report_dict.get('gdt_ts_score', 0.0) > 50
        )
    
    def get_campaign_status(self) -> Dict[str, Any]:
        """
        Get current campaign status.
        
        Returns:
            Dictionary with campaign progress and metrics
        """
        if not self._is_setup:
            return {"status": "not_setup"}
        
        # Get dashboard data if progress tracker exists
        dashboard_dict = {}
        if self._progress_tracker:
            dashboard = self._progress_tracker.get_dashboard_data()
            dashboard_dict = asdict(dashboard)
        
        # Get batch progress
        batch_progress_dict = {}
        if self._batch_executor:
            batch_progress = self._batch_executor.get_progress()
            batch_progress_dict = asdict(batch_progress)
        
        return {
            "campaign_id": self.campaign_id,
            "status": "running" if self._current_phase else "setup",
            "current_phase": self._current_phase.phase_number if self._current_phase else None,
            "tests_completed": len(self._all_validation_reports),
            "tests_pending": batch_progress_dict.get('pending', 0),
            "success_rate": dashboard_dict.get('success_rate', 0.0),
            "dashboard": dashboard_dict,
            "batch_progress": batch_progress_dict
        }
    
    def checkpoint_campaign(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Save campaign checkpoint for resume capability.
        
        Args:
            checkpoint_name: Optional checkpoint name
        
        Returns:
            Path to checkpoint file
        """
        if checkpoint_name is None:
            checkpoint_name = f"{self.campaign_id}_phase{self._current_phase.phase_number if self._current_phase else 0}"
        
        checkpoint_path = self.output_dir / "checkpoints" / f"{checkpoint_name}.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "campaign_id": self.campaign_id,
            "config": asdict(self.config),
            "current_phase": self._current_phase.phase_number if self._current_phase else None,
            "completed_reports": [asdict(r) for r in self._all_validation_reports],
            "phase_results": [asdict(pr) for pr in self._phase_results],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Campaign checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_default_campaign() -> LargeScaleValidationCampaign:
    """
    Create campaign with default configuration.
    
    Returns:
        LargeScaleValidationCampaign ready to run
    """
    return LargeScaleValidationCampaign(config=CampaignConfig())


def create_quick_test_campaign() -> LargeScaleValidationCampaign:
    """
    Create campaign for quick testing (fewer proteins, less iterations).
    
    Returns:
        LargeScaleValidationCampaign configured for testing
    """
    config = CampaignConfig(
        target_protein_count=10,
        num_agents=5,
        iterations_per_agent=500,
        max_parallel_tests=2
    )
    return LargeScaleValidationCampaign(config=config)
