"""
Phase Manager for Large-Scale Validation

Organizes testing into 4 progressive phases with quality gates and parameter adjustment.
Enables iterative research by validating system performance before proceeding to larger batches.

Key Features:
- 4-phase progressive testing (10, 15, 25, remaining proteins)
- Quality gate checking (60% success threshold for Phase 1)
- Phase transition management with status tracking
- Phase summary report generation
- Parameter adjustment interface between phases
- Difficulty-based protein distribution
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

from .protein_selector import ProteinMetadata

logger = logging.getLogger(__name__)


class PhaseStatus(Enum):
    """Status of a testing phase."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED_GATE = "failed_gate"


@dataclass
class Phase:
    """
    A single phase in the progressive testing campaign.
    
    Attributes:
        phase_number: Phase identifier (1, 2, 3, 4)
        protein_count: Target number of proteins for this phase
        proteins: List of proteins assigned to this phase
        status: Current status (pending, in_progress, completed, failed_gate)
        start_time: When phase execution started
        end_time: When phase execution ended
        success_rate: Percentage of successful predictions (0-100)
        average_rmsd: Average RMSD across all predictions in this phase
        average_gdt_ts: Average GDT-TS score across predictions
        average_tm_score: Average TM-score across predictions
        average_energy: Average final energy across predictions
        failed_proteins: List of PDB IDs that failed
        execution_times: Dict of PDB ID -> execution time (seconds)
    """
    phase_number: int
    protein_count: int
    proteins: List[ProteinMetadata]
    status: PhaseStatus = PhaseStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success_rate: float = 0.0
    average_rmsd: float = 0.0
    average_gdt_ts: float = 0.0
    average_tm_score: float = 0.0
    average_energy: float = 0.0
    failed_proteins: List[str] = field(default_factory=list)
    execution_times: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate phase after initialization."""
        if self.phase_number not in [1, 2, 3, 4]:
            raise ValueError(f"phase_number must be 1-4, got {self.phase_number}")
        
        if not isinstance(self.status, PhaseStatus):
            # Convert string to enum if needed
            if isinstance(self.status, str):
                self.status = PhaseStatus(self.status)
    
    def is_complete(self) -> bool:
        """Check if phase has been completed."""
        return self.status in [PhaseStatus.COMPLETED, PhaseStatus.FAILED_GATE]
    
    def get_duration_seconds(self) -> Optional[float]:
        """Get phase execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class QualityGateResult:
    """
    Result of quality gate checking for a phase.
    
    Quality gates ensure system performance before proceeding to next phase.
    Phase 1 requires 60% success rate to pass.
    
    Attributes:
        passed: Whether the phase passed the quality gate
        success_rate: Actual success rate achieved (0-100)
        threshold: Required success rate threshold (0-100)
        issues_identified: List of issues detected
        recommendations: List of recommended actions
        phase_number: Which phase was checked
    """
    passed: bool
    success_rate: float
    threshold: float
    issues_identified: List[str]
    recommendations: List[str]
    phase_number: int
    
    def get_summary(self) -> str:
        """Get human-readable summary of quality gate result."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        
        summary = f"""
Quality Gate - Phase {self.phase_number}: {status}
{'=' * 60}
Success Rate:    {self.success_rate:.1f}% (threshold: {self.threshold:.1f}%)
Gate Status:     {status}

Issues Identified:
"""
        if self.issues_identified:
            for issue in self.issues_identified:
                summary += f"  - {issue}\n"
        else:
            summary += "  None\n"
        
        summary += "\nRecommendations:\n"
        if self.recommendations:
            for rec in self.recommendations:
                summary += f"  - {rec}\n"
        else:
            summary += "  Continue to next phase\n"
        
        return summary


@dataclass
class PhaseSummaryReport:
    """
    Summary report for a completed phase.
    
    Provides comprehensive overview of phase execution and results.
    
    Attributes:
        phase_number: Phase identifier
        proteins_tested: Number of proteins tested
        proteins_succeeded: Number of successful predictions
        proteins_failed: Number of failed predictions
        success_rate: Success rate percentage
        average_rmsd: Average RMSD across all predictions
        average_gdt_ts: Average GDT-TS score
        average_tm_score: Average TM-score
        average_energy: Average final energy
        execution_time_seconds: Total execution time
        quality_gate_result: Quality gate checking result
        top_performers: List of best predictions (PDB IDs)
        worst_performers: List of worst predictions (PDB IDs)
        generated_timestamp: When report was generated
    """
    phase_number: int
    proteins_tested: int
    proteins_succeeded: int
    proteins_failed: int
    success_rate: float
    average_rmsd: float
    average_gdt_ts: float
    average_tm_score: float
    average_energy: float
    execution_time_seconds: float
    quality_gate_result: QualityGateResult
    top_performers: List[str]
    worst_performers: List[str]
    generated_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        report = f"""
# Phase {self.phase_number} Summary Report

**Generated:** {self.generated_timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Proteins Tested:** {self.proteins_tested}
- **Succeeded:** {self.proteins_succeeded} ({self.success_rate:.1f}%)
- **Failed:** {self.proteins_failed}
- **Execution Time:** {self.execution_time_seconds/60:.1f} minutes

## Performance Metrics

| Metric | Value | Quality |
|--------|-------|---------|
| Average RMSD | {self.average_rmsd:.2f} Å | {'✓ Excellent' if self.average_rmsd < 3.0 else '○ Good' if self.average_rmsd < 5.0 else '✗ Poor'} |
| Average GDT-TS | {self.average_gdt_ts:.1f} | {'✓ Excellent' if self.average_gdt_ts > 70 else '○ Good' if self.average_gdt_ts > 50 else '✗ Poor'} |
| Average TM-Score | {self.average_tm_score:.3f} | {'✓ Excellent' if self.average_tm_score > 0.7 else '○ Good' if self.average_tm_score > 0.5 else '✗ Poor'} |
| Average Energy | {self.average_energy:.1f} kcal/mol | {'✓ Stable' if self.average_energy < 0 else '✗ Unstable'} |

## Quality Gate

{self.quality_gate_result.get_summary()}

## Top Performers

"""
        for i, pdb_id in enumerate(self.top_performers, 1):
            report += f"{i}. {pdb_id}\n"
        
        report += "\n## Worst Performers\n\n"
        for i, pdb_id in enumerate(self.worst_performers, 1):
            report += f"{i}. {pdb_id}\n"
        
        return report


class PhaseManager:
    """
    Manage progressive testing phases with quality gates.
    
    Organizes proteins into 4 phases:
    - Phase 1: 10 proteins (easy, well-studied, high resolution)
    - Phase 2: 15 proteins (mixed difficulty)
    - Phase 3: 25 proteins (diverse characteristics)
    - Phase 4: Remaining proteins (challenging cases)
    
    Each phase has quality gate checking to ensure system performance
    before proceeding. Phase 1 requires 60% success rate.
    
    Example:
        manager = PhaseManager()
        phases = manager.initialize_phases(proteins)
        
        # Execute phase 1
        current_phase = manager.get_current_phase()
        # ... run tests ...
        manager.update_phase_results(current_phase, results)
        
        # Check quality gate
        gate_result = manager.check_quality_gate(current_phase)
        if gate_result.passed:
            manager.advance_to_next_phase()
    """
    
    def __init__(self,
                 phase1_count: int = 10,
                 phase2_count: int = 15,
                 phase3_count: int = 25,
                 quality_gate_threshold: float = 60.0):
        """
        Initialize phase manager.
        
        Args:
            phase1_count: Number of proteins for Phase 1 (default: 10)
            phase2_count: Number of proteins for Phase 2 (default: 15)
            phase3_count: Number of proteins for Phase 3 (default: 25)
            quality_gate_threshold: Success rate threshold for quality gates (default: 60%)
        """
        self.phase1_count = phase1_count
        self.phase2_count = phase2_count
        self.phase3_count = phase3_count
        self.quality_gate_threshold = quality_gate_threshold
        
        self.phases: Dict[int, Phase] = {}
        self.current_phase_number = 1
        
        logger.info(f"Initialized PhaseManager with {phase1_count}, {phase2_count}, {phase3_count} proteins per phase")
        logger.info(f"Quality gate threshold: {quality_gate_threshold}%")
    
    def initialize_phases(self, proteins: List[ProteinMetadata]) -> Dict[int, Phase]:
        """
        Initialize phases by distributing proteins.
        
        Distribution strategy:
        - Phase 1: Small proteins with high resolution (easy)
        - Phase 2: Mixed sizes and resolutions (moderate)
        - Phase 3: Diverse characteristics (challenging)
        - Phase 4: Remaining proteins
        
        Args:
            proteins: List of all proteins to distribute
        
        Returns:
            Dictionary of phase_number -> Phase
        
        Raises:
            ValueError: If not enough proteins for minimum distribution
        """
        if len(proteins) < (self.phase1_count + self.phase2_count + self.phase3_count):
            logger.warning(
                f"Only {len(proteins)} proteins available, but need "
                f"{self.phase1_count + self.phase2_count + self.phase3_count} "
                f"for first 3 phases. Adjusting phase sizes."
            )
        
        # Sort proteins by difficulty (easier first)
        sorted_proteins = self._sort_by_difficulty(proteins)
        
        # Distribute proteins to phases
        phase1_proteins = sorted_proteins[:self.phase1_count]
        phase2_proteins = sorted_proteins[self.phase1_count:self.phase1_count + self.phase2_count]
        phase3_proteins = sorted_proteins[
            self.phase1_count + self.phase2_count:
            self.phase1_count + self.phase2_count + self.phase3_count
        ]
        phase4_proteins = sorted_proteins[self.phase1_count + self.phase2_count + self.phase3_count:]
        
        # Create phase objects
        self.phases = {
            1: Phase(
                phase_number=1,
                protein_count=len(phase1_proteins),
                proteins=phase1_proteins,
                status=PhaseStatus.PENDING
            ),
            2: Phase(
                phase_number=2,
                protein_count=len(phase2_proteins),
                proteins=phase2_proteins,
                status=PhaseStatus.PENDING
            ),
            3: Phase(
                phase_number=3,
                protein_count=len(phase3_proteins),
                proteins=phase3_proteins,
                status=PhaseStatus.PENDING
            ),
            4: Phase(
                phase_number=4,
                protein_count=len(phase4_proteins),
                proteins=phase4_proteins,
                status=PhaseStatus.PENDING
            )
        }
        
        logger.info(f"Initialized 4 phases with {len(phase1_proteins)}, {len(phase2_proteins)}, "
                   f"{len(phase3_proteins)}, {len(phase4_proteins)} proteins")
        
        return self.phases
    
    def _sort_by_difficulty(self, proteins: List[ProteinMetadata]) -> List[ProteinMetadata]:
        """
        Sort proteins by difficulty (easier first).
        
        Difficulty factors:
        1. Size: Smaller is easier
        2. Resolution: Better resolution is easier (X-ray only)
        3. Structural class: Some classes are easier than others
        
        Args:
            proteins: List of proteins to sort
        
        Returns:
            Sorted list (easiest first)
        """
        def difficulty_score(protein: ProteinMetadata) -> float:
            """Calculate difficulty score (lower = easier)."""
            score = 0.0
            
            # Size factor (normalized 0-1)
            # Assume max size is 500 residues
            score += protein.sequence_length / 500.0
            
            # Resolution factor (if X-ray)
            if protein.experimental_method == 'X-ray' and protein.resolution:
                # Better resolution (lower value) = easier
                # Normalize: 1.0Å = 0, 3.0Å = 1
                score += min((protein.resolution - 1.0) / 2.0, 1.0)
            else:
                # NMR structures: moderate difficulty
                score += 0.5
            
            # Structural class factor
            class_difficulty = {
                'all-alpha': 0.3,      # Relatively easy
                'all-beta': 0.5,       # Moderate
                'alpha+beta': 0.6,     # Moderate-hard
                'alpha-beta': 0.7,     # Hard (complex topology)
                'irregular': 0.8       # Hardest
            }
            score += class_difficulty.get(protein.structural_class, 0.5)
            
            return score
        
        return sorted(proteins, key=difficulty_score)
    
    def get_current_phase(self) -> Phase:
        """
        Get the current active phase.
        
        Returns:
            Current Phase object
        
        Raises:
            ValueError: If no phases initialized or invalid phase number
        """
        if not self.phases:
            raise ValueError("No phases initialized. Call initialize_phases() first.")
        
        if self.current_phase_number not in self.phases:
            raise ValueError(f"Invalid phase number: {self.current_phase_number}")
        
        return self.phases[self.current_phase_number]
    
    def get_phase(self, phase_number: int) -> Phase:
        """
        Get specific phase by number.
        
        Args:
            phase_number: Phase identifier (1-4)
        
        Returns:
            Phase object
        
        Raises:
            ValueError: If phase doesn't exist
        """
        if phase_number not in self.phases:
            raise ValueError(f"Phase {phase_number} does not exist")
        
        return self.phases[phase_number]
    
    def advance_to_next_phase(self) -> bool:
        """
        Advance to the next phase.
        
        Returns:
            True if advanced, False if no more phases
        
        Raises:
            ValueError: If current phase not completed
        """
        current = self.get_current_phase()
        
        if not current.is_complete():
            raise ValueError(f"Phase {self.current_phase_number} is not complete")
        
        if self.current_phase_number < 4:
            self.current_phase_number += 1
            logger.info(f"Advanced to Phase {self.current_phase_number}")
            return True
        else:
            logger.info("All phases complete")
            return False
    
    def start_phase(self, phase_number: int) -> None:
        """
        Mark a phase as started.
        
        Args:
            phase_number: Phase to start
        """
        phase = self.get_phase(phase_number)
        phase.status = PhaseStatus.IN_PROGRESS
        phase.start_time = datetime.now()
        logger.info(f"Started Phase {phase_number}")
    
    def complete_phase(self, phase_number: int) -> None:
        """
        Mark a phase as completed.
        
        Args:
            phase_number: Phase to complete
        """
        phase = self.get_phase(phase_number)
        phase.status = PhaseStatus.COMPLETED
        phase.end_time = datetime.now()
        logger.info(f"Completed Phase {phase_number}")
    
    def fail_phase(self, phase_number: int) -> None:
        """
        Mark a phase as failed quality gate.
        
        Args:
            phase_number: Phase that failed
        """
        phase = self.get_phase(phase_number)
        phase.status = PhaseStatus.FAILED_GATE
        phase.end_time = datetime.now()
        logger.warning(f"Phase {phase_number} failed quality gate")
    
    def update_phase_results(self,
                            phase: Phase,
                            results: List[Dict[str, Any]]) -> None:
        """
        Update phase with test results.
        
        Args:
            phase: Phase to update
            results: List of test result dictionaries with keys:
                    - pdb_id: str
                    - success: bool
                    - rmsd: float
                    - gdt_ts: float
                    - tm_score: float
                    - energy: float
                    - execution_time: float (seconds)
        """
        if not results:
            logger.warning(f"No results to update for Phase {phase.phase_number}")
            return
        
        # Calculate metrics
        successes = sum(1 for r in results if r.get('success', False))
        phase.success_rate = (successes / len(results)) * 100.0
        
        # Average metrics (only from successful predictions)
        successful_results = [r for r in results if r.get('success', False)]
        
        if successful_results:
            phase.average_rmsd = sum(r['rmsd'] for r in successful_results) / len(successful_results)
            phase.average_gdt_ts = sum(r['gdt_ts'] for r in successful_results) / len(successful_results)
            phase.average_tm_score = sum(r['tm_score'] for r in successful_results) / len(successful_results)
            phase.average_energy = sum(r['energy'] for r in successful_results) / len(successful_results)
        
        # Track failed proteins
        phase.failed_proteins = [r['pdb_id'] for r in results if not r.get('success', False)]
        
        # Track execution times
        phase.execution_times = {r['pdb_id']: r.get('execution_time', 0.0) for r in results}
        
        logger.info(f"Updated Phase {phase.phase_number} with {len(results)} results")
        logger.info(f"  Success rate: {phase.success_rate:.1f}%")
        logger.info(f"  Average RMSD: {phase.average_rmsd:.2f} Å")
        logger.info(f"  Average GDT-TS: {phase.average_gdt_ts:.1f}")
    
    def check_quality_gate(self, phase: Phase) -> QualityGateResult:
        """
        Check if phase passes quality gate.
        
        Quality gate criteria:
        - Phase 1: 60% success rate required
        - Other phases: No strict requirement (informational)
        
        Args:
            phase: Phase to check
        
        Returns:
            QualityGateResult with pass/fail and recommendations
        """
        issues = []
        recommendations = []
        
        # Check success rate
        passed = phase.success_rate >= self.quality_gate_threshold
        
        if not passed:
            issues.append(
                f"Success rate {phase.success_rate:.1f}% below threshold {self.quality_gate_threshold:.1f}%"
            )
            
            # Analyze failures
            if phase.average_rmsd > 5.0:
                issues.append(f"High average RMSD: {phase.average_rmsd:.2f} Å")
                recommendations.append("Consider increasing agent count or iterations")
            
            if phase.average_gdt_ts < 50:
                issues.append(f"Low average GDT-TS: {phase.average_gdt_ts:.1f}")
                recommendations.append("Review structural similarity metrics")
            
            if phase.average_energy > 0:
                issues.append(f"Positive average energy: {phase.average_energy:.1f} kcal/mol")
                recommendations.append("Energy minimization may need adjustment")
            
            # Check for systematic failures
            if len(phase.failed_proteins) > len(phase.proteins) * 0.5:
                issues.append(f"High failure rate: {len(phase.failed_proteins)}/{len(phase.proteins)} proteins failed")
                recommendations.append("Review system parameters before continuing")
        else:
            recommendations.append("Performance meets quality gate threshold")
            recommendations.append("Proceed to next phase")
        
        result = QualityGateResult(
            passed=passed,
            success_rate=phase.success_rate,
            threshold=self.quality_gate_threshold,
            issues_identified=issues,
            recommendations=recommendations,
            phase_number=phase.phase_number
        )
        
        logger.info(f"Quality gate for Phase {phase.phase_number}: {'PASSED' if passed else 'FAILED'}")
        
        return result
    
    def generate_phase_summary(self, phase: Phase, results: List[Dict[str, Any]]) -> PhaseSummaryReport:
        """
        Generate comprehensive summary report for a phase.
        
        Args:
            phase: Phase to summarize
            results: Test results for the phase
        
        Returns:
            PhaseSummaryReport with comprehensive metrics
        """
        # Get quality gate result
        gate_result = self.check_quality_gate(phase)
        
        # Find top and worst performers
        successful_results = [r for r in results if r.get('success', False)]
        
        # Sort by RMSD (lower is better)
        sorted_by_rmsd = sorted(successful_results, key=lambda r: r['rmsd'])
        top_performers = [r['pdb_id'] for r in sorted_by_rmsd[:5]]
        worst_performers = [r['pdb_id'] for r in sorted_by_rmsd[-5:]]
        
        # Calculate execution time
        execution_time = phase.get_duration_seconds() or 0.0
        
        # Create summary report
        report = PhaseSummaryReport(
            phase_number=phase.phase_number,
            proteins_tested=len(results),
            proteins_succeeded=len(successful_results),
            proteins_failed=len(phase.failed_proteins),
            success_rate=phase.success_rate,
            average_rmsd=phase.average_rmsd,
            average_gdt_ts=phase.average_gdt_ts,
            average_tm_score=phase.average_tm_score,
            average_energy=phase.average_energy,
            execution_time_seconds=execution_time,
            quality_gate_result=gate_result,
            top_performers=top_performers,
            worst_performers=worst_performers
        )
        
        logger.info(f"Generated summary report for Phase {phase.phase_number}")
        
        return report
    
    def allow_parameter_adjustment(self, phase: Phase) -> Dict[str, Any]:
        """
        Generate parameter adjustment recommendations based on phase results.
        
        Analyzes phase performance and suggests parameter changes for next phase.
        
        Args:
            phase: Completed phase to analyze
        
        Returns:
            Dictionary of recommended parameter adjustments
        """
        adjustments = {}
        
        # If success rate is low, increase computational resources
        if phase.success_rate < 70:
            adjustments['num_agents'] = 'increase'
            adjustments['num_agents_recommendation'] = "Increase from 10 to 15 agents"
            
            adjustments['iterations'] = 'increase'
            adjustments['iterations_recommendation'] = "Increase from 1000 to 1500 iterations"
        
        # If RMSD is high, need more exploration
        if phase.average_rmsd > 4.0:
            adjustments['exploration'] = 'increase'
            adjustments['exploration_recommendation'] = "Enable more diverse agent behaviors"
        
        # If energy is positive, need better energy minimization
        if phase.average_energy > 0:
            adjustments['energy_minimization'] = 'improve'
            adjustments['energy_minimization_recommendation'] = "Adjust energy function weights"
        
        # If GDT-TS is low, need better structural alignment
        if phase.average_gdt_ts < 60:
            adjustments['structural_guidance'] = 'increase'
            adjustments['structural_guidance_recommendation'] = "Increase QCPP integration weight"
        
        if not adjustments:
            adjustments['status'] = 'no_changes_needed'
            adjustments['message'] = "Performance is good, continue with current parameters"
        
        logger.info(f"Generated {len(adjustments)} parameter adjustment recommendations")
        
        return adjustments
    
    def export_phases(self, output_path: str) -> None:
        """
        Export phase configuration and results to JSON.
        
        Args:
            output_path: Path to JSON file
        """
        data = {
            'phase_config': {
                'phase1_count': self.phase1_count,
                'phase2_count': self.phase2_count,
                'phase3_count': self.phase3_count,
                'quality_gate_threshold': self.quality_gate_threshold
            },
            'current_phase': self.current_phase_number,
            'phases': {}
        }
        
        for phase_num, phase in self.phases.items():
            # Convert phase to dict
            phase_dict = asdict(phase)
            # Convert enum to string
            phase_dict['status'] = phase.status.value
            # Convert datetime to string
            if phase.start_time:
                phase_dict['start_time'] = phase.start_time.isoformat()
            if phase.end_time:
                phase_dict['end_time'] = phase.end_time.isoformat()
            
            data['phases'][phase_num] = phase_dict
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported phases to {output_path}")
    
    def load_phases(self, input_path: str) -> None:
        """
        Load phase configuration and results from JSON.
        
        Args:
            input_path: Path to JSON file
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Restore configuration
        config = data['phase_config']
        self.phase1_count = config['phase1_count']
        self.phase2_count = config['phase2_count']
        self.phase3_count = config['phase3_count']
        self.quality_gate_threshold = config['quality_gate_threshold']
        self.current_phase_number = data['current_phase']
        
        # Restore phases
        self.phases = {}
        for phase_num_str, phase_dict in data['phases'].items():
            phase_num = int(phase_num_str)
            
            # Convert status string to enum
            phase_dict['status'] = PhaseStatus(phase_dict['status'])
            
            # Convert datetime strings back to datetime
            if phase_dict.get('start_time'):
                phase_dict['start_time'] = datetime.fromisoformat(phase_dict['start_time'])
            if phase_dict.get('end_time'):
                phase_dict['end_time'] = datetime.fromisoformat(phase_dict['end_time'])
            
            # Reconstruct ProteinMetadata objects
            proteins = [ProteinMetadata(**p) for p in phase_dict['proteins']]
            phase_dict['proteins'] = proteins
            
            self.phases[phase_num] = Phase(**phase_dict)
        
        logger.info(f"Loaded phases from {input_path}")


# Example usage
if __name__ == '__main__':
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Import ProteinSelector
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from validation.protein_selector import ProteinSelector
    
    # Select proteins
    selector = ProteinSelector()
    proteins = selector.select_proteins(target_count=60)
    
    # Initialize phase manager
    manager = PhaseManager(
        phase1_count=10,
        phase2_count=15,
        phase3_count=25,
        quality_gate_threshold=60.0
    )
    
    # Initialize phases
    phases = manager.initialize_phases(proteins)
    
    print("\n" + "=" * 70)
    print("Phase Distribution")
    print("=" * 70)
    for phase_num, phase in phases.items():
        print(f"\nPhase {phase_num}:")
        print(f"  Proteins: {phase.protein_count}")
        print(f"  Status: {phase.status.value}")
        print(f"  First 3 proteins:")
        for p in phase.proteins[:3]:
            print(f"    - {p.pdb_id}: {p.sequence_length} residues, {p.description}")
