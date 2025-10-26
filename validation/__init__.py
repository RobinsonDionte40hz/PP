"""
Large-Scale Protein Validation Framework

This package provides comprehensive validation for testing 50-75 proteins
using the integrated QCPP-UBF platform.

Components:
- ProteinSelector: Curate diverse test sets from PDB
- PhaseManager: Organize testing into progressive phases
- ResultsRepository: Centralized storage for test results
- BatchExecutor: Parallel execution with resource management
- ProgressTracker: Real-time monitoring and dashboards
- StatisticalAnalyzer: Pattern detection and correlation analysis
- FailureAnalyzer: Detailed failure analysis
- DocumentationGenerator: Automated research reports
- LargeScaleValidationCampaign: Main orchestrator
"""

from .protein_selector import ProteinSelector, ProteinMetadata
from .phase_manager import (
    PhaseManager,
    Phase,
    PhaseStatus,
    QualityGateResult,
    PhaseSummaryReport
)

__all__ = [
    'ProteinSelector',
    'ProteinMetadata',
    'PhaseManager',
    'Phase',
    'PhaseStatus',
    'QualityGateResult',
    'PhaseSummaryReport',
]
