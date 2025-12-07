"""
Prediction Engine Public API

This module defines the PUBLIC interface for the prediction engine.
All external code (backend, CLI tools) should ONLY import from this module.

Internal implementation details (multi_agent_coordinator, energy_function, etc.)
should NEVER be imported directly by external code.

This follows the Dependency Inversion Principle (DIP):
- High-level modules (backend) depend on abstractions (this API)
- Low-level modules (engine internals) implement the abstractions

Usage:
    from ubf_protein.api import PredictionRunner, PredictionConfig
    
    config = PredictionConfig(sequence="ACDEFGH...")
    runner = PredictionRunner(config)
    results = runner.run()
"""

from .interfaces import (
    IPredictionRunner,
    IProgressCallback,
    IResultsExporter,
)

from .schemas import (
    PredictionConfig,
    PredictionResults,
    ProgressUpdate,
    ScreeningConfig,
    ScreeningResults,
    AggregationRegion,
    AggregationRisk,
    QCPPConfig,
    ValidationMetrics,
)

from .runner import (
    PredictionRunner,
    get_optimal_settings,
    get_quick_test_settings,
)

from .screening import (
    AggregationScreener,
)

from .exporters import (
    PDBExporter,
    JSONExporter,
)

__all__ = [
    # Interfaces (for type hints and dependency injection)
    'IPredictionRunner',
    'IProgressCallback', 
    'IResultsExporter',
    
    # Data classes / Schemas
    'PredictionConfig',
    'PredictionResults',
    'ProgressUpdate',
    'ScreeningConfig',
    'ScreeningResults',
    'AggregationRegion',
    'AggregationRisk',
    'QCPPConfig',
    'ValidationMetrics',
    
    # Concrete implementations
    'PredictionRunner',
    'AggregationScreener',
    'PDBExporter',
    'JSONExporter',
    
    # Utility functions
    'get_optimal_settings',
    'get_quick_test_settings',
]
