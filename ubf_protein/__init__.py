# UBF Protein System Package
"""
Universal Behavioral Framework (UBF) integration with quantum-inspired protein structure prediction.

This package implements a mappless conformational navigation system where autonomous agents
explore protein folding space using consciousness coordinates, behavioral states, and experiential memory.

Key Modules:
- prediction_runner: Unified prediction interface (use this for all predictions)
- multi_agent_coordinator: Multi-agent exploration coordination
- qcpp_integration: Quantum Coherence Protein Predictor integration
- rmsd_calculator: Structure validation and RMSD calculation
- geometric_attractor: Golden ratio and Platonic solid analysis
"""

__version__ = "0.1.0"
__author__ = "UBF Protein Team"

# Export key classes for easy access
from .prediction_runner import (
    PredictionRunner,
    PredictionConfig,
    PredictionResults,
    ProgressUpdate,
    run_prediction,
    get_optimal_settings,
    get_fast_settings,
    get_quick_test_settings,
)