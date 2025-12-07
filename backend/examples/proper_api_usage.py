"""
Example: How to properly import from the prediction engine.

This file demonstrates the CORRECT way to use the prediction engine
from external code (backend, CLI tools, etc.).

SOLID Principle: Dependency Inversion
- High-level modules (backend) should not depend on low-level modules (engine internals)
- Both should depend on abstractions (ubf_protein.api)
"""

# ============================================================================
# CORRECT: Import from public API
# ============================================================================

from ubf_protein.api import (
    # Main runner
    PredictionRunner,
    
    # Configuration schemas
    PredictionConfig,
    PredictionResults,
    ProgressUpdate,
    ValidationMetrics,
    
    # Screening
    AggregationScreener,
    ScreeningConfig,
    ScreeningResults,
    
    # Exporters
    PDBExporter,
    JSONExporter,
    
    # Utility functions
    get_optimal_settings,
    get_quick_test_settings,
    
    # Interfaces (for type hints)
    IPredictionRunner,
    IProgressCallback,
)


# ============================================================================
# WRONG: Don't import internal modules directly
# ============================================================================

# These imports break the API boundary and create tight coupling:
#
# from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator  # ❌
# from ubf_protein.energy_function import EnergyFunction  # ❌
# from ubf_protein.qcpp_integration import QCPPIntegrationAdapter  # ❌
# from ubf_protein.rmsd_calculator import RMSDCalculator  # ❌
# from ubf_protein.geometric_attractor import GeometricAttractorAnalyzer  # ❌


# ============================================================================
# Example Usage
# ============================================================================

def run_prediction_example():
    """Example of running a prediction through the public API."""
    
    # Create configuration
    config = PredictionConfig(
        sequence="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        native_pdb="1UBQ",  # For RMSD validation
        agents=10,
        iterations=500,
        enable_refinement=True,
        qcpp_config="default",
    )
    
    # Create runner
    runner = PredictionRunner(config)
    
    # Define progress callback
    def on_progress(update: ProgressUpdate):
        print(f"[{update.phase}] {update.percentage:.1f}% - {update.message}")
        if update.best_energy:
            print(f"  Best energy: {update.best_energy:.2f}")
    
    # Run prediction
    results = runner.run(progress_callback=on_progress)
    
    # Access results through typed schema
    print(f"\n=== Prediction Complete ===")
    print(f"Runtime: {results.runtime_seconds:.1f}s")
    print(f"RMSD: {results.metrics.rmsd}")
    print(f"Energy: {results.metrics.energy_total}")
    
    # Export results
    pdb_exporter = PDBExporter()
    pdb_exporter.export(results, "output.pdb")
    
    json_exporter = JSONExporter()
    json_exporter.export(results, "output.json")
    
    return results


def run_screening_example():
    """Example of running aggregation screening."""
    
    screener = AggregationScreener()
    
    config = ScreeningConfig(
        window_size=7,
        threshold=0.5,
    )
    
    results = screener.screen(
        sequence="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        config=config,
    )
    
    print(f"\n=== Screening Results ===")
    print(f"Aggregation score: {results.aggregation_score:.2f}")
    print(f"Risk level: {results.risk_level.value}")
    print(f"Regions found: {len(results.regions)}")
    
    for region in results.regions:
        print(f"  {region.start}-{region.end}: {region.sequence} (score: {region.score:.2f})")
    
    return results


def type_hint_example(runner: IPredictionRunner):
    """
    Example showing how to use interfaces for type hints.
    
    This enables dependency injection and testing with mock implementations.
    """
    # The function accepts any implementation of IPredictionRunner
    # Could be the real runner or a mock for testing
    results = runner.run()
    return results


if __name__ == "__main__":
    # Run examples
    run_prediction_example()
    run_screening_example()
