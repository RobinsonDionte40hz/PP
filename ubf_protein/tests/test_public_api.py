"""
Tests for the public API module (ubf_protein.api).

These tests verify that:
1. All public API exports are accessible
2. External code can use the API without importing internals
3. Data classes and interfaces work correctly
4. The API boundary is maintained

This ensures SOLID compliance - external code only depends on abstractions.
"""
import pytest
from typing import Optional
from dataclasses import is_dataclass


class TestAPIImports:
    """Test that all expected exports are available from the public API."""
    
    def test_import_main_api(self):
        """Test importing the main API module."""
        from ubf_protein import api
        assert api is not None
    
    def test_import_prediction_runner(self):
        """Test PredictionRunner is exported."""
        from ubf_protein.api import PredictionRunner
        assert PredictionRunner is not None
    
    def test_import_prediction_config(self):
        """Test PredictionConfig is exported."""
        from ubf_protein.api import PredictionConfig
        assert PredictionConfig is not None
        assert is_dataclass(PredictionConfig)
    
    def test_import_prediction_results(self):
        """Test PredictionResults is exported."""
        from ubf_protein.api import PredictionResults
        assert PredictionResults is not None
    
    def test_import_screener(self):
        """Test AggregationScreener is exported."""
        from ubf_protein.api import AggregationScreener
        assert AggregationScreener is not None
    
    def test_import_screening_config(self):
        """Test ScreeningConfig is exported."""
        from ubf_protein.api import ScreeningConfig
        assert ScreeningConfig is not None
        assert is_dataclass(ScreeningConfig)
    
    def test_import_aggregation_risk(self):
        """Test AggregationRisk enum is exported."""
        from ubf_protein.api import AggregationRisk
        assert AggregationRisk is not None
        # Verify it's an enum with expected values
        assert hasattr(AggregationRisk, 'LOW')
        assert hasattr(AggregationRisk, 'MODERATE')
        assert hasattr(AggregationRisk, 'HIGH')
        assert hasattr(AggregationRisk, 'CRITICAL')
    
    def test_import_interfaces(self):
        """Test interfaces are exported."""
        from ubf_protein.api import (
            IPredictionRunner,
            IProgressCallback,
            IResultsExporter,
        )
        assert IPredictionRunner is not None
        assert IProgressCallback is not None
        assert IResultsExporter is not None
    
    def test_import_exporters(self):
        """Test exporters are exported."""
        from ubf_protein.api import PDBExporter, JSONExporter
        assert PDBExporter is not None
        assert JSONExporter is not None
    
    def test_import_utility_functions(self):
        """Test utility functions are exported."""
        from ubf_protein.api import get_optimal_settings, get_quick_test_settings
        assert callable(get_optimal_settings)
        assert callable(get_quick_test_settings)


class TestPredictionConfig:
    """Test PredictionConfig dataclass."""
    
    def test_create_minimal_config(self):
        """Test creating config with minimal parameters."""
        from ubf_protein.api import PredictionConfig
        
        config = PredictionConfig(sequence="ACDEFGHIKLMNPQRSTVWY")
        assert config.sequence == "ACDEFGHIKLMNPQRSTVWY"
        assert config.agents > 0
        assert config.iterations > 0
    
    def test_create_full_config(self):
        """Test creating config with all parameters."""
        from ubf_protein.api import PredictionConfig
        
        config = PredictionConfig(
            sequence="ACDEFGH",
            agents=10,
            iterations=500,
            enable_refinement=True,
            enable_mediators=True,
            qcpp_config="default",
        )
        assert config.agents == 10
        assert config.iterations == 500
        assert config.enable_refinement is True
    
    def test_config_mutable(self):
        """Test that config is a regular dataclass (mutable)."""
        from ubf_protein.api import PredictionConfig
        
        config = PredictionConfig(sequence="ACDEFGH")
        # PredictionConfig is not frozen, mutation is allowed
        config.sequence = "DIFFERENT"
        assert config.sequence == "DIFFERENT"


class TestScreeningConfig:
    """Test ScreeningConfig dataclass."""
    
    def test_create_default_config(self):
        """Test creating default screening config."""
        from ubf_protein.api import ScreeningConfig
        
        config = ScreeningConfig()
        assert config.window_size > 0
        assert 0 <= config.threshold <= 1
    
    def test_create_custom_config(self):
        """Test creating custom screening config."""
        from ubf_protein.api import ScreeningConfig
        
        config = ScreeningConfig(window_size=9, threshold=0.4)
        assert config.window_size == 9
        assert config.threshold == 0.4


class TestUtilityFunctions:
    """Test utility functions exported from API."""
    
    def test_get_optimal_settings_small(self):
        """Test optimal settings for small proteins."""
        from ubf_protein.api import get_optimal_settings, PredictionConfig
        
        config = get_optimal_settings(30)
        assert isinstance(config, PredictionConfig)
        assert config.agents > 0
        assert config.iterations > 0
    
    def test_get_optimal_settings_large(self):
        """Test optimal settings for large proteins."""
        from ubf_protein.api import get_optimal_settings, PredictionConfig
        
        config = get_optimal_settings(200)
        assert isinstance(config, PredictionConfig)
        # Larger proteins should have more resources
        small_config = get_optimal_settings(30)
        assert config.agents >= small_config.agents or config.iterations >= small_config.iterations
    
    def test_get_quick_test_settings(self):
        """Test quick test settings."""
        from ubf_protein.api import get_quick_test_settings, PredictionConfig
        
        config = get_quick_test_settings(50)
        assert isinstance(config, PredictionConfig)
        # Quick settings should have fewer iterations
        assert config.iterations <= 200
    
    def test_quick_vs_optimal(self):
        """Test that quick settings are faster than optimal."""
        from ubf_protein.api import get_optimal_settings, get_quick_test_settings
        
        optimal = get_optimal_settings(50)
        quick = get_quick_test_settings(50)
        
        # Quick should have fewer iterations or agents
        assert (quick.iterations < optimal.iterations or 
                quick.agents < optimal.agents)


class TestAggregationRisk:
    """Test AggregationRisk enum."""
    
    def test_risk_levels_exist(self):
        """Test all expected risk levels exist."""
        from ubf_protein.api import AggregationRisk
        
        levels = [AggregationRisk.LOW, AggregationRisk.MODERATE, AggregationRisk.HIGH, AggregationRisk.CRITICAL]
        assert len(levels) == 4
    
    def test_risk_values(self):
        """Test risk level values."""
        from ubf_protein.api import AggregationRisk
        
        # Values should be strings for JSON serialization
        assert isinstance(AggregationRisk.LOW.value, str)
        assert isinstance(AggregationRisk.MODERATE.value, str)
        assert isinstance(AggregationRisk.HIGH.value, str)
        assert isinstance(AggregationRisk.CRITICAL.value, str)


class TestAPIBoundary:
    """Test that the API boundary is properly maintained."""
    
    def test_internal_modules_not_in_api(self):
        """Test that internal modules are not exported from API."""
        from ubf_protein import api
        
        # These should NOT be accessible via API
        internal_modules = [
            'MultiAgentCoordinator',
            'EnergyFunction',
            'ProteinAgent',
            'QCPPIntegrationAdapter',
            'RMSDCalculator',
        ]
        
        for module in internal_modules:
            assert not hasattr(api, module), f"Internal {module} should not be in API"
    
    def test_only_public_exports(self):
        """Test that __all__ defines the public API."""
        from ubf_protein import api
        
        # Should have __all__ defined
        assert hasattr(api, '__all__')
        assert len(api.__all__) > 0
        
        # All items in __all__ should be accessible
        for name in api.__all__:
            assert hasattr(api, name), f"{name} in __all__ but not accessible"


class TestPredictionRunnerInterface:
    """Test that PredictionRunner implements the interface correctly."""
    
    def test_runner_has_run_method(self):
        """Test PredictionRunner has run method."""
        from ubf_protein.api import PredictionRunner, PredictionConfig
        
        config = PredictionConfig(sequence="ACDEFGH")
        runner = PredictionRunner(config)
        
        assert hasattr(runner, 'run')
        assert callable(runner.run)
    
    def test_runner_has_cancel_method(self):
        """Test PredictionRunner has cancel method."""
        from ubf_protein.api import PredictionRunner, PredictionConfig
        
        config = PredictionConfig(sequence="ACDEFGH")
        runner = PredictionRunner(config)
        
        assert hasattr(runner, 'cancel')
        assert callable(runner.cancel)


class TestScreenerInterface:
    """Test that AggregationScreener works correctly."""
    
    def test_screener_has_screen_method(self):
        """Test AggregationScreener has screen method."""
        from ubf_protein.api import AggregationScreener, ScreeningConfig
        
        screener = AggregationScreener()
        assert hasattr(screener, 'screen')
        assert callable(screener.screen)
    
    def test_screener_basic_usage(self):
        """Test basic screener usage."""
        from ubf_protein.api import AggregationScreener, ScreeningConfig
        
        screener = AggregationScreener()
        config = ScreeningConfig(window_size=5, threshold=0.5)
        
        # Should not raise
        result = screener.screen("ACDEFGHIKLMNPQRSTVWY", config)
        assert result is not None


class TestExporters:
    """Test exporter classes."""
    
    def test_pdb_exporter_exists(self):
        """Test PDBExporter is functional."""
        from ubf_protein.api import PDBExporter
        
        exporter = PDBExporter()
        assert hasattr(exporter, 'export')
    
    def test_json_exporter_exists(self):
        """Test JSONExporter is functional."""
        from ubf_protein.api import JSONExporter
        
        exporter = JSONExporter()
        assert hasattr(exporter, 'export')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
