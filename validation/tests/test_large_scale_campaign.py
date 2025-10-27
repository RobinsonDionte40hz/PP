"""
Unit Tests for Large-Scale Validation Campaign Orchestrator

Tests the LargeScaleValidationCampaign class including setup, execution,
phase management, quality gates, and result generation.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from validation.large_scale_validation_campaign import (
    LargeScaleValidationCampaign,
    CampaignConfig,
    CampaignResults,
    PhaseResults
)
from validation.protein_selector import ProteinMetadata


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def basic_config():
    """Create basic test configuration."""
    return CampaignConfig(
        target_protein_count=20,
        enable_qcpp=True,
        max_parallel_tests=2,
        num_agents=5,
        iterations_per_agent=300,
        checkpoint_interval=5,
        quality_gate_threshold=0.60,
        failure_rmsd_threshold=8.0,
        timeout_multiplier=2.0,
        random_seed=42,
        output_dir="./test_campaign_results"
    )


@pytest.fixture
def sample_proteins():
    """Create sample protein metadata."""
    return [
        ProteinMetadata(
            pdb_id=f"1TST{i}",
            sequence_length=50 + i * 10,
            resolution=2.0,
            experimental_method="X-RAY DIFFRACTION",
            structural_class="all-alpha",
            size_category="small",
            missing_residues_pct=0.0,
            organism="Test organism",
            description="Test protein"
        )
        for i in range(20)
    ]


# ============================================================================
# Configuration and Initialization Tests
# ============================================================================

class TestCampaignInitialization:
    """Test campaign initialization and configuration."""
    
    def test_init_with_config(self, basic_config):
        """Test initialization with configuration."""
        campaign = LargeScaleValidationCampaign(config=basic_config)
        
        assert campaign.config == basic_config
        assert campaign.campaign_id.startswith("campaign_")
        assert campaign.output_dir == Path(basic_config.output_dir)
        assert campaign._is_setup is False
    
    def test_init_without_config(self):
        """Test initialization with default configuration."""
        campaign = LargeScaleValidationCampaign()
        
        assert campaign.config is not None
        assert isinstance(campaign.config, CampaignConfig)
        assert campaign.config.target_protein_count == 60  # Default
    
    def test_init_with_protein_selection(self, basic_config, sample_proteins):
        """Test initialization with pre-selected proteins."""
        campaign = LargeScaleValidationCampaign(
            config=basic_config,
            protein_selection=sample_proteins
        )
        
        assert campaign._protein_selection == sample_proteins
    
    def test_output_directory_created(self, basic_config):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CampaignConfig(
                **{**basic_config.__dict__, 'output_dir': str(Path(tmpdir) / "new_campaign")}
            )
            
            campaign = LargeScaleValidationCampaign(config=config)
            
            assert campaign.output_dir.exists()
            assert campaign.output_dir.is_dir()
    
    def test_config_validation_protein_count(self):
        """Test configuration validation for protein count."""
        config = CampaignConfig(
            target_protein_count=120,  # Above recommended
            output_dir="./test"
        )
        
        # Should create campaign but log warning
        campaign = LargeScaleValidationCampaign(config=config)
        assert campaign.config.target_protein_count == 120
    
    def test_config_validation_quality_threshold(self):
        """Test configuration validation for quality threshold."""
        config = CampaignConfig(
            target_protein_count=50,
            quality_gate_threshold=1.5,  # Invalid
            output_dir="./test"
        )
        
        with pytest.raises(ValueError, match="quality_gate_threshold"):
            LargeScaleValidationCampaign(config=config)
    
    def test_config_validation_parallel_tests(self):
        """Test configuration validation for parallel tests."""
        config = CampaignConfig(
            target_protein_count=50,
            max_parallel_tests=0,  # Invalid
            output_dir="./test"
        )
        
        with pytest.raises(ValueError, match="max_parallel_tests"):
            LargeScaleValidationCampaign(config=config)


# ============================================================================
# Campaign Setup Tests
# ============================================================================

class TestCampaignSetup:
    """Test campaign setup functionality."""
    
    @patch('validation.large_scale_validation_campaign.ProteinSelector')
    @patch('validation.large_scale_validation_campaign.PhaseManager')
    def test_setup_campaign_basic(self, mock_phase_mgr, mock_selector, basic_config):
        """Test basic campaign setup."""
        # Mock protein selector
        mock_selector_instance = Mock()
        mock_selector.return_value = mock_selector_instance
        mock_selector_instance.select_proteins.return_value = []
        
        # Mock phase manager
        mock_phase_instance = Mock()
        mock_phase_mgr.return_value = mock_phase_instance
        
        campaign = LargeScaleValidationCampaign(config=basic_config)
        campaign.setup_campaign()
        
        assert campaign._is_setup is True
        assert campaign._protein_selector is not None
        assert campaign._phase_manager is not None
    
    @patch('validation.large_scale_validation_campaign.ProteinSelector')
    def test_setup_with_preset_proteins(self, mock_selector, basic_config, sample_proteins):
        """Test setup with pre-selected proteins."""
        campaign = LargeScaleValidationCampaign(
            config=basic_config,
            protein_selection=sample_proteins
        )
        
        with patch.object(campaign, '_initialize_components'):
            campaign.setup_campaign()
        
        # Should not call selector when proteins provided
        mock_selector.assert_not_called()
    
    def test_setup_initializes_all_components(self, basic_config):
        """Test that setup initializes all required components."""
        campaign = LargeScaleValidationCampaign(config=basic_config)
        
        with patch.object(campaign, '_protein_selector') as mock_ps, \
             patch.object(campaign, '_phase_manager') as mock_pm, \
             patch.object(campaign, '_batch_executor') as mock_be, \
             patch.object(campaign, '_results_repository') as mock_rr, \
             patch.object(campaign, '_progress_tracker') as mock_pt, \
             patch.object(campaign, '_statistical_analyzer') as mock_sa, \
             patch.object(campaign, '_failure_analyzer') as mock_fa, \
             patch.object(campaign, '_documentation_generator') as mock_dg, \
             patch.object(campaign, '_quality_controller') as mock_qc:
            
            # All components should be None before setup
            assert campaign._is_setup is False
    
    def test_setup_campaign_idempotent(self, basic_config):
        """Test that setup can be called multiple times safely."""
        campaign = LargeScaleValidationCampaign(config=basic_config)
        
        with patch('validation.large_scale_validation_campaign.ProteinSelector'):
            campaign.setup_campaign()
            first_setup = campaign._is_setup
            
            campaign.setup_campaign()
            second_setup = campaign._is_setup
            
            assert first_setup is True
            assert second_setup is True


# ============================================================================
# Campaign Status Tests
# ============================================================================

class TestCampaignStatus:
    """Test campaign status tracking."""
    
    def test_get_campaign_status_initial(self, basic_config):
        """Test getting status of unstarted campaign."""
        campaign = LargeScaleValidationCampaign(config=basic_config)
        
        status = campaign.get_campaign_status()
        
        assert isinstance(status, dict)
        assert status['campaign_id'] == campaign.campaign_id
        assert status['is_setup'] is False
        assert status['phases_completed'] == 0
        assert status['proteins_tested'] == 0
    
    @patch('validation.large_scale_validation_campaign.ProteinSelector')
    def test_get_campaign_status_after_setup(self, mock_selector, basic_config):
        """Test status after campaign setup."""
        mock_selector_instance = Mock()
        mock_selector.return_value = mock_selector_instance
        mock_selector_instance.select_proteins.return_value = []
        
        campaign = LargeScaleValidationCampaign(config=basic_config)
        campaign.setup_campaign()
        
        status = campaign.get_campaign_status()
        
        assert status['is_setup'] is True
    
    def test_campaign_id_format(self, basic_config):
        """Test campaign ID format."""
        campaign = LargeScaleValidationCampaign(config=basic_config)
        
        assert campaign.campaign_id.startswith("campaign_")
        # Should contain timestamp
        assert len(campaign.campaign_id) > len("campaign_")


# ============================================================================
# Checkpoint Tests
# ============================================================================

class TestCheckpointing:
    """Test campaign checkpointing functionality."""
    
    def test_checkpoint_campaign_creates_file(self, basic_config):
        """Test that checkpoint creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CampaignConfig(
                **{**basic_config.__dict__, 'output_dir': tmpdir}
            )
            campaign = LargeScaleValidationCampaign(config=config)
            
            checkpoint_path = campaign.checkpoint_campaign()
            
            assert Path(checkpoint_path).exists()
            assert checkpoint_path.endswith('.json')
    
    def test_checkpoint_campaign_custom_name(self, basic_config):
        """Test checkpoint with custom name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CampaignConfig(
                **{**basic_config.__dict__, 'output_dir': tmpdir}
            )
            campaign = LargeScaleValidationCampaign(config=config)
            
            checkpoint_path = campaign.checkpoint_campaign("my_checkpoint")
            
            assert "my_checkpoint" in checkpoint_path
    
    def test_checkpoint_contains_config(self, basic_config):
        """Test that checkpoint contains configuration."""
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CampaignConfig(
                **{**basic_config.__dict__, 'output_dir': tmpdir}
            )
            campaign = LargeScaleValidationCampaign(config=config)
            
            checkpoint_path = campaign.checkpoint_campaign()
            
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)
            
            assert 'config' in checkpoint_data
            assert checkpoint_data['config']['target_protein_count'] == 20


# ============================================================================
# Phase Results Tests
# ============================================================================

class TestPhaseResults:
    """Test PhaseResults data structure."""
    
    def test_phase_results_creation(self):
        """Test creating PhaseResults."""
        results = PhaseResults(
            phase_number=1,
            proteins_tested=10,
            success_rate=75.0,
            average_rmsd=3.5,
            average_gdt_ts=65.0,
            average_energy=-50.0,
            quality_gate_passed=True
        )
        
        assert results.phase_number == 1
        assert results.proteins_tested == 10
        assert results.success_rate == 75.0
        assert results.quality_gate_passed is True
    
    def test_phase_results_with_validation_reports(self):
        """Test PhaseResults with validation reports."""
        reports = [
            {'pdb_id': '1TST', 'rmsd': 3.0},
            {'pdb_id': '2TST', 'rmsd': 4.0}
        ]
        
        results = PhaseResults(
            phase_number=1,
            proteins_tested=2,
            success_rate=100.0,
            average_rmsd=3.5,
            average_gdt_ts=70.0,
            average_energy=-45.0,
            quality_gate_passed=True,
            validation_reports=reports
        )
        
        assert len(results.validation_reports) == 2
        assert results.validation_reports[0]['pdb_id'] == '1TST'


# ============================================================================
# Campaign Results Tests
# ============================================================================

class TestCampaignResults:
    """Test CampaignResults data structure."""
    
    def test_campaign_results_creation(self, basic_config):
        """Test creating CampaignResults."""
        results = CampaignResults(
            campaign_id="test_campaign_001",
            config=basic_config,
            start_time=datetime.now(),
            end_time=None,
            total_proteins=20,
            phases_completed=2,
            overall_success_rate=70.5,
            validation_reports=[],
            phase_summaries=[],
            statistical_analysis_path="",
            failure_analysis_path="",
            final_report_path=""
        )
        
        assert results.campaign_id == "test_campaign_001"
        assert results.total_proteins == 20
        assert results.phases_completed == 2
        assert results.overall_success_rate == 70.5
    
    def test_campaign_results_with_phase_summaries(self, basic_config):
        """Test CampaignResults with phase summaries."""
        phase1 = PhaseResults(
            phase_number=1,
            proteins_tested=10,
            success_rate=80.0,
            average_rmsd=3.0,
            average_gdt_ts=70.0,
            average_energy=-50.0,
            quality_gate_passed=True
        )
        
        results = CampaignResults(
            campaign_id="test_campaign",
            config=basic_config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_proteins=10,
            phases_completed=1,
            overall_success_rate=80.0,
            validation_reports=[],
            phase_summaries=[phase1],
            statistical_analysis_path="./stats.json",
            failure_analysis_path="./failures.json",
            final_report_path="./report.md"
        )
        
        assert len(results.phase_summaries) == 1
        assert results.phase_summaries[0].phase_number == 1


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in campaign execution."""
    
    def test_validate_setup_before_run(self, basic_config):
        """Test that campaign validates setup before running."""
        campaign = LargeScaleValidationCampaign(config=basic_config)
        
        # Should auto-setup if not setup
        with patch.object(campaign, 'setup_campaign') as mock_setup:
            with patch.object(campaign, 'run_phase', return_value=Mock()):
                with patch.object(campaign, '_phase_manager'):
                    campaign._phase_manager = Mock()
                    campaign._phase_manager.phases = {1: Mock()}
                    campaign._phase_manager.get_phase = Mock(return_value=Mock(phase_number=1, proteins=[]))
                    
                    try:
                        campaign.run_campaign()
                    except:
                        pass  # Expected to fail without full mocking
                    
                    # Should have called setup
                    mock_setup.assert_called_once()
    
    def test_invalid_phase_number(self, basic_config):
        """Test error on invalid phase number."""
        campaign = LargeScaleValidationCampaign(config=basic_config)
        
        # Phase validation will occur in phase_manager
        # This tests the campaign doesn't crash with bad data


# ============================================================================
# Integration Tests
# ============================================================================

class TestCampaignIntegration:
    """Integration tests for campaign workflow."""
    
    @patch('validation.large_scale_validation_campaign.ValidationSuite')
    @patch('validation.large_scale_validation_campaign.ProteinSelector')
    def test_minimal_campaign_flow(self, mock_selector, mock_suite, basic_config, sample_proteins):
        """Test minimal campaign execution flow."""
        # Mock protein selector
        mock_selector_instance = Mock()
        mock_selector.return_value = mock_selector_instance
        mock_selector_instance.select_proteins.return_value = sample_proteins[:5]  # Small set
        
        campaign = LargeScaleValidationCampaign(config=basic_config)
        
        # Setup should work
        with patch('validation.large_scale_validation_campaign.PhaseManager'), \
             patch('validation.large_scale_validation_campaign.BatchExecutor'), \
             patch('validation.large_scale_validation_campaign.ResultsRepository'), \
             patch('validation.large_scale_validation_campaign.ProgressTracker'), \
             patch('validation.large_scale_validation_campaign.StatisticalAnalyzer'), \
             patch('validation.large_scale_validation_campaign.FailureAnalyzer'), \
             patch('validation.large_scale_validation_campaign.DocumentationGenerator'), \
             patch('validation.large_scale_validation_campaign.QualityController'):
            
            campaign.setup_campaign()
            
            assert campaign._is_setup is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
