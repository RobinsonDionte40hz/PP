"""
Unit tests for CLI functionality (Task 12).

Tests cover:
- Argument parsing
- Configuration loading and creation
- Logging setup
- Interactive mode execution
- Batch mode execution
- Resume functionality
- Benchmark mode
- Error handling and validation
"""

import pytest
import tempfile
import json
import argparse
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

# Import components to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_validation_campaign import (
    setup_logging,
    load_config_from_file,
    create_config_from_args,
    save_config_to_file,
    run_interactive_mode,
    run_batch_mode,
    run_resume_mode,
    run_benchmark_mode,
    create_argument_parser,
    main
)
from large_scale_validation_campaign import CampaignConfig, CampaignResults


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "total_proteins": 50,
        "num_agents": 10,
        "iterations_per_agent": 1000,
        "max_parallel_proteins": 2,
        "enable_qcpp_integration": True,
        "qcpp_update_frequency": 1,
        "output_dir": "./test_results"
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict):
    """Create temporary configuration file."""
    config_path = Path(temp_dir) / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config_dict, f)
    return str(config_path)


@pytest.fixture
def mock_campaign():
    """Mock LargeScaleValidationCampaign."""
    campaign = Mock()
    campaign.config = CampaignConfig(
        target_protein_count=10,
        num_agents=5,
        iterations_per_agent=500
    )
    campaign.results_dir = Path("./test_results")
    return campaign


@pytest.fixture
def mock_campaign_results():
    """Mock CampaignResults."""
    from datetime import datetime
    return CampaignResults(
        campaign_id="test_123",
        config=CampaignConfig(),
        start_time=datetime(2024, 1, 15, 10, 0, 0),
        end_time=datetime(2024, 1, 15, 12, 0, 0),
        total_proteins=10,
        phases_completed=3,
        overall_success_rate=80.0,
        validation_reports=[],
        phase_summaries=[]
    )


# ============================================================================
# Test Logging Setup
# ============================================================================

class TestLoggingSetup:
    """Tests for logging configuration."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()
        
        logger = logging.getLogger()
        assert logger.level <= logging.DEBUG
        assert len(logger.handlers) > 0
    
    def test_setup_logging_with_level(self):
        """Test logging setup with custom level."""
        setup_logging(log_level="WARNING")
        
        # Verify console handler has correct level
        logger = logging.getLogger()
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert any(h.level == logging.WARNING for h in console_handlers)
    
    def test_setup_logging_with_file(self, temp_dir):
        """Test logging setup with file output."""
        log_file = str(Path(temp_dir) / "test.log")
        setup_logging(log_level="INFO", log_file=log_file)
        
        # Log a message
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        # Verify file was created
        assert Path(log_file).exists()
        
        # Verify content
        with open(log_file) as f:
            content = f.read()
        assert "Test message" in content


# ============================================================================
# Test Configuration Management
# ============================================================================

class TestConfigurationLoading:
    """Tests for configuration loading."""
    
    def test_load_config_from_file(self, sample_config_file, sample_config_dict):
        """Test loading configuration from JSON file."""
        config = load_config_from_file(sample_config_file)
        
        assert isinstance(config, dict)
        assert config["total_proteins"] == sample_config_dict["total_proteins"]
        assert config["num_agents"] == sample_config_dict["num_agents"]
    
    def test_load_config_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_file("nonexistent.json")
    
    def test_load_config_invalid_json(self, temp_dir):
        """Test loading invalid JSON file."""
        bad_file = Path(temp_dir) / "bad.json"
        bad_file.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            load_config_from_file(str(bad_file))


class TestConfigurationCreation:
    """Tests for configuration creation from arguments."""
    
    def test_create_config_from_args_minimal(self):
        """Test creating config with minimal arguments."""
        args = argparse.Namespace(
            proteins=50,
            agents=10,
            iterations=1000,
            parallel=2,
            qcpp=True,
            qcpp_freq=1,
            qcpp_cache=1000,
            output="./results",
            checkpoint_interval=10,
            phases=3,
            proteins_per_phase=None
        )
        
        config = create_config_from_args(args)
        
        assert isinstance(config, CampaignConfig)
        assert config.target_protein_count == 50
        assert config.num_agents == 10
        assert config.iterations_per_agent == 1000
    
    def test_create_config_from_args_full(self):
        """Test creating config with all arguments."""
        args = argparse.Namespace(
            proteins=100,
            agents=20,
            iterations=2000,
            parallel=4,
            qcpp=True,
            qcpp_freq=5,
            qcpp_cache=5000,
            output="./custom_results",
            checkpoint_interval=20,
            phases=5,
            proteins_per_phase=[20, 20, 20, 20, 20]
        )
        
        config = create_config_from_args(args)
        
        assert config.target_protein_count == 100
        assert config.max_parallel_tests == 4
        assert config.checkpoint_interval == 20
        # Note: num_phases is not a config attribute, phases are managed by campaign


class TestConfigurationSaving:
    """Tests for saving configuration to file."""
    
    def test_save_config_to_file(self, temp_dir):
        """Test saving configuration to JSON file."""
        config = CampaignConfig(
            target_protein_count=30,
            num_agents=8,
            iterations_per_agent=800
        )
        
        output_path = str(Path(temp_dir) / "saved_config.json")
        save_config_to_file(config, output_path)
        
        assert Path(output_path).exists()
        
        # Verify content
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["target_protein_count"] == 30
        assert data["num_agents"] == 8


# ============================================================================
# Test Interactive Mode
# ============================================================================

class TestInteractiveMode:
    """Tests for interactive mode execution."""
    
    @patch('builtins.input', side_effect=['y', 'y', 'y'])
    @patch('validation.run_validation_campaign.time.sleep')
    def test_run_interactive_mode_all_approved(self, mock_sleep, mock_input, 
                                              mock_campaign, mock_campaign_results):
        """Test interactive mode with all phases approved."""
        mock_campaign.run_phase.return_value = Mock(
            phase_number=1,
            proteins_tested=10,
            success_rate=80.0,
            quality_gate_passed=True
        )
        mock_campaign.get_campaign_results.return_value = mock_campaign_results
        
        results = run_interactive_mode(mock_campaign)
        
        assert isinstance(results, CampaignResults)
        assert mock_input.call_count >= 1
    
    @patch('builtins.input', side_effect=['y', 'n'])
    @patch('validation.run_validation_campaign.time.sleep')
    def test_run_interactive_mode_early_stop(self, mock_sleep, mock_input, 
                                            mock_campaign, mock_campaign_results):
        """Test interactive mode with early stop."""
        mock_campaign.run_phase.return_value = Mock(
            phase_number=1,
            proteins_tested=10,
            success_rate=80.0,
            quality_gate_passed=True
        )
        mock_campaign.get_campaign_results.return_value = mock_campaign_results
        
        results = run_interactive_mode(mock_campaign)
        
        assert isinstance(results, CampaignResults)
        # Should stop after 'n' response
        assert mock_input.call_count >= 2
    
    @patch('builtins.input', side_effect=['skip', 'y', 'y'])
    @patch('validation.run_validation_campaign.time.sleep')
    def test_run_interactive_mode_skip_phase(self, mock_sleep, mock_input, 
                                            mock_campaign, mock_campaign_results):
        """Test interactive mode with phase skip."""
        mock_campaign.run_phase.return_value = Mock(
            phase_number=1,
            proteins_tested=10,
            success_rate=80.0,
            quality_gate_passed=True
        )
        mock_campaign.get_campaign_results.return_value = mock_campaign_results
        
        results = run_interactive_mode(mock_campaign)
        
        assert isinstance(results, CampaignResults)


# ============================================================================
# Test Batch Mode
# ============================================================================

class TestBatchMode:
    """Tests for batch mode execution."""
    
    @patch('validation.run_validation_campaign.time.sleep')
    def test_run_batch_mode_success(self, mock_sleep, mock_campaign, 
                                    mock_campaign_results):
        """Test successful batch mode execution."""
        mock_campaign.run_campaign.return_value = mock_campaign_results
        
        results = run_batch_mode(mock_campaign)
        
        assert isinstance(results, CampaignResults)
        mock_campaign.run_campaign.assert_called_once()
    
    @patch('validation.run_validation_campaign.time.sleep')
    def test_run_batch_mode_with_error(self, mock_sleep, mock_campaign):
        """Test batch mode with execution error."""
        mock_campaign.run_campaign.side_effect = RuntimeError("Campaign failed")
        
        with pytest.raises(RuntimeError):
            run_batch_mode(mock_campaign)


# ============================================================================
# Test Resume Mode
# ============================================================================

class TestResumeMode:
    """Tests for resume functionality."""
    
    def test_run_resume_mode_nonexistent_checkpoint(self):
        """Test resume with nonexistent checkpoint."""
        with pytest.raises((FileNotFoundError, ValueError)):
            run_resume_mode("nonexistent_checkpoint.json")
    
    @patch('validation.large_scale_validation_campaign.LargeScaleValidationCampaign.resume_from_checkpoint')
    def test_run_resume_mode_batch(self, mock_resume, temp_dir, 
                                   mock_campaign_results):
        """Test resume in batch mode."""
        # Create mock checkpoint
        checkpoint_path = Path(temp_dir) / "checkpoint.json"
        checkpoint_data = {
            "campaign_id": "test_123",
            "config": {"total_proteins": 10},
            "current_phase": 1,
            "completed_phases": []
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        mock_campaign = Mock()
        mock_campaign.run_campaign.return_value = mock_campaign_results
        mock_resume.return_value = mock_campaign
        
        results = run_resume_mode(str(checkpoint_path), interactive=False)
        
        assert isinstance(results, CampaignResults)
    
    @patch('validation.large_scale_validation_campaign.LargeScaleValidationCampaign.resume_from_checkpoint')
    @patch('builtins.input', return_value='y')
    def test_run_resume_mode_interactive(self, mock_input, mock_resume, 
                                        temp_dir, mock_campaign_results):
        """Test resume in interactive mode."""
        checkpoint_path = Path(temp_dir) / "checkpoint.json"
        checkpoint_data = {
            "campaign_id": "test_123",
            "config": {"total_proteins": 10},
            "current_phase": 1
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        
        mock_campaign = Mock()
        mock_campaign.run_phase.return_value = Mock(quality_gate_passed=True)
        mock_campaign.get_campaign_results.return_value = mock_campaign_results
        mock_resume.return_value = mock_campaign
        
        results = run_resume_mode(str(checkpoint_path), interactive=True)
        
        assert isinstance(results, CampaignResults)


# ============================================================================
# Test Benchmark Mode
# ============================================================================

class TestBenchmarkMode:
    """Tests for benchmark mode execution."""
    
    @patch('validation.comparative_benchmarking.ComparativeBenchmark.run_benchmark')
    @patch('validation.protein_selector.ProteinSelector.select_proteins')
    def test_run_benchmark_mode(self, mock_select, mock_run_benchmark, temp_dir):
        """Test benchmark mode execution."""
        args = argparse.Namespace(
            proteins=20,
            agents=10,
            iterations=1000,
            parallel=2,
            output=temp_dir,
            log_level="INFO",
            log_file=None
        )
        
        # Mock protein selection
        mock_select.return_value = [Mock(pdb_id=f"1TST{i}") for i in range(20)]
        
        # Mock benchmark results
        from validation.comparative_benchmarking import BenchmarkReport
        mock_report = Mock(spec=BenchmarkReport)
        mock_report.benchmark_id = "bench_001"
        mock_run_benchmark.return_value = mock_report
        
        # Should not raise
        run_benchmark_mode(args)
        
        mock_select.assert_called_once()
        mock_run_benchmark.assert_called_once()


# ============================================================================
# Test Argument Parser
# ============================================================================

class TestArgumentParser:
    """Tests for argument parsing."""
    
    def test_create_argument_parser(self):
        """Test argument parser creation."""
        parser = create_argument_parser()
        
        assert isinstance(parser, argparse.ArgumentParser)
        
        # Test parsing minimal interactive command
        args = parser.parse_args(['--interactive'])
        assert args.interactive is True
        assert args.batch is False
    
    def test_parse_batch_mode_args(self):
        """Test parsing batch mode arguments."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            '--batch',
            '--proteins', '50',
            '--agents', '10',
            '--iterations', '1000'
        ])
        
        assert args.batch is True
        assert args.proteins == 50
        assert args.agents == 10
        assert args.iterations == 1000
    
    def test_parse_resume_args(self):
        """Test parsing resume arguments."""
        parser = create_argument_parser()
        
        args = parser.parse_args(['--resume', './checkpoint.json'])
        
        assert args.resume == './checkpoint.json'
    
    def test_parse_benchmark_args(self):
        """Test parsing benchmark arguments."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            '--benchmark',
            '--proteins', '30',
            '--output', './benchmark_results'
        ])
        
        assert args.benchmark is True
        assert args.proteins == 30
        assert args.output == './benchmark_results'
    
    def test_parse_qcpp_args(self):
        """Test parsing QCPP integration arguments."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            '--batch',
            '--qcpp',
            '--qcpp-freq', '5',
            '--qcpp-cache', '5000'
        ])
        
        assert args.qcpp is True
        assert args.qcpp_freq == 5
        assert args.qcpp_cache == 5000
    
    def test_parse_config_file_arg(self):
        """Test parsing config file argument."""
        parser = create_argument_parser()
        
        args = parser.parse_args(['--config', 'my_config.json', '--batch'])
        
        assert args.config == 'my_config.json'
        assert args.batch is True


# ============================================================================
# Test Main Function
# ============================================================================

class TestMainFunction:
    """Tests for main CLI entry point."""
    
    @patch('validation.run_validation_campaign.run_batch_mode')
    @patch('validation.large_scale_validation_campaign.LargeScaleValidationCampaign')
    @patch('sys.argv', ['prog', '--batch', '--proteins', '10'])
    def test_main_batch_mode(self, mock_campaign_class, mock_run_batch, 
                            mock_campaign_results):
        """Test main function in batch mode."""
        mock_campaign = Mock()
        mock_campaign_class.return_value = mock_campaign
        mock_run_batch.return_value = mock_campaign_results
        
        exit_code = main()
        
        assert exit_code == 0
        mock_run_batch.assert_called_once_with(mock_campaign)
    
    @patch('validation.run_validation_campaign.run_interactive_mode')
    @patch('validation.large_scale_validation_campaign.LargeScaleValidationCampaign')
    @patch('sys.argv', ['prog', '--interactive'])
    def test_main_interactive_mode(self, mock_campaign_class, mock_run_interactive, 
                                   mock_campaign_results):
        """Test main function in interactive mode."""
        mock_campaign = Mock()
        mock_campaign_class.return_value = mock_campaign
        mock_run_interactive.return_value = mock_campaign_results
        
        exit_code = main()
        
        assert exit_code == 0
        mock_run_interactive.assert_called_once_with(mock_campaign)
    
    @patch('validation.run_validation_campaign.run_resume_mode')
    @patch('sys.argv', ['prog', '--resume', './checkpoint.json'])
    def test_main_resume_mode(self, mock_run_resume, mock_campaign_results):
        """Test main function in resume mode."""
        mock_run_resume.return_value = mock_campaign_results
        
        exit_code = main()
        
        assert exit_code == 0
        mock_run_resume.assert_called_once()
    
    @patch('validation.run_validation_campaign.run_benchmark_mode')
    @patch('sys.argv', ['prog', '--benchmark', '--proteins', '20'])
    def test_main_benchmark_mode(self, mock_run_benchmark):
        """Test main function in benchmark mode."""
        mock_run_benchmark.return_value = None
        
        exit_code = main()
        
        assert exit_code == 0
        mock_run_benchmark.assert_called_once()
    
    @patch('sys.argv', ['prog'])
    def test_main_no_mode_specified(self):
        """Test main function with no mode specified."""
        # Should show help and exit with non-zero
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code != 0
    
    @patch('validation.run_validation_campaign.run_batch_mode')
    @patch('validation.large_scale_validation_campaign.LargeScaleValidationCampaign')
    @patch('sys.argv', ['prog', '--batch', '--proteins', '10'])
    def test_main_with_error(self, mock_campaign_class, mock_run_batch):
        """Test main function with execution error."""
        mock_campaign_class.return_value = Mock()
        mock_run_batch.side_effect = RuntimeError("Execution failed")
        
        exit_code = main()
        
        assert exit_code == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestCLIIntegration:
    """Integration tests for complete CLI workflows."""
    
    @patch('validation.large_scale_validation_campaign.LargeScaleValidationCampaign')
    def test_full_batch_workflow_with_config_file(self, mock_campaign_class, 
                                                  temp_dir, sample_config_file,
                                                  mock_campaign_results):
        """Test complete batch workflow with config file."""
        parser = create_argument_parser()
        args = parser.parse_args([
            '--config', sample_config_file,
            '--batch'
        ])
        
        # Load config
        config_dict = load_config_from_file(sample_config_file)
        assert isinstance(config_dict, dict)
        
        # Create campaign mock
        mock_campaign = Mock()
        mock_campaign.run_campaign.return_value = mock_campaign_results
        mock_campaign_class.return_value = mock_campaign
        
        # Run batch mode
        results = run_batch_mode(mock_campaign)
        
        assert isinstance(results, CampaignResults)
    
    def test_config_save_and_load_roundtrip(self, temp_dir):
        """Test saving and loading configuration."""
        # Create config
        original_config = CampaignConfig(
            target_protein_count=100,
            num_agents=20,
            iterations_per_agent=2000
        )
        
        # Save to file
        config_path = str(Path(temp_dir) / "roundtrip_config.json")
        save_config_to_file(original_config, config_path)
        
        # Load back
        loaded_dict = load_config_from_file(config_path)
        
        # Verify key values
        assert loaded_dict["target_protein_count"] == 100
        assert loaded_dict["num_agents"] == 20
        assert loaded_dict["iterations_per_agent"] == 2000
