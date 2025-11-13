"""
Unit tests for PP integration layer.
"""

import pytest
from pathlib import Path
from app.integrations.config_mapper import ConfigMapper
from app.integrations.file_manager import FileManager
from app.integrations.result_parser import ResultParser

class TestConfigMapper:
    """Tests for ConfigMapper."""
    
    def test_get_preset_balanced(self):
        """Test getting balanced preset."""
        config = ConfigMapper.get_preset("balanced")
        assert config["iterations"] == 1000
        assert config["agents"] == 10
        assert config["enable_qcpp"] is True
    
    def test_get_preset_fast(self):
        """Test getting fast preset."""
        config = ConfigMapper.get_preset("fast")
        assert config["iterations"] == 500
        assert config["agents"] == 5
        assert config["enable_qcpp"] is False
    
    def test_get_preset_unknown(self):
        """Test getting unknown preset defaults to balanced."""
        config = ConfigMapper.get_preset("unknown")
        assert config["iterations"] == 1000  # balanced default
    
    def test_map_api_to_pp_config(self):
        """Test mapping API config to PP config."""
        api_config = {
            "preset": "fast",
            "iterations": 750,
            "agents": 8,
        }
        pp_config = ConfigMapper.map_api_to_pp_config(api_config)
        assert pp_config["iterations"] == 750  # Override
        assert pp_config["agents"] == 8  # Override
    
    def test_validate_config_valid(self):
        """Test validating valid config."""
        config = {"iterations": 1000, "agents": 10}
        is_valid, error = ConfigMapper.validate_config(config)
        assert is_valid is True
        assert error is None
    
    def test_validate_config_invalid_iterations(self):
        """Test validating invalid iterations."""
        config = {"iterations": 20000, "agents": 10}
        is_valid, error = ConfigMapper.validate_config(config)
        assert is_valid is False
        assert error is not None and "Iterations" in error
    
    def test_validate_config_invalid_agents(self):
        """Test validating invalid agents."""
        config = {"iterations": 1000, "agents": 200}
        is_valid, error = ConfigMapper.validate_config(config)
        assert is_valid is False
        assert error is not None and "Agents" in error


class TestResultParser:
    """Tests for ResultParser."""
    
    def test_extract_metrics_with_all_fields(self):
        """Test extracting metrics with all fields present."""
        result_data = {
            "final_rmsd": 2.5,
            "final_energy": -120.5,
            "gdt_ts": 75.0,
            "tm_score": 0.85,
            "iterations": 1000,
        }
        metrics = ResultParser.extract_metrics(result_data)
        assert metrics["rmsd"] == 2.5
        assert metrics["energy"] == -120.5
        assert metrics["gdt_ts"] == 75.0
        assert metrics["tm_score"] == 0.85
        assert metrics["iterations"] == 1000
    
    def test_extract_metrics_with_partial_fields(self):
        """Test extracting metrics with partial fields."""
        result_data = {
            "final_rmsd": 2.5,
            "final_energy": -120.5,
        }
        metrics = ResultParser.extract_metrics(result_data)
        assert metrics["rmsd"] == 2.5
        assert metrics["energy"] == -120.5
        assert "gdt_ts" not in metrics
    
    def test_extract_metrics_with_empty_data(self):
        """Test extracting metrics with empty data."""
        result_data = {}
        metrics = ResultParser.extract_metrics(result_data)
        assert len(metrics) == 0


class TestFileManager:
    """Tests for FileManager."""
    
    def test_ensure_directories(self):
        """Test that directories are created."""
        fm = FileManager()
        assert fm.results_dir.exists()
        assert fm.checkpoints_dir.exists()
        assert fm.pdb_cache_dir.exists()
    
    def test_get_disk_usage(self):
        """Test getting disk usage statistics."""
        fm = FileManager()
        usage = fm.get_disk_usage()
        assert "results" in usage
        assert "checkpoints" in usage
        assert "pdb_cache" in usage
        assert "size_bytes" in usage["results"]
        assert "file_count" in usage["results"]
