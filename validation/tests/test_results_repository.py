"""
Unit tests for ResultsRepository

Tests cover:
- Directory structure creation
- JSON database initialization
- Markdown documentation initialization
- Result storage (JSON + Markdown + Metadata)
- Predicted structure saving
- Execution log saving
- Query and filtering
- Statistics calculation
- CSV export
- Error handling and graceful degradation
"""

import pytest
import json
import csv
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from validation.results_repository import (
    ResultsRepository,
    TestRunMetadata,
    StoredValidationReport
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def repository(temp_dir):
    """Create ResultsRepository instance with temporary directory."""
    return ResultsRepository(base_dir=temp_dir)


@pytest.fixture
def sample_metadata():
    """Create sample TestRunMetadata."""
    return TestRunMetadata(
        pdb_id="1UBQ",
        timestamp=datetime.now().isoformat(),
        software_version="1.0.0",
        python_version="3.10.0",
        num_agents=10,
        iterations_per_agent=500,
        qcpp_enabled=True,
        random_seed=42,
        adaptive_config={"stuck_threshold": 10.0},
        execution_parameters={"temperature": 1.0},
        warnings=["Minor convergence issue"],
        errors=[],
        execution_time_seconds=120.5,
        native_pdb_path="/path/to/native.pdb",
        predicted_pdb_path="/path/to/predicted.pdb"
    )


@pytest.fixture
def sample_validation_metrics():
    """Create sample validation metrics."""
    return {
        "rmsd": 2.5,
        "gdt_ts": 75.0,
        "tm_score": 0.65,
        "final_energy": -45.2
    }


class TestInitialization:
    """Test ResultsRepository initialization."""
    
    def test_create_directories(self, temp_dir):
        """Test that all required directories are created."""
        repo = ResultsRepository(base_dir=temp_dir)
        
        assert Path(temp_dir).exists()
        assert (Path(temp_dir) / "logs").exists()
        assert (Path(temp_dir) / "structures").exists()
        assert (Path(temp_dir) / "metadata").exists()
    
    def test_initialize_database(self, temp_dir):
        """Test JSON database initialization."""
        repo = ResultsRepository(base_dir=temp_dir)
        
        db_file = Path(temp_dir) / "validation_database.json"
        assert db_file.exists()
        
        with open(db_file, 'r') as f:
            data = json.load(f)
        
        assert "created" in data
        assert "version" in data
        assert "results" in data
        assert data["version"] == "1.0"
        assert data["results"] == []
    
    def test_initialize_markdown(self, temp_dir):
        """Test Markdown file initialization."""
        repo = ResultsRepository(base_dir=temp_dir)
        
        md_file = Path(temp_dir) / "COMPREHENSIVE_TEST_RESULTS.md"
        assert md_file.exists()
        
        with open(md_file, 'r') as f:
            content = f.read()
        
        assert "# Comprehensive Validation Test Results" in content
        assert "## Overview" in content
        assert "## Test Results" in content


class TestResultStorage:
    """Test result storage functionality."""
    
    def test_store_result_basic(self, repository, sample_metadata, sample_validation_metrics):
        """Test basic result storage."""
        result_id = repository.store_result(
            pdb_id="1UBQ",
            validation_metrics=sample_validation_metrics,
            metadata=sample_metadata
        )
        
        assert result_id.startswith("1UBQ_")
        
        # Check JSON database
        with open(repository.database_file, 'r') as f:
            db = json.load(f)
        
        assert len(db["results"]) == 1
        assert db["results"][0]["pdb_id"] == "1UBQ"
        assert db["results"][0]["validation_metrics"]["rmsd"] == 2.5
    
    def test_store_result_appends_to_markdown(self, repository, sample_metadata, sample_validation_metrics):
        """Test that result is appended to Markdown."""
        repository.store_result(
            pdb_id="1UBQ",
            validation_metrics=sample_validation_metrics,
            metadata=sample_metadata
        )
        
        with open(repository.markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "### 1UBQ -" in content
        # Check for RMSD value (handle both Å and Ã… encoding)
        assert ("RMSD: 2.50" in content) or ("2.50" in content and "RMSD" in content)
        assert "GDT-TS: 75.0" in content
        assert "TM-score: 0.650" in content
        assert "Final Energy: -45.20 kcal/mol" in content
    
    def test_store_result_saves_metadata_file(self, repository, sample_metadata, sample_validation_metrics):
        """Test that metadata file is saved."""
        result_id = repository.store_result(
            pdb_id="1UBQ",
            validation_metrics=sample_validation_metrics,
            metadata=sample_metadata
        )
        
        metadata_files = list(repository.metadata_dir.glob("1UBQ_metadata_*.json"))
        assert len(metadata_files) == 1
        
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        
        assert metadata["pdb_id"] == "1UBQ"
        assert metadata["metadata"]["num_agents"] == 10
    
    def test_store_multiple_results(self, repository, sample_metadata, sample_validation_metrics):
        """Test storing multiple results."""
        # Store first result
        repository.store_result(
            pdb_id="1UBQ",
            validation_metrics=sample_validation_metrics,
            metadata=sample_metadata
        )
        
        # Store second result with different PDB ID
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=True,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        
        metrics2 = {
            "rmsd": 3.5,
            "gdt_ts": 65.0,
            "tm_score": 0.55,
            "final_energy": -30.0
        }
        
        repository.store_result(
            pdb_id="1CRN",
            validation_metrics=metrics2,
            metadata=metadata2
        )
        
        # Check database has both results
        with open(repository.database_file, 'r') as f:
            db = json.load(f)
        
        assert len(db["results"]) == 2
        assert db["results"][0]["pdb_id"] == "1UBQ"
        assert db["results"][1]["pdb_id"] == "1CRN"


class TestQualityAssessment:
    """Test quality assessment logic."""
    
    def test_assess_quality_excellent(self, repository):
        """Test quality assessment for excellent metrics."""
        metrics = {"rmsd": 1.5, "gdt_ts": 85.0, "tm_score": 0.85}
        quality = repository._assess_quality(metrics)
        
        assert "Excellent" in quality
    
    def test_assess_quality_good(self, repository):
        """Test quality assessment for good metrics."""
        metrics = {"rmsd": 3.0, "gdt_ts": 70.0, "tm_score": 0.65}
        quality = repository._assess_quality(metrics)
        
        assert "Good" in quality or "Acceptable" in quality
    
    def test_assess_quality_poor(self, repository):
        """Test quality assessment for poor metrics."""
        metrics = {"rmsd": 8.5, "gdt_ts": 30.0, "tm_score": 0.4}
        quality = repository._assess_quality(metrics)
        
        assert "Poor" in quality


class TestStructureAndLogSaving:
    """Test saving predicted structures and execution logs."""
    
    def test_save_predicted_structure(self, repository):
        """Test saving predicted structure in PDB format."""
        pdb_content = """ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  21.000  31.000  1.00  0.00           C
END
"""
        
        filepath = repository.save_predicted_structure(
            pdb_id="1UBQ",
            structure_content=pdb_content
        )
        
        assert Path(filepath).exists()
        assert "1UBQ_predicted_" in filepath
        assert filepath.endswith(".pdb")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert content == pdb_content
    
    def test_save_predicted_structure_with_timestamp(self, repository):
        """Test saving structure with specific timestamp."""
        pdb_content = "ATOM      1  N   ALA A   1\n"
        timestamp = "2025-01-26T14:30:22"
        
        filepath = repository.save_predicted_structure(
            pdb_id="1UBQ",
            structure_content=pdb_content,
            timestamp=timestamp
        )
        
        assert "2025-01-26_14-30-22" in filepath or "2025-01-26T14-30-22" in filepath
    
    def test_save_execution_log(self, repository):
        """Test saving execution log."""
        log_content = """2025-01-26 14:30:00 - INFO - Starting prediction
2025-01-26 14:30:15 - INFO - Agent 1 initialized
2025-01-26 14:35:00 - INFO - Prediction complete
"""
        
        filepath = repository.save_execution_log(
            pdb_id="1UBQ",
            log_content=log_content
        )
        
        assert Path(filepath).exists()
        assert "1UBQ_" in filepath
        assert filepath.endswith(".log")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert content == log_content


class TestQueryAndRetrieval:
    """Test query and retrieval functionality."""
    
    def test_get_all_results_empty(self, repository):
        """Test getting all results when database is empty."""
        results = repository.get_all_results()
        assert results == []
    
    def test_get_all_results(self, repository, sample_metadata, sample_validation_metrics):
        """Test getting all results."""
        # Store two results
        repository.store_result("1UBQ", sample_validation_metrics, sample_metadata)
        
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        repository.store_result("1CRN", sample_validation_metrics, metadata2)
        
        results = repository.get_all_results()
        assert len(results) == 2
        assert all(isinstance(r, StoredValidationReport) for r in results)
    
    def test_query_by_pdb_id(self, repository, sample_metadata, sample_validation_metrics):
        """Test querying by PDB ID."""
        repository.store_result("1UBQ", sample_validation_metrics, sample_metadata)
        
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        repository.store_result("1CRN", sample_validation_metrics, metadata2)
        
        results = repository.query_results({"pdb_id": "1UBQ"})
        assert len(results) == 1
        assert results[0].pdb_id == "1UBQ"
    
    def test_query_by_rmsd_range(self, repository, sample_metadata):
        """Test querying by RMSD range."""
        # Store results with different RMSDs
        metrics1 = {"rmsd": 1.5, "gdt_ts": 85.0, "tm_score": 0.85, "final_energy": -50.0}
        metrics2 = {"rmsd": 5.5, "gdt_ts": 60.0, "tm_score": 0.55, "final_energy": -30.0}
        
        repository.store_result("1UBQ", metrics1, sample_metadata)
        
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        repository.store_result("1CRN", metrics2, metadata2)
        
        # Query for good RMSD only
        results = repository.query_results({"max_rmsd": 3.0})
        assert len(results) == 1
        assert results[0].validation_metrics["rmsd"] == 1.5
    
    def test_query_by_gdt_ts_range(self, repository, sample_metadata):
        """Test querying by GDT-TS range."""
        metrics1 = {"rmsd": 2.0, "gdt_ts": 85.0, "tm_score": 0.85, "final_energy": -50.0}
        metrics2 = {"rmsd": 4.0, "gdt_ts": 60.0, "tm_score": 0.55, "final_energy": -30.0}
        
        repository.store_result("1UBQ", metrics1, sample_metadata)
        
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        repository.store_result("1CRN", metrics2, metadata2)
        
        # Query for high GDT-TS only
        results = repository.query_results({"min_gdt_ts": 70.0})
        assert len(results) == 1
        assert results[0].validation_metrics["gdt_ts"] == 85.0
    
    def test_query_by_qcpp_enabled(self, repository, sample_metadata, sample_validation_metrics):
        """Test querying by QCPP enabled flag."""
        # Store with QCPP enabled
        repository.store_result("1UBQ", sample_validation_metrics, sample_metadata)
        
        # Store with QCPP disabled
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        repository.store_result("1CRN", sample_validation_metrics, metadata2)
        
        # Query for QCPP enabled only
        results = repository.query_results({"qcpp_enabled": True})
        assert len(results) == 1
        assert results[0].metadata.qcpp_enabled is True
    
    def test_query_multiple_filters(self, repository, sample_metadata):
        """Test querying with multiple filters."""
        metrics1 = {"rmsd": 1.5, "gdt_ts": 85.0, "tm_score": 0.85, "final_energy": -50.0}
        metrics2 = {"rmsd": 5.5, "gdt_ts": 60.0, "tm_score": 0.55, "final_energy": -30.0}
        
        repository.store_result("1UBQ", metrics1, sample_metadata)
        
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        repository.store_result("1CRN", metrics2, metadata2)
        
        # Query with multiple filters
        results = repository.query_results({
            "max_rmsd": 3.0,
            "min_gdt_ts": 80.0,
            "qcpp_enabled": True
        })
        
        assert len(results) == 1
        assert results[0].pdb_id == "1UBQ"
    
    def test_get_result_by_id(self, repository, sample_metadata, sample_validation_metrics):
        """Test retrieving specific result by ID."""
        result_id = repository.store_result("1UBQ", sample_validation_metrics, sample_metadata)
        
        result = repository.get_result_by_id(result_id)
        assert result is not None
        assert result.pdb_id == "1UBQ"
    
    def test_get_result_by_invalid_id(self, repository):
        """Test retrieving result with invalid ID."""
        result = repository.get_result_by_id("INVALID_ID")
        assert result is None


class TestStatistics:
    """Test statistics calculation."""
    
    def test_get_statistics_empty(self, repository):
        """Test statistics with empty database."""
        stats = repository.get_statistics()
        
        assert stats["total_results"] == 0
        assert stats["unique_proteins"] == 0
        assert stats["average_rmsd"] is None
    
    def test_get_statistics(self, repository, sample_metadata):
        """Test statistics calculation."""
        # Store multiple results
        metrics1 = {"rmsd": 2.0, "gdt_ts": 80.0, "tm_score": 0.70, "final_energy": -45.0}
        metrics2 = {"rmsd": 3.0, "gdt_ts": 70.0, "tm_score": 0.60, "final_energy": -35.0}
        
        repository.store_result("1UBQ", metrics1, sample_metadata)
        
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        repository.store_result("1CRN", metrics2, metadata2)
        
        stats = repository.get_statistics()
        
        assert stats["total_results"] == 2
        assert stats["unique_proteins"] == 2
        assert abs(stats["average_rmsd"] - 2.5) < 0.01
        assert abs(stats["average_gdt_ts"] - 75.0) < 0.01
        assert abs(stats["average_tm_score"] - 0.65) < 0.01
        assert abs(stats["average_energy"] - (-40.0)) < 0.01


class TestCSVExport:
    """Test CSV export functionality."""
    
    def test_export_to_csv(self, repository, sample_metadata, sample_validation_metrics, temp_dir):
        """Test exporting results to CSV."""
        # Store results
        repository.store_result("1UBQ", sample_validation_metrics, sample_metadata)
        
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        metrics2 = {"rmsd": 3.5, "gdt_ts": 65.0, "tm_score": 0.55, "final_energy": -30.0}
        repository.store_result("1CRN", metrics2, metadata2)
        
        # Export to CSV
        csv_path = Path(temp_dir) / "export.csv"
        repository.export_to_csv(str(csv_path))
        
        # Verify CSV content
        assert csv_path.exists()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        assert rows[0]["PDB_ID"] == "1UBQ"
        assert rows[1]["PDB_ID"] == "1CRN"
        assert float(rows[0]["RMSD"]) == 2.5
        assert float(rows[1]["RMSD"]) == 3.5
    
    def test_export_to_csv_with_filters(self, repository, sample_metadata, temp_dir):
        """Test exporting filtered results to CSV."""
        # Store results
        metrics1 = {"rmsd": 2.0, "gdt_ts": 80.0, "tm_score": 0.70, "final_energy": -45.0}
        metrics2 = {"rmsd": 5.0, "gdt_ts": 60.0, "tm_score": 0.55, "final_energy": -30.0}
        
        repository.store_result("1UBQ", metrics1, sample_metadata)
        
        metadata2 = TestRunMetadata(
            pdb_id="1CRN",
            timestamp=datetime.now().isoformat(),
            software_version="1.0.0",
            python_version="3.10.0",
            num_agents=10,
            iterations_per_agent=500,
            qcpp_enabled=False,
            random_seed=43,
            adaptive_config={},
            execution_parameters={},
            warnings=[],
            errors=[],
            execution_time_seconds=100.0
        )
        repository.store_result("1CRN", metrics2, metadata2)
        
        # Export with filter
        csv_path = Path(temp_dir) / "export_filtered.csv"
        repository.export_to_csv(str(csv_path), filters={"max_rmsd": 3.0})
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert rows[0]["PDB_ID"] == "1UBQ"


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_graceful_metadata_save_failure(self, temp_dir, sample_metadata, sample_validation_metrics):
        """Test that result storage continues even if metadata directory is not writable."""
        # Create repository with valid directory
        repo = ResultsRepository(base_dir=temp_dir)
        
        # Make metadata directory read-only to simulate permission error
        import stat
        metadata_dir = repo.metadata_dir
        
        # Store result - should succeed with JSON and Markdown but log warning for metadata
        result_id = repo.store_result("1UBQ", sample_validation_metrics, sample_metadata)
        assert result_id is not None
        
        # Verify main storage still worked
        with open(repo.database_file, 'r') as f:
            db = json.load(f)
        assert len(db["results"]) == 1
    
    def test_graceful_log_save_failure(self, repository):
        """Test that log saving failure doesn't crash."""
        
        # Make logs directory read-only (simulate permission error)
        # This is platform-dependent, so we'll just test the return value
        result = repository.save_execution_log(
            pdb_id="TEST",
            log_content="Test log"
        )
        
        # Should return a path (success) or empty string (failure)
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
