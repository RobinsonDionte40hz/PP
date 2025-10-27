"""
Unit Tests for Quality Control Module

Tests comprehensive quality control checks and reproducibility features including:
- Native structure validation
- Output file validation
- Metadata recording
- Abnormal termination detection
- Reproducibility script generation
"""

import pytest
import json
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from validation.quality_control import (
    QualityCheckResult,
    ReproducibilityMetadata,
    QualityController
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_pdb_content():
    """Sample PDB file content with 10 CA atoms."""
    return """HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C
ATOM      2  CA  VAL A   2      13.800  10.000  10.000  1.00 20.00           C
ATOM      3  CA  GLY A   3      17.600  10.000  10.000  1.00 20.00           C
ATOM      4  CA  LEU A   4      21.400  10.000  10.000  1.00 20.00           C
ATOM      5  CA  SER A   5      25.200  10.000  10.000  1.00 20.00           C
ATOM      6  CA  THR A   6      29.000  10.000  10.000  1.00 20.00           C
ATOM      7  CA  PHE A   7      32.800  10.000  10.000  1.00 20.00           C
ATOM      8  CA  TYR A   8      36.600  10.000  10.000  1.00 20.00           C
ATOM      9  CA  TRP A   9      40.400  10.000  10.000  1.00 20.00           C
ATOM     10  CA  ASP A  10      44.200  10.000  10.000  1.00 20.00           C
END
"""


@pytest.fixture
def sample_config():
    """Sample test configuration."""
    return {
        'num_agents': 10,
        'iterations_per_agent': 1000,
        'qcpp_enabled': True,
        'random_seed': 42,
        'adaptive_config': {
            'stuck_window': 30,
            'stuck_threshold': 10.0
        },
        'timeout': 3600
    }


@pytest.fixture
def qc_controller():
    """Create QualityController instance."""
    return QualityController(
        strict_mode=False,
        capture_env_vars=False,
        validate_checksums=True
    )


# ============================================================================
# QualityCheckResult Tests
# ============================================================================

def test_quality_check_result_creation():
    """Test QualityCheckResult creation."""
    result = QualityCheckResult(
        check_name="test_check",
        passed=True,
        severity="info",
        message="Test passed",
        details={'key': 'value'}
    )
    
    assert result.check_name == "test_check"
    assert result.passed is True
    assert result.severity == "info"
    assert result.message == "Test passed"
    assert result.details == {'key': 'value'}
    assert isinstance(result.timestamp, datetime)


def test_quality_check_result_to_dict():
    """Test QualityCheckResult serialization."""
    result = QualityCheckResult(
        check_name="test_check",
        passed=False,
        severity="error",
        message="Test failed"
    )
    
    data = result.to_dict()
    
    assert data['check_name'] == "test_check"
    assert data['passed'] is False
    assert data['severity'] == "error"
    assert data['message'] == "Test failed"
    assert 'timestamp' in data
    assert isinstance(data['timestamp'], str)


# ============================================================================
# ReproducibilityMetadata Tests
# ============================================================================

def test_reproducibility_metadata_creation(sample_config):
    """Test ReproducibilityMetadata creation."""
    metadata = ReproducibilityMetadata(
        test_id="TEST_001",
        pdb_id="1UBQ",
        timestamp=datetime.now(),
        python_version="3.10.0",
        system_platform="Linux-5.4.0",
        ubf_version="1.0.0",
        dependencies={'pytest': '7.0.0'},
        num_agents=10,
        iterations_per_agent=1000,
        qcpp_enabled=True,
        random_seed=42,
        adaptive_config=sample_config['adaptive_config'],
        execution_parameters={'timeout': 3600},
        native_structure_checksum="abc123",
        predicted_structure_checksum=None,
        working_directory="/test",
        command_line="python test.py",
        environment_variables={}
    )
    
    assert metadata.test_id == "TEST_001"
    assert metadata.pdb_id == "1UBQ"
    assert metadata.num_agents == 10
    assert metadata.random_seed == 42
    assert metadata.completed_normally is True


def test_reproducibility_metadata_to_dict():
    """Test ReproducibilityMetadata serialization."""
    metadata = ReproducibilityMetadata(
        test_id="TEST_001",
        pdb_id="1UBQ",
        timestamp=datetime.now(),
        python_version="3.10.0",
        system_platform="Linux",
        ubf_version="1.0.0",
        dependencies={},
        num_agents=10,
        iterations_per_agent=1000,
        qcpp_enabled=True,
        random_seed=42,
        adaptive_config={},
        execution_parameters={},
        native_structure_checksum=None,
        predicted_structure_checksum=None,
        working_directory="/test",
        command_line="python test.py",
        environment_variables={}
    )
    
    data = metadata.to_dict()
    
    assert data['test_id'] == "TEST_001"
    assert data['pdb_id'] == "1UBQ"
    assert isinstance(data['timestamp'], str)
    assert data['num_agents'] == 10


def test_reproducibility_metadata_json_roundtrip(temp_dir):
    """Test ReproducibilityMetadata JSON save/load."""
    filepath = os.path.join(temp_dir, 'metadata.json')
    
    original = ReproducibilityMetadata(
        test_id="TEST_001",
        pdb_id="1UBQ",
        timestamp=datetime.now(),
        python_version="3.10.0",
        system_platform="Linux",
        ubf_version="1.0.0",
        dependencies={'pytest': '7.0.0'},
        num_agents=10,
        iterations_per_agent=1000,
        qcpp_enabled=True,
        random_seed=42,
        adaptive_config={'key': 'value'},
        execution_parameters={'timeout': 3600},
        native_structure_checksum="abc123",
        predicted_structure_checksum="def456",
        working_directory="/test",
        command_line="python test.py",
        environment_variables={}
    )
    
    # Save
    original.to_json(filepath)
    assert os.path.exists(filepath)
    
    # Load
    loaded = ReproducibilityMetadata.from_json(filepath)
    
    assert loaded.test_id == original.test_id
    assert loaded.pdb_id == original.pdb_id
    assert loaded.num_agents == original.num_agents
    assert loaded.random_seed == original.random_seed
    assert loaded.dependencies == original.dependencies


# ============================================================================
# QualityController Creation Tests
# ============================================================================

def test_quality_controller_creation():
    """Test QualityController creation with default parameters."""
    qc = QualityController()
    
    assert qc.strict_mode is False
    assert qc.capture_env_vars is False
    assert qc.validate_checksums is True


def test_quality_controller_strict_mode():
    """Test QualityController in strict mode."""
    qc = QualityController(strict_mode=True)
    
    assert qc.strict_mode is True


# ============================================================================
# Native Structure Validation Tests
# ============================================================================

def test_validate_native_structure_valid(temp_dir, sample_pdb_content, qc_controller):
    """Test native structure validation with valid PDB."""
    pdb_file = os.path.join(temp_dir, 'test.pdb')
    with open(pdb_file, 'w') as f:
        f.write(sample_pdb_content)
    
    result = qc_controller.validate_native_structure(pdb_file)
    
    assert result.passed is True
    assert result.severity == "info"
    assert "validated" in result.message.lower()
    assert result.details['ca_atoms'] == 10


def test_validate_native_structure_missing_file(qc_controller):
    """Test native structure validation with missing file."""
    result = qc_controller.validate_native_structure("nonexistent.pdb")
    
    assert result.passed is False
    assert result.severity == "critical"
    assert "not found" in result.message.lower()


def test_validate_native_structure_empty_file(temp_dir, qc_controller):
    """Test native structure validation with empty file."""
    pdb_file = os.path.join(temp_dir, 'empty.pdb')
    with open(pdb_file, 'w') as f:
        f.write("")
    
    result = qc_controller.validate_native_structure(pdb_file)
    
    assert result.passed is False
    assert result.severity == "error"
    assert "too small" in result.message.lower()


def test_validate_native_structure_no_atoms(temp_dir, qc_controller):
    """Test native structure validation with no ATOM records."""
    pdb_file = os.path.join(temp_dir, 'no_atoms.pdb')
    # Make it large enough to pass size check
    with open(pdb_file, 'w') as f:
        f.write("HEADER    TEST PROTEIN STRUCTURE\n" + " " * 100 + "\nEND\n")
    
    result = qc_controller.validate_native_structure(pdb_file)
    
    assert result.passed is False
    assert result.severity == "error"
    assert "no atom" in result.message.lower()


def test_validate_native_structure_few_atoms(temp_dir, qc_controller):
    """Test native structure validation with too few CA atoms."""
    pdb_file = os.path.join(temp_dir, 'few_atoms.pdb')
    with open(pdb_file, 'w') as f:
        f.write("""HEADER    TEST
ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C
ATOM      2  CA  VAL A   2      13.800  10.000  10.000  1.00 20.00           C
END
""")
    
    result = qc_controller.validate_native_structure(pdb_file)
    
    assert result.passed is False
    assert result.severity == "error"
    assert "too few" in result.message.lower()


def test_validate_native_structure_missing_residues(temp_dir, qc_controller):
    """Test native structure validation with excessive missing residues."""
    pdb_file = os.path.join(temp_dir, 'gaps.pdb')
    # Create with 12 CA atoms but large gaps
    content = """HEADER    TEST
ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C
ATOM      2  CA  VAL A   2      13.800  10.000  10.000  1.00 20.00           C
ATOM      3  CA  GLY A   3      17.600  10.000  10.000  1.00 20.00           C
ATOM      4  CA  LEU A  20      21.400  10.000  10.000  1.00 20.00           C
ATOM      5  CA  SER A  21      25.200  10.000  10.000  1.00 20.00           C
ATOM      6  CA  THR A  22      29.000  10.000  10.000  1.00 20.00           C
ATOM      7  CA  PHE A  40      32.800  10.000  10.000  1.00 20.00           C
ATOM      8  CA  TYR A  41      36.600  10.000  10.000  1.00 20.00           C
ATOM      9  CA  TRP A  60      40.400  10.000  10.000  1.00 20.00           C
ATOM     10  CA  ASP A  61      44.200  10.000  10.000  1.00 20.00           C
ATOM     11  CA  GLU A  80      48.000  10.000  10.000  1.00 20.00           C
ATOM     12  CA  LYS A  81      51.800  10.000  10.000  1.00 20.00           C
END
"""
    with open(pdb_file, 'w') as f:
        f.write(content)
    
    result = qc_controller.validate_native_structure(pdb_file)
    
    # Warning but still passes
    assert result.passed is True
    assert result.severity == "warning"
    assert "missing residues" in result.message.lower()


def test_validate_native_structure_wrong_extension(temp_dir, qc_controller):
    """Test native structure validation with wrong file extension."""
    txt_file = os.path.join(temp_dir, 'structure.txt')
    with open(txt_file, 'w') as f:
        f.write("ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n" * 20)
    
    result = qc_controller.validate_native_structure(txt_file)
    
    assert result.passed is False
    assert result.severity == "warning"
    assert "extension" in result.message.lower()


# ============================================================================
# Output File Validation Tests
# ============================================================================

def test_validate_output_files_all_valid(temp_dir, sample_pdb_content, qc_controller):
    """Test output file validation with all valid files."""
    predicted_pdb = os.path.join(temp_dir, 'predicted.pdb')
    results_json = os.path.join(temp_dir, 'results.json')
    log_file = os.path.join(temp_dir, 'test.log')
    
    # Create valid files
    with open(predicted_pdb, 'w') as f:
        f.write(sample_pdb_content)
    
    with open(results_json, 'w') as f:
        json.dump({
            'pdb_id': '1UBQ',
            'best_rmsd': 3.5,
            'best_energy': -50.0
        }, f)
    
    with open(log_file, 'w') as f:
        f.write("Starting test\nIteration 1\nAgent 0\nCompleted\n")
    
    result = qc_controller.validate_output_files(
        predicted_pdb=predicted_pdb,
        results_json=results_json,
        log_file=log_file
    )
    
    assert result.passed is True
    assert result.severity == "info"
    assert "validated successfully" in result.message.lower()


def test_validate_output_files_missing_predicted(temp_dir, qc_controller):
    """Test output file validation with missing predicted PDB."""
    result = qc_controller.validate_output_files(
        predicted_pdb=os.path.join(temp_dir, 'nonexistent.pdb')
    )
    
    assert result.passed is False
    assert "not found" in result.message.lower()


def test_validate_output_files_invalid_json(temp_dir, qc_controller):
    """Test output file validation with invalid JSON."""
    results_json = os.path.join(temp_dir, 'results.json')
    with open(results_json, 'w') as f:
        f.write("{ invalid json")
    
    result = qc_controller.validate_output_files(results_json=results_json)
    
    assert result.passed is False
    assert "invalid" in result.message.lower()


def test_validate_output_files_missing_keys(temp_dir, qc_controller):
    """Test output file validation with missing required keys in JSON."""
    results_json = os.path.join(temp_dir, 'results.json')
    with open(results_json, 'w') as f:
        json.dump({'pdb_id': '1UBQ'}, f)  # Missing best_rmsd, best_energy
    
    result = qc_controller.validate_output_files(results_json=results_json)
    
    assert result.passed is False
    assert "missing keys" in result.message.lower()


def test_validate_output_files_small_log(temp_dir, qc_controller):
    """Test output file validation with missing expected markers."""
    log_file = os.path.join(temp_dir, 'test.log')
    with open(log_file, 'w') as f:
        f.write("X" * 150)  # Content but no execution markers
    
    result = qc_controller.validate_output_files(log_file=log_file)
    
    assert result.passed is False
    assert "missing expected markers" in result.message.lower()


# ============================================================================
# Metadata Recording Tests
# ============================================================================

def test_record_test_metadata(sample_config, qc_controller):
    """Test test metadata recording."""
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config
    )
    
    assert metadata.pdb_id == "1UBQ"
    assert metadata.num_agents == 10
    assert metadata.iterations_per_agent == 1000
    assert metadata.qcpp_enabled is True
    assert metadata.random_seed == 42
    assert metadata.adaptive_config == sample_config['adaptive_config']
    assert metadata.python_version
    assert metadata.system_platform
    assert metadata.working_directory


def test_record_test_metadata_with_native(temp_dir, sample_pdb_content, sample_config, qc_controller):
    """Test metadata recording with native structure checksum."""
    native_pdb = os.path.join(temp_dir, 'native.pdb')
    with open(native_pdb, 'w') as f:
        f.write(sample_pdb_content)
    
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config,
        native_pdb=native_pdb
    )
    
    assert metadata.native_structure_checksum is not None
    assert len(metadata.native_structure_checksum) == 64  # SHA256 hex length


def test_record_test_metadata_custom_test_id(sample_config, qc_controller):
    """Test metadata recording with custom test ID."""
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config,
        test_id="CUSTOM_TEST_001"
    )
    
    assert metadata.test_id == "CUSTOM_TEST_001"


def test_update_predicted_checksum(temp_dir, sample_pdb_content, sample_config, qc_controller):
    """Test updating metadata with predicted structure checksum."""
    predicted_pdb = os.path.join(temp_dir, 'predicted.pdb')
    with open(predicted_pdb, 'w') as f:
        f.write(sample_pdb_content)
    
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config
    )
    
    assert metadata.predicted_structure_checksum is None
    
    qc_controller.update_predicted_checksum(metadata, predicted_pdb)
    
    assert metadata.predicted_structure_checksum is not None
    assert len(metadata.predicted_structure_checksum) == 64


# ============================================================================
# Abnormal Termination Detection Tests
# ============================================================================

def test_detect_normal_termination(temp_dir, qc_controller):
    """Test detection of normal termination."""
    log_file = os.path.join(temp_dir, 'normal.log')
    with open(log_file, 'w') as f:
        f.write("Starting test\nIteration 1\nAgent 0\nTest completed successfully\n")
    
    result = qc_controller.detect_abnormal_termination(log_file)
    
    assert result.passed is True
    assert result.severity == "info"
    assert "completed normally" in result.message.lower()


def test_detect_abnormal_termination_no_completion(temp_dir, qc_controller):
    """Test detection of abnormal termination without completion marker."""
    log_file = os.path.join(temp_dir, 'incomplete.log')
    with open(log_file, 'w') as f:
        f.write("Starting test\nIteration 1\nAgent 0\n")  # No completion
    
    result = qc_controller.detect_abnormal_termination(log_file)
    
    assert result.passed is False
    assert "did not complete" in result.message.lower()


def test_detect_abnormal_termination_with_error(temp_dir, qc_controller):
    """Test detection of abnormal termination with errors."""
    log_file = os.path.join(temp_dir, 'error.log')
    with open(log_file, 'w') as f:
        f.write("Starting test\nError: Something went wrong\nException: ValueError\n")
    
    result = qc_controller.detect_abnormal_termination(log_file)
    
    assert result.passed is False
    assert result.severity == "critical"
    assert "error" in result.message.lower()


def test_detect_abnormal_termination_with_timeout(temp_dir, qc_controller):
    """Test detection of timeout."""
    log_file = os.path.join(temp_dir, 'timeout.log')
    with open(log_file, 'w') as f:
        f.write("Starting test\nIteration 1\nTimeout: exceeded time limit\n")
    
    result = qc_controller.detect_abnormal_termination(log_file)
    
    assert result.passed is False
    assert "timeout" in result.message.lower()


def test_detect_abnormal_termination_missing_log(qc_controller):
    """Test detection with missing log file."""
    result = qc_controller.detect_abnormal_termination("nonexistent.log")
    
    assert result.passed is False
    assert result.severity == "error"
    assert "not found" in result.message.lower()


def test_detect_abnormal_termination_custom_markers(temp_dir, qc_controller):
    """Test detection with custom completion markers."""
    log_file = os.path.join(temp_dir, 'custom.log')
    with open(log_file, 'w') as f:
        f.write("Starting test\nTest PASSED\n")
    
    result = qc_controller.detect_abnormal_termination(
        log_file,
        expected_markers=['PASSED', 'SUCCESS']
    )
    
    assert result.passed is True


# ============================================================================
# Reproducibility Script Generation Tests
# ============================================================================

def test_generate_python_reproducibility_script(sample_config, qc_controller):
    """Test Python reproducibility script generation."""
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config,
        test_id="TEST_001"
    )
    
    script = qc_controller.generate_reproducibility_script(metadata, template="python")
    
    assert "#!/usr/bin/env python3" in script
    assert "TEST_001" in script
    assert "1UBQ" in script
    assert "NUM_AGENTS = 10" in script
    assert "ITERATIONS_PER_AGENT = 1000" in script
    assert "RANDOM_SEED = 42" in script
    assert "def validate_environment" in script
    assert "def run_test" in script


def test_generate_bash_reproducibility_script(sample_config, qc_controller):
    """Test Bash reproducibility script generation."""
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config,
        test_id="TEST_001"
    )
    
    script = qc_controller.generate_reproducibility_script(metadata, template="bash")
    
    assert "#!/bin/bash" in script
    assert "TEST_001" in script
    assert "1UBQ" in script


def test_generate_batch_reproducibility_script(sample_config, qc_controller):
    """Test Windows batch reproducibility script generation."""
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config,
        test_id="TEST_001"
    )
    
    script = qc_controller.generate_reproducibility_script(metadata, template="batch")
    
    assert "@echo off" in script
    assert "TEST_001" in script
    assert "1UBQ" in script


def test_generate_reproducibility_script_invalid_template(sample_config, qc_controller):
    """Test reproducibility script generation with invalid template."""
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config
    )
    
    with pytest.raises(ValueError, match="Unknown template"):
        qc_controller.generate_reproducibility_script(metadata, template="invalid")


def test_save_reproducibility_script(temp_dir, sample_config, qc_controller):
    """Test saving reproducibility script to file."""
    metadata = qc_controller.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config,
        test_id="TEST_001"
    )
    
    script = qc_controller.generate_reproducibility_script(metadata)
    output_path = os.path.join(temp_dir, "reproduce.py")
    
    qc_controller.save_reproducibility_script(script, output_path)
    
    assert os.path.exists(output_path)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        saved_script = f.read()
    
    assert saved_script == script


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_quality_control_workflow(temp_dir, sample_pdb_content, sample_config):
    """Test complete quality control workflow."""
    qc = QualityController(strict_mode=False, validate_checksums=True)
    
    # Setup files
    native_pdb = os.path.join(temp_dir, 'native.pdb')
    with open(native_pdb, 'w') as f:
        f.write(sample_pdb_content)
    
    # Pre-execution: Validate native structure
    native_check = qc.validate_native_structure(native_pdb)
    assert native_check.passed is True
    
    # Record metadata
    metadata = qc.record_test_metadata(
        pdb_id="1UBQ",
        config=sample_config,
        native_pdb=native_pdb,
        test_id="WORKFLOW_TEST"
    )
    metadata.quality_checks.append(native_check)
    
    # Simulate test execution
    predicted_pdb = os.path.join(temp_dir, 'predicted.pdb')
    results_json = os.path.join(temp_dir, 'results.json')
    log_file = os.path.join(temp_dir, 'test.log')
    
    with open(predicted_pdb, 'w') as f:
        f.write(sample_pdb_content)
    
    with open(results_json, 'w') as f:
        json.dump({'pdb_id': '1UBQ', 'best_rmsd': 2.5, 'best_energy': -60.0}, f)
    
    with open(log_file, 'w') as f:
        f.write("Starting\nIteration 100\nAgent 5\nCompleted successfully\n")
    
    # Post-execution: Update checksum and validate outputs
    qc.update_predicted_checksum(metadata, predicted_pdb)
    
    output_check = qc.validate_output_files(
        predicted_pdb=predicted_pdb,
        results_json=results_json,
        log_file=log_file
    )
    metadata.quality_checks.append(output_check)
    assert output_check.passed is True
    
    # Check for abnormal termination
    termination_check = qc.detect_abnormal_termination(log_file)
    metadata.quality_checks.append(termination_check)
    assert termination_check.passed is True
    
    # Generate reproducibility script
    script = qc.generate_reproducibility_script(metadata)
    script_path = os.path.join(temp_dir, 'reproduce.py')
    qc.save_reproducibility_script(script, script_path)
    
    # Save metadata
    metadata_path = os.path.join(temp_dir, 'metadata.json')
    metadata.to_json(metadata_path)
    
    # Verify all files exist
    assert os.path.exists(script_path)
    assert os.path.exists(metadata_path)
    
    # Verify metadata can be loaded
    loaded_metadata = ReproducibilityMetadata.from_json(metadata_path)
    assert loaded_metadata.test_id == "WORKFLOW_TEST"
    assert loaded_metadata.pdb_id == "1UBQ"
    assert len(loaded_metadata.quality_checks) == 3


def test_strict_mode_workflow(temp_dir, sample_pdb_content, sample_config):
    """Test workflow in strict mode with warnings."""
    qc = QualityController(strict_mode=True)
    
    # Create file with wrong extension
    txt_file = os.path.join(temp_dir, 'structure.txt')
    with open(txt_file, 'w') as f:
        f.write(sample_pdb_content)
    
    result = qc.validate_native_structure(txt_file)
    
    # In strict mode, warnings should be treated as failures
    assert result.passed is False
    assert result.severity == "warning"
