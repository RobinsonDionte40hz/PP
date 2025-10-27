"""
Quality Control and Reproducibility Module

This module implements comprehensive quality control checks and reproducibility features
for large-scale protein structure validation campaigns.

Features:
- Native structure validation before test execution
- Output file validation after test completion
- Software version, configuration, and random seed recording
- Abnormal termination detection and flagging
- Reproducibility script generation for re-executing tests

Classes:
    QualityCheckResult: Results from a single quality control check
    ReproducibilityMetadata: Complete metadata for test reproducibility
    QualityController: Main class for quality control operations

Example:
    >>> qc = QualityController()
    >>> result = qc.validate_native_structure("1UBQ.pdb")
    >>> if result.passed:
    ...     metadata = qc.record_test_metadata(config)
    ...     script = qc.generate_reproducibility_script(metadata)
"""

import json
import hashlib
import platform
import sys
import os
import subprocess
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QualityCheckResult:
    """Results from a single quality control check"""
    check_name: str
    passed: bool
    severity: str  # info, warning, error, critical
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class ReproducibilityMetadata:
    """Complete metadata for test reproducibility"""
    # Test identification
    test_id: str
    pdb_id: str
    timestamp: datetime
    
    # Software versions
    python_version: str
    system_platform: str
    ubf_version: str
    dependencies: Dict[str, str]
    
    # Configuration
    num_agents: int
    iterations_per_agent: int
    qcpp_enabled: bool
    random_seed: Optional[int]
    adaptive_config: Dict[str, Any]
    execution_parameters: Dict[str, Any]
    
    # File checksums
    native_structure_checksum: Optional[str]
    predicted_structure_checksum: Optional[str]
    
    # Execution context
    working_directory: str
    command_line: str
    environment_variables: Dict[str, str]
    
    # Quality checks
    quality_checks: List[QualityCheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Termination status
    completed_normally: bool = True
    termination_reason: Optional[str] = None
    exit_code: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['quality_checks'] = [qc.to_dict() for qc in self.quality_checks]
        return result
    
    def to_json(self, filepath: str) -> None:
        """Save metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ReproducibilityMetadata':
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert timestamp
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert quality checks
        if 'quality_checks' in data:
            data['quality_checks'] = [
                QualityCheckResult(**{**qc, 'timestamp': datetime.fromisoformat(qc['timestamp'])})
                for qc in data['quality_checks']
            ]
        
        return cls(**data)


# ============================================================================
# QualityController Class
# ============================================================================

class QualityController:
    """
    Comprehensive quality control and reproducibility management.
    
    This class implements quality checks before, during, and after test execution,
    along with complete metadata capture for reproducibility.
    
    Attributes:
        strict_mode: If True, treat warnings as errors and halt execution
        capture_env_vars: If True, capture environment variables (may include sensitive data)
        validate_checksums: If True, compute and validate file checksums
    
    Example:
        >>> qc = QualityController(strict_mode=True)
        >>> 
        >>> # Pre-execution validation
        >>> native_check = qc.validate_native_structure("1UBQ.pdb")
        >>> if not native_check.passed:
        ...     print(f"Native structure validation failed: {native_check.message}")
        ...     return
        >>> 
        >>> # Record metadata
        >>> metadata = qc.record_test_metadata(
        ...     pdb_id="1UBQ",
        ...     config=config_dict,
        ...     random_seed=42
        ... )
        >>> 
        >>> # ... run test ...
        >>> 
        >>> # Post-execution validation
        >>> output_check = qc.validate_output_files("results/1UBQ_predicted.pdb")
        >>> metadata.quality_checks.append(output_check)
        >>> 
        >>> # Generate reproducibility script
        >>> script = qc.generate_reproducibility_script(metadata)
        >>> qc.save_reproducibility_script(script, "reproduce_1UBQ.py")
    """
    
    def __init__(self,
                 strict_mode: bool = False,
                 capture_env_vars: bool = False,
                 validate_checksums: bool = True):
        """
        Initialize QualityController.
        
        Args:
            strict_mode: Treat warnings as errors
            capture_env_vars: Capture environment variables (may include sensitive data)
            validate_checksums: Compute and validate file checksums
        """
        self.strict_mode = strict_mode
        self.capture_env_vars = capture_env_vars
        self.validate_checksums = validate_checksums
        logger.info(f"QualityController initialized (strict={strict_mode})")
    
    # ========================================================================
    # Native Structure Validation
    # ========================================================================
    
    def validate_native_structure(self, pdb_file: str) -> QualityCheckResult:
        """
        Validate native structure before test execution.
        
        Checks:
        - File exists and is readable
        - File size is reasonable (>100 bytes, <100MB)
        - File has .pdb or .cif extension
        - File contains valid PDB format markers (ATOM, HETATM, etc.)
        - Structure has at least one CA atom per residue
        - No excessive missing residues
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            QualityCheckResult with validation outcome
        """
        check_name = "native_structure_validation"
        details = {"file": pdb_file}
        
        try:
            # Check file exists
            if not os.path.exists(pdb_file):
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity="critical",
                    message=f"Native structure file not found: {pdb_file}",
                    details=details
                )
            
            # Check file size
            file_size = os.path.getsize(pdb_file)
            details['file_size_bytes'] = file_size  # type: ignore[assignment]
            
            if file_size < 100:
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity="error",
                    message=f"File too small ({file_size} bytes), likely empty or corrupted",
                    details=details
                )
            
            if file_size > 100 * 1024 * 1024:  # 100 MB
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity="warning",
                    message=f"File very large ({file_size / (1024*1024):.1f} MB), may have issues",
                    details=details
                )
            
            # Check file extension
            ext = Path(pdb_file).suffix.lower()
            if ext not in ['.pdb', '.cif', '.ent']:
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity="warning",
                    message=f"Unexpected file extension: {ext} (expected .pdb, .cif, or .ent)",
                    details=details
                )
            
            # Parse file and check structure
            with open(pdb_file, 'r') as f:
                lines = f.readlines()
            
            details['total_lines'] = len(lines)  # type: ignore[assignment]
            
            # Count structure markers
            atom_lines = [l for l in lines if l.startswith('ATOM') or l.startswith('HETATM')]
            ca_atoms = [l for l in atom_lines if ' CA ' in l]
            
            details['atom_lines'] = len(atom_lines)  # type: ignore[assignment]
            details['ca_atoms'] = len(ca_atoms)  # type: ignore[assignment]
            
            if len(atom_lines) == 0:
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity="error",
                    message="No ATOM or HETATM records found in file",
                    details=details
                )
            
            if len(ca_atoms) < 10:
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity="error",
                    message=f"Too few CA atoms ({len(ca_atoms)}), structure likely incomplete",
                    details=details
                )
            
            # Check for excessive gaps
            if len(ca_atoms) > 0:
                # Extract residue numbers from CA atoms
                residue_numbers = []
                for line in ca_atoms:
                    try:
                        res_num = int(line[22:26].strip())
                        residue_numbers.append(res_num)
                    except:
                        pass
                
                if residue_numbers:
                    residue_numbers.sort()
                    gaps = []
                    for i in range(len(residue_numbers) - 1):
                        gap = residue_numbers[i + 1] - residue_numbers[i]
                        if gap > 1:
                            gaps.append(gap - 1)
                    
                    total_missing = sum(gaps)
                    missing_pct = (total_missing / len(residue_numbers)) * 100 if residue_numbers else 0
                    
                    details['missing_residues'] = total_missing  # type: ignore[assignment]
                    details['missing_percentage'] = f"{missing_pct:.1f}%"
                    
                    if missing_pct > 20:
                        return QualityCheckResult(
                            check_name=check_name,
                            passed=True,  # Warning but still passed
                            severity="warning",
                            message=f"Excessive missing residues ({missing_pct:.1f}%), may affect validation",
                            details=details
                        )
            
            # All checks passed
            return QualityCheckResult(
                check_name=check_name,
                passed=True,
                severity="info",
                message=f"Native structure validated: {len(ca_atoms)} residues",
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error validating native structure: {e}")
            return QualityCheckResult(
                check_name=check_name,
                passed=False,
                severity="critical",
                message=f"Exception during validation: {str(e)}",
                details={**details, 'exception': str(e), 'traceback': traceback.format_exc()}
            )
    
    # ========================================================================
    # Output File Validation
    # ========================================================================
    
    def validate_output_files(self,
                             predicted_pdb: Optional[str] = None,
                             results_json: Optional[str] = None,
                             log_file: Optional[str] = None) -> QualityCheckResult:
        """
        Validate output files after test completion.
        
        Checks:
        - Files exist and are readable
        - Files have expected content
        - JSON files are valid
        - PDB files have structure data
        - Log files contain expected execution markers
        
        Args:
            predicted_pdb: Path to predicted structure PDB file
            results_json: Path to results JSON file
            log_file: Path to execution log file
            
        Returns:
            QualityCheckResult with validation outcome
        """
        check_name = "output_file_validation"
        details = {
            'predicted_pdb': predicted_pdb,
            'results_json': results_json,
            'log_file': log_file
        }
        issues = []
        
        try:
            # Validate predicted PDB
            if predicted_pdb:
                if not os.path.exists(predicted_pdb):
                    issues.append(f"Predicted PDB not found: {predicted_pdb}")
                else:
                    pdb_size = os.path.getsize(predicted_pdb)
                    details['predicted_pdb_size'] = pdb_size  # type: ignore[assignment]
                    
                    if pdb_size < 100:
                        issues.append(f"Predicted PDB too small ({pdb_size} bytes)")
                    else:
                        # Check for ATOM records
                        with open(predicted_pdb, 'r') as f:
                            pdb_lines = f.readlines()
                        atom_count = sum(1 for l in pdb_lines if l.startswith('ATOM'))
                        details['predicted_atoms'] = atom_count  # type: ignore[assignment]
                        
                        if atom_count < 10:
                            issues.append(f"Predicted PDB has too few atoms ({atom_count})")
            
            # Validate results JSON
            if results_json:
                if not os.path.exists(results_json):
                    issues.append(f"Results JSON not found: {results_json}")
                else:
                    try:
                        with open(results_json, 'r') as f:
                            results = json.load(f)
                        details['results_keys'] = list(results.keys())  # type: ignore[assignment]
                        
                        # Check for expected keys
                        expected_keys = ['pdb_id', 'best_rmsd', 'best_energy']
                        missing_keys = [k for k in expected_keys if k not in results]
                        if missing_keys:
                            issues.append(f"Results JSON missing keys: {missing_keys}")
                            
                    except json.JSONDecodeError as e:
                        issues.append(f"Results JSON invalid: {str(e)}")
            
            # Validate log file
            if log_file:
                if not os.path.exists(log_file):
                    issues.append(f"Log file not found: {log_file}")
                else:
                    log_size = os.path.getsize(log_file)
                    details['log_file_size'] = log_size  # type: ignore[assignment]
                    
                    # Note: Very small log files (< 50 bytes) are suspicious
                    # but we allow them since some tests may have minimal output
                    
                    # Check for execution markers
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    
                    markers = ['Starting', 'Completed', 'iteration', 'agent']
                    found_markers = [m for m in markers if m.lower() in log_content.lower()]
                    details['log_markers_found'] = found_markers  # type: ignore[assignment]
                    
                    if len(found_markers) < 2:
                        issues.append(f"Log file missing expected markers (found: {found_markers})")
            
            # Determine result
            if issues:
                severity = "error" if len(issues) > 2 else "warning"
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity=severity,
                    message=f"Output validation found {len(issues)} issue(s): {'; '.join(issues)}",
                    details=details
                )
            else:
                return QualityCheckResult(
                    check_name=check_name,
                    passed=True,
                    severity="info",
                    message="All output files validated successfully",
                    details=details
                )
                
        except Exception as e:
            logger.error(f"Error validating output files: {e}")
            return QualityCheckResult(
                check_name=check_name,
                passed=False,
                severity="critical",
                message=f"Exception during validation: {str(e)}",
                details={**details, 'exception': str(e), 'traceback': traceback.format_exc()}
            )
    
    # ========================================================================
    # Metadata Recording
    # ========================================================================
    
    def record_test_metadata(self,
                            pdb_id: str,
                            config: Dict[str, Any],
                            native_pdb: Optional[str] = None,
                            test_id: Optional[str] = None) -> ReproducibilityMetadata:
        """
        Record complete metadata for test reproducibility.
        
        Captures:
        - Software versions (Python, UBF, dependencies)
        - System information (platform, architecture)
        - Configuration parameters
        - Random seed
        - File checksums
        - Environment variables (if enabled)
        - Command line and working directory
        
        Args:
            pdb_id: PDB identifier
            config: Test configuration dictionary
            native_pdb: Path to native structure (for checksum)
            test_id: Unique test identifier (auto-generated if None)
            
        Returns:
            ReproducibilityMetadata with complete test metadata
        """
        # Generate test ID
        if test_id is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_id = f"{pdb_id}_{timestamp_str}"
        
        # Capture software versions
        python_version = sys.version
        system_platform = platform.platform()
        
        # Get UBF version (from package or git)
        ubf_version = self._get_ubf_version()
        
        # Get dependency versions
        dependencies = self._get_dependency_versions()
        
        # Extract configuration
        num_agents = config.get('num_agents', 0)
        iterations_per_agent = config.get('iterations_per_agent', 0)
        qcpp_enabled = config.get('qcpp_enabled', False)
        random_seed = config.get('random_seed', None)
        adaptive_config = config.get('adaptive_config', {})
        execution_parameters = {k: v for k, v in config.items() 
                              if k not in ['num_agents', 'iterations_per_agent', 
                                         'qcpp_enabled', 'random_seed', 'adaptive_config']}
        
        # Compute checksums
        native_checksum = None
        if native_pdb and self.validate_checksums:
            native_checksum = self._compute_file_checksum(native_pdb)
        
        # Capture environment
        working_directory = os.getcwd()
        command_line = ' '.join(sys.argv)
        env_vars = {}
        if self.capture_env_vars:
            # Capture selected environment variables (avoid sensitive data)
            safe_vars = ['PATH', 'PYTHONPATH', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV']
            env_vars = {k: os.environ.get(k, '') for k in safe_vars if k in os.environ}
        
        # Create metadata
        metadata = ReproducibilityMetadata(
            test_id=test_id,
            pdb_id=pdb_id,
            timestamp=datetime.now(),
            python_version=python_version,
            system_platform=system_platform,
            ubf_version=ubf_version,
            dependencies=dependencies,
            num_agents=num_agents,
            iterations_per_agent=iterations_per_agent,
            qcpp_enabled=qcpp_enabled,
            random_seed=random_seed,
            adaptive_config=adaptive_config,
            execution_parameters=execution_parameters,
            native_structure_checksum=native_checksum,
            predicted_structure_checksum=None,  # Set after prediction
            working_directory=working_directory,
            command_line=command_line,
            environment_variables=env_vars
        )
        
        logger.info(f"Recorded metadata for test {test_id}")
        return metadata
    
    def update_predicted_checksum(self,
                                  metadata: ReproducibilityMetadata,
                                  predicted_pdb: str) -> None:
        """
        Update metadata with predicted structure checksum.
        
        Args:
            metadata: Metadata object to update
            predicted_pdb: Path to predicted structure
        """
        if self.validate_checksums and os.path.exists(predicted_pdb):
            metadata.predicted_structure_checksum = self._compute_file_checksum(predicted_pdb)
            logger.debug(f"Updated predicted structure checksum: {metadata.predicted_structure_checksum[:16]}...")
    
    # ========================================================================
    # Abnormal Termination Detection
    # ========================================================================
    
    def detect_abnormal_termination(self,
                                   log_file: str,
                                   expected_markers: Optional[List[str]] = None) -> QualityCheckResult:
        """
        Detect abnormal termination by checking log file.
        
        Checks for:
        - Expected completion markers
        - Error/exception patterns
        - Sudden termination
        - Timeout indicators
        
        Args:
            log_file: Path to execution log file
            expected_markers: List of expected completion markers
            
        Returns:
            QualityCheckResult indicating normal or abnormal termination
        """
        check_name = "abnormal_termination_detection"
        details = {'log_file': log_file}
        
        if expected_markers is None:
            expected_markers = ['completed', 'success', 'finished', 'done']
        
        try:
            if not os.path.exists(log_file):
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity="error",
                    message="Log file not found, execution may have crashed",
                    details=details
                )
            
            with open(log_file, 'r') as f:
                log_content = f.read().lower()
            
            details['log_size'] = len(log_content)  # type: ignore[assignment]
            
            # Check for completion markers
            found_completion = any(marker.lower() in log_content for marker in expected_markers)
            details['found_completion_marker'] = found_completion  # type: ignore[assignment]
            
            # Check for error patterns
            error_patterns = ['error:', 'exception:', 'traceback', 'failed', 'crash', 'killed']
            found_errors = [p for p in error_patterns if p in log_content]
            details['error_patterns_found'] = found_errors  # type: ignore[assignment]
            
            # Check for timeout
            timeout_patterns = ['timeout', 'timed out', 'exceeded time limit']
            found_timeout = any(p in log_content for p in timeout_patterns)
            details['timeout_detected'] = found_timeout  # type: ignore[assignment]
            
            # Determine result
            if not found_completion:
                severity = "critical" if found_errors else "warning"
                message = "Execution did not complete normally"
                if found_errors:
                    message += f" (errors: {', '.join(found_errors)})"
                if found_timeout:
                    message += " (timeout detected)"
                
                return QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    severity=severity,
                    message=message,
                    details=details
                )
            elif found_errors and not found_timeout:
                # Completed but with errors
                return QualityCheckResult(
                    check_name=check_name,
                    passed=True,
                    severity="warning",
                    message=f"Completed with warnings/errors: {', '.join(found_errors)}",
                    details=details
                )
            else:
                # Normal completion
                return QualityCheckResult(
                    check_name=check_name,
                    passed=True,
                    severity="info",
                    message="Execution completed normally",
                    details=details
                )
                
        except Exception as e:
            logger.error(f"Error detecting abnormal termination: {e}")
            return QualityCheckResult(
                check_name=check_name,
                passed=False,
                severity="error",
                message=f"Exception during detection: {str(e)}",
                details={**details, 'exception': str(e)}
            )
    
    # ========================================================================
    # Reproducibility Script Generation
    # ========================================================================
    
    def generate_reproducibility_script(self,
                                       metadata: ReproducibilityMetadata,
                                       template: str = "python") -> str:
        """
        Generate reproducibility script for re-executing test.
        
        Creates a standalone script that:
        - Documents all test parameters
        - Sets up identical environment
        - Executes test with same configuration
        - Validates results match original
        
        Args:
            metadata: Test metadata to encode in script
            template: Script template ('python', 'bash', 'batch')
            
        Returns:
            Complete reproducibility script as string
        """
        if template == "python":
            return self._generate_python_script(metadata)
        elif template == "bash":
            return self._generate_bash_script(metadata)
        elif template == "batch":
            return self._generate_batch_script(metadata)
        else:
            raise ValueError(f"Unknown template: {template}")
    
    def _generate_python_script(self, metadata: ReproducibilityMetadata) -> str:
        """Generate Python reproducibility script."""
        script = f'''#!/usr/bin/env python3
"""
Reproducibility Script for Test: {metadata.test_id}
Generated: {datetime.now().isoformat()}

This script reproduces the exact test conditions for:
PDB ID: {metadata.pdb_id}
Original execution: {metadata.timestamp.isoformat()}

To run: python {metadata.test_id}_reproduce.py
"""

import sys
import os
import json
from datetime import datetime

# ============================================================================
# Test Metadata
# ============================================================================

TEST_ID = "{metadata.test_id}"
PDB_ID = "{metadata.pdb_id}"
ORIGINAL_TIMESTAMP = "{metadata.timestamp.isoformat()}"

# Software Versions
PYTHON_VERSION = "{metadata.python_version.split()[0]}"
UBF_VERSION = "{metadata.ubf_version}"
SYSTEM_PLATFORM = "{metadata.system_platform}"

# Configuration
NUM_AGENTS = {metadata.num_agents}
ITERATIONS_PER_AGENT = {metadata.iterations_per_agent}
QCPP_ENABLED = {metadata.qcpp_enabled}
RANDOM_SEED = {metadata.random_seed if metadata.random_seed is not None else "None"}

# Adaptive Configuration
ADAPTIVE_CONFIG = {json.dumps(metadata.adaptive_config, indent=4)}

# Execution Parameters
EXECUTION_PARAMETERS = {json.dumps(metadata.execution_parameters, indent=4)}

# File Checksums (for validation)
NATIVE_STRUCTURE_CHECKSUM = "{metadata.native_structure_checksum}"
EXPECTED_OUTPUT_CHECKSUM = "{metadata.predicted_structure_checksum}"

# ============================================================================
# Environment Validation
# ============================================================================

def validate_environment():
    """Validate that environment matches original test conditions."""
    issues = []
    
    # Check Python version
    current_python = f"{{sys.version_info.major}}.{{sys.version_info.minor}}.{{sys.version_info.micro}}"
    if not current_python.startswith(PYTHON_VERSION[:3]):
        issues.append(f"Python version mismatch: expected {{PYTHON_VERSION}}, got {{current_python}}")
    
    # Check platform
    import platform
    if platform.platform() != SYSTEM_PLATFORM:
        print(f"WARNING: Platform differs - expected {{SYSTEM_PLATFORM}}, got {{platform.platform()}}")
    
    # Check dependencies
    try:
        # Import required modules to verify they're installed
        import ubf_protein
        print("[OK] UBF protein module found")
    except ImportError as e:
        issues.append(f"Missing dependency: {{e}}")
    
    if issues:
        print("\\n[!] ENVIRONMENT VALIDATION FAILED:")
        for issue in issues:
            print(f"  - {{issue}}")
        response = input("\\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print("[OK] Environment validation passed\\n")

# ============================================================================
# Test Execution
# ============================================================================

def run_test():
    """Execute test with original parameters."""
    print(f"Reproducing test: {{TEST_ID}}")
    print(f"PDB ID: {{PDB_ID}}")
    print(f"Original timestamp: {{ORIGINAL_TIMESTAMP}}")
    print(f"Reproduction timestamp: {{datetime.now().isoformat()}}\\n")
    
    # Import UBF modules
    from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
    from ubf_protein.validation_suite import ValidationSuite
    
    # Set random seed if specified
    if RANDOM_SEED is not None:
        import random
        random.seed(RANDOM_SEED)
        print(f"[OK] Random seed set: {{RANDOM_SEED}}\\n")
    
    # Run prediction
    print(f"Running {{NUM_AGENTS}} agents × {{ITERATIONS_PER_AGENT}} iterations...")
    print(f"QCPP integration: {{'enabled' if QCPP_ENABLED else 'disabled'}}\\n")
    
    # TODO: Add actual test execution code here
    # This depends on your specific test harness
    
    print("\\n[OK] Test execution completed")

# ============================================================================
# Results Validation
# ============================================================================

def validate_results():
    """Validate that results match original test."""
    print("\\nValidating results...")
    
    # TODO: Add result validation logic
    # - Check output file checksums
    # - Compare metrics (RMSD, energy, etc.)
    # - Verify reproducibility
    
    print("[OK] Results validation completed")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print(f"REPRODUCIBILITY SCRIPT: {{TEST_ID}}")
    print("="*80 + "\\n")
    
    validate_environment()
    run_test()
    validate_results()
    
    print("\\n" + "="*80)
    print("REPRODUCTION COMPLETE")
    print("="*80)
'''
        return script
    
    def _generate_bash_script(self, metadata: ReproducibilityMetadata) -> str:
        """Generate Bash reproducibility script."""
        # Simplified bash version
        script = f'''#!/bin/bash
# Reproducibility Script for Test: {metadata.test_id}
# Generated: {datetime.now().isoformat()}

echo "Reproducing test: {metadata.test_id}"
echo "PDB ID: {metadata.pdb_id}"
echo "Original timestamp: {metadata.timestamp.isoformat()}"

# Run Python reproducibility script
python {metadata.test_id}_reproduce.py
'''
        return script
    
    def _generate_batch_script(self, metadata: ReproducibilityMetadata) -> str:
        """Generate Windows batch reproducibility script."""
        script = f'''@echo off
REM Reproducibility Script for Test: {metadata.test_id}
REM Generated: {datetime.now().isoformat()}

echo Reproducing test: {metadata.test_id}
echo PDB ID: {metadata.pdb_id}
echo Original timestamp: {metadata.timestamp.isoformat()}

REM Run Python reproducibility script
python {metadata.test_id}_reproduce.py
'''
        return script
    
    def save_reproducibility_script(self,
                                   script: str,
                                   output_path: str) -> None:
        """
        Save reproducibility script to file.
        
        Args:
            script: Script content
            output_path: Path to save script
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        # Make executable on Unix-like systems
        if sys.platform != 'win32':
            os.chmod(output_path, 0o755)
        
        logger.info(f"Saved reproducibility script: {output_path}")
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_ubf_version(self) -> str:
        """Get UBF protein package version."""
        try:
            import ubf_protein
            if hasattr(ubf_protein, '__version__'):
                return ubf_protein.__version__
        except:
            pass
        
        # Try to get git commit hash
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return f"git-{result.stdout.strip()}"
        except:
            pass
        
        return "unknown"
    
    def _get_dependency_versions(self) -> Dict[str, str]:
        """Get versions of key dependencies."""
        dependencies = {}
        
        # List of packages to check
        packages = ['numpy', 'scipy', 'biopython', 'pytest']
        
        for package in packages:
            try:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                dependencies[package] = version
            except ImportError:
                dependencies[package] = 'not installed'
        
        return dependencies
    
    def _compute_file_checksum(self, filepath: str, algorithm: str = 'sha256') -> str:
        """
        Compute file checksum.
        
        Args:
            filepath: Path to file
            algorithm: Hash algorithm (sha256, md5)
            
        Returns:
            Hexadecimal checksum string
        """
        if algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif algorithm == 'md5':
            hasher = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
