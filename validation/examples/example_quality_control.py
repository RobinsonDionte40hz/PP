"""
Example Usage: Quality Control and Reproducibility

This example demonstrates the quality control and reproducibility features
for large-scale protein structure validation campaigns.

Features demonstrated:
1. Native structure validation before test execution
2. Complete metadata recording for reproducibility
3. Output file validation after test completion
4. Abnormal termination detection
5. Reproducibility script generation

Run this example:
    python validation/examples/example_quality_control.py
"""

import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.quality_control import QualityController


def create_sample_pdb(filepath: str) -> None:
    """Create a sample PDB file for testing."""
    content = """HEADER    EXAMPLE PROTEIN
ATOM      1  CA  MET A   1      10.000  10.000  10.000  1.00 20.00           C
ATOM      2  CA  GLN A   2      13.800  10.000  10.000  1.00 20.00           C
ATOM      3  CA  ILE A   3      17.600  10.000  10.000  1.00 20.00           C
ATOM      4  CA  PHE A   4      21.400  10.000  10.000  1.00 20.00           C
ATOM      5  CA  VAL A   5      25.200  10.000  10.000  1.00 20.00           C
ATOM      6  CA  LYS A   6      29.000  10.000  10.000  1.00 20.00           C
ATOM      7  CA  THR A   7      32.800  10.000  10.000  1.00 20.00           C
ATOM      8  CA  LEU A   8      36.600  10.000  10.000  1.00 20.00           C
ATOM      9  CA  THR A   9      40.400  10.000  10.000  1.00 20.00           C
ATOM     10  CA  GLY A  10      44.200  10.000  10.000  1.00 20.00           C
ATOM     11  CA  LYS A  11      48.000  10.000  10.000  1.00 20.00           C
ATOM     12  CA  THR A  12      51.800  10.000  10.000  1.00 20.00           C
ATOM     13  CA  ILE A  13      55.600  10.000  10.000  1.00 20.00           C
ATOM     14  CA  THR A  14      59.400  10.000  10.000  1.00 20.00           C
ATOM     15  CA  LEU A  15      63.200  10.000  10.000  1.00 20.00           C
END
"""
    with open(filepath, 'w') as f:
        f.write(content)


def example_1_native_structure_validation():
    """
    Example 1: Validate native structure before test execution
    
    This ensures the native PDB file is valid before running predictions.
    """
    print("=" * 80)
    print("Example 1: Native Structure Validation")
    print("=" * 80 + "\n")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create quality controller
        qc = QualityController()
        
        # Create a valid PDB file
        pdb_file = os.path.join(temp_dir, '1UBQ.pdb')
        create_sample_pdb(pdb_file)
        
        print(f"Validating native structure: {pdb_file}\n")
        result = qc.validate_native_structure(pdb_file)
        
        print(f"Check: {result.check_name}")
        print(f"Passed: {result.passed}")
        print(f"Severity: {result.severity}")
        print(f"Message: {result.message}")
        print(f"\nDetails:")
        for key, value in result.details.items():
            print(f"  {key}: {value}")
        
        # Test with missing file
        print("\n" + "-" * 80)
        print("Testing with missing file:\n")
        
        result = qc.validate_native_structure("nonexistent.pdb")
        print(f"Passed: {result.passed}")
        print(f"Severity: {result.severity}")
        print(f"Message: {result.message}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n")


def example_2_metadata_recording():
    """
    Example 2: Record complete metadata for reproducibility
    
    Captures all test parameters, software versions, and configuration.
    """
    print("=" * 80)
    print("Example 2: Metadata Recording")
    print("=" * 80 + "\n")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create quality controller
        qc = QualityController(validate_checksums=True)
        
        # Create native structure for checksum
        native_pdb = os.path.join(temp_dir, 'native.pdb')
        create_sample_pdb(native_pdb)
        
        # Test configuration
        config = {
            'num_agents': 10,
            'iterations_per_agent': 1000,
            'qcpp_enabled': True,
            'random_seed': 42,
            'adaptive_config': {
                'stuck_window': 30,
                'stuck_threshold': 10.0,
                'max_iterations': 5000
            },
            'timeout': 3600,
            'checkpoint_interval': 50
        }
        
        print("Recording test metadata...\n")
        metadata = qc.record_test_metadata(
            pdb_id="1UBQ",
            config=config,
            native_pdb=native_pdb,
            test_id="EXAMPLE_TEST_001"
        )
        
        print(f"Test ID: {metadata.test_id}")
        print(f"PDB ID: {metadata.pdb_id}")
        print(f"Timestamp: {metadata.timestamp.isoformat()}")
        print(f"\nConfiguration:")
        print(f"  Agents: {metadata.num_agents}")
        print(f"  Iterations: {metadata.iterations_per_agent}")
        print(f"  QCPP: {metadata.qcpp_enabled}")
        print(f"  Random Seed: {metadata.random_seed}")
        print(f"\nSystem:")
        print(f"  Python: {metadata.python_version}")
        print(f"  Platform: {metadata.system_platform}")
        print(f"  UBF Version: {metadata.ubf_version}")
        print(f"\nChecksums:")
        if metadata.native_structure_checksum:
            print(f"  Native: {metadata.native_structure_checksum[:16]}...")
        else:
            print(f"  Native: Not computed")
        
        # Save metadata to JSON
        metadata_path = os.path.join(temp_dir, 'metadata.json')
        metadata.to_json(metadata_path)
        print(f"\nMetadata saved to: {metadata_path}")
        
        # Load and verify
        from validation.quality_control import ReproducibilityMetadata
        loaded = ReproducibilityMetadata.from_json(metadata_path)
        print(f"✓ Metadata successfully loaded from file")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n")


def example_3_output_validation():
    """
    Example 3: Validate output files after test completion
    
    Ensures all expected output files exist and contain valid data.
    """
    print("=" * 80)
    print("Example 3: Output File Validation")
    print("=" * 80 + "\n")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create quality controller
        qc = QualityController()
        
        # Create output files
        predicted_pdb = os.path.join(temp_dir, 'predicted.pdb')
        results_json = os.path.join(temp_dir, 'results.json')
        log_file = os.path.join(temp_dir, 'test.log')
        
        create_sample_pdb(predicted_pdb)
        
        with open(results_json, 'w') as f:
            json.dump({
                'pdb_id': '1UBQ',
                'sequence_length': 76,
                'best_rmsd': 2.8,
                'best_energy': -55.3,
                'gdt_ts_score': 72.5,
                'tm_score': 0.82,
                'runtime_seconds': 450.2
            }, f)
        
        with open(log_file, 'w') as f:
            f.write("""2025-01-26 14:30:00 - INFO - Starting test for 1UBQ
2025-01-26 14:30:01 - INFO - Initialized 10 agents
2025-01-26 14:30:02 - INFO - Beginning exploration
2025-01-26 14:35:00 - INFO - Iteration 100 completed
2025-01-26 14:40:00 - INFO - Iteration 200 completed
2025-01-26 14:45:00 - INFO - Agent 0 best RMSD: 3.2
2025-01-26 14:50:00 - INFO - Agent 1 best RMSD: 2.8
2025-01-26 14:55:00 - INFO - Test completed successfully
2025-01-26 14:55:01 - INFO - Final RMSD: 2.8 Å
""")
        
        print("Validating output files...\n")
        result = qc.validate_output_files(
            predicted_pdb=predicted_pdb,
            results_json=results_json,
            log_file=log_file
        )
        
        print(f"Check: {result.check_name}")
        print(f"Passed: {result.passed}")
        print(f"Severity: {result.severity}")
        print(f"Message: {result.message}")
        print(f"\nDetails:")
        for key, value in result.details.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(map(str, value))}")
            else:
                print(f"  {key}: {value}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n")


def example_4_termination_detection():
    """
    Example 4: Detect abnormal termination
    
    Analyzes log files to detect crashes, timeouts, or other issues.
    """
    print("=" * 80)
    print("Example 4: Abnormal Termination Detection")
    print("=" * 80 + "\n")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create quality controller
        qc = QualityController()
        
        # Test 1: Normal completion
        print("Test 1: Normal completion\n")
        log_file = os.path.join(temp_dir, 'normal.log')
        with open(log_file, 'w') as f:
            f.write("Starting test\nIteration 100\nAgent 5\nTest completed successfully\n")
        
        result = qc.detect_abnormal_termination(log_file)
        print(f"Passed: {result.passed}")
        print(f"Message: {result.message}\n")
        
        # Test 2: Incomplete execution
        print("Test 2: Incomplete execution (no completion marker)\n")
        log_file = os.path.join(temp_dir, 'incomplete.log')
        with open(log_file, 'w') as f:
            f.write("Starting test\nIteration 50\nAgent 3\n")
        
        result = qc.detect_abnormal_termination(log_file)
        print(f"Passed: {result.passed}")
        print(f"Severity: {result.severity}")
        print(f"Message: {result.message}\n")
        
        # Test 3: Crash with error
        print("Test 3: Crash with error\n")
        log_file = os.path.join(temp_dir, 'crash.log')
        with open(log_file, 'w') as f:
            f.write("Starting test\nIteration 25\nError: Out of memory\nException: MemoryError\n")
        
        result = qc.detect_abnormal_termination(log_file)
        print(f"Passed: {result.passed}")
        print(f"Severity: {result.severity}")
        print(f"Message: {result.message}")
        print(f"Errors found: {result.details.get('error_patterns_found', [])}\n")
        
        # Test 4: Timeout
        print("Test 4: Timeout\n")
        log_file = os.path.join(temp_dir, 'timeout.log')
        with open(log_file, 'w') as f:
            f.write("Starting test\nIteration 10\nTimeout: exceeded time limit\n")
        
        result = qc.detect_abnormal_termination(log_file)
        print(f"Passed: {result.passed}")
        print(f"Message: {result.message}")
        print(f"Timeout detected: {result.details.get('timeout_detected', False)}\n")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n")


def example_5_reproducibility_script():
    """
    Example 5: Generate reproducibility script
    
    Creates standalone scripts to reproduce test conditions exactly.
    """
    print("=" * 80)
    print("Example 5: Reproducibility Script Generation")
    print("=" * 80 + "\n")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create quality controller
        qc = QualityController(validate_checksums=True)
        
        # Create native structure
        native_pdb = os.path.join(temp_dir, 'native.pdb')
        create_sample_pdb(native_pdb)
        
        # Test configuration
        config = {
            'num_agents': 10,
            'iterations_per_agent': 1000,
            'qcpp_enabled': True,
            'random_seed': 42,
            'adaptive_config': {
                'stuck_window': 30,
                'stuck_threshold': 10.0
            }
        }
        
        # Record metadata
        metadata = qc.record_test_metadata(
            pdb_id="1UBQ",
            config=config,
            native_pdb=native_pdb,
            test_id="REPRO_TEST_001"
        )
        
        # Generate Python script
        print("Generating Python reproducibility script...\n")
        python_script = qc.generate_reproducibility_script(metadata, template="python")
        python_path = os.path.join(temp_dir, "reproduce_1UBQ.py")
        qc.save_reproducibility_script(python_script, python_path)
        print(f"✓ Saved Python script: {python_path}")
        
        # Generate Bash script
        bash_script = qc.generate_reproducibility_script(metadata, template="bash")
        bash_path = os.path.join(temp_dir, "reproduce_1UBQ.sh")
        qc.save_reproducibility_script(bash_script, bash_path)
        print(f"✓ Saved Bash script: {bash_path}")
        
        # Generate Batch script
        batch_script = qc.generate_reproducibility_script(metadata, template="batch")
        batch_path = os.path.join(temp_dir, "reproduce_1UBQ.bat")
        qc.save_reproducibility_script(batch_script, batch_path)
        print(f"✓ Saved Batch script: {batch_path}")
        
        # Show preview of Python script
        print(f"\nPreview of Python script (first 30 lines):")
        print("-" * 80)
        lines = python_script.split('\n')[:30]
        for line in lines:
            print(line)
        print("-" * 80)
        print(f"... ({len(python_script.split(chr(10))) - 30} more lines)")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n")


def example_6_complete_workflow():
    """
    Example 6: Complete quality control workflow
    
    Demonstrates a complete workflow from pre-execution validation through
    post-execution checks and reproducibility documentation.
    """
    print("=" * 80)
    print("Example 6: Complete Quality Control Workflow")
    print("=" * 80 + "\n")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize quality controller
        print("Step 1: Initialize QualityController\n")
        qc = QualityController(
            strict_mode=False,
            capture_env_vars=False,
            validate_checksums=True
        )
        print("✓ QualityController initialized\n")
        
        # Pre-execution: Validate native structure
        print("Step 2: Pre-execution - Validate native structure\n")
        native_pdb = os.path.join(temp_dir, '1UBQ.pdb')
        create_sample_pdb(native_pdb)
        
        native_check = qc.validate_native_structure(native_pdb)
        if not native_check.passed:
            print(f"✗ Native structure validation failed: {native_check.message}")
            return
        print(f"✓ Native structure validated: {native_check.message}\n")
        
        # Record metadata
        print("Step 3: Record test metadata\n")
        config = {
            'num_agents': 10,
            'iterations_per_agent': 1000,
            'qcpp_enabled': True,
            'random_seed': 42,
            'adaptive_config': {'stuck_window': 30}
        }
        
        metadata = qc.record_test_metadata(
            pdb_id="1UBQ",
            config=config,
            native_pdb=native_pdb,
            test_id="WORKFLOW_001"
        )
        metadata.quality_checks.append(native_check)
        print(f"✓ Metadata recorded: Test ID {metadata.test_id}\n")
        
        # Simulate test execution
        print("Step 4: Simulate test execution\n")
        print("  (In real workflow, run actual prediction here)\n")
        
        # Create output files
        predicted_pdb = os.path.join(temp_dir, 'predicted.pdb')
        results_json = os.path.join(temp_dir, 'results.json')
        log_file = os.path.join(temp_dir, 'test.log')
        
        create_sample_pdb(predicted_pdb)
        
        with open(results_json, 'w') as f:
            json.dump({
                'pdb_id': '1UBQ',
                'best_rmsd': 2.8,
                'best_energy': -55.3,
                'gdt_ts_score': 72.5
            }, f)
        
        with open(log_file, 'w') as f:
            f.write("Starting\nIteration 500\nCompleted successfully\n")
        
        # Post-execution: Update checksum
        print("Step 5: Post-execution - Update predicted checksum\n")
        qc.update_predicted_checksum(metadata, predicted_pdb)
        if metadata.predicted_structure_checksum:
            print(f"✓ Predicted checksum: {metadata.predicted_structure_checksum[:16]}...\n")
        else:
            print(f"✓ Predicted checksum updated\n")
        
        # Validate outputs
        print("Step 6: Validate output files\n")
        output_check = qc.validate_output_files(
            predicted_pdb=predicted_pdb,
            results_json=results_json,
            log_file=log_file
        )
        metadata.quality_checks.append(output_check)
        
        if output_check.passed:
            print(f"✓ Output files validated\n")
        else:
            print(f"✗ Output validation issues: {output_check.message}\n")
        
        # Check termination
        print("Step 7: Check for abnormal termination\n")
        termination_check = qc.detect_abnormal_termination(log_file)
        metadata.quality_checks.append(termination_check)
        
        if termination_check.passed:
            print(f"✓ Execution completed normally\n")
        else:
            print(f"✗ Abnormal termination: {termination_check.message}\n")
        
        # Generate documentation
        print("Step 8: Generate reproducibility documentation\n")
        
        # Save metadata
        metadata_path = os.path.join(temp_dir, 'metadata.json')
        metadata.to_json(metadata_path)
        print(f"✓ Saved metadata: {metadata_path}")
        
        # Generate script
        script = qc.generate_reproducibility_script(metadata)
        script_path = os.path.join(temp_dir, 'reproduce.py')
        qc.save_reproducibility_script(script, script_path)
        print(f"✓ Saved reproducibility script: {script_path}\n")
        
        # Summary
        print("=" * 80)
        print("WORKFLOW SUMMARY")
        print("=" * 80)
        print(f"Test ID: {metadata.test_id}")
        print(f"PDB ID: {metadata.pdb_id}")
        print(f"Quality Checks: {len(metadata.quality_checks)}")
        print(f"  Passed: {sum(1 for qc in metadata.quality_checks if qc.passed)}")
        print(f"  Failed: {sum(1 for qc in metadata.quality_checks if not qc.passed)}")
        print(f"\nAll quality checks passed: {all(qc.passed for qc in metadata.quality_checks)}")
        print(f"Ready for reproducibility: {metadata.completed_normally}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n")


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("QUALITY CONTROL AND REPRODUCIBILITY - EXAMPLE USAGE")
    print("=" * 80)
    print("\n")
    
    example_1_native_structure_validation()
    example_2_metadata_recording()
    example_3_output_validation()
    example_4_termination_detection()
    example_5_reproducibility_script()
    example_6_complete_workflow()
    
    print("=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\n")


if __name__ == "__main__":
    main()
