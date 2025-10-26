"""
Unit tests for BatchExecutor

Tests cover:
- Initialization and configuration
- Resource monitoring
- Size-based prioritization
- Batch execution with mock test function
- Progress tracking
- Completion time estimation
- Checkpointing
- Resume from checkpoint
- Throttling behavior
- Parallel execution
- Error handling

Note: psutil is required for resource monitoring
Install with: pip install psutil
"""

import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    from validation.batch_executor import (
        BatchExecutor,
        ResourceMetrics,
        BatchProgress,
        BatchCheckpoint
    )
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    pytest.skip("psutil not available", allow_module_level=True)


@dataclass
class MockProteinMetadata:
    """Mock protein metadata for testing."""
    pdb_id: str
    sequence_length: int
    size_category: str


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def executor(temp_dir):
    """Create BatchExecutor instance."""
    return BatchExecutor(
        max_parallel=2,
        checkpoint_interval=2,
        checkpoint_dir=temp_dir,
        enable_throttling=False  # Disable for predictable testing
    )


@pytest.fixture
def sample_proteins():
    """Create sample protein list."""
    return [
        MockProteinMetadata("1UBQ", 76, "small"),
        MockProteinMetadata("1CRN", 46, "tiny"),
        MockProteinMetadata("2MR9", 35, "tiny"),
        MockProteinMetadata("1VII", 36, "tiny"),
        MockProteinMetadata("1LYZ", 129, "medium"),
    ]


def mock_test_success(protein):
    """Mock test function that succeeds."""
    time.sleep(0.1)  # Simulate work
    return {"pdb_id": protein.pdb_id, "rmsd": 2.5, "success": True}


def mock_test_failure(protein):
    """Mock test function that fails."""
    if protein.pdb_id == "1VII":
        raise ValueError(f"Simulated failure for {protein.pdb_id}")
    time.sleep(0.1)
    return {"pdb_id": protein.pdb_id, "rmsd": 2.5, "success": True}


def mock_test_variable_time(protein):
    """Mock test function with variable execution time."""
    # Small proteins faster
    delay = protein.sequence_length / 1000.0
    time.sleep(delay)
    return {"pdb_id": protein.pdb_id, "rmsd": 2.5, "success": True}


class TestInitialization:
    """Test BatchExecutor initialization."""
    
    def test_create_with_defaults(self):
        """Test creation with default parameters."""
        executor = BatchExecutor()
        assert executor.max_parallel == 3
        assert executor.checkpoint_interval == 5
        assert executor.enable_throttling is True
    
    def test_create_with_custom_params(self, temp_dir):
        """Test creation with custom parameters."""
        executor = BatchExecutor(
            max_parallel=5,
            checkpoint_interval=10,
            checkpoint_dir=temp_dir,
            cpu_threshold=90.0,
            memory_threshold=85.0,
            enable_throttling=False
        )
        
        assert executor.max_parallel == 5
        assert executor.checkpoint_interval == 10
        assert executor.cpu_threshold == 90.0
        assert executor.memory_threshold == 85.0
        assert executor.enable_throttling is False
    
    def test_checkpoint_dir_created(self, temp_dir):
        """Test that checkpoint directory is created."""
        checkpoint_path = Path(temp_dir) / "checkpoints"
        executor = BatchExecutor(checkpoint_dir=str(checkpoint_path))
        
        assert checkpoint_path.exists()
        assert checkpoint_path.is_dir()


class TestResourceMonitoring:
    """Test resource monitoring functionality."""
    
    def test_monitor_resources(self, executor):
        """Test basic resource monitoring."""
        metrics = executor.monitor_resources()
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.cpu_usage_percent >= 0
        assert metrics.memory_usage_mb > 0
        assert metrics.memory_usage_percent >= 0
        assert metrics.disk_usage_mb > 0
        assert metrics.active_processes >= 0
        assert isinstance(metrics.throttle_recommended, bool)
        assert metrics.timestamp is not None
    
    def test_throttle_recommendation(self):
        """Test throttle recommendation logic."""
        # High thresholds - should not recommend throttle
        executor1 = BatchExecutor(
            cpu_threshold=99.0,
            memory_threshold=99.0,
            enable_throttling=True
        )
        metrics1 = executor1.monitor_resources()
        # Likely not to throttle with 99% thresholds
        
        # Low thresholds - might recommend throttle
        executor2 = BatchExecutor(
            cpu_threshold=1.0,
            memory_threshold=1.0,
            enable_throttling=True
        )
        metrics2 = executor2.monitor_resources()
        # More likely to throttle with 1% thresholds
        
        # Just verify we can call it without errors
        assert isinstance(metrics1.throttle_recommended, bool)
        assert isinstance(metrics2.throttle_recommended, bool)


class TestPrioritization:
    """Test size-based prioritization."""
    
    def test_prioritize_by_size(self, executor, sample_proteins):
        """Test prioritization puts small proteins first."""
        prioritized = executor.prioritize_by_size(sample_proteins)
        
        # Check ascending order by sequence_length
        lengths = [p.sequence_length for p in prioritized]
        assert lengths == sorted(lengths)
        
        # Check smallest first
        assert prioritized[0].pdb_id == "2MR9"  # 35 residues
        assert prioritized[-1].pdb_id == "1LYZ"  # 129 residues
    
    def test_prioritize_preserves_all_proteins(self, executor, sample_proteins):
        """Test that prioritization doesn't lose proteins."""
        prioritized = executor.prioritize_by_size(sample_proteins)
        
        assert len(prioritized) == len(sample_proteins)
        
        original_ids = {p.pdb_id for p in sample_proteins}
        prioritized_ids = {p.pdb_id for p in prioritized}
        assert original_ids == prioritized_ids


class TestBatchExecution:
    """Test batch execution functionality."""
    
    def test_execute_batch_success(self, executor, sample_proteins):
        """Test successful batch execution."""
        results = executor.execute_batch(
            proteins=sample_proteins,
            test_function=mock_test_success,
            prioritize=False
        )
        
        assert len(results) == len(sample_proteins)
        
        # Check all succeeded
        for result in results:
            assert result is not None
            assert result["success"] is True
    
    def test_execute_batch_with_failure(self, executor, sample_proteins):
        """Test batch execution with one failure."""
        results = executor.execute_batch(
            proteins=sample_proteins,
            test_function=mock_test_failure,
            prioritize=False
        )
        
        assert len(results) == len(sample_proteins)
        
        # Check 1VII failed, others succeeded
        for i, protein in enumerate(sample_proteins):
            if protein.pdb_id == "1VII":
                assert results[i] is None
            else:
                assert results[i] is not None
                assert results[i]["success"] is True
    
    def test_execute_batch_with_prioritization(self, executor, sample_proteins):
        """Test batch execution with size prioritization."""
        results = executor.execute_batch(
            proteins=sample_proteins,
            test_function=mock_test_success,
            prioritize=True
        )
        
        assert len(results) == len(sample_proteins)
        
        # Results should be in original order despite prioritization
        for i, protein in enumerate(sample_proteins):
            assert results[i]["pdb_id"] == protein.pdb_id
    
    def test_execution_times_tracked(self, executor, sample_proteins):
        """Test that execution times are tracked."""
        executor.execute_batch(
            proteins=sample_proteins,
            test_function=mock_test_success,
            prioritize=False
        )
        
        # Check execution times recorded
        assert len(executor._execution_times) == len(sample_proteins)
        
        for protein in sample_proteins:
            assert protein.pdb_id in executor._execution_times
            assert executor._execution_times[protein.pdb_id] > 0


class TestProgressTracking:
    """Test progress tracking functionality."""
    
    def test_get_progress_initial(self, executor):
        """Test progress before execution."""
        # Start time not set yet
        executor._start_time = time.time()
        
        progress = executor.get_progress()
        
        assert isinstance(progress, BatchProgress)
        assert progress.total_proteins == 0
        assert progress.completed == 0
        assert progress.in_progress == 0
        assert progress.pending == 0
        assert progress.failed == 0
    
    def test_get_progress_during_execution(self, executor, sample_proteins):
        """Test progress during execution."""
        # Execute batch
        executor.execute_batch(
            proteins=sample_proteins[:2],  # Just 2 for speed
            test_function=mock_test_success,
            prioritize=False
        )
        
        progress = executor.get_progress()
        
        assert progress.completed == 2
        assert progress.average_time_per_protein > 0
        assert progress.elapsed_time > 0


class TestCompletionEstimation:
    """Test completion time estimation."""
    
    def test_estimate_completion_time_no_data(self, executor):
        """Test estimation with no execution data."""
        estimate = executor.estimate_completion_time(remaining=10)
        assert estimate is None
    
    def test_estimate_completion_time_with_data(self, executor, sample_proteins):
        """Test estimation after some executions."""
        # Execute a few proteins
        executor.execute_batch(
            proteins=sample_proteins[:2],
            test_function=mock_test_success,
            prioritize=False
        )
        
        # Estimate for remaining
        estimate = executor.estimate_completion_time(remaining=3)
        
        assert estimate is not None
        assert isinstance(estimate, timedelta)
        assert estimate.total_seconds() > 0


class TestCheckpointing:
    """Test checkpointing functionality."""
    
    def test_checkpoint_creation(self, executor, sample_proteins, temp_dir):
        """Test checkpoint file creation."""
        batch_id = "test_batch_001"
        
        # Execute with checkpointing
        executor.execute_batch(
            proteins=sample_proteins[:3],
            test_function=mock_test_success,
            batch_id=batch_id,
            prioritize=False
        )
        
        # Check checkpoint file exists
        checkpoint_files = list(Path(temp_dir).glob(f"{batch_id}_checkpoint.json"))
        assert len(checkpoint_files) > 0
        
        # Verify checkpoint content
        with open(checkpoint_files[0], 'r') as f:
            checkpoint_data = json.load(f)
        
        assert checkpoint_data["batch_id"] == batch_id
        assert checkpoint_data["total_proteins"] == 3
        assert len(checkpoint_data["completed_proteins"]) == 3
    
    def test_checkpoint_interval(self, executor, sample_proteins, temp_dir):
        """Test checkpoint interval respected."""
        # Executor has checkpoint_interval=2
        batch_id = "test_batch_002"
        
        executor.execute_batch(
            proteins=sample_proteins[:4],  # 4 proteins, should checkpoint at 2 and 4
            test_function=mock_test_success,
            batch_id=batch_id,
            prioritize=False
        )
        
        # Should have checkpoint file
        checkpoint_files = list(Path(temp_dir).glob(f"{batch_id}_checkpoint.json"))
        assert len(checkpoint_files) > 0


class TestResumeFromCheckpoint:
    """Test resume from checkpoint functionality."""
    
    def test_resume_from_checkpoint(self, executor, sample_proteins, temp_dir):
        """Test resuming batch from checkpoint."""
        batch_id = "test_batch_resume"
        
        # Execute partial batch
        proteins_subset = sample_proteins[:3]
        executor.execute_batch(
            proteins=proteins_subset,
            test_function=mock_test_success,
            batch_id=batch_id,
            prioritize=False
        )
        
        # Get checkpoint file
        checkpoint_file = str(list(Path(temp_dir).glob(f"{batch_id}_checkpoint.json"))[0])
        
        # Create new executor and resume
        executor2 = BatchExecutor(
            max_parallel=2,
            checkpoint_interval=2,
            checkpoint_dir=temp_dir,
            enable_throttling=False
        )
        
        # Manually modify checkpoint to have pending proteins
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        checkpoint_data["completed_proteins"] = [proteins_subset[0].pdb_id]
        checkpoint_data["pending_proteins"] = [p.pdb_id for p in proteins_subset[1:]]
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Resume
        results = executor2.resume_from_checkpoint(
            checkpoint_file=checkpoint_file,
            proteins=proteins_subset,
            test_function=mock_test_success
        )
        
        # Should complete the pending proteins
        assert len(results) == 2


class TestParallelExecution:
    """Test parallel execution behavior."""
    
    def test_parallel_execution_faster(self, temp_dir):
        """Test that parallel execution is faster than serial."""
        proteins = [
            MockProteinMetadata(f"TEST{i}", 50, "small")
            for i in range(4)
        ]
        
        def slow_test(protein):
            time.sleep(0.2)
            return {"pdb_id": protein.pdb_id}
        
        # Serial execution (max_parallel=1)
        executor_serial = BatchExecutor(
            max_parallel=1,
            checkpoint_dir=temp_dir,
            enable_throttling=False
        )
        start_serial = time.time()
        executor_serial.execute_batch(proteins, slow_test, prioritize=False)
        time_serial = time.time() - start_serial
        
        # Parallel execution (max_parallel=2)
        executor_parallel = BatchExecutor(
            max_parallel=2,
            checkpoint_dir=temp_dir,
            enable_throttling=False
        )
        start_parallel = time.time()
        executor_parallel.execute_batch(proteins, slow_test, prioritize=False)
        time_parallel = time.time() - start_parallel
        
        # Parallel should be noticeably faster
        assert time_parallel < time_serial * 0.8


class TestErrorHandling:
    """Test error handling."""
    
    def test_handles_test_function_failure(self, executor, sample_proteins):
        """Test graceful handling of test function failures."""
        results = executor.execute_batch(
            proteins=sample_proteins,
            test_function=mock_test_failure,
            prioritize=False
        )
        
        # Should complete despite failure
        assert len(results) == len(sample_proteins)
        
        # Check failed protein tracked
        assert "1VII" in executor._failed
    
    def test_invalid_checkpoint_file(self, executor, sample_proteins):
        """Test handling of invalid checkpoint file."""
        with pytest.raises(Exception):
            executor.resume_from_checkpoint(
                checkpoint_file="nonexistent.json",
                proteins=sample_proteins,
                test_function=mock_test_success
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
