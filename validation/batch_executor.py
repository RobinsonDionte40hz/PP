"""
Batch Executor for Large-Scale Validation

Executes multiple protein validation tests efficiently with:
- Parallel execution (up to max_parallel concurrent tests)
- Resource monitoring (CPU, memory, disk)
- Adaptive throttling based on resource constraints
- Size-based prioritization (small proteins first)
- Checkpointing every N completed tests
- Resume from checkpoint after interruption
- Completion time estimation

Key Features:
- Execute 50-75 proteins in batch mode
- Monitor system resources and throttle if needed
- Checkpoint frequently for recovery
- Prioritize small proteins for faster early results
- Estimate remaining time based on averages
"""

import json
import logging
import os
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """
    Current system resource usage metrics.
    
    Attributes:
        cpu_usage_percent: CPU usage percentage (0-100 per core, can exceed 100)
        memory_usage_mb: Memory usage in megabytes
        memory_usage_percent: Memory usage as percentage
        disk_usage_mb: Disk usage in megabytes
        active_processes: Number of active processes in executor
        throttle_recommended: Whether throttling is recommended
        timestamp: When metrics were captured
    """
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_usage_mb: float
    active_processes: int
    throttle_recommended: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BatchProgress:
    """
    Progress tracking for batch execution.
    
    Attributes:
        total_proteins: Total number of proteins in batch
        completed: Number of completed tests
        in_progress: Number of tests currently running
        pending: Number of pending tests
        failed: Number of failed tests
        estimated_completion: Estimated completion datetime
        average_time_per_protein: Average execution time per protein (seconds)
        elapsed_time: Total elapsed time (seconds)
    """
    total_proteins: int
    completed: int
    in_progress: int
    pending: int
    failed: int
    estimated_completion: Optional[datetime]
    average_time_per_protein: float
    elapsed_time: float


@dataclass
class BatchCheckpoint:
    """
    Checkpoint data for batch execution resume.
    
    Attributes:
        batch_id: Unique identifier for this batch
        checkpoint_time: When checkpoint was created
        total_proteins: Total proteins in batch
        completed_proteins: List of completed protein IDs
        failed_proteins: List of failed protein IDs
        pending_proteins: List of pending protein IDs
        execution_times: Dict mapping protein ID to execution time
        configuration: Batch execution configuration
    """
    batch_id: str
    checkpoint_time: str
    total_proteins: int
    completed_proteins: List[str]
    failed_proteins: List[str]
    pending_proteins: List[str]
    execution_times: Dict[str, float]
    configuration: Dict[str, Any]


class BatchExecutor:
    """
    Executes multiple protein validation tests in parallel with resource management.
    
    Features:
    - Parallel execution (configurable max concurrent tests)
    - Resource monitoring and adaptive throttling
    - Size-based prioritization (small proteins first)
    - Automatic checkpointing for recovery
    - Resume from checkpoint
    - Progress tracking and time estimation
    
    Usage:
        executor = BatchExecutor(
            max_parallel=3,
            checkpoint_interval=5,
            checkpoint_dir="checkpoints"
        )
        
        results = executor.execute_batch(
            proteins=protein_list,
            test_function=run_single_test
        )
    """
    
    def __init__(self,
                 max_parallel: int = 3,
                 checkpoint_interval: int = 5,
                 checkpoint_dir: str = "checkpoints",
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 80.0,
                 enable_throttling: bool = True):
        """
        Initialize BatchExecutor.
        
        Args:
            max_parallel: Maximum concurrent test executions
            checkpoint_interval: Checkpoint every N completed tests
            checkpoint_dir: Directory for checkpoint files
            cpu_threshold: CPU usage threshold for throttling (percent)
            memory_threshold: Memory usage threshold for throttling (percent)
            enable_throttling: Enable adaptive throttling based on resources
        """
        self.max_parallel = max_parallel
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.enable_throttling = enable_throttling
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self._completed: List[str] = []
        self._failed: List[str] = []
        self._in_progress: set = set()
        self._execution_times: Dict[str, float] = {}
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        
        logger.info(f"BatchExecutor initialized: max_parallel={max_parallel}, "
                   f"checkpoint_interval={checkpoint_interval}, "
                   f"throttling={'enabled' if enable_throttling else 'disabled'}")
    
    def execute_batch(self,
                     proteins: List[Any],
                     test_function: Callable,
                     batch_id: Optional[str] = None,
                     prioritize: bool = True) -> List[Any]:
        """
        Execute batch of protein tests with parallel execution and checkpointing.
        
        Args:
            proteins: List of protein metadata objects (must have pdb_id attribute)
            test_function: Function to execute for each protein (takes protein, returns result)
            batch_id: Optional batch identifier (generated if None)
            prioritize: Whether to prioritize by size (small first)
        
        Returns:
            List of test results (same order as input proteins)
        """
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._start_time = time.time()
        self._completed = []
        self._failed = []
        self._in_progress = set()
        self._execution_times = {}
        
        # Prioritize proteins if requested
        if prioritize:
            proteins_sorted = self.prioritize_by_size(proteins)
        else:
            proteins_sorted = list(proteins)
        
        # Track results in original order
        protein_ids = [p.pdb_id for p in proteins]
        results_map: Dict[str, Any] = {}
        
        logger.info(f"Starting batch {batch_id}: {len(proteins)} proteins, "
                   f"max_parallel={self.max_parallel}, prioritize={prioritize}")
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures: Dict[Future, Any] = {}
            
            for protein in proteins_sorted:
                # Check resources and throttle if needed
                if self.enable_throttling:
                    self._throttle_if_needed()
                
                # Submit task
                future = executor.submit(self._execute_single, protein, test_function)
                futures[future] = protein
                
                with self._lock:
                    self._in_progress.add(protein.pdb_id)
            
            # Collect results as they complete
            for future in as_completed(futures):
                protein = futures[future]
                pdb_id = protein.pdb_id
                
                try:
                    result = future.result()
                    results_map[pdb_id] = result
                    
                    with self._lock:
                        self._completed.append(pdb_id)
                        self._in_progress.discard(pdb_id)
                    
                    logger.info(f"Completed {pdb_id} ({len(self._completed)}/{len(proteins)})")
                    
                    # Checkpoint if interval reached
                    if len(self._completed) % self.checkpoint_interval == 0:
                        self._create_checkpoint(batch_id, proteins_sorted, protein_ids)
                
                except Exception as e:
                    logger.error(f"Failed {pdb_id}: {e}")
                    
                    with self._lock:
                        self._failed.append(pdb_id)
                        self._in_progress.discard(pdb_id)
                        results_map[pdb_id] = None
        
        # Final checkpoint
        self._create_checkpoint(batch_id, proteins_sorted, protein_ids)
        
        elapsed = time.time() - self._start_time
        logger.info(f"Batch {batch_id} complete: {len(self._completed)} succeeded, "
                   f"{len(self._failed)} failed, elapsed={elapsed:.1f}s")
        
        # Return results in original order
        results = [results_map.get(pdb_id) for pdb_id in protein_ids]
        return results
    
    def _execute_single(self, protein: Any, test_function: Callable) -> Any:
        """
        Execute single protein test with timing.
        
        Args:
            protein: Protein metadata
            test_function: Test function to execute
        
        Returns:
            Test result
        """
        pdb_id = protein.pdb_id
        start = time.time()
        
        try:
            result = test_function(protein)
            elapsed = time.time() - start
            
            with self._lock:
                self._execution_times[pdb_id] = elapsed
            
            return result
        
        except Exception as e:
            elapsed = time.time() - start
            with self._lock:
                self._execution_times[pdb_id] = elapsed
            raise
    
    def _throttle_if_needed(self) -> None:
        """Check resources and sleep if thresholds exceeded."""
        metrics = self.monitor_resources()
        
        if metrics.throttle_recommended:
            throttle_time = 5.0
            logger.warning(f"Resource threshold exceeded: CPU={metrics.cpu_usage_percent:.1f}%, "
                          f"Memory={metrics.memory_usage_percent:.1f}%. "
                          f"Throttling for {throttle_time}s")
            time.sleep(throttle_time)
    
    def prioritize_by_size(self, proteins: List[Any]) -> List[Any]:
        """
        Prioritize proteins by size (small first) for faster early results.
        
        Args:
            proteins: List of protein metadata objects (must have sequence_length)
        
        Returns:
            Sorted list of proteins (smallest first)
        """
        sorted_proteins = sorted(proteins, key=lambda p: p.sequence_length)
        
        logger.debug(f"Prioritized proteins by size: "
                    f"{sorted_proteins[0].pdb_id} ({sorted_proteins[0].sequence_length} residues) first, "
                    f"{sorted_proteins[-1].pdb_id} ({sorted_proteins[-1].sequence_length} residues) last")
        
        return sorted_proteins
    
    def monitor_resources(self) -> ResourceMetrics:
        """
        Monitor current system resource usage.
        
        Returns:
            ResourceMetrics with current usage and throttle recommendation
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent
        
        # Disk usage (current directory)
        disk = psutil.disk_usage('.')
        disk_mb = disk.used / (1024 * 1024)
        
        # Active processes
        with self._lock:
            active_processes = len(self._in_progress)
        
        # Throttle recommendation
        throttle = (cpu_percent > self.cpu_threshold or 
                   memory_percent > self.memory_threshold)
        
        return ResourceMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            memory_usage_percent=memory_percent,
            disk_usage_mb=disk_mb,
            active_processes=active_processes,
            throttle_recommended=throttle
        )
    
    def get_progress(self) -> BatchProgress:
        """
        Get current batch execution progress.
        
        Returns:
            BatchProgress with current status and estimates
        """
        with self._lock:
            completed = len(self._completed)
            in_progress = len(self._in_progress)
            failed = len(self._failed)
            
            total = completed + in_progress + failed
            pending = total - completed - in_progress - failed
            
            # Calculate average time
            if self._execution_times:
                avg_time = sum(self._execution_times.values()) / len(self._execution_times)
            else:
                avg_time = 0.0
            
            # Estimate completion
            if avg_time > 0 and pending > 0:
                remaining_seconds = avg_time * pending
                estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)
            else:
                estimated_completion = None
            
            # Elapsed time
            elapsed = time.time() - self._start_time if self._start_time else 0.0
            
            return BatchProgress(
                total_proteins=total,
                completed=completed,
                in_progress=in_progress,
                pending=pending,
                failed=failed,
                estimated_completion=estimated_completion,
                average_time_per_protein=avg_time,
                elapsed_time=elapsed
            )
    
    def estimate_completion_time(self, remaining: int) -> Optional[timedelta]:
        """
        Estimate completion time for remaining proteins.
        
        Args:
            remaining: Number of remaining proteins
        
        Returns:
            Estimated time remaining as timedelta, or None if insufficient data
        """
        with self._lock:
            if not self._execution_times:
                return None
            
            avg_time = sum(self._execution_times.values()) / len(self._execution_times)
        
        # Account for parallelization
        parallel_time = avg_time * remaining / self.max_parallel
        
        return timedelta(seconds=parallel_time)
    
    def _create_checkpoint(self, 
                          batch_id: str, 
                          proteins: List[Any],
                          original_order: List[str]) -> str:
        """
        Create checkpoint file for batch execution.
        
        Args:
            batch_id: Batch identifier
            proteins: List of all proteins
            original_order: Original protein ID order
        
        Returns:
            Path to checkpoint file
        """
        with self._lock:
            completed = list(self._completed)
            failed = list(self._failed)
            execution_times = dict(self._execution_times)
        
        # Determine pending proteins
        all_ids = {p.pdb_id for p in proteins}
        done_ids = set(completed + failed)
        pending_ids = list(all_ids - done_ids)
        
        checkpoint = BatchCheckpoint(
            batch_id=batch_id,
            checkpoint_time=datetime.now().isoformat(),
            total_proteins=len(proteins),
            completed_proteins=completed,
            failed_proteins=failed,
            pending_proteins=pending_ids,
            execution_times=execution_times,
            configuration={
                "max_parallel": self.max_parallel,
                "checkpoint_interval": self.checkpoint_interval,
                "cpu_threshold": self.cpu_threshold,
                "memory_threshold": self.memory_threshold
            }
        )
        
        checkpoint_file = self.checkpoint_dir / f"{batch_id}_checkpoint.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2)
            
            logger.info(f"Created checkpoint: {checkpoint_file} "
                       f"({len(completed)} completed, {len(pending_ids)} pending)")
            
            return str(checkpoint_file)
        
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return ""
    
    def resume_from_checkpoint(self, 
                               checkpoint_file: str,
                               proteins: List[Any],
                               test_function: Callable) -> List[Any]:
        """
        Resume batch execution from checkpoint.
        
        Args:
            checkpoint_file: Path to checkpoint file
            proteins: Original list of proteins
            test_function: Test function to execute
        
        Returns:
            List of test results (same order as input proteins)
        """
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            checkpoint = BatchCheckpoint(**checkpoint_data)
            
            logger.info(f"Resuming from checkpoint: {checkpoint.batch_id} "
                       f"({len(checkpoint.completed_proteins)} completed, "
                       f"{len(checkpoint.pending_proteins)} pending)")
            
            # Restore state
            self._completed = checkpoint.completed_proteins
            self._failed = checkpoint.failed_proteins
            self._execution_times = checkpoint.execution_times
            
            # Filter to pending proteins only
            pending_proteins = [p for p in proteins 
                               if p.pdb_id in checkpoint.pending_proteins]
            
            # Execute remaining proteins
            if pending_proteins:
                results = self.execute_batch(
                    proteins=pending_proteins,
                    test_function=test_function,
                    batch_id=checkpoint.batch_id,
                    prioritize=False  # Keep checkpoint order
                )
                return results
            else:
                logger.info("No pending proteins to execute")
                return []
        
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise
