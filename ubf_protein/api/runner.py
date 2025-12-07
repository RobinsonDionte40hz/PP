"""
PredictionRunner - Public API Wrapper

This module provides the public-facing PredictionRunner that wraps
the internal implementation. External code should use this instead
of importing from the internal prediction_runner module directly.

This separation allows:
1. Stable public API even when internals change
2. Clear documentation of what's public vs internal
3. Type-safe interfaces for external consumers
"""

from typing import Optional, Callable
import logging

from .interfaces import IPredictionRunner
from .schemas import (
    PredictionConfig,
    PredictionResults,
    ProgressUpdate,
    ValidationMetrics,
)

logger = logging.getLogger(__name__)


class PredictionRunner(IPredictionRunner):
    """
    Public interface for protein structure prediction.
    
    This class wraps the internal prediction engine, providing a
    stable API for external consumers (backend, CLI tools).
    
    Usage:
        from ubf_protein.api import PredictionRunner, PredictionConfig
        
        config = PredictionConfig(
            sequence="MQIFVKTLTGK...",
            agents=10,
            iterations=500,
            qcpp_config="default"
        )
        
        runner = PredictionRunner(config)
        results = runner.run(progress_callback=my_callback)
        
        print(f"RMSD: {results.metrics.rmsd}")
        print(results.pdb_string)
    """
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize the prediction runner.
        
        Args:
            config: Prediction configuration
        """
        self._config = config
        self._is_running = False
        self._cancelled = False
        self._internal_runner = None
        
    def run(
        self, 
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> PredictionResults:
        """
        Execute the prediction.
        
        This method delegates to the internal prediction runner while
        providing API stability and proper type conversion.
        
        Args:
            progress_callback: Optional callback for progress updates.
                              Called periodically with ProgressUpdate objects.
        
        Returns:
            PredictionResults containing predicted structure and metrics
            
        Raises:
            ValueError: If sequence is invalid
            RuntimeError: If prediction fails
        """
        # Validate sequence first
        is_valid, error_msg = self.validate_sequence(self._config.sequence)
        if not is_valid:
            raise ValueError(f"Invalid sequence: {error_msg}")
        
        self._is_running = True
        self._cancelled = False
        
        try:
            # Import internal runner (lazy import to avoid circular deps)
            from ..prediction_runner import (
                PredictionRunner as InternalRunner,
                PredictionConfig as InternalConfig,
            )
            
            # Convert public config to internal config
            internal_config = InternalConfig(
                sequence=self._config.sequence,
                native_pdb=self._config.native_pdb,
                native_pdb_path=self._config.native_pdb_path,
                agents=self._config.agents,
                iterations=self._config.iterations,
                enable_refinement=self._config.enable_refinement,
                enable_mediators=self._config.enable_mediators,
                enable_geometric_attractors=self._config.enable_geometric_attractors,
                qcpp_config=self._config.qcpp_config,
                output_dir=self._config.output_dir,
                checkpoint_interval=self._config.checkpoint_interval,
            )
            
            # Create internal runner
            self._internal_runner = InternalRunner(internal_config)
            
            # Wrap progress callback if provided
            wrapped_callback = None
            if progress_callback:
                def wrapped_callback(internal_update):
                    # Convert internal ProgressUpdate to public schema
                    public_update = ProgressUpdate(
                        iteration=internal_update.iteration,
                        total_iterations=internal_update.total_iterations,
                        phase=internal_update.phase,
                        percentage=internal_update.percentage,
                        best_energy=internal_update.best_energy,
                        current_rmsd=getattr(internal_update, 'current_rmsd', None),
                        message=internal_update.message,
                        metrics=getattr(internal_update, 'metrics', {}),
                    )
                    progress_callback(public_update)
            
            # Run prediction
            internal_results = self._internal_runner.run(
                progress_callback=wrapped_callback
            )
            
            # Convert internal results to public schema
            return self._convert_results(internal_results)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
        finally:
            self._is_running = False
            self._internal_runner = None
    
    def _convert_results(self, internal_results) -> PredictionResults:
        """Convert internal results to public schema."""
        # Extract metrics
        metrics = ValidationMetrics(
            rmsd=getattr(internal_results, 'rmsd', None),
            tm_score=getattr(internal_results, 'tm_score', None),
            energy_total=getattr(internal_results, 'best_energy', None),
            qcp_score=getattr(internal_results, 'qcp_score', None),
        )
        
        # Build results
        return PredictionResults(
            sequence=internal_results.sequence,
            pdb_string=internal_results.pdb_string,
            coordinates=internal_results.coordinates,
            metrics=metrics,
            trajectory=getattr(internal_results, 'trajectory', []),
            runtime_seconds=getattr(internal_results, 'runtime_seconds', 0.0),
            config=self._config.to_dict(),
            metadata={
                'engine_version': '1.0.0',
                'qcpp_enabled': self._config.qcpp_config != 'none',
            }
        )
    
    def get_config(self) -> PredictionConfig:
        """Get the current configuration."""
        return self._config
    
    def validate_sequence(self, sequence: str) -> tuple[bool, str]:
        """
        Validate a protein sequence.
        
        Args:
            sequence: Amino acid sequence to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sequence:
            return False, "Sequence is empty"
        
        # Standard amino acids
        valid_residues = set('ACDEFGHIKLMNPQRSTVWY')
        
        # Clean sequence
        clean_seq = sequence.upper().strip()
        
        if len(clean_seq) < 5:
            return False, "Sequence too short (minimum 5 residues)"
        
        if len(clean_seq) > 2000:
            return False, "Sequence too long (maximum 2000 residues)"
        
        invalid_chars = set(clean_seq) - valid_residues
        if invalid_chars:
            return False, f"Invalid amino acids: {', '.join(sorted(invalid_chars))}"
        
        return True, ""
    
    @property
    def is_running(self) -> bool:
        """Whether a prediction is currently in progress."""
        return self._is_running
    
    def cancel(self) -> bool:
        """
        Request cancellation of the current prediction.
        
        Returns:
            True if cancellation was requested, False if no prediction running
        """
        if not self._is_running:
            return False
        
        self._cancelled = True
        if self._internal_runner and hasattr(self._internal_runner, 'cancel'):
            self._internal_runner.cancel()
        return True


def get_optimal_settings(sequence_length: int) -> PredictionConfig:
    """
    Get optimal prediction settings based on sequence length.
    
    Args:
        sequence_length: Number of residues in the sequence
        
    Returns:
        PredictionConfig with optimized parameters
    """
    if sequence_length < 50:
        return PredictionConfig(
            sequence="",  # Placeholder, caller should set
            agents=8,
            iterations=300,
            enable_refinement=True,
        )
    elif sequence_length < 150:
        return PredictionConfig(
            sequence="",
            agents=10,
            iterations=500,
            enable_refinement=True,
        )
    else:
        return PredictionConfig(
            sequence="",
            agents=12,
            iterations=800,
            enable_refinement=True,
        )


def get_quick_test_settings(sequence_length: int = 50) -> PredictionConfig:
    """
    Get settings for quick testing (reduced accuracy, fast completion).
    
    Args:
        sequence_length: Number of residues (used to scale agent count)
    
    Returns:
        PredictionConfig optimized for speed over accuracy
    """
    if sequence_length < 50:
        agents = 4
        iterations = 50
    elif sequence_length < 100:
        agents = 5
        iterations = 80
    else:
        agents = 6
        iterations = 100
    
    return PredictionConfig(
        sequence="",
        agents=agents,
        iterations=iterations,
        enable_refinement=False,
        qcpp_config="none",
    )
