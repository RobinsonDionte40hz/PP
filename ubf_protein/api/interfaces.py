"""
Abstract interfaces for the Prediction Engine.

These interfaces define the contracts that external code depends on.
Concrete implementations are hidden from external consumers.

This enables:
1. Dependency Injection - swap implementations without changing callers
2. Testing - mock implementations for unit tests
3. Future flexibility - add new prediction strategies without breaking API
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass


class IProgressCallback(ABC):
    """
    Interface for receiving progress updates during prediction.
    
    Implement this interface to receive real-time updates from the
    prediction engine. Used by CLI for console output and backend
    for WebSocket notifications.
    """
    
    @abstractmethod
    def on_progress(self, update: 'ProgressUpdate') -> None:
        """
        Called when prediction progress changes.
        
        Args:
            update: Progress information including iteration, percentage, metrics
        """
        pass
    
    @abstractmethod
    def on_phase_change(self, phase: str, description: str) -> None:
        """
        Called when prediction enters a new phase.
        
        Args:
            phase: Phase identifier (e.g., 'exploration', 'refinement')
            description: Human-readable phase description
        """
        pass
    
    @abstractmethod
    def on_complete(self, results: 'PredictionResults') -> None:
        """
        Called when prediction completes successfully.
        
        Args:
            results: Final prediction results
        """
        pass
    
    @abstractmethod
    def on_error(self, error: Exception, recoverable: bool) -> None:
        """
        Called when an error occurs.
        
        Args:
            error: The exception that occurred
            recoverable: Whether prediction can continue
        """
        pass


class IPredictionRunner(ABC):
    """
    Interface for protein structure prediction.
    
    This is the main entry point for running predictions.
    All prediction strategies must implement this interface.
    """
    
    @abstractmethod
    def run(
        self, 
        progress_callback: Optional[Callable[['ProgressUpdate'], None]] = None
    ) -> 'PredictionResults':
        """
        Execute the prediction.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            PredictionResults containing the predicted structure and metrics
        """
        pass
    
    @abstractmethod
    def get_config(self) -> 'PredictionConfig':
        """Get the current configuration."""
        pass
    
    @abstractmethod
    def validate_sequence(self, sequence: str) -> tuple[bool, str]:
        """
        Validate a protein sequence.
        
        Args:
            sequence: Amino acid sequence to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether a prediction is currently in progress."""
        pass
    
    @abstractmethod
    def cancel(self) -> bool:
        """
        Request cancellation of the current prediction.
        
        Returns:
            True if cancellation was requested, False if no prediction running
        """
        pass


class IResultsExporter(ABC):
    """
    Interface for exporting prediction results.
    
    Different exporters handle different formats (PDB, JSON, etc.)
    """
    
    @abstractmethod
    def export(self, results: 'PredictionResults', output_path: str) -> str:
        """
        Export results to a file.
        
        Args:
            results: Prediction results to export
            output_path: Path for the output file
            
        Returns:
            Path to the created file
        """
        pass
    
    @abstractmethod
    def export_string(self, results: 'PredictionResults') -> str:
        """
        Export results to a string.
        
        Args:
            results: Prediction results to export
            
        Returns:
            String representation in the exporter's format
        """
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this export format."""
        pass
    
    @property
    @abstractmethod
    def mime_type(self) -> str:
        """MIME type for this export format."""
        pass


class IScreener(ABC):
    """
    Interface for protein screening operations.
    
    Screening analyzes sequences for specific properties
    (aggregation propensity, solubility, etc.)
    """
    
    @abstractmethod
    def screen(
        self, 
        sequence: str,
        config: Optional['ScreeningConfig'] = None
    ) -> 'ScreeningResults':
        """
        Screen a protein sequence.
        
        Args:
            sequence: Amino acid sequence to screen
            config: Optional screening configuration
            
        Returns:
            Screening results with identified regions and scores
        """
        pass
    
    @abstractmethod
    def batch_screen(
        self,
        sequences: List[str],
        config: Optional['ScreeningConfig'] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List['ScreeningResults']:
        """
        Screen multiple sequences.
        
        Args:
            sequences: List of sequences to screen
            config: Optional screening configuration
            progress_callback: Callback with (current, total) progress
            
        Returns:
            List of screening results
        """
        pass


class IQCPPAdapter(ABC):
    """
    Interface for QCPP (Quantum Coherence) integration.
    
    This abstracts the quantum coherence system, allowing the core
    prediction engine to work with or without QCPP.
    """
    
    @abstractmethod
    def calculate_coherence(
        self, 
        structure: Any,  # Will be typed properly in implementation
        sequence: str
    ) -> Dict[str, float]:
        """
        Calculate quantum coherence metrics for a structure.
        
        Args:
            structure: Protein structure representation
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of coherence metrics
        """
        pass
    
    @abstractmethod
    def get_stability_feedback(
        self,
        structure: Any,
        sequence: str
    ) -> Dict[str, Any]:
        """
        Get stability feedback for guiding exploration.
        
        Args:
            structure: Current structure
            sequence: Amino acid sequence
            
        Returns:
            Feedback dictionary with scores and suggestions
        """
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether QCPP is available and configured."""
        pass


# Import type references for forward declarations
# These will be defined in schemas.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .schemas import (
        PredictionConfig,
        PredictionResults,
        ProgressUpdate,
        ScreeningConfig,
        ScreeningResults,
    )
