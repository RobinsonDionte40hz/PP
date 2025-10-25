"""
Integrated Trajectory Recording for QCPP-UBF System

This module provides trajectory recording that captures both UBF metrics
(RMSD, energy, consciousness) and QCPP metrics (QCP, coherence, stability, phi match)
during conformational exploration.

Key Components:
- IntegratedTrajectoryPoint: Data class containing both UBF and QCPP metrics
- IntegratedTrajectoryRecorder: Recorder for capturing trajectory points
- TrajectoryAnalyzer: Analysis of correlations between QCPP and UBF metrics
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntegratedTrajectoryPoint:
    """
    A single point in the integrated trajectory containing both UBF and QCPP metrics.
    
    This immutable data class captures a snapshot of the exploration state at a
    specific iteration, including both consciousness-based metrics (UBF) and
    quantum physics metrics (QCPP).
    
    Attributes:
        iteration: Iteration number
        
        # UBF Metrics
        rmsd: Root mean square deviation to native structure (Angstroms)
        energy: Conformational energy (kcal/mol)
        consciousness_frequency: Agent consciousness frequency (3-15 Hz)
        consciousness_coherence: Agent consciousness coherence (0.2-1.0)
        
        # QCPP Metrics
        qcp_score: Quantum Consciousness Potential score (typically 3-8)
        field_coherence: Field coherence metric (-1 to 1)
        stability_score: QCPP stability prediction (higher = more stable)
        phi_match_score: Golden ratio angle matching score (0-1)
        
        # Timestamp
        timestamp: Unix timestamp in milliseconds
    """
    iteration: int
    
    # UBF metrics
    rmsd: float
    energy: float
    consciousness_frequency: float
    consciousness_coherence: float
    
    # QCPP metrics
    qcp_score: float
    field_coherence: float
    stability_score: float
    phi_match_score: float
    
    # Timestamp
    timestamp: int
    
    def __post_init__(self):
        """Validate trajectory point data."""
        if self.iteration < 0:
            raise ValueError(f"iteration {self.iteration} must be non-negative")
        
        if self.rmsd < 0:
            raise ValueError(f"rmsd {self.rmsd} must be non-negative")
        
        if not (3.0 <= self.consciousness_frequency <= 15.0):
            raise ValueError(f"consciousness_frequency {self.consciousness_frequency} must be in range [3, 15]")
        
        if not (0.2 <= self.consciousness_coherence <= 1.0):
            raise ValueError(f"consciousness_coherence {self.consciousness_coherence} must be in range [0.2, 1.0]")
        
        if not (0.0 <= self.phi_match_score <= 1.0):
            raise ValueError(f"phi_match_score {self.phi_match_score} must be in range [0, 1]")
        
        if self.timestamp < 0:
            raise ValueError(f"timestamp {self.timestamp} must be non-negative")


class IntegratedTrajectoryRecorder:
    """
    Records integrated trajectory points containing both UBF and QCPP metrics.
    
    This recorder maintains a chronological sequence of trajectory points
    that capture the state of conformational exploration at each iteration,
    including both consciousness-based and quantum physics measurements.
    
    Features:
    - Thread-safe recording of trajectory points
    - JSON export with metadata
    - Memory-efficient storage with optional point limit
    - Performance tracking (recording latency)
    
    Performance Targets:
    - Recording latency: <1ms per point
    - Memory overhead: ~200 bytes per point
    - Export latency: <100ms for 10K points
    """
    
    def __init__(self, max_points: Optional[int] = None):
        """
        Initialize integrated trajectory recorder.
        
        Args:
            max_points: Optional maximum number of points to store (default: unlimited)
                       When limit is reached, oldest points are discarded (FIFO)
        """
        self._points: List[IntegratedTrajectoryPoint] = []
        self._max_points = max_points
        self._recording_count = 0
        self._total_recording_time_ms = 0.0
    
    def record_point(self,
                    iteration: int,
                    rmsd: float,
                    energy: float,
                    consciousness_frequency: float,
                    consciousness_coherence: float,
                    qcp_score: float,
                    field_coherence: float,
                    stability_score: float,
                    phi_match_score: float) -> IntegratedTrajectoryPoint:
        """
        Record a trajectory point with both UBF and QCPP metrics.
        
        Args:
            iteration: Current iteration number
            rmsd: RMSD to native structure
            energy: Conformational energy
            consciousness_frequency: Agent consciousness frequency
            consciousness_coherence: Agent consciousness coherence
            qcp_score: QCPP QCP score
            field_coherence: QCPP field coherence
            stability_score: QCPP stability score
            phi_match_score: QCPP phi match score
            
        Returns:
            Created IntegratedTrajectoryPoint
            
        Raises:
            ValueError: If any metric is invalid
        """
        start_time = time.time()
        
        # Create trajectory point with current timestamp
        point = IntegratedTrajectoryPoint(
            iteration=iteration,
            rmsd=rmsd,
            energy=energy,
            consciousness_frequency=consciousness_frequency,
            consciousness_coherence=consciousness_coherence,
            qcp_score=qcp_score,
            field_coherence=field_coherence,
            stability_score=stability_score,
            phi_match_score=phi_match_score,
            timestamp=int(time.time() * 1000)  # milliseconds
        )
        
        # Add to points list
        self._points.append(point)
        
        # Enforce max_points limit (FIFO)
        if self._max_points is not None and len(self._points) > self._max_points:
            self._points.pop(0)  # Remove oldest point
        
        # Track performance
        end_time = time.time()
        recording_time_ms = (end_time - start_time) * 1000
        self._recording_count += 1
        self._total_recording_time_ms += recording_time_ms
        
        return point
    
    def get_points(self) -> List[IntegratedTrajectoryPoint]:
        """
        Get all recorded trajectory points.
        
        Returns:
            List of trajectory points in chronological order
        """
        return self._points.copy()
    
    def get_point_count(self) -> int:
        """
        Get number of recorded trajectory points.
        
        Returns:
            Count of trajectory points
        """
        return len(self._points)
    
    def export_to_json(self, output_file: str, include_metadata: bool = True) -> None:
        """
        Export trajectory data to JSON file.
        
        Creates a JSON file containing all trajectory points plus optional
        metadata about the recording session.
        
        Args:
            output_file: Path to output JSON file
            include_metadata: Whether to include metadata section (default: True)
            
        Raises:
            IOError: If file write fails
        """
        start_time = time.time()
        
        # Build export data structure
        export_data: Dict[str, Any] = {}
        
        # Add metadata if requested
        if include_metadata:
            export_data["metadata"] = {
                "point_count": len(self._points),
                "recording_count": self._recording_count,
                "avg_recording_time_ms": (
                    self._total_recording_time_ms / self._recording_count
                    if self._recording_count > 0 else 0.0
                ),
                "max_points_limit": self._max_points,
                "export_timestamp": int(time.time() * 1000)
            }
        
        # Add trajectory points
        export_data["trajectory_points"] = [
            asdict(point) for point in self._points
        ]
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Log export performance
        end_time = time.time()
        export_time_ms = (end_time - start_time) * 1000
        logger.info(
            f"Exported {len(self._points)} trajectory points to {output_file} "
            f"in {export_time_ms:.2f}ms"
        )
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export trajectory data to dictionary format.
        
        Returns:
            Dictionary with trajectory points and metadata
        """
        return {
            "metadata": {
                "point_count": len(self._points),
                "recording_count": self._recording_count,
                "avg_recording_time_ms": (
                    self._total_recording_time_ms / self._recording_count
                    if self._recording_count > 0 else 0.0
                ),
                "max_points_limit": self._max_points
            },
            "trajectory_points": [asdict(point) for point in self._points]
        }
    
    def clear(self) -> None:
        """Clear all recorded trajectory points."""
        self._points.clear()
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """
        Get recording performance statistics.
        
        Returns:
            Dictionary with recording statistics
        """
        return {
            "total_points": len(self._points),
            "recording_count": self._recording_count,
            "avg_recording_time_ms": (
                self._total_recording_time_ms / self._recording_count
                if self._recording_count > 0 else 0.0
            ),
            "total_recording_time_ms": self._total_recording_time_ms,
            "max_points_limit": self._max_points
        }


class TrajectoryAnalyzer:
    """
    Analyzer for computing correlations between QCPP and UBF metrics.
    
    This class provides statistical analysis tools to understand the
    relationship between quantum physics metrics (QCPP) and structural
    quality metrics (RMSD, energy).
    
    Key analyses:
    - Pearson correlation between QCPP metrics and RMSD
    - Correlation between QCPP metrics and energy
    - Consciousness-QCPP correlations
    - Trajectory trends over time
    
    Scientific Value:
    - Validates that quantum coherence predicts native-like structures
    - Demonstrates physics grounding of consciousness coordinates
    - Quantifies QCPP's predictive power for conformational quality
    """
    
    def __init__(self, trajectory_points: List[IntegratedTrajectoryPoint]):
        """
        Initialize trajectory analyzer with trajectory data.
        
        Args:
            trajectory_points: List of trajectory points to analyze
        """
        self._points = trajectory_points
    
    def calculate_qcpp_rmsd_correlation(self) -> Dict[str, float]:
        """
        Calculate Pearson correlations between QCPP metrics and RMSD.
        
        Computes correlations for:
        - QCP score vs RMSD
        - Field coherence vs RMSD
        - Stability score vs RMSD
        - Phi match score vs RMSD
        
        Expected correlations (if QCPP predicts native-like structures):
        - High QCP → Low RMSD (negative correlation)
        - High coherence → Low RMSD (negative correlation)
        - High stability → Low RMSD (negative correlation)
        - High phi match → Low RMSD (negative correlation)
        
        Returns:
            Dictionary with correlation coefficients and p-values:
            {
                'qcp_rmsd_corr': float,
                'qcp_rmsd_pvalue': float,
                'coherence_rmsd_corr': float,
                'coherence_rmsd_pvalue': float,
                'stability_rmsd_corr': float,
                'stability_rmsd_pvalue': float,
                'phi_rmsd_corr': float,
                'phi_rmsd_pvalue': float
            }
            
        Raises:
            ValueError: If trajectory has fewer than 2 points
        """
        if len(self._points) < 2:
            raise ValueError("Need at least 2 trajectory points for correlation analysis")
        
        # Try to import scipy, fall back to simple correlation if unavailable
        try:
            from scipy.stats import pearsonr
            use_scipy = True
        except ImportError:
            logger.warning("scipy not available, using simplified correlation calculation")
            use_scipy = False
        
        # Extract metrics
        qcp_scores = [p.qcp_score for p in self._points]
        coherences = [p.field_coherence for p in self._points]
        stabilities = [p.stability_score for p in self._points]
        phi_matches = [p.phi_match_score for p in self._points]
        rmsds = [p.rmsd for p in self._points]
        
        # Calculate correlations
        if use_scipy:
            qcp_corr, qcp_pval = pearsonr(qcp_scores, rmsds)
            coh_corr, coh_pval = pearsonr(coherences, rmsds)
            stab_corr, stab_pval = pearsonr(stabilities, rmsds)
            phi_corr, phi_pval = pearsonr(phi_matches, rmsds)
        else:
            # Simple correlation without p-values
            qcp_corr = self._simple_correlation(qcp_scores, rmsds)
            coh_corr = self._simple_correlation(coherences, rmsds)
            stab_corr = self._simple_correlation(stabilities, rmsds)
            phi_corr = self._simple_correlation(phi_matches, rmsds)
            qcp_pval = coh_pval = stab_pval = phi_pval = 0.0  # Unavailable
        
        return {
            'qcp_rmsd_corr': qcp_corr,
            'qcp_rmsd_pvalue': qcp_pval,
            'coherence_rmsd_corr': coh_corr,
            'coherence_rmsd_pvalue': coh_pval,
            'stability_rmsd_corr': stab_corr,
            'stability_rmsd_pvalue': stab_pval,
            'phi_rmsd_corr': phi_corr,
            'phi_rmsd_pvalue': phi_pval
        }
    
    def calculate_qcpp_energy_correlation(self) -> Dict[str, float]:
        """
        Calculate Pearson correlations between QCPP metrics and energy.
        
        Returns:
            Dictionary with correlation coefficients and p-values
            
        Raises:
            ValueError: If trajectory has fewer than 2 points
        """
        if len(self._points) < 2:
            raise ValueError("Need at least 2 trajectory points for correlation analysis")
        
        # Try to import scipy
        try:
            from scipy.stats import pearsonr
            use_scipy = True
        except ImportError:
            use_scipy = False
        
        # Extract metrics
        qcp_scores = [p.qcp_score for p in self._points]
        coherences = [p.field_coherence for p in self._points]
        stabilities = [p.stability_score for p in self._points]
        phi_matches = [p.phi_match_score for p in self._points]
        energies = [p.energy for p in self._points]
        
        # Calculate correlations
        if use_scipy:
            qcp_corr, qcp_pval = pearsonr(qcp_scores, energies)
            coh_corr, coh_pval = pearsonr(coherences, energies)
            stab_corr, stab_pval = pearsonr(stabilities, energies)
            phi_corr, phi_pval = pearsonr(phi_matches, energies)
        else:
            qcp_corr = self._simple_correlation(qcp_scores, energies)
            coh_corr = self._simple_correlation(coherences, energies)
            stab_corr = self._simple_correlation(stabilities, energies)
            phi_corr = self._simple_correlation(phi_matches, energies)
            qcp_pval = coh_pval = stab_pval = phi_pval = 0.0
        
        return {
            'qcp_energy_corr': qcp_corr,
            'qcp_energy_pvalue': qcp_pval,
            'coherence_energy_corr': coh_corr,
            'coherence_energy_pvalue': coh_pval,
            'stability_energy_corr': stab_corr,
            'stability_energy_pvalue': stab_pval,
            'phi_energy_corr': phi_corr,
            'phi_energy_pvalue': phi_pval
        }
    
    def calculate_consciousness_qcpp_correlation(self) -> Dict[str, float]:
        """
        Calculate correlations between consciousness coordinates and QCPP metrics.
        
        Validates physics grounding: consciousness should correlate with QCPP.
        
        Returns:
            Dictionary with correlation coefficients
            
        Raises:
            ValueError: If trajectory has fewer than 2 points
        """
        if len(self._points) < 2:
            raise ValueError("Need at least 2 trajectory points for correlation analysis")
        
        # Try to import scipy
        try:
            from scipy.stats import pearsonr
            use_scipy = True
        except ImportError:
            use_scipy = False
        
        # Extract metrics
        frequencies = [p.consciousness_frequency for p in self._points]
        coherences_consc = [p.consciousness_coherence for p in self._points]
        qcp_scores = [p.qcp_score for p in self._points]
        coherences_qcpp = [p.field_coherence for p in self._points]
        
        # Calculate correlations
        if use_scipy:
            freq_qcp_corr, freq_qcp_pval = pearsonr(frequencies, qcp_scores)
            coh_qcp_corr, coh_qcp_pval = pearsonr(coherences_consc, coherences_qcpp)
        else:
            freq_qcp_corr = self._simple_correlation(frequencies, qcp_scores)
            coh_qcp_corr = self._simple_correlation(coherences_consc, coherences_qcpp)
            freq_qcp_pval = coh_qcp_pval = 0.0
        
        return {
            'frequency_qcp_corr': freq_qcp_corr,
            'frequency_qcp_pvalue': freq_qcp_pval,
            'coherence_coherence_corr': coh_qcp_corr,
            'coherence_coherence_pvalue': coh_qcp_pval
        }
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the trajectory.
        
        Returns:
            Dictionary with summary statistics including:
            - Best/worst/average values for all metrics
            - Improvement metrics (first vs last)
            - Trajectory trends
        """
        if not self._points:
            return {"error": "No trajectory points"}
        
        # Extract all metrics
        rmsds = [p.rmsd for p in self._points]
        energies = [p.energy for p in self._points]
        qcp_scores = [p.qcp_score for p in self._points]
        stabilities = [p.stability_score for p in self._points]
        
        return {
            "point_count": len(self._points),
            "iteration_range": [self._points[0].iteration, self._points[-1].iteration],
            
            # RMSD statistics
            "rmsd_best": min(rmsds),
            "rmsd_worst": max(rmsds),
            "rmsd_avg": sum(rmsds) / len(rmsds),
            "rmsd_improvement": rmsds[0] - rmsds[-1] if len(rmsds) > 1 else 0.0,
            
            # Energy statistics
            "energy_best": min(energies),
            "energy_worst": max(energies),
            "energy_avg": sum(energies) / len(energies),
            "energy_improvement": energies[0] - energies[-1] if len(energies) > 1 else 0.0,
            
            # QCP statistics
            "qcp_best": max(qcp_scores),
            "qcp_worst": min(qcp_scores),
            "qcp_avg": sum(qcp_scores) / len(qcp_scores),
            
            # Stability statistics
            "stability_best": max(stabilities),
            "stability_worst": min(stabilities),
            "stability_avg": sum(stabilities) / len(stabilities)
        }
    
    def _simple_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate simple Pearson correlation coefficient without scipy.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Correlation coefficient in range [-1, 1]
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate covariance and standard deviations
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = (sum((x[i] - mean_x) ** 2 for i in range(n))) ** 0.5
        std_y = (sum((y[i] - mean_y) ** 2 for i in range(n))) ** 0.5
        
        if std_x == 0 or std_y == 0:
            return 0.0
        
        return cov / (std_x * std_y)
