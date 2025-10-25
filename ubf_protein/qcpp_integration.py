"""
QCPP Integration Layer for UBF System

This module provides the integration layer between the Quantum Coherence Protein
Predictor (QCPP) and the Universal Behavioral Framework (UBF). It wraps QCPP
functionality for use in real-time conformational exploration.

Key Components:
- QCPPMetrics: Data class containing QCPP analysis results
- QCPPIntegrationAdapter: Main adapter class providing QCPP functionality to UBF
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Any, List
from functools import lru_cache
import hashlib
import numpy as np
import logging
import time

if TYPE_CHECKING:
    from .models import Conformation

# Setup logger for performance monitoring
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QCPPMetrics:
    """
    QCPP analysis results for a protein conformation.
    
    All fields are immutable to support caching.
    
    Attributes:
        qcp_score: Quantum Consciousness Potential score (typically 3-8)
        field_coherence: Field coherence metric (-1 to 1, normalized)
        stability_score: Overall stability prediction (higher = more stable)
        phi_match_score: Golden ratio angle matching score (0-1)
        calculation_time_ms: Time taken for QCPP analysis (performance tracking)
    """
    qcp_score: float
    field_coherence: float
    stability_score: float
    phi_match_score: float
    calculation_time_ms: float
    
    def __post_init__(self):
        """Validate metrics are in expected ranges."""
        # QCP typically ranges from 3-8 based on formula: 4 + (2^n × φ^l × m)
        if not (0.0 <= self.qcp_score <= 20.0):
            raise ValueError(f"qcp_score {self.qcp_score} outside expected range [0, 20]")
        
        # Coherence is normalized to [-1, 1]
        if not (-2.0 <= self.field_coherence <= 2.0):
            raise ValueError(f"field_coherence {self.field_coherence} outside expected range [-2, 2]")
        
        # Stability score should be positive
        if self.stability_score < 0:
            raise ValueError(f"stability_score {self.stability_score} should be non-negative")
        
        # Phi match is a probability-like score
        if not (0.0 <= self.phi_match_score <= 1.0):
            raise ValueError(f"phi_match_score {self.phi_match_score} outside expected range [0, 1]")
        
        # Calculation time should be positive
        if self.calculation_time_ms < 0:
            raise ValueError(f"calculation_time_ms should be non-negative")


class QCPPIntegrationAdapter:
    """
    Adapter for integrating QCPP with UBF system.
    
    This class wraps the QuantumCoherenceProteinPredictor and provides
    optimized, cached access to QCPP metrics for real-time conformational
    exploration.
    
    Features:
    - LRU caching of conformation analysis for performance
    - Fallback to default metrics on QCPP calculation failures
    - Performance monitoring via calculation_time_ms
    - Thread-safe operation (caching is thread-safe)
    
    Performance Targets:
    - QCPP analysis: <5ms per conformation (cached: <10μs)
    - Memory overhead: ~1KB per cached conformation
    - Cache hit rate: >80% for typical exploration
    """
    
    def __init__(self, predictor: Any, cache_size: int = 1000):
        """
        Initialize QCPP integration adapter.
        
        Args:
            predictor: Instance of QuantumCoherenceProteinPredictor (or mock)
            cache_size: Maximum number of conformations to cache (default: 1000)
        """
        self.predictor = predictor
        self.cache_size = cache_size
        
        # Statistics tracking
        self.analysis_count = 0
        self.cache_hits = 0
        self.total_calculation_time_ms = 0.0
        
        # Performance monitoring (Task 12)
        self.slow_analyses_count = 0  # Count of analyses exceeding 5ms
        self.calculation_times: List[float] = []  # Recent calculation times
        self.max_calculation_time_ms = 0.0  # Peak calculation time
        self.performance_warning_threshold_ms = 5.0  # Threshold for warnings
        
        # Golden ratio constant (cached for performance)
        self.phi = (1 + np.sqrt(5)) / 2  # ≈ 1.618
        
    def _hash_conformation(self, conformation: 'Conformation') -> str:
        """
        Create a hash key for a conformation to use in caching.
        
        Uses atom coordinates as the unique identifier. Two conformations
        with identical atom positions will have the same hash.
        
        Args:
            conformation: Conformation to hash
            
        Returns:
            SHA256 hash string (first 16 characters for compactness)
        """
        # Convert coordinates to bytes for hashing
        coords_array = np.array(conformation.atom_coordinates, dtype=np.float32)
        coords_bytes = coords_array.tobytes()
        
        # Create hash
        hash_obj = hashlib.sha256(coords_bytes)
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for speed
    
    @lru_cache(maxsize=1000)
    def _analyze_conformation_cached(self, conformation_hash: str, 
                                     coords_tuple: tuple) -> QCPPMetrics:
        """
        Internal cached method for QCPP analysis.
        
        This method is decorated with lru_cache for automatic caching.
        The conformation is represented by its hash and coordinate tuple
        to make it hashable for caching.
        
        Args:
            conformation_hash: Hash of the conformation
            coords_tuple: Tuple of coordinate tuples (hashable representation)
            
        Returns:
            QCPPMetrics containing analysis results
        """
        import time
        start_time = time.time()
        
        # Reconstruct conformation data for QCPP analysis
        # Note: This is a simplified version - in production, we'd need to
        # reconstruct the full QCPP structure from the conformation
        
        # For now, we'll use a simplified analysis based on coordinate geometry
        coords = np.array(coords_tuple)
        
        # Calculate QCP score (simplified version)
        qcp_score = self._calculate_qcp_score(coords)
        
        # Calculate field coherence (simplified version)
        field_coherence = self._calculate_field_coherence(coords)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(qcp_score, field_coherence)
        
        # Calculate phi match score
        phi_match_score = self._calculate_phi_match_score(coords)
        
        # Calculate time taken
        end_time = time.time()
        calculation_time_ms = (end_time - start_time) * 1000
        
        self.total_calculation_time_ms += calculation_time_ms
        
        return QCPPMetrics(
            qcp_score=qcp_score,
            field_coherence=field_coherence,
            stability_score=stability_score,
            phi_match_score=phi_match_score,
            calculation_time_ms=calculation_time_ms
        )
    
    def analyze_conformation(self, conformation: 'Conformation') -> QCPPMetrics:
        """
        Analyze a conformation using QCPP and return metrics.
        
        This method uses LRU caching to avoid redundant calculations.
        Results are cached based on the conformation's atomic coordinates.
        
        Task 12: Now includes performance monitoring with timing and warnings.
        
        Args:
            conformation: Conformation to analyze
            
        Returns:
            QCPPMetrics containing QCP score, coherence, stability, and phi match
            
        Raises:
            ValueError: If conformation is invalid or analysis fails
        """
        # Task 12: Start timing for performance monitoring
        start_time = time.perf_counter()
        
        self.analysis_count += 1
        
        # Create hash and hashable representation
        conf_hash = self._hash_conformation(conformation)
        coords_tuple = tuple(tuple(coord) for coord in conformation.atom_coordinates)
        
        # Check if this is a cache hit (for statistics)
        cache_info = self._analyze_conformation_cached.cache_info()
        previous_hits = cache_info.hits
        
        # Call cached method
        try:
            metrics = self._analyze_conformation_cached(conf_hash, coords_tuple)
            
            # Update cache hit statistics
            new_cache_info = self._analyze_conformation_cached.cache_info()
            is_cache_hit = new_cache_info.hits > previous_hits
            if is_cache_hit:
                self.cache_hits += 1
            
            # Task 12: Record timing and check threshold
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            
            # Store calculation time for monitoring (keep last 100)
            self.calculation_times.append(total_time_ms)
            if len(self.calculation_times) > 100:
                self.calculation_times.pop(0)
            
            # Update peak calculation time
            if total_time_ms > self.max_calculation_time_ms:
                self.max_calculation_time_ms = total_time_ms
            
            # Task 12: Log warning if exceeds threshold (only for non-cached calls)
            if not is_cache_hit and total_time_ms > self.performance_warning_threshold_ms:
                self.slow_analyses_count += 1
                logger.warning(
                    f"QCPP analysis exceeded {self.performance_warning_threshold_ms}ms threshold: "
                    f"{total_time_ms:.2f}ms (analysis #{self.analysis_count})"
                )
            
            return metrics
            
        except Exception as e:
            # Fallback to default metrics on failure
            logger.error(f"QCPP analysis failed: {e}. Using default metrics.")
            return QCPPMetrics(
                qcp_score=4.0,  # Base energy level
                field_coherence=0.0,  # Neutral coherence
                stability_score=1.0,  # Neutral stability
                phi_match_score=0.5,  # Medium phi match
                calculation_time_ms=0.0
            )
    
    def calculate_quantum_alignment(self, qcpp_metrics: QCPPMetrics) -> float:
        """
        Calculate quantum alignment factor from QCPP metrics for move evaluation.
        
        This factor replaces the QAAP-based quantum alignment in the move evaluator.
        Formula: 0.5 + min(1.0, (qcp/5.0)*0.4 + (coherence+1.0)*0.3 + phi_match*0.3)
        
        Result range: [0.5, 1.5]
        - 0.5: Poor quantum alignment (low QCP, low coherence, poor phi match)
        - 1.0: Neutral alignment (baseline)
        - 1.5: Excellent alignment (high QCP, high coherence, strong phi match)
        
        Args:
            qcpp_metrics: QCPP metrics from analyze_conformation()
            
        Returns:
            Quantum alignment factor in range [0.5, 1.5]
        """
        # Normalize QCP score (divide by 5 to get typical range 0.6-1.6)
        qcp_component = (qcpp_metrics.qcp_score / 5.0) * 0.4
        
        # Normalize coherence (shift from [-1,1] to [0,2], divide by 2)
        coherence_component = ((qcpp_metrics.field_coherence + 1.0) / 2.0) * 0.3
        
        # Phi match is already in [0,1]
        phi_component = qcpp_metrics.phi_match_score * 0.3
        
        # Combine components (max contribution: 0.4+0.3+0.3 = 1.0)
        alignment = 0.5 + min(1.0, qcp_component + coherence_component + phi_component)
        
        return alignment
    
    def get_stability_score(self, qcpp_metrics: QCPPMetrics) -> float:
        """
        Get stability score from QCPP metrics.
        
        Convenience method to extract stability score for dynamic parameter
        adjustment and memory significance calculations.
        
        Args:
            qcpp_metrics: QCPP metrics from analyze_conformation()
            
        Returns:
            Stability score (higher = more stable)
        """
        return qcpp_metrics.stability_score
    
    def get_phi_match_score(self, qcpp_metrics: QCPPMetrics) -> float:
        """
        Get phi match score from QCPP metrics.
        
        Convenience method to extract phi match score for phi pattern rewards.
        
        Args:
            qcpp_metrics: QCPP metrics from analyze_conformation()
            
        Returns:
            Phi match score in range [0, 1]
        """
        return qcpp_metrics.phi_match_score
    
    def get_cache_stats(self) -> dict:
        """
        Get caching and performance statistics for monitoring.
        
        Task 12: Enhanced with detailed performance metrics including
        slow analysis tracking, peak times, and recent calculation times.
        
        Returns:
            Dictionary with cache and performance statistics:
            - total_analyses: Total number of analyze_conformation() calls
            - cache_hits: Number of cache hits
            - cache_hit_rate: Percentage of cache hits
            - avg_calculation_time_ms: Average calculation time
            - max_calculation_time_ms: Peak calculation time
            - slow_analyses_count: Number of analyses exceeding threshold
            - slow_analyses_rate: Percentage of slow analyses
            - recent_avg_time_ms: Average of last 100 calculations
            - performance_warning_threshold_ms: Current warning threshold
        """
        cache_info = self._analyze_conformation_cached.cache_info()
        
        # Calculate recent average (last 100 calculations)
        recent_avg = 0.0
        if self.calculation_times:
            recent_avg = sum(self.calculation_times) / len(self.calculation_times)
        
        return {
            'total_analyses': self.analysis_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': (self.cache_hits / self.analysis_count * 100) if self.analysis_count > 0 else 0.0,
            'cache_size': cache_info.currsize,
            'cache_maxsize': cache_info.maxsize,
            'avg_calculation_time_ms': (self.total_calculation_time_ms / self.analysis_count) if self.analysis_count > 0 else 0.0,
            # Task 12: Enhanced performance metrics
            'max_calculation_time_ms': self.max_calculation_time_ms,
            'slow_analyses_count': self.slow_analyses_count,
            'slow_analyses_rate': (self.slow_analyses_count / self.analysis_count * 100) if self.analysis_count > 0 else 0.0,
            'recent_avg_time_ms': recent_avg,
            'performance_warning_threshold_ms': self.performance_warning_threshold_ms
        }
    
    def get_performance_recommendations(self) -> List[str]:
        """
        Analyze performance and provide optimization recommendations.
        
        Task 12: Provides actionable recommendations based on performance metrics.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if self.analysis_count == 0:
            return ["No analyses performed yet"]
        
        stats = self.get_cache_stats()
        
        # Check slow analyses
        if stats['slow_analyses_rate'] > 10.0:
            recommendations.append(
                f"High slow analysis rate ({stats['slow_analyses_rate']:.1f}%): "
                "Consider increasing analysis frequency interval or cache size"
            )
        
        # Check cache hit rate
        if stats['cache_hit_rate'] < 20.0 and self.analysis_count > 50:
            recommendations.append(
                f"Low cache hit rate ({stats['cache_hit_rate']:.1f}%): "
                "Consider increasing cache size for better performance"
            )
        elif stats['cache_hit_rate'] > 70.0 and self.analysis_count > 50:
            recommendations.append(
                f"Excellent cache hit rate ({stats['cache_hit_rate']:.1f}%): "
                "Current cache size is working well"
            )
        
        # Check average calculation time
        if stats['recent_avg_time_ms'] > 3.0:
            recommendations.append(
                f"Recent calculations averaging {stats['recent_avg_time_ms']:.2f}ms: "
                "Consider reducing analysis frequency to maintain performance"
            )
        elif stats['recent_avg_time_ms'] < 1.0:
            recommendations.append(
                f"Fast calculations ({stats['recent_avg_time_ms']:.2f}ms avg): "
                "Can afford to analyze more frequently"
            )
        
        # Check peak time
        if stats['max_calculation_time_ms'] > 10.0:
            recommendations.append(
                f"Peak calculation time very high ({stats['max_calculation_time_ms']:.2f}ms): "
                "Some conformations are expensive to analyze"
            )
        
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges")
        
        return recommendations
    
    def should_reduce_analysis_frequency(self) -> bool:
        """
        Determine if analysis frequency should be reduced for performance.
        
        Task 12: Adaptive frequency adjustment based on calculation time.
        
        Returns:
            True if recent calculations are slow and frequency should be reduced
        """
        if len(self.calculation_times) < 10:
            return False  # Not enough data
        
        stats = self.get_cache_stats()
        
        # Reduce frequency if recent calculations are consistently slow
        return stats['recent_avg_time_ms'] > 4.0 or stats['slow_analyses_rate'] > 20.0
    
    # ========================================================================
    # Private helper methods for simplified QCPP calculations
    # ========================================================================
    
    def _calculate_qcp_score(self, coords: np.ndarray) -> float:
        """
        Calculate simplified QCP score based on geometry.
        
        Full QCP calculation: QCP = 4 + (2^n × φ^l × m)
        This simplified version estimates QCP from coordinate compactness
        and spatial distribution.
        
        Args:
            coords: Array of atomic coordinates (N x 3)
            
        Returns:
            QCP score (typically 3-8)
        """
        if len(coords) == 0:
            return 4.0  # Base energy level
        
        # Calculate radius of gyration (compactness measure)
        centroid = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        rg = np.sqrt(np.mean(distances ** 2))
        
        # Calculate spatial organization (relates to n in QCP formula)
        # More organized = higher n = higher QCP
        organization = 1.0 / (1.0 + rg / 10.0)
        
        # Estimate QCP: base (4.0) + structural contribution
        qcp = 4.0 + organization * 2.0 * (self.phi ** 1.5)
        
        return float(qcp)
    
    def _calculate_field_coherence(self, coords: np.ndarray) -> float:
        """
        Calculate simplified field coherence from coordinate geometry.
        
        Full coherence: C = ∑(ψᵢ × e^(iφ)) × D(t)
        This simplified version estimates coherence from spatial regularity.
        
        Args:
            coords: Array of atomic coordinates (N x 3)
            
        Returns:
            Field coherence in range [-1, 1]
        """
        if len(coords) < 2:
            return 0.0
        
        # Calculate pairwise distances
        n = len(coords)
        total_coherence = 0.0
        count = 0
        
        for i in range(min(n, 50)):  # Sample first 50 for performance
            for j in range(i+1, min(n, 50)):
                dist = np.linalg.norm(coords[i] - coords[j])
                
                # Phi-based phase factor
                phase = np.cos(2 * np.pi * dist / (3.8 * self.phi))
                
                # Resonance coupling (simplified)
                resonance = np.exp(-abs(dist - 5.0) / 2.0)
                
                total_coherence += phase * resonance
                count += 1
        
        if count == 0:
            return 0.0
        
        # Normalize to [-1, 1]
        coherence = total_coherence / count
        coherence = max(-1.0, min(1.0, coherence))
        
        return float(coherence)
    
    def _calculate_stability_score(self, qcp: float, coherence: float) -> float:
        """
        Calculate stability score from QCP and coherence.
        
        Stability formula: 0.4 × QCP + 0.4 × coherence + 0.2 × phi_score
        For this helper, we use a simplified version without phi_score.
        
        Args:
            qcp: QCP score
            coherence: Field coherence
            
        Returns:
            Stability score (typically 0-3)
        """
        # Normalize coherence from [-1,1] to [0,1]
        norm_coherence = (coherence + 1.0) / 2.0
        
        # Combine QCP and coherence (weights: 0.5, 0.5 without phi)
        stability = 0.5 * qcp + 0.5 * norm_coherence * 4.0
        
        return float(stability)
    
    def _calculate_phi_match_score(self, coords: np.ndarray) -> float:
        """
        Calculate phi angle matching score.
        
        Looks for angles close to 137.5° and 222.5° (golden ratio angles)
        between residue triplets.
        
        Args:
            coords: Array of atomic coordinates (N x 3)
            
        Returns:
            Phi match score in range [0, 1]
        """
        if len(coords) < 3:
            return 0.5  # Default for insufficient data
        
        phi_angle_deg = 2 * 180 / self.phi  # ≈ 137.5°
        target_angles = [phi_angle_deg, 360 - phi_angle_deg]  # 137.5° and 222.5°
        
        matches = 0
        total = 0
        
        # Sample triplets for performance
        step = max(1, len(coords) // 20)  # Sample ~20 triplets
        
        for i in range(0, len(coords) - 2, step):
            j = min(i + 1, len(coords) - 2)
            k = min(j + 1, len(coords) - 1)
            
            # Calculate angle between vectors
            v1 = coords[i] - coords[j]
            v2 = coords[k] - coords[j]
            
            # Normalize
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Calculate angle
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle_rad = np.arccos(dot)
                angle_deg = angle_rad * 180 / np.pi
                
                # Check match to target angles
                best_match = min(abs(angle_deg - target_angles[0]), 
                               abs(angle_deg - target_angles[1]))
                
                if best_match < 15.0:  # Within 15° tolerance
                    match_score = np.exp(-0.01 * best_match**2)
                    if match_score > 0.5:
                        matches += 1
                
                total += 1
        
        if total == 0:
            return 0.5
        
        # Return ratio of good matches
        phi_match = matches / total
        
        return float(phi_match)
    
    def clear_cache(self):
        """Clear the LRU cache to free memory."""
        self._analyze_conformation_cached.cache_clear()
        self.cache_hits = 0
