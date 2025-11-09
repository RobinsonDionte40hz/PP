"""
Mediator Agent Module

This module implements Mediator Agents - specialized agents for pattern detection
and information relay in the UBF protein folding system.

Mediator Agents:
- Detect THz resonance patterns
- Identify folding dynamics (secondary structure formation)
- Find geometric similarities between conformations
- Relay information to QCPP for validation
- Broadcast patterns to exploration agents via shared memory

Author: UBF Protein System
Date: November 9, 2025
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import time
import hashlib
import numpy as np
from sklearn.cluster import DBSCAN

from ubf_protein.interfaces import IProteinAgent, IConsciousnessState, IBehavioralState, IMemorySystem, ISharedMemoryPool
from ubf_protein.models import Conformation, ConformationalOutcome, ConformationalMove
from ubf_protein.consciousness import ConsciousnessState
from ubf_protein.behavioral_state import BehavioralState
from ubf_protein.memory_system import MemorySystem
from ubf_protein.mediator_config import MediatorConfig
from ubf_protein.pattern_detection import (
    PatternDetection, PatternType, PatternSignificance,
    THzResonanceData, FoldingDynamicsData, GeometricSimilarityData
)
from ubf_protein.geometric_attractor import GeometricAttractorAnalyzer
from ubf_protein.rmsd_calculator import RMSDCalculator, RMSDResult


class BroadcastThrottler:
    """
    Throttle mechanism to prevent shared memory overflow.
    
    This class implements a sliding window rate limiter that prevents Mediator Agents
    from overwhelming the shared memory pool with too many broadcasts.
    
    The throttler:
    - Tracks broadcast timestamps in a sliding window (default 1 second)
    - Enforces maximum broadcasts per window (from config)
    - Allows high-significance patterns to bypass throttle
    - Automatically expires old timestamps
    
    Attributes:
        max_broadcasts_per_window: Maximum broadcasts allowed in window
        window_duration: Sliding window duration in seconds (default 1.0)
        broadcast_times: List of timestamps of recent broadcasts
        high_priority_bypass: Allow HIGH significance patterns to bypass throttle
    
    Example:
        >>> throttler = BroadcastThrottler(max_broadcasts_per_window=10)
        >>> 
        >>> # Check if broadcast is allowed
        >>> if throttler.can_broadcast():
        ...     throttler.record_broadcast()
        ...     # ... perform broadcast
        >>> 
        >>> # High priority pattern can bypass
        >>> if throttler.can_broadcast(priority='high'):
        ...     # Always allowed for high priority
        ...     throttler.record_broadcast()
    """
    
    def __init__(
        self,
        max_broadcasts_per_window: int = 10,
        window_duration: float = 1.0,
        high_priority_bypass: bool = True
    ):
        """
        Initialize broadcast throttler.
        
        Args:
            max_broadcasts_per_window: Maximum broadcasts allowed per window
            window_duration: Sliding window duration in seconds
            high_priority_bypass: Allow high priority to bypass throttle
        """
        self.max_broadcasts_per_window = max_broadcasts_per_window
        self.window_duration = window_duration
        self.high_priority_bypass = high_priority_bypass
        self.broadcast_times: List[float] = []
    
    def can_broadcast(self, priority: Optional[str] = None) -> bool:
        """
        Check if broadcast is allowed based on current rate.
        
        Args:
            priority: Pattern priority ('high', 'medium', 'low', None)
        
        Returns:
            True if broadcast is allowed, False if throttled
        """
        # High priority patterns bypass throttle
        if self.high_priority_bypass and priority == 'high':
            return True
        
        # Clean up expired timestamps
        current_time = time.time()
        self._expire_old_timestamps(current_time)
        
        # Check if we're under the limit
        return len(self.broadcast_times) < self.max_broadcasts_per_window
    
    def record_broadcast(self) -> None:
        """
        Record a broadcast timestamp.
        
        Call this immediately after a successful broadcast to update the throttle state.
        """
        current_time = time.time()
        self.broadcast_times.append(current_time)
        
        # Clean up old timestamps periodically
        self._expire_old_timestamps(current_time)
    
    def prioritize_message(self, pattern: PatternDetection) -> str:
        """
        Determine priority level for a pattern.
        
        Args:
            pattern: Pattern to prioritize
        
        Returns:
            Priority string: 'high', 'medium', or 'low'
        """
        return pattern.significance.value
    
    def get_current_rate(self) -> float:
        """
        Get current broadcast rate (broadcasts per second).
        
        Returns:
            Current rate in broadcasts/second
        """
        current_time = time.time()
        self._expire_old_timestamps(current_time)
        
        if self.window_duration > 0:
            return len(self.broadcast_times) / self.window_duration
        return 0.0
    
    def reset(self) -> None:
        """
        Reset throttle state (clear all timestamps).
        
        Useful for testing or when starting a new exploration phase.
        """
        self.broadcast_times.clear()
    
    def _expire_old_timestamps(self, current_time: float) -> None:
        """
        Remove timestamps outside the sliding window.
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - self.window_duration
        self.broadcast_times = [
            t for t in self.broadcast_times
            if t > cutoff_time
        ]


class MediatorAgent(IProteinAgent):
    """
    Mediator Agent for pattern detection and information relay.
    
    Mediator Agents operate alongside exploration agents to detect emergent patterns
    in protein conformations and relay information between QCPP physics engine and
    autonomous exploration agents.
    
    Key responsibilities:
    - Pattern Detection: Identify THz resonance, folding dynamics, geometric similarities
    - QCPP Relay: Request physics-based validation for significant patterns
    - Information Broadcast: Share patterns with exploration agents via shared memory
    - Performance Optimization: Cache pattern detections to minimize redundant calculations
    
    Attributes:
        protein_sequence: Amino acid sequence string
        config: Configuration parameters for detection and relay
        qcpp_adapter: Integration adapter for QCPP physics engine
        geometric_analyzer: Analyzer for geometric pattern detection
        shared_memory: Shared memory pool for inter-agent communication
        
        # Internal state
        consciousness: Consciousness coordinates (frequency, coherence)
        behavioral_state: Derived behavioral parameters
        memory_system: Memory storage (minimal for Mediators)
        pattern_cache: LRU cache for pattern detections
        reference_conformations: List of reference structures for RMSD comparison
        detection_statistics: Counters for detected patterns
    
    Example:
        >>> from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
        >>> from ubf_protein.geometric_attractor import GeometricAttractorAnalyzer
        >>> from ubf_protein.multi_agent_coordinator import SharedMemoryPool
        >>> 
        >>> # Initialize components
        >>> qcpp_adapter = QCPPIntegrationAdapter(...)
        >>> geo_analyzer = GeometricAttractorAnalyzer()
        >>> shared_mem = SharedMemoryPool()
        >>> config = MediatorConfig()
        >>> 
        >>> # Create Mediator Agent
        >>> mediator = MediatorAgent(
        ...     protein_sequence="ACDEFGHIKLM",
        ...     qcpp_adapter=qcpp_adapter,
        ...     geometric_analyzer=geo_analyzer,
        ...     shared_memory=shared_mem,
        ...     config=config
        ... )
        >>> 
        >>> # Detection cycle (called every N iterations)
        >>> outcome = mediator.explore_step()
    """
    
    def __init__(
        self,
        protein_sequence: str,
        qcpp_adapter: Any,  # QCPPIntegrationAdapter
        geometric_analyzer: GeometricAttractorAnalyzer,
        shared_memory: ISharedMemoryPool,
        config: Optional[MediatorConfig] = None,
    ):
        """
        Initialize Mediator Agent.
        
        Args:
            protein_sequence: Amino acid sequence string
            qcpp_adapter: QCPP integration adapter for physics validation
            geometric_analyzer: Geometric attractor analyzer instance
            shared_memory: Shared memory pool for inter-agent communication
            config: Configuration parameters (uses default if None)
        
        Raises:
            ValueError: If protein_sequence is empty or config is invalid
        """
        if not protein_sequence:
            raise ValueError("protein_sequence cannot be empty")
        
        self.protein_sequence = protein_sequence
        self.config = config or MediatorConfig()
        self.qcpp_adapter = qcpp_adapter
        self.geometric_analyzer = geometric_analyzer
        self.shared_memory = shared_memory
        
        # Initialize consciousness state (Mediators have high coherence, moderate frequency)
        # Frequency: 9.0 Hz (moderate exploration tempo, between cautious and aggressive)
        # Coherence: 0.8 (high behavioral consistency for reliable pattern detection)
        self.consciousness = ConsciousnessState(
            frequency=9.0,
            coherence=0.8
        )
        
        # Initialize behavioral state (derived from consciousness)
        self.behavioral_state = BehavioralState(self.consciousness.get_coordinates())
        
        # Initialize minimal memory system (Mediators don't explore, so minimal memory needed)
        self.memory_system = MemorySystem()
        
        # Initialize pattern cache (OrderedDict for LRU behavior)
        self.pattern_cache: OrderedDict = OrderedDict()
        
        # Initialize THz signature cache (conformation_hash -> THz spectrum data)
        self.thz_signature_cache: OrderedDict = OrderedDict()
        
        # Initialize reference conformations for geometric similarity detection
        # Each entry: {hash, coordinates, energy, geometric_score, timestamp, agent_id}
        self.reference_conformations: List[Dict[str, Any]] = []
        self.max_references = 100  # Limit from requirement 7.1
        
        # Initialize RMSD calculator with alignment enabled
        self.rmsd_calculator = RMSDCalculator(align_structures=True)
        
        # Initialize detection statistics
        self.detection_statistics = {
            'total_detections': 0,
            'thz_detections': 0,
            'folding_detections': 0,
            'geometric_detections': 0,
            'broadcasts': 0,
            'qcpp_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        # Current iteration counter (for pattern age calculation)
        self.current_iteration = 0
        
        # Broadcast throttle tracking
        self.broadcast_throttler = BroadcastThrottler(
            max_broadcasts_per_window=self.config.broadcast_throttle_rate,
            window_duration=1.0,  # 1 second window
            high_priority_bypass=True
        )
    
    # ========================================================================
    # IProteinAgent Interface Implementation
    # ========================================================================
    
    def get_consciousness_state(self) -> IConsciousnessState:
        """
        Get current consciousness coordinates.
        
        Returns:
            Consciousness state with frequency and coherence
        """
        return self.consciousness
    
    def get_behavioral_state(self) -> IBehavioralState:
        """
        Get cached behavioral state.
        
        Returns:
            Behavioral state derived from consciousness
        """
        return self.behavioral_state
    
    def get_memory_system(self) -> IMemorySystem:
        """
        Get agent's memory system.
        
        Returns:
            Memory system instance (minimal for Mediators)
        """
        return self.memory_system
    
    def explore_step(self) -> ConformationalOutcome:
        """
        Execute one detection cycle (called by coordinator).
        
        This method is called by MultiAgentCoordinator during parallel exploration.
        Mediators don't perform actual conformational moves - instead they detect
        patterns in the best conformations found by exploration agents.
        
        Returns:
            ConformationalOutcome with dummy move (no actual conformational change)
        
        Note:
            This is a placeholder implementation. The actual pattern detection
            logic will be implemented in Task 5.3.
        """
        # Placeholder: Mediators don't perform moves
        # Actual detection will be implemented in Task 5.3
        # For now, raise NotImplementedError to make it clear this needs implementation
        
        raise NotImplementedError(
            "Mediator Agent explore_step() will be implemented in Task 5.3. "
            "Mediators don't perform conformational moves - they detect patterns."
        )
    
    def get_current_conformation(self) -> Conformation:
        """
        Get current protein conformation.
        
        Note:
            Mediators don't maintain a current conformation - they analyze
            conformations from exploration agents. This raises NotImplementedError.
        
        Raises:
            NotImplementedError: Mediators don't have conformations
        """
        raise NotImplementedError(
            "Mediators don't maintain conformations - they analyze conformations "
            "from exploration agents"
        )
    
    def get_exploration_metrics(self) -> Dict[str, float]:
        """
        Get current exploration metrics.
        
        For Mediators, this returns detection statistics instead of
        exploration metrics.
        
        Returns:
            Dictionary with detection statistics
        """
        return {
            'total_detections': float(self.detection_statistics['total_detections']),
            'thz_detections': float(self.detection_statistics['thz_detections']),
            'folding_detections': float(self.detection_statistics['folding_detections']),
            'geometric_detections': float(self.detection_statistics['geometric_detections']),
            'broadcasts': float(self.detection_statistics['broadcasts']),
            'cache_hit_rate': self._get_cache_hit_rate(),
        }
    
    # ========================================================================
    # Mediator-Specific Methods
    # ========================================================================
    
    def detect_patterns(self, conformation: Conformation) -> List[PatternDetection]:
        """
        Detect all enabled patterns in a conformation.
        
        This is the main pattern detection entry point. Checks cache first,
        then performs expensive calculations if needed.
        
        Args:
            conformation: Protein conformation to analyze
        
        Returns:
            List of detected patterns (may be empty)
        
        Note:
            Implementation for Tasks 6-8 (pattern detection).
            Task 6 (THz) is now complete.
            Tasks 7-8 (Folding, Geometric) will be added later.
        """
        patterns = []
        
        # Generate conformation hash for caching
        conf_hash = self._generate_conformation_hash(conformation)
        
        # Check cache first
        if conf_hash in self.pattern_cache:
            self.detection_statistics['cache_hits'] += 1
            return self.pattern_cache[conf_hash]
        
        self.detection_statistics['cache_misses'] += 1
        
        # Task 6: THz resonance detection (COMPLETED)
        if self.config.enable_thz_detection:
            thz_pattern = self._detect_thz_resonance(conformation)
            if thz_pattern:
                patterns.append(thz_pattern)
                self.detection_statistics['thz_detections'] += 1
                self.detection_statistics['total_detections'] += 1

        # Task 7: Folding dynamics detection
        if self.config.enable_folding_detection:
            folding_pattern = self._detect_folding_dynamics(conformation)
            if folding_pattern:
                patterns.append(folding_pattern)
                self.detection_statistics['folding_detections'] += 1
                self.detection_statistics['total_detections'] += 1
        
        # Task 8: Geometric similarity detection
        if self.config.enable_geometric_detection:
            geometric_pattern = self._detect_geometric_similarity(conformation)
            if geometric_pattern:
                patterns.append(geometric_pattern)
                self.detection_statistics['geometric_detections'] += 1
                self.detection_statistics['total_detections'] += 1
        
        # Cache results
        self.pattern_cache[conf_hash] = patterns
        
        # Maintain cache size limit (LRU eviction)
        if len(self.pattern_cache) > self.config.cache_size:
            self.pattern_cache.popitem(last=False)  # Remove oldest
        
        return patterns
    
    def relay_to_qcpp(self, pattern: PatternDetection, conformation: Conformation) -> Optional[Any]:
        """
        Request QCPP analysis for detected pattern.
        
        This method validates detected patterns using the QCPP physics engine,
        providing quantum coherence predictions and structural stability metrics.
        
        The relay process:
        1. Extract conformation from pattern
        2. Invoke qcpp_adapter.analyze_conformation()
        3. Parse QCPP metrics (QCP score, coherence, stability)
        4. Return metrics for pattern augmentation
        5. Handle failures gracefully (log warning, continue)
        
        Args:
            pattern: Detected pattern to validate
            conformation: Conformation where pattern was detected
        
        Returns:
            QCPP metrics dictionary if successful, None if failed
            
            Metrics structure:
            {
                'qcp_score': float,
                'field_coherence': float,
                'stability_prediction': str,
                'golden_ratio_score': float,
                'thz_spectrum': dict (optional)
            }
        
        Example:
            >>> pattern = PatternDetection(...)
            >>> conformation = Conformation(...)
            >>> qcpp_metrics = mediator.relay_to_qcpp(pattern, conformation)
            >>> if qcpp_metrics:
            ...     print(f"QCP score: {qcpp_metrics['qcp_score']:.2f}")
        """
        try:
            # Increment validation counter
            self.detection_statistics['qcpp_validations'] += 1
            
            # Validate inputs
            if not pattern or not conformation:
                return None
            
            # Check if QCPP adapter is available
            if not self.qcpp_adapter:
                return None
            
            # Convert conformation to format expected by QCPP
            # QCPP expects dict with coordinates and optionally sequence
            conformation_dict = {
                'coordinates': conformation.atom_coordinates,
                'sequence': conformation.sequence if hasattr(conformation, 'sequence') else None
            }
            
            # Invoke QCPP analysis
            # Note: This depends on QCPPIntegrationAdapter implementation
            try:
                # Check if adapter has analyze_conformation method
                if not hasattr(self.qcpp_adapter, 'analyze_conformation'):
                    # Fallback: Try to extract metrics from predictor
                    if hasattr(self.qcpp_adapter, 'predictor'):
                        # This is a simplified fallback - actual implementation depends on adapter
                        return {
                            'qcp_score': 0.0,
                            'field_coherence': 0.0,
                            'stability_prediction': 'unknown',
                            'golden_ratio_score': 0.0,
                        }
                    return None
                
                # Call QCPP adapter
                qcpp_result = self.qcpp_adapter.analyze_conformation(conformation_dict)
                
                # Extract metrics from result
                # The exact structure depends on QCPPIntegrationAdapter implementation
                # Assume it returns a dict-like object with relevant metrics
                metrics = {
                    'qcp_score': float(getattr(qcpp_result, 'qcp_score', 0.0)),
                    'field_coherence': float(getattr(qcpp_result, 'field_coherence', 0.0)),
                    'stability_prediction': str(getattr(qcpp_result, 'stability_prediction', 'unknown')),
                    'golden_ratio_score': float(getattr(qcpp_result, 'golden_ratio_score', 0.0)),
                }
                
                # Include THz spectrum if available and relevant
                if pattern.pattern_type == PatternType.THZ:
                    if hasattr(qcpp_result, 'thz_spectrum'):
                        metrics['thz_spectrum'] = qcpp_result.thz_spectrum
                
                return metrics
            
            except AttributeError as e:
                # QCPP adapter doesn't have expected methods
                print(f"Warning: QCPP adapter missing expected methods: {e}")
                return None
            
            except Exception as e:
                # QCPP analysis failed
                print(f"Warning: QCPP analysis failed: {e}")
                return None
        
        except Exception as e:
            # Outer exception handler for unexpected errors
            print(f"Warning: QCPP relay failed unexpectedly: {e}")
            return None
    
    def broadcast_to_agents(self, pattern: PatternDetection, qcpp_metrics: Optional[Dict] = None) -> bool:
        """
        Broadcast pattern information to exploration agents via shared memory.
        
        This method shares detected patterns with all exploration agents through
        the shared memory pool, enabling collective learning and convergence.
        
        The broadcast process:
        1. Check throttle to prevent overflow
        2. Prioritize high-significance patterns (bypass throttle)
        3. Create broadcast entry with pattern metadata
        4. Include QCPP validation metrics if available
        5. Store via shared_memory.broadcast_pattern()
        
        Args:
            pattern: Pattern to broadcast
            qcpp_metrics: Optional QCPP validation metrics from relay_to_qcpp()
        
        Returns:
            True if broadcast succeeded, False if throttled or failed
        
        Example:
            >>> pattern = PatternDetection(...)
            >>> qcpp_metrics = mediator.relay_to_qcpp(pattern, conformation)
            >>> success = mediator.broadcast_to_agents(pattern, qcpp_metrics)
            >>> if success:
            ...     print("Pattern broadcast to all agents")
        """
        try:
            # Determine pattern priority
            priority = self.broadcast_throttler.prioritize_message(pattern)
            
            # Check throttle
            if not self.broadcast_throttler.can_broadcast(priority):
                # Throttled - too many broadcasts recently
                return False
            
            # Create pattern broadcast entry using pattern.to_dict()
            broadcast_entry = pattern.to_dict()
            
            # Add QCPP metrics if available
            if qcpp_metrics:
                broadcast_entry['qcpp_metrics'] = qcpp_metrics
            
            # Store in shared memory pool via proper interface
            self.shared_memory.broadcast_pattern(broadcast_entry)
            
            # Record broadcast
            self.broadcast_throttler.record_broadcast()
            self.detection_statistics['broadcasts'] += 1
            
            return True
        
        except Exception as e:
            print(f"Warning: Pattern broadcast failed: {e}")
            return False
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Return pattern detection statistics.
        
        Returns:
            Dictionary with detection counts, cache stats, and performance metrics
        """
        return {
            **self.detection_statistics,
            'cache_hit_rate': self._get_cache_hit_rate(),
            'cache_size': len(self.pattern_cache),
            'reference_conformations': len(self.reference_conformations),
            'current_iteration': self.current_iteration,
        }
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _detect_thz_resonance(self, conformation: Conformation) -> Optional[PatternDetection]:
        """
        Detect THz resonance patterns using QCPP analysis and clustering.
        
        This method:
        1. Checks cache for existing THz signature
        2. Calculates THz spectrum via QCPP if not cached
        3. Collects recent THz signatures for clustering
        4. Identifies resonant clusters
        5. Returns PatternDetection if significant resonance found
        
        Args:
            conformation: Protein conformation to analyze
        
        Returns:
            PatternDetection with THzResonanceData if resonance found, None otherwise
        
        Raises:
            None - gracefully handles all errors
        """
        if not self.config.enable_thz_detection:
            return None
        
        try:
            # Generate conformation hash
            conf_hash = self._generate_conformation_hash(conformation)
            
            # Check THz signature cache
            if conf_hash in self.thz_signature_cache:
                thz_signature = self.thz_signature_cache[conf_hash]
            else:
                # Calculate THz spectrum via QCPP
                try:
                    # Use QCPP predictor to get THz spectrum
                    # Note: This assumes the predictor has a predict_thz_spectrum method
                    thz_spectrum_df = self.qcpp_adapter.predictor.predict_thz_spectrum()
                    
                    # Extract frequencies and intensities as numpy arrays
                    frequencies = thz_spectrum_df['frequency'].values
                    intensities = thz_spectrum_df['intensity'].values
                    
                    thz_signature = {
                        'frequencies': frequencies,
                        'intensities': intensities,
                        'hash': conf_hash,
                        'timestamp': time.time(),
                    }
                    
                    # Store in cache
                    self.thz_signature_cache[conf_hash] = thz_signature
                    
                    # Maintain cache size limit
                    if len(self.thz_signature_cache) > self.config.cache_size:
                        self.thz_signature_cache.popitem(last=False)
                
                except Exception as e:
                    # Log warning and skip THz detection for this conformation
                    print(f"Warning: THz spectrum calculation failed: {e}")
                    return None
            
            # Collect recent THz signatures for clustering (window size: 50)
            recent_signatures = list(self.thz_signature_cache.values())[-50:]
            
            if len(recent_signatures) < 2:
                # Not enough data for clustering
                return None
            
            # Perform clustering using spectral correlation
            patterns = self._cluster_thz_signatures(recent_signatures, conf_hash)
            
            return patterns
        
        except Exception as e:
            print(f"Warning: THz resonance detection failed: {e}")
            return None
    
    def _cluster_thz_signatures(
        self, 
        signatures: List[Dict],
        current_hash: str
    ) -> Optional[PatternDetection]:
        """
        Cluster THz signatures using spectral correlation and DBSCAN.
        
        Args:
            signatures: List of THz signature dictionaries
            current_hash: Hash of current conformation
        
        Returns:
            PatternDetection if current conformation belongs to significant cluster, None otherwise
        """
        if len(signatures) < 2:
            return None
        
        # Build pairwise correlation matrix
        n = len(signatures)
        correlation_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = self._calculate_spectral_correlation(
                    signatures[i]['frequencies'],
                    signatures[i]['intensities'],
                    signatures[j]['frequencies'],
                    signatures[j]['intensities']
                )
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # Convert correlation to distance (1 - correlation)
        distance_matrix = 1.0 - correlation_matrix
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=1.0 - self.config.thz_similarity_threshold,  # eps=0.3 when threshold=0.7
            min_samples=2,
            metric='precomputed'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Find cluster of current conformation
        current_idx = -1
        for i, sig in enumerate(signatures):
            if sig['hash'] == current_hash:
                current_idx = i
                break
        
        if current_idx == -1 or labels[current_idx] == -1:
            # Current conformation not in a cluster (noise point)
            return None
        
        current_cluster = labels[current_idx]
        cluster_size = np.sum(labels == current_cluster)
        
        # Calculate significance
        significance_score = cluster_size / len(signatures)
        
        if significance_score < 0.1:
            # Cluster too small to be significant
            return None
        
        # Determine significance level
        if significance_score >= 0.3:
            significance = PatternSignificance.HIGH
        elif significance_score >= 0.15:
            significance = PatternSignificance.MEDIUM
        else:
            significance = PatternSignificance.LOW
        
        # Calculate cluster statistics
        cluster_signatures = [sig for i, sig in enumerate(signatures) if labels[i] == current_cluster]
        
        # Find dominant frequency (average peak frequency across cluster)
        dominant_frequencies = []
        for sig in cluster_signatures:
            peak_idx = np.argmax(sig['intensities'])
            dominant_frequencies.append(sig['frequencies'][peak_idx])
        dominant_frequency = float(np.mean(dominant_frequencies))
        
        # Calculate spectral entropy (average across cluster)
        entropies = []
        for sig in cluster_signatures:
            # Normalize intensities to probability distribution
            probs = sig['intensities'] / np.sum(sig['intensities'])
            # Calculate Shannon entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        spectral_entropy = float(np.mean(entropies))
        
        # Calculate average similarity within cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == current_cluster]
        similarity_scores = []
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx_i = cluster_indices[i]
                idx_j = cluster_indices[j]
                similarity_scores.append(correlation_matrix[idx_i, idx_j])
        avg_similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0
        
        # Create THzResonanceData
        thz_data = THzResonanceData(
            cluster_id=int(current_cluster),
            cluster_size=int(cluster_size),
            similarity_score=avg_similarity,
            dominant_frequency=dominant_frequency,
            spectral_entropy=spectral_entropy
        )
        
        # Create PatternDetection
        pattern = PatternDetection(
            pattern_type=PatternType.THZ,
            significance=significance,
            timestamp=time.time(),
            iteration=self.current_iteration,
            conformation_hash=current_hash,
            thz_data=thz_data
        )
        
        return pattern
    
    def _calculate_spectral_correlation(
        self,
        freq1: np.ndarray,
        int1: np.ndarray,
        freq2: np.ndarray,
        int2: np.ndarray
    ) -> float:
        """
        Calculate spectral correlation between two THz spectra.
        
        Interpolates both spectra onto a common frequency grid and computes
        Pearson correlation coefficient.
        
        Args:
            freq1: Frequencies of first spectrum (THz)
            int1: Intensities of first spectrum
            freq2: Frequencies of second spectrum (THz)
            int2: Intensities of second spectrum
        
        Returns:
            Correlation coefficient in range [0.0, 1.0]
        """
        try:
            # Create common frequency grid (0-10 THz with 1000 points)
            common_freq = np.linspace(0, 10, 1000)
            
            # Interpolate both spectra onto common grid
            int1_interp = np.interp(common_freq, freq1, int1)
            int2_interp = np.interp(common_freq, freq2, int2)
            
            # Normalize to unit vectors
            norm1 = np.linalg.norm(int1_interp)
            norm2 = np.linalg.norm(int2_interp)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                # One or both spectra are essentially zero
                return 0.0
            
            int1_norm = int1_interp / norm1
            int2_norm = int2_interp / norm2
            
            # Calculate Pearson correlation
            corr_matrix = np.corrcoef(int1_norm, int2_norm)
            corr = corr_matrix[0, 1]
            
            # Clamp to [0, 1] range (negative correlations treated as 0)
            return float(max(0.0, min(1.0, corr)))
        
        except Exception as e:
            print(f"Warning: Spectral correlation calculation failed: {e}")
            return 0.0

    def _detect_folding_dynamics(self, conformation: "Conformation") -> Optional[PatternDetection]:
        """
        Detect folding dynamics (secondary structure) from phi/psi angles.

        Assumptions:
        - Helix criteria: phi in [-70, -50], psi in [-50, -30]
        - Sheet criteria: phi in [-150, -90], psi in [120, 170]
        - Turn criteria (approximation): phi in [-90, -30], psi in [0, 90]
        - Residue indices in regions are 1-based and inclusive (start, end)

        Returns:
            PatternDetection with FoldingDynamicsData if significant, otherwise None
        """
        if not self.config.enable_folding_detection:
            return None

        try:
            phi = getattr(conformation, 'phi_angles', None)
            psi = getattr(conformation, 'psi_angles', None)

            if not phi or not psi or len(phi) != len(psi):
                return None

            n = len(phi)
            labels: List[str] = []  # 'H' helix, 'E' sheet, 'T' turn, 'C' coil

            for i in range(n):
                p = phi[i]
                q = psi[i]

                # If angles are missing or NaN, treat as coil
                if p is None or q is None:
                    labels.append('C')
                    continue

                # Helix
                if -70.0 <= p <= -50.0 and -50.0 <= q <= -30.0:
                    labels.append('H')
                    continue

                # Sheet (beta) - approximate
                if -150.0 <= p <= -90.0 and 120.0 <= q <= 170.0:
                    labels.append('E')
                    continue

                # Turn - approximate window
                if -90.0 <= p <= -30.0 and 0.0 <= q <= 90.0:
                    labels.append('T')
                    continue

                # Otherwise coil
                labels.append('C')

            # Counts and percentages
            helix_count = sum(1 for l in labels if l == 'H')
            sheet_count = sum(1 for l in labels if l == 'E')
            turn_count = sum(1 for l in labels if l == 'T')
            coil_count = sum(1 for l in labels if l == 'C')

            helix_pct = (helix_count / n) * 100.0
            sheet_pct = (sheet_count / n) * 100.0
            turn_pct = (turn_count / n) * 100.0
            coil_pct = (coil_count / n) * 100.0

            # Identify continuous regions
            helix_regions = self._find_continuous_regions(labels, 'H', self.config.secondary_structure_min_length)
            sheet_regions = self._find_continuous_regions(labels, 'E', 3)
            turn_regions = self._find_continuous_regions(labels, 'T', 3)

            # Compute significance: high if helix>30% or sheet>20%
            if helix_pct > 30.0 or sheet_pct > 20.0:
                significance = PatternSignificance.HIGH
            elif helix_pct > 15.0 or sheet_pct > 10.0:
                significance = PatternSignificance.MEDIUM
            else:
                significance = PatternSignificance.LOW

            # Create FoldingDynamicsData
            folding_data = FoldingDynamicsData(
                helix_percentage=round(helix_pct, 2),
                sheet_percentage=round(sheet_pct, 2),
                turn_percentage=round(turn_pct, 2),
                coil_percentage=round(coil_pct, 2),
                helix_regions=helix_regions,
                sheet_regions=sheet_regions,
                turn_regions=turn_regions,
            )

            conf_hash = self._generate_conformation_hash(conformation)

            pattern = PatternDetection(
                pattern_type=PatternType.FOLDING,
                significance=significance,
                timestamp=time.time(),
                iteration=self.current_iteration,
                conformation_hash=conf_hash,
                thz_data=None,
                folding_data=folding_data,
                geometric_data=None,
            )

            # Only broadcast/return if at least MEDIUM significance (configurable policy)
            # For now, return any detected pattern (policy can filter later)
            return pattern if pattern else None

        except Exception as e:
            print(f"Warning: Folding dynamics detection failed: {e}")
            return None

    def _find_continuous_regions(self, labels: List[str], target: str, min_length: int) -> List[Tuple[int, int]]:
        """
        Find continuous regions of `target` label in labels list.

        Returns list of (start, end) tuples using 1-based inclusive indices.
        Only regions with length >= min_length are returned.
        """
        regions: List[Tuple[int, int]] = []
        n = len(labels)
        i = 0
        while i < n:
            if labels[i] == target:
                start = i
                j = i + 1
                while j < n and labels[j] == target:
                    j += 1
                length = j - start
                if length >= min_length:
                    # convert to 1-based inclusive indices
                    regions.append((start + 1, j))
                i = j
            else:
                i += 1

        return regions
    
    def _detect_geometric_similarity(self, conformation: Conformation) -> Optional[PatternDetection]:
        """
        Detect geometric similarity to reference conformations using RMSD.
        
        This method:
        1. Extracts CA coordinates from current conformation
        2. Calculates RMSD to each reference conformation
        3. Identifies similar conformations (RMSD < threshold)
        4. Performs detailed geometric analysis on most similar reference
        5. Calculates structural overlap percentage
        6. Returns PatternDetection if significant similarity found
        
        Args:
            conformation: Protein conformation to analyze
        
        Returns:
            PatternDetection with GeometricSimilarityData if similarity found, None otherwise
        
        Raises:
            None - gracefully handles all errors
        """
        if not self.config.enable_geometric_detection:
            return None
        
        # Need at least one reference for comparison
        if len(self.reference_conformations) == 0:
            return None
        
        try:
            # Extract CA coordinates from current conformation
            current_coords = conformation.atom_coordinates
            conf_hash = self._generate_conformation_hash(conformation)
            
            if len(current_coords) == 0:
                return None
            
            # Find most similar reference conformation
            best_rmsd = float('inf')
            best_reference = None
            best_rmsd_result = None
            
            for ref in self.reference_conformations:
                ref_coords = ref['coordinates']
                
                # Skip if coordinate counts don't match
                if len(ref_coords) != len(current_coords):
                    continue
                
                try:
                    # Calculate RMSD with alignment
                    rmsd_result = self.rmsd_calculator.calculate_rmsd(
                        predicted_coords=current_coords,
                        native_coords=ref_coords,
                        calculate_metrics=False  # Don't need GDT-TS/TM-score for similarity
                    )
                    
                    if rmsd_result.rmsd < best_rmsd:
                        best_rmsd = rmsd_result.rmsd
                        best_reference = ref
                        best_rmsd_result = rmsd_result
                
                except Exception as e:
                    # Skip this reference if RMSD calculation fails
                    continue
            
            # Check if similarity is significant
            if best_reference is None or best_rmsd > self.config.geometric_similarity_threshold:
                # No similar reference found
                return None
            
            # Calculate structural overlap (residues within 2.0 Å)
            overlap_pct = self._calculate_structural_overlap(
                current_coords,
                best_reference['coordinates']
            )
            
            # Perform detailed geometric analysis
            # Convert conformation to dict format expected by analyzer
            conformation_dict = {
                'coordinates': current_coords,
                'sequence': conformation.sequence if hasattr(conformation, 'sequence') else None
            }
            geometric_analysis = self.geometric_analyzer.analyze_conformation(
                conformation_dict,
                sequence=conformation.sequence if hasattr(conformation, 'sequence') else None
            )
            
            # Calculate significance based on RMSD and overlap
            # High: RMSD < 1.0 Å AND overlap > 80%
            # Medium: RMSD < 1.5 Å AND overlap > 60%
            # Low: Otherwise (but still similar)
            if best_rmsd < 1.0 and overlap_pct > 80.0:
                significance = PatternSignificance.HIGH
            elif best_rmsd < 1.5 and overlap_pct > 60.0:
                significance = PatternSignificance.MEDIUM
            else:
                significance = PatternSignificance.LOW
            
            # Find dominant Platonic solid from geometric analysis
            # Build dict of similarities
            platonic_similarities = {
                'tetrahedron': geometric_analysis.tetrahedron_similarity,
                'cube': geometric_analysis.cube_similarity,
                'octahedron': geometric_analysis.octahedron_similarity,
                'dodecahedron': geometric_analysis.dodecahedron_similarity,
                'icosahedron': geometric_analysis.icosahedron_similarity,
            }
            dominant_solid = max(platonic_similarities.items(), key=lambda x: x[1])
            dominant_solid_name = dominant_solid[0]
            dominant_solid_score = dominant_solid[1]
            
            # Create GeometricSimilarityData
            geometric_data = GeometricSimilarityData(
                rmsd_to_reference=round(best_rmsd, 3),
                overlap_percentage=round(overlap_pct, 2),
                reference_conformation_hash=best_reference['hash'],
                golden_ratio_percentage=round(geometric_analysis.golden_ratio_percentage, 2),
                dominant_platonic_solid=dominant_solid_name,
                platonic_similarity_score=round(dominant_solid_score, 3)
            )
            
            # Create PatternDetection
            pattern = PatternDetection(
                pattern_type=PatternType.GEOMETRIC,
                significance=significance,
                timestamp=time.time(),
                iteration=self.current_iteration,
                conformation_hash=conf_hash,
                thz_data=None,
                folding_data=None,
                geometric_data=geometric_data
            )
            
            return pattern
        
        except Exception as e:
            print(f"Warning: Geometric similarity detection failed: {e}")
            return None
    
    def _calculate_structural_overlap(
        self,
        coords1: List[Tuple[float, float, float]],
        coords2: List[Tuple[float, float, float]],
        distance_threshold: float = 2.0
    ) -> float:
        """
        Calculate percentage of residues within distance threshold.
        
        Args:
            coords1: First set of coordinates
            coords2: Second set of coordinates
            distance_threshold: Distance threshold in Ångströms (default 2.0)
        
        Returns:
            Overlap percentage (0.0-100.0)
        """
        if len(coords1) != len(coords2):
            return 0.0
        
        if len(coords1) == 0:
            return 0.0
        
        matching_count = 0
        threshold_squared = distance_threshold ** 2
        
        for (x1, y1, z1), (x2, y2, z2) in zip(coords1, coords2):
            # Calculate squared distance (avoid sqrt for performance)
            dist_squared = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
            
            if dist_squared <= threshold_squared:
                matching_count += 1
        
        overlap_pct = (matching_count / len(coords1)) * 100.0
        return overlap_pct
    
    def add_reference_conformation(
        self,
        conformation: Conformation,
        agent_id: Optional[str] = None,
        geometric_score: Optional[float] = None
    ) -> None:
        """
        Add a conformation to the reference set for geometric similarity detection.
        
        Reference conformations are used to identify convergent exploration pathways
        where multiple agents find similar structures. References are stored with:
        - Coordinates for RMSD calculation
        - Energy for quality assessment
        - Geometric score for prioritization
        - Agent ID to track diversity
        - Timestamp for age-based eviction
        
        Args:
            conformation: Conformation to add as reference
            agent_id: ID of agent that found this conformation (optional)
            geometric_score: Geometric analysis score (golden ratio %, optional)
        
        Note:
            Implements LRU eviction when max_references (100) is reached.
            Conformations with higher geometric scores are prioritized.
        """
        try:
            # Extract data from conformation
            conf_hash = self._generate_conformation_hash(conformation)
            
            # Check if this conformation already exists in references
            for ref in self.reference_conformations:
                if ref['hash'] == conf_hash:
                    # Already have this reference, update timestamp
                    ref['timestamp'] = time.time()
                    return
            
            # Create reference entry
            reference = {
                'hash': conf_hash,
                'coordinates': conformation.atom_coordinates,
                'energy': conformation.energy,
                'geometric_score': geometric_score or 0.0,
                'timestamp': time.time(),
                'agent_id': agent_id,
            }
            
            # Add to reference list
            self.reference_conformations.append(reference)
            
            # Check if eviction needed
            if len(self.reference_conformations) > self.max_references:
                # Evict lowest priority reference
                # Priority: geometric_score (higher is better), then oldest
                self.reference_conformations.sort(
                    key=lambda x: (x['geometric_score'], x['timestamp']),
                    reverse=False  # Ascending (lowest score/oldest first)
                )
                # Remove first (lowest priority)
                self.reference_conformations.pop(0)
        
        except Exception as e:
            print(f"Warning: Failed to add reference conformation: {e}")
    
    def clear_reference_conformations(self) -> None:
        """
        Clear all reference conformations.
        
        Useful for resetting between different protein targets or
        when starting a new exploration phase.
        """
        self.reference_conformations.clear()
    
    # ========================================================================
    # Private Helper Methods (Original)
    # ========================================================================
    
    def _generate_conformation_hash(self, conformation: Conformation) -> str:
        """
        Generate SHA256 hash of conformation for caching.
        
        Args:
            conformation: Protein conformation
        
        Returns:
            First 16 characters of SHA256 hash
        """
        # Use atom coordinates for hashing
        coords = conformation.atom_coordinates
        
        # Round to 2 decimal places for stability
        rounded_coords = []
        for coord in coords:
            if isinstance(coord, (list, tuple)) and len(coord) == 3:
                rounded_coords.extend([
                    round(coord[0], 2),
                    round(coord[1], 2),
                    round(coord[2], 2),
                ])
        
        # Convert to bytes and hash
        coord_bytes = str(rounded_coords).encode('utf-8')
        hash_obj = hashlib.sha256(coord_bytes)
        
        return hash_obj.hexdigest()[:16]
    
    def _get_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            Hit rate as fraction (0.0-1.0)
        """
        total = (self.detection_statistics['cache_hits'] + 
                 self.detection_statistics['cache_misses'])
        
        if total == 0:
            return 0.0
        
        return self.detection_statistics['cache_hits'] / total
