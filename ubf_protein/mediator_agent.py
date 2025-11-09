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
        self.reference_conformations: List[Dict] = []
        
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
        self.broadcast_times: List[float] = []
    
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
        
        # TODO (Task 7): Folding dynamics detection
        # if self.config.enable_folding_detection:
        #     folding_pattern = self._detect_folding_dynamics(conformation)
        #     if folding_pattern:
        #         patterns.append(folding_pattern)
        
        # TODO (Task 8): Geometric similarity detection
        # if self.config.enable_geometric_detection:
        #     geometric_pattern = self._detect_geometric_similarity(conformation)
        #     if geometric_pattern:
        #         patterns.append(geometric_pattern)
        
        # Cache results
        self.pattern_cache[conf_hash] = patterns
        
        # Maintain cache size limit (LRU eviction)
        if len(self.pattern_cache) > self.config.cache_size:
            self.pattern_cache.popitem(last=False)  # Remove oldest
        
        return patterns
    
    def relay_to_qcpp(self, pattern: PatternDetection) -> Optional[Any]:
        """
        Request QCPP analysis for detected pattern.
        
        Args:
            pattern: Detected pattern to validate
        
        Returns:
            QCPP metrics if successful, None if failed
        
        Note:
            Implementation will be added in Task 9 (information relay).
            This is a skeleton method for Task 5.2.
        """
        # TODO (Task 9): Implement QCPP relay
        self.detection_statistics['qcpp_validations'] += 1
        return None
    
    def broadcast_to_agents(self, pattern: PatternDetection) -> None:
        """
        Broadcast pattern information to exploration agents.
        
        Args:
            pattern: Pattern to broadcast via shared memory
        
        Note:
            Implementation will be added in Task 9 (information relay).
            This is a skeleton method for Task 5.2.
        """
        # TODO (Task 9): Implement broadcast with throttling
        self.detection_statistics['broadcasts'] += 1
    
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
