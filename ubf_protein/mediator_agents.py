"""
Mediator Agents Module - Pattern Relaying and System Coordination

This module implements Mediator Agents that act as intermediaries between the
QCPP system and protein exploration agents, facilitating information flow and
coordination.

Key Responsibilities:
1. THz Resonance Pattern Detection - Monitor and cluster THz signatures
2. Folding Dynamics Detection - Track secondary structure formation
3. Geometric Similarity Detection - Identify structural convergence patterns
4. Information Relaying - Broadcast significant patterns to all agents
5. Memory Flow Coordination - Manage shared memory pool efficiently
6. Data Caching - Cache QCPP results and patterns for performance

Design Principles:
- Non-blocking beacon model (agents pull data when needed)
- Significance-based filtering (only relay important patterns)
- Distributed caching (reduce redundant QCPP calculations)
- Pure Python (PyPy-optimized)

Author: UBF Protein System
Date: November 9, 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, TYPE_CHECKING
from collections import defaultdict
from enum import Enum
import time
import math

if TYPE_CHECKING:
    from .models import Conformation, ConformationalMemory
    from .qcpp_integration import QCPPMetrics
    from .geometric_attractor_v2 import GeometricRelationshipScores


# ============================================================================
# Pattern Detection Data Models
# ============================================================================

class PatternSignificance(Enum):
    """Significance levels for detected patterns."""
    LOW = "low"        # 0.0-0.4: Minor patterns, cache only
    MEDIUM = "medium"  # 0.4-0.7: Moderate patterns, selective relay
    HIGH = "high"      # 0.7-1.0: Major patterns, broadcast to all agents


@dataclass(frozen=True)
class THzResonancePattern:
    """
    THz resonance pattern detected by mediator.
    
    Attributes:
        cluster_id: Unique identifier for this THz signature cluster
        cluster_size: Number of conformations in this cluster
        dominant_frequency_thz: Primary THz frequency (THz)
        similarity_score: Average spectral correlation (0.0-1.0)
        representative_conformation_hash: Hash of exemplar conformation
        significance: Pattern significance level
        timestamp: Detection timestamp
    """
    cluster_id: int
    cluster_size: int
    dominant_frequency_thz: float
    similarity_score: float
    representative_conformation_hash: str
    significance: PatternSignificance
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'type': 'thz_resonance',
            'cluster_id': self.cluster_id,
            'cluster_size': self.cluster_size,
            'dominant_frequency_thz': self.dominant_frequency_thz,
            'similarity_score': self.similarity_score,
            'representative': self.representative_conformation_hash,
            'significance': self.significance.value,
            'timestamp': self.timestamp,
        }


@dataclass(frozen=True)
class FoldingDynamicsPattern:
    """
    Secondary structure folding pattern detected by mediator.
    
    Attributes:
        pattern_type: Type of folding pattern ('helix', 'sheet', 'turn')
        start_residue: Starting residue index
        end_residue: Ending residue index
        length: Number of residues in pattern
        stability_score: Stability assessment (0.0-1.0)
        occurrence_count: How many times this pattern was observed
        significance: Pattern significance level
        timestamp: Detection timestamp
    """
    pattern_type: str  # 'helix', 'sheet', 'turn'
    start_residue: int
    end_residue: int
    length: int
    stability_score: float
    occurrence_count: int
    significance: PatternSignificance
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'type': 'folding_dynamics',
            'pattern_type': self.pattern_type,
            'region': (self.start_residue, self.end_residue),
            'length': self.length,
            'stability': self.stability_score,
            'occurrences': self.occurrence_count,
            'significance': self.significance.value,
            'timestamp': self.timestamp,
        }


@dataclass(frozen=True)
class GeometricSimilarityPattern:
    """
    Geometric convergence pattern detected by mediator.
    
    Attributes:
        cluster_id: Unique identifier for geometric cluster
        cluster_size: Number of conformations in cluster
        representative_hash: Hash of cluster centroid
        average_rmsd: Average RMSD within cluster (Angstroms)
        geometric_score: Overall geometric organization percentage
        phi_pattern_strength: Golden ratio pattern strength percentage
        platonic_similarity: Best Platonic solid match percentage
        significance: Pattern significance level
        timestamp: Detection timestamp
    """
    cluster_id: int
    cluster_size: int
    representative_hash: str
    average_rmsd: float
    geometric_score: float
    phi_pattern_strength: float
    platonic_similarity: float
    significance: PatternSignificance
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'type': 'geometric_similarity',
            'cluster_id': self.cluster_id,
            'cluster_size': self.cluster_size,
            'representative': self.representative_hash,
            'avg_rmsd': self.average_rmsd,
            'geometric_score': self.geometric_score,
            'phi_strength': self.phi_pattern_strength,
            'platonic_sim': self.platonic_similarity,
            'significance': self.significance.value,
            'timestamp': self.timestamp,
        }


# ============================================================================
# Mediator Agent Implementation
# ============================================================================

@dataclass
class MediatorAgentConfig:
    """Configuration for mediator agent behavior."""
    
    # Detection thresholds
    thz_similarity_threshold: float = 0.7  # Min similarity for THz cluster
    geometric_rmsd_threshold: float = 3.0  # Max RMSD for geometric cluster (Angstroms)
    folding_min_length: int = 4  # Minimum length for folding pattern
    
    # Significance thresholds
    high_significance_threshold: float = 0.7
    medium_significance_threshold: float = 0.4
    
    # Cache settings
    qcpp_cache_size: int = 5000
    pattern_cache_size: int = 1000
    memory_cache_size: int = 2000
    
    # Relay settings
    broadcast_interval_ms: float = 100.0  # Minimum time between broadcasts
    max_patterns_per_broadcast: int = 10  # Limit patterns per broadcast


class MediatorAgent:
    """
    Mediator Agent - Intelligent pattern detection and information relay.
    
    Acts as a beacon between QCPP system and exploration agents:
    - Detects THz resonance patterns through signature clustering
    - Tracks folding dynamics (helix, sheet, turn formation)
    - Identifies geometric convergence via similarity analysis
    - Relays significant patterns to exploration agents
    - Manages shared memory pool and caching
    
    Architecture:
    - Non-blocking: Agents pull data when needed
    - Significance-filtered: Only important patterns are broadcast
    - Performance-optimized: Extensive caching reduces redundant work
    
    Usage:
        >>> mediator = MediatorAgent()
        >>> mediator.observe_conformation(conformation, qcpp_metrics, geo_scores)
        >>> patterns = mediator.get_significant_patterns()
        >>> for pattern in patterns:
        ...     # Relay pattern to agents
        ...     agent.receive_pattern(pattern)
    """
    
    def __init__(self, config: Optional[MediatorAgentConfig] = None):
        """
        Initialize mediator agent.
        
        Args:
            config: Configuration settings (uses defaults if None)
        """
        self.config = config or MediatorAgentConfig()
        
        # Pattern storage
        self.thz_patterns: List[THzResonancePattern] = []
        self.folding_patterns: List[FoldingDynamicsPattern] = []
        self.geometric_patterns: List[GeometricSimilarityPattern] = []
        
        # Detection state
        self.thz_clusters: Dict[int, List[str]] = defaultdict(list)  # cluster_id -> conf_hashes
        self.geometric_clusters: Dict[int, List[str]] = defaultdict(list)
        self.folding_observations: Dict[str, int] = defaultdict(int)  # pattern_key -> count
        
        # Caches
        self.qcpp_cache: Dict[str, 'QCPPMetrics'] = {}
        self.geometric_cache: Dict[str, 'GeometricRelationshipScores'] = {}
        self.memory_cache: Dict[str, 'ConformationalMemory'] = {}
        
        # Statistics
        self.total_observations = 0
        self.patterns_detected = 0
        self.patterns_relayed = 0
        self.cache_hits = 0
        self.last_broadcast_time = 0.0
        
        # Conformation tracking
        self.observed_hashes: Set[str] = set()
        self.next_cluster_id = 0
    
    def observe_conformation(
        self,
        conformation: 'Conformation',
        qcpp_metrics: Optional['QCPPMetrics'] = None,
        geometric_scores: Optional['GeometricRelationshipScores'] = None
    ) -> None:
        """
        Observe a conformation and detect patterns.
        
        This is the main entry point for mediator analysis. Call this each time
        a conformation is evaluated to allow pattern detection.
        
        Args:
            conformation: Protein conformation to observe
            qcpp_metrics: QCPP analysis results (optional, will use cache if available)
            geometric_scores: Geometric analysis results (optional)
        """
        self.total_observations += 1
        
        # Generate conformation hash
        conf_hash = self._hash_conformation(conformation)
        
        # Skip if already observed (avoid duplicates)
        if conf_hash in self.observed_hashes:
            return
        
        self.observed_hashes.add(conf_hash)
        
        # Cache QCPP metrics if provided
        if qcpp_metrics is not None:
            self._cache_qcpp_metrics(conf_hash, qcpp_metrics)
        
        # Cache geometric scores if provided
        if geometric_scores is not None:
            self._cache_geometric_scores(conf_hash, geometric_scores)
        
        # Detect patterns (only if we have data)
        if qcpp_metrics is not None:
            self._detect_thz_patterns(conf_hash, qcpp_metrics)
        
        if geometric_scores is not None:
            self._detect_geometric_patterns(conf_hash, geometric_scores)
            self._detect_folding_patterns(conf_hash, conformation, geometric_scores)
    
    def _hash_conformation(self, conformation: 'Conformation') -> str:
        """Generate hash from conformation for caching."""
        import hashlib
        
        # Use atom coordinates (first 16 chars of SHA256)
        coords_str = str(conformation.atom_coordinates)
        return hashlib.sha256(coords_str.encode()).hexdigest()[:16]
    
    def _cache_qcpp_metrics(self, conf_hash: str, metrics: 'QCPPMetrics') -> None:
        """Cache QCPP metrics for this conformation."""
        if len(self.qcpp_cache) >= self.config.qcpp_cache_size:
            # Remove oldest entry
            oldest = next(iter(self.qcpp_cache))
            del self.qcpp_cache[oldest]
        
        self.qcpp_cache[conf_hash] = metrics
    
    def _cache_geometric_scores(self, conf_hash: str, scores: 'GeometricRelationshipScores') -> None:
        """Cache geometric scores for this conformation."""
        if len(self.geometric_cache) >= self.config.pattern_cache_size:
            # Remove oldest entry
            oldest = next(iter(self.geometric_cache))
            del self.geometric_cache[oldest]
        
        self.geometric_cache[conf_hash] = scores
    
    def _detect_thz_patterns(self, conf_hash: str, metrics: 'QCPPMetrics') -> None:
        """
        Detect THz resonance patterns through clustering.
        
        Clusters conformations by their QCPP metrics (QCP score, field coherence).
        Forms clusters when multiple conformations have similar quantum properties.
        """
        # For now, use simple QCP-based clustering
        # In production, would use full THz spectral clustering
        
        qcp_cluster = int(metrics.qcp_score)  # Simple binning by integer QCP
        
        # Add to cluster
        self.thz_clusters[qcp_cluster].append(conf_hash)
        cluster_size = len(self.thz_clusters[qcp_cluster])
        
        # Detect pattern if cluster reaches threshold
        if cluster_size >= 5:  # Minimum cluster size
            # Calculate significance
            avg_coherence = metrics.field_coherence
            similarity = abs(avg_coherence)  # Use coherence as proxy for similarity
            
            significance = self._determine_significance(similarity)
            
            # Create pattern
            pattern = THzResonancePattern(
                cluster_id=qcp_cluster,
                cluster_size=cluster_size,
                dominant_frequency_thz=metrics.qcp_score * 0.1,  # Simplified mapping
                similarity_score=similarity,
                representative_conformation_hash=conf_hash,
                significance=significance,
                timestamp=time.time()
            )
            
            # Store if significant
            if significance != PatternSignificance.LOW:
                self.thz_patterns.append(pattern)
                self.patterns_detected += 1
    
    def _detect_geometric_patterns(self, conf_hash: str, scores: 'GeometricRelationshipScores') -> None:
        """
        Detect geometric convergence patterns.
        
        Identifies when multiple conformations converge to similar geometric
        organization (high phi patterns, Platonic solid similarity).
        """
        # Cluster by overall geometric organization (binned by 10%)
        geo_cluster = int(scores.overall_geometric_organization / 10.0)
        
        # Add to cluster
        self.geometric_clusters[geo_cluster].append(conf_hash)
        cluster_size = len(self.geometric_clusters[geo_cluster])
        
        # Detect pattern if cluster reaches threshold
        if cluster_size >= 3:  # Minimum cluster size
            # Calculate average metrics for cluster
            avg_geo_score = scores.overall_geometric_organization
            phi_strength = scores.phi_distance_patterns
            
            # Best Platonic similarity
            platonic_best = max(
                scores.icosahedron_similarity,
                scores.dodecahedron_similarity,
                scores.octahedron_similarity,
                scores.tetrahedron_similarity,
                scores.cube_similarity
            )
            
            # Calculate significance
            significance = self._determine_significance(avg_geo_score / 100.0)
            
            # Create pattern
            pattern = GeometricSimilarityPattern(
                cluster_id=geo_cluster,
                cluster_size=cluster_size,
                representative_hash=conf_hash,
                average_rmsd=2.0,  # Placeholder (would calculate from cluster)
                geometric_score=avg_geo_score,
                phi_pattern_strength=phi_strength,
                platonic_similarity=platonic_best,
                significance=significance,
                timestamp=time.time()
            )
            
            # Store if significant
            if significance != PatternSignificance.LOW:
                self.geometric_patterns.append(pattern)
                self.patterns_detected += 1
    
    def _detect_folding_patterns(
        self,
        conf_hash: str,
        conformation: 'Conformation',
        scores: 'GeometricRelationshipScores'
    ) -> None:
        """
        Detect secondary structure folding patterns.
        
        Identifies formation of helices, sheets, and turns based on:
        - Geometric regularity (high local symmetry suggests ordered structure)
        - Platonic similarities (helices ~ cylindrical, sheets ~ planar)
        - Phi patterns (helices often show φ-based geometry)
        """
        # Estimate folding patterns from geometric scores
        # In production, would use actual secondary structure assignment
        
        # Helix detection: high elongation + phi patterns
        if scores.elongation > 60 and scores.phi_angle_patterns > 20:
            pattern_key = f"helix_{conf_hash[:8]}"
            self.folding_observations[pattern_key] += 1
            
            if self.folding_observations[pattern_key] >= 3:  # Observed multiple times
                pattern = FoldingDynamicsPattern(
                    pattern_type='helix',
                    start_residue=0,  # Placeholder
                    end_residue=scores.num_residues - 1,
                    length=scores.num_residues,
                    stability_score=scores.elongation / 100.0,
                    occurrence_count=self.folding_observations[pattern_key],
                    significance=PatternSignificance.MEDIUM,
                    timestamp=time.time()
                )
                
                self.folding_patterns.append(pattern)
                self.patterns_detected += 1
        
        # Sheet detection: high planarity + local symmetry
        if scores.planarity > 50 and scores.local_symmetry > 60:
            pattern_key = f"sheet_{conf_hash[:8]}"
            self.folding_observations[pattern_key] += 1
            
            if self.folding_observations[pattern_key] >= 3:
                pattern = FoldingDynamicsPattern(
                    pattern_type='sheet',
                    start_residue=0,
                    end_residue=scores.num_residues - 1,
                    length=scores.num_residues,
                    stability_score=scores.planarity / 100.0,
                    occurrence_count=self.folding_observations[pattern_key],
                    significance=PatternSignificance.MEDIUM,
                    timestamp=time.time()
                )
                
                self.folding_patterns.append(pattern)
                self.patterns_detected += 1
    
    def _determine_significance(self, score: float) -> PatternSignificance:
        """Determine pattern significance from score (0.0-1.0)."""
        if score >= self.config.high_significance_threshold:
            return PatternSignificance.HIGH
        elif score >= self.config.medium_significance_threshold:
            return PatternSignificance.MEDIUM
        else:
            return PatternSignificance.LOW
    
    def get_significant_patterns(
        self,
        min_significance: PatternSignificance = PatternSignificance.MEDIUM
    ) -> List[Dict]:
        """
        Get all significant patterns detected since last broadcast.
        
        Args:
            min_significance: Minimum significance level to return
        
        Returns:
            List of pattern dictionaries (ready for relay to agents)
        """
        patterns = []
        
        # Collect THz patterns
        for pattern in self.thz_patterns:
            if self._is_significant_enough(pattern.significance, min_significance):
                patterns.append(pattern.to_dict())
        
        # Collect folding patterns
        for pattern in self.folding_patterns:
            if self._is_significant_enough(pattern.significance, min_significance):
                patterns.append(pattern.to_dict())
        
        # Collect geometric patterns
        for pattern in self.geometric_patterns:
            if self._is_significant_enough(pattern.significance, min_significance):
                patterns.append(pattern.to_dict())
        
        # Limit to max patterns per broadcast
        if len(patterns) > self.config.max_patterns_per_broadcast:
            # Sort by timestamp (most recent first) and take top N
            patterns.sort(key=lambda p: p['timestamp'], reverse=True)
            patterns = patterns[:self.config.max_patterns_per_broadcast]
        
        return patterns
    
    def _is_significant_enough(
        self,
        pattern_sig: PatternSignificance,
        min_sig: PatternSignificance
    ) -> bool:
        """Check if pattern meets minimum significance threshold."""
        sig_order = {
            PatternSignificance.LOW: 0,
            PatternSignificance.MEDIUM: 1,
            PatternSignificance.HIGH: 2
        }
        return sig_order[pattern_sig] >= sig_order[min_sig]
    
    def cache_memory(self, memory: 'ConformationalMemory') -> None:
        """
        Cache a conformational memory for shared access.
        
        Args:
            memory: Memory to cache
        """
        # Generate key from memory
        memory_key = f"{memory.move_type}_{memory.energy_change:.2f}"
        
        if len(self.memory_cache) >= self.config.memory_cache_size:
            # Remove oldest entry
            oldest = next(iter(self.memory_cache))
            del self.memory_cache[oldest]
        
        self.memory_cache[memory_key] = memory
    
    def get_cached_qcpp_metrics(self, conf_hash: str) -> Optional['QCPPMetrics']:
        """
        Retrieve cached QCPP metrics.
        
        Args:
            conf_hash: Conformation hash
        
        Returns:
            Cached QCPPMetrics or None if not in cache
        """
        if conf_hash in self.qcpp_cache:
            self.cache_hits += 1
            return self.qcpp_cache[conf_hash]
        return None
    
    def get_cached_geometric_scores(self, conf_hash: str) -> Optional['GeometricRelationshipScores']:
        """
        Retrieve cached geometric scores.
        
        Args:
            conf_hash: Conformation hash
        
        Returns:
            Cached GeometricRelationshipScores or None if not in cache
        """
        if conf_hash in self.geometric_cache:
            self.cache_hits += 1
            return self.geometric_cache[conf_hash]
        return None
    
    def clear_patterns(self) -> None:
        """Clear all detected patterns (call after successful broadcast)."""
        self.thz_patterns.clear()
        self.folding_patterns.clear()
        self.geometric_patterns.clear()
        self.last_broadcast_time = time.time()
        self.patterns_relayed += self.patterns_detected
    
    def get_statistics(self) -> Dict:
        """
        Get mediator statistics.
        
        Returns:
            Dictionary with performance and detection statistics
        """
        return {
            'total_observations': self.total_observations,
            'unique_conformations': len(self.observed_hashes),
            'patterns_detected': self.patterns_detected,
            'patterns_relayed': self.patterns_relayed,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': (self.cache_hits / max(1, self.total_observations) * 100),
            'qcpp_cache_size': len(self.qcpp_cache),
            'geometric_cache_size': len(self.geometric_cache),
            'memory_cache_size': len(self.memory_cache),
            'thz_clusters': len(self.thz_clusters),
            'geometric_clusters': len(self.geometric_clusters),
            'folding_observations': len(self.folding_observations),
            'pending_thz_patterns': len(self.thz_patterns),
            'pending_folding_patterns': len(self.folding_patterns),
            'pending_geometric_patterns': len(self.geometric_patterns),
        }
    
    def print_summary(self) -> None:
        """Print human-readable summary of mediator activity."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 70)
        print("MEDIATOR AGENT SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Observations:")
        print(f"  Total conformations observed: {stats['total_observations']}")
        print(f"  Unique conformations: {stats['unique_conformations']}")
        
        print(f"\n🔍 Pattern Detection:")
        print(f"  Patterns detected: {stats['patterns_detected']}")
        print(f"  Patterns relayed: {stats['patterns_relayed']}")
        print(f"  Pending THz patterns: {stats['pending_thz_patterns']}")
        print(f"  Pending folding patterns: {stats['pending_folding_patterns']}")
        print(f"  Pending geometric patterns: {stats['pending_geometric_patterns']}")
        
        print(f"\n💾 Caching Performance:")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        print(f"  QCPP cache: {stats['qcpp_cache_size']} entries")
        print(f"  Geometric cache: {stats['geometric_cache_size']} entries")
        print(f"  Memory cache: {stats['memory_cache_size']} entries")
        
        print(f"\n🎯 Clustering:")
        print(f"  THz clusters: {stats['thz_clusters']}")
        print(f"  Geometric clusters: {stats['geometric_clusters']}")
        print(f"  Folding observations: {stats['folding_observations']}")
        
        print("=" * 70 + "\n")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_mediator(
    thz_threshold: float = 0.7,
    geometric_rmsd_threshold: float = 3.0,
    cache_size: int = 5000
) -> MediatorAgent:
    """
    Create a mediator agent with custom configuration.
    
    Args:
        thz_threshold: Minimum similarity for THz clustering
        geometric_rmsd_threshold: Maximum RMSD for geometric clustering
        cache_size: QCPP cache size
    
    Returns:
        Configured MediatorAgent
    
    Example:
        >>> mediator = create_mediator(thz_threshold=0.8, cache_size=10000)
    """
    config = MediatorAgentConfig(
        thz_similarity_threshold=thz_threshold,
        geometric_rmsd_threshold=geometric_rmsd_threshold,
        qcpp_cache_size=cache_size
    )
    
    return MediatorAgent(config)
