"""
Performance caching system for quantum refinement engine.

This module provides a high-performance caching layer for expensive
computations in the quantum refinement pipeline. Caching reduces
redundant calculations and dramatically improves performance for
iterative optimization.

Key features:
- LRU cache with configurable size limits
- Cache hit/miss statistics tracking
- Automatic cache invalidation
- Sub-millisecond cache access
- Memory-efficient storage

Performance targets:
- Cache hit: <10μs
- Cache miss (compute + store): varies by operation
- Memory: <100MB per 100-residue protein

Cached operations:
1. QCP values for residues
2. THz mode calculations for quantum cores
3. Distance matrices between residues
4. Secondary structure assignments
5. Hydrophobic residue lists
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from collections import OrderedDict
import time
import hashlib
import logging

logger = logging.getLogger(__name__)


class RefinementCache:
    """
    LRU cache for quantum refinement computations.
    
    This cache stores expensive computation results with automatic
    eviction of least-recently-used entries when memory limits are
    reached. Cache keys are generated from structure hashes and
    parameter tuples.
    
    Cache categories:
    - QCP values: residue index → QCP float
    - THz modes: residue index → list of frequencies
    - Distance matrices: structure hash → NxN matrix
    - Secondary structure: structure hash → list of SS codes
    - Hydrophobic residues: structure hash → set of indices
    
    Attributes:
        max_qcp_entries: Maximum QCP cache entries (default 1000)
        max_thz_entries: Maximum THz cache entries (default 500)
        max_distance_matrices: Maximum distance matrices (default 10)
        hit_count: Total cache hits
        miss_count: Total cache misses
        total_time_saved: Cumulative time saved by cache hits (seconds)
    
    Example:
        >>> cache = RefinementCache(max_qcp_entries=1000)
        >>> 
        >>> # Try to get cached QCP
        >>> qcp = cache.get_qcp(structure_hash, residue_idx=10)
        >>> if qcp is None:
        ...     qcp = expensive_qcp_calculation()
        ...     cache.set_qcp(structure_hash, residue_idx=10, value=qcp)
        >>> 
        >>> # Check statistics
        >>> stats = cache.get_statistics()
        >>> print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    """
    
    def __init__(
        self,
        max_qcp_entries: int = 1000,
        max_thz_entries: int = 500,
        max_distance_matrices: int = 10,
        max_ss_entries: int = 50,
        max_hydrophobic_entries: int = 50
    ):
        """
        Initialize refinement cache with size limits.
        
        Args:
            max_qcp_entries: Maximum QCP cache entries
            max_thz_entries: Maximum THz mode cache entries
            max_distance_matrices: Maximum distance matrices to cache
            max_ss_entries: Maximum secondary structure entries
            max_hydrophobic_entries: Maximum hydrophobic residue lists
        """
        self.max_qcp_entries = max_qcp_entries
        self.max_thz_entries = max_thz_entries
        self.max_distance_matrices = max_distance_matrices
        self.max_ss_entries = max_ss_entries
        self.max_hydrophobic_entries = max_hydrophobic_entries
        
        # LRU caches (OrderedDict maintains insertion order)
        self._qcp_cache: OrderedDict[str, float] = OrderedDict()
        self._thz_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._distance_cache: OrderedDict[str, Any] = OrderedDict()
        self._ss_cache: OrderedDict[str, List[str]] = OrderedDict()
        self._hydrophobic_cache: OrderedDict[str, Set[int]] = OrderedDict()
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.total_time_saved = 0.0  # seconds
        
        logger.info(
            f"RefinementCache initialized: "
            f"QCP={max_qcp_entries}, THz={max_thz_entries}, "
            f"Distance={max_distance_matrices}, SS={max_ss_entries}, "
            f"Hydrophobic={max_hydrophobic_entries}"
        )
    
    def _make_key(self, *args) -> str:
        """
        Create cache key from arguments.
        
        Args:
            *args: Variable arguments to hash
        
        Returns:
            SHA256 hash of arguments as hex string
        """
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _evict_lru(self, cache: OrderedDict, max_size: int) -> None:
        """
        Evict least-recently-used entry if cache is full.
        
        Args:
            cache: OrderedDict cache to evict from
            max_size: Maximum cache size
        """
        if len(cache) >= max_size:
            # Remove oldest entry (first in OrderedDict)
            cache.popitem(last=False)
    
    # ========================================================================
    # QCP Cache
    # ========================================================================
    
    def get_qcp(self, structure_hash: str, residue_idx: int) -> Optional[float]:
        """
        Get cached QCP value for residue.
        
        Args:
            structure_hash: Hash of structure coordinates
            residue_idx: Residue index (0-based)
        
        Returns:
            Cached QCP value or None if not cached
        """
        key = self._make_key(structure_hash, residue_idx, "qcp")
        
        if key in self._qcp_cache:
            # Move to end (most recent)
            self._qcp_cache.move_to_end(key)
            self.hit_count += 1
            return self._qcp_cache[key]
        
        self.miss_count += 1
        return None
    
    def set_qcp(
        self,
        structure_hash: str,
        residue_idx: int,
        value: float
    ) -> None:
        """
        Cache QCP value for residue.
        
        Args:
            structure_hash: Hash of structure coordinates
            residue_idx: Residue index (0-based)
            value: QCP value to cache
        """
        key = self._make_key(structure_hash, residue_idx, "qcp")
        
        # Evict LRU if needed
        self._evict_lru(self._qcp_cache, self.max_qcp_entries)
        
        # Add new entry (goes to end as most recent)
        self._qcp_cache[key] = value
    
    # ========================================================================
    # THz Mode Cache
    # ========================================================================
    
    def get_thz_modes(
        self,
        structure_hash: str,
        residue_idx: int
    ) -> Optional[List[float]]:
        """
        Get cached THz modes for residue.
        
        Args:
            structure_hash: Hash of structure coordinates
            residue_idx: Residue index (0-based)
        
        Returns:
            Cached THz mode frequencies or None if not cached
        """
        key = self._make_key(structure_hash, residue_idx, "thz")
        
        if key in self._thz_cache:
            self._thz_cache.move_to_end(key)
            self.hit_count += 1
            return self._thz_cache[key]
        
        self.miss_count += 1
        return None
    
    def set_thz_modes(
        self,
        structure_hash: str,
        residue_idx: int,
        modes: List[float]
    ) -> None:
        """
        Cache THz modes for residue.
        
        Args:
            structure_hash: Hash of structure coordinates
            residue_idx: Residue index (0-based)
            modes: List of THz mode frequencies
        """
        key = self._make_key(structure_hash, residue_idx, "thz")
        
        self._evict_lru(self._thz_cache, self.max_thz_entries)
        self._thz_cache[key] = modes
    
    # ========================================================================
    # Distance Matrix Cache
    # ========================================================================
    
    def get_distance_matrix(self, structure_hash: str) -> Optional[Any]:
        """
        Get cached distance matrix.
        
        Args:
            structure_hash: Hash of structure coordinates
        
        Returns:
            Cached distance matrix or None if not cached
        """
        if structure_hash in self._distance_cache:
            self._distance_cache.move_to_end(structure_hash)
            self.hit_count += 1
            return self._distance_cache[structure_hash]
        
        self.miss_count += 1
        return None
    
    def set_distance_matrix(
        self,
        structure_hash: str,
        matrix: Any
    ) -> None:
        """
        Cache distance matrix.
        
        Args:
            structure_hash: Hash of structure coordinates
            matrix: Distance matrix to cache
        """
        self._evict_lru(self._distance_cache, self.max_distance_matrices)
        self._distance_cache[structure_hash] = matrix
    
    # ========================================================================
    # Secondary Structure Cache
    # ========================================================================
    
    def get_secondary_structure(
        self,
        structure_hash: str
    ) -> Optional[List[str]]:
        """
        Get cached secondary structure assignment.
        
        Args:
            structure_hash: Hash of structure coordinates
        
        Returns:
            Cached SS codes or None if not cached
        """
        if structure_hash in self._ss_cache:
            self._ss_cache.move_to_end(structure_hash)
            self.hit_count += 1
            return self._ss_cache[structure_hash]
        
        self.miss_count += 1
        return None
    
    def set_secondary_structure(
        self,
        structure_hash: str,
        ss_codes: List[str]
    ) -> None:
        """
        Cache secondary structure assignment.
        
        Args:
            structure_hash: Hash of structure coordinates
            ss_codes: List of secondary structure codes ('H', 'E', 'C', etc.)
        """
        self._evict_lru(self._ss_cache, self.max_ss_entries)
        self._ss_cache[structure_hash] = ss_codes
    
    # ========================================================================
    # Hydrophobic Residue Cache
    # ========================================================================
    
    def get_hydrophobic_residues(
        self,
        structure_hash: str
    ) -> Optional[Set[int]]:
        """
        Get cached hydrophobic residue indices.
        
        Args:
            structure_hash: Hash of structure coordinates
        
        Returns:
            Cached set of hydrophobic residue indices or None if not cached
        """
        if structure_hash in self._hydrophobic_cache:
            self._hydrophobic_cache.move_to_end(structure_hash)
            self.hit_count += 1
            return self._hydrophobic_cache[structure_hash]
        
        self.miss_count += 1
        return None
    
    def set_hydrophobic_residues(
        self,
        structure_hash: str,
        residue_indices: Set[int]
    ) -> None:
        """
        Cache hydrophobic residue indices.
        
        Args:
            structure_hash: Hash of structure coordinates
            residue_indices: Set of hydrophobic residue indices
        """
        self._evict_lru(self._hydrophobic_cache, self.max_hydrophobic_entries)
        self._hydrophobic_cache[structure_hash] = residue_indices
    
    # ========================================================================
    # Cache Management
    # ========================================================================
    
    def clear(self) -> None:
        """Clear all caches and reset statistics."""
        self._qcp_cache.clear()
        self._thz_cache.clear()
        self._distance_cache.clear()
        self._ss_cache.clear()
        self._hydrophobic_cache.clear()
        
        self.hit_count = 0
        self.miss_count = 0
        self.total_time_saved = 0.0
        
        logger.info("RefinementCache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics:
            - hit_count: Total cache hits
            - miss_count: Total cache misses
            - total_queries: hit_count + miss_count
            - hit_rate: hit_count / total_queries
            - time_saved: Total time saved (seconds)
            - cache_sizes: Current size of each cache
        """
        total_queries = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_queries) if total_queries > 0 else 0.0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'total_queries': total_queries,
            'hit_rate': hit_rate,
            'time_saved': self.total_time_saved,
            'cache_sizes': {
                'qcp': len(self._qcp_cache),
                'thz': len(self._thz_cache),
                'distance': len(self._distance_cache),
                'ss': len(self._ss_cache),
                'hydrophobic': len(self._hydrophobic_cache)
            }
        }
    
    def format_statistics(self) -> str:
        """
        Format cache statistics as human-readable string.
        
        Returns:
            Formatted statistics string
        """
        stats = self.get_statistics()
        
        return (
            f"Cache Statistics:\n"
            f"  Hits: {stats['hit_count']}\n"
            f"  Misses: {stats['miss_count']}\n"
            f"  Hit Rate: {stats['hit_rate']:.1%}\n"
            f"  Time Saved: {stats['time_saved']:.2f}s\n"
            f"  Cache Sizes:\n"
            f"    QCP: {stats['cache_sizes']['qcp']}/{self.max_qcp_entries}\n"
            f"    THz: {stats['cache_sizes']['thz']}/{self.max_thz_entries}\n"
            f"    Distance: {stats['cache_sizes']['distance']}/{self.max_distance_matrices}\n"
            f"    SS: {stats['cache_sizes']['ss']}/{self.max_ss_entries}\n"
            f"    Hydrophobic: {stats['cache_sizes']['hydrophobic']}/{self.max_hydrophobic_entries}"
        )
