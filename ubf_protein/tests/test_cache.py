"""
Unit Tests for LRU Cache with TTL

Tests cover:
- Cache hit performance (< 1ms target)
- LRU eviction when size limit reached
- TTL expiration removes stale entries
- Cache statistics accuracy
- Memory limit behavior

Author: UBF Protein System
Date: November 9, 2025
"""

import pytest
import time
from ubf_protein.geometric_attractor import (
    LRUCache,
    GeometricAnalysisResult,
    GeometricAttractorAnalyzer,
)


def create_test_result(index: int = 0, timestamp: float | None = None) -> GeometricAnalysisResult:
    """Helper to create test GeometricAnalysisResult with valid 16-char hash."""
    if timestamp is None:
        timestamp = time.time()
    return GeometricAnalysisResult(
        golden_ratio_percentage=float(index * 10) % 100.0,
        phi_pattern_count=index,
        tetrahedron_similarity=0.5,
        cube_similarity=0.5,
        octahedron_similarity=0.5,
        dodecahedron_similarity=0.5,
        icosahedron_similarity=0.5,
        rotational_symmetry=0.5,
        local_symmetry=0.5,
        radius_of_gyration=10.0,
        asphericity=0.3,
        conformation_hash=f"{index:016d}",  # Exactly 16 digits
        timestamp=timestamp,
        num_residues=10,
    )


class TestLRUCache:
    """Tests for LRUCache class."""
    
    def test_cache_initialization(self):
        """Test cache initializes with correct defaults."""
        cache = LRUCache()
        
        assert cache.max_size == 1000
        assert cache.ttl_seconds == 3600.0
        
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['evictions'] == 0
    
    def test_cache_initialization_custom(self):
        """Test cache initializes with custom parameters."""
        cache = LRUCache(max_size=500, ttl_seconds=1800.0)
        
        assert cache.max_size == 500
        assert cache.ttl_seconds == 1800.0
    
    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache()
        
        # Create a mock result
        result = GeometricAnalysisResult(
            golden_ratio_percentage=50.0,
            phi_pattern_count=10,
            tetrahedron_similarity=0.5,
            cube_similarity=0.6,
            octahedron_similarity=0.7,
            dodecahedron_similarity=0.8,
            icosahedron_similarity=0.9,
            rotational_symmetry=0.7,
            local_symmetry=0.6,
            radius_of_gyration=10.0,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=time.time(),
            num_residues=10,
        )
        
        # Put and get
        cache.put("test_key", result)
        retrieved = cache.get("test_key")
        
        assert retrieved is not None
        assert retrieved.conformation_hash == result.conformation_hash
        assert retrieved.golden_ratio_percentage == result.golden_ratio_percentage
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LRUCache()
        
        result = cache.get("nonexistent_key")
        
        assert result is None
        
        stats = cache.get_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0
    
    def test_cache_hit_performance(self):
        """Test cache hit returns results in < 1ms."""
        cache = LRUCache()
        analyzer = GeometricAttractorAnalyzer()
        
        # Create and cache a result
        conformation = {
            'coordinates': [(float(i), float(i), float(i)) for i in range(50)]
        }
        
        # First call (cache miss)
        result1 = analyzer.analyze_conformation(conformation)
        
        # Measure cache hit time
        start = time.perf_counter()
        result2 = analyzer.analyze_conformation(conformation)
        elapsed = time.perf_counter() - start
        
        # Should be very fast (< 1ms = 0.001s)
        # We'll be generous and allow up to 10ms for slower systems
        assert elapsed < 0.01  # 10ms max
        
        # Results should be identical
        assert result1.conformation_hash == result2.conformation_hash
    
    def test_lru_eviction(self):
        """Test LRU eviction when size limit reached."""
        cache = LRUCache(max_size=3)  # Small cache for testing
        
        # Add 5 items to cache with max_size=3
        for i in range(5):
            result = create_test_result(i)
            cache.put(f"key{i}", result)
        
        # Cache should only hold 3 items (last 3 added)
        stats = cache.get_stats()
        assert stats['size'] == 3
        assert stats['evictions'] == 2  # 2 items evicted
        
        # First two items should be evicted (LRU)
        assert cache.get("key0") is None
        assert cache.get("key1") is None
        
        # Last three should be present
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None
    
    def test_lru_ordering(self):
        """Test that accessing items updates LRU order."""
        cache = LRUCache(max_size=3)
        
        # Fill cache
        for i in range(3):
            result = create_test_result(i)
            cache.put(f"key{i}", result)
        
        # Access key0 to make it most recently used
        cache.get("key0")
        
        # Add new item (should evict key1, the least recently used)
        new_result = create_test_result(99)
        cache.put("key3", new_result)
        
        # key0 should still be present (was accessed)
        assert cache.get("key0") is not None
        # key1 should be evicted (least recently used)
        assert cache.get("key1") is None
        # key2 and key3 should be present
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
    
    def test_ttl_expiration(self):
        """Test TTL expiration removes stale entries."""
        cache = LRUCache(max_size=100, ttl_seconds=0.1)  # 100ms TTL
        
        result = GeometricAnalysisResult(
            golden_ratio_percentage=50.0,
            phi_pattern_count=10,
            tetrahedron_similarity=0.5,
            cube_similarity=0.5,
            octahedron_similarity=0.5,
            dodecahedron_similarity=0.5,
            icosahedron_similarity=0.5,
            rotational_symmetry=0.5,
            local_symmetry=0.5,
            radius_of_gyration=10.0,
            asphericity=0.3,
            conformation_hash="a1b2c3d4e5f6g7h8",
            timestamp=time.time(),
            num_residues=10,
        )
        
        cache.put("test_key", result)
        
        # Should be retrievable immediately
        assert cache.get("test_key") is not None
        
        # Wait for TTL to expire
        time.sleep(0.15)  # 150ms
        
        # Should be None after expiration
        expired_result = cache.get("test_key")
        assert expired_result is None
        
        # Cache should be empty after expiration cleanup
        stats = cache.get_stats()
        assert stats['size'] == 0
    
    def test_cache_update_existing_key(self):
        """Test updating an existing key refreshes timestamp."""
        cache = LRUCache(ttl_seconds=0.2)  # 200ms TTL
        
        result1 = create_test_result(1)
        cache.put("test_key", result1)
        
        # Wait 100ms
        time.sleep(0.1)
        
        # Update with new result
        result2 = create_test_result(2)
        cache.put("test_key", result2)
        
        # Wait another 150ms (total 250ms from first put, but only 150ms from update)
        time.sleep(0.15)
        
        # Should still be valid (timestamp was refreshed on update)
        retrieved = cache.get("test_key")
        assert retrieved is not None
        assert retrieved.phi_pattern_count == 2  # Updated value
    
    def test_cache_statistics(self):
        """Test cache statistics accuracy."""
        cache = LRUCache(max_size=10)
        
        # Add 5 entries
        for i in range(5):
            result = create_test_result(i)
            cache.put(f"key{i}", result)
        
        # Perform some gets (hits and misses)
        cache.get("key0")  # hit
        cache.get("key1")  # hit
        cache.get("key99")  # miss
        cache.get("key2")  # hit
        cache.get("key98")  # miss
        
        stats = cache.get_stats()
        
        assert stats['size'] == 5
        assert stats['max_size'] == 10
        assert stats['hits'] == 3
        assert stats['misses'] == 2
        assert stats['total_requests'] == 5
        assert stats['hit_rate'] == 0.6  # 3/5 = 0.6
        assert stats['evictions'] == 0
    
    def test_cache_clear(self):
        """Test cache clear removes all entries."""
        cache = LRUCache()
        
        # Add some entries
        for i in range(5):
            result = create_test_result(i)
            cache.put(f"key{i}", result)
        
        assert cache.get_stats()['size'] == 5
        
        # Clear cache
        cache.clear()
        
        stats = cache.get_stats()
        assert stats['size'] == 0
        
        # All entries should be gone
        for i in range(5):
            assert cache.get(f"key{i}") is None


class TestGeometricAttractorCacheIntegration:
    """Integration tests for cache in GeometricAttractorAnalyzer."""
    
    def test_analyzer_uses_cache(self):
        """Test that analyzer properly uses cache."""
        analyzer = GeometricAttractorAnalyzer()
        
        conformation = {
            'coordinates': [(float(i), 0.0, 0.0) for i in range(10)]
        }
        
        # First call (cache miss)
        result1 = analyzer.analyze_conformation(conformation)
        stats1 = analyzer.get_cache_stats()
        
        assert stats1['misses'] == 1
        assert stats1['hits'] == 0
        assert stats1['size'] == 1
        
        # Second call (cache hit)
        result2 = analyzer.analyze_conformation(conformation)
        stats2 = analyzer.get_cache_stats()
        
        assert stats2['misses'] == 1
        assert stats2['hits'] == 1
        assert stats2['size'] == 1
        
        # Results should be identical
        assert result1.conformation_hash == result2.conformation_hash
    
    def test_analyzer_cache_different_conformations(self):
        """Test analyzer caches different conformations separately."""
        analyzer = GeometricAttractorAnalyzer()
        
        conf1 = {'coordinates': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]}
        conf2 = {'coordinates': [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)]}
        
        result1 = analyzer.analyze_conformation(conf1)
        result2 = analyzer.analyze_conformation(conf2)
        
        # Different hashes
        assert result1.conformation_hash != result2.conformation_hash
        
        # Cache should have 2 entries
        stats = analyzer.get_cache_stats()
        assert stats['size'] == 2
    
    def test_analyzer_clear_cache(self):
        """Test analyzer clear_cache method."""
        analyzer = GeometricAttractorAnalyzer()
        
        # Populate cache
        for i in range(5):
            conf = {'coordinates': [(float(j+i), 0.0, 0.0) for j in range(10)]}
            analyzer.analyze_conformation(conf)
        
        assert analyzer.get_cache_stats()['size'] == 5
        
        # Clear cache
        analyzer.clear_cache()
        
        assert analyzer.get_cache_stats()['size'] == 0
    
    def test_analyzer_cache_with_custom_settings(self):
        """Test analyzer with custom cache settings."""
        analyzer = GeometricAttractorAnalyzer(
            cache_size=50,
            cache_ttl=1800.0,
        )
        
        assert analyzer.cache.max_size == 50
        assert analyzer.cache.ttl_seconds == 1800.0


class TestCacheMemoryBehavior:
    """Tests for cache memory behavior."""
    
    def test_cache_respects_max_size(self):
        """Test that cache never exceeds max_size."""
        cache = LRUCache(max_size=10)
        
        # Add 20 items
        for i in range(20):
            result = create_test_result(i)
            cache.put(f"key{i}", result)
            
            # Size should never exceed max_size
            assert cache.get_stats()['size'] <= 10
    
    def test_eviction_count_accuracy(self):
        """Test that eviction count is accurate."""
        cache = LRUCache(max_size=5)
        
        # Add 10 items (should cause 5 evictions)
        for i in range(10):
            result = create_test_result(i)
            cache.put(f"key{i}", result)
        
        stats = cache.get_stats()
        assert stats['evictions'] == 5
        assert stats['size'] == 5
