"""
Unit Tests for THz Resonance Detection in Mediator Agents

This module tests the THz resonance detection functionality implemented in Task 6.

Test Coverage:
- Spectral correlation calculation
- THz signature caching
- DBSCAN clustering with spectral distance
- Significance scoring
- Pattern detection creation
- Error handling

Author: UBF Protein System
Date: November 9, 2025
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock
import pandas as pd

from ubf_protein.mediator_agent import MediatorAgent
from ubf_protein.mediator_config import MediatorConfig
from ubf_protein.models import Conformation
from ubf_protein.pattern_detection import (
    PatternDetection, PatternType, PatternSignificance, THzResonanceData
)
from ubf_protein.geometric_attractor import GeometricAttractorAnalyzer


class TestSpectralCorrelation:
    """Test spectral correlation calculations."""
    
    def test_identical_spectra_perfect_correlation(self):
        """Identical spectra should have correlation = 1.0."""
        # Create mock mediator
        mediator = self._create_mock_mediator()
        
        # Create identical spectra
        freq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        intensities = np.array([0.5, 1.0, 0.8, 0.3, 0.1])
        
        # Calculate correlation
        corr = mediator._calculate_spectral_correlation(freq, intensities, freq, intensities)
        
        # Should be very close to 1.0 (allow small numerical error)
        assert 0.99 <= corr <= 1.0
    
    def test_uncorrelated_spectra_low_correlation(self):
        """Uncorrelated spectra should have correlation near 0."""
        mediator = self._create_mock_mediator()
        
        # Create uncorrelated spectra
        freq1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        int1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Peak at 1 THz
        
        freq2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        int2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # Peak at 5 THz
        
        corr = mediator._calculate_spectral_correlation(freq1, int1, freq2, int2)
        
        # Should be low correlation
        assert corr < 0.5
    
    def test_similar_spectra_high_correlation(self):
        """Similar spectra should have high correlation."""
        mediator = self._create_mock_mediator()
        
        freq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        int1 = np.array([0.5, 1.0, 0.8, 0.3, 0.1])
        int2 = np.array([0.4, 0.9, 0.7, 0.4, 0.2])  # Similar but not identical
        
        corr = mediator._calculate_spectral_correlation(freq, int1, freq, int2)
        
        # Should have high correlation
        assert corr > 0.8
    
    def test_zero_spectrum_returns_zero(self):
        """Zero spectrum should return correlation = 0."""
        mediator = self._create_mock_mediator()
        
        freq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        int1 = np.array([0.5, 1.0, 0.8, 0.3, 0.1])
        int2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Zero spectrum
        
        corr = mediator._calculate_spectral_correlation(freq, int1, freq, int2)
        
        assert corr == 0.0
    
    def test_different_frequency_grids_interpolated(self):
        """Different frequency grids should be interpolated correctly."""
        mediator = self._create_mock_mediator()
        
        # Different frequency grids
        freq1 = np.array([1.0, 3.0, 5.0])
        int1 = np.array([1.0, 0.5, 0.2])
        
        freq2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        int2 = np.array([0.9, 0.7, 0.4, 0.3, 0.1])
        
        corr = mediator._calculate_spectral_correlation(freq1, int1, freq2, int2)
        
        # Should compute correlation successfully
        assert 0.0 <= corr <= 1.0
    
    def _create_mock_mediator(self):
        """Create mock mediator for testing."""
        config = MediatorConfig()
        qcpp_adapter = Mock()
        geometric_analyzer = Mock()
        shared_memory = Mock()
        
        return MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )


class TestTHzSignatureCaching:
    """Test THz signature caching functionality."""
    
    def test_cache_stores_thz_signature(self):
        """THz signature should be cached after first calculation."""
        mediator, qcpp_adapter = self._create_mediator_with_mock_qcpp()
        conformation = self._create_mock_conformation()
        
        # First call - should calculate
        mediator._detect_thz_resonance(conformation)
        
        # Check cache
        conf_hash = mediator._generate_conformation_hash(conformation)
        assert conf_hash in mediator.thz_signature_cache
        
        # Verify QCPP was called
        assert qcpp_adapter.predictor.predict_thz_spectrum.call_count == 1
    
    def test_cache_hit_skips_qcpp_calculation(self):
        """Cached THz signature should skip QCPP calculation."""
        mediator, qcpp_adapter = self._create_mediator_with_mock_qcpp()
        conformation = self._create_mock_conformation()
        
        # First call - calculates and caches
        mediator._detect_thz_resonance(conformation)
        
        # Second call - should use cache
        mediator._detect_thz_resonance(conformation)
        
        # QCPP should only be called once
        assert qcpp_adapter.predictor.predict_thz_spectrum.call_count == 1
    
    def test_cache_eviction_at_size_limit(self):
        """Cache should evict oldest entries when size limit reached."""
        config = MediatorConfig(cache_size=5)  # Small cache for testing
        mediator, qcpp_adapter = self._create_mediator_with_mock_qcpp(config)
        
        # Create 6 different conformations
        for i in range(6):
            conf = self._create_mock_conformation(coords_offset=i * 10.0)
            mediator._detect_thz_resonance(conf)
        
        # Cache should have max 5 entries
        assert len(mediator.thz_signature_cache) <= 5
    
    def test_cache_stores_frequency_and_intensity(self):
        """Cached signature should include frequencies and intensities."""
        mediator, qcpp_adapter = self._create_mediator_with_mock_qcpp()
        conformation = self._create_mock_conformation()
        
        mediator._detect_thz_resonance(conformation)
        
        conf_hash = mediator._generate_conformation_hash(conformation)
        signature = mediator.thz_signature_cache[conf_hash]
        
        assert 'frequencies' in signature
        assert 'intensities' in signature
        assert isinstance(signature['frequencies'], np.ndarray)
        assert isinstance(signature['intensities'], np.ndarray)
    
    def _create_mediator_with_mock_qcpp(self, config=None):
        """Create mediator with mock QCPP adapter."""
        if config is None:
            config = MediatorConfig()
        
        # Create mock QCPP adapter with predict_thz_spectrum method
        qcpp_adapter = Mock()
        predictor_mock = Mock()
        
        # Mock THz spectrum data
        thz_df = pd.DataFrame({
            'frequency': [1.0, 1.5, 2.0, 2.5, 3.0],
            'intensity': [0.5, 1.0, 0.8, 0.3, 0.1]
        })
        predictor_mock.predict_thz_spectrum.return_value = thz_df
        qcpp_adapter.predictor = predictor_mock
        
        geometric_analyzer = Mock()
        shared_memory = Mock()
        
        mediator = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )
        
        return mediator, qcpp_adapter
    
    def _create_mock_conformation(self, coords_offset=0.0):
        """Create mock conformation with unique coordinates."""
        coords = [
            (1.0 + coords_offset, 2.0, 3.0),
            (4.0 + coords_offset, 5.0, 6.0),
            (7.0 + coords_offset, 8.0, 9.0),
        ]
        return Conformation(
            conformation_id="test_conf_001",
            sequence="ACE",
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=5.0,
            secondary_structure=['C', 'C', 'C'],
            phi_angles=[0.0, 0.0, 0.0],
            psi_angles=[0.0, 0.0, 0.0],
            available_move_types=['rotation', 'translation'],
            structural_constraints={}
        )


class TestTHzClustering:
    """Test THz signature clustering with DBSCAN."""
    
    def test_single_signature_no_clustering(self):
        """Single signature should not create clusters."""
        mediator, _ = self._create_mediator_with_signatures(1)
        
        # Should return None (not enough data)
        signatures = list(mediator.thz_signature_cache.values())
        result = mediator._cluster_thz_signatures(signatures, signatures[0]['hash'])
        
        assert result is None
    
    def test_identical_signatures_form_cluster(self):
        """Identical signatures should form a cluster."""
        mediator = self._create_mediator_with_identical_signatures(5)
        
        signatures = list(mediator.thz_signature_cache.values())
        current_hash = signatures[0]['hash']
        
        result = mediator._cluster_thz_signatures(signatures, current_hash)
        
        # Should detect cluster
        assert result is not None
        assert result.pattern_type == PatternType.THZ
        assert result.thz_data is not None
        assert result.thz_data.cluster_size == 5
    
    def test_diverse_signatures_no_cluster(self):
        """Diverse signatures should not form clusters."""
        mediator = self._create_mediator_with_diverse_signatures(5)
        
        signatures = list(mediator.thz_signature_cache.values())
        current_hash = signatures[0]['hash']
        
        result = mediator._cluster_thz_signatures(signatures, current_hash)
        
        # Should not detect cluster (noise point)
        assert result is None
    
    def test_cluster_size_affects_significance(self):
        """Larger clusters should have higher significance."""
        # Create mediator with 10 identical signatures
        mediator = self._create_mediator_with_identical_signatures(10)
        
        signatures = list(mediator.thz_signature_cache.values())
        current_hash = signatures[0]['hash']
        
        result = mediator._cluster_thz_signatures(signatures, current_hash)
        
        # All 10 should be in same cluster
        assert result is not None
        assert result.thz_data is not None
        assert result.thz_data.cluster_size == 10
        # Significance should be high (10/10 = 1.0)
        assert result.significance == PatternSignificance.HIGH
    
    def test_minimum_significance_threshold(self):
        """Clusters below 10% significance should be ignored."""
        # Create 20 signatures: 1 similar cluster + 19 diverse
        mediator = self._create_mediator_with_mixed_signatures()
        
        signatures = list(mediator.thz_signature_cache.values())
        # Use hash from first signature (in small cluster)
        current_hash = signatures[0]['hash']
        
        result = mediator._cluster_thz_signatures(signatures, current_hash)
        
        # Should return None if cluster < 10% of total
        # Small cluster (2/20 = 10%) should barely pass
        if result:
            assert result.thz_data is not None
            assert result.thz_data.cluster_size / len(signatures) >= 0.1
    
    def test_thz_data_fields_populated(self):
        """THzResonanceData should have all required fields."""
        mediator = self._create_mediator_with_identical_signatures(5)
        
        signatures = list(mediator.thz_signature_cache.values())
        current_hash = signatures[0]['hash']
        
        result = mediator._cluster_thz_signatures(signatures, current_hash)
        
        assert result is not None
        thz_data = result.thz_data
        assert thz_data is not None
        
        # Check all fields present and valid
        assert thz_data.cluster_id >= 0
        assert thz_data.cluster_size > 0
        assert 0.0 <= thz_data.similarity_score <= 1.0
        assert thz_data.dominant_frequency > 0.0
        assert thz_data.spectral_entropy >= 0.0
    
    def _create_mediator_with_signatures(self, count):
        """Create mediator with specified number of signatures."""
        config = MediatorConfig()
        qcpp_adapter = Mock()
        geometric_analyzer = Mock()
        shared_memory = Mock()
        
        mediator = MediatorAgent(
            protein_sequence="A" * count,
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )
        
        # Add signatures to cache
        for i in range(count):
            sig = {
                'frequencies': np.array([1.0, 2.0, 3.0]),
                'intensities': np.array([0.5, 1.0, 0.3]),
                'hash': f"hash{i:012d}",  # 16 characters total (hash + 12 digits)
                'timestamp': time.time(),
            }
            mediator.thz_signature_cache[f"hash{i:012d}"] = sig
        
        return mediator, qcpp_adapter
    
    def _create_mediator_with_identical_signatures(self, count):
        """Create mediator with identical THz signatures."""
        config = MediatorConfig()
        qcpp_adapter = Mock()
        geometric_analyzer = Mock()
        shared_memory = Mock()
        
        mediator = MediatorAgent(
            protein_sequence="A" * count,
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )
        
        # All signatures identical
        base_freq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base_int = np.array([0.5, 1.0, 0.8, 0.3, 0.1])
        
        for i in range(count):
            sig = {
                'frequencies': base_freq.copy(),
                'intensities': base_int.copy(),
                'hash': f"hash{i:012d}",  # 16 characters total
                'timestamp': time.time(),
            }
            mediator.thz_signature_cache[f"hash{i:012d}"] = sig
        
        return mediator
    
    def _create_mediator_with_diverse_signatures(self, count):
        """Create mediator with diverse THz signatures."""
        config = MediatorConfig()
        qcpp_adapter = Mock()
        geometric_analyzer = Mock()
        shared_memory = Mock()
        
        mediator = MediatorAgent(
            protein_sequence="A" * count,
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )
        
        # Each signature very different
        for i in range(count):
            # Shift peak to different frequency for each signature
            intensities = np.zeros(5)
            intensities[i % 5] = 1.0
            
            sig = {
                'frequencies': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                'intensities': intensities,
                'hash': f"hash{i:012d}",  # 16 characters total
                'timestamp': time.time(),
            }
            mediator.thz_signature_cache[f"hash{i:012d}"] = sig
        
        return mediator
    
    def _create_mediator_with_mixed_signatures(self):
        """Create mediator with 2 similar + 18 diverse signatures."""
        config = MediatorConfig()
        qcpp_adapter = Mock()
        geometric_analyzer = Mock()
        shared_memory = Mock()
        
        mediator = MediatorAgent(
            protein_sequence="A" * 20,
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )
        
        # First 2 similar
        base_int = np.array([0.5, 1.0, 0.8, 0.3, 0.1])
        for i in range(2):
            sig = {
                'frequencies': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                'intensities': base_int.copy(),
                'hash': f"hash{i:012d}",  # 16 characters total
                'timestamp': time.time(),
            }
            mediator.thz_signature_cache[f"hash{i:012d}"] = sig
        
        # Remaining 18 diverse
        for i in range(2, 20):
            intensities = np.zeros(5)
            intensities[i % 5] = 1.0
            
            sig = {
                'frequencies': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                'intensities': intensities,
                'hash': f"hash{i:012d}",  # 16 characters total
                'timestamp': time.time(),
            }
            mediator.thz_signature_cache[f"hash{i:012d}"] = sig
        
        return mediator


class TestTHzDetectionIntegration:
    """Integration tests for full THz detection pipeline."""
    
    def test_detect_patterns_calls_thz_detection(self):
        """detect_patterns should call THz detection when enabled."""
        mediator, qcpp_adapter = self._create_mediator_with_qcpp()
        conformation = self._create_mock_conformation()
        
        # Enable THz detection
        mediator.config.enable_thz_detection = True
        
        # Call detect_patterns
        patterns = mediator.detect_patterns(conformation)
        
        # Should have called QCPP
        assert qcpp_adapter.predictor.predict_thz_spectrum.called
    
    def test_detect_patterns_skips_when_disabled(self):
        """detect_patterns should skip THz detection when disabled."""
        mediator, qcpp_adapter = self._create_mediator_with_qcpp()
        conformation = self._create_mock_conformation()
        
        # Disable THz detection
        mediator.config.enable_thz_detection = False
        
        # Call detect_patterns
        patterns = mediator.detect_patterns(conformation)
        
        # Should not have called QCPP
        assert not qcpp_adapter.predictor.predict_thz_spectrum.called
    
    def test_detection_statistics_updated(self):
        """Statistics should be updated after detection."""
        mediator, _ = self._create_mediator_with_qcpp()
        
        # Create multiple similar conformations to form cluster
        for i in range(5):
            conf = self._create_mock_conformation()
            mediator.detect_patterns(conf)
        
        # Check statistics
        stats = mediator.detection_statistics
        assert stats['cache_misses'] > 0  # First detection is cache miss
    
    def test_qcpp_failure_handled_gracefully(self):
        """QCPP failures should not crash detection."""
        mediator, qcpp_adapter = self._create_mediator_with_qcpp()
        
        # Make QCPP raise exception
        qcpp_adapter.predictor.predict_thz_spectrum.side_effect = Exception("QCPP error")
        
        conformation = self._create_mock_conformation()
        
        # Should not raise exception
        patterns = mediator.detect_patterns(conformation)
        
        # Folding detection is QCPP-independent, so should still return folding patterns
        # even when QCPP/THz fails. THz pattern should be absent.
        assert len(patterns) >= 1  # At least folding pattern
        pattern_types = [p.pattern_type for p in patterns]
        assert PatternType.FOLDING in pattern_types  # Folding detection works
        assert PatternType.THZ not in pattern_types  # THz detection failed as expected
    
    def _create_mediator_with_qcpp(self):
        """Create mediator with mock QCPP."""
        config = MediatorConfig()
        
        qcpp_adapter = Mock()
        predictor_mock = Mock()
        
        thz_df = pd.DataFrame({
            'frequency': [1.0, 1.5, 2.0, 2.5, 3.0],
            'intensity': [0.5, 1.0, 0.8, 0.3, 0.1]
        })
        predictor_mock.predict_thz_spectrum.return_value = thz_df
        qcpp_adapter.predictor = predictor_mock
        
        geometric_analyzer = Mock()
        shared_memory = Mock()
        
        return MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        ), qcpp_adapter
    
    def _create_mock_conformation(self):
        """Create mock conformation."""
        coords = [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0),
        ]
        return Conformation(
            conformation_id="test_conf_001",
            sequence="ACE",
            atom_coordinates=coords,
            energy=-100.0,
            rmsd_to_native=5.0,
            secondary_structure=['C', 'C', 'C'],
            phi_angles=[0.0, 0.0, 0.0],
            psi_angles=[0.0, 0.0, 0.0],
            available_move_types=['rotation', 'translation'],
            structural_constraints={}
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
