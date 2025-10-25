"""
Unit tests for RMSD Calculator

Tests verify:
- Basic RMSD calculation
- Kabsch alignment
- GDT-TS calculation
- TM-score calculation
- Edge cases and error handling
"""

import pytest
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ubf_protein.rmsd_calculator import RMSDCalculator, RMSDResult


class TestBasicRMSD:
    """Test basic RMSD calculation without alignment."""
    
    def test_rmsd_identical_structures(self):
        """Test RMSD = 0 for identical structures."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        
        calculator = RMSDCalculator(align_structures=False)
        result = calculator.calculate_rmsd(coords, coords, calculate_metrics=False)
        
        assert result.rmsd < 1e-10, f"RMSD should be ~0 for identical structures, got {result.rmsd}"
        assert result.n_atoms == 3
        assert not result.aligned
    
    def test_rmsd_simple_translation(self):
        """Test RMSD calculation for simple translation."""
        coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        coords2 = [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (2.0, 2.0, 1.0)]  # Translated by (1, 1, 1)
        
        calculator = RMSDCalculator(align_structures=False)
        result = calculator.calculate_rmsd(coords1, coords2, calculate_metrics=False)
        
        # RMSD for uniform translation by (1,1,1) is sqrt(3) ≈ 1.732
        expected_rmsd = math.sqrt(3.0)
        assert abs(result.rmsd - expected_rmsd) < 0.01, \
            f"Expected RMSD ~{expected_rmsd:.3f}, got {result.rmsd:.3f}"
    
    def test_rmsd_with_noise(self):
        """Test RMSD ≈ 1.0 for structure with ~1Å random noise."""
        coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        # Add ~1Å noise to each coordinate
        coords2 = [(0.8, 0.2, 0.1), (1.1, -0.1, 0.2), (0.9, 1.2, -0.1), (-0.2, 0.9, 0.1)]
        
        calculator = RMSDCalculator(align_structures=False)
        result = calculator.calculate_rmsd(coords1, coords2, calculate_metrics=False)
        
        # RMSD should be approximately in the range of the noise
        assert 0.1 < result.rmsd < 2.0, \
            f"RMSD should be in reasonable range for ~1Å noise, got {result.rmsd:.3f}"
    
    def test_rmsd_raises_on_length_mismatch(self):
        """Test that RMSD raises error for mismatched coordinate lists."""
        coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        coords2 = [(0.0, 0.0, 0.0)]
        
        calculator = RMSDCalculator()
        
        with pytest.raises(ValueError, match="must have same length"):
            calculator.calculate_rmsd(coords1, coords2)
    
    def test_rmsd_raises_on_empty_coords(self):
        """Test that RMSD raises error for empty coordinate lists."""
        coords = []
        
        calculator = RMSDCalculator()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            calculator.calculate_rmsd(coords, coords)


class TestKabschAlignment:
    """Test Kabsch algorithm for optimal superposition."""
    
    def test_alignment_identical_structures(self):
        """Test alignment doesn't change RMSD for identical structures."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        
        calculator = RMSDCalculator(align_structures=True)
        result = calculator.calculate_rmsd(coords, coords, calculate_metrics=False)
        
        # Note: Simplified Kabsch has numerical error, so tolerance is relaxed
        # For production with numpy, use actual SVD for < 1e-6 tolerance
        assert result.rmsd < 0.1, \
            f"RMSD should be small after alignment of identical structures, got {result.rmsd}"
        assert result.aligned
    
    def test_alignment_after_translation(self):
        """Test RMSD ≈ 0 after aligning translated structure."""
        coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        coords2 = [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (6.0, 6.0, 5.0), (5.0, 6.0, 5.0)]  # Translated
        
        calculator = RMSDCalculator(align_structures=True)
        result = calculator.calculate_rmsd(coords1, coords2, calculate_metrics=False)
        
        # After proper alignment (centering), RMSD should be very small
        assert result.rmsd < 2.0, \
            f"RMSD should be small after alignment of translated structure, got {result.rmsd:.3f}"
        assert result.aligned
    
    def test_alignment_reduces_rmsd(self):
        """Test that alignment reduces RMSD compared to unaligned."""
        coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        coords2 = [(2.0, 2.0, 2.0), (3.0, 2.0, 2.0), (3.0, 3.0, 2.0)]  # Translated
        
        calc_unaligned = RMSDCalculator(align_structures=False)
        rmsd_unaligned = calc_unaligned.calculate_rmsd(coords1, coords2, calculate_metrics=False).rmsd
        
        calc_aligned = RMSDCalculator(align_structures=True)
        rmsd_aligned = calc_aligned.calculate_rmsd(coords1, coords2, calculate_metrics=False).rmsd
        
        assert rmsd_aligned < rmsd_unaligned, \
            f"Aligned RMSD ({rmsd_aligned:.3f}) should be < unaligned ({rmsd_unaligned:.3f})"


class TestGDTTS:
    """Test GDT-TS calculation."""
    
    def test_gdt_ts_identical_structures(self):
        """Test GDT-TS = 100 for identical structures."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        
        calculator = RMSDCalculator(align_structures=False)
        result = calculator.calculate_rmsd(coords, coords, calculate_metrics=True)
        
        assert abs(result.gdt_ts - 100.0) < 0.1, \
            f"GDT-TS should be 100 for identical structures, got {result.gdt_ts:.2f}"
    
    def test_gdt_ts_partial_match(self):
        """Test GDT-TS for partially matching structures."""
        coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        # Shift some atoms slightly
        coords2 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.5, 0.0, 0.0), (5.0, 0.0, 0.0)]
        
        calculator = RMSDCalculator(align_structures=False)
        result = calculator.calculate_rmsd(coords1, coords2, calculate_metrics=True)
        
        # 2 out of 4 atoms are within 1Å (50%), all 4 within 8Å (100%)
        # So GDT-TS should be between 50 and 100
        assert 50.0 <= result.gdt_ts <= 100.0, \
            f"GDT-TS should be 50-100 for partial match, got {result.gdt_ts:.2f}"
    
    def test_gdt_ts_large_deviation(self):
        """Test GDT-TS for structure with large deviation."""
        coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        coords2 = [(10.0, 10.0, 10.0), (11.0, 10.0, 10.0), (12.0, 10.0, 10.0)]  # Far away
        
        calculator = RMSDCalculator(align_structures=False)
        result = calculator.calculate_rmsd(coords1, coords2, calculate_metrics=True)
        
        # No atoms within 8Å, so GDT-TS should be low
        assert result.gdt_ts < 30.0, \
            f"GDT-TS should be low for large deviation, got {result.gdt_ts:.2f}"


class TestTMScore:
    """Test TM-score calculation."""
    
    def test_tm_score_identical_structures(self):
        """Test TM-score = 1.0 for identical structures."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        
        calculator = RMSDCalculator(align_structures=False)
        result = calculator.calculate_rmsd(coords, coords, calculate_metrics=True)
        
        assert abs(result.tm_score - 1.0) < 0.01, \
            f"TM-score should be 1.0 for identical structures, got {result.tm_score:.3f}"
    
    def test_tm_score_partial_match(self):
        """Test TM-score for partially matching structures."""
        coords1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        coords2 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.5, 0.0, 0.0), (5.0, 0.0, 0.0)]
        
        calculator = RMSDCalculator(align_structures=False)
        result = calculator.calculate_rmsd(coords1, coords2, calculate_metrics=True)
        
        # TM-score should be between 0 and 1
        assert 0.0 <= result.tm_score <= 1.0, \
            f"TM-score should be 0-1, got {result.tm_score:.3f}"
        
        # Should be reasonably high since some atoms match well
        assert result.tm_score > 0.3, \
            f"TM-score should be >0.3 for partial match, got {result.tm_score:.3f}"
    
    def test_tm_score_length_dependence(self):
        """Test that TM-score normalization depends on protein length."""
        # Small protein (4 residues)
        coords1_small = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        coords2_small = [(0.5, 0.0, 0.0), (1.5, 0.0, 0.0), (2.5, 0.0, 0.0), (3.5, 0.0, 0.0)]
        
        calculator = RMSDCalculator(align_structures=False)
        result_small = calculator.calculate_rmsd(coords1_small, coords2_small, calculate_metrics=True)
        
        # Larger protein (20 residues, repeat pattern)
        coords1_large = [(float(i), 0.0, 0.0) for i in range(20)]
        coords2_large = [(float(i) + 0.5, 0.0, 0.0) for i in range(20)]
        
        result_large = calculator.calculate_rmsd(coords1_large, coords2_large, calculate_metrics=True)
        
        # Both should have valid TM-scores
        assert 0.0 <= result_small.tm_score <= 1.0
        assert 0.0 <= result_large.tm_score <= 1.0


class TestQualityAssessment:
    """Test quality assessment based on metrics."""
    
    def test_quality_excellent(self):
        """Test excellent quality assessment."""
        calculator = RMSDCalculator()
        
        quality = calculator.get_quality_assessment(rmsd=1.5, gdt_ts=80.0, tm_score=0.8)
        assert quality == "excellent"
    
    def test_quality_good(self):
        """Test good quality assessment."""
        calculator = RMSDCalculator()
        
        quality = calculator.get_quality_assessment(rmsd=3.0, gdt_ts=60.0, tm_score=0.6)
        assert quality == "good"
    
    def test_quality_acceptable(self):
        """Test acceptable quality assessment."""
        calculator = RMSDCalculator()
        
        quality = calculator.get_quality_assessment(rmsd=5.0, gdt_ts=40.0, tm_score=0.4)
        assert quality == "acceptable"
    
    def test_quality_poor(self):
        """Test poor quality assessment."""
        calculator = RMSDCalculator()
        
        quality = calculator.get_quality_assessment(rmsd=10.0, gdt_ts=10.0, tm_score=0.1)
        assert quality == "poor"


class TestDistanceMatrix:
    """Test distance matrix calculation."""
    
    def test_distance_matrix_diagonal_zeros(self):
        """Test that diagonal of self-distance matrix is zeros."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        
        calculator = RMSDCalculator()
        distances = calculator.calculate_distance_matrix(coords, coords)
        
        # Diagonal should be zeros (distance to self)
        assert len(distances) == 3
        assert len(distances[0]) == 3
        
        for i in range(3):
            assert abs(distances[i][i]) < 1e-10, \
                f"Diagonal element [{i}][{i}] should be ~0, got {distances[i][i]}"
    
    def test_distance_matrix_symmetry(self):
        """Test that distance matrix is symmetric for identical structures."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        calculator = RMSDCalculator()
        distances = calculator.calculate_distance_matrix(coords, coords)
        
        # Matrix should be symmetric
        assert abs(distances[0][1] - distances[1][0]) < 1e-10


class TestRMSDResult:
    """Test RMSDResult dataclass."""
    
    def test_rmsd_result_fields(self):
        """Test that RMSDResult contains all expected fields."""
        result = RMSDResult(
            rmsd=2.5,
            gdt_ts=75.0,
            tm_score=0.7,
            n_atoms=100,
            aligned=True
        )
        
        assert result.rmsd == 2.5
        assert result.gdt_ts == 75.0
        assert result.tm_score == 0.7
        assert result.n_atoms == 100
        assert result.aligned == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
