"""
Unit tests for dynamic parameter adjustment.

Tests the DynamicParameterAdjuster class that adjusts exploration
parameters (frequency and temperature) based on QCPP stability metrics.
"""

import pytest

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ubf_protein.dynamic_adjustment import DynamicParameterAdjuster
from ubf_protein.qcpp_integration import QCPPMetrics


class TestDynamicParameterAdjusterInitialization:
    """Test suite for DynamicParameterAdjuster initialization."""
    
    def test_initialization(self):
        """Test adjuster initializes with correct defaults."""
        adjuster = DynamicParameterAdjuster()
        
        assert adjuster._adjustment_count == 0
        assert adjuster._total_frequency_change == 0.0
        assert adjuster._total_temperature_change == 0.0
    
    def test_constants_defined(self):
        """Test that adjustment constants are correctly defined."""
        adjuster = DynamicParameterAdjuster()
        
        # Thresholds
        assert adjuster.LOW_STABILITY_THRESHOLD == 1.0
        assert adjuster.HIGH_STABILITY_THRESHOLD == 2.0
        
        # Adjustments
        assert adjuster.FREQUENCY_INCREASE_UNSTABLE == 2.0
        assert adjuster.FREQUENCY_DECREASE_STABLE == 1.0
        assert adjuster.TEMPERATURE_INCREASE_UNSTABLE == 50.0
        assert adjuster.TEMPERATURE_DECREASE_STABLE == 20.0
        
        # Bounds
        assert adjuster.MIN_FREQUENCY == 3.0
        assert adjuster.MAX_FREQUENCY == 15.0
        assert adjuster.MIN_TEMPERATURE == 100.0
        assert adjuster.MAX_TEMPERATURE == 500.0


class TestUnstableRegionAdjustments:
    """Test suite for adjustments in unstable regions (stability < 1.0)."""
    
    def test_increase_parameters_low_stability(self):
        """Test parameters increase when stability < 1.0."""
        adjuster = DynamicParameterAdjuster()
        
        current_freq = 8.0
        current_temp = 300.0
        stability = 0.5  # Low stability
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        # Should increase exploration
        assert new_freq == current_freq + 2.0  # +2.0 Hz
        assert new_temp == current_temp + 50.0  # +50 K
    
    def test_increase_parameters_very_low_stability(self):
        """Test parameters increase with very low stability."""
        adjuster = DynamicParameterAdjuster()
        
        current_freq = 5.0
        current_temp = 200.0
        stability = 0.1  # Very low stability
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        assert new_freq == 7.0  # 5 + 2
        assert new_temp == 250.0  # 200 + 50
    
    def test_increase_respects_upper_bounds(self):
        """Test that increases respect upper bounds."""
        adjuster = DynamicParameterAdjuster()
        
        # Start near upper bounds
        current_freq = 14.0  # Near max of 15
        current_temp = 480.0  # Near max of 500
        stability = 0.3  # Low stability (would increase)
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        # Should be clamped to maximum
        assert new_freq == 15.0  # Clamped (14 + 2 = 16 -> 15)
        assert new_temp == 500.0  # Clamped (480 + 50 = 530 -> 500)
    
    def test_statistics_tracked_for_increases(self):
        """Test that adjustment statistics are tracked."""
        adjuster = DynamicParameterAdjuster()
        
        adjuster.adjust_from_stability(8.0, 300.0, 0.5)
        
        stats = adjuster.get_adjustment_statistics()
        assert stats['adjustment_count'] == 1
        assert stats['total_frequency_change'] == 2.0
        assert stats['total_temperature_change'] == 50.0


class TestStableRegionAdjustments:
    """Test suite for adjustments in stable regions (stability > 2.0)."""
    
    def test_decrease_parameters_high_stability(self):
        """Test parameters decrease when stability > 2.0."""
        adjuster = DynamicParameterAdjuster()
        
        current_freq = 10.0
        current_temp = 350.0
        stability = 2.5  # High stability
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        # Should increase exploitation
        assert new_freq == current_freq - 1.0  # -1.0 Hz
        assert new_temp == current_temp - 20.0  # -20 K
    
    def test_decrease_parameters_very_high_stability(self):
        """Test parameters decrease with very high stability."""
        adjuster = DynamicParameterAdjuster()
        
        current_freq = 12.0
        current_temp = 400.0
        stability = 3.0  # Very high stability
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        assert new_freq == 11.0  # 12 - 1
        assert new_temp == 380.0  # 400 - 20
    
    def test_decrease_respects_lower_bounds(self):
        """Test that decreases respect lower bounds."""
        adjuster = DynamicParameterAdjuster()
        
        # Start near lower bounds
        current_freq = 3.5  # Near min of 3
        current_temp = 110.0  # Near min of 100
        stability = 2.8  # High stability (would decrease)
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        # Should be clamped to minimum
        assert new_freq == 3.0  # Clamped (3.5 - 1 = 2.5 -> 3)
        assert new_temp == 100.0  # Clamped (110 - 20 = 90 -> 100)
    
    def test_statistics_tracked_for_decreases(self):
        """Test that negative adjustments are tracked correctly."""
        adjuster = DynamicParameterAdjuster()
        
        adjuster.adjust_from_stability(10.0, 350.0, 2.5)
        
        stats = adjuster.get_adjustment_statistics()
        assert stats['adjustment_count'] == 1
        assert stats['total_frequency_change'] == -1.0
        assert stats['total_temperature_change'] == -20.0


class TestNormalStabilityNoAdjustment:
    """Test suite for normal stability (1.0 <= stability <= 2.0)."""
    
    def test_no_adjustment_at_low_threshold(self):
        """Test no adjustment at low threshold boundary."""
        adjuster = DynamicParameterAdjuster()
        
        current_freq = 9.0
        current_temp = 300.0
        stability = 1.0  # At low threshold
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        # Should remain unchanged
        assert new_freq == current_freq
        assert new_temp == current_temp
    
    def test_no_adjustment_at_high_threshold(self):
        """Test no adjustment at high threshold boundary."""
        adjuster = DynamicParameterAdjuster()
        
        current_freq = 9.0
        current_temp = 300.0
        stability = 2.0  # At high threshold
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        # Should remain unchanged
        assert new_freq == current_freq
        assert new_temp == current_temp
    
    def test_no_adjustment_mid_range(self):
        """Test no adjustment in mid-stability range."""
        adjuster = DynamicParameterAdjuster()
        
        current_freq = 8.5
        current_temp = 280.0
        stability = 1.5  # Mid-range
        
        new_freq, new_temp = adjuster.adjust_from_stability(
            current_freq,
            current_temp,
            stability
        )
        
        assert new_freq == current_freq
        assert new_temp == current_temp
    
    def test_statistics_not_tracked_for_no_adjustment(self):
        """Test that statistics are not updated when no adjustment occurs."""
        adjuster = DynamicParameterAdjuster()
        
        adjuster.adjust_from_stability(9.0, 300.0, 1.5)
        
        stats = adjuster.get_adjustment_statistics()
        assert stats['adjustment_count'] == 0
        assert stats['total_frequency_change'] == 0.0
        assert stats['total_temperature_change'] == 0.0


class TestBoundsEnforcement:
    """Test suite for parameter bounds enforcement."""
    
    def test_frequency_lower_bound_enforcement(self):
        """Test frequency cannot go below 3 Hz."""
        adjuster = DynamicParameterAdjuster()
        
        # Try to decrease below minimum
        current_freq = 3.0
        stability = 2.5  # Would decrease by 1 Hz
        
        new_freq, _ = adjuster.adjust_from_stability(current_freq, 300.0, stability)
        
        assert new_freq == 3.0  # Clamped to minimum
    
    def test_frequency_upper_bound_enforcement(self):
        """Test frequency cannot go above 15 Hz."""
        adjuster = DynamicParameterAdjuster()
        
        # Try to increase above maximum
        current_freq = 15.0
        stability = 0.5  # Would increase by 2 Hz
        
        new_freq, _ = adjuster.adjust_from_stability(current_freq, 300.0, stability)
        
        assert new_freq == 15.0  # Clamped to maximum
    
    def test_temperature_lower_bound_enforcement(self):
        """Test temperature cannot go below 100 K."""
        adjuster = DynamicParameterAdjuster()
        
        # Try to decrease below minimum
        current_temp = 100.0
        stability = 2.5  # Would decrease by 20 K
        
        _, new_temp = adjuster.adjust_from_stability(9.0, current_temp, stability)
        
        assert new_temp == 100.0  # Clamped to minimum
    
    def test_temperature_upper_bound_enforcement(self):
        """Test temperature cannot go above 500 K."""
        adjuster = DynamicParameterAdjuster()
        
        # Try to increase above maximum
        current_temp = 500.0
        stability = 0.5  # Would increase by 50 K
        
        _, new_temp = adjuster.adjust_from_stability(9.0, current_temp, stability)
        
        assert new_temp == 500.0  # Clamped to maximum
    
    def test_invalid_input_frequency_raises_error(self):
        """Test that invalid input frequency raises ValueError."""
        adjuster = DynamicParameterAdjuster()
        
        with pytest.raises(ValueError, match="outside valid range"):
            adjuster.adjust_from_stability(2.0, 300.0, 1.5)  # Below min
        
        with pytest.raises(ValueError, match="outside valid range"):
            adjuster.adjust_from_stability(16.0, 300.0, 1.5)  # Above max
    
    def test_invalid_input_temperature_raises_error(self):
        """Test that invalid input temperature raises ValueError."""
        adjuster = DynamicParameterAdjuster()
        
        with pytest.raises(ValueError, match="outside valid range"):
            adjuster.adjust_from_stability(9.0, 50.0, 1.5)  # Below min
        
        with pytest.raises(ValueError, match="outside valid range"):
            adjuster.adjust_from_stability(9.0, 600.0, 1.5)  # Above max


class TestQCPPMetricsIntegration:
    """Test suite for integration with QCPPMetrics."""
    
    def test_adjust_from_qcpp_metrics_low_stability(self):
        """Test adjustment from QCPPMetrics with low stability."""
        adjuster = DynamicParameterAdjuster()
        
        qcpp_metrics = QCPPMetrics(
            qcp_score=4.0,
            field_coherence=0.3,
            stability_score=0.7,  # Low stability
            phi_match_score=0.5,
            calculation_time_ms=1.2
        )
        
        new_freq, new_temp = adjuster.adjust_from_qcpp_metrics(
            current_frequency=8.0,
            current_temperature=300.0,
            qcpp_metrics=qcpp_metrics
        )
        
        # Should increase exploration
        assert new_freq == 10.0  # 8 + 2
        assert new_temp == 350.0  # 300 + 50
    
    def test_adjust_from_qcpp_metrics_high_stability(self):
        """Test adjustment from QCPPMetrics with high stability."""
        adjuster = DynamicParameterAdjuster()
        
        qcpp_metrics = QCPPMetrics(
            qcp_score=6.5,
            field_coherence=0.9,
            stability_score=2.3,  # High stability
            phi_match_score=0.85,
            calculation_time_ms=1.5
        )
        
        new_freq, new_temp = adjuster.adjust_from_qcpp_metrics(
            current_frequency=10.0,
            current_temperature=350.0,
            qcpp_metrics=qcpp_metrics
        )
        
        # Should increase exploitation
        assert new_freq == 9.0  # 10 - 1
        assert new_temp == 330.0  # 350 - 20
    
    def test_adjust_from_qcpp_metrics_normal_stability(self):
        """Test adjustment from QCPPMetrics with normal stability."""
        adjuster = DynamicParameterAdjuster()
        
        qcpp_metrics = QCPPMetrics(
            qcp_score=5.2,
            field_coherence=0.6,
            stability_score=1.5,  # Normal stability
            phi_match_score=0.7,
            calculation_time_ms=1.3
        )
        
        new_freq, new_temp = adjuster.adjust_from_qcpp_metrics(
            current_frequency=9.0,
            current_temperature=300.0,
            qcpp_metrics=qcpp_metrics
        )
        
        # No adjustment
        assert new_freq == 9.0
        assert new_temp == 300.0
    
    def test_adjust_from_invalid_qcpp_metrics(self):
        """Test that invalid QCPP metrics raises ValueError."""
        adjuster = DynamicParameterAdjuster()
        
        class InvalidMetrics:
            pass
        
        invalid = InvalidMetrics()
        
        with pytest.raises(ValueError, match="must have stability_score"):
            adjuster.adjust_from_qcpp_metrics(9.0, 300.0, invalid)


class TestAdjustmentStatistics:
    """Test suite for adjustment statistics tracking."""
    
    def test_statistics_after_multiple_adjustments(self):
        """Test statistics after multiple adjustments."""
        adjuster = DynamicParameterAdjuster()
        
        # Make several adjustments
        adjuster.adjust_from_stability(8.0, 300.0, 0.5)  # +2, +50
        adjuster.adjust_from_stability(10.0, 350.0, 2.5)  # -1, -20
        adjuster.adjust_from_stability(9.0, 330.0, 0.8)  # +2, +50
        
        stats = adjuster.get_adjustment_statistics()
        
        assert stats['adjustment_count'] == 3
        assert stats['total_frequency_change'] == 3.0  # 2 - 1 + 2
        assert stats['total_temperature_change'] == 80.0  # 50 - 20 + 50
        assert abs(stats['avg_frequency_change'] - 1.0) < 0.01  # 3 / 3
        assert abs(stats['avg_temperature_change'] - 26.67) < 0.01  # 80 / 3
    
    def test_statistics_reset(self):
        """Test that statistics can be reset."""
        adjuster = DynamicParameterAdjuster()
        
        # Make some adjustments
        adjuster.adjust_from_stability(8.0, 300.0, 0.5)
        adjuster.adjust_from_stability(10.0, 350.0, 2.5)
        
        # Reset
        adjuster.reset_statistics()
        
        stats = adjuster.get_adjustment_statistics()
        assert stats['adjustment_count'] == 0
        assert stats['total_frequency_change'] == 0.0
        assert stats['total_temperature_change'] == 0.0
        assert stats['avg_frequency_change'] == 0.0
        assert stats['avg_temperature_change'] == 0.0
    
    def test_statistics_with_no_adjustments(self):
        """Test statistics when no adjustments have been made."""
        adjuster = DynamicParameterAdjuster()
        
        stats = adjuster.get_adjustment_statistics()
        
        assert stats['adjustment_count'] == 0
        assert stats['total_frequency_change'] == 0.0
        assert stats['total_temperature_change'] == 0.0
        assert stats['avg_frequency_change'] == 0.0
        assert stats['avg_temperature_change'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
