"""
Unit tests for consciousness coordinate system.

Tests consciousness state management, behavioral state derivation,
and coordinate updates based on exploration outcomes.
"""

import pytest
import time
from unittest.mock import Mock

from ubf_protein.consciousness import ConsciousnessState
from ubf_protein.behavioral_state import BehavioralState
from ubf_protein.models import ConsciousnessCoordinates, ConformationalOutcome, BehavioralStateData
from ubf_protein.config import (
    FREQUENCY_MIN, FREQUENCY_MAX, COHERENCE_MIN, COHERENCE_MAX,
    BEHAVIORAL_STATE_REGEN_THRESHOLD
)


class TestConsciousnessState:
    """Test ConsciousnessState class implementation"""

    def test_frequency_bounds_enforcement(self):
        """Test frequency stays within 3-15 Hz bounds"""
        # Valid frequency
        state = ConsciousnessState(frequency=7.5, coherence=0.7)
        assert state.get_frequency() == 7.5

        # Test lower bound
        with pytest.raises(ValueError, match="Frequency must be between"):
            ConsciousnessState(frequency=2.0, coherence=0.7)

        # Test upper bound
        with pytest.raises(ValueError, match="Frequency must be between"):
            ConsciousnessState(frequency=16.0, coherence=0.7)

    def test_coherence_bounds_enforcement(self):
        """Test coherence stays within 0.2-1.0 bounds"""
        # Valid coherence
        state = ConsciousnessState(frequency=7.5, coherence=0.7)
        assert state.get_coherence() == 0.7

        # Test lower bound
        with pytest.raises(ValueError, match="Coherence must be between"):
            ConsciousnessState(frequency=7.5, coherence=0.1)

        # Test upper bound
        with pytest.raises(ValueError, match="Coherence must be between"):
            ConsciousnessState(frequency=7.5, coherence=1.1)

    def test_initial_timestamp(self):
        """Test timestamp is set on initialization"""
        before = int(time.time() * 1000)
        state = ConsciousnessState(frequency=7.5, coherence=0.7)
        after = int(time.time() * 1000)

        coords = state.get_coordinates()
        assert before <= coords.last_update_timestamp <= after

    def test_update_from_large_energy_decrease(self):
        """Test consciousness updates from large energy decrease"""
        state = ConsciousnessState(frequency=7.5, coherence=0.7)

        # Create mock outcome with large energy decrease
        outcome = Mock(spec=ConformationalOutcome)
        outcome.energy_change = -150  # Large decrease
        outcome.success = True

        state.update_from_outcome(outcome)

        # Should increase frequency and coherence
        assert state.get_frequency() > 7.5
        assert state.get_coherence() > 0.7

        # Should stay within bounds
        assert FREQUENCY_MIN <= state.get_frequency() <= FREQUENCY_MAX
        assert COHERENCE_MIN <= state.get_coherence() <= COHERENCE_MAX

    def test_update_from_energy_increase(self):
        """Test consciousness updates from energy increase"""
        state = ConsciousnessState(frequency=7.5, coherence=0.7)

        # Create mock outcome with energy increase
        outcome = Mock(spec=ConformationalOutcome)
        outcome.energy_change = 75  # Increase
        outcome.success = False

        state.update_from_outcome(outcome)

        # Should decrease frequency and coherence
        assert state.get_frequency() < 7.5
        assert state.get_coherence() < 0.7

    def test_bounds_enforcement_on_update(self):
        """Test bounds are enforced during updates"""
        # Start near upper bound
        state = ConsciousnessState(frequency=14.5, coherence=0.95)

        outcome = Mock(spec=ConformationalOutcome)
        outcome.energy_change = -150  # Large increase
        outcome.success = True

        state.update_from_outcome(outcome)

        # Should not exceed maximum bounds
        assert state.get_frequency() <= FREQUENCY_MAX
        assert state.get_coherence() <= COHERENCE_MAX

    def test_timestamp_update_on_outcome(self):
        """Test timestamp is updated when processing outcomes"""
        state = ConsciousnessState(frequency=7.5, coherence=0.7)
        initial_timestamp = state.get_coordinates().last_update_timestamp

        time.sleep(0.001)  # Small delay

        outcome = Mock(spec=ConformationalOutcome)
        outcome.energy_change = -50
        outcome.success = True

        state.update_from_outcome(outcome)

        # Timestamp should be updated
        assert state.get_coordinates().last_update_timestamp > initial_timestamp


class TestBehavioralState:
    """Test BehavioralState class implementation"""

    def test_from_consciousness_static_method(self):
        """Test static method creates behavioral state from coordinates"""
        behavioral = BehavioralState.from_consciousness(frequency=8.0, coherence=0.8)

        assert isinstance(behavioral, BehavioralState)
        assert behavioral.get_exploration_energy() > 0
        assert behavioral.get_structural_focus() > 0

    def test_behavioral_dimensions(self):
        """Test all behavioral dimension getters return valid values"""
        behavioral = BehavioralState.from_consciousness(frequency=10.0, coherence=0.6)

        # All dimensions should be between 0 and 1
        assert 0.0 <= behavioral.get_exploration_energy() <= 1.0
        assert 0.0 <= behavioral.get_structural_focus() <= 1.0
        assert 0.0 <= behavioral.get_hydrophobic_drive() <= 1.0
        assert 0.0 <= behavioral.get_risk_tolerance() <= 1.0
        assert 0.0 <= behavioral.get_native_state_ambition() <= 1.0

    def test_high_frequency_behavioral_state(self):
        """Test behavioral state for high frequency (aggressive agent)"""
        behavioral = BehavioralState.from_consciousness(frequency=12.0, coherence=0.4)

        # High frequency should mean high exploration energy and risk tolerance
        assert behavioral.get_exploration_energy() > 0.7  # High exploration
        assert behavioral.get_risk_tolerance() > 0.5     # High risk tolerance
        assert behavioral.get_hydrophobic_drive() > 0.5  # High hydrophobic drive

    def test_high_coherence_behavioral_state(self):
        """Test behavioral state for high coherence (cautious agent)"""
        behavioral = BehavioralState.from_consciousness(frequency=6.0, coherence=0.9)

        # High coherence should mean high structural focus and ambition
        assert behavioral.get_structural_focus() > 0.8   # High focus
        assert behavioral.get_native_state_ambition() > 0.5  # High ambition

    def test_should_regenerate_threshold(self):
        """Test behavioral state regeneration threshold (0.3)"""
        behavioral = BehavioralState.from_consciousness(frequency=7.5, coherence=0.7)

        # Small change should not trigger regeneration
        assert not behavioral.should_regenerate(0.2)

        # Threshold change should trigger regeneration
        assert behavioral.should_regenerate(0.3)

        # Large change should trigger regeneration
        assert behavioral.should_regenerate(0.5)

    def test_regenerate_if_needed(self):
        """Test conditional regeneration based on coordinate changes"""
        initial_coords = ConsciousnessCoordinates(
            frequency=7.5, coherence=0.7, last_update_timestamp=1000
        )
        behavioral = BehavioralState(initial_coords)

        # Small change - should not regenerate
        small_change_coords = ConsciousnessCoordinates(
            frequency=7.6, coherence=0.75, last_update_timestamp=1001
        )
        result = behavioral.regenerate_if_needed(small_change_coords)
        assert result is None

        # Large change - should regenerate
        large_change_coords = ConsciousnessCoordinates(
            frequency=8.5, coherence=0.8, last_update_timestamp=1002
        )
        result = behavioral.regenerate_if_needed(large_change_coords)
        assert result is not None
        assert isinstance(result, BehavioralState)

    def test_caching_behavior(self):
        """Test behavioral state caching mechanism"""
        behavioral = BehavioralState.from_consciousness(frequency=7.5, coherence=0.7)

        # Should have cached timestamp
        assert behavioral.get_cached_timestamp() > 0

        # Multiple calls should return same values (cached)
        energy1 = behavioral.get_exploration_energy()
        energy2 = behavioral.get_exploration_energy()
        assert energy1 == energy2

    def test_behavioral_state_data_access(self):
        """Test access to underlying behavioral state data"""
        behavioral = BehavioralState.from_consciousness(frequency=8.0, coherence=0.6)

        data = behavioral.get_behavioral_data()
        assert isinstance(data, BehavioralStateData)
        assert data.exploration_energy == behavioral.get_exploration_energy()
        assert data.structural_focus == behavioral.get_structural_focus()


class TestConsciousnessBehavioralIntegration:
    """Test integration between consciousness and behavioral states"""

    def test_behavioral_state_regeneration_after_coordinate_change(self):
        """Test behavioral state regenerates when coordinates change significantly"""
        # Create consciousness state
        consciousness = ConsciousnessState(frequency=7.5, coherence=0.7)

        # Create initial behavioral state
        behavioral = BehavioralState(consciousness.get_coordinates())

        # Small coordinate change - behavioral state should remain valid
        consciousness._coordinates.frequency = 7.6  # Change of 0.1
        regenerated = behavioral.regenerate_if_needed(consciousness.get_coordinates())
        assert regenerated is None  # No regeneration needed

        # Large coordinate change - behavioral state should regenerate
        consciousness._coordinates.frequency = 8.5  # Change of 1.0 > 0.3 threshold
        regenerated = behavioral.regenerate_if_needed(consciousness.get_coordinates())
        assert regenerated is not None  # Regeneration occurred

        # New behavioral state should reflect new coordinates
        assert regenerated.get_exploration_energy() > behavioral.get_exploration_energy()

    def test_coordinate_bounds_preserve_behavioral_validity(self):
        """Test that coordinate bounds preserve behavioral state validity"""
        # Test at bounds
        min_freq_state = BehavioralState.from_consciousness(FREQUENCY_MIN, COHERENCE_MAX)
        max_freq_state = BehavioralState.from_consciousness(FREQUENCY_MAX, COHERENCE_MIN)

        # All behavioral dimensions should be valid (0-1 range)
        for behavioral in [min_freq_state, max_freq_state]:
            assert 0.0 <= behavioral.get_exploration_energy() <= 1.0
            assert 0.0 <= behavioral.get_structural_focus() <= 1.0
            assert 0.0 <= behavioral.get_hydrophobic_drive() <= 1.0
            assert 0.0 <= behavioral.get_risk_tolerance() <= 1.0
            assert 0.0 <= behavioral.get_native_state_ambition() <= 1.0