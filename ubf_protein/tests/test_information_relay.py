"""
Unit tests for Information Relay System (Task 9).

Tests for:
- QCPP relay functionality
- Broadcast throttling mechanism
- Pattern broadcasting to shared memory
- Pattern consumption by exploration agents
- Pattern age filtering

Author: UBF Protein System
Date: November 9, 2025
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from ubf_protein.mediator_agent import MediatorAgent, BroadcastThrottler
from ubf_protein.mediator_config import MediatorConfig
from ubf_protein.pattern_detection import (
    PatternDetection, PatternType, PatternSignificance,
    THzResonanceData, FoldingDynamicsData, GeometricSimilarityData
)
from ubf_protein.models import Conformation
from ubf_protein.memory_system import SharedMemoryPool
from ubf_protein.geometric_attractor import GeometricAttractorAnalyzer


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_qcpp_adapter():
    """Create mock QCPP adapter for testing."""
    adapter = Mock()
    adapter.analyze_conformation = Mock(return_value=Mock(
        qcp_score=4.5,
        field_coherence=0.85,
        stability_prediction='stable',
        golden_ratio_score=0.72
    ))
    adapter.predictor = Mock()
    return adapter


@pytest.fixture
def geometric_analyzer():
    """Create real geometric analyzer for testing."""
    return GeometricAttractorAnalyzer()


@pytest.fixture
def shared_memory():
    """Create shared memory pool for testing."""
    return SharedMemoryPool()


@pytest.fixture
def mediator_config():
    """Create test mediator configuration."""
    return MediatorConfig(
        relay_frequency=10,
        broadcast_throttle_rate=5,
        cache_size=100
    )


@pytest.fixture
def mediator_agent(mock_qcpp_adapter, geometric_analyzer, shared_memory, mediator_config):
    """Create mediator agent for testing."""
    return MediatorAgent(
        protein_sequence="ACDEFGH",
        qcpp_adapter=mock_qcpp_adapter,
        geometric_analyzer=geometric_analyzer,
        shared_memory=shared_memory,
        config=mediator_config
    )


@pytest.fixture
def sample_thz_pattern():
    """Create sample THz resonance pattern."""
    return PatternDetection(
        pattern_type=PatternType.THZ,
        significance=PatternSignificance.HIGH,
        timestamp=time.time(),
        iteration=100,
        conformation_hash="a1b2c3d4e5f6g7h8",
        thz_data=THzResonanceData(
            cluster_id=3,
            cluster_size=12,
            similarity_score=0.85,
            dominant_frequency=2.45,
            spectral_entropy=1.23
        )
    )


@pytest.fixture
def sample_folding_pattern():
    """Create sample folding dynamics pattern."""
    return PatternDetection(
        pattern_type=PatternType.FOLDING,
        significance=PatternSignificance.MEDIUM,
        timestamp=time.time(),
        iteration=150,
        conformation_hash="b2c3d4e5f6g7h8i9",
        folding_data=FoldingDynamicsData(
            helix_percentage=35.2,
            sheet_percentage=22.1,
            turn_percentage=12.5,
            coil_percentage=30.2,
            helix_regions=[(5, 18), (25, 38)],
            sheet_regions=[(42, 48)],
            turn_regions=[(19, 22)]
        )
    )


@pytest.fixture
def sample_geometric_pattern():
    """Create sample geometric similarity pattern."""
    return PatternDetection(
        pattern_type=PatternType.GEOMETRIC,
        significance=PatternSignificance.HIGH,
        timestamp=time.time(),
        iteration=200,
        conformation_hash="c3d4e5f6g7h8i9j0",
        geometric_data=GeometricSimilarityData(
            rmsd_to_reference=1.85,
            overlap_percentage=78.5,
            reference_conformation_hash="d4e5f6g7h8i9j0k1",
            golden_ratio_percentage=24.3,
            dominant_platonic_solid="icosahedron",
            platonic_similarity_score=0.82
        )
    )


@pytest.fixture
def sample_conformation():
    """Create sample conformation for testing."""
    coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)]
    return Conformation(
        conformation_id="test_conf_001",
        atom_coordinates=coords,
        energy=-100.0,
        rmsd_to_native=None,
        sequence="ACE",
        secondary_structure=['C', 'C', 'C'],
        phi_angles=[-60.0, -60.0, -60.0],
        psi_angles=[-45.0, -45.0, -45.0],
        available_move_types=[],
        structural_constraints={}
    )


# ============================================================================
# Test BroadcastThrottler
# ============================================================================

def test_throttler_initialization():
    """Test BroadcastThrottler initialization."""
    throttler = BroadcastThrottler(max_broadcasts_per_window=10)
    
    assert throttler.max_broadcasts_per_window == 10
    assert throttler.window_duration == 1.0
    assert throttler.high_priority_bypass is True
    assert len(throttler.broadcast_times) == 0


def test_throttler_allows_broadcasts_under_limit():
    """Test throttler allows broadcasts when under limit."""
    throttler = BroadcastThrottler(max_broadcasts_per_window=5)
    
    # Should allow first 5 broadcasts
    for i in range(5):
        assert throttler.can_broadcast() is True
        throttler.record_broadcast()
    
    # Should throttle 6th broadcast
    assert throttler.can_broadcast() is False


def test_throttler_high_priority_bypass():
    """Test high priority patterns bypass throttle."""
    throttler = BroadcastThrottler(max_broadcasts_per_window=2)
    
    # Fill throttle with 2 broadcasts
    throttler.record_broadcast()
    throttler.record_broadcast()
    
    # Normal priority should be throttled
    assert throttler.can_broadcast() is False
    
    # High priority should bypass
    assert throttler.can_broadcast(priority='high') is True


def test_throttler_sliding_window_expiration():
    """Test throttler expires old timestamps."""
    throttler = BroadcastThrottler(
        max_broadcasts_per_window=2,
        window_duration=0.1  # 100ms window
    )
    
    # Fill throttle
    throttler.record_broadcast()
    throttler.record_broadcast()
    assert throttler.can_broadcast() is False
    
    # Wait for window to expire
    time.sleep(0.15)
    
    # Should allow broadcasts again
    assert throttler.can_broadcast() is True


def test_throttler_prioritize_message():
    """Test throttler extracts priority from pattern."""
    throttler = BroadcastThrottler()
    
    high_pattern = Mock(significance=PatternSignificance.HIGH)
    medium_pattern = Mock(significance=PatternSignificance.MEDIUM)
    low_pattern = Mock(significance=PatternSignificance.LOW)
    
    assert throttler.prioritize_message(high_pattern) == 'high'
    assert throttler.prioritize_message(medium_pattern) == 'medium'
    assert throttler.prioritize_message(low_pattern) == 'low'


def test_throttler_get_current_rate():
    """Test throttler reports current broadcast rate."""
    throttler = BroadcastThrottler(window_duration=1.0)
    
    # No broadcasts
    assert throttler.get_current_rate() == 0.0
    
    # Add some broadcasts
    throttler.record_broadcast()
    throttler.record_broadcast()
    throttler.record_broadcast()
    
    # Should be 3 broadcasts per second
    assert throttler.get_current_rate() == 3.0


def test_throttler_reset():
    """Test throttler reset clears state."""
    throttler = BroadcastThrottler()
    
    # Add broadcasts
    throttler.record_broadcast()
    throttler.record_broadcast()
    assert len(throttler.broadcast_times) == 2
    
    # Reset
    throttler.reset()
    assert len(throttler.broadcast_times) == 0


# ============================================================================
# Test QCPP Relay
# ============================================================================

def test_relay_to_qcpp_success(mediator_agent, sample_thz_pattern, sample_conformation):
    """Test successful QCPP relay."""
    qcpp_metrics = mediator_agent.relay_to_qcpp(sample_thz_pattern, sample_conformation)
    
    assert qcpp_metrics is not None
    assert 'qcp_score' in qcpp_metrics
    assert 'field_coherence' in qcpp_metrics
    assert 'stability_prediction' in qcpp_metrics
    assert 'golden_ratio_score' in qcpp_metrics
    
    # Verify statistics updated
    assert mediator_agent.detection_statistics['qcpp_validations'] == 1


def test_relay_to_qcpp_handles_failure(mediator_agent, sample_thz_pattern, sample_conformation):
    """Test QCPP relay handles failures gracefully."""
    # Make QCPP adapter raise exception
    mediator_agent.qcpp_adapter.analyze_conformation.side_effect = Exception("QCPP failed")
    
    qcpp_metrics = mediator_agent.relay_to_qcpp(sample_thz_pattern, sample_conformation)
    
    # Should return None on failure
    assert qcpp_metrics is None
    
    # Statistics should still be updated
    assert mediator_agent.detection_statistics['qcpp_validations'] == 1


def test_relay_to_qcpp_no_adapter(sample_thz_pattern, sample_conformation):
    """Test QCPP relay handles missing adapter."""
    mediator = MediatorAgent(
        protein_sequence="ACDEFGH",
        qcpp_adapter=None,  # No adapter
        geometric_analyzer=GeometricAttractorAnalyzer(),
        shared_memory=SharedMemoryPool(),
        config=MediatorConfig()
    )
    
    qcpp_metrics = mediator.relay_to_qcpp(sample_thz_pattern, sample_conformation)
    
    # Should return None when no adapter
    assert qcpp_metrics is None


# ============================================================================
# Test Pattern Broadcasting
# ============================================================================

def test_broadcast_to_agents_success(mediator_agent, sample_thz_pattern):
    """Test successful pattern broadcast."""
    qcpp_metrics = {'qcp_score': 4.5, 'coherence': 0.85}
    
    success = mediator_agent.broadcast_to_agents(sample_thz_pattern, qcpp_metrics)
    
    assert success is True
    assert mediator_agent.detection_statistics['broadcasts'] == 1
    
    # Verify pattern stored in shared memory
    patterns = mediator_agent.shared_memory.pattern_broadcasts
    assert len(patterns) == 1
    assert patterns[0]['pattern_type'] == 'thz_resonance'
    assert patterns[0]['significance'] == 'high'
    assert 'qcpp_metrics' in patterns[0]


def test_broadcast_throttling(mediator_agent):
    """Test broadcast throttling prevents overflow."""
    # Mediator config has throttle_rate=5
    # Create LOW priority pattern (won't bypass throttle)
    low_pattern = PatternDetection(
        pattern_type=PatternType.THZ,
        significance=PatternSignificance.LOW,  # LOW priority - no bypass
        timestamp=time.time(),
        iteration=100,
        conformation_hash="test12345678hash",
        thz_data=THzResonanceData(1, 2, 0.5, 1.0, 0.5)
    )
    
    # Fill throttle
    for i in range(5):
        success = mediator_agent.broadcast_to_agents(low_pattern)
        assert success is True
    
    # Next broadcast should be throttled
    success = mediator_agent.broadcast_to_agents(low_pattern)
    assert success is False


def test_broadcast_high_priority_bypass(mediator_agent, sample_thz_pattern):
    """Test high priority patterns bypass throttle."""
    # Fill throttle with 5 broadcasts
    low_pattern = PatternDetection(
        pattern_type=PatternType.THZ,
        significance=PatternSignificance.LOW,
        timestamp=time.time(),
        iteration=100,
        conformation_hash="test12345678hash",
        thz_data=THzResonanceData(1, 2, 0.5, 1.0, 0.5)
    )
    
    for i in range(5):
        mediator_agent.broadcast_to_agents(low_pattern)
    
    # High priority should still broadcast
    success = mediator_agent.broadcast_to_agents(sample_thz_pattern)
    assert success is True  # HIGH priority bypasses


def test_broadcast_all_pattern_types(mediator_agent, sample_thz_pattern, 
                                    sample_folding_pattern, sample_geometric_pattern):
    """Test broadcasting all pattern types."""
    # Broadcast THz pattern
    mediator_agent.broadcast_to_agents(sample_thz_pattern)
    
    # Broadcast folding pattern
    mediator_agent.broadcast_to_agents(sample_folding_pattern)
    
    # Broadcast geometric pattern
    mediator_agent.broadcast_to_agents(sample_geometric_pattern)
    
    # Verify all stored
    patterns = mediator_agent.shared_memory.pattern_broadcasts
    assert len(patterns) == 3
    
    pattern_types = [p['pattern_type'] for p in patterns]
    assert 'thz_resonance' in pattern_types
    assert 'folding_dynamics' in pattern_types
    assert 'geometric_similarity' in pattern_types


# ============================================================================
# Test Pattern Retrieval
# ============================================================================

def test_retrieve_recent_patterns(shared_memory):
    """Test retrieving recent patterns from shared memory."""
    # Add patterns at different iterations
    pattern1 = {'iteration': 100, 'pattern_type': 'thz'}
    pattern2 = {'iteration': 150, 'pattern_type': 'folding'}
    pattern3 = {'iteration': 200, 'pattern_type': 'geometric'}
    pattern4 = {'iteration': 50, 'pattern_type': 'old'}  # Old pattern
    
    shared_memory.broadcast_pattern(pattern1)
    shared_memory.broadcast_pattern(pattern2)
    shared_memory.broadcast_pattern(pattern3)
    shared_memory.broadcast_pattern(pattern4)
    
    # Retrieve patterns from iteration 200 with max_age=100
    recent = shared_memory.retrieve_recent_patterns(
        current_iteration=200,
        max_age=100
    )
    
    # Should get patterns from iterations 100-200
    assert len(recent) == 3
    iterations = [p['iteration'] for p in recent]
    assert 50 not in iterations  # Old pattern excluded
    assert 100 in iterations
    assert 150 in iterations
    assert 200 in iterations


def test_pattern_age_filtering(shared_memory):
    """Test pattern age filtering works correctly."""
    # Add pattern at iteration 50
    old_pattern = {'iteration': 50, 'pattern_type': 'old'}
    shared_memory.broadcast_pattern(old_pattern)
    
    # Retrieve from iteration 200 with max_age=100
    recent = shared_memory.retrieve_recent_patterns(
        current_iteration=200,
        max_age=100
    )
    
    # Old pattern should be filtered out
    assert len(recent) == 0


def test_get_pattern_count(shared_memory):
    """Test getting total pattern count."""
    assert shared_memory.get_pattern_count() == 0
    
    shared_memory.broadcast_pattern({'iteration': 1})
    shared_memory.broadcast_pattern({'iteration': 2})
    shared_memory.broadcast_pattern({'iteration': 3})
    
    assert shared_memory.get_pattern_count() == 3


# ============================================================================
# Test Pattern Consumption by Agents
# ============================================================================

def test_pattern_guidance_no_patterns():
    """Test pattern guidance returns neutral when no patterns."""
    from ubf_protein.protein_agent import ProteinAgent
    from ubf_protein.models import Conformation
    from ubf_protein.interfaces import MoveType
    from ubf_protein.models import ConformationalMove
    
    # Create minimal agent
    agent = ProteinAgent(protein_sequence="ACE")
    
    # Create mock move
    move = Mock()
    move.move_type = MoveType.BACKBONE_ROTATION
    
    # Get guidance with no shared memory
    guidance = agent._get_pattern_guidance(move, None)
    
    assert guidance == 1.0  # Neutral


def test_pattern_guidance_geometric_bonus():
    """Test pattern guidance applies geometric bonuses."""
    from ubf_protein.protein_agent import ProteinAgent
    from ubf_protein.interfaces import MoveType
    
    agent = ProteinAgent(protein_sequence="ACE")
    agent._iterations_completed = 200
    
    # Create shared memory with geometric pattern
    shared_mem = SharedMemoryPool()
    shared_mem.broadcast_pattern({
        'iteration': 190,
        'pattern_type': 'geometric_similarity',
        'significance': 'high',
        'geometric_data': {
            'golden_ratio_percentage': 25.0,
            'dominant_platonic_solid': 'icosahedron'
        }
    })
    
    # Create mock move
    move = Mock()
    move.move_type = MoveType.BACKBONE_ROTATION
    
    # Get guidance
    guidance = agent._get_pattern_guidance(move, shared_mem)
    
    # Should have geometric bonus (high significance = 1.20, plus icosahedron bonus)
    assert guidance > 1.0


def test_pattern_guidance_thz_bonus():
    """Test pattern guidance applies THz bonuses."""
    from ubf_protein.protein_agent import ProteinAgent
    from ubf_protein.interfaces import MoveType
    
    agent = ProteinAgent(protein_sequence="ACE")
    agent._iterations_completed = 200
    
    # Create shared memory with THz pattern
    shared_mem = SharedMemoryPool()
    shared_mem.broadcast_pattern({
        'iteration': 195,
        'pattern_type': 'thz_resonance',
        'significance': 'high',
        'thz_data': {
            'cluster_id': 1,
            'cluster_size': 10
        }
    })
    
    # Create backbone rotation move (should get bonus)
    move = Mock()
    move.move_type = MoveType.BACKBONE_ROTATION
    
    guidance = agent._get_pattern_guidance(move, shared_mem)
    assert guidance > 1.0
    
    # Create large jump move (should get penalty)
    move_jump = Mock()
    move_jump.move_type = MoveType.LARGE_CONFORMATIONAL_JUMP
    
    guidance_jump = agent._get_pattern_guidance(move_jump, shared_mem)
    assert guidance_jump < 1.0


def test_pattern_guidance_folding_bonus():
    """Test pattern guidance applies folding bonuses."""
    from ubf_protein.protein_agent import ProteinAgent
    from ubf_protein.interfaces import MoveType
    
    agent = ProteinAgent(protein_sequence="ACE")
    agent._iterations_completed = 200
    
    # Create shared memory with folding pattern (high helix)
    shared_mem = SharedMemoryPool()
    shared_mem.broadcast_pattern({
        'iteration': 198,
        'pattern_type': 'folding_dynamics',
        'significance': 'medium',
        'folding_data': {
            'helix_percentage': 35.0,
            'sheet_percentage': 10.0
        }
    })
    
    # Helix formation move should get bonus
    move = Mock()
    move.move_type = MoveType.HELIX_FORMATION
    
    guidance = agent._get_pattern_guidance(move, shared_mem)
    assert guidance > 1.0


def test_pattern_guidance_clamping():
    """Test pattern guidance is clamped to [0.8, 1.5]."""
    from ubf_protein.protein_agent import ProteinAgent
    from ubf_protein.interfaces import MoveType
    
    agent = ProteinAgent(protein_sequence="ACE")
    agent._iterations_completed = 200
    
    # Create shared memory with many high-significance patterns
    shared_mem = SharedMemoryPool()
    for i in range(10):
        shared_mem.broadcast_pattern({
            'iteration': 195 + i,
            'pattern_type': 'geometric_similarity',
            'significance': 'high',
            'geometric_data': {
                'golden_ratio_percentage': 30.0,
                'dominant_platonic_solid': 'dodecahedron'
            }
        })
    
    move = Mock()
    move.move_type = MoveType.BACKBONE_ROTATION
    
    guidance = agent._get_pattern_guidance(move, shared_mem)
    
    # Should be clamped to max 1.5
    assert guidance <= 1.5


# ============================================================================
# Integration Test
# ============================================================================

def test_full_relay_workflow(mediator_agent, sample_geometric_pattern, sample_conformation):
    """Test complete relay workflow: detect -> relay -> broadcast."""
    # Step 1: Relay to QCPP
    qcpp_metrics = mediator_agent.relay_to_qcpp(sample_geometric_pattern, sample_conformation)
    assert qcpp_metrics is not None
    
    # Step 2: Broadcast to agents
    success = mediator_agent.broadcast_to_agents(sample_geometric_pattern, qcpp_metrics)
    assert success is True
    
    # Step 3: Verify pattern in shared memory
    patterns = mediator_agent.shared_memory.retrieve_recent_patterns(
        current_iteration=sample_geometric_pattern.iteration,
        max_age=100
    )
    
    assert len(patterns) == 1
    assert patterns[0]['pattern_type'] == 'geometric_similarity'
    assert 'qcpp_metrics' in patterns[0]
    assert patterns[0]['qcpp_metrics']['qcp_score'] == 4.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
