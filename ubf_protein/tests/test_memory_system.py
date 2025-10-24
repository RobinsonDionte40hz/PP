"""
Unit tests for memory system implementation.
"""

import pytest
import time
from unittest.mock import Mock

from ubf_protein.memory_system import MemorySystem
from ubf_protein.models import (
    ConformationalMemory, ConformationalOutcome, ConformationalMove,
    ConsciousnessCoordinates, BehavioralStateData, MoveType
)
from ubf_protein.config import MEMORY_SIGNIFICANCE_THRESHOLD, MAX_MEMORIES_PER_AGENT


class TestMemorySystem:
    """Test basic memory system functionality"""

    def test_initialization(self):
        """Test memory system initializes empty"""
        memory_system = MemorySystem()
        assert memory_system._memory_count == 0
        assert len(memory_system._memories) == 0

    def test_store_high_significance_memory(self):
        """Test storing memory with high significance"""
        memory_system = MemorySystem()

        # Create a high-significance memory
        memory = self._create_test_memory(significance=0.8)

        memory_system.store_memory(memory)

        assert memory_system._memory_count == 1
        assert len(memory_system._memories[memory.move_type]) == 1

    def test_reject_low_significance_memory(self):
        """Test rejecting memory below significance threshold"""
        memory_system = MemorySystem()

        # Create a low-significance memory
        memory = self._create_test_memory(significance=MEMORY_SIGNIFICANCE_THRESHOLD - 0.1)

        memory_system.store_memory(memory)

        assert memory_system._memory_count == 0
        assert len(memory_system._memories) == 0

    def test_retrieve_relevant_memories(self):
        """Test retrieving memories for a specific move type"""
        memory_system = MemorySystem()

        # Store memories for different move types
        helix_memory = self._create_test_memory(move_type="helix_formation", significance=0.8)
        sheet_memory = self._create_test_memory(move_type="sheet_formation", significance=0.7)
        rotation_memory = self._create_test_memory(move_type="backbone_rotation", significance=0.9)

        memory_system.store_memory(helix_memory)
        memory_system.store_memory(sheet_memory)
        memory_system.store_memory(rotation_memory)

        # Retrieve helix memories
        helix_memories = memory_system.retrieve_relevant_memories("helix_formation")
        assert len(helix_memories) == 1
        assert helix_memories[0].memory_id == helix_memory.memory_id

    def test_memory_sorting_by_influence(self):
        """Test memories are sorted by influence weight"""
        memory_system = MemorySystem()

        # Create memories with different timestamps (older = lower influence)
        recent_memory = self._create_test_memory(significance=0.8, timestamp=int(time.time() * 1000))
        old_memory = self._create_test_memory(significance=0.8, timestamp=int(time.time() * 1000) - 3600000)  # 1 hour ago

        memory_system.store_memory(recent_memory)
        memory_system.store_memory(old_memory)

        retrieved = memory_system.retrieve_relevant_memories(recent_memory.move_type)
        assert len(retrieved) == 2
        # Recent memory should come first (higher influence)
        assert retrieved[0].timestamp > retrieved[1].timestamp

    def test_calculate_memory_influence_high_success(self):
        """Test influence calculation with high success rate"""
        memory_system = MemorySystem()

        # Add successful memories
        for _ in range(8):
            memory = self._create_test_memory(success=True)
            memory_system.store_memory(memory)

        # Add one failure
        failure_memory = self._create_test_memory(success=False)
        memory_system.store_memory(failure_memory)

        influence = memory_system.calculate_memory_influence("backbone_rotation")
        # High success rate should give high influence
        assert influence > 1.2  # Above neutral

    def test_calculate_memory_influence_low_success(self):
        """Test influence calculation with low success rate"""
        memory_system = MemorySystem()

        # Add mostly failed memories
        for _ in range(8):
            memory = self._create_test_memory(success=False)
            memory_system.store_memory(memory)

        # Add one success
        success_memory = self._create_test_memory(success=True)
        memory_system.store_memory(success_memory)

        influence = memory_system.calculate_memory_influence("backbone_rotation")
        # Low success rate should give low influence
        assert influence < 0.9  # Below neutral

    def test_calculate_memory_influence_no_memories(self):
        """Test influence calculation with no memories"""
        memory_system = MemorySystem()

        influence = memory_system.calculate_memory_influence("backbone_rotation")
        assert influence == 1.0  # Neutral influence

    def test_memory_pruning(self):
        """Test automatic memory pruning when exceeding max count"""
        memory_system = MemorySystem()

        # Store more memories than the limit
        memories_to_store = MAX_MEMORIES_PER_AGENT + 5

        for i in range(memories_to_store):
            memory = self._create_test_memory(significance=0.8, timestamp=int(time.time() * 1000) - i * 1000)
            memory_system.store_memory(memory)

        # Should have pruned down to max
        assert memory_system._memory_count <= MAX_MEMORIES_PER_AGENT

    def test_create_memory_from_outcome(self):
        """Test creating memory from conformational outcome"""
        memory_system = MemorySystem()

        # Create mock outcome
        move = ConformationalMove(
            move_id="test_move_123456",
            move_type=MoveType.BACKBONE_ROTATION,
            target_residues=[1, 2, 3],
            estimated_energy_change=-50.0,
            estimated_rmsd_change=2.0,
            required_capabilities={"can_large_rotation": True},
            energy_barrier=10.0,
            structural_feasibility=0.8
        )

        outcome = ConformationalOutcome(
            move_executed=move,
            new_conformation=Mock(),  # Mock conformation
            energy_change=-50.0,
            rmsd_change=2.0,
            success=True,
            significance=0.7
        )

        consciousness = ConsciousnessCoordinates(
            frequency=8.0,
            coherence=0.7,
            last_update_timestamp=int(time.time() * 1000)
        )

        behavioral = BehavioralStateData(
            exploration_energy=0.6,
            structural_focus=0.7,
            conformational_bias=0.1,
            hydrophobic_drive=0.5,
            risk_tolerance=0.4,
            native_state_ambition=0.49,
            cached_timestamp=int(time.time() * 1000)
        )

        memory = memory_system.create_memory_from_outcome(outcome, consciousness, behavioral)

        assert isinstance(memory, ConformationalMemory)
        assert memory.move_type == "backbone_rotation"
        assert memory.energy_change == -50.0
        assert memory.success == True
        assert memory.significance > 0.0

    def test_get_memory_stats(self):
        """Test getting memory statistics"""
        memory_system = MemorySystem()

        # Store memories of different types
        memory_system.store_memory(self._create_test_memory(move_type="helix_formation"))
        memory_system.store_memory(self._create_test_memory(move_type="sheet_formation"))
        memory_system.store_memory(self._create_test_memory(move_type="helix_formation"))

        stats = memory_system.get_memory_stats()

        assert stats["helix_formation"] == 2
        assert stats["sheet_formation"] == 1
        assert "backbone_rotation" not in stats or stats["backbone_rotation"] == 0

    def _create_test_memory(self,
                          move_type: str = "backbone_rotation",
                          significance: float = 0.8,
                          success: bool = True,
                          timestamp: int = None) -> ConformationalMemory:
        """Helper to create test memory"""
        if timestamp is None:
            timestamp = int(time.time() * 1000)

        return ConformationalMemory(
            memory_id=f"test_memory_{timestamp}",
            move_type=move_type,
            significance=significance,
            energy_change=-25.0 if success else 25.0,
            rmsd_change=1.5,
            success=success,
            timestamp=timestamp,
            consciousness_state=ConsciousnessCoordinates(
                frequency=8.0, coherence=0.7, last_update_timestamp=timestamp
            ),
            behavioral_state=BehavioralStateData(
                exploration_energy=0.6,
                structural_focus=0.7,
                conformational_bias=0.1,
                hydrophobic_drive=0.5,
                risk_tolerance=0.4,
                native_state_ambition=0.49,
                cached_timestamp=timestamp
            )
        )