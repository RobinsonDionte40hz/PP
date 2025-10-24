"""
Memory system implementation for UBF protein system.

This module implements the memory system that stores and retrieves
significant conformational transitions to guide future exploration.
"""

import uuid
from typing import List, Dict
from collections import defaultdict

from .interfaces import IMemorySystem
from .models import ConformationalMemory, ConformationalOutcome, ConsciousnessCoordinates, BehavioralStateData
from .config import MEMORY_SIGNIFICANCE_THRESHOLD, MAX_MEMORIES_PER_AGENT, MEMORY_INFLUENCE_MIN, MEMORY_INFLUENCE_MAX


class MemorySystem(IMemorySystem):
    """
    Implementation of experience memory management.

    Stores significant conformational outcomes and provides memory-based
    influence for future move selection. Memories are pruned to maintain
    performance and relevance.
    """

    def __init__(self):
        """
        Initialize empty memory system.
        """
        self._memories: Dict[str, List[ConformationalMemory]] = defaultdict(list)
        self._memory_count = 0

    def store_memory(self, memory: ConformationalMemory) -> None:
        """
        Store memory if significance >= threshold, auto-prune if > max memories.

        Args:
            memory: The conformational memory to potentially store
        """
        if memory.significance >= MEMORY_SIGNIFICANCE_THRESHOLD:
            self._memories[memory.move_type].append(memory)
            self._memory_count += 1

            # Auto-prune if we exceed the limit
            if self._memory_count > MAX_MEMORIES_PER_AGENT:
                self._prune_memories()

    def retrieve_relevant_memories(self, move_type: str, max_count: int = 10) -> List[ConformationalMemory]:
        """
        Retrieve relevant memories for move evaluation.

        Returns most significant memories for the given move type,
        sorted by influence weight.

        Args:
            move_type: Type of move to get memories for
            max_count: Maximum number of memories to return

        Returns:
            List of relevant memories, sorted by influence weight
        """
        memories = self._memories.get(move_type, [])

        # Sort by influence weight (descending)
        sorted_memories = sorted(
            memories,
            key=lambda m: m.get_influence_weight(),
            reverse=True
        )

        return sorted_memories[:max_count]

    def calculate_memory_influence(self, move_type: str) -> float:
        """
        Calculate memory influence multiplier (0.8-1.5).

        Based on historical success rate for this move type.
        Higher success rates = higher influence (more conservative).
        Lower success rates = lower influence (more exploratory).

        Args:
            move_type: Type of move to calculate influence for

        Returns:
            Influence multiplier between MEMORY_INFLUENCE_MIN and MEMORY_INFLUENCE_MAX
        """
        memories = self._memories.get(move_type, [])

        if not memories:
            # No memories = neutral influence
            return 1.0

        # Calculate success rate
        successful_memories = [m for m in memories if m.success]
        success_rate = len(successful_memories) / len(memories)

        # Map success rate to influence range
        # High success (1.0) -> high influence (1.5) = more conservative
        # Low success (0.0) -> low influence (0.8) = more exploratory
        influence_range = MEMORY_INFLUENCE_MAX - MEMORY_INFLUENCE_MIN
        influence = MEMORY_INFLUENCE_MIN + (success_rate * influence_range)

        return influence

    def create_memory_from_outcome(self,
                                 outcome: ConformationalOutcome,
                                 consciousness_state: ConsciousnessCoordinates,
                                 behavioral_state: BehavioralStateData) -> ConformationalMemory:
        """
        Create a memory from a conformational outcome.

        Args:
            outcome: The outcome to create memory from
            consciousness_state: Consciousness state when outcome occurred
            behavioral_state: Behavioral state when outcome occurred

        Returns:
            New ConformationalMemory instance
        """
        # Calculate significance based on outcome impact
        significance = self._calculate_significance(outcome)

        return ConformationalMemory(
            memory_id=str(uuid.uuid4()),
            move_type=outcome.move_executed.move_type.value,  # Convert enum to string
            significance=significance,
            energy_change=outcome.energy_change,
            rmsd_change=outcome.rmsd_change,
            success=outcome.success,
            timestamp=outcome.move_executed.move_id.split('_')[1] if '_' in outcome.move_executed.move_id else 0,  # Extract timestamp from move_id
            consciousness_state=consciousness_state,
            behavioral_state=behavioral_state
        )

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get statistics about stored memories.

        Returns:
            Dictionary with memory counts by move type
        """
        return {
            move_type: len(memories)
            for move_type, memories in self._memories.items()
        }

    def _calculate_significance(self, outcome: ConformationalOutcome) -> float:
        """
        Calculate significance score for an outcome (0.0-1.0).

        Based on energy change magnitude and structural impact.

        Args:
            outcome: The outcome to evaluate

        Returns:
            Significance score between 0.0 and 1.0
        """
        # Energy significance (larger changes = more significant)
        energy_significance = min(1.0, abs(outcome.energy_change) / 100.0)

        # Structural significance (larger RMSD changes = more significant)
        structural_significance = min(1.0, outcome.rmsd_change / 5.0)

        # Success bonus
        success_bonus = 0.2 if outcome.success else 0.0

        # Combine factors (weighted average)
        significance = (
            energy_significance * 0.5 +
            structural_significance * 0.3 +
            success_bonus
        )

        return min(1.0, significance)

    def _prune_memories(self) -> None:
        """
        Prune memories to stay within MAX_MEMORIES_PER_AGENT limit.

        Removes least influential memories first.
        """
        # Collect all memories with their influence weights
        all_memories = []
        for move_type, memories in self._memories.items():
            for memory in memories:
                all_memories.append((memory, memory.get_influence_weight()))

        # Sort by influence weight (ascending - least influential first)
        all_memories.sort(key=lambda x: x[1])

        # Remove excess memories
        excess_count = self._memory_count - MAX_MEMORIES_PER_AGENT
        if excess_count > 0:
            memories_to_remove = all_memories[:excess_count]

            for memory, _ in memories_to_remove:
                # Remove from the appropriate move type list
                move_type = memory.move_type
                if memory in self._memories[move_type]:
                    self._memories[move_type].remove(memory)
                    self._memory_count -= 1