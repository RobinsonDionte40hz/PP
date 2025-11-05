"""
Memory system implementation for UBF protein system.

This module implements the memory system that stores and retrieves
significant conformational transitions to guide future exploration.
"""

import uuid
import logging
from typing import List, Dict, Optional, Any
from collections import defaultdict

from .interfaces import IMemorySystem, ISharedMemoryPool
from .models import ConformationalMemory, QCPPValidatedMemory, ConformationalOutcome, ConsciousnessCoordinates, BehavioralStateData
from .config import MEMORY_SIGNIFICANCE_THRESHOLD, MAX_MEMORIES_PER_AGENT, MEMORY_INFLUENCE_MIN, MEMORY_INFLUENCE_MAX
from .config import SHARED_MEMORY_SIGNIFICANCE_THRESHOLD, MAX_SHARED_MEMORY_POOL_SIZE

# Set up logging
logger = logging.getLogger(__name__)


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
        try:
            # Validate memory data
            if not hasattr(memory, 'significance') or not hasattr(memory, 'move_type'):
                logger.warning(f"Invalid memory object, missing required attributes")
                return
            
            if memory.significance >= MEMORY_SIGNIFICANCE_THRESHOLD:
                self._memories[memory.move_type].append(memory)
                self._memory_count += 1

                # Auto-prune if we exceed the limit
                if self._memory_count > MAX_MEMORIES_PER_AGENT:
                    self._prune_memories()
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            # Continue execution - memory storage is non-critical

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
        try:
            # Type hint for PyPy JIT optimization
            memories: List[ConformationalMemory] = self._memories.get(move_type, [])
            
            if not memories:
                return []

            # Sort by influence weight (descending)
            # Cache weights to avoid recalculation during sort
            memory_weights: List[tuple] = []
            for m in memories:
                memory_weights.append((m, m.get_influence_weight()))
            
            memory_weights.sort(key=lambda x: x[1], reverse=True)
            
            sorted_memories: List[ConformationalMemory] = [m for m, _ in memory_weights[:max_count]]

            return sorted_memories
        except Exception as e:
            logger.error(f"Error retrieving memories for {move_type}: {e}")
            # Return empty list on error - allows execution to continue
            return []

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
        # Type hints for PyPy JIT optimization
        memories: List[ConformationalMemory] = self._memories.get(move_type, [])
        memory_count: int = len(memories)

        if memory_count == 0:
            # No memories = neutral influence
            return 1.0

        # Calculate success rate with explicit types
        success_count: int = sum(1 for m in memories if m.success)
        success_rate: float = success_count / memory_count

        # Map success rate to influence range
        # High success (1.0) -> high influence (1.5) = more conservative
        # Low success (0.0) -> low influence (0.8) = more exploratory
        influence_range: float = MEMORY_INFLUENCE_MAX - MEMORY_INFLUENCE_MIN
        influence: float = MEMORY_INFLUENCE_MIN + (success_rate * influence_range)

        return influence

    def create_memory_from_outcome(self,
                                 outcome: ConformationalOutcome,
                                 consciousness_state: ConsciousnessCoordinates,
                                 behavioral_state: BehavioralStateData,
                                 qcpp_metrics: Optional[Any] = None,
                                 conformation: Optional[Any] = None) -> ConformationalMemory:
        """
        Create a memory from a conformational outcome.
        
        If qcpp_metrics is provided, creates a QCPPValidatedMemory with
        enhanced significance calculation including QCPP stability.
        If conformation is provided, generates hash for QCPP metrics reuse.

        Args:
            outcome: The outcome to create memory from
            consciousness_state: Consciousness state when outcome occurred
            behavioral_state: Behavioral state when outcome occurred
            qcpp_metrics: Optional QCPP metrics for validation (QCPPMetrics instance)
            conformation: Optional conformation for hash generation

        Returns:
            New ConformationalMemory or QCPPValidatedMemory instance
        """
        # Calculate base significance with QCPP metrics if available
        base_significance = self._calculate_significance(outcome, qcpp_metrics)
        
        # Extract timestamp from move_id if possible
        timestamp = 0
        if '_' in outcome.move_executed.move_id:
            parts = outcome.move_executed.move_id.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                timestamp = int(parts[1])
        
        # Create QCPP-validated memory if metrics provided
        if qcpp_metrics is not None:
            qcpp_significance = self._calculate_qcpp_significance(qcpp_metrics, outcome)
            
            # Combine base and QCPP significance (weighted: 70% base, 30% QCPP)
            total_significance = min(1.0, base_significance * 0.7 + qcpp_significance * 0.3)
            
            # Generate conformation hash if conformation provided
            conf_hash = None
            if conformation is not None:
                conf_hash = self._hash_conformation(conformation)
            
            return QCPPValidatedMemory(
                memory_id=str(uuid.uuid4()),
                move_type=outcome.move_executed.move_type.value,
                significance=total_significance,
                energy_change=outcome.energy_change,
                rmsd_change=outcome.rmsd_change,
                success=outcome.success,
                timestamp=timestamp,
                consciousness_state=consciousness_state,
                behavioral_state=behavioral_state,
                qcpp_metrics=qcpp_metrics,
                qcpp_significance=qcpp_significance,
                conformation_hash=conf_hash
            )
        else:
            # Create standard memory without QCPP validation
            return ConformationalMemory(
                memory_id=str(uuid.uuid4()),
                move_type=outcome.move_executed.move_type.value,
                significance=base_significance,
                energy_change=outcome.energy_change,
                rmsd_change=outcome.rmsd_change,
                success=outcome.success,
                timestamp=timestamp,
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

    def _calculate_significance(self, outcome: ConformationalOutcome, qcpp_metrics: Optional[Any] = None) -> float:
        """
        Calculate significance score for an outcome (0.0-1.0).
        
        Now uses 8 signals for comprehensive learning (NEW: geometric targeting):
        1. Energy impact (25%): Magnitude of energy change  [reduced from 30%]
        2. Structural novelty (20%): RMSD change
        3. THz activity (15%): 40 Hz resonance patterns from QCPP
        4. Geometric patterns (10%): Golden ratio (φ) angle matching
        5. Field coherence (10%): Quantum field alignment
        6. Hydrophobic clustering (5%): Core formation patterns  [reduced from 10%]
        7. Secondary structure (5%): Helix/sheet formation
        8. Geometric targeting (10%): Similarity to target Platonic solid [NEW]

        Args:
            outcome: The outcome to evaluate
            qcpp_metrics: Optional QCPP metrics for physics-based signals

        Returns:
            Significance score between 0.0 and 1.0
        """
        # Signal 1: Energy impact (25%) - larger changes = more significant [reduced from 30%]
        energy_significance = min(1.0, abs(outcome.energy_change) / 100.0)
        
        # Signal 2: Structural novelty (20%) - larger RMSD changes = more significant
        structural_significance = min(1.0, outcome.rmsd_change / 5.0)
        
        # Initialize physics-based signals to neutral values
        thz_significance = 0.5  # Default: moderate significance
        geometric_significance = 0.5
        coherence_significance = 0.5
        geometric_targeting_significance = 0.0  # NEW: Default 0 (no target)
        
        # Signals 3-5, 8: Physics-based signals from QCPP (if available)
        if qcpp_metrics is not None:
            try:
                # Signal 3: THz activity (15%) - resonance with consciousness frequencies
                # 40 Hz is the target "gamma band" for consciousness
                # QCPP's QCP score correlates with THz resonance strength
                # Normalize QCP score (typically 3-8) to significance
                thz_significance = min(1.0, max(0.0, (qcpp_metrics.qcp_score - 3.0) / 5.0))
                
                # Signal 4: Geometric patterns (10%) - golden ratio matching
                # phi_match_score is already 0-1, use directly
                geometric_significance = qcpp_metrics.phi_match_score
                
                # Signal 5: Field coherence (10%) - quantum field alignment
                # Normalize coherence from [-1, 1] to [0, 1]
                coherence_significance = (qcpp_metrics.field_coherence + 1.0) / 2.0
                
                # Signal 8: Geometric targeting (10%) - NEW: Prescriptive geometric guidance
                # geometric_similarity is 0.0 if no target, 0.0-1.0 if targeting enabled
                geometric_targeting_significance = qcpp_metrics.geometric_similarity
                
                # Boost significance for high geometric similarity (strong attractor)
                if geometric_targeting_significance > 0.7:
                    geometric_targeting_significance *= 1.5  # Extra weight for good matches
                    geometric_targeting_significance = min(1.0, geometric_targeting_significance)
                
            except Exception as e:
                logger.warning(f"Error extracting QCPP signals for significance: {e}")
        
        # Signal 6: Hydrophobic clustering (5%) - core formation [reduced from 10%]
        hydrophobic_significance = self._calculate_hydrophobic_significance(outcome.new_conformation)
        
        # Signal 7: Secondary structure (5%) - helix/sheet formation
        ss_significance = self._calculate_secondary_structure_significance(outcome.new_conformation)
        
        # Weighted combination of all 8 signals (total: 100%)
        significance = (
            energy_significance * 0.25 +            # Reduced from 0.30
            structural_significance * 0.20 +
            thz_significance * 0.15 +
            geometric_significance * 0.10 +
            coherence_significance * 0.10 +
            hydrophobic_significance * 0.05 +       # Reduced from 0.10
            ss_significance * 0.05 +
            geometric_targeting_significance * 0.10  # NEW: 8th signal
        )
        
        # Success bonus (up to +0.2, capped at 1.0)
        if outcome.success:
            significance = min(1.0, significance + 0.2)
        
        return min(1.0, significance)
    
    def _calculate_hydrophobic_significance(self, conformation: Any) -> float:
        """
        Calculate hydrophobic clustering significance (0.0-1.0).
        
        Measures how well hydrophobic residues cluster together,
        indicating core formation - a key folding event.
        
        Args:
            conformation: Conformation to analyze
            
        Returns:
            Hydrophobic significance score (0-1)
        """
        try:
            # Hydrophobic residues
            hydrophobic = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}
            
            # Get sequence and check hydrophobic count
            sequence = conformation.sequence
            hydrophobic_indices = [i for i, aa in enumerate(sequence) if aa in hydrophobic]
            
            if len(hydrophobic_indices) < 2:
                return 0.3  # Neutral if insufficient hydrophobic residues
            
            # Calculate average distance between hydrophobic residues
            coords = conformation.atom_coordinates
            total_distance = 0.0
            pair_count = 0
            
            for i in range(len(hydrophobic_indices)):
                for j in range(i + 1, len(hydrophobic_indices)):
                    idx1, idx2 = hydrophobic_indices[i], hydrophobic_indices[j]
                    if idx1 < len(coords) and idx2 < len(coords):
                        c1, c2 = coords[idx1], coords[idx2]
                        dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)**0.5
                        total_distance += dist
                        pair_count += 1
            
            if pair_count == 0:
                return 0.3
            
            avg_distance = total_distance / pair_count
            
            # Closer clustering = higher significance
            # Target: < 10 Å is good clustering, > 20 Å is poor
            if avg_distance < 10.0:
                return 1.0  # Excellent clustering
            elif avg_distance < 15.0:
                return 0.7  # Good clustering
            elif avg_distance < 20.0:
                return 0.5  # Moderate clustering
            else:
                return 0.3  # Poor clustering
                
        except Exception as e:
            logger.warning(f"Error calculating hydrophobic significance: {e}")
            return 0.5  # Neutral default
    
    def _calculate_secondary_structure_significance(self, conformation: Any) -> float:
        """
        Calculate secondary structure significance (0.0-1.0).
        
        Measures presence of helices and sheets, indicating
        structured folding progress.
        
        Args:
            conformation: Conformation to analyze
            
        Returns:
            Secondary structure significance score (0-1)
        """
        try:
            ss = conformation.secondary_structure
            total_residues = len(ss)
            
            if total_residues == 0:
                return 0.5
            
            # Count structured elements
            helix_count = ss.count('H')
            sheet_count = ss.count('E')
            structured_count = helix_count + sheet_count
            
            # Calculate structured fraction
            structured_fraction = structured_count / total_residues
            
            # Reward continuous stretches (not isolated residues)
            continuous_bonus = 0.0
            stretch_length = 0
            
            for element in ss:
                if element in ['H', 'E']:
                    stretch_length += 1
                else:
                    if stretch_length >= 3:  # Meaningful stretch
                        continuous_bonus += 0.1
                    stretch_length = 0
            
            # Final check for last stretch
            if stretch_length >= 3:
                continuous_bonus += 0.1
            
            # Combine fraction and bonus (capped at 1.0)
            significance = min(1.0, structured_fraction * 0.7 + min(0.3, continuous_bonus))
            
            return significance
            
        except Exception as e:
            logger.warning(f"Error calculating secondary structure significance: {e}")
            return 0.5  # Neutral default
    
    def _calculate_qcpp_significance(self, qcpp_metrics: Any, outcome: ConformationalOutcome) -> float:
        """
        Calculate QCPP contribution to memory significance (0.0-1.0).
        
        Based on QCPP stability score with high-significance detection.
        High significance is triggered when:
        - QCPP stability > 1.5 (stable structure)
        - Energy change < -20 kcal/mol (favorable)
        
        Args:
            qcpp_metrics: QCPP metrics for the conformation (QCPPMetrics instance)
            outcome: Conformational outcome for energy information
            
        Returns:
            QCPP significance score between 0.0 and 1.0
        """
        try:
            # Extract stability score from QCPP metrics
            stability = qcpp_metrics.stability_score if hasattr(qcpp_metrics, 'stability_score') else 0.0
            
            # High-significance detection
            is_high_significance = (stability > 1.5) and (outcome.energy_change < -20.0)
            
            if is_high_significance:
                # High significance: boost to 0.9-1.0 range
                return min(1.0, 0.9 + (stability - 1.5) * 0.1)
            else:
                # Normal significance: map stability to 0.0-0.8 range
                # Stability typically ranges 0-3, so normalize
                normalized_stability = min(1.0, stability / 3.0)
                return normalized_stability * 0.8
        except Exception as e:
            logger.warning(f"Error calculating QCPP significance: {e}")
            return 0.0

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

    def get_qcpp_for_conformation(self, conformation: Any) -> Optional[Any]:
        """
        Query memory for QCPP metrics of a conformation.
        
        This enables reusing QCPP calculations when revisiting conformations,
        avoiding redundant computation (0.3-2.0ms per analysis).
        
        Args:
            conformation: Conformation to query QCPP metrics for
            
        Returns:
            QCPPMetrics if found in memory, None if never analyzed before
        """
        try:
            # Generate hash for conformation (coordinate-based)
            conf_hash = self._hash_conformation(conformation)
            
            # Search through all memories for matching conformation
            for move_type, memories in self._memories.items():
                for memory in memories:
                    # Check if this is a QCPP-validated memory with matching conformation
                    if isinstance(memory, QCPPValidatedMemory) and hasattr(memory, 'qcpp_metrics'):
                        # Check if memory has conformation hash
                        if hasattr(memory, 'conformation_hash'):
                            if memory.conformation_hash == conf_hash:
                                logger.debug(f"✓ Found QCPP metrics in memory (self-revisit)")
                                return memory.qcpp_metrics
            
            return None
        except Exception as e:
            logger.warning(f"Error querying QCPP from memory: {e}")
            return None
    
    def store_qcpp_metrics(self, conformation: Any, qcpp_metrics: Any) -> None:
        """
        Store QCPP metrics for a conformation (lightweight storage).
        
        Note: This method is now simplified since conformation_hash is set
        during memory creation in create_memory_from_outcome().
        
        This method is kept for backward compatibility but is essentially
        a no-op since QCPP metrics are already stored with hash during
        memory creation.
        
        Args:
            conformation: Conformation these metrics apply to
            qcpp_metrics: QCPP metrics to store
        """
        # No-op: QCPP metrics are now stored with hash during memory creation
        logger.debug(f"✓ QCPP metrics stored via memory creation")
        pass
    
    def _hash_conformation(self, conformation: Any) -> str:
        """
        Generate hash for conformation based on atom coordinates.
        
        Uses first 10 CA atom coordinates (rounded to 1 decimal) to create
        a compact hash that identifies unique conformations while being
        tolerant to minor numerical differences.
        
        Args:
            conformation: Conformation to hash
            
        Returns:
            Hash string for conformation lookup
        """
        try:
            import hashlib
            
            # Extract coordinates (first 10 atoms for speed)
            coords = []
            if hasattr(conformation, 'atom_coordinates'):
                atom_coords = conformation.atom_coordinates
                # atom_coordinates is a List[Tuple[float, float, float]]
                # Take first 10 atoms, round to 1 decimal place
                for coord in atom_coords[:10]:
                    if len(coord) >= 3:
                        coords.extend([round(coord[0], 1), round(coord[1], 1), round(coord[2], 1)])
            
            # Create hash from coordinate string
            coord_str = '_'.join(str(c) for c in coords)
            return hashlib.sha256(coord_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error hashing conformation: {e}")
            return ""


class SharedMemoryPool(ISharedMemoryPool):
    """
    Implementation of shared memory pool across all agents.

    Stores high-significance memories that can be shared between agents
    to accelerate collective learning and avoid redundant exploration.
    """

    def __init__(self):
        """
        Initialize empty shared memory pool.
        """
        self._shared_memories: List[ConformationalMemory] = []
        self._memory_count = 0

    def share_memory(self, memory: ConformationalMemory) -> None:
        """
        Share high-significance memory (>= 0.7) with all agents.

        Args:
            memory: The memory to potentially share
        """
        if memory.significance >= SHARED_MEMORY_SIGNIFICANCE_THRESHOLD:
            self._shared_memories.append(memory)
            self._memory_count += 1

            # Auto-prune if we exceed the limit
            if self._memory_count > MAX_SHARED_MEMORY_POOL_SIZE:
                self.prune_pool()

    def retrieve_shared_memories(self, move_type: str, max_count: int = 10) -> List[ConformationalMemory]:
        """
        Retrieve relevant shared memories for move evaluation.

        Returns most significant shared memories for the given move type,
        sorted by influence weight.

        Args:
            move_type: Type of move to get memories for
            max_count: Maximum number of memories to return

        Returns:
            List of relevant shared memories, sorted by influence weight
        """
        # Filter memories by move type
        relevant_memories = [
            memory for memory in self._shared_memories
            if memory.move_type == move_type
        ]

        # Sort by influence weight (descending)
        sorted_memories = sorted(
            relevant_memories,
            key=lambda m: m.get_influence_weight(),
            reverse=True
        )

        return sorted_memories[:max_count]

    def prune_pool(self, max_size: int = MAX_SHARED_MEMORY_POOL_SIZE) -> None:
        """
        Prune pool to maintain max size by weighted significance.

        Removes least influential memories first.

        Args:
            max_size: Maximum size to prune to
        """
        if self._memory_count <= max_size:
            return

        # Sort all memories by influence weight (ascending - least influential first)
        sorted_memories = sorted(
            self._shared_memories,
            key=lambda m: m.get_influence_weight()
        )

        # Keep only the most influential memories
        excess_count = self._memory_count - max_size
        self._shared_memories = sorted_memories[excess_count:]
        self._memory_count = len(self._shared_memories)

    def get_pool_stats(self) -> Dict[str, int]:
        """
        Get statistics about the shared memory pool.

        Returns:
            Dictionary with memory counts by move type
        """
        stats = {}
        for memory in self._shared_memories:
            move_type = memory.move_type
            stats[move_type] = stats.get(move_type, 0) + 1
        return stats

    def get_total_memories(self) -> int:
        """
        Get total number of memories in the pool.

        Returns:
            Total memory count
        """
        return self._memory_count