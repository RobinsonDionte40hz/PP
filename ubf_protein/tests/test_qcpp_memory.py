"""
Unit tests for QCPP-validated memory system.

Tests the integration of QCPP metrics into the memory system,
including QCPPValidatedMemory dataclass, significance calculation
with QCPP metrics, and high-significance threshold detection.
"""

import pytest
import time
from dataclasses import replace

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ubf_protein.models import (
    QCPPValidatedMemory,
    ConformationalMemory,
    ConformationalOutcome,
    ConformationalMove,
    Conformation,
    ConsciousnessCoordinates,
    BehavioralStateData
)
from ubf_protein.memory_system import MemorySystem
from ubf_protein.qcpp_integration import QCPPMetrics
from ubf_protein.interfaces import MoveType


def create_test_conformation(sequence="ACDEF", energy=-100.0, rmsd=5.0):
    """Helper to create test conformation."""
    n_residues = len(sequence)
    return Conformation(
        conformation_id="test-conf",
        sequence=sequence,
        atom_coordinates=[(0.0, 0.0, 0.0)] * (n_residues * 3),  # 3 atoms per residue (simplified)
        energy=energy,
        rmsd_to_native=rmsd,
        secondary_structure=['C'] * n_residues,
        phi_angles=[0.0] * n_residues,
        psi_angles=[0.0] * n_residues,
        available_move_types=[],
        structural_constraints={}
    )


def create_test_move(move_id="move_123", move_type=MoveType.BACKBONE_ROTATION):
    """Helper to create test conformational move."""
    return ConformationalMove(
        move_id=move_id,
        move_type=move_type,
        target_residues=[0, 1],
        estimated_energy_change=-10.0,
        estimated_rmsd_change=1.0,
        required_capabilities={},
        energy_barrier=5.0,
        structural_feasibility=0.8
    )


def create_test_outcome(energy_change=-20.0, rmsd_change=1.5, success=True):
    """Helper to create test conformational outcome."""
    move = create_test_move()
    conf = create_test_conformation(energy=-100.0 + energy_change)
    
    return ConformationalOutcome(
        move_executed=move,
        new_conformation=conf,
        energy_change=energy_change,
        rmsd_change=rmsd_change,
        success=success,
        significance=0.5  # Default significance
    )


class TestQCPPValidatedMemoryDataclass:
    """Test suite for QCPPValidatedMemory dataclass."""
    
    def test_creation_with_qcpp_metrics(self):
        """Test creating QCPPValidatedMemory with QCPP metrics."""
        qcpp_metrics = QCPPMetrics(
            qcp_score=5.0,
            field_coherence=0.5,
            stability_score=2.0,
            phi_match_score=0.8,
            calculation_time_ms=1.5
        )
        
        consciousness = ConsciousnessCoordinates(
            frequency=9.0,
            coherence=0.6,
            last_update_timestamp=int(time.time() * 1000)
        )
        
        behavioral = BehavioralStateData.from_consciousness(9.0, 0.6)
        
        memory = QCPPValidatedMemory(
            memory_id="test-123",
            move_type="PHI_PATTERNED",
            significance=0.8,
            energy_change=-25.0,
            rmsd_change=1.5,
            success=True,
            timestamp=int(time.time() * 1000),
            consciousness_state=consciousness,
            behavioral_state=behavioral,
            qcpp_metrics=qcpp_metrics,
            qcpp_significance=0.75
        )
        
        assert memory.memory_id == "test-123"
        assert memory.move_type == "PHI_PATTERNED"
        assert memory.significance == 0.8
        assert memory.qcpp_metrics == qcpp_metrics
        assert memory.qcpp_significance == 0.75
    
    def test_creation_without_qcpp_metrics(self):
        """Test creating QCPPValidatedMemory without QCPP metrics (defaults)."""
        consciousness = ConsciousnessCoordinates(
            frequency=9.0,
            coherence=0.6,
            last_update_timestamp=int(time.time() * 1000)
        )
        
        behavioral = BehavioralStateData.from_consciousness(9.0, 0.6)
        
        memory = QCPPValidatedMemory(
            memory_id="test-456",
            move_type="BACKBONE_ROTATION",
            significance=0.5,
            energy_change=-10.0,
            rmsd_change=0.8,
            success=True,
            timestamp=int(time.time() * 1000),
            consciousness_state=consciousness,
            behavioral_state=behavioral
        )
        
        assert memory.qcpp_metrics is None
        assert memory.qcpp_significance == 0.0
    
    def test_invalid_qcpp_significance_range(self):
        """Test that invalid qcpp_significance raises ValueError."""
        consciousness = ConsciousnessCoordinates(
            frequency=9.0,
            coherence=0.6,
            last_update_timestamp=int(time.time() * 1000)
        )
        
        behavioral = BehavioralStateData.from_consciousness(9.0, 0.6)
        
        qcpp_metrics = QCPPMetrics(5.0, 0.5, 2.0, 0.8, 1.0)
        
        with pytest.raises(ValueError, match="qcpp_significance.*must be in"):
            QCPPValidatedMemory(
                memory_id="test-789",
                move_type="SIDECHAIN_ROTATION",
                significance=0.6,
                energy_change=-15.0,
                rmsd_change=1.2,
                success=True,
                timestamp=int(time.time() * 1000),
                consciousness_state=consciousness,
                behavioral_state=behavioral,
                qcpp_metrics=qcpp_metrics,
                qcpp_significance=1.5  # Invalid: > 1.0
            )
    
    def test_is_high_significance_true(self):
        """Test is_high_significance returns True for qualifying memories."""
        qcpp_metrics = QCPPMetrics(
            qcp_score=6.0,
            field_coherence=0.8,
            stability_score=2.0,  # > 1.5 ✓
            phi_match_score=0.9,
            calculation_time_ms=1.2
        )
        
        consciousness = ConsciousnessCoordinates(9.0, 0.7, int(time.time() * 1000))
        behavioral = BehavioralStateData.from_consciousness(9.0, 0.7)
        
        memory = QCPPValidatedMemory(
            memory_id="high-sig",
            move_type="PHI_PATTERNED",
            significance=0.9,
            energy_change=-30.0,  # < -20 ✓
            rmsd_change=2.0,
            success=True,
            timestamp=int(time.time() * 1000),
            consciousness_state=consciousness,
            behavioral_state=behavioral,
            qcpp_metrics=qcpp_metrics,
            qcpp_significance=0.9
        )
        
        assert memory.is_high_significance() is True
    
    def test_is_high_significance_false_low_stability(self):
        """Test is_high_significance returns False for low stability."""
        qcpp_metrics = QCPPMetrics(
            qcp_score=4.5,
            field_coherence=0.3,
            stability_score=1.0,  # ≤ 1.5 ✗
            phi_match_score=0.5,
            calculation_time_ms=1.0
        )
        
        consciousness = ConsciousnessCoordinates(8.0, 0.5, int(time.time() * 1000))
        behavioral = BehavioralStateData.from_consciousness(8.0, 0.5)
        
        memory = QCPPValidatedMemory(
            memory_id="low-stab",
            move_type="BACKBONE_ROTATION",
            significance=0.6,
            energy_change=-25.0,  # < -20 ✓
            rmsd_change=1.5,
            success=True,
            timestamp=int(time.time() * 1000),
            consciousness_state=consciousness,
            behavioral_state=behavioral,
            qcpp_metrics=qcpp_metrics,
            qcpp_significance=0.4
        )
        
        assert memory.is_high_significance() is False
    
    def test_is_high_significance_false_high_energy(self):
        """Test is_high_significance returns False for unfavorable energy."""
        qcpp_metrics = QCPPMetrics(
            qcp_score=6.5,
            field_coherence=0.9,
            stability_score=2.5,  # > 1.5 ✓
            phi_match_score=0.85,
            calculation_time_ms=1.3
        )
        
        consciousness = ConsciousnessCoordinates(10.0, 0.8, int(time.time() * 1000))
        behavioral = BehavioralStateData.from_consciousness(10.0, 0.8)
        
        memory = QCPPValidatedMemory(
            memory_id="high-energy",
            move_type="SIDECHAIN_ROTATION",
            significance=0.7,
            energy_change=-5.0,  # ≥ -20 ✗
            rmsd_change=0.5,
            success=True,
            timestamp=int(time.time() * 1000),
            consciousness_state=consciousness,
            behavioral_state=behavioral,
            qcpp_metrics=qcpp_metrics,
            qcpp_significance=0.8
        )
        
        assert memory.is_high_significance() is False
    
    def test_is_high_significance_false_no_qcpp(self):
        """Test is_high_significance returns False without QCPP metrics."""
        consciousness = ConsciousnessCoordinates(9.0, 0.6, int(time.time() * 1000))
        behavioral = BehavioralStateData.from_consciousness(9.0, 0.6)
        
        memory = QCPPValidatedMemory(
            memory_id="no-qcpp",
            move_type="BACKBONE_ROTATION",
            significance=0.8,
            energy_change=-30.0,
            rmsd_change=2.0,
            success=True,
            timestamp=int(time.time() * 1000),
            consciousness_state=consciousness,
            behavioral_state=behavioral
            # No qcpp_metrics
        )
        
        assert memory.is_high_significance() is False


class TestMemorySystemQCPPIntegration:
    """Test suite for MemorySystem with QCPP integration."""
    
    def test_create_memory_with_qcpp_metrics(self):
        """Test creating memory with QCPP metrics."""
        memory_system = MemorySystem()
        
        qcpp_metrics = QCPPMetrics(
            qcp_score=5.5,
            field_coherence=0.6,
            stability_score=1.8,
            phi_match_score=0.75,
            calculation_time_ms=1.8
        )
        
        outcome = create_test_outcome(energy_change=-22.0, rmsd_change=1.2, success=True)
        
        consciousness = ConsciousnessCoordinates(9.0, 0.7, 123456789)
        behavioral = BehavioralStateData.from_consciousness(9.0, 0.7)
        
        memory = memory_system.create_memory_from_outcome(
            outcome,
            consciousness,
            behavioral,
            qcpp_metrics=qcpp_metrics
        )
        
        # Should return QCPPValidatedMemory
        assert isinstance(memory, QCPPValidatedMemory)
        assert memory.qcpp_metrics == qcpp_metrics
        assert memory.qcpp_significance > 0.0
        assert 0.0 <= memory.significance <= 1.0
    
    def test_create_memory_without_qcpp_metrics(self):
        """Test creating memory without QCPP metrics (backward compatibility)."""
        memory_system = MemorySystem()
        
        outcome = create_test_outcome(energy_change=-15.0, rmsd_change=0.8, success=True)
        
        consciousness = ConsciousnessCoordinates(8.0, 0.6, 987654321)
        behavioral = BehavioralStateData.from_consciousness(8.0, 0.6)
        
        memory = memory_system.create_memory_from_outcome(
            outcome,
            consciousness,
            behavioral
            # No qcpp_metrics
        )
        
        # Should return standard ConformationalMemory
        assert isinstance(memory, ConformationalMemory)
        assert not isinstance(memory, QCPPValidatedMemory)
    
    def test_qcpp_significance_calculation_high(self):
        """Test QCPP significance calculation for high-significance case."""
        memory_system = MemorySystem()
        
        # High stability + favorable energy = high significance
        qcpp_metrics = QCPPMetrics(
            qcp_score=6.0,
            field_coherence=0.8,
            stability_score=2.0,  # > 1.5
            phi_match_score=0.85,
            calculation_time_ms=1.5
        )
        
        outcome = create_test_outcome(energy_change=-25.0, rmsd_change=1.8, success=True)
        
        qcpp_sig = memory_system._calculate_qcpp_significance(qcpp_metrics, outcome)
        
        # Should be in high significance range (0.9-1.0)
        assert 0.9 <= qcpp_sig <= 1.0
    
    def test_qcpp_significance_calculation_normal(self):
        """Test QCPP significance calculation for normal case."""
        memory_system = MemorySystem()
        
        # Normal stability and energy
        qcpp_metrics = QCPPMetrics(
            qcp_score=4.5,
            field_coherence=0.4,
            stability_score=1.2,  # ≤ 1.5
            phi_match_score=0.6,
            calculation_time_ms=1.2
        )
        
        outcome = create_test_outcome(energy_change=-12.0, rmsd_change=0.9, success=True)
        
        qcpp_sig = memory_system._calculate_qcpp_significance(qcpp_metrics, outcome)
        
        # Should be in normal significance range (0.0-0.8)
        assert 0.0 <= qcpp_sig <= 0.8
    
    def test_combined_significance_calculation(self):
        """Test that combined significance includes both base and QCPP."""
        memory_system = MemorySystem()
        
        qcpp_metrics = QCPPMetrics(
            qcp_score=5.2,
            field_coherence=0.5,
            stability_score=1.6,
            phi_match_score=0.7,
            calculation_time_ms=1.4
        )
        
        # Create outcome with moderate impact
        outcome = create_test_outcome(energy_change=-18.0, rmsd_change=1.5, success=True)
        
        consciousness = ConsciousnessCoordinates(9.5, 0.65, 777888999)
        behavioral = BehavioralStateData.from_consciousness(9.5, 0.65)
        
        memory = memory_system.create_memory_from_outcome(
            outcome,
            consciousness,
            behavioral,
            qcpp_metrics=qcpp_metrics
        )
        
        # Significance should be combination of base (70%) and QCPP (30%)
        assert 0.0 <= memory.significance <= 1.0
        assert isinstance(memory, QCPPValidatedMemory)  # Ensure it's the right type
        assert memory.qcpp_significance > 0.0
        
        # Calculate expected range
        base_sig = memory_system._calculate_significance(outcome)
        qcpp_sig = memory_system._calculate_qcpp_significance(qcpp_metrics, outcome)
        expected_sig = base_sig * 0.7 + qcpp_sig * 0.3
        
        assert abs(memory.significance - expected_sig) < 0.01


class TestMemoryStorageAndRetrieval:
    """Test suite for storing and retrieving QCPP-validated memories."""
    
    def test_store_qcpp_memory_above_threshold(self):
        """Test storing QCPP-validated memory above significance threshold."""
        memory_system = MemorySystem()
        
        qcpp_metrics = QCPPMetrics(6.0, 0.8, 2.2, 0.9, 1.6)
        
        consciousness = ConsciousnessCoordinates(10.0, 0.75, int(time.time() * 1000))
        behavioral = BehavioralStateData.from_consciousness(10.0, 0.75)
        
        memory = QCPPValidatedMemory(
            memory_id="store-test-1",
            move_type="PHI_PATTERNED",
            significance=0.85,  # Above typical threshold
            energy_change=-28.0,
            rmsd_change=2.2,
            success=True,
            timestamp=int(time.time() * 1000),
            consciousness_state=consciousness,
            behavioral_state=behavioral,
            qcpp_metrics=qcpp_metrics,
            qcpp_significance=0.92
        )
        
        memory_system.store_memory(memory)
        
        # Should be stored
        retrieved = memory_system.retrieve_relevant_memories("PHI_PATTERNED", max_count=10)
        assert len(retrieved) == 1
        assert retrieved[0].memory_id == "store-test-1"
    
    def test_retrieve_qcpp_memories_sorted_by_influence(self):
        """Test retrieving QCPP memories sorted by influence weight."""
        memory_system = MemorySystem()
        
        consciousness = ConsciousnessCoordinates(9.0, 0.7, int(time.time() * 1000))
        behavioral = BehavioralStateData.from_consciousness(9.0, 0.7)
        
        # Create multiple memories with different significances
        for i in range(3):
            qcpp_metrics = QCPPMetrics(
                qcp_score=5.0 + i * 0.5,
                field_coherence=0.5 + i * 0.1,
                stability_score=1.5 + i * 0.3,
                phi_match_score=0.7 + i * 0.1,
                calculation_time_ms=1.0 + i * 0.2
            )
            
            memory = QCPPValidatedMemory(
                memory_id=f"retrieve-test-{i}",
                move_type="BACKBONE_ROTATION",
                significance=0.5 + i * 0.15,  # Increasing significance
                energy_change=-15.0 - i * 5.0,
                rmsd_change=1.0 + i * 0.3,
                success=True,
                timestamp=int(time.time() * 1000) - i * 1000,  # Earlier timestamps
                consciousness_state=consciousness,
                behavioral_state=behavioral,
                qcpp_metrics=qcpp_metrics,
                qcpp_significance=0.6 + i * 0.1
            )
            
            memory_system.store_memory(memory)
        
        # Retrieve memories
        retrieved = memory_system.retrieve_relevant_memories("BACKBONE_ROTATION", max_count=10)
        
        assert len(retrieved) == 3
        
        # Should be sorted by influence weight (descending)
        # Higher significance and more recent = higher influence
        influences = [m.get_influence_weight() for m in retrieved]
        assert influences == sorted(influences, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
