"""
Tests for structural validation and error handling.

Tests validation detection, conformation repair, agent resilience,
and memory system error handling.
"""

import pytest
import math
from unittest.mock import Mock, patch

from ubf_protein.structural_validation import StructuralValidation, ValidationResult
from ubf_protein.models import Conformation, AdaptiveConfig, ProteinSizeClass
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.memory_system import MemorySystem


class TestStructuralValidation:
    """Test StructuralValidation functionality."""

    def test_initialization(self):
        """Test validator initializes correctly."""
        validator = StructuralValidation()
        assert validator is not None
        assert validator.IDEAL_BOND_LENGTH == 3.8
        assert validator.MIN_BOND_LENGTH == 1.0
        assert validator.MAX_BOND_LENGTH == 5.0

    def test_validate_valid_conformation(self):
        """Test validation passes for valid conformation."""
        validator = StructuralValidation()
        
        # Create valid conformation with proper bond lengths
        coords = []
        for i in range(10):
            coords.append((i * 3.8, 0.0, 0.0))  # Ideal spacing
        
        conformation = Conformation(
            conformation_id="valid_conf",
            sequence="ACDEFGHIKL",
            atom_coordinates=coords,
            energy=100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 10,
            phi_angles=[-60.0] * 10,
            psi_angles=[-40.0] * 10,
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )
        
        result = validator.validate_conformation(conformation)
        assert result.is_valid
        assert len(result.issues) == 0

    def test_detect_invalid_bond_lengths(self):
        """Test detection of invalid bond lengths."""
        validator = StructuralValidation()
        
        # Create conformation with too-long bonds
        coords = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]  # 10 Å apart - too long
        
        conformation = Conformation(
            conformation_id="invalid_bonds",
            sequence="AC",
            atom_coordinates=coords,
            energy=100.0,
            rmsd_to_native=None,
            secondary_structure=['C', 'C'],
            phi_angles=[-60.0, -60.0],
            psi_angles=[-40.0, -40.0],
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )
        
        result = validator.validate_conformation(conformation)
        assert not result.is_valid
        assert any("Too long" in issue for issue in result.issues)

    def test_detect_steric_clashes(self):
        """Test detection of steric clashes."""
        validator = StructuralValidation()
        
        # Create conformation with clash (residues 0 and 4 too close)
        coords = [
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (0.5, 0.0, 0.0)  # Too close to residue 0
        ]
        
        conformation = Conformation(
            conformation_id="clashing",
            sequence="ACDEF",
            atom_coordinates=coords,
            energy=100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 5,
            phi_angles=[-60.0] * 5,
            psi_angles=[-40.0] * 5,
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )
        
        result = validator.validate_conformation(conformation)
        assert not result.is_valid
        assert any("clash" in issue.lower() for issue in result.issues)

    def test_detect_invalid_coordinates(self):
        """Test detection of invalid coordinates (NaN, inf)."""
        validator = StructuralValidation()
        
        # Create conformation with NaN coordinate
        coords = [(0.0, 0.0, 0.0), (float('nan'), 0.0, 0.0)]
        
        conformation = Conformation(
            conformation_id="nan_coords",
            sequence="AC",
            atom_coordinates=coords,
            energy=100.0,
            rmsd_to_native=None,
            secondary_structure=['C', 'C'],
            phi_angles=[-60.0, -60.0],
            psi_angles=[-40.0, -40.0],
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )
        
        result = validator.validate_conformation(conformation)
        assert not result.is_valid
        assert any("NaN" in issue for issue in result.issues)

    def test_detect_backbone_discontinuity(self):
        """Test detection of backbone breaks."""
        validator = StructuralValidation()
        
        # Create conformation with gap
        coords = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)]  # Huge gap
        
        conformation = Conformation(
            conformation_id="broken_backbone",
            sequence="AC",
            atom_coordinates=coords,
            energy=100.0,
            rmsd_to_native=None,
            secondary_structure=['C', 'C'],
            phi_angles=[-60.0, -60.0],
            psi_angles=[-40.0, -40.0],
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )
        
        result = validator.validate_conformation(conformation)
        assert not result.is_valid
        assert any("break" in issue.lower() for issue in result.issues)

    def test_repair_bond_lengths(self):
        """Test repair of invalid bond lengths."""
        validator = StructuralValidation()
        
        # Create conformation with too-long bonds
        coords = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        
        conformation = Conformation(
            conformation_id="to_repair",
            sequence="AC",
            atom_coordinates=coords,
            energy=100.0,
            rmsd_to_native=None,
            secondary_structure=['C', 'C'],
            phi_angles=[-60.0, -60.0],
            psi_angles=[-40.0, -40.0],
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )
        
        repaired, success = validator.repair_conformation(conformation)
        
        # Check repair was successful
        assert success or len(repaired.atom_coordinates) == 2
        
        # Check bond length is closer to ideal
        x1, y1, z1 = repaired.atom_coordinates[0]
        x2, y2, z2 = repaired.atom_coordinates[1]
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        assert 3.0 < dist < 5.0  # Should be closer to ideal

    def test_distance_calculation(self):
        """Test distance calculation helper."""
        validator = StructuralValidation()
        
        coord1 = (0.0, 0.0, 0.0)
        coord2 = (3.0, 4.0, 0.0)
        
        dist = validator._calculate_distance(coord1, coord2)
        assert abs(dist - 5.0) < 0.001  # 3-4-5 triangle


class TestProteinAgentErrorHandling:
    """Test ProteinAgent error handling and resilience."""

    def test_agent_handles_validation_failure(self):
        """Test agent continues after validation failure."""
        config = AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=20,
            initial_frequency_range=(3.0, 15.0),
            initial_coherence_range=(0.2, 1.0),
            stuck_detection_window=10,
            stuck_detection_threshold=5.0,
            memory_significance_threshold=0.3,
            max_memories_per_agent=50,
            convergence_energy_threshold=10.0,
            convergence_rmsd_threshold=2.0,
            max_iterations=1000,
            checkpoint_interval=100
        )
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGHIKLMNPQRSTVWY",
            adaptive_config=config
        )
        
        # Run several exploration steps
        for _ in range(5):
            outcome = agent.explore_step()
            # Agent should not crash
            assert outcome is not None
        
        # Check metrics include validation stats
        metrics = agent.get_exploration_metrics()
        assert 'validation_failures' in metrics
        assert 'repair_attempts' in metrics
        assert 'repair_successes' in metrics

    @patch('ubf_protein.mapless_moves.MaplessMoveGenerator')
    def test_agent_handles_move_generation_error(self, mock_generator):
        """Test agent handles errors during move generation."""
        # Mock generator to raise exception
        mock_generator.return_value.generate_moves.side_effect = Exception("Generation error")
        
        config = AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=20,
            initial_frequency_range=(3.0, 15.0),
            initial_coherence_range=(0.2, 1.0),
            stuck_detection_window=10,
            stuck_detection_threshold=5.0,
            memory_significance_threshold=0.3,
            max_memories_per_agent=50,
            convergence_energy_threshold=10.0,
            convergence_rmsd_threshold=2.0,
            max_iterations=1000,
            checkpoint_interval=100
        )
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGHIKLMNPQRSTVWY",
            adaptive_config=config
        )
        
        # Should not crash, should return minimal outcome
        outcome = agent.explore_step()
        assert outcome is not None
        assert not outcome.success

    def test_agent_tracks_validation_metrics(self):
        """Test agent properly tracks validation and repair metrics."""
        config = AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=20,
            initial_frequency_range=(3.0, 15.0),
            initial_coherence_range=(0.2, 1.0),
            stuck_detection_window=10,
            stuck_detection_threshold=5.0,
            memory_significance_threshold=0.3,
            max_memories_per_agent=50,
            convergence_energy_threshold=10.0,
            convergence_rmsd_threshold=2.0,
            max_iterations=1000,
            checkpoint_interval=100
        )
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGHIKLMNPQRSTVWY",
            adaptive_config=config
        )
        
        initial_metrics = agent.get_exploration_metrics()
        assert initial_metrics['validation_failures'] == 0
        assert initial_metrics['repair_attempts'] == 0
        assert initial_metrics['repair_successes'] == 0


class TestMemorySystemErrorHandling:
    """Test MemorySystem error handling."""

    def test_memory_system_handles_invalid_memory(self):
        """Test memory system handles invalid memory objects gracefully."""
        memory_system = MemorySystem()
        
        # Try to store invalid memory (missing attributes)
        invalid_memory = Mock()
        invalid_memory.significance = None  # Invalid
        
        # Should not crash
        memory_system.store_memory(invalid_memory)
        
        # Memory count should be 0
        stats = memory_system.get_memory_stats()
        assert all(count == 0 for count in stats.values())

    def test_memory_system_handles_retrieval_error(self):
        """Test memory system handles retrieval errors gracefully."""
        memory_system = MemorySystem()
        
        # Mock a memory that will cause error on get_influence_weight
        bad_memory = Mock()
        bad_memory.move_type = "test_move"
        bad_memory.significance = 0.8
        bad_memory.get_influence_weight.side_effect = Exception("Weight error")
        
        # Store the memory
        memory_system._memories["test_move"].append(bad_memory)
        memory_system._memory_count += 1
        
        # Retrieve should return empty list on error
        memories = memory_system.retrieve_relevant_memories("test_move")
        assert isinstance(memories, list)
        # May be empty or may work depending on implementation

    def test_memory_system_continues_after_store_error(self):
        """Test memory system continues execution after storage error."""
        memory_system = MemorySystem()
        
        # Create valid memory
        from ubf_protein.models import ConformationalMemory, ConsciousnessCoordinates, BehavioralStateData
        
        memory = ConformationalMemory(
            memory_id="mem_1",
            move_type="backbone_rotation",
            significance=0.8,
            energy_change=-10.0,
            rmsd_change=-0.5,
            success=True,
            timestamp=1000,
            consciousness_state=ConsciousnessCoordinates(8.0, 0.6, 1000),
            behavioral_state=BehavioralStateData(0.5, 0.6, 0.5, 0.4, 0.6, 0.8, 1000)
        )
        
        # Store should succeed
        memory_system.store_memory(memory)
        
        stats = memory_system.get_memory_stats()
        assert stats.get("backbone_rotation", 0) >= 1


class TestValidationIntegration:
    """Test integration of validation with agent exploration."""

    def test_validation_improves_conformation_quality(self):
        """Test that validation and repair improve conformation quality."""
        validator = StructuralValidation()
        
        # Create severely invalid conformation
        coords = []
        for i in range(10):
            # Random-ish coordinates with issues
            coords.append((i * 8.0, 0.0, 0.0))  # Too far apart
        
        conformation = Conformation(
            conformation_id="invalid",
            sequence="ACDEFGHIKL",
            atom_coordinates=coords,
            energy=1000.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 10,
            phi_angles=[-60.0] * 10,
            psi_angles=[-40.0] * 10,
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )
        
        # Validate - should fail
        initial_result = validator.validate_conformation(conformation)
        assert not initial_result.is_valid
        
        # Repair
        repaired, success = validator.repair_conformation(conformation)
        
        # Validate repaired - should have fewer issues
        final_result = validator.validate_conformation(repaired)
        assert len(final_result.issues) < len(initial_result.issues)
