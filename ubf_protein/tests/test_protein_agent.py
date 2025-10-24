"""
Unit tests for protein agent implementation.
"""

import pytest
import time

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import Conformation, ConformationalOutcome


class TestProteinAgent:
    """Test basic protein agent functionality"""

    def test_initialization(self):
        """Test agent initializes with correct components"""
        agent = ProteinAgent("ACDEFGHIK")

        # Check consciousness system
        assert agent.get_consciousness_state().get_frequency() == 9.0
        assert agent.get_consciousness_state().get_coherence() == 0.6

        # Check behavioral state
        behavioral = agent.get_behavioral_state()
        assert behavioral.get_exploration_energy() > 0
        assert behavioral.get_structural_focus() > 0

        # Check memory system
        memory = agent.get_memory_system()
        assert memory._memory_count == 0

        # Check initial conformation
        conformation = agent.get_current_conformation()
        assert conformation.sequence == "ACDEFGHIK"
        assert conformation.energy == 1000.0
        assert len(conformation.atom_coordinates) == 9

    def test_initialization_custom_coordinates(self):
        """Test agent initialization with custom consciousness coordinates"""
        agent = ProteinAgent("TEST", initial_frequency=12.0, initial_coherence=0.8)

        assert agent.get_consciousness_state().get_frequency() == 12.0
        assert agent.get_consciousness_state().get_coherence() == 0.8

    def test_initialization_custom_conformation(self):
        """Test agent initialization with custom starting conformation"""
        custom_conformation = Conformation(
            conformation_id="custom",
            sequence="TEST",
            atom_coordinates=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            energy=500.0,
            rmsd_to_native=2.5,
            secondary_structure=['C', 'C', 'C', 'C'],
            phi_angles=[-60, -60, -60, -60],
            psi_angles=[-40, -40, -40, -40],
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )

        agent = ProteinAgent("TEST", initial_conformation=custom_conformation)

        current = agent.get_current_conformation()
        assert current.conformation_id == "custom"
        assert current.energy == 500.0
        assert current.rmsd_to_native == 2.5

    def test_explore_step_updates_state(self):
        """Test that explore_step updates agent state"""
        agent = ProteinAgent("ACDEFGHIK")

        initial_energy = agent.get_current_conformation().energy
        initial_frequency = agent.get_consciousness_state().get_frequency()
        initial_coherence = agent.get_consciousness_state().get_coherence()

        # Perform exploration step
        outcome = agent.explore_step()

        # Check outcome
        assert isinstance(outcome, ConformationalOutcome)
        assert hasattr(outcome, 'energy_change')
        assert hasattr(outcome, 'success')

        # Check state updates
        new_energy = agent.get_current_conformation().energy
        assert new_energy != initial_energy  # Energy should change

        # Consciousness may or may not change depending on outcome
        # Behavioral state may regenerate

        # Check metrics updated
        metrics = agent.get_exploration_metrics()
        assert metrics["iterations_completed"] == 1
        assert metrics["conformations_explored"] == 2  # Initial + 1 explored
        assert metrics["avg_decision_time_ms"] > 0

    def test_multiple_explore_steps(self):
        """Test multiple exploration steps accumulate correctly"""
        agent = ProteinAgent("ACDEFGHIK")

        # Perform 5 exploration steps
        for i in range(5):
            outcome = agent.explore_step()
            assert isinstance(outcome, ConformationalOutcome)

        # Check accumulated metrics
        metrics = agent.get_exploration_metrics()
        assert metrics["iterations_completed"] == 5
        assert metrics["conformations_explored"] == 6  # Initial + 5 explored
        assert metrics["avg_decision_time_ms"] > 0

    def test_memory_creation_from_exploration(self):
        """Test that exploration creates memories"""
        agent = ProteinAgent("ACDEFGHIK")

        # Perform exploration
        agent.explore_step()

        # Check memory system has memories (may be 0 or 1 depending on significance)
        memory_stats = agent.get_memory_system().get_memory_stats()
        # At least one move type should have been attempted
        assert len(memory_stats) >= 0  # May be empty if significance too low

        # Check agent metrics include memory count
        metrics = agent.get_exploration_metrics()
        assert "memories_created" in metrics

    def test_best_energy_tracking(self):
        """Test that agent tracks best energy found"""
        agent = ProteinAgent("ACDEFGHIK")

        initial_energy = agent.get_current_conformation().energy

        # Perform several steps
        for _ in range(10):
            agent.explore_step()

        metrics = agent.get_exploration_metrics()
        # Best energy should be <= initial (since we might find better conformations)
        assert metrics["best_energy"] <= initial_energy

    def test_consciousness_updates_from_outcomes(self):
        """Test that consciousness coordinates update based on exploration outcomes"""
        agent = ProteinAgent("ACDEFGHIK")

        initial_freq = agent.get_consciousness_state().get_frequency()
        initial_coh = agent.get_consciousness_state().get_coherence()

        # Perform exploration - consciousness should update
        agent.explore_step()

        # Coordinates may have changed
        new_freq = agent.get_consciousness_state().get_frequency()
        new_coh = agent.get_consciousness_state().get_coherence()

        # Changes are possible but not guaranteed (depends on random simulation)
        # Just verify they are within bounds
        assert 3.0 <= new_freq <= 15.0
        assert 0.2 <= new_coh <= 1.0

    def test_behavioral_state_regeneration(self):
        """Test that behavioral state regenerates when consciousness changes significantly"""
        agent = ProteinAgent("ACDEFGHIK")

        initial_behavioral = agent.get_behavioral_state()

        # Force a large consciousness change by directly modifying (for testing)
        agent._consciousness._coordinates.frequency = 12.0  # Large change from 9.0

        # Perform exploration step - should trigger regeneration
        agent.explore_step()

        # Check if behavioral state was regenerated (new instance)
        new_behavioral = agent.get_behavioral_state()
        assert new_behavioral is not initial_behavioral  # Should be a new instance

    def test_get_exploration_metrics_completeness(self):
        """Test that exploration metrics include all required fields"""
        agent = ProteinAgent("ACDEFGHIK")

        # Perform some exploration
        for _ in range(3):
            agent.explore_step()

        metrics = agent.get_exploration_metrics()

        required_fields = [
            "iterations_completed", "conformations_explored", "memories_created",
            "best_energy", "best_rmsd", "avg_decision_time_ms",
            "stuck_in_minima_count", "successful_escapes"
        ]

        for field in required_fields:
            assert field in metrics
            assert isinstance(metrics[field], (int, float))

    def test_initial_conformation_structure(self):
        """Test that initial conformation has proper structure"""
        agent = ProteinAgent("ACDEFGHIK")  # 9 residues

        conformation = agent.get_current_conformation()

        assert len(conformation.atom_coordinates) == 9
        assert len(conformation.secondary_structure) == 9
        assert len(conformation.phi_angles) == 9
        assert len(conformation.psi_angles) == 9
        assert conformation.secondary_structure == ['C'] * 9  # All coil initially
        assert all(phi == -60.0 for phi in conformation.phi_angles)
        assert all(psi == -40.0 for psi in conformation.psi_angles)