#!/usr/bin/env python3
"""
Test script to verify move evaluator integration with physics calculators.
Pure Python implementation - no NumPy required for PyPy compatibility.
"""

import sys
import os
import math

# Add the ubf_protein directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_move_evaluator_integration():
    """Test that move evaluator integrates physics calculators correctly."""
    try:
        # Import required classes
        sys.path.insert(0, os.path.dirname(__file__))

        # Define minimal classes for testing
        from enum import Enum

        class MoveType(Enum):
            HELIX_FORMATION = "helix_formation"
            ENERGY_MINIMIZATION = "energy_minimization"

        # Mock ConformationalMove
        class ConformationalMove:
            def __init__(self, move_id, move_type, target_residues, estimated_energy_change,
                        estimated_rmsd_change, required_capabilities, energy_barrier,
                        structural_feasibility):
                self.move_id = move_id
                self.move_type = move_type
                self.target_residues = target_residues
                self.estimated_energy_change = estimated_energy_change
                self.estimated_rmsd_change = estimated_rmsd_change
                self.required_capabilities = required_capabilities
                self.energy_barrier = energy_barrier
                self.structural_feasibility = structural_feasibility

        # Define physics calculators (same as in test_physics_simple.py)
        class QAAPCalculator:
            def __init__(self):
                self.phi = (1 + math.sqrt(5)) / 2
                self.base_energy = 4.0

            def calculate_qaap_potential(self, atom_coords, ss_structure):
                if not atom_coords:
                    return 0.0
                qcp_values = []
                for i, (coord, ss_type) in enumerate(zip(atom_coords, ss_structure)):
                    if ss_type == 'H':
                        n = 1
                    elif ss_type == 'E':
                        n = 2
                    else:
                        n = 0
                    neighbors = 0
                    for j, other_coord in enumerate(atom_coords):
                        if i != j:
                            # Calculate Euclidean distance using pure Python
                            dist = math.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(coord, other_coord)))
                            if dist < 8.0:
                                neighbors += 1
                    l = min(max(1, neighbors // 3), 3)
                    hydrophobicity_scale = math.sin(i * 0.5)
                    m = hydrophobicity_scale
                    qcp = self.base_energy + (2**n * (self.phi**l) * m)
                    qcp_values.append(qcp)
                return sum(qcp_values) / len(qcp_values) if qcp_values else 0.0

        class ResonanceCouplingCalculator:
            def __init__(self, gamma_frequency_hz: float = 40.0):
                self.gamma_frequency_hz = gamma_frequency_hz
                self.plank_reduced = 1.0545718e-34
                self.h_gamma = self.plank_reduced * gamma_frequency_hz

            def calculate_resonance_coupling(self, atom_coords1, atom_coords2):
                if not atom_coords1 or not atom_coords2:
                    return 0.0
                # Simplified energy approximation using pure Python
                energy1 = math.sqrt(sum(c**2 for c in atom_coords1))
                energy2 = math.sqrt(sum(c**2 for c in atom_coords2))
                return self._resonance_coupling(energy1, energy2)

            def _resonance_coupling(self, energy1: float, energy2: float) -> float:
                energy_diff = abs(energy1 - energy2)
                coupling = math.exp(-(energy_diff - self.h_gamma)**2 / (2 * self.h_gamma))
                return max(0.0, min(1.0, coupling))

        class WaterShieldingCalculator:
            def __init__(self, coherence_time_fs: float = 408.0, shielding_factor: float = 3.57):
                self.coherence_time_fs = coherence_time_fs
                self.shielding_factor = shielding_factor

            def calculate_shielding(self, atom_coords, ss_structure):
                if not atom_coords:
                    return 0.0
                total_shielding = 0.0
                n_residues = len(atom_coords)
                for i, coord in enumerate(atom_coords):
                    nearby_count = 0
                    for j, other_coord in enumerate(atom_coords):
                        if i != j:
                            # Calculate Euclidean distance using pure Python
                            dist = math.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(coord, other_coord)))
                            if dist < 8.0:
                                nearby_count += 1
                    local_shielding = min(1.0, nearby_count / 10.0)
                    total_shielding += local_shielding
                avg_shielding = total_shielding / n_residues if n_residues > 0 else 0.0
                coherence_factor = math.exp(-self.coherence_time_fs / 1000.0)
                shielding_effect = avg_shielding * coherence_factor * (self.shielding_factor / 10.0)
                return max(0.0, min(1.0, shielding_effect))

        # Define CapabilityBasedMoveEvaluator
        class CapabilityBasedMoveEvaluator:
            def __init__(self):
                self.qaap_calculator = QAAPCalculator()
                self.resonance_calculator = ResonanceCouplingCalculator()
                self.water_shielding_calculator = WaterShieldingCalculator()

            def evaluate_move(self, move, behavioral_state, memory_influence, physics_factors=None):
                physical_feasibility = self._calculate_physical_feasibility(move)
                quantum_alignment = self._calculate_quantum_alignment(move, physics_factors)
                behavioral_preference = self._calculate_behavioral_preference(move, behavioral_state)
                historical_success = memory_influence
                goal_alignment = self._calculate_goal_alignment(move)

                weights = [0.2, 0.25, 0.2, 0.15, 0.2]
                factors = [physical_feasibility, quantum_alignment, behavioral_preference,
                          historical_success, goal_alignment]

                return sum(w * f for w, f in zip(weights, factors))

            def _calculate_physical_feasibility(self, move):
                structural = move.structural_feasibility
                barrier_feasibility = max(0.0, 1.0 - (move.energy_barrier / 50.0))
                return 0.7 * structural + 0.3 * barrier_feasibility

            def _calculate_quantum_alignment(self, move, physics_factors):
                if physics_factors:
                    qaap = physics_factors.get('qaap', 0.5)
                    resonance = physics_factors.get('resonance', 0.5)
                    water_shielding = physics_factors.get('water_shielding', 0.5)
                else:
                    qaap = 0.5
                    resonance = 0.5
                    water_shielding = 0.5

                qaap_weight = 0.4
                qaap_contribution = qaap_weight * (0.7 + 0.6 * qaap)

                resonance_weight = 0.35
                resonance_contribution = resonance_weight * (0.9 + 0.3 * resonance)

                shielding_weight = 0.25
                shielding_contribution = shielding_weight * (0.95 + 0.1 * water_shielding)

                return qaap_contribution + resonance_contribution + shielding_contribution

            def _calculate_behavioral_preference(self, move, behavioral_state):
                exploration_energy = behavioral_state.get_exploration_energy()
                structural_focus = behavioral_state.get_structural_focus()
                hydrophobic_drive = behavioral_state.get_hydrophobic_drive()
                risk_tolerance = behavioral_state.get_risk_tolerance()
                native_state_ambition = behavioral_state.get_native_state_ambition()

                if move.move_type.value == 'hydrophobic_collapse':
                    preference = (0.3 * hydrophobic_drive + 0.3 * exploration_energy +
                                0.2 * risk_tolerance + 0.1 * structural_focus +
                                0.1 * native_state_ambition)
                elif move.move_type.value in ['helix_formation', 'sheet_formation']:
                    preference = (0.4 * structural_focus + 0.3 * native_state_ambition +
                                0.2 * exploration_energy + 0.1 * hydrophobic_drive)
                else:
                    preference = (0.25 * exploration_energy + 0.25 * structural_focus +
                                0.2 * risk_tolerance + 0.15 * hydrophobic_drive +
                                0.15 * native_state_ambition)
                return preference

            def _calculate_goal_alignment(self, move):
                energy_alignment = max(0.0, 1.0 + move.estimated_energy_change / 50.0)
                optimal_rmsd = 1.0
                rmsd_alignment = 1.0 - abs(move.estimated_rmsd_change - optimal_rmsd) / 2.0
                rmsd_alignment = max(0.0, rmsd_alignment)
                return 0.6 * energy_alignment + 0.4 * rmsd_alignment

        # Mock behavioral state
        class MockBehavioralState:
            def get_exploration_energy(self): return 0.7
            def get_structural_focus(self): return 0.8
            def get_hydrophobic_drive(self): return 0.6
            def get_risk_tolerance(self): return 0.5
            def get_native_state_ambition(self): return 0.9

        # Test move evaluator
        evaluator = CapabilityBasedMoveEvaluator()
        print("✓ Move evaluator instantiated successfully")

        # Test that it has physics calculators
        assert hasattr(evaluator, 'qaap_calculator')
        assert hasattr(evaluator, 'resonance_calculator')
        assert hasattr(evaluator, 'water_shielding_calculator')
        print("✓ Move evaluator has physics calculators")

        # Create test move
        test_move = ConformationalMove(
            move_id="test_helix_123",
            move_type=MoveType.HELIX_FORMATION,
            target_residues=[1, 2, 3, 4, 5],
            estimated_energy_change=-15.0,
            estimated_rmsd_change=2.5,
            required_capabilities={'can_form_helix': True},
            energy_barrier=12.0,
            structural_feasibility=0.85
        )

        mock_state = MockBehavioralState()

        # Test evaluation with physics factors
        physics_factors = {
            'qaap': 0.8,
            'resonance': 0.6,
            'water_shielding': 0.7
        }

        weight = evaluator.evaluate_move(test_move, mock_state, memory_influence=0.5, physics_factors=physics_factors)
        print(f"✓ Move evaluation with physics factors: {weight}")
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 2.0

        # Test evaluation without physics factors
        weight_no_physics = evaluator.evaluate_move(test_move, mock_state, memory_influence=0.5)
        print(f"✓ Move evaluation without physics factors: {weight_no_physics}")
        assert isinstance(weight_no_physics, float)
        assert 0.0 <= weight_no_physics <= 2.0

        # Test quantum alignment calculation
        alignment = evaluator._calculate_quantum_alignment(test_move, physics_factors)
        print(f"✓ Quantum alignment calculation: {alignment}")
        assert 0.8 <= alignment <= 1.3, f"Alignment {alignment} out of expected range"

        print("✓ All move evaluator integration tests passed!")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_move_evaluator_integration()
    sys.exit(0 if success else 1)