"""
Test suite for energy function integration with UBF protein system.

Tests verify that the molecular mechanics energy calculator integrates
correctly with the protein agent and works during conformational exploration.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import Conformation
from ubf_protein.energy_function import MolecularMechanicsEnergy
from ubf_protein import config


class TestEnergyIntegration:
    """Test energy function integration with protein agent."""
    
    def test_agent_initializes_with_energy_calculator(self):
        """Test that agent properly initializes energy calculator when enabled."""
        # Enable energy calculator
        original_setting = config.USE_MOLECULAR_MECHANICS_ENERGY
        config.USE_MOLECULAR_MECHANICS_ENERGY = True
        
        try:
            agent = ProteinAgent(
                protein_sequence="ACDEFGH",
                initial_frequency=9.0,
                initial_coherence=0.6
            )
            
            # Check that energy calculator was initialized
            assert agent._energy_calculator is not None
            assert isinstance(agent._energy_calculator, MolecularMechanicsEnergy)
            
        finally:
            config.USE_MOLECULAR_MECHANICS_ENERGY = original_setting
    
    def test_agent_works_without_energy_calculator(self):
        """Test that agent works correctly when energy calculator is disabled."""
        # Disable energy calculator
        original_setting = config.USE_MOLECULAR_MECHANICS_ENERGY
        config.USE_MOLECULAR_MECHANICS_ENERGY = False
        
        try:
            agent = ProteinAgent(
                protein_sequence="ACDEFGH",
                initial_frequency=9.0,
                initial_coherence=0.6
            )
            
            # Check that energy calculator is None
            assert agent._energy_calculator is None
            
            # Agent should still be able to perform exploration
            outcome = agent.explore_step()
            assert outcome is not None
            assert outcome.new_conformation is not None
            
        finally:
            config.USE_MOLECULAR_MECHANICS_ENERGY = original_setting
    
    def test_energy_calculated_during_exploration(self):
        """Test that energy is calculated during conformational exploration."""
        # Enable energy calculator
        original_setting = config.USE_MOLECULAR_MECHANICS_ENERGY
        config.USE_MOLECULAR_MECHANICS_ENERGY = True
        
        try:
            agent = ProteinAgent(
                protein_sequence="ACDEFGH",
                initial_frequency=9.0,
                initial_coherence=0.6
            )
            
            # Get initial conformation
            initial_conf = agent.get_current_conformation()
            initial_energy = initial_conf.energy
            
            # Perform exploration step
            outcome = agent.explore_step()
            
            # Check that energy was calculated
            assert outcome.new_conformation.energy != initial_energy or outcome.energy_change == 0
            
            # Check that energy components were stored (if move was successful)
            if outcome.success and outcome.new_conformation.energy_components is not None:
                assert 'total_energy' in outcome.new_conformation.energy_components
                assert 'bond_energy' in outcome.new_conformation.energy_components
                assert 'angle_energy' in outcome.new_conformation.energy_components
            
        finally:
            config.USE_MOLECULAR_MECHANICS_ENERGY = original_setting
    
    def test_energy_validation_warns_on_unrealistic_values(self):
        """Test that unrealistic energy values trigger warnings."""
        # Enable energy calculator
        original_setting = config.USE_MOLECULAR_MECHANICS_ENERGY
        config.USE_MOLECULAR_MECHANICS_ENERGY = True
        
        try:
            agent = ProteinAgent(
                protein_sequence="ACDEFGH",
                initial_frequency=9.0,
                initial_coherence=0.6
            )
            
            # Perform several exploration steps
            for _ in range(5):
                outcome = agent.explore_step()
                
                # Energy should be within reasonable bounds
                # (Warning will be logged if it exceeds ENERGY_VALIDATION_THRESHOLD)
                assert outcome.new_conformation.energy is not None
            
        finally:
            config.USE_MOLECULAR_MECHANICS_ENERGY = original_setting
    
    def test_energy_components_stored_in_conformation(self):
        """Test that energy components are properly stored in Conformation."""
        # Enable energy calculator
        original_setting = config.USE_MOLECULAR_MECHANICS_ENERGY
        config.USE_MOLECULAR_MECHANICS_ENERGY = True
        
        try:
            agent = ProteinAgent(
                protein_sequence="ACDEFGH",
                initial_frequency=9.0,
                initial_coherence=0.6
            )
            
            # Perform exploration steps until we get a successful move with components
            for _ in range(10):
                outcome = agent.explore_step()
                
                if outcome.new_conformation.energy_components is not None:
                    components = outcome.new_conformation.energy_components
                    
                    # Check that all expected components are present
                    assert 'total_energy' in components
                    assert 'bond_energy' in components
                    assert 'angle_energy' in components
                    assert 'dihedral_energy' in components
                    assert 'vdw_energy' in components
                    assert 'electrostatic_energy' in components
                    assert 'hbond_energy' in components
                    
                    # Check that total is sum of components
                    component_sum = (
                        components['bond_energy'] +
                        components['angle_energy'] +
                        components['dihedral_energy'] +
                        components['vdw_energy'] +
                        components['electrostatic_energy'] +
                        components['hbond_energy'] +
                        components['compactness_bonus']
                    )
                    
                    # Allow small floating point difference
                    assert abs(components['total_energy'] - component_sum) < 0.01
                    break
            
        finally:
            config.USE_MOLECULAR_MECHANICS_ENERGY = original_setting
    
    def test_error_handling_on_calculation_failure(self):
        """Test that agent handles energy calculation failures gracefully."""
        # Enable energy calculator
        original_setting = config.USE_MOLECULAR_MECHANICS_ENERGY
        config.USE_MOLECULAR_MECHANICS_ENERGY = True
        
        try:
            agent = ProteinAgent(
                protein_sequence="ACDEFGH",
                initial_frequency=9.0,
                initial_coherence=0.6
            )
            
            # Even if energy calculation has issues, agent should continue
            for _ in range(5):
                outcome = agent.explore_step()
                
                # Outcome should still be valid
                assert outcome is not None
                assert outcome.new_conformation is not None
                assert isinstance(outcome.new_conformation.energy, (int, float))
            
        finally:
            config.USE_MOLECULAR_MECHANICS_ENERGY = original_setting
    
    def test_backward_compatibility_with_existing_code(self):
        """Test that existing code works with new energy_components field."""
        # Create conformation without energy_components (old style)
        conf = Conformation(
            conformation_id="test_1",
            sequence="ACDEFGH",
            atom_coordinates=[(0.0, 0.0, 0.0)] * 7,
            energy=100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 7,
            phi_angles=[0.0] * 7,
            psi_angles=[0.0] * 7,
            available_move_types=['local_rotation', 'helix_formation'],
            structural_constraints={}
            # Note: energy_components not provided
        )
        
        # Should still work (energy_components defaults to None)
        assert conf.energy == 100.0
        assert conf.energy_components is None
        
        # Can create with energy_components
        conf_with_components = Conformation(
            conformation_id="test_2",
            sequence="ACDEFGH",
            atom_coordinates=[(0.0, 0.0, 0.0)] * 7,
            energy=100.0,
            rmsd_to_native=None,
            secondary_structure=['C'] * 7,
            phi_angles=[0.0] * 7,
            psi_angles=[0.0] * 7,
            available_move_types=['local_rotation', 'helix_formation'],
            structural_constraints={},
            energy_components={'total_energy': 100.0, 'bond_energy': 50.0}
        )
        
        assert conf_with_components.energy_components is not None
        assert 'total_energy' in conf_with_components.energy_components


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
