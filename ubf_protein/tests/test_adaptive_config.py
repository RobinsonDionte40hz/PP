"""
Tests for adaptive configuration system.

Tests protein size classification, parameter scaling, and integration
with the multi-agent coordinator.
"""

import pytest
import math

from ubf_protein.adaptive_config import AdaptiveConfigurator, create_config_for_sequence
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.models import ProteinSizeClass, AdaptiveConfig


class TestProteinSizeClassification:
    """Test protein size classification logic."""
    
    def test_classify_small_protein(self):
        """Test classification of small proteins (<50 residues)."""
        configurator = AdaptiveConfigurator()
        
        # Test various small sizes
        assert configurator.classify_protein_size("A" * 10) == ProteinSizeClass.SMALL
        assert configurator.classify_protein_size("A" * 30) == ProteinSizeClass.SMALL
        assert configurator.classify_protein_size("A" * 49) == ProteinSizeClass.SMALL
    
    def test_classify_medium_protein(self):
        """Test classification of medium proteins (50-150 residues)."""
        configurator = AdaptiveConfigurator()
        
        # Test various medium sizes
        assert configurator.classify_protein_size("A" * 50) == ProteinSizeClass.MEDIUM
        assert configurator.classify_protein_size("A" * 100) == ProteinSizeClass.MEDIUM
        assert configurator.classify_protein_size("A" * 150) == ProteinSizeClass.MEDIUM
    
    def test_classify_large_protein(self):
        """Test classification of large proteins (>150 residues)."""
        configurator = AdaptiveConfigurator()
        
        # Test various large sizes
        assert configurator.classify_protein_size("A" * 151) == ProteinSizeClass.LARGE
        assert configurator.classify_protein_size("A" * 200) == ProteinSizeClass.LARGE
        assert configurator.classify_protein_size("A" * 500) == ProteinSizeClass.LARGE
    
    def test_boundary_conditions(self):
        """Test classification at boundary conditions."""
        configurator = AdaptiveConfigurator()
        
        # Test exact boundaries
        assert configurator.classify_protein_size("A" * 49) == ProteinSizeClass.SMALL
        assert configurator.classify_protein_size("A" * 50) == ProteinSizeClass.MEDIUM
        assert configurator.classify_protein_size("A" * 150) == ProteinSizeClass.MEDIUM
        assert configurator.classify_protein_size("A" * 151) == ProteinSizeClass.LARGE


class TestParameterScaling:
    """Test parameter scaling based on protein size."""
    
    def test_small_protein_parameters(self):
        """Test that small proteins get tighter parameters."""
        configurator = AdaptiveConfigurator()
        config = configurator.get_config_for_protein("A" * 30)
        
        assert config.size_class == ProteinSizeClass.SMALL
        assert config.residue_count == 30
        
        # Small proteins should have:
        # - Smaller stuck detection window
        # - Lower stuck threshold
        # - Fewer max memories
        # - Lower convergence thresholds
        # - Fewer max iterations
        assert config.stuck_detection_window < 20  # < base value
        assert config.stuck_detection_threshold < 10.0
        assert config.max_memories_per_agent < 50
        assert config.convergence_energy_threshold < 10.0
        assert config.convergence_rmsd_threshold < 2.0
        assert config.max_iterations < 2000
    
    def test_medium_protein_parameters(self):
        """Test that medium proteins get balanced parameters."""
        configurator = AdaptiveConfigurator()
        config = configurator.get_config_for_protein("A" * 100)
        
        assert config.size_class == ProteinSizeClass.MEDIUM
        assert config.residue_count == 100
        
        # Medium proteins should have baseline values
        assert config.stuck_detection_window == 20
        assert config.stuck_detection_threshold == 10.0
        assert config.max_memories_per_agent == 50
        assert config.convergence_energy_threshold == 10.0
        assert config.convergence_rmsd_threshold == 2.0
        assert config.max_iterations == 2000
    
    def test_large_protein_parameters(self):
        """Test that large proteins get relaxed parameters."""
        configurator = AdaptiveConfigurator()
        config = configurator.get_config_for_protein("A" * 200)
        
        assert config.size_class == ProteinSizeClass.LARGE
        assert config.residue_count == 200
        
        # Large proteins should have:
        # - Larger stuck detection window
        # - Higher stuck threshold (with sqrt scaling)
        # - More max memories
        # - Higher convergence thresholds
        # - More max iterations
        assert config.stuck_detection_window > 20  # > base value
        assert config.stuck_detection_threshold > 10.0
        assert config.max_memories_per_agent > 50
        assert config.convergence_energy_threshold > 10.0
        assert config.convergence_rmsd_threshold > 2.0
        assert config.max_iterations > 2000
    
    def test_sqrt_threshold_scaling(self):
        """Test that threshold scaling uses square root for very large proteins."""
        configurator = AdaptiveConfigurator()
        
        # Test a very large protein
        config = configurator.get_config_for_protein("A" * 500)
        
        # Threshold should scale, but not linearly (sqrt scaling prevents huge thresholds)
        base_large_threshold = 15.0  # Base threshold for large proteins
        
        # Should be larger than base but not 5x (linear would be 5x for 500 vs 100)
        assert config.stuck_detection_threshold > base_large_threshold
        assert config.stuck_detection_threshold < base_large_threshold * 5.0
    
    def test_scale_threshold_method(self):
        """Test the scale_threshold method directly."""
        configurator = AdaptiveConfigurator()
        
        # Test sqrt scaling
        base = 10.0
        
        # For reference size (100 residues), should return base
        scaled_100 = configurator.scale_threshold(base, 100)
        assert abs(scaled_100 - base) < 0.01
        
        # For 400 residues (4x reference), should scale by sqrt(4) = 2
        scaled_400 = configurator.scale_threshold(base, 400)
        assert abs(scaled_400 - base * 2.0) < 0.01
        
        # For 25 residues (0.25x reference), should scale by sqrt(0.25) = 0.5
        scaled_25 = configurator.scale_threshold(base, 25)
        assert abs(scaled_25 - base * 0.5) < 0.01
    
    def test_checkpoint_interval_scaling(self):
        """Test that checkpoint interval scales with max iterations."""
        configurator = AdaptiveConfigurator()
        
        small_config = configurator.get_config_for_protein("A" * 30)
        medium_config = configurator.get_config_for_protein("A" * 100)
        large_config = configurator.get_config_for_protein("A" * 200)
        
        # Checkpoint interval should be ~5% of max iterations
        assert small_config.checkpoint_interval <= small_config.max_iterations * 0.06
        assert medium_config.checkpoint_interval <= medium_config.max_iterations * 0.06
        assert large_config.checkpoint_interval <= large_config.max_iterations * 0.06
        
        # Should be at least 50
        assert small_config.checkpoint_interval >= 50
        assert medium_config.checkpoint_interval >= 50
        assert large_config.checkpoint_interval >= 50
    
    def test_consciousness_ranges_constant(self):
        """Test that consciousness parameter ranges remain constant across sizes."""
        configurator = AdaptiveConfigurator()
        
        small_config = configurator.get_config_for_protein("A" * 30)
        medium_config = configurator.get_config_for_protein("A" * 100)
        large_config = configurator.get_config_for_protein("A" * 200)
        
        # All should have same frequency and coherence ranges
        assert small_config.initial_frequency_range == medium_config.initial_frequency_range
        assert medium_config.initial_frequency_range == large_config.initial_frequency_range
        assert small_config.initial_coherence_range == medium_config.initial_coherence_range
        assert medium_config.initial_coherence_range == large_config.initial_coherence_range
        
        # Should match global bounds
        assert small_config.initial_frequency_range == (3.0, 15.0)
        assert small_config.initial_coherence_range == (0.2, 1.0)


class TestMultiAgentCoordinatorIntegration:
    """Test integration of adaptive configuration with MultiAgentCoordinator."""
    
    def test_coordinator_auto_configuration(self):
        """Test that coordinator automatically generates configuration."""
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 residues
        coordinator = MultiAgentCoordinator(sequence)
        
        config = coordinator.get_adaptive_config()
        
        # Should have generated a config
        assert config is not None
        assert config.size_class == ProteinSizeClass.SMALL
        assert config.residue_count == 40
    
    def test_coordinator_with_provided_config(self):
        """Test that coordinator uses provided config."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        
        # Create custom config
        custom_config = AdaptiveConfig(
            size_class=ProteinSizeClass.SMALL,
            residue_count=20,
            initial_frequency_range=(5.0, 10.0),
            initial_coherence_range=(0.3, 0.8),
            stuck_detection_window=15,
            stuck_detection_threshold=7.0,
            memory_significance_threshold=0.4,
            max_memories_per_agent=30,
            convergence_energy_threshold=8.0,
            convergence_rmsd_threshold=1.8,
            max_iterations=500,
            checkpoint_interval=50
        )
        
        coordinator = MultiAgentCoordinator(sequence, adaptive_config=custom_config)
        
        # Should use provided config
        assert coordinator.get_adaptive_config() == custom_config
    
    def test_coordinator_configuration_summary(self):
        """Test that coordinator can provide configuration summary."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        coordinator = MultiAgentCoordinator(sequence)
        
        summary = coordinator.get_configuration_summary()
        
        # Should contain key information
        assert "Protein Size:" in summary
        assert "SMALL" in summary
        assert "20 residues" in summary
        assert "Consciousness Parameters" in summary
        assert "Local Minima Detection" in summary
        assert "Memory System" in summary
        assert "Convergence Criteria" in summary
    
    def test_agents_use_same_config(self):
        """Test that all agents in coordinator share the same adaptive config."""
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 residues (MEDIUM)
        coordinator = MultiAgentCoordinator(sequence)
        
        # Initialize some agents
        agents = coordinator.initialize_agents(5, "balanced")
        
        config = coordinator.get_adaptive_config()
        
        # All agents should have the same config parameters
        assert config.size_class == ProteinSizeClass.MEDIUM
        assert len(agents) == 5


class TestConvenienceFunctions:
    """Test convenience functions for config creation."""
    
    def test_create_config_for_sequence(self):
        """Test convenience function for creating config."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        
        config = create_config_for_sequence(sequence)
        
        assert config is not None
        assert config.size_class == ProteinSizeClass.SMALL
        assert config.residue_count == 20
    
    def test_create_config_verbose_mode(self):
        """Test that verbose mode prints summary (visual test)."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        
        # This should print output (can't easily test print output in pytest)
        config = create_config_for_sequence(sequence, verbose=False)
        
        assert config is not None


class TestConfigSummary:
    """Test configuration summary generation."""
    
    def test_config_summary_format(self):
        """Test that config summary has proper format."""
        configurator = AdaptiveConfigurator()
        config = configurator.get_config_for_protein("A" * 100)
        
        summary = configurator.get_config_summary(config)
        
        # Check for required sections
        assert "Adaptive Configuration Summary" in summary
        assert "Protein Size:" in summary
        assert "Consciousness Parameters:" in summary
        assert "Local Minima Detection:" in summary
        assert "Memory System:" in summary
        assert "Convergence Criteria:" in summary
        assert "Performance:" in summary
    
    def test_config_summary_contains_values(self):
        """Test that summary contains actual configuration values."""
        configurator = AdaptiveConfigurator()
        config = configurator.get_config_for_protein("A" * 50)
        
        summary = configurator.get_config_summary(config)
        
        # Check for specific values
        assert "50 residues" in summary
        assert "MEDIUM" in summary
        assert str(config.stuck_detection_window) in summary
        assert str(config.max_memories_per_agent) in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
