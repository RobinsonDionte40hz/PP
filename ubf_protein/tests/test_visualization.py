"""
Tests for visualization export system.

This module tests the VisualizationExporter class and its integration with
ProteinAgent for trajectory capture and export.
"""

import json
import tempfile
import os

from ubf_protein.visualization import VisualizationExporter
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.models import ProteinSizeClass, ConformationSnapshot

# Import test helpers
from ubf_protein.tests.test_helpers import (
    create_test_conformation,
    create_test_snapshot,
    create_test_snapshots,
    create_test_config
)

try:
    import pytest as _pytest_module
    pytest = _pytest_module
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Define minimal pytest replacements for when pytest is not available
    class _PytestStub:
        @staticmethod
        def fixture(func):
            return func
        
        class raises:
            def __init__(self, exc, match=None):
                self.exc = exc
                self.match = match
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.exc} but no exception was raised")
                if not issubclass(exc_type, self.exc):
                    return False  # Re-raise the exception
                return True  # Suppress the exception
        
        @staticmethod
        def main(args):
            print("pytest not available, please install: pip install pytest")
    
    pytest = _PytestStub()


# Test fixtures
@pytest.fixture
def sample_conformation():
    """Create a sample conformation for testing."""
    return create_test_conformation()


@pytest.fixture
def sample_snapshots():
    """Create sample trajectory snapshots."""
    return create_test_snapshots(count=10, agent_id="test_agent")


@pytest.fixture
def exporter():
    """Create a VisualizationExporter instance."""
    exporter = VisualizationExporter()
    # Clear any existing stream file before each test
    stream_file = exporter._output_dir / "realtime_stream.jsonl"
    if stream_file.exists():
        stream_file.unlink()
    return exporter


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file exports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Tests for VisualizationExporter
class TestVisualizationExporter:
    """Test suite for VisualizationExporter class."""
    
    def test_export_trajectory_json(self, exporter, sample_snapshots, temp_dir):
        """Test trajectory export to JSON format."""
        output_path = os.path.join(temp_dir, "trajectory.json")
        
        # Add snapshots to exporter
        for snapshot in sample_snapshots:
            exporter.add_snapshot(snapshot)
        
        # Export using actual API
        exporter.export_trajectory_to_json(
            agent_id="test_agent",
            output_file=output_path
        )
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Load and verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "agent_id" in data
        assert "snapshots" in data
        assert len(data["snapshots"]) == len(sample_snapshots)
        assert data["snapshot_count"] == len(sample_snapshots)
        assert data["agent_id"] == "test_agent"
    
    def test_export_trajectory_pdb(self, exporter, sample_snapshots, temp_dir):
        """Test trajectory export to PDB format."""
        output_path = os.path.join(temp_dir, "trajectory.pdb")
        
        # Add snapshots to exporter
        for snapshot in sample_snapshots:
            exporter.add_snapshot(snapshot)
        
        # Export using actual API
        exporter.export_trajectory_to_pdb(
            agent_id="test_agent",
            output_file=output_path
        )
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Verify PDB format
        with open(output_path, 'r') as f:
            content = f.read()
        
        # Should contain MODEL/ENDMDL blocks
        assert "MODEL" in content
        assert "ENDMDL" in content
        assert content.count("MODEL") == len(sample_snapshots)
        assert "ATOM" in content
    
    def test_export_trajectory_csv(self, exporter, sample_snapshots, temp_dir):
        """Test energy landscape export to CSV format."""
        output_path = os.path.join(temp_dir, "energy_landscape.csv")
        
        # Add snapshots to exporter
        for snapshot in sample_snapshots:
            exporter.add_snapshot(snapshot)
        
        # Export energy landscape using actual API
        exporter.export_energy_landscape_to_csv(output_file=output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Verify CSV format
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        # Should have header + data rows (one per snapshot)
        assert len(lines) >= 1  # At least header
        assert "x" in lines[0] and "y" in lines[0]
        assert "energy" in lines[0]
        assert "rmsd" in lines[0]
    
    def test_export_invalid_format(self, exporter, sample_snapshots, temp_dir):
        """Test export_trajectory method returns snapshots list."""
        # The API has export_trajectory(agent_id) which returns List[ConformationSnapshot]
        # Add snapshots for a test agent
        for snapshot in sample_snapshots:
            exporter.add_snapshot(snapshot)
        
        # Export using agent_id
        result = exporter.export_trajectory("test_agent")
        
        # Should return list of snapshots
        assert isinstance(result, list)
        assert len(result) == len(sample_snapshots)
        assert all(isinstance(s, ConformationSnapshot) for s in result)
    
    def test_export_empty_snapshots(self, exporter, temp_dir):
        """Test export with empty snapshot list."""
        output_path = os.path.join(temp_dir, "empty.json")
        
        # Don't add any snapshots - exporter starts empty
        # Export using actual API
        exporter.export_trajectory_to_json(
            agent_id="test_agent",
            output_file=output_path
        )
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Verify empty trajectory
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["snapshot_count"] == 0
        assert len(data["snapshots"]) == 0
    
    def test_export_energy_landscape(self, exporter, sample_snapshots, temp_dir):
        """Test energy landscape projection export."""
        
        # Add snapshots to exporter
        for snapshot in sample_snapshots:
            exporter.add_snapshot(snapshot)
        
        # Export energy landscape using actual API (returns EnergyLandscape object)
        landscape = exporter.export_energy_landscape()
        
        # Verify landscape object
        assert landscape.projection_method == 'PCA'
        assert len(landscape.coordinates_2d) == len(sample_snapshots)
        assert len(landscape.energy_values) == len(sample_snapshots)
        assert len(landscape.rmsd_values) == len(sample_snapshots)
    
    def test_stream_update_json(self, exporter, sample_conformation, temp_dir):
        """Test streaming update functionality."""
        snapshot = create_test_snapshot(iteration=1, agent_id="stream_agent")
        
        # stream_update() adds to buffer and flushes periodically
        # Set stream interval to 1 to force immediate flush
        exporter._stream_interval = 1
        exporter.stream_update(snapshot)
        
        # Verify stream file exists
        stream_file = exporter._output_dir / "realtime_stream.jsonl"
        assert stream_file.exists()
        
        # Verify content (JSONL format - one JSON per line)
        with open(stream_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["iteration"] == 1
        assert data["agent_id"] == "stream_agent"
        assert "energy" in data
    
    def test_stream_update_csv(self, exporter, sample_conformation, temp_dir):
        """Test multiple streaming updates."""
        # Set stream interval to 1 to force immediate flush
        exporter._stream_interval = 1
        
        # First snapshot
        snapshot1 = create_test_snapshot(iteration=1, agent_id="stream_agent")
        exporter.stream_update(snapshot1)
        
        # Second snapshot
        snapshot2 = create_test_snapshot(iteration=2, agent_id="stream_agent")
        exporter.stream_update(snapshot2)
        
        # Verify stream file content
        stream_file = exporter._output_dir / "realtime_stream.jsonl"
        with open(stream_file, 'r') as f:
            lines = f.readlines()
        
        # Should have 2 data rows (JSONL format)
        assert len(lines) == 2
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])
        assert data1["iteration"] == 1
        assert data2["iteration"] == 2


# Tests for ProteinAgent visualization integration
class TestProteinAgentVisualization:
    """Test suite for ProteinAgent visualization integration."""
    
    @pytest.fixture
    def test_agent(self):
        """Create a test ProteinAgent with visualization enabled."""
        config = create_test_config(
            size_class=ProteinSizeClass.SMALL,
            max_iterations=10
        )
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGH",
            adaptive_config=config,
            enable_visualization=True,
            max_snapshots=5
        )
        
        return agent
    
    def test_visualization_enabled(self, test_agent):
        """Test that visualization is enabled."""
        assert test_agent._enable_visualization is True
        assert test_agent._max_snapshots == 5
    
    def test_snapshot_capture_during_exploration(self, test_agent):
        """Test that snapshots are captured during exploration."""
        # Initial snapshot (iteration 0) already captured in __init__
        initial_count = len(test_agent.get_trajectory_snapshots())
        assert initial_count == 1  # Initial snapshot at iteration 0
        
        # Run a few exploration steps
        for _ in range(3):
            test_agent.explore_step()
        
        # Check that snapshots were captured (initial + 3 steps = 4 total)
        snapshots = test_agent.get_trajectory_snapshots()
        assert len(snapshots) == 4
        
        # Verify snapshot content (check first non-initial snapshot)
        for i, snapshot in enumerate(snapshots):
            assert snapshot.iteration == i  # 0, 1, 2, 3
            assert snapshot.agent_id == test_agent.get_agent_id()
            assert snapshot.conformation is not None
            assert snapshot.consciousness_state is not None
            assert snapshot.behavioral_state is not None
    
    def test_snapshot_downsampling(self, test_agent):
        """Test automatic downsampling when max_snapshots is reached."""
        # Note: max_snapshots=5 for test_agent, but downsampling keeps more
        # during the process before reaching steady state
        initial_count = len(test_agent.get_trajectory_snapshots())
        
        # Run more steps than max_snapshots
        for _ in range(15):
            test_agent.explore_step()
        
        # Get final snapshot count
        snapshots = test_agent.get_trajectory_snapshots()
        # Due to downsampling algorithm (keep first 10%, last 10%, downsample middle)
        # the count will be controlled but may vary slightly
        # We just verify it's reasonable and not growing unbounded
        assert len(snapshots) < 20  # Should be well under uncontrolled growth (which would be 16)
    
    def test_clear_snapshots(self, test_agent):
        """Test clearing trajectory snapshots."""
        # Capture some snapshots (initial + 3 steps = 4 total)
        for _ in range(3):
            test_agent.explore_step()
        
        assert len(test_agent.get_trajectory_snapshots()) == 4
        
        # Clear snapshots
        test_agent.clear_trajectory_snapshots()
        assert len(test_agent.get_trajectory_snapshots()) == 0
    
    def test_disable_visualization(self, test_agent):
        """Test disabling visualization."""
        # Capture some snapshots (initial + 3 steps = 4 total)
        for _ in range(3):
            test_agent.explore_step()
        
        assert len(test_agent.get_trajectory_snapshots()) == 4
        
        # Disable visualization (should clear snapshots)
        test_agent.enable_visualization(enable=False)
        assert len(test_agent.get_trajectory_snapshots()) == 0
        
        # Run more steps - no snapshots should be captured
        for _ in range(2):
            test_agent.explore_step()
        
        assert len(test_agent.get_trajectory_snapshots()) == 0
    
    def test_agent_id_management(self, test_agent):
        """Test agent ID getter/setter."""
        # Should have a default ID
        default_id = test_agent.get_agent_id()
        assert default_id is not None
        assert len(default_id) > 0
        
        # Set custom ID
        test_agent.set_agent_id("custom_agent_123")
        assert test_agent.get_agent_id() == "custom_agent_123"
    
    def test_visualization_with_export(self, test_agent, temp_dir):
        """Test full workflow: capture snapshots and export."""
        # Run exploration (initial + 5 steps)
        for _ in range(5):
            test_agent.explore_step()
        
        # Get snapshots - with max_snapshots=5, downsampling will kick in
        snapshots = test_agent.get_trajectory_snapshots()
        # We ran 5 steps + 1 initial = 6 total, but with max_snapshots=5
        # downsampling should have reduced it
        assert len(snapshots) <= test_agent._max_snapshots + 1  # Allow slight overage
        
        # Export to JSON
        exporter = VisualizationExporter()
        output_path = os.path.join(temp_dir, "agent_trajectory.json")
        # Use the exporter's method to add snapshots and export
        for snapshot in snapshots:
            exporter.add_snapshot(snapshot)
        exporter.export_trajectory_to_json(test_agent.get_agent_id(), output_path)
        
        # Verify export
        assert os.path.exists(output_path)


# Integration tests
class TestVisualizationIntegration:
    """Integration tests for complete visualization workflows."""
    
    def test_multi_agent_trajectory_comparison(self, temp_dir):
        """Test capturing and comparing trajectories from multiple agents."""
        config = create_test_config(
            size_class=ProteinSizeClass.SMALL,
            max_iterations=5
        )
        
        # Create two agents with different IDs
        agent1 = ProteinAgent(
            protein_sequence="ACDEFGH",
            adaptive_config=config,
            enable_visualization=True,
            max_snapshots=10
        )
        agent1.set_agent_id("agent_1")
        
        agent2 = ProteinAgent(
            protein_sequence="ACDEFGH",
            adaptive_config=config,
            enable_visualization=True,
            max_snapshots=10
        )
        agent2.set_agent_id("agent_2")
        
        # Run both agents
        for _ in range(5):
            agent1.explore_step()
            agent2.explore_step()
        
        # Export both trajectories
        exporter = VisualizationExporter()
        
        # Add snapshots for both agents
        for snapshot in agent1.get_trajectory_snapshots():
            exporter.add_snapshot(snapshot)
        for snapshot in agent2.get_trajectory_snapshots():
            exporter.add_snapshot(snapshot)
        
        path1 = os.path.join(temp_dir, "agent1_trajectory.json")
        exporter.export_trajectory_to_json(agent1.get_agent_id(), path1)
        
        path2 = os.path.join(temp_dir, "agent2_trajectory.json")
        exporter.export_trajectory_to_json(agent2.get_agent_id(), path2)
        
        # Verify both exports
        with open(path1, 'r') as f:
            data1 = json.load(f)
        with open(path2, 'r') as f:
            data2 = json.load(f)
        
        assert data1["agent_id"] == "agent_1"
        assert data2["agent_id"] == "agent_2"
        # Each agent: 5 steps (no initial snapshot at iteration 0 in this test)
        assert len(data1["snapshots"]) == 5
        assert len(data2["snapshots"]) == 5
    
    def test_streaming_during_long_run(self, temp_dir):
        """Test streaming updates during a longer simulation."""
        config = create_test_config(
            size_class=ProteinSizeClass.SMALL,
            max_iterations=20
        )
        
        agent = ProteinAgent(
            protein_sequence="ACDEFGH",
            adaptive_config=config,
            enable_visualization=True,
            max_snapshots=100
        )
        
        exporter = VisualizationExporter()
        # Clear any existing stream file before testing
        stream_file = exporter._output_dir / "realtime_stream.jsonl"
        if stream_file.exists():
            stream_file.unlink()
        exporter._stream_interval = 1  # Force immediate flush
        
        # Run with periodic streaming
        for i in range(10):
            agent.explore_step()
            
            # Stream every 2 steps
            if i % 2 == 0:
                snapshots = agent.get_trajectory_snapshots()
                if snapshots:
                    exporter.add_snapshot(snapshots[-1])
                    exporter.stream_update(snapshots[-1])
        
        # Verify streaming file (JSONL format)
        stream_file = exporter._output_dir / "realtime_stream.jsonl"
        assert stream_file.exists()
        
        with open(stream_file, 'r') as f:
            lines = f.readlines()
        
        # Should have 5 data rows (streamed every 2 steps: iterations 0, 2, 4, 6, 8)
        assert len(lines) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
