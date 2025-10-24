"""
Tests for checkpoint and resume system.

This module tests the CheckpointManager class and its integration with
MultiAgentCoordinator for state serialization and recovery.
"""

import os
import json
import tempfile
import time
from pathlib import Path

from ubf_protein.checkpoint import CheckpointManager
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.protein_agent import ProteinAgent
from ubf_protein.memory_system import SharedMemoryPool
from ubf_protein.tests.test_helpers import create_test_config
from ubf_protein.models import ProteinSizeClass

try:
    import pytest  # type: ignore[import-untyped]
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    class _PytestStub:
        @staticmethod
        def fixture(func):
            return func
        
        @staticmethod
        def main(args):
            print("pytest not available, please install: pip install pytest")
    
    pytest = _PytestStub()


# Test fixtures
@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create a CheckpointManager instance."""
    return CheckpointManager(checkpoint_dir=temp_checkpoint_dir, max_checkpoints=3)


@pytest.fixture
def sample_agent():
    """Create a sample ProteinAgent for testing."""
    config = create_test_config(
        size_class=ProteinSizeClass.SMALL,
        max_iterations=50
    )
    
    agent = ProteinAgent(
        protein_sequence="ACDEFGH",
        adaptive_config=config
    )
    
    # Run a few iterations to generate state
    for _ in range(5):
        agent.explore_step()
    
    return agent


@pytest.fixture
def sample_agents():
    """Create multiple agents for testing."""
    config = create_test_config(
        size_class=ProteinSizeClass.SMALL,
        max_iterations=50
    )
    
    agents = []
    for i in range(3):
        agent = ProteinAgent(
            protein_sequence="ACDEFGH",
            adaptive_config=config
        )
        agent.set_agent_id(f"test_agent_{i}")
        
        # Run a few iterations
        for _ in range(5):
            agent.explore_step()
        
        agents.append(agent)
    
    return agents


# Tests for CheckpointManager
class TestCheckpointManager:
    """Test suite for CheckpointManager class."""
    
    def test_save_checkpoint(self, checkpoint_manager, sample_agents, temp_checkpoint_dir):
        """Test checkpoint saving."""
        shared_pool = SharedMemoryPool()
        
        # Save checkpoint
        checkpoint_file = checkpoint_manager.save_checkpoint(
            agents=sample_agents,
            shared_pool=shared_pool,
            iteration=10,
            metadata={"test": "metadata"}
        )
        
        # Verify file exists
        assert os.path.exists(checkpoint_file)
        
        # Verify file contains valid JSON
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        assert data["version"] == "1.0.0"
        assert data["iteration"] == 10
        assert data["metadata"]["test"] == "metadata"
        assert len(data["agents"]) == 3
        assert "shared_pool" in data
        assert "integrity_hash" in data
    
    def test_load_checkpoint(self, checkpoint_manager, sample_agents):
        """Test checkpoint loading."""
        shared_pool = SharedMemoryPool()
        
        # Save checkpoint
        checkpoint_file = checkpoint_manager.save_checkpoint(
            agents=sample_agents,
            shared_pool=shared_pool,
            iteration=15,
            metadata={"protein": "test"}
        )
        
        # Load checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_file)
        
        assert checkpoint_data["version"] == "1.0.0"
        assert checkpoint_data["iteration"] == 15
        assert checkpoint_data["metadata"]["protein"] == "test"
        assert len(checkpoint_data["agents"]) == 3
    
    def test_restore_agents(self, checkpoint_manager, sample_agents):
        """Test agent restoration from checkpoint."""
        shared_pool = SharedMemoryPool()
        
        # Get original states
        original_ids = [agent.get_agent_id() for agent in sample_agents]
        original_iterations = [agent._iterations_completed for agent in sample_agents]
        
        # Save checkpoint
        checkpoint_file = checkpoint_manager.save_checkpoint(
            agents=sample_agents,
            shared_pool=shared_pool,
            iteration=20,
            metadata={}
        )
        
        # Load and restore
        checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_file)
        restored_agents, restored_pool, iteration = checkpoint_manager.restore_agents(
            checkpoint_data,
            ProteinAgent
        )
        
        # Verify restoration
        assert iteration == 20
        assert len(restored_agents) == len(sample_agents)
        
        # Check agent IDs match
        restored_ids = [agent.get_agent_id() for agent in restored_agents]
        assert set(restored_ids) == set(original_ids)
        
        # Check iterations completed match
        restored_iterations = [agent._iterations_completed for agent in restored_agents]
        assert restored_iterations == original_iterations
    
    def test_integrity_check(self, checkpoint_manager, sample_agents, temp_checkpoint_dir):
        """Test integrity hash validation."""
        shared_pool = SharedMemoryPool()
        
        # Save checkpoint
        checkpoint_file = checkpoint_manager.save_checkpoint(
            agents=sample_agents,
            shared_pool=shared_pool,
            iteration=25,
            metadata={}
        )
        
        # Corrupt the checkpoint file
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        # Change data without updating hash
        data["iteration"] = 999
        
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)
        
        # Loading should fail integrity check
        try:
            checkpoint_manager.load_checkpoint(checkpoint_file)
            assert False, "Should have raised ValueError for corrupted checkpoint"
        except ValueError as e:
            assert "integrity" in str(e).lower()
    
    def test_checkpoint_rotation(self, checkpoint_manager, sample_agents):
        """Test automatic checkpoint rotation."""
        shared_pool = SharedMemoryPool()
        
        # Save more checkpoints than max_checkpoints (3)
        checkpoint_files = []
        for i in range(5):
            checkpoint_file = checkpoint_manager.save_checkpoint(
                agents=sample_agents,
                shared_pool=shared_pool,
                iteration=i * 10,
                metadata={}
            )
            checkpoint_files.append(checkpoint_file)
            time.sleep(0.1)  # Ensure different timestamps
        
        # List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        
        # Should only have 3 checkpoints (max_checkpoints)
        assert len(checkpoints) <= 3
        
        # Oldest checkpoints should be removed
        existing_files = [c["file"] for c in checkpoints]
        assert checkpoint_files[0] not in existing_files  # First should be gone
        assert checkpoint_files[-1] in existing_files  # Last should still be there
    
    def test_auto_save_interval(self, checkpoint_manager, sample_agents):
        """Test auto-save interval checking."""
        shared_pool = SharedMemoryPool()
        
        # Set auto-save interval
        checkpoint_manager.set_auto_save_interval(10)
        
        # Should not save at iteration 5
        assert not checkpoint_manager.should_auto_save(5)
        
        # Should save at iteration 10
        assert checkpoint_manager.should_auto_save(10)
        
        # Perform auto-save
        checkpoint_file = checkpoint_manager.auto_save(
            agents=sample_agents,
            shared_pool=shared_pool,
            iteration=10,
            metadata={}
        )
        
        assert checkpoint_file is not None
        assert os.path.exists(checkpoint_file)
        
        # Should not save again at iteration 15
        assert not checkpoint_manager.should_auto_save(15)
        
        # Should save again at iteration 20
        assert checkpoint_manager.should_auto_save(20)
    
    def test_list_checkpoints(self, checkpoint_manager, sample_agents):
        """Test checkpoint listing."""
        shared_pool = SharedMemoryPool()
        
        # Save multiple checkpoints
        for i in range(3):
            checkpoint_manager.save_checkpoint(
                agents=sample_agents,
                shared_pool=shared_pool,
                iteration=i * 10,
                metadata={"index": i}
            )
            time.sleep(0.1)
        
        # List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        
        assert len(checkpoints) == 3
        
        # Verify checkpoint info
        for checkpoint in checkpoints:
            assert "file" in checkpoint
            assert "iteration" in checkpoint
            assert "timestamp" in checkpoint
            assert "version" in checkpoint
            assert "size_bytes" in checkpoint
    
    def test_get_latest_checkpoint(self, checkpoint_manager, sample_agents):
        """Test getting latest checkpoint."""
        shared_pool = SharedMemoryPool()
        
        # Initially no checkpoints
        assert checkpoint_manager.get_latest_checkpoint() is None
        
        # Save checkpoints
        checkpoint_files = []
        for i in range(3):
            checkpoint_file = checkpoint_manager.save_checkpoint(
                agents=sample_agents,
                shared_pool=shared_pool,
                iteration=i * 10,
                metadata={}
            )
            checkpoint_files.append(checkpoint_file)
            time.sleep(0.1)
        
        # Get latest
        latest = checkpoint_manager.get_latest_checkpoint()
        
        assert latest is not None
        assert latest == checkpoint_files[-1]  # Should be the last one saved


# Tests for MultiAgentCoordinator integration
class TestCheckpointIntegration:
    """Test suite for checkpoint integration with MultiAgentCoordinator."""
    
    def test_coordinator_with_checkpointing(self, temp_checkpoint_dir):
        """Test coordinator initialization with checkpointing enabled."""
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        
        assert coordinator._enable_checkpointing is True
        assert coordinator._checkpoint_manager is not None
    
    def test_coordinator_auto_save(self, temp_checkpoint_dir):
        """Test automatic checkpoint saving during exploration."""
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # Set short auto-save interval for testing
        assert coordinator._checkpoint_manager is not None, "Checkpoint manager should be initialized"
        coordinator._checkpoint_manager.set_auto_save_interval(5)
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        
        # Run exploration (should trigger auto-save at iteration 5)
        results = coordinator.run_parallel_exploration(iterations=10)
        
        # Check that checkpoint was created
        checkpoints = coordinator._checkpoint_manager.list_checkpoints()
        assert len(checkpoints) > 0
    
    def test_coordinator_manual_save(self, temp_checkpoint_dir):
        """Test manual checkpoint saving."""
        coordinator = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        coordinator.initialize_agents(count=2, diversity_profile="balanced")
        
        # Run a few iterations
        coordinator.run_parallel_exploration(iterations=5)
        
        # Manually save checkpoint
        checkpoint_file = coordinator.save_checkpoint(checkpoint_name="manual_test")
        
        assert os.path.exists(checkpoint_file)
        
        # Load and verify
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        assert data["metadata"]["manual_save"] is True
        assert data["metadata"]["checkpoint_name"] == "manual_test"
    
    def test_coordinator_resume(self, temp_checkpoint_dir):
        """Test resuming exploration from checkpoint."""
        # Create and run initial coordinator
        coordinator1 = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        coordinator1.initialize_agents(count=2, diversity_profile="balanced")
        coordinator1.run_parallel_exploration(iterations=10)
        
        # Save checkpoint
        checkpoint_file = coordinator1.save_checkpoint()
        
        # Create new coordinator and resume
        coordinator2 = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # Resume from checkpoint
        iteration = coordinator2.resume_from_checkpoint(checkpoint_file)
        
        assert iteration == 10
        assert len(coordinator2._agents) == 2
        assert coordinator2._total_iterations == 10
        
        # Continue exploration
        results = coordinator2.run_parallel_exploration(iterations=5)
        
        # Should have completed 15 iterations total
        assert coordinator2._total_iterations == 15
    
    def test_checkpoint_preserves_state(self, temp_checkpoint_dir):
        """Test that checkpoint preserves complete agent state."""
        coordinator1 = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        coordinator1.initialize_agents(count=2, diversity_profile="balanced")
        coordinator1.run_parallel_exploration(iterations=15)
        
        # Get agent states before checkpoint (cast to ProteinAgent for type checking)
        agent1 = coordinator1._agents[0]
        assert isinstance(agent1, ProteinAgent), "Agent should be ProteinAgent instance"
        agent1_iterations = agent1._iterations_completed
        agent1_energy = agent1._best_energy
        agent1_memories = len(agent1._memory._memories)
        
        # Save and reload
        checkpoint_file = coordinator1.save_checkpoint()
        
        coordinator2 = MultiAgentCoordinator(
            protein_sequence="ACDEFGH",
            enable_checkpointing=True,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        coordinator2.resume_from_checkpoint(checkpoint_file)
        
        # Verify state preservation (cast to ProteinAgent for type checking)
        agent2 = coordinator2._agents[0]
        assert isinstance(agent2, ProteinAgent), "Agent should be ProteinAgent instance"
        assert agent2._iterations_completed == agent1_iterations
        assert agent2._best_energy == agent1_energy
        assert len(agent2._memory._memories) == agent1_memories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
