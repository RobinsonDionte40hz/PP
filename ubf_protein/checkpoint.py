"""
Checkpoint and resume system for UBF protein folding.

This module implements full state serialization and resumption for long-running
simulations, enabling recovery from interruptions and distributed computation.
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from .interfaces import ICheckpointManager
from .models import (
    Conformation, ConsciousnessCoordinates, BehavioralStateData,
    ConformationalMemory, AdaptiveConfig, ProteinSizeClass
)

logger = logging.getLogger(__name__)

# Checkpoint format version for compatibility tracking
CHECKPOINT_VERSION = "1.0.0"


class CheckpointManager(ICheckpointManager):
    """
    Manages checkpoint saving and loading for protein folding simulations.
    
    Features:
    - Full state serialization (agents, memories, configuration)
    - Version and integrity checking
    - Automatic checkpoint rotation
    - Graceful error recovery
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
            max_checkpoints: Maximum number of checkpoints to keep (rotation)
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._max_checkpoints = max_checkpoints
        self._last_checkpoint_time = 0
        self._auto_save_interval = 100  # Default: save every 100 iterations
        self._last_auto_save_iteration = 0

    def set_auto_save_interval(self, interval: int) -> None:
        """
        Set the auto-save interval.
        
        Args:
            interval: Number of iterations between auto-saves (0 to disable)
        """
        self._auto_save_interval = interval
        logger.info(f"Auto-save interval set to {interval} iterations")

    def save_checkpoint(
        self,
        agents: List[Any],  # List of ProteinAgent instances
        shared_pool: Any,  # SharedMemoryPool instance
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save checkpoint with all agent states and shared memory.
        
        Args:
            agents: List of ProteinAgent instances
            shared_pool: SharedMemoryPool instance
            iteration: Current iteration number
            metadata: Optional additional metadata
            
        Returns:
            Path to saved checkpoint file
        """
        try:
            # Generate checkpoint filename
            timestamp = int(time.time())
            checkpoint_file = self._checkpoint_dir / f"checkpoint_iter{iteration}_{timestamp}.json"
            
            # Build checkpoint data
            checkpoint_data = {
                "version": CHECKPOINT_VERSION,
                "timestamp": timestamp,
                "iteration": iteration,
                "metadata": metadata or {},
                "agents": self._serialize_agents(agents),
                "shared_pool": self._serialize_shared_pool(shared_pool)
            }
            
            # Calculate integrity hash
            checkpoint_json = json.dumps(checkpoint_data, sort_keys=True)
            integrity_hash = hashlib.sha256(checkpoint_json.encode()).hexdigest()
            checkpoint_data["integrity_hash"] = integrity_hash
            
            # Write checkpoint file
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"Checkpoint saved: {checkpoint_file} (iteration {iteration})")
            
            # Rotate old checkpoints
            self._rotate_checkpoints()
            
            self._last_checkpoint_time = timestamp
            return str(checkpoint_file)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Load and validate checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
            
        Raises:
            ValueError: If checkpoint is invalid or corrupted
        """
        try:
            checkpoint_path = Path(checkpoint_file)
            
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
            
            # Load checkpoint data
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Validate version
            if checkpoint_data.get("version") != CHECKPOINT_VERSION:
                logger.warning(
                    f"Checkpoint version mismatch: {checkpoint_data.get('version')} "
                    f"!= {CHECKPOINT_VERSION}"
                )
            
            # Validate integrity
            stored_hash = checkpoint_data.pop("integrity_hash", None)
            if stored_hash:
                checkpoint_json = json.dumps(checkpoint_data, sort_keys=True)
                calculated_hash = hashlib.sha256(checkpoint_json.encode()).hexdigest()
                
                if stored_hash != calculated_hash:
                    raise ValueError("Checkpoint integrity check failed - file may be corrupted")
            
            logger.info(
                f"Checkpoint loaded: {checkpoint_file} "
                f"(iteration {checkpoint_data.get('iteration')})"
            )
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def restore_agents(
        self,
        checkpoint_data: Dict[str, Any],
        agent_class: type
    ) -> Tuple[List[Any], Any, int]:
        """
        Restore agents and shared pool from checkpoint data.
        
        Args:
            checkpoint_data: Loaded checkpoint data
            agent_class: ProteinAgent class for reconstruction
            
        Returns:
            Tuple of (restored_agents, restored_shared_pool, iteration)
        """
        try:
            iteration = checkpoint_data["iteration"]
            
            # Restore shared pool
            from .memory_system import SharedMemoryPool
            shared_pool = self._deserialize_shared_pool(
                checkpoint_data["shared_pool"],
                SharedMemoryPool
            )
            
            # Restore agents
            agents = self._deserialize_agents(
                checkpoint_data["agents"],
                agent_class,
                shared_pool
            )
            
            logger.info(
                f"Restored {len(agents)} agents from checkpoint "
                f"(iteration {iteration})"
            )
            
            return agents, shared_pool, iteration
            
        except Exception as e:
            logger.error(f"Failed to restore agents from checkpoint: {e}")
            raise

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []
        
        for checkpoint_file in sorted(self._checkpoint_dir.glob("checkpoint_*.json")):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                checkpoints.append({
                    "file": str(checkpoint_file),
                    "iteration": data.get("iteration"),
                    "timestamp": data.get("timestamp"),
                    "version": data.get("version"),
                    "size_bytes": checkpoint_file.stat().st_size
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
        
        return checkpoints

    def should_auto_save(self, current_iteration: int) -> bool:
        """
        Check if an auto-save should be triggered.
        
        Args:
            current_iteration: Current iteration number
            
        Returns:
            True if auto-save should be triggered
        """
        if self._auto_save_interval <= 0:
            return False
        
        iterations_since_last = current_iteration - self._last_auto_save_iteration
        
        return iterations_since_last >= self._auto_save_interval

    def auto_save(
        self,
        agents: List[Any],
        shared_pool: Any,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Perform auto-save if interval has elapsed.
        
        Args:
            agents: List of ProteinAgent instances
            shared_pool: SharedMemoryPool instance
            iteration: Current iteration number
            metadata: Optional additional metadata
            
        Returns:
            Path to saved checkpoint file, or None if no save performed
        """
        if not self.should_auto_save(iteration):
            return None
        
        try:
            # Add auto-save marker to metadata
            save_metadata = metadata.copy() if metadata else {}
            save_metadata["auto_save"] = True
            save_metadata["auto_save_interval"] = self._auto_save_interval
            
            checkpoint_file = self.save_checkpoint(
                agents=agents,
                shared_pool=shared_pool,
                iteration=iteration,
                metadata=save_metadata
            )
            
            self._last_auto_save_iteration = iteration
            logger.info(f"Auto-save completed at iteration {iteration}")
            
            return checkpoint_file
            
        except Exception as e:
            # Log error but don't crash - auto-save is non-critical
            logger.error(f"Auto-save failed at iteration {iteration}: {e}")
            return None

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint file.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Sort by iteration (descending)
        checkpoints.sort(key=lambda x: x["iteration"], reverse=True)
        
        return checkpoints[0]["file"]

    def _serialize_agents(self, agents: List[Any]) -> List[Dict[str, Any]]:
        """Serialize list of agents to JSON-compatible format."""
        serialized_agents = []
        
        for agent in agents:
            agent_data = {
                "agent_id": agent.get_agent_id(),
                "protein_sequence": agent._protein_sequence,
                "current_conformation": self._serialize_conformation(agent._current_conformation),
                "consciousness": self._serialize_consciousness(agent._consciousness.get_coordinates()),
                "behavioral": self._serialize_behavioral(agent._behavioral.get_behavioral_data()),
                "memories": {
                    move_type: [self._serialize_memory(m) for m in memories]
                    for move_type, memories in agent._memory._memories.items()
                },
                "memory_count": agent._memory._memory_count,
                "metrics": {
                    "iterations_completed": agent._iterations_completed,
                    "conformations_explored": agent._conformations_explored,
                    "memories_created": agent._memories_created,
                    "best_energy": agent._best_energy,
                    "best_rmsd": agent._best_rmsd,
                    "stuck_in_minima_count": agent._stuck_in_minima_count,
                    "successful_escapes": agent._successful_escapes,
                    "validation_failures": agent._validation_failures,
                    "repair_attempts": agent._repair_attempts,
                    "repair_successes": agent._repair_successes
                },
                "adaptive_config": self._serialize_config(agent._adaptive_config),
                "enable_visualization": agent._enable_visualization,
                "max_snapshots": agent._max_snapshots
            }
            
            serialized_agents.append(agent_data)
        
        return serialized_agents

    def _deserialize_agents(
        self,
        agents_data: List[Dict[str, Any]],
        agent_class: type,
        shared_pool: Any
    ) -> List[Any]:
        """Deserialize agents from JSON data."""
        from .protein_agent import ProteinAgent
        
        agents = []
        
        for agent_data in agents_data:
            # Reconstruct adaptive config
            config = self._deserialize_config(agent_data["adaptive_config"])
            
            # Create new agent (ProteinAgent doesn't take shared_pool in constructor)
            agent = agent_class(
                protein_sequence=agent_data["protein_sequence"],
                adaptive_config=config,
                enable_visualization=agent_data.get("enable_visualization", False),
                max_snapshots=agent_data.get("max_snapshots", 1000)
            )
            
            # Restore agent state
            agent.set_agent_id(agent_data["agent_id"])
            agent._current_conformation = self._deserialize_conformation(
                agent_data["current_conformation"]
            )
            
            # Restore consciousness
            consciousness_data = agent_data["consciousness"]
            agent._consciousness._coordinates.frequency = consciousness_data["frequency"]
            agent._consciousness._coordinates.coherence = consciousness_data["coherence"]
            agent._consciousness._coordinates.last_update_timestamp = consciousness_data["last_update_timestamp"]
            
            # Restore behavioral state
            behavioral_data = agent_data["behavioral"]
            agent._behavioral._cached_state = self._deserialize_behavioral(behavioral_data)
            
            # Restore memories (as dict of move_type -> list of memories)
            from collections import defaultdict
            agent._memory._memories = defaultdict(list)
            for move_type, memories in agent_data["memories"].items():
                agent._memory._memories[move_type] = [
                    self._deserialize_memory(m) for m in memories
                ]
            agent._memory._memory_count = agent_data.get("memory_count", 0)
            
            # Restore metrics
            metrics = agent_data["metrics"]
            agent._iterations_completed = metrics["iterations_completed"]
            agent._conformations_explored = metrics["conformations_explored"]
            agent._memories_created = metrics["memories_created"]
            agent._best_energy = metrics["best_energy"]
            agent._best_rmsd = metrics["best_rmsd"]
            agent._stuck_in_minima_count = metrics["stuck_in_minima_count"]
            agent._successful_escapes = metrics["successful_escapes"]
            agent._validation_failures = metrics.get("validation_failures", 0)
            agent._repair_attempts = metrics.get("repair_attempts", 0)
            agent._repair_successes = metrics.get("repair_successes", 0)
            
            agents.append(agent)
        
        return agents

    def _serialize_shared_pool(self, shared_pool: Any) -> Dict[str, Any]:
        """Serialize shared memory pool."""
        from .config import MAX_SHARED_MEMORY_POOL_SIZE, SHARED_MEMORY_SIGNIFICANCE_THRESHOLD
        
        return {
            "memories": [self._serialize_memory(m) for m in shared_pool._shared_memories],
            "memory_count": shared_pool._memory_count,
            "max_pool_size": MAX_SHARED_MEMORY_POOL_SIZE,
            "share_threshold": SHARED_MEMORY_SIGNIFICANCE_THRESHOLD
        }

    def _deserialize_shared_pool(
        self,
        pool_data: Dict[str, Any],
        pool_class: type
    ) -> Any:
        """Deserialize shared memory pool."""
        pool = pool_class()  # SharedMemoryPool doesn't take constructor args
        
        pool._shared_memories = [
            self._deserialize_memory(m) for m in pool_data["memories"]
        ]
        pool._memory_count = pool_data.get("memory_count", len(pool._shared_memories))
        
        return pool

    def _serialize_conformation(self, conformation: Conformation) -> Dict[str, Any]:
        """Serialize conformation to dict."""
        return {
            "conformation_id": conformation.conformation_id,
            "sequence": conformation.sequence,
            "atom_coordinates": conformation.atom_coordinates,
            "energy": conformation.energy,
            "rmsd_to_native": conformation.rmsd_to_native,
            "secondary_structure": conformation.secondary_structure,
            "phi_angles": conformation.phi_angles,
            "psi_angles": conformation.psi_angles,
            "available_move_types": conformation.available_move_types,
            "structural_constraints": conformation.structural_constraints
        }

    def _deserialize_conformation(self, data: Dict[str, Any]) -> Conformation:
        """Deserialize conformation from dict."""
        return Conformation(
            conformation_id=data["conformation_id"],
            sequence=data["sequence"],
            atom_coordinates=[tuple(coord) for coord in data["atom_coordinates"]],
            energy=data["energy"],
            rmsd_to_native=data["rmsd_to_native"],
            secondary_structure=data["secondary_structure"],
            phi_angles=data["phi_angles"],
            psi_angles=data["psi_angles"],
            available_move_types=data["available_move_types"],
            structural_constraints=data["structural_constraints"]
        )

    def _serialize_consciousness(self, consciousness: ConsciousnessCoordinates) -> Dict[str, Any]:
        """Serialize consciousness coordinates."""
        return {
            "frequency": consciousness.frequency,
            "coherence": consciousness.coherence,
            "last_update_timestamp": consciousness.last_update_timestamp
        }

    def _serialize_behavioral(self, behavioral: BehavioralStateData) -> Dict[str, Any]:
        """Serialize behavioral state."""
        return {
            "exploration_energy": behavioral.exploration_energy,
            "structural_focus": behavioral.structural_focus,
            "conformational_bias": behavioral.conformational_bias,
            "hydrophobic_drive": behavioral.hydrophobic_drive,
            "risk_tolerance": behavioral.risk_tolerance,
            "native_state_ambition": behavioral.native_state_ambition,
            "cached_timestamp": behavioral.cached_timestamp
        }

    def _deserialize_behavioral(self, data: Dict[str, Any]) -> BehavioralStateData:
        """Deserialize behavioral state."""
        return BehavioralStateData(
            exploration_energy=data["exploration_energy"],
            structural_focus=data["structural_focus"],
            conformational_bias=data["conformational_bias"],
            hydrophobic_drive=data["hydrophobic_drive"],
            risk_tolerance=data["risk_tolerance"],
            native_state_ambition=data["native_state_ambition"],
            cached_timestamp=data["cached_timestamp"]
        )

    def _serialize_memory(self, memory: ConformationalMemory) -> Dict[str, Any]:
        """Serialize conformational memory."""
        return {
            "memory_id": memory.memory_id,
            "move_type": memory.move_type,
            "significance": memory.significance,
            "energy_change": memory.energy_change,
            "rmsd_change": memory.rmsd_change,
            "success": memory.success,
            "timestamp": memory.timestamp,
            "consciousness_state": self._serialize_consciousness(memory.consciousness_state),
            "behavioral_state": self._serialize_behavioral(memory.behavioral_state)
        }

    def _deserialize_memory(self, data: Dict[str, Any]) -> ConformationalMemory:
        """Deserialize conformational memory."""
        return ConformationalMemory(
            memory_id=data["memory_id"],
            move_type=data["move_type"],
            significance=data["significance"],
            energy_change=data["energy_change"],
            rmsd_change=data["rmsd_change"],
            success=data["success"],
            timestamp=data["timestamp"],
            consciousness_state=ConsciousnessCoordinates(
                frequency=data["consciousness_state"]["frequency"],
                coherence=data["consciousness_state"]["coherence"],
                last_update_timestamp=data["consciousness_state"]["last_update_timestamp"]
            ),
            behavioral_state=self._deserialize_behavioral(data["behavioral_state"])
        )

    def _serialize_config(self, config: AdaptiveConfig) -> Dict[str, Any]:
        """Serialize adaptive configuration."""
        return {
            "size_class": config.size_class.value,
            "residue_count": config.residue_count,
            "initial_frequency_range": config.initial_frequency_range,
            "initial_coherence_range": config.initial_coherence_range,
            "stuck_detection_window": config.stuck_detection_window,
            "stuck_detection_threshold": config.stuck_detection_threshold,
            "memory_significance_threshold": config.memory_significance_threshold,
            "max_memories_per_agent": config.max_memories_per_agent,
            "convergence_energy_threshold": config.convergence_energy_threshold,
            "convergence_rmsd_threshold": config.convergence_rmsd_threshold,
            "max_iterations": config.max_iterations,
            "checkpoint_interval": config.checkpoint_interval
        }

    def _deserialize_config(self, data: Dict[str, Any]) -> AdaptiveConfig:
        """Deserialize adaptive configuration."""
        return AdaptiveConfig(
            size_class=ProteinSizeClass(data["size_class"]),
            residue_count=data["residue_count"],
            initial_frequency_range=tuple(data["initial_frequency_range"]),
            initial_coherence_range=tuple(data["initial_coherence_range"]),
            stuck_detection_window=data["stuck_detection_window"],
            stuck_detection_threshold=data["stuck_detection_threshold"],
            memory_significance_threshold=data["memory_significance_threshold"],
            max_memories_per_agent=data["max_memories_per_agent"],
            convergence_energy_threshold=data["convergence_energy_threshold"],
            convergence_rmsd_threshold=data["convergence_rmsd_threshold"],
            max_iterations=data["max_iterations"],
            checkpoint_interval=data["checkpoint_interval"]
        )

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the most recent N."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self._max_checkpoints:
            return
        
        # Sort by iteration (ascending)
        checkpoints.sort(key=lambda x: x["iteration"])
        
        # Remove oldest checkpoints
        to_remove = checkpoints[:len(checkpoints) - self._max_checkpoints]
        
        for checkpoint in to_remove:
            try:
                Path(checkpoint["file"]).unlink()
                logger.info(f"Removed old checkpoint: {checkpoint['file']}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint['file']}: {e}")
