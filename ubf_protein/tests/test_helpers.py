"""
Helper functions and fixtures for testing.

This module provides utility functions to create test data structures
that match the actual model definitions.
"""

import time
import uuid
from typing import List, Optional

from ubf_protein.models import (
    Conformation, ConformationSnapshot, ConsciousnessCoordinates,
    BehavioralStateData, AdaptiveConfig, ProteinSizeClass, MoveType
)


def create_test_conformation(sequence: str = "ACDEFGH",
                              energy: float = -15.5,
                              rmsd: float = 2.3) -> Conformation:
    """
    Create a test conformation with all required fields.
    
    Args:
        sequence: Protein sequence
        energy: Energy value
        rmsd: RMSD to native
        
    Returns:
        Configured Conformation object
    """
    n_residues = len(sequence)
    
    return Conformation(
        conformation_id=str(uuid.uuid4()),
        sequence=sequence,
        atom_coordinates=[(float(i), float(i), float(i)) for i in range(n_residues * 4)],  # 4 atoms per residue
        energy=energy,
        rmsd_to_native=rmsd,
        secondary_structure=['C'] * n_residues,  # All coil
        phi_angles=[120.0 + i for i in range(n_residues)],
        psi_angles=[-140.0 + i for i in range(n_residues)],
        available_move_types=[mt.value for mt in MoveType],
        structural_constraints={}
    )


def create_test_consciousness(frequency: float = 8.0,
                               coherence: float = 0.7) -> ConsciousnessCoordinates:
    """
    Create test consciousness coordinates.
    
    Args:
        frequency: Frequency value (3-15 Hz)
        coherence: Coherence value (0.2-1.0)
        
    Returns:
        ConsciousnessCoordinates object
    """
    return ConsciousnessCoordinates(
        frequency=frequency,
        coherence=coherence,
        last_update_timestamp=int(time.time() * 1000)
    )


def create_test_behavioral() -> BehavioralStateData:
    """
    Create test behavioral state data.
    
    Returns:
        BehavioralStateData object
    """
    return BehavioralStateData.from_consciousness(8.0, 0.7)


def create_test_snapshot(iteration: int,
                         agent_id: str = "test_agent",
                         conformation: Optional[Conformation] = None) -> ConformationSnapshot:
    """
    Create a test conformation snapshot.
    
    Args:
        iteration: Iteration number
        agent_id: Agent identifier
        conformation: Conformation to use (creates default if None)
        
    Returns:
        ConformationSnapshot object
    """
    if conformation is None:
        conformation = create_test_conformation()
    
    return ConformationSnapshot(
        iteration=iteration,
        timestamp=time.time(),
        conformation=conformation,
        agent_id=agent_id,
        consciousness_state=create_test_consciousness(),
        behavioral_state=create_test_behavioral()
    )


def create_test_config(size_class: ProteinSizeClass = ProteinSizeClass.SMALL,
                       max_iterations: int = 10) -> AdaptiveConfig:
    """
    Create test adaptive configuration.
    
    Args:
        size_class: Protein size class
        max_iterations: Maximum iterations
        
    Returns:
        AdaptiveConfig object
    """
    return AdaptiveConfig(
        size_class=size_class,
        residue_count=30 if size_class == ProteinSizeClass.SMALL else 100,
        initial_frequency_range=(7.0, 10.0),
        initial_coherence_range=(0.6, 0.8),
        stuck_detection_window=10,
        stuck_detection_threshold=0.5,
        memory_significance_threshold=0.3,
        max_memories_per_agent=100,
        convergence_energy_threshold=-50.0,
        convergence_rmsd_threshold=1.0,
        max_iterations=max_iterations,
        checkpoint_interval=10
    )


def create_test_snapshots(count: int = 10,
                          agent_id: str = "test_agent") -> List[ConformationSnapshot]:
    """
    Create a list of test snapshots with varying properties.
    
    Args:
        count: Number of snapshots to create
        agent_id: Agent identifier
        
    Returns:
        List of ConformationSnapshot objects
    """
    snapshots = []
    
    for i in range(count):
        # Create conformation with progressively better energy
        conformation = create_test_conformation(
            energy=-15.5 - i * 0.5,
            rmsd=2.3 - i * 0.1
        )
        
        snapshot = create_test_snapshot(
            iteration=i,
            agent_id=agent_id,
            conformation=conformation
        )
        
        snapshots.append(snapshot)
    
    return snapshots
