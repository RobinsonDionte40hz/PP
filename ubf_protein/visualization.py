"""
Visualization export system for UBF protein system.

This module implements trajectory export, energy landscape visualization,
and real-time monitoring capabilities for protein folding exploration.
"""

import json
import time
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .interfaces import IVisualizationExporter
from .models import ConformationSnapshot, EnergyLandscape, Conformation


class VisualizationExporter(IVisualizationExporter):
    """
    Implementation of visualization export system.
    
    Supports:
    - Trajectory export (complete agent exploration history)
    - Energy landscape 2D projection (PCA visualization)
    - Real-time streaming for monitoring
    - Multiple output formats (JSON, PDB, CSV)
    """

    def __init__(self, output_dir: str = "visualization_output"):
        """
        Initialize visualization exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trajectory storage: agent_id -> list of snapshots
        self._trajectories: Dict[str, List[ConformationSnapshot]] = {}
        
        # Real-time streaming buffer
        self._stream_buffer: List[ConformationSnapshot] = []
        self._stream_interval = 10  # Save every N snapshots
        self._stream_counter = 0

    def register_agent(self, agent_id: str) -> None:
        """
        Register an agent for trajectory tracking.
        
        Args:
            agent_id: Unique agent identifier
        """
        if agent_id not in self._trajectories:
            self._trajectories[agent_id] = []

    def add_snapshot(self, snapshot: ConformationSnapshot) -> None:
        """
        Add a conformation snapshot to agent's trajectory.
        
        Args:
            snapshot: ConformationSnapshot to add
        """
        agent_id = snapshot.agent_id
        
        if agent_id not in self._trajectories:
            self.register_agent(agent_id)
        
        self._trajectories[agent_id].append(snapshot)

    def export_trajectory(self, agent_id: str) -> List[ConformationSnapshot]:
        """
        Export complete trajectory for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of ConformationSnapshots for the agent
        """
        if agent_id not in self._trajectories:
            return []
        
        return self._trajectories[agent_id].copy()

    def export_energy_landscape(self) -> EnergyLandscape:
        """
        Export 2D projection of explored conformational space using PCA.
        
        Returns:
            EnergyLandscape with 2D coordinates and energy/RMSD values
        """
        # Collect all conformations from all agents
        all_snapshots = []
        for agent_id in self._trajectories:
            all_snapshots.extend(self._trajectories[agent_id])
        
        if len(all_snapshots) < 2:
            # Not enough data for projection
            return EnergyLandscape(
                projection_method='PCA',
                coordinates_2d=[],
                energy_values=[],
                rmsd_values=[]
            )
        
        # Extract energy and RMSD values
        energy_values = [s.conformation.energy for s in all_snapshots]
        rmsd_values = [s.conformation.rmsd_to_native or 0.0 for s in all_snapshots]
        
        # Perform simplified PCA (energy vs RMSD as 2D projection)
        # In a full implementation, would use actual structural coordinates
        coordinates_2d = self._compute_pca_projection(all_snapshots)
        
        return EnergyLandscape(
            projection_method='PCA',
            coordinates_2d=coordinates_2d,
            energy_values=energy_values,
            rmsd_values=rmsd_values
        )

    def stream_update(self, snapshot: ConformationSnapshot) -> None:
        """
        Stream real-time update (non-blocking).
        
        Args:
            snapshot: ConformationSnapshot to stream
        """
        # Add to trajectory
        self.add_snapshot(snapshot)
        
        # Add to stream buffer
        self._stream_buffer.append(snapshot)
        self._stream_counter += 1
        
        # Write to file periodically
        if self._stream_counter >= self._stream_interval:
            self._flush_stream_buffer()
            self._stream_counter = 0

    def _flush_stream_buffer(self) -> None:
        """Flush stream buffer to file (non-blocking)."""
        if not self._stream_buffer:
            return
        
        # Write to streaming log file
        stream_file = self._output_dir / "realtime_stream.jsonl"
        
        try:
            with open(stream_file, 'a') as f:
                for snapshot in self._stream_buffer:
                    data = {
                        'timestamp': snapshot.timestamp,
                        'iteration': snapshot.iteration,
                        'agent_id': snapshot.agent_id,
                        'energy': snapshot.conformation.energy,
                        'rmsd': snapshot.conformation.rmsd_to_native or 0.0,
                        'frequency': snapshot.consciousness_state.frequency,
                        'coherence': snapshot.consciousness_state.coherence
                    }
                    f.write(json.dumps(data) + '\n')
            
            self._stream_buffer.clear()
            
        except Exception as e:
            # Non-blocking: log error but don't raise
            print(f"Warning: Stream buffer flush failed: {e}")

    def _compute_pca_projection(self, snapshots: List[ConformationSnapshot]) -> List[Tuple[float, float]]:
        """
        Compute simplified 2D PCA projection.
        
        Uses energy and RMSD as features for 2D projection.
        In a full implementation, would use actual structural coordinates.
        
        Args:
            snapshots: List of ConformationSnapshots
            
        Returns:
            List of (x, y) coordinates in 2D projection
        """
        if len(snapshots) < 2:
            return []
        
        # Extract features (energy and RMSD)
        energies = [s.conformation.energy for s in snapshots]
        rmsds = [s.conformation.rmsd_to_native or 0.0 for s in snapshots]
        
        # Normalize to 0-1 range
        min_energy, max_energy = min(energies), max(energies)
        min_rmsd, max_rmsd = min(rmsds), max(rmsds)
        
        energy_range = max_energy - min_energy if max_energy != min_energy else 1.0
        rmsd_range = max_rmsd - min_rmsd if max_rmsd != min_rmsd else 1.0
        
        # Create 2D coordinates (normalized energy vs RMSD)
        coordinates = []
        for energy, rmsd in zip(energies, rmsds):
            x = (energy - min_energy) / energy_range
            y = (rmsd - min_rmsd) / rmsd_range
            coordinates.append((x, y))
        
        return coordinates

    def export_trajectory_to_json(self, agent_id: str, output_file: Optional[str] = None) -> str:
        """
        Export agent trajectory to JSON format.
        
        Args:
            agent_id: Agent identifier
            output_file: Optional output file path
            
        Returns:
            Path to exported file
        """
        trajectory = self.export_trajectory(agent_id)
        
        if output_file is None:
            output_file = str(self._output_dir / f"{agent_id}_trajectory.json")
        
        # Convert to serializable format
        data = {
            'agent_id': agent_id,
            'snapshot_count': len(trajectory),
            'snapshots': [
                {
                    'iteration': s.iteration,
                    'timestamp': s.timestamp,
                    'energy': s.conformation.energy,
                    'rmsd': s.conformation.rmsd_to_native or 0.0,
                    'consciousness': {
                        'frequency': s.consciousness_state.frequency,
                        'coherence': s.consciousness_state.coherence
                    },
                    'behavioral': {
                        'exploration_energy': s.behavioral_state.exploration_energy,
                        'structural_focus': s.behavioral_state.structural_focus,
                        'hydrophobic_drive': s.behavioral_state.hydrophobic_drive,
                        'risk_tolerance': s.behavioral_state.risk_tolerance,
                        'native_state_ambition': s.behavioral_state.native_state_ambition
                    },
                    'secondary_structure': s.conformation.secondary_structure
                }
                for s in trajectory
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_file

    def export_energy_landscape_to_csv(self, output_file: Optional[str] = None) -> str:
        """
        Export energy landscape to CSV format for plotting.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to exported file
        """
        landscape = self.export_energy_landscape()
        
        if output_file is None:
            output_file = str(self._output_dir / "energy_landscape.csv")
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("x,y,energy,rmsd\n")
            
            # Write data rows
            for (x, y), energy, rmsd in zip(
                landscape.coordinates_2d,
                landscape.energy_values,
                landscape.rmsd_values
            ):
                f.write(f"{x:.6f},{y:.6f},{energy:.2f},{rmsd:.2f}\n")
        
        return output_file

    def export_trajectory_to_pdb(self, agent_id: str, output_file: Optional[str] = None) -> str:
        """
        Export agent trajectory to multi-model PDB format.
        
        Compatible with PyMOL, VMD, ChimeraX for visualization.
        
        Args:
            agent_id: Agent identifier
            output_file: Optional output file path
            
        Returns:
            Path to exported file
        """
        trajectory = self.export_trajectory(agent_id)
        
        if output_file is None:
            output_file = str(self._output_dir / f"{agent_id}_trajectory.pdb")
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("HEADER    PROTEIN FOLDING TRAJECTORY\n")
            f.write(f"TITLE     UBF PROTEIN AGENT {agent_id} TRAJECTORY\n")
            f.write(f"REMARK    SNAPSHOT COUNT: {len(trajectory)}\n")
            f.write("\n")
            
            # Write each snapshot as a MODEL
            for model_num, snapshot in enumerate(trajectory, 1):
                f.write(f"MODEL     {model_num:4d}\n")
                f.write(f"REMARK    ITERATION: {snapshot.iteration}\n")
                f.write(f"REMARK    ENERGY: {snapshot.conformation.energy:.2f}\n")
                f.write(f"REMARK    RMSD: {snapshot.conformation.rmsd_to_native or 0.0:.2f}\n")
                
                # Write atom coordinates (simplified - would use actual coordinates)
                for i, coord in enumerate(snapshot.conformation.atom_coordinates[:100], 1):  # Limit to 100 atoms
                    x, y, z = coord
                    # PDB ATOM format
                    f.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                
                f.write("ENDMDL\n")
            
            f.write("END\n")
        
        return output_file

    def get_trajectory_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for an agent's trajectory.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with trajectory statistics
        """
        trajectory = self.export_trajectory(agent_id)
        
        if not trajectory:
            return {
                'agent_id': agent_id,
                'snapshot_count': 0,
                'duration': 0.0
            }
        
        energies = [s.conformation.energy for s in trajectory]
        rmsds = [s.conformation.rmsd_to_native or 0.0 for s in trajectory]
        
        return {
            'agent_id': agent_id,
            'snapshot_count': len(trajectory),
            'duration': trajectory[-1].timestamp - trajectory[0].timestamp,
            'energy': {
                'initial': energies[0],
                'final': energies[-1],
                'best': min(energies),
                'worst': max(energies),
                'improvement': energies[0] - energies[-1]
            },
            'rmsd': {
                'initial': rmsds[0],
                'final': rmsds[-1],
                'best': min(rmsds),
                'worst': max(rmsds),
                'improvement': rmsds[0] - rmsds[-1]
            }
        }

    def export_all_trajectories(self, format: str = 'json') -> List[str]:
        """
        Export all agent trajectories in specified format.
        
        Args:
            format: Output format ('json', 'pdb', or 'csv')
            
        Returns:
            List of exported file paths
        """
        exported_files = []
        
        for agent_id in self._trajectories:
            if format == 'json':
                file_path = self.export_trajectory_to_json(agent_id)
            elif format == 'pdb':
                file_path = self.export_trajectory_to_pdb(agent_id)
            else:
                continue  # Skip unsupported formats
            
            exported_files.append(file_path)
        
        return exported_files

    def clear_trajectories(self) -> None:
        """Clear all stored trajectories to free memory."""
        self._trajectories.clear()
        self._stream_buffer.clear()
        self._stream_counter = 0

    def get_agent_count(self) -> int:
        """Get number of agents being tracked."""
        return len(self._trajectories)

    def get_total_snapshots(self) -> int:
        """Get total number of snapshots across all agents."""
        return sum(len(traj) for traj in self._trajectories.values())
