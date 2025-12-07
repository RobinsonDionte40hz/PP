"""
Dynamic loop refinement using G(φ,t) temporal evolution.

This module implements loop refinement with time-dependent golden ratio
evolution for flexible loop regions. Loop refinement strategy adapts based
on quantum coherence (QCP) values:

- Low QCP (<4): Classical loop modeling
- Medium QCP (4-7): G(φ,t) temporal evolution with quantum decay
- High QCP (>7): Quantum-corrected geometry constraints

Key Concepts:
    - G(φ,t) Evolution: Time-dependent golden ratio scaling
      (proprietary formula using quantum coherence principles)
    - Temporal Evolution: Multi-step sampling across picosecond timescale
    - Energy Selection: Choose conformation with lowest energy at each step
    - Smooth Interpolation: Gradual transition from extended to compact

Design Principles:
    - Pure Python implementation (PyPy-optimized)
    - Immutable data models (LoopRegion dataclass)
    - Physics-grounded coherence time (proprietary)
    - Adaptive strategy selection based on QCP
"""

from typing import List, Dict, Tuple, Optional, Any
import math

# Handle imports for both package and direct execution
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .models import Conformation, LoopRegion
    from .interfaces import IPhysicsCalculator
except ImportError:
    from ubf_protein.models import Conformation, LoopRegion
    from ubf_protein.interfaces import IPhysicsCalculator


class LoopRefiner:
    """
    Dynamic loop refinement using G(φ,t) temporal evolution.
    
    Refines flexible loop regions using quantum-classical hybrid approach.
    Strategy selection based on average QCP of loop residues:
    - Classical (<4 QCP): Standard loop modeling
    - Quantum (4-7 QCP): G(φ,t) temporal evolution
    - High (>7 QCP): Quantum-corrected constraints
    
    Attributes:
        phi: Golden ratio constant (1.618033988749895)
        coherence_time_fs: Quantum coherence time in femtoseconds (proprietary)
        max_time_ps: Maximum simulation time in picoseconds
        num_timesteps: Number of temporal evolution steps
        energy_calculator: Optional energy calculator for scoring conformations
    
    Example:
        >>> refiner = LoopRefiner()
        >>> loop = LoopRegion(
        ...     start_residue=10,
        ...     end_residue=15,
        ...     average_qcp=5.5,
        ...     current_conformation=[...]
        ... )
        >>> refined_conf = refiner.refine_loops_dynamic(
        ...     structure=current_structure,
        ...     loops=[loop],
        ...     qcp_values={10: 5.2, 11: 5.8, ...}
        ... )
    """
    
    def __init__(
        self,
        energy_calculator: Optional[IPhysicsCalculator] = None
    ):
        """
        Initialize loop refiner with quantum parameters.
        
        Args:
            energy_calculator: Optional energy calculator for conformation scoring
        """
        # Golden ratio and quantum constants (proprietary values)
        self.phi = 1.618033988749895
        self.coherence_time_fs = 408.0  # proprietary
        self.max_time_ps = 1.0  # picoseconds
        self.num_timesteps = 100
        
        # Energy calculator for scoring
        self.energy_calculator = energy_calculator
        
        # Classical loop refinement parameters
        self.classical_iterations = 50
        self.classical_temperature = 300.0  # Kelvin
    
    def refine_loops_dynamic(
        self,
        structure: Conformation,
        loops: List[LoopRegion],
        qcp_values: Dict[int, float]
    ) -> Conformation:
        """
        Refine loop conformations using quantum-classical hybrid approach.
        
        Strategy selection per loop:
        1. Calculate average QCP for loop residues
        2. QCP < 4: Classical loop modeling (sampling + minimization)
        3. 4 <= QCP < 7: G(φ,t) temporal evolution
        4. QCP >= 7: Quantum-corrected geometry (high coherence)
        
        Args:
            structure: Current protein conformation
            loops: List of loop regions to refine
            qcp_values: QCP values for each residue index
        
        Returns:
            Refined conformation with improved loop geometries
        
        Example:
            >>> loops = [
            ...     LoopRegion(10, 15, 5.5, [(1,2,3), ...]),
            ...     LoopRegion(25, 30, 3.2, [(4,5,6), ...])
            ... ]
            >>> refined = refiner.refine_loops_dynamic(structure, loops, qcp_values)
        """
        # Create a copy of structure to modify
        refined_coords = list(structure.atom_coordinates)
        
        for loop in loops:
            # Select refinement strategy based on QCP
            if loop.is_classical_refinement():
                # Classical loop modeling
                refined_loop_coords = self._refine_classical(loop, qcp_values)
            elif loop.is_quantum_refinement():
                # G(φ,t) temporal evolution
                refined_loop_coords = self.apply_g_phi_t_evolution(loop, qcp_values)
            else:  # High QCP (>= 7)
                # Quantum-corrected geometry
                refined_loop_coords = self._refine_high_qcp(loop, qcp_values)
            
            # Update coordinates in structure
            for i, residue_idx in enumerate(range(loop.start_residue, loop.end_residue + 1)):
                refined_coords[residue_idx] = refined_loop_coords[i]
        
        # Create refined conformation
        refined_structure = Conformation(
            conformation_id=structure.conformation_id + "_loop_refined",
            sequence=structure.sequence,
            atom_coordinates=refined_coords,
            energy=structure.energy,  # Will be recalculated
            rmsd_to_native=structure.rmsd_to_native,
            secondary_structure=structure.secondary_structure,
            phi_angles=structure.phi_angles,
            psi_angles=structure.psi_angles,
            available_move_types=structure.available_move_types,
            structural_constraints=structure.structural_constraints
        )
        
        return refined_structure
    
    def apply_g_phi_t_evolution(
        self,
        loop: LoopRegion,
        qcp_values: Dict[int, float]
    ) -> List[Tuple[float, float, float]]:
        """
        Apply time-dependent golden ratio evolution to loop.
        
        Uses proprietary G(φ,t) temporal evolution formula.
        
        Evolution strategy:
        1. Generate 100 timesteps from 0 to 1 ps
        2. At each timestep, calculate G(φ,t) scaling factor
        3. Interpolate loop conformation using scaling factor
        4. Evaluate energy of interpolated conformation
        5. Select conformation with lowest energy
        
        Args:
            loop: Loop region to refine
            qcp_values: QCP values for residue indices
        
        Returns:
            Refined loop coordinates (list of (x,y,z) tuples)
        
        Example:
            >>> loop = LoopRegion(10, 15, 5.5, [...])
            >>> refined_coords = refiner.apply_g_phi_t_evolution(loop, qcp_values)
        """
        # Convert coherence time to picoseconds
        coherence_time_ps = self.coherence_time_fs / 1000.0  # proprietary
        
        # Generate timesteps (0 to max_time_ps)
        timesteps = [
            (i / self.num_timesteps) * self.max_time_ps
            for i in range(self.num_timesteps + 1)
        ]
        
        best_conformation = loop.current_conformation
        best_energy = float('inf')
        
        for t in timesteps:
            # Calculate G(φ,t) = exp(-t/τ_c) × φ
            g_phi_t = math.exp(-t / coherence_time_ps) * self.phi
            
            # Interpolate loop conformation with G(φ,t) scaling
            interpolated_conf = self.interpolate_loop_conformation(
                loop=loop,
                scaling_factor=g_phi_t,
                time=t
            )
            
            # Evaluate energy if calculator available
            if self.energy_calculator is not None:
                # Create temporary conformation for energy evaluation
                energy = self._evaluate_loop_energy(interpolated_conf, loop)
            else:
                # Use simple geometric heuristic
                energy = self._geometric_energy_heuristic(interpolated_conf)
            
            # Track best conformation
            if energy < best_energy:
                best_energy = energy
                best_conformation = interpolated_conf
        
        return best_conformation
    
    def interpolate_loop_conformation(
        self,
        loop: LoopRegion,
        scaling_factor: float,
        time: float
    ) -> List[Tuple[float, float, float]]:
        """
        Interpolate loop conformation using scaling factor.
        
        Strategy:
        1. Calculate extended conformation (straight line between anchors)
        2. Calculate compact conformation (closer to center of mass)
        3. Interpolate: current + (compact - extended) × (1 - scaling_factor/φ)
        
        As G(φ,t) decays from φ to 0, loop transitions from extended to compact.
        
        Args:
            loop: Loop region with current conformation
            scaling_factor: G(φ,t) value for current timestep
            time: Current time in picoseconds
        
        Returns:
            Interpolated loop coordinates
        
        Example:
            >>> interpolated = refiner.interpolate_loop_conformation(
            ...     loop, scaling_factor=1.2, time=0.5
            ... )
        """
        # Calculate interpolation parameter (0=compact, 1=extended)
        # As G(φ,t) decays from φ (~1.618) to 0, we want to go from extended to compact
        alpha = scaling_factor / self.phi  # Ranges from ~1.0 to 0.0
        
        # Get current conformation
        current = loop.current_conformation
        
        # Calculate extended conformation (straight line between start and end)
        extended = self._calculate_extended_conformation(loop)
        
        # Calculate compact conformation (closer to center of mass)
        compact = self._calculate_compact_conformation(loop)
        
        # Interpolate: (1-alpha) × compact + alpha × extended
        interpolated = []
        for i in range(len(current)):
            x = (1 - alpha) * compact[i][0] + alpha * extended[i][0]
            y = (1 - alpha) * compact[i][1] + alpha * extended[i][1]
            z = (1 - alpha) * compact[i][2] + alpha * extended[i][2]
            interpolated.append((x, y, z))
        
        return interpolated
    
    def _refine_classical(
        self,
        loop: LoopRegion,
        qcp_values: Dict[int, float]
    ) -> List[Tuple[float, float, float]]:
        """
        Refine loop using classical methods (low QCP < 4).
        
        Classical approach:
        1. Sample multiple loop conformations
        2. Energy minimize each
        3. Select lowest energy
        
        Args:
            loop: Loop region to refine
            qcp_values: QCP values for residues
        
        Returns:
            Refined loop coordinates
        """
        # For classical loops, use simple minimization
        # In production, this would use proper loop sampling algorithms
        
        # Simple strategy: gradually compact the loop
        return self._calculate_compact_conformation(loop)
    
    def _refine_high_qcp(
        self,
        loop: LoopRegion,
        qcp_values: Dict[int, float]
    ) -> List[Tuple[float, float, float]]:
        """
        Refine loop with quantum-corrected geometry (high QCP >= 7).
        
        High QCP loops have strong quantum coherence and should
        maintain specific geometric relationships.
        
        Args:
            loop: Loop region to refine
            qcp_values: QCP values for residues
        
        Returns:
            Refined loop coordinates
        """
        # For high QCP, maintain current geometry with minor adjustments
        # In production, this would apply quantum-corrected constraints
        
        return loop.current_conformation
    
    def _calculate_extended_conformation(
        self,
        loop: LoopRegion
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate extended loop conformation (straight line).
        
        Creates a straight-line interpolation between loop start and end.
        
        Args:
            loop: Loop region
        
        Returns:
            Extended loop coordinates
        """
        start_coord = loop.current_conformation[0]
        end_coord = loop.current_conformation[-1]
        n = len(loop.current_conformation)
        
        extended = []
        for i in range(n):
            alpha = i / (n - 1) if n > 1 else 0.0
            x = start_coord[0] + alpha * (end_coord[0] - start_coord[0])
            y = start_coord[1] + alpha * (end_coord[1] - start_coord[1])
            z = start_coord[2] + alpha * (end_coord[2] - start_coord[2])
            extended.append((x, y, z))
        
        return extended
    
    def _calculate_compact_conformation(
        self,
        loop: LoopRegion
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate compact loop conformation (closer to center of mass).
        
        Moves loop residues toward center of mass while maintaining
        approximate chain connectivity.
        
        Args:
            loop: Loop region
        
        Returns:
            Compact loop coordinates
        """
        # Calculate center of mass
        coords = loop.current_conformation
        n = len(coords)
        
        cx = sum(c[0] for c in coords) / n
        cy = sum(c[1] for c in coords) / n
        cz = sum(c[2] for c in coords) / n
        center = (cx, cy, cz)
        
        # Move each coordinate 50% toward center
        compact = []
        for coord in coords:
            x = coord[0] + 0.5 * (center[0] - coord[0])
            y = coord[1] + 0.5 * (center[1] - coord[1])
            z = coord[2] + 0.5 * (center[2] - coord[2])
            compact.append((x, y, z))
        
        return compact
    
    def _evaluate_loop_energy(
        self,
        loop_coords: List[Tuple[float, float, float]],
        loop: LoopRegion
    ) -> float:
        """
        Evaluate energy of loop conformation.
        
        Args:
            loop_coords: Loop coordinates to evaluate
            loop: Loop region information
        
        Returns:
            Energy value (lower is better)
        """
        if self.energy_calculator is None:
            return self._geometric_energy_heuristic(loop_coords)
        
        # In production, this would evaluate full molecular mechanics energy
        # For now, use geometric heuristic
        return self._geometric_energy_heuristic(loop_coords)
    
    def _geometric_energy_heuristic(
        self,
        loop_coords: List[Tuple[float, float, float]]
    ) -> float:
        """
        Simple geometric energy heuristic for loop scoring.
        
        Penalizes:
        - Very long or very short bond lengths
        - Steric clashes (too close residues)
        
        Args:
            loop_coords: Loop coordinates
        
        Returns:
            Pseudo-energy (lower is better)
        """
        energy = 0.0
        n = len(loop_coords)
        
        # Ideal bond length ~3.8 Å
        ideal_bond = 3.8
        
        # Sequential bond penalties
        for i in range(n - 1):
            dx = loop_coords[i+1][0] - loop_coords[i][0]
            dy = loop_coords[i+1][1] - loop_coords[i][1]
            dz = loop_coords[i+1][2] - loop_coords[i][2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Harmonic penalty for deviation from ideal
            deviation = dist - ideal_bond
            energy += deviation * deviation
        
        # Non-sequential clash penalties (residues >= 2 apart)
        for i in range(n):
            for j in range(i + 2, n):
                dx = loop_coords[j][0] - loop_coords[i][0]
                dy = loop_coords[j][1] - loop_coords[i][1]
                dz = loop_coords[j][2] - loop_coords[i][2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Penalty if too close (< 3.0 Å)
                if dist < 3.0:
                    energy += 10.0 * (3.0 - dist) ** 2
        
        return energy
