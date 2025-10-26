"""
Local Refinement Module for UBF Protein System.

This module implements gradient descent optimization for local energy minimization
of protein conformations. Uses numerical gradients with central differences and
adaptive step size control.

Target performance: <5s for 100 residues
"""

import math
import copy
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Handle imports for both package and direct execution
try:
    from .interfaces import IPhysicsCalculator
    from .models import Conformation
except ImportError:
    from ubf_protein.interfaces import IPhysicsCalculator
    from ubf_protein.models import Conformation


@dataclass
class RefinementResult:
    """
    Results from local refinement optimization.
    
    Attributes:
        refined_conformation: Optimized conformation
        initial_energy: Starting energy (kcal/mol)
        final_energy: Final energy after optimization (kcal/mol)
        energy_change: Total energy change (kcal/mol)
        n_iterations: Number of gradient descent steps taken
        converged: Whether convergence criterion was met
        reason: Reason for termination (converged, max_iterations, geometry_violation)
        gradient_norm: Final gradient norm
    """
    refined_conformation: Conformation
    initial_energy: float
    final_energy: float
    energy_change: float
    n_iterations: int
    converged: bool
    reason: str
    gradient_norm: float


class LocalRefinement:
    """
    Local refinement optimizer using gradient descent.
    
    Implements energy minimization through:
    1. Numerical gradient calculation (central differences)
    2. Coordinate updates with adaptive step size
    3. Geometry validation after each step
    4. Convergence checking
    
    Features:
    - Adaptive step size: starts at 0.01 Å, reduces by 0.5 on problems
    - Central differences: epsilon = 0.01 Å for gradient accuracy
    - Convergence tolerance: 0.001 kcal/mol energy change
    - Maximum iterations: 100 steps
    - Geometry validation: ensures valid bond lengths and angles
    
    Attributes:
        energy_calculator: Physics calculator for energy evaluation
        initial_step_size: Starting step size in Angstroms (default 0.01)
        epsilon: Finite difference epsilon for gradients (default 0.01)
        convergence_tolerance: Energy change threshold (default 0.001 kcal/mol)
        max_iterations: Maximum optimization steps (default 100)
        step_reduction_factor: Factor to reduce step size on failure (default 0.5)
    
    Example:
        >>> from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
        >>> calc = EnhancedEnergyCalculator(sequence="ACDEFG")
        >>> refiner = LocalRefinement(calc)
        >>> result = refiner.refine(conformation)
        >>> print(f"Energy: {result.initial_energy:.2f} → {result.final_energy:.2f}")
    """
    
    def __init__(self,
                 energy_calculator: IPhysicsCalculator,
                 initial_step_size: float = 0.01,
                 epsilon: float = 0.01,
                 convergence_tolerance: float = 0.001,
                 max_iterations: int = 100,
                 step_reduction_factor: float = 0.5,
                 min_step_size: float = 1e-6):
        """
        Initialize local refinement optimizer.
        
        Args:
            energy_calculator: Physics calculator implementing IPhysicsCalculator
            initial_step_size: Starting step size in Å (default 0.01)
            epsilon: Finite difference epsilon for gradients in Å (default 0.01)
            convergence_tolerance: Energy change threshold in kcal/mol (default 0.001)
            max_iterations: Maximum optimization steps (default 100)
            step_reduction_factor: Factor to reduce step on failure (default 0.5)
            min_step_size: Minimum allowed step size in Å (default 1e-6)
        
        Raises:
            ValueError: If parameters are invalid
        """
        if initial_step_size <= 0:
            raise ValueError(f"initial_step_size must be positive, got {initial_step_size}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if convergence_tolerance < 0:
            raise ValueError(f"convergence_tolerance must be non-negative, got {convergence_tolerance}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
        if not 0 < step_reduction_factor < 1:
            raise ValueError(f"step_reduction_factor must be in (0, 1), got {step_reduction_factor}")
        if min_step_size <= 0:
            raise ValueError(f"min_step_size must be positive, got {min_step_size}")
        
        self.energy_calculator = energy_calculator
        self.initial_step_size = initial_step_size
        self.epsilon = epsilon
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.step_reduction_factor = step_reduction_factor
        self.min_step_size = min_step_size
    
    def refine(self, conformation: Conformation) -> RefinementResult:
        """
        Refine conformation through gradient descent optimization.
        
        Performs iterative energy minimization:
        1. Calculate numerical gradient
        2. Update coordinates along negative gradient
        3. Validate geometry
        4. Check convergence
        5. Reduce step size if needed
        
        Args:
            conformation: Initial conformation to refine
            
        Returns:
            RefinementResult with optimization details
        """
        # Initialize
        current_conf = conformation
        current_energy = self.energy_calculator.calculate(current_conf)
        initial_energy = current_energy
        step_size = self.initial_step_size
        
        for iteration in range(self.max_iterations):
            # Calculate gradient
            gradient = self._calculate_gradient(current_conf)
            gradient_norm = self._calculate_norm(gradient)
            
            # Try to update coordinates
            updated_conf = self._update_coordinates(current_conf, gradient, step_size)
            
            # Validate geometry
            is_valid = self._validate_geometry(updated_conf)
            
            if not is_valid:
                # Reduce step size and retry
                step_size *= self.step_reduction_factor
                
                if step_size < self.min_step_size:
                    # Step size too small, terminate
                    return RefinementResult(
                        refined_conformation=current_conf,
                        initial_energy=initial_energy,
                        final_energy=current_energy,
                        energy_change=current_energy - initial_energy,
                        n_iterations=iteration,
                        converged=False,
                        reason="step_size_too_small",
                        gradient_norm=gradient_norm
                    )
                continue
            
            # Calculate new energy
            new_energy = self.energy_calculator.calculate(updated_conf)
            
            # Check if energy increased
            if new_energy > current_energy:
                # Reduce step size and retry
                step_size *= self.step_reduction_factor
                
                if step_size < self.min_step_size:
                    return RefinementResult(
                        refined_conformation=current_conf,
                        initial_energy=initial_energy,
                        final_energy=current_energy,
                        energy_change=current_energy - initial_energy,
                        n_iterations=iteration,
                        converged=False,
                        reason="step_size_too_small",
                        gradient_norm=gradient_norm
                    )
                continue
            
            # Accept update
            energy_change = abs(new_energy - current_energy)
            current_conf = updated_conf
            current_energy = new_energy
            
            # Check convergence
            if energy_change < self.convergence_tolerance:
                return RefinementResult(
                    refined_conformation=current_conf,
                    initial_energy=initial_energy,
                    final_energy=current_energy,
                    energy_change=current_energy - initial_energy,
                    n_iterations=iteration + 1,
                    converged=True,
                    reason="converged",
                    gradient_norm=gradient_norm
                )
        
        # Maximum iterations reached
        gradient = self._calculate_gradient(current_conf)
        gradient_norm = self._calculate_norm(gradient)
        
        return RefinementResult(
            refined_conformation=current_conf,
            initial_energy=initial_energy,
            final_energy=current_energy,
            energy_change=current_energy - initial_energy,
            n_iterations=self.max_iterations,
            converged=False,
            reason="max_iterations",
            gradient_norm=gradient_norm
        )
    
    def _calculate_gradient(self, conformation: Conformation) -> List[Tuple[float, float, float]]:
        """
        Calculate numerical gradient using central differences.
        
        For each coordinate x_i:
        ∂E/∂x_i ≈ (E(x_i + ε) - E(x_i - ε)) / (2ε)
        
        Args:
            conformation: Current conformation
            
        Returns:
            List of gradient vectors (∂E/∂x, ∂E/∂y, ∂E/∂z) for each atom
        """
        coords = conformation.atom_coordinates
        n_atoms = len(coords)
        gradient = []
        
        for i in range(n_atoms):
            grad_x = self._partial_derivative(conformation, i, 0)
            grad_y = self._partial_derivative(conformation, i, 1)
            grad_z = self._partial_derivative(conformation, i, 2)
            
            gradient.append((grad_x, grad_y, grad_z))
        
        return gradient
    
    def _partial_derivative(self, conformation: Conformation, atom_idx: int, 
                           coord_idx: int) -> float:
        """
        Calculate partial derivative using central differences.
        
        Args:
            conformation: Current conformation
            atom_idx: Index of atom
            coord_idx: Coordinate index (0=x, 1=y, 2=z)
            
        Returns:
            Partial derivative ∂E/∂coord
        """
        # Create forward perturbation
        coords_forward = [list(c) for c in conformation.atom_coordinates]
        coords_forward[atom_idx][coord_idx] += self.epsilon
        conf_forward = self._create_conformation(conformation, coords_forward)
        energy_forward = self.energy_calculator.calculate(conf_forward)
        
        # Create backward perturbation
        coords_backward = [list(c) for c in conformation.atom_coordinates]
        coords_backward[atom_idx][coord_idx] -= self.epsilon
        conf_backward = self._create_conformation(conformation, coords_backward)
        energy_backward = self.energy_calculator.calculate(conf_backward)
        
        # Central difference
        derivative = (energy_forward - energy_backward) / (2.0 * self.epsilon)
        
        return derivative
    
    def _update_coordinates(self, conformation: Conformation,
                           gradient: List[Tuple[float, float, float]],
                           step_size: float) -> Conformation:
        """
        Update coordinates along negative gradient direction.
        
        new_coords = old_coords - step_size * gradient
        
        Args:
            conformation: Current conformation
            gradient: Gradient vectors for each atom
            step_size: Step size in Angstroms
            
        Returns:
            New conformation with updated coordinates
        """
        old_coords = conformation.atom_coordinates
        new_coords = []
        
        for i, (x, y, z) in enumerate(old_coords):
            grad_x, grad_y, grad_z = gradient[i]
            
            # Move in negative gradient direction
            new_x = x - step_size * grad_x
            new_y = y - step_size * grad_y
            new_z = z - step_size * grad_z
            
            new_coords.append((new_x, new_y, new_z))
        
        return self._create_conformation(conformation, new_coords)
    
    def _create_conformation(self, template: Conformation,
                            new_coords: List[List[float]]) -> Conformation:
        """
        Create new conformation with updated coordinates.
        
        Args:
            template: Template conformation
            new_coords: New coordinates
            
        Returns:
            New conformation
        """
        # Convert to proper tuples with exactly 3 elements
        coord_tuples: List[Tuple[float, float, float]] = []
        for c in new_coords:
            if len(c) == 3:
                coord_tuples.append((float(c[0]), float(c[1]), float(c[2])))
            else:
                coord_tuples.append((float(c[0]), float(c[1]), float(c[2])))
        
        return Conformation(
            conformation_id=template.conformation_id,
            sequence=template.sequence,
            atom_coordinates=coord_tuples,
            energy=template.energy,
            rmsd_to_native=template.rmsd_to_native,
            secondary_structure=template.secondary_structure,
            phi_angles=template.phi_angles,
            psi_angles=template.psi_angles,
            available_move_types=template.available_move_types,
            structural_constraints=template.structural_constraints
        )
    
    def _validate_geometry(self, conformation: Conformation) -> bool:
        """
        Validate geometry of conformation.
        
        Checks:
        - Bond lengths between consecutive atoms (2.5-5.0 Å)
        - No extreme coordinate values (< 1000 Å)
        
        Args:
            conformation: Conformation to validate
            
        Returns:
            True if geometry is valid, False otherwise
        """
        coords = conformation.atom_coordinates
        
        # Check coordinate magnitudes
        for x, y, z in coords:
            if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 1000:
                return False
        
        # Check consecutive bond lengths
        for i in range(len(coords) - 1):
            x1, y1, z1 = coords[i]
            x2, y2, z2 = coords[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Bond length should be reasonable (2.5-7.0 Å for CA-CA)
            if dist < 2.5 or dist > 7.0:
                return False
        
        return True
    
    def _calculate_norm(self, gradient: List[Tuple[float, float, float]]) -> float:
        """
        Calculate L2 norm of gradient.
        
        ||grad|| = sqrt(Σ(grad_x² + grad_y² + grad_z²))
        
        Args:
            gradient: Gradient vectors
            
        Returns:
            L2 norm of gradient
        """
        sum_squares = 0.0
        
        for grad_x, grad_y, grad_z in gradient:
            sum_squares += grad_x**2 + grad_y**2 + grad_z**2
        
        return math.sqrt(sum_squares)
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current optimization parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'initial_step_size': self.initial_step_size,
            'epsilon': self.epsilon,
            'convergence_tolerance': self.convergence_tolerance,
            'max_iterations': self.max_iterations,
            'step_reduction_factor': self.step_reduction_factor,
            'min_step_size': self.min_step_size
        }


if __name__ == '__main__':
    # Simple test
    from ubf_protein.enhanced_energy_calculator import EnhancedEnergyCalculator
    from ubf_protein.models import Conformation
    
    print("Testing LocalRefinement...")
    
    # Create test sequence
    sequence = "AAAAA"
    
    # Create energy calculator
    calc = EnhancedEnergyCalculator(sequence)
    
    # Create refiner
    refiner = LocalRefinement(calc)
    
    # Create test conformation with slightly perturbed geometry
    coords = [
        (0.0, 0.0, 0.0),
        (3.9, 0.1, 0.0),  # Slightly off
        (7.7, 0.2, 0.0),
        (11.5, 0.1, 0.0),
        (15.3, 0.0, 0.0)
    ]
    
    conf = Conformation(
        conformation_id="test",
        sequence=sequence,
        atom_coordinates=coords,
        energy=-100.0,
        rmsd_to_native=None,
        secondary_structure=['C'] * 5,
        phi_angles=[0.0] * 5,
        psi_angles=[0.0] * 5,
        available_move_types=[],
        structural_constraints={}
    )
    
    print(f"\nInitial energy: {calc.calculate(conf):.4f} kcal/mol")
    print(f"Refining with gradient descent...")
    
    # Refine
    result = refiner.refine(conf)
    
    print(f"\nRefinement Results:")
    print(f"  Initial energy:  {result.initial_energy:.4f} kcal/mol")
    print(f"  Final energy:    {result.final_energy:.4f} kcal/mol")
    print(f"  Energy change:   {result.energy_change:.4f} kcal/mol")
    print(f"  Iterations:      {result.n_iterations}")
    print(f"  Converged:       {result.converged}")
    print(f"  Reason:          {result.reason}")
    print(f"  Gradient norm:   {result.gradient_norm:.6f}")
    
    # Display parameters
    print(f"\nOptimization parameters:")
    params = refiner.get_parameters()
    for name, value in params.items():
        print(f"  {name:25s}: {value}")
    
    # Test performance
    print(f"\n✅ LocalRefinement test complete!")
