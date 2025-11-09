"""
Tertiary Contact Predictor for UBF Protein System

This module predicts and enforces long-range tertiary contacts using quantum
resonance coupling between residue pairs. Tertiary contacts are critical for
protein fold stability and are predicted using physics-based calculations.

Key Concepts:
    - Tertiary Contact: Long-range interaction between distant residues (≥5 apart)
    - Resonance Coupling: Quantum energy matching between residue pairs
    - Contact Map: Matrix of predicted residue-residue proximities
    - Contact Enforcement: Attractive forces to form missing contacts

Physics Background:
    - Resonance coupling captures quantum energy correlations
    - γ frequency (40 Hz) represents fundamental biological oscillation
    - G(φ,t) describes golden ratio temporal evolution
    - Strong resonance (>0.7) indicates probable contact formation

Resonance Coupling Formula:
    R(E₁,E₂,t) = exp[-(E₁(t) - E₂(t) - ℏωγ)²/(2ℏωγ)] × G(φ,t)
    
    where:
        - E₁, E₂: Quantum energies from QCP
        - ωγ = 2π × 40 Hz (gamma frequency)
        - ℏ = 1.0545718 × 10⁻³⁴ J·s (Planck's constant)
        - G(φ,t) = exp(-t/τ_c) × φ, τ_c = 408 fs
        - φ = 1.618033988749895 (golden ratio)

Contact Criteria:
    - Resonance strength > 0.7 (probable contact)
    - Sequence separation ≥ 5 residues (long-range)
    - Spatial distance < 8.0Å (feasible contact)
    - Predicted distance ≈ 6.0Å (optimal contact)

Force Application:
    - Attractive force for missing contacts (distance > 8Å)
    - Magnitude: (distance - 6.0) × 10.0
    - Momentum conservation: equal and opposite forces
    - Applied along unit vector between residues

Performance Targets:
    - Contact prediction: <100ms for 100 residues
    - Contact enforcement: <50ms per optimization step
    - Memory: <2MB for contact map

Requirements Addressed:
    - 5.1: Calculate quantum energy for pairs ≥5 positions apart
    - 5.2: Calculate R(E₁,E₂,t) with 40 Hz gamma frequency
    - 5.3: Classify resonance > 0.7 as probable contact
    - 5.4: Verify spatial distance < 8.0Å
    - 5.5: Return residue indices and resonance strength
    - 8.1-8.5: Contact map enforcement with forces
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import logging

try:
    from .models import Conformation, TertiaryContact
    from .qcpp_integration import QCPPIntegrationAdapter
except ImportError:
    from ubf_protein.models import Conformation, TertiaryContact
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter

# Setup logger
logger = logging.getLogger(__name__)


class TertiaryContactPredictor:
    """
    Predicts and enforces tertiary contacts using quantum resonance coupling.
    
    This predictor uses quantum mechanics principles to identify long-range
    residue-residue interactions that stabilize protein folds. Resonance
    coupling calculations reveal probable contacts that should form during
    folding.
    
    The prediction process:
        1. Calculate QCP-based quantum energies for all residue pairs
        2. Compute resonance coupling R(E₁,E₂,t) using 40 Hz gamma frequency
        3. Filter contacts by resonance > 0.7 and separation ≥ 5
        4. Validate feasibility using spatial distance < 8.0Å
        5. Return predicted contacts with resonance strengths
    
    The enforcement process:
        1. Identify missing contacts (predicted but not formed)
        2. Calculate attractive force vectors between residue pairs
        3. Apply force magnitude: (distance - 6.0) × 10.0
        4. Maintain momentum conservation (equal/opposite forces)
        5. Update structure coordinates based on forces
    
    Attributes:
        qcpp_adapter: QCPP integration for QCP and energy calculations
        phi: Golden ratio constant (1.618033988749895)
        h_bar: Reduced Planck constant (1.0545718e-34 J·s)
        gamma_frequency: Gamma oscillation frequency (40.0 Hz)
        coherence_time: Coherence time (408e-15 seconds)
        resonance_threshold: Minimum resonance for contact (0.7)
        min_sequence_separation: Minimum residue separation (5)
        max_contact_distance: Maximum feasible distance (8.0 Å)
        optimal_contact_distance: Target contact distance (6.0 Å)
        force_constant: Force magnitude scaling (10.0)
    
    Example:
        >>> predictor = TertiaryContactPredictor(qcpp_adapter)
        >>> contacts = predictor.predict_tertiary_contacts_quantum(
        ...     sequence="ACDEFGHIKLMNPQRSTVWY",
        ...     qcp_values={0: 8.5, 5: 9.2, 10: 8.8, 15: 9.5}
        ... )
        >>> print(f"Predicted {len(contacts)} tertiary contacts")
        >>> 
        >>> refined_structure = predictor.enforce_contact_map(
        ...     structure=coarse_structure,
        ...     predicted_contacts=contacts
        ... )
    """
    
    def __init__(
        self,
        qcpp_adapter: Optional[QCPPIntegrationAdapter] = None,
        phi: float = 1.618033988749895,
        h_bar: float = 1.0545718e-34,  # J·s
        gamma_frequency: float = 40.0,  # Hz
        coherence_time: float = 408e-15,  # seconds
        resonance_threshold: float = 0.7,
        min_sequence_separation: int = 5,
        max_contact_distance: float = 8.0,  # Ångströms
        optimal_contact_distance: float = 6.0,  # Ångströms
        force_constant: float = 10.0
    ):
        """
        Initialize tertiary contact predictor.
        
        Args:
            qcpp_adapter: Optional QCPP integration for QCP calculations
            phi: Golden ratio constant (default: 1.618033988749895)
            h_bar: Reduced Planck constant (default: 1.0545718e-34)
            gamma_frequency: Gamma frequency in Hz (default: 40.0)
            coherence_time: Coherence time in seconds (default: 408e-15)
            resonance_threshold: Minimum resonance for contact (default: 0.7)
            min_sequence_separation: Minimum residue separation (default: 5)
            max_contact_distance: Maximum feasible distance in Å (default: 8.0)
            optimal_contact_distance: Target contact distance in Å (default: 6.0)
            force_constant: Force magnitude scaling factor (default: 10.0)
        """
        self.qcpp_adapter = qcpp_adapter
        self.phi = phi
        self.h_bar = h_bar
        self.gamma_frequency = gamma_frequency
        self.omega_gamma = 2.0 * math.pi * gamma_frequency  # Angular frequency
        self.coherence_time = coherence_time
        self.resonance_threshold = resonance_threshold
        self.min_sequence_separation = min_sequence_separation
        self.max_contact_distance = max_contact_distance
        self.optimal_contact_distance = optimal_contact_distance
        self.force_constant = force_constant
        
        logger.info(
            f"Initialized TertiaryContactPredictor with gamma={gamma_frequency}Hz, "
            f"resonance_threshold={resonance_threshold}"
        )
    
    def calculate_resonance_coupling(
        self,
        energy_i: float,
        energy_j: float,
        time: float = 0.0
    ) -> float:
        """
        Calculate resonance coupling R(E₁,E₂,t) between two residues.
        
        This method implements the quantum resonance coupling formula:
            R(E₁,E₂,t) = exp[-(E₁ - E₂ - ℏωγ)²/(2ℏωγ)] × G(φ,t)
        
        where G(φ,t) = exp(-t/τ_c) × φ is the golden ratio temporal evolution.
        
        The resonance captures the degree to which two residues' quantum
        energies are matched at the gamma frequency (40 Hz). Strong resonance
        (>0.7) indicates the residues are likely to form a stable contact.
        
        Physical Interpretation:
            - Resonance near 1.0: Perfect energy matching at gamma frequency
            - Resonance > 0.7: Strong coupling, probable contact
            - Resonance < 0.7: Weak coupling, unlikely contact
            - Time evolution: Coherence decays with τ_c = 408 fs
        
        Args:
            energy_i: Quantum energy of residue i (arbitrary units from QCP)
            energy_j: Quantum energy of residue j (arbitrary units from QCP)
            time: Time in seconds (default: 0.0 for static analysis)
        
        Returns:
            Resonance coupling strength (0-1 range, can exceed 1 briefly)
        
        Example:
            >>> predictor = TertiaryContactPredictor()
            >>> # Perfect resonance at gamma frequency
            >>> E1 = 0.0
            >>> E2 = predictor.h_bar * predictor.omega_gamma
            >>> resonance = predictor.calculate_resonance_coupling(E1, E2)
            >>> print(f"Resonance: {resonance:.3f}")  # Should be near 1.0
            >>> 
            >>> # Poor resonance (large energy mismatch)
            >>> E3 = 100.0
            >>> resonance2 = predictor.calculate_resonance_coupling(E1, E3)
            >>> print(f"Resonance: {resonance2:.3f}")  # Should be near 0.0
        """
        # Energy difference from resonance condition
        h_omega_gamma = self.h_bar * self.omega_gamma
        energy_diff = energy_i - energy_j - h_omega_gamma
        
        # Gaussian envelope for resonance matching
        # exp[-(ΔE)²/(2ℏωγ)] peaks when E₁ - E₂ = ℏωγ
        exponent = -(energy_diff ** 2) / (2.0 * h_omega_gamma)
        resonance_factor = math.exp(exponent)
        
        # Golden ratio temporal evolution G(φ,t)
        # Coherence decays with characteristic time τ_c = 408 fs
        temporal_factor = math.exp(-time / self.coherence_time) * self.phi
        
        # Combined resonance coupling
        resonance = resonance_factor * temporal_factor
        
        logger.debug(
            f"Resonance coupling: E_diff={energy_diff:.2e}, "
            f"resonance={resonance:.4f}"
        )
        
        return resonance
    
    def predict_tertiary_contacts_quantum(
        self,
        sequence: str,
        qcp_values: Dict[int, float],
        structure: Optional[Conformation] = None,
        time: float = 0.0
    ) -> List[TertiaryContact]:
        """
        Predict tertiary contacts using quantum resonance coupling.
        
        This method predicts long-range residue-residue contacts by:
            1. Calculating quantum energies from QCP values
            2. Computing resonance coupling for all pairs ≥5 apart
            3. Filtering by resonance threshold (>0.7)
            4. Validating spatial feasibility (distance <8Å if structure given)
            5. Assigning predicted optimal distance (6.0Å)
        
        The prediction is physics-based and does not require machine learning
        or homology information. It captures quantum energy correlations that
        drive protein folding.
        
        Contact Selection Criteria:
            - Sequence separation ≥ min_sequence_separation (default 5)
            - Resonance coupling > resonance_threshold (default 0.7)
            - Spatial distance < max_contact_distance (default 8Å, if structure given)
            - Both residues have QCP values available
        
        Args:
            sequence: Amino acid sequence (single-letter codes)
            qcp_values: Dictionary mapping residue index to QCP value
            structure: Optional current structure for distance validation
            time: Time in seconds for temporal evolution (default: 0.0)
        
        Returns:
            List of predicted tertiary contacts sorted by resonance strength
            (highest resonance first)
        
        Raises:
            ValueError: If sequence is empty or QCP values are insufficient
        
        Example:
            >>> predictor = TertiaryContactPredictor(qcpp_adapter)
            >>> sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues
            >>> qcp_values = {0: 8.5, 5: 9.2, 10: 8.8, 15: 9.5}
            >>> contacts = predictor.predict_tertiary_contacts_quantum(
            ...     sequence=sequence,
            ...     qcp_values=qcp_values
            ... )
            >>> for contact in contacts[:5]:  # Show top 5
            ...     print(f"Contact {contact.residue_i}-{contact.residue_j}: "
            ...           f"resonance={contact.resonance_strength:.3f}")
        """
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        
        if len(qcp_values) < 2:
            raise ValueError(f"Need at least 2 QCP values, got {len(qcp_values)}")
        
        logger.info(
            f"Predicting tertiary contacts for {len(sequence)} residues, "
            f"{len(qcp_values)} with QCP values"
        )
        
        contacts = []
        
        # Iterate over all residue pairs with sufficient sequence separation
        residue_indices = sorted(qcp_values.keys())
        
        for i in range(len(residue_indices)):
            for j in range(i + 1, len(residue_indices)):
                res_i = residue_indices[i]
                res_j = residue_indices[j]
                
                # Check sequence separation
                separation = abs(res_j - res_i)
                if separation < self.min_sequence_separation:
                    continue
                
                # Get QCP values (use as proxy for quantum energy)
                qcp_i = qcp_values[res_i]
                qcp_j = qcp_values[res_j]
                
                # Calculate resonance coupling
                # Scale QCP to energy range: QCP × ℏωγ to match formula scale
                # This ensures resonance values are in meaningful range
                energy_scale = self.h_bar * self.omega_gamma
                energy_i = qcp_i * energy_scale
                energy_j = qcp_j * energy_scale
                
                resonance = self.calculate_resonance_coupling(
                    energy_i=energy_i,
                    energy_j=energy_j,
                    time=time
                )
                
                # Normalize resonance to [0, 1] range
                # φ × exp(...) can exceed 1.0, so we need to normalize
                # We use φ as the maximum theoretical value
                resonance_normalized = min(resonance / self.phi, 1.0)
                
                # Normalize resonance to [0, 1] range
                # φ × exp(...) can exceed 1.0, so we need to normalize
                # We use φ as the maximum theoretical value
                resonance_normalized = min(resonance / self.phi, 1.0)
                
                # Filter by resonance threshold
                if resonance_normalized < self.resonance_threshold:
                    continue
                
                # Validate spatial feasibility if structure provided
                if structure is not None:
                    distance = self._calculate_distance(
                        structure.atom_coordinates[res_i],
                        structure.atom_coordinates[res_j]
                    )
                    
                    # Skip if distance too large (not feasible)
                    if distance > self.max_contact_distance:
                        logger.debug(
                            f"Contact {res_i}-{res_j} too far: {distance:.2f}Å"
                        )
                        continue
                
                # Create contact prediction
                contact = TertiaryContact(
                    residue_i=res_i,
                    residue_j=res_j,
                    resonance_strength=resonance_normalized,
                    predicted_distance=self.optimal_contact_distance,
                    sequence_separation=separation
                )
                
                contacts.append(contact)
                
                logger.debug(
                    f"Predicted contact {res_i}-{res_j}: "
                    f"resonance={resonance_normalized:.4f}, separation={separation}"
                )
        
        # Sort by resonance strength (highest first)
        contacts.sort(key=lambda c: c.resonance_strength, reverse=True)
        
        logger.info(f"Predicted {len(contacts)} tertiary contacts")
        
        return contacts
    
    def enforce_contact_map(
        self,
        structure: Conformation,
        predicted_contacts: List[TertiaryContact],
        max_force: float = 100.0
    ) -> Conformation:
        """
        Enforce predicted contact map by applying attractive forces.
        
        This method identifies missing contacts (predicted but not yet formed)
        and applies attractive forces to bring residues together. Forces are
        applied along the vector connecting residue pairs, with magnitude
        proportional to distance deviation from optimal.
        
        Force Calculation:
            - Only applied to missing contacts (distance > 8Å)
            - Magnitude: (distance - 6.0) × force_constant
            - Direction: Unit vector from i to j
            - Momentum conservation: Equal and opposite forces
            - Capped at max_force to prevent instability
        
        The enforcement process:
            1. Calculate current distances for all predicted contacts
            2. Identify missing contacts (distance > 8Å)
            3. For each missing contact:
                a. Calculate distance and deviation from optimal (6.0Å)
                b. Compute force magnitude: (distance - 6.0) × 10.0
                c. Calculate unit vector from residue i to j
                d. Apply force to residue j, opposite force to i
            4. Update coordinates based on accumulated forces
            5. Return modified structure
        
        Args:
            structure: Current protein conformation
            predicted_contacts: List of predicted contacts to enforce
            max_force: Maximum force magnitude to apply (default: 100.0)
        
        Returns:
            Modified conformation with updated coordinates
        
        Raises:
            ValueError: If structure or predicted_contacts are invalid
        
        Example:
            >>> predictor = TertiaryContactPredictor()
            >>> contacts = predictor.predict_tertiary_contacts_quantum(
            ...     sequence="ACDEFGHIKLMNPQRSTVWY",
            ...     qcp_values={0: 8.5, 5: 9.2, 10: 8.8, 15: 9.5}
            ... )
            >>> refined = predictor.enforce_contact_map(
            ...     structure=coarse_structure,
            ...     predicted_contacts=contacts
            ... )
            >>> print(f"Enforced {len(contacts)} contacts")
        
        Notes:
            - This is a single-step force application
            - For full optimization, call repeatedly or integrate with optimizer
            - Forces are applied simultaneously (not sequentially)
            - Momentum is conserved for each contact pair
        """
        if not predicted_contacts:
            logger.warning("No predicted contacts to enforce")
            # Still return a copy with updated ID
            return Conformation(
                conformation_id=structure.conformation_id + "_enforced",
                sequence=structure.sequence,
                atom_coordinates=structure.atom_coordinates[:],
                energy=structure.energy,
                rmsd_to_native=structure.rmsd_to_native,
                secondary_structure=structure.secondary_structure,
                phi_angles=structure.phi_angles,
                psi_angles=structure.psi_angles,
                available_move_types=structure.available_move_types,
                structural_constraints=structure.structural_constraints,
                energy_components=structure.energy_components,
                native_structure_ref=structure.native_structure_ref,
                gdt_ts_score=structure.gdt_ts_score,
                tm_score=structure.tm_score
            )
        
        logger.info(
            f"Enforcing {len(predicted_contacts)} predicted contacts"
        )
        
        # Initialize force accumulator (zero forces)
        forces = [(0.0, 0.0, 0.0) for _ in structure.atom_coordinates]
        
        # Count missing and formed contacts
        missing_count = 0
        formed_count = 0
        
        # Calculate forces for each predicted contact
        for contact in predicted_contacts:
            res_i = contact.residue_i
            res_j = contact.residue_j
            
            # Get current coordinates
            coord_i = structure.atom_coordinates[res_i]
            coord_j = structure.atom_coordinates[res_j]
            
            # Calculate current distance
            distance = self._calculate_distance(coord_i, coord_j)
            
            # Only enforce missing contacts (distance > threshold)
            if distance <= self.max_contact_distance:
                formed_count += 1
                continue
            
            missing_count += 1
            
            # Calculate force magnitude
            # Larger distance = stronger attractive force
            deviation = distance - self.optimal_contact_distance
            force_magnitude = min(deviation * self.force_constant, max_force)
            
            # Calculate unit vector from i to j
            dx = coord_j[0] - coord_i[0]
            dy = coord_j[1] - coord_i[1]
            dz = coord_j[2] - coord_i[2]
            
            # Normalize to unit vector
            unit_x = dx / distance
            unit_y = dy / distance
            unit_z = dz / distance
            
            # Force on j (toward i, so negative direction)
            force_j = (
                -force_magnitude * unit_x,
                -force_magnitude * unit_y,
                -force_magnitude * unit_z
            )
            
            # Force on i (equal and opposite, toward j)
            force_i = (
                force_magnitude * unit_x,
                force_magnitude * unit_y,
                force_magnitude * unit_z
            )
            
            # Accumulate forces (momentum conservation)
            forces[res_i] = (
                forces[res_i][0] + force_i[0],
                forces[res_i][1] + force_i[1],
                forces[res_i][2] + force_i[2]
            )
            
            forces[res_j] = (
                forces[res_j][0] + force_j[0],
                forces[res_j][1] + force_j[1],
                forces[res_j][2] + force_j[2]
            )
            
            logger.debug(
                f"Contact {res_i}-{res_j}: distance={distance:.2f}Å, "
                f"force={force_magnitude:.2f}"
            )
        
        logger.info(
            f"Contact enforcement: {formed_count} already formed, "
            f"{missing_count} missing (forces applied)"
        )
        
        # Apply forces to update coordinates
        # Simple Euler integration: x_new = x_old + force × step_size
        # Use small step size to maintain stability
        step_size = 0.01  # Ångströms per force unit
        
        new_coordinates = []
        for i, (coord, force) in enumerate(zip(structure.atom_coordinates, forces)):
            new_coord = (
                coord[0] + force[0] * step_size,
                coord[1] + force[1] * step_size,
                coord[2] + force[2] * step_size
            )
            new_coordinates.append(new_coord)
        
        # Create updated conformation
        # Deep copy to avoid modifying original
        updated_structure = Conformation(
            conformation_id=structure.conformation_id + "_enforced",
            sequence=structure.sequence,
            atom_coordinates=new_coordinates,
            energy=structure.energy,  # Energy will be recalculated
            rmsd_to_native=structure.rmsd_to_native,
            secondary_structure=structure.secondary_structure,
            phi_angles=structure.phi_angles,
            psi_angles=structure.psi_angles,
            available_move_types=structure.available_move_types,
            structural_constraints=structure.structural_constraints,
            energy_components=structure.energy_components,
            native_structure_ref=structure.native_structure_ref,
            gdt_ts_score=structure.gdt_ts_score,
            tm_score=structure.tm_score
        )
        
        logger.info("Contact map enforcement complete")
        
        return updated_structure
    
    def _calculate_distance(
        self,
        coord_i: Tuple[float, float, float],
        coord_j: Tuple[float, float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two 3D coordinates.
        
        Args:
            coord_i: (x, y, z) coordinates of first point
            coord_j: (x, y, z) coordinates of second point
        
        Returns:
            Distance in Ångströms
        """
        dx = coord_j[0] - coord_i[0]
        dy = coord_j[1] - coord_i[1]
        dz = coord_j[2] - coord_i[2]
        
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def get_current_contacts(
        self,
        structure: Conformation,
        distance_threshold: float = 8.0
    ) -> List[Tuple[int, int, float]]:
        """
        Get list of current contacts in structure.
        
        A contact is defined as two residues within distance_threshold.
        
        Args:
            structure: Current protein conformation
            distance_threshold: Maximum distance for contact (default: 8.0Å)
        
        Returns:
            List of (residue_i, residue_j, distance) tuples for all contacts
        """
        contacts = []
        n_residues = len(structure.atom_coordinates)
        
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                distance = self._calculate_distance(
                    structure.atom_coordinates[i],
                    structure.atom_coordinates[j]
                )
                
                if distance <= distance_threshold:
                    contacts.append((i, j, distance))
        
        return contacts
    
    def calculate_contact_satisfaction(
        self,
        structure: Conformation,
        predicted_contacts: List[TertiaryContact]
    ) -> float:
        """
        Calculate percentage of predicted contacts that are satisfied.
        
        A contact is satisfied if current distance < max_contact_distance.
        
        Args:
            structure: Current protein conformation
            predicted_contacts: List of predicted contacts
        
        Returns:
            Satisfaction rate (0.0-1.0)
        """
        if not predicted_contacts:
            return 1.0
        
        satisfied = 0
        
        for contact in predicted_contacts:
            distance = self._calculate_distance(
                structure.atom_coordinates[contact.residue_i],
                structure.atom_coordinates[contact.residue_j]
            )
            
            if distance <= self.max_contact_distance:
                satisfied += 1
        
        return satisfied / len(predicted_contacts)
