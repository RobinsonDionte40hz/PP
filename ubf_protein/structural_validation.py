"""
Structural validation and repair system for UBF protein system.

This module implements validation of protein conformations and repair
mechanisms for common structural issues, including disulfide bond
constraint validation.
"""

from typing import List, Tuple, Optional, Dict, Any
import math

from .models import Conformation, DisulfideBond


class ValidationResult:
    """Result of conformation validation."""
    
    def __init__(self, is_valid: bool, issues: List[str]):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether conformation is valid
            issues: List of validation issue descriptions
        """
        self.is_valid = is_valid
        self.issues = issues
    
    def __repr__(self) -> str:
        if self.is_valid:
            return "ValidationResult(valid)"
        return f"ValidationResult(invalid, {len(self.issues)} issues)"


class StructuralValidation:
    """
    Validates and repairs protein conformations.
    
    Checks for common structural issues like invalid bond lengths,
    steric clashes, backbone discontinuities, and invalid coordinates.
    Provides repair mechanisms for fixable issues.
    """
    
    # Validation thresholds
    MIN_BOND_LENGTH = 1.0  # Å - minimum CA-CA distance
    MAX_BOND_LENGTH = 7.0  # Å - maximum CA-CA distance (relaxed for enhanced physics exploration)
    IDEAL_BOND_LENGTH = 3.8  # Å - ideal CA-CA distance
    MIN_CLASH_DISTANCE = 2.0  # Å - minimum distance to avoid clash
    MAX_COORDINATE = 1000.0  # Å - maximum valid coordinate value
    
    # Note: MAX_BOND_LENGTH = 7.0 Å allows extended conformations during folding:
    # - Accepts structured regions (3.5-4.5 Å) and extended chains (~6-7 Å)
    # - 5.0 Å was too strict and rejected too many valid exploration moves
    # - Energy function penalizes overstretched bonds, so validation can be more permissive
    # - Quality maintained through energy rejection threshold (5000 kcal/mol)
    
    def __init__(self):
        """Initialize structural validator."""
        pass
    
    def validate_conformation(
        self, 
        conformation: Conformation,
        disulfide_bonds: Optional[List[DisulfideBond]] = None
    ) -> ValidationResult:
        """
        Validate a protein conformation for structural integrity.
        
        Checks:
        - Bond lengths between consecutive CA atoms
        - Steric clashes between non-consecutive residues
        - Backbone continuity
        - Coordinate validity (no NaN, inf, or extreme values)
        - Disulfide bond constraints (if provided)
        
        Args:
            conformation: Conformation to validate
            disulfide_bonds: Optional list of disulfide bonds to validate
            
        Returns:
            ValidationResult with validity status and issue descriptions
        """
        issues = []
        
        # Check coordinate validity
        coord_issues = self._check_coordinate_validity(conformation)
        issues.extend(coord_issues)
        
        # Check bond lengths
        bond_issues = self._check_bond_lengths(conformation)
        issues.extend(bond_issues)
        
        # Check for steric clashes
        clash_issues = self._check_steric_clashes(conformation)
        issues.extend(clash_issues)
        
        # Check backbone continuity
        continuity_issues = self._check_backbone_continuity(conformation)
        issues.extend(continuity_issues)
        
        # Check disulfide bond constraints if provided
        if disulfide_bonds:
            disulfide_issues = self._check_disulfide_bonds(conformation, disulfide_bonds)
            issues.extend(disulfide_issues)
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, issues)
    
    def validate_disulfide_bonds(
        self,
        conformation: Conformation,
        disulfide_bonds: List[DisulfideBond]
    ) -> Tuple[bool, List[str]]:
        """
        Validate disulfide bond constraints for a conformation.
        
        Checks that CA-CA distances for all disulfide-bonded cysteine pairs
        are within the acceptable range (target ± tolerance).
        
        This is a standalone method that can be called independently or
        as part of the full validate_conformation() workflow.
        
        Args:
            conformation: Conformation to validate
            disulfide_bonds: List of disulfide bonds to check
            
        Returns:
            Tuple of (is_valid, violation_messages)
            - is_valid: True if all bonds satisfy constraints
            - violation_messages: List of specific violations (empty if valid)
            
        Performance:
            Completes in <5ms for typical proteins (<300 residues, <10 bonds)
            
        Example:
            >>> validator = StructuralValidation()
            >>> bonds = [DisulfideBond(residue_i=5, residue_j=55)]
            >>> is_valid, violations = validator.validate_disulfide_bonds(conf, bonds)
            >>> if not is_valid:
            ...     print(f"Violations: {violations}")
        """
        violations = self._check_disulfide_bonds(conformation, disulfide_bonds)
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _check_disulfide_bonds(
        self,
        conformation: Conformation,
        disulfide_bonds: List[DisulfideBond]
    ) -> List[str]:
        """
        Internal method to check disulfide bond constraints.
        
        Args:
            conformation: Conformation to check
            disulfide_bonds: List of disulfide bonds to validate
            
        Returns:
            List of violation descriptions (empty if all satisfied)
        """
        violations = []
        coords = conformation.atom_coordinates
        seq_len = len(coords)
        
        for bond in disulfide_bonds:
            # Check that indices are within bounds
            if bond.residue_i >= seq_len or bond.residue_j >= seq_len:
                violations.append(
                    f"Disulfide bond {bond.residue_i}-{bond.residue_j}: "
                    f"Index out of bounds (sequence length: {seq_len})"
                )
                continue
            
            # Calculate CA-CA distance
            ca_i = coords[bond.residue_i]
            ca_j = coords[bond.residue_j]
            distance = self._calculate_distance(ca_i, ca_j)
            
            # Check if bond constraint is satisfied
            if not bond.is_satisfied(distance):
                violation_mag = bond.get_violation(distance)
                violations.append(
                    f"Disulfide bond CYS{bond.residue_i}-CYS{bond.residue_j}: "
                    f"Distance {distance:.2f} Å violates constraint "
                    f"(target {bond.distance:.1f}±{bond.tolerance:.1f} Å, "
                    f"violation: {violation_mag:.2f} Å)"
                )
        
        return violations
    
    def repair_conformation(self, conformation: Conformation) -> Tuple[Conformation, bool]:
        """
        Attempt to repair an invalid conformation.
        
        Tries to fix:
        - Invalid bond lengths
        - Steric clashes (minor ones)
        - Backbone discontinuities
        
        Args:
            conformation: Conformation to repair
            
        Returns:
            Tuple of (repaired_conformation, success)
            success is True if all issues were fixed
        """
        # Create a copy to modify
        repaired = Conformation(
            conformation_id=conformation.conformation_id + "_repaired",
            sequence=conformation.sequence,
            atom_coordinates=list(conformation.atom_coordinates),
            energy=conformation.energy,
            rmsd_to_native=conformation.rmsd_to_native,
            secondary_structure=list(conformation.secondary_structure),
            phi_angles=list(conformation.phi_angles),
            psi_angles=list(conformation.psi_angles),
            available_move_types=list(conformation.available_move_types),
            structural_constraints=dict(conformation.structural_constraints)
        )
        
        # Attempt repairs
        repaired = self._fix_bond_lengths(repaired)
        repaired = self._fix_backbone(repaired)
        repaired = self._resolve_clashes(repaired)
        
        # Validate repaired conformation
        result = self.validate_conformation(repaired)
        
        return repaired, result.is_valid
    
    def _check_coordinate_validity(self, conformation: Conformation) -> List[str]:
        """
        Check if coordinates contain invalid values.
        
        Args:
            conformation: Conformation to check
            
        Returns:
            List of issue descriptions
        """
        issues = []
        
        for i, coord in enumerate(conformation.atom_coordinates):
            x, y, z = coord
            
            # Check for NaN or inf
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                issues.append(f"Residue {i}: NaN coordinate detected")
            elif math.isinf(x) or math.isinf(y) or math.isinf(z):
                issues.append(f"Residue {i}: Infinite coordinate detected")
            
            # Check for extreme values
            elif (abs(x) > self.MAX_COORDINATE or 
                  abs(y) > self.MAX_COORDINATE or 
                  abs(z) > self.MAX_COORDINATE):
                issues.append(f"Residue {i}: Coordinate exceeds maximum value ({self.MAX_COORDINATE} Å)")
        
        return issues
    
    def _check_bond_lengths(self, conformation: Conformation) -> List[str]:
        """
        Check CA-CA bond lengths between consecutive residues.
        
        Args:
            conformation: Conformation to check
            
        Returns:
            List of issue descriptions
        """
        issues = []
        coords = conformation.atom_coordinates
        
        for i in range(len(coords) - 1):
            dist = self._calculate_distance(coords[i], coords[i + 1])
            
            if dist < self.MIN_BOND_LENGTH:
                issues.append(f"Bond {i}-{i+1}: Too short ({dist:.2f} Å < {self.MIN_BOND_LENGTH} Å)")
            elif dist > self.MAX_BOND_LENGTH:
                issues.append(f"Bond {i}-{i+1}: Too long ({dist:.2f} Å > {self.MAX_BOND_LENGTH} Å)")
        
        return issues
    
    def _check_steric_clashes(self, conformation: Conformation) -> List[str]:
        """
        Check for steric clashes between non-consecutive residues.
        
        Args:
            conformation: Conformation to check
            
        Returns:
            List of issue descriptions
        """
        issues = []
        coords = conformation.atom_coordinates
        
        # Only check a subset of non-consecutive pairs to avoid O(n²) complexity
        # Check residues at least 3 positions apart
        for i in range(len(coords)):
            for j in range(i + 3, min(i + 10, len(coords))):  # Check next 7 residues
                dist = self._calculate_distance(coords[i], coords[j])
                
                if dist < self.MIN_CLASH_DISTANCE:
                    issues.append(f"Steric clash: Residues {i} and {j} too close ({dist:.2f} Å)")
        
        return issues
    
    def _check_backbone_continuity(self, conformation: Conformation) -> List[str]:
        """
        Check backbone continuity (no major breaks).
        
        Args:
            conformation: Conformation to check
            
        Returns:
            List of issue descriptions
        """
        issues = []
        coords = conformation.atom_coordinates
        
        # Check for major discontinuities (gaps > 2x ideal bond length)
        max_gap = 2 * self.IDEAL_BOND_LENGTH
        
        for i in range(len(coords) - 1):
            dist = self._calculate_distance(coords[i], coords[i + 1])
            
            if dist > max_gap:
                issues.append(f"Backbone break: Gap between residues {i} and {i+1} ({dist:.2f} Å)")
        
        return issues
    
    def _fix_bond_lengths(self, conformation: Conformation) -> Conformation:
        """
        Fix invalid bond lengths by adjusting coordinates.
        
        Uses a simple relaxation approach to bring bonds closer to ideal length.
        
        Args:
            conformation: Conformation to fix
            
        Returns:
            Conformation with adjusted bond lengths
        """
        coords = list(conformation.atom_coordinates)
        
        # Iterate through consecutive pairs
        for i in range(len(coords) - 1):
            dist = self._calculate_distance(coords[i], coords[i + 1])
            
            # If bond is too short or too long, adjust
            if dist < self.MIN_BOND_LENGTH or dist > self.MAX_BOND_LENGTH:
                # Move second residue to ideal distance
                x1, y1, z1 = coords[i]
                x2, y2, z2 = coords[i + 1]
                
                # Calculate direction vector
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                
                # Normalize
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    dz /= dist
                    
                    # Place at ideal distance
                    coords[i + 1] = (
                        x1 + dx * self.IDEAL_BOND_LENGTH,
                        y1 + dy * self.IDEAL_BOND_LENGTH,
                        z1 + dz * self.IDEAL_BOND_LENGTH
                    )
        
        conformation.atom_coordinates = coords
        return conformation
    
    def _resolve_clashes(self, conformation: Conformation) -> Conformation:
        """
        Resolve steric clashes by slightly adjusting coordinates.
        
        Uses a simple repulsion approach to push clashing residues apart.
        
        Args:
            conformation: Conformation to fix
            
        Returns:
            Conformation with reduced clashes
        """
        coords = list(conformation.atom_coordinates)
        
        # Simple clash resolution: push apart if too close
        for i in range(len(coords)):
            for j in range(i + 3, min(i + 10, len(coords))):
                dist = self._calculate_distance(coords[i], coords[j])
                
                if dist < self.MIN_CLASH_DISTANCE and dist > 0:
                    # Calculate repulsion vector
                    x1, y1, z1 = coords[i]
                    x2, y2, z2 = coords[j]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    dz = z2 - z1
                    
                    # Normalize
                    dx /= dist
                    dy /= dist
                    dz /= dist
                    
                    # Push apart by small amount (0.5 Å)
                    push = 0.5
                    coords[j] = (
                        x2 + dx * push,
                        y2 + dy * push,
                        z2 + dz * push
                    )
        
        conformation.atom_coordinates = coords
        return conformation
    
    def _fix_backbone(self, conformation: Conformation) -> Conformation:
        """
        Fix backbone discontinuities.
        
        Interpolates coordinates for major gaps.
        
        Args:
            conformation: Conformation to fix
            
        Returns:
            Conformation with continuous backbone
        """
        coords = list(conformation.atom_coordinates)
        max_gap = 2 * self.IDEAL_BOND_LENGTH
        
        for i in range(len(coords) - 1):
            dist = self._calculate_distance(coords[i], coords[i + 1])
            
            if dist > max_gap:
                # Interpolate to close gap
                x1, y1, z1 = coords[i]
                x2, y2, z2 = coords[i + 1]
                
                # Move second residue closer
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                
                # Normalize and place at ideal distance
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    dz /= dist
                    
                    coords[i + 1] = (
                        x1 + dx * self.IDEAL_BOND_LENGTH,
                        y1 + dy * self.IDEAL_BOND_LENGTH,
                        z1 + dz * self.IDEAL_BOND_LENGTH
                    )
        
        conformation.atom_coordinates = coords
        return conformation
    
    def _calculate_distance(self, coord1: Tuple[float, float, float], 
                          coord2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D coordinates.
        
        Args:
            coord1: First coordinate (x, y, z)
            coord2: Second coordinate (x, y, z)
            
        Returns:
            Distance in Ångströms
        """
        x1, y1, z1 = coord1
        x2, y2, z2 = coord2
        
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        
        return math.sqrt(dx * dx + dy * dy + dz * dz)
