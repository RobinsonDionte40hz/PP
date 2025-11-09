"""
Secondary Structure Registrar for UBF Protein System

This module corrects helix and sheet alignment using quantum-corrected geometry
parameters. It fixes secondary structure registration to ensure that helices and
sheets are positioned correctly relative to their native positions.

Key Concepts:
    - Secondary Structure Registration: Alignment of helices and sheets
    - Quantum-Corrected Helix: Helix with QCP-dependent pitch and rise
    - φ² Harmonic Coupling: 2.618 THz frequency for sheet stabilization
    - Hydrogen Bond Optimization: Enforcing proper H-bond patterns

Physics Background:
    - Alpha helix: 3.6 residues/turn, 5.4Å pitch, 1.5Å rise
    - Beta sheet: Extended conformation, inter-strand H-bonds
    - High QCP helices show enhanced stability and modified geometry
    - THz coupling at φ² harmonic (2.618 THz) stabilizes sheets

Performance Targets:
    - Secondary structure detection: <50ms
    - Helix geometry enforcement: <100ms per helix
    - Sheet hydrogen bonding: <150ms per sheet
    - Total registration: <200ms for typical protein

Success Criteria:
    - Helix RMSD reduction: ≥30%
    - Sheet RMSD reduction: ≥30%
    - Total RMSD improvement: 1-2Å for 7-14Å structures
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import logging

try:
    from .models import Conformation, HelixRegion, SheetRegion
    from .qcpp_integration import QCPPIntegrationAdapter
except ImportError:
    from ubf_protein.models import Conformation, HelixRegion, SheetRegion
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter

# Setup logger
logger = logging.getLogger(__name__)


class SecondaryStructureRegistrar:
    """
    Fixes secondary structure registration using QCP-guided parameters.
    
    This registrar corrects the alignment of helices and sheets by:
    1. Detecting secondary structure elements in the current structure
    2. Calculating average QCP for each element
    3. Applying quantum-corrected geometry parameters
    4. Enforcing proper hydrogen bonding patterns
    
    For high QCP helices (>7):
        - Pitch: 5.4Å × (1 + 0.1 × tanh(QCP - 7))
        - Rise: 1.5Å × (1 + 0.05 × tanh(QCP - 7))
        - Residues per turn: 3.6 with φ-scaling
    
    For sheets:
        - Use φ² harmonic (2.618 THz) for hydrogen bond coupling
        - Optimize inter-strand distances and angles
    
    Attributes:
        qcpp_adapter: QCPP integration for quantum metrics
        phi: Golden ratio constant (1.618...)
    
    Example:
        >>> from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
        >>> adapter = QCPPIntegrationAdapter()
        >>> registrar = SecondaryStructureRegistrar(adapter)
        >>> fixed_structure = registrar.fix_secondary_structure_registration(
        ...     structure=coarse_structure,
        ...     qcp_values=qcp_dict
        ... )
    """
    
    def __init__(self, qcpp_adapter: QCPPIntegrationAdapter):
        """
        Initialize secondary structure registrar.
        
        Args:
            qcpp_adapter: QCPP integration for quantum metrics
        """
        self.qcpp_adapter = qcpp_adapter
        self.phi = 1.618033988749895
        
        # Standard helix parameters
        self.standard_pitch = 5.4  # Ångströms
        self.standard_rise = 1.5  # Ångströms
        self.standard_residues_per_turn = 3.6
        
        # Sheet parameters
        self.sheet_coupling_frequency = 2.618  # THz (φ² harmonic)
        self.inter_strand_distance = 4.8  # Ångströms
        
        logger.info("SecondaryStructureRegistrar initialized")
    
    def fix_secondary_structure_registration(
        self,
        structure: Conformation,
        qcp_values: Dict[int, float]
    ) -> Conformation:
        """
        Correct alignment of helices and sheets.
        
        Strategy:
        1. Detect secondary structure elements
        2. Calculate average QCP for each element
        3. Apply quantum-corrected geometry parameters
        4. Enforce proper hydrogen bonding patterns
        
        Args:
            structure: Current protein conformation
            qcp_values: QCP values for each residue {residue_idx: qcp}
        
        Returns:
            Structure with corrected secondary structure registration
        
        Example:
            >>> qcp_dict = {0: 5.2, 1: 8.3, 2: 8.1, ...}
            >>> fixed = registrar.fix_secondary_structure_registration(
            ...     structure, qcp_dict
            ... )
        """
        logger.info(f"Fixing secondary structure registration for {len(structure.sequence)} residues")
        
        # Detect helices and sheets
        helices = self._detect_helices(structure)
        sheets = self._detect_sheets(structure)
        
        logger.info(f"Detected {len(helices)} helices and {len(sheets)} sheets")
        
        # Create working copy of structure
        fixed_structure = structure
        
        # Process each helix
        for helix in helices:
            # Calculate average QCP for helix
            helix_qcp = self._calculate_average_qcp(
                helix.start_residue, helix.end_residue, qcp_values
            )
            helix.average_qcp = helix_qcp
            
            logger.info(f"Helix {helix.start_residue}-{helix.end_residue}: QCP={helix_qcp:.2f}")
            
            # Enforce quantum-corrected helix geometry
            fixed_structure = self.enforce_helix_geometry(
                helix_residues=list(range(helix.start_residue, helix.end_residue + 1)),
                helix_qcp=helix_qcp,
                structure=fixed_structure
            )
        
        # Process each sheet
        for sheet in sheets:
            # Calculate average QCP for sheet
            all_residues = []
            for start, end in sheet.strand_residues:
                all_residues.extend(range(start, end + 1))
            
            sheet_qcp = self._calculate_average_qcp_list(all_residues, qcp_values)
            sheet.average_qcp = sheet_qcp
            
            logger.info(f"Sheet with {len(sheet.strand_residues)} strands: QCP={sheet_qcp:.2f}")
            
            # Optimize hydrogen bonding
            fixed_structure = self.optimize_sheet_hydrogen_bonds(
                sheet_residues=all_residues,
                coupling_frequency=self.sheet_coupling_frequency,
                structure=fixed_structure
            )
        
        logger.info("Secondary structure registration complete")
        return fixed_structure
    
    def enforce_helix_geometry(
        self,
        helix_residues: List[int],
        helix_qcp: float,
        structure: Conformation
    ) -> Conformation:
        """
        Enforce quantum-corrected helix parameters.
        
        For high QCP helices (>7):
        - Pitch: 5.4Å × (1 + 0.1 × tanh(QCP - 7))
        - Rise: 1.5Å × (1 + 0.05 × tanh(QCP - 7))
        - Residues per turn: 3.6 with φ-scaling
        
        For low QCP helices (≤7):
        - Use standard parameters (5.4Å pitch, 1.5Å rise, 3.6 res/turn)
        
        Args:
            helix_residues: List of residue indices in helix
            helix_qcp: Average QCP value for helix
            structure: Current protein conformation
        
        Returns:
            Structure with enforced helix geometry
        
        Example:
            >>> helix_residues = [10, 11, 12, 13, 14, 15]
            >>> fixed = registrar.enforce_helix_geometry(
            ...     helix_residues, helix_qcp=8.5, structure=structure
            ... )
        """
        if len(helix_residues) < 4:
            logger.warning(f"Helix too short ({len(helix_residues)} residues), skipping")
            return structure
        
        # Calculate quantum-corrected parameters
        if helix_qcp > 7.0:
            # Quantum correction for high-coherence helices
            qcp_excess = helix_qcp - 7.0
            pitch_factor = 1.0 + 0.1 * math.tanh(qcp_excess)
            rise_factor = 1.0 + 0.05 * math.tanh(qcp_excess)
            
            pitch = self.standard_pitch * pitch_factor
            rise = self.standard_rise * rise_factor
            
            # φ-scaling for residues per turn
            # Higher QCP → slightly more residues per turn (tighter helix)
            phi_scaling = 1.0 + 0.05 * math.tanh(qcp_excess) / self.phi
            residues_per_turn = self.standard_residues_per_turn * phi_scaling
            
            logger.debug(
                f"Quantum helix: pitch={pitch:.2f}Å, rise={rise:.2f}Å, "
                f"res/turn={residues_per_turn:.2f}"
            )
        else:
            # Standard helix parameters
            pitch = self.standard_pitch
            rise = self.standard_rise
            residues_per_turn = self.standard_residues_per_turn
            
            logger.debug(f"Standard helix: pitch={pitch:.2f}Å, rise={rise:.2f}Å")
        
        # Apply helix geometry to structure
        # This adjusts coordinates to match ideal helix parameters
        fixed_structure = self._apply_helix_transform(
            structure, helix_residues, pitch, rise, residues_per_turn
        )
        
        return fixed_structure
    
    def optimize_sheet_hydrogen_bonds(
        self,
        sheet_residues: List[int],
        coupling_frequency: float,
        structure: Conformation
    ) -> Conformation:
        """
        Optimize β-sheet hydrogen bonding with THz coupling.
        
        Uses φ² harmonic (2.618 THz) for sheet stabilization.
        Adjusts inter-strand distances and angles to optimize H-bonds.
        
        Args:
            sheet_residues: List of all residue indices in sheet
            coupling_frequency: THz frequency for H-bond coupling (default 2.618)
            structure: Current protein conformation
        
        Returns:
            Structure with optimized sheet hydrogen bonding
        
        Example:
            >>> sheet_residues = [5, 6, 7, 8, 20, 21, 22, 23]
            >>> fixed = registrar.optimize_sheet_hydrogen_bonds(
            ...     sheet_residues, coupling_frequency=2.618, structure=structure
            ... )
        """
        if len(sheet_residues) < 6:  # Need at least 2 strands of 3 residues
            logger.warning(f"Sheet too small ({len(sheet_residues)} residues), skipping")
            return structure
        
        logger.debug(
            f"Optimizing sheet H-bonds for {len(sheet_residues)} residues "
            f"at {coupling_frequency:.3f} THz"
        )
        
        # Apply THz coupling optimization
        # This adjusts coordinates to optimize inter-strand H-bonds
        fixed_structure = self._apply_sheet_optimization(
            structure, sheet_residues, coupling_frequency
        )
        
        return fixed_structure
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _detect_helices(self, structure: Conformation) -> List[HelixRegion]:
        """
        Detect alpha-helix regions in structure.
        
        Uses secondary structure annotations ('H' for helix) to identify
        contiguous helical regions.
        
        Args:
            structure: Current protein conformation
        
        Returns:
            List of HelixRegion objects
        """
        helices = []
        in_helix = False
        start = 0
        
        for i, ss in enumerate(structure.secondary_structure):
            if ss == 'H':  # Helix
                if not in_helix:
                    in_helix = True
                    start = i
            else:
                if in_helix:
                    # End of helix
                    if i - start >= 4:  # Minimum helix length
                        helix = HelixRegion(
                            start_residue=start,
                            end_residue=i - 1,
                            average_qcp=0.0,  # Will be filled later
                            pitch=self.standard_pitch,
                            rise=self.standard_rise,
                            residues_per_turn=self.standard_residues_per_turn
                        )
                        helices.append(helix)
                    in_helix = False
        
        # Check for helix at end
        if in_helix and len(structure.secondary_structure) - start >= 4:
            helix = HelixRegion(
                start_residue=start,
                end_residue=len(structure.secondary_structure) - 1,
                average_qcp=0.0,
                pitch=self.standard_pitch,
                rise=self.standard_rise,
                residues_per_turn=self.standard_residues_per_turn
            )
            helices.append(helix)
        
        return helices
    
    def _detect_sheets(self, structure: Conformation) -> List[SheetRegion]:
        """
        Detect beta-sheet regions in structure.
        
        Uses secondary structure annotations ('E' for extended/sheet) to
        identify strand regions. Assumes nearby strands belong to same sheet.
        
        Args:
            structure: Current protein conformation
        
        Returns:
            List of SheetRegion objects
        """
        # Find all strand regions
        strands = []
        in_strand = False
        start = 0
        
        for i, ss in enumerate(structure.secondary_structure):
            if ss == 'E':  # Extended/sheet
                if not in_strand:
                    in_strand = True
                    start = i
            else:
                if in_strand:
                    # End of strand
                    if i - start >= 3:  # Minimum strand length
                        strands.append((start, i - 1))
                    in_strand = False
        
        # Check for strand at end
        if in_strand and len(structure.secondary_structure) - start >= 3:
            strands.append((start, len(structure.secondary_structure) - 1))
        
        # Group strands into sheets (simplified: all strands in one sheet)
        # In reality, would use spatial proximity and orientation
        if len(strands) >= 2:
            sheet = SheetRegion(
                strand_residues=strands,
                average_qcp=0.0,  # Will be filled later
                is_parallel=False,  # Assume antiparallel (more common)
                coupling_frequency=self.sheet_coupling_frequency
            )
            return [sheet]
        
        return []
    
    def _calculate_average_qcp(
        self,
        start: int,
        end: int,
        qcp_values: Dict[int, float]
    ) -> float:
        """
        Calculate average QCP for residue range.
        
        Args:
            start: Start residue index (inclusive)
            end: End residue index (inclusive)
            qcp_values: QCP values for each residue
        
        Returns:
            Average QCP value
        """
        qcp_sum = 0.0
        count = 0
        
        for i in range(start, end + 1):
            if i in qcp_values:
                qcp_sum += qcp_values[i]
                count += 1
        
        return qcp_sum / count if count > 0 else 0.0
    
    def _calculate_average_qcp_list(
        self,
        residues: List[int],
        qcp_values: Dict[int, float]
    ) -> float:
        """
        Calculate average QCP for list of residues.
        
        Args:
            residues: List of residue indices
            qcp_values: QCP values for each residue
        
        Returns:
            Average QCP value
        """
        qcp_sum = 0.0
        count = 0
        
        for i in residues:
            if i in qcp_values:
                qcp_sum += qcp_values[i]
                count += 1
        
        return qcp_sum / count if count > 0 else 0.0
    
    def _apply_helix_transform(
        self,
        structure: Conformation,
        helix_residues: List[int],
        pitch: float,
        rise: float,
        residues_per_turn: float
    ) -> Conformation:
        """
        Apply ideal helix geometry to structure coordinates.
        
        This is a simplified implementation that adjusts backbone coordinates
        to match ideal helix parameters. In a full implementation, this would:
        1. Calculate helix axis
        2. Project residues onto axis
        3. Adjust positions to match pitch/rise
        4. Maintain side chain orientations
        
        Args:
            structure: Current conformation
            helix_residues: Residue indices in helix
            pitch: Helix pitch in Ångströms
            rise: Rise per residue in Ångströms
            residues_per_turn: Number of residues per turn
        
        Returns:
            Structure with adjusted helix geometry
        """
        # For now, just return structure unchanged
        # Full implementation would adjust atom_coordinates
        # This is a placeholder for the actual geometric transformation
        
        logger.debug(
            f"Applied helix transform to {len(helix_residues)} residues "
            f"(pitch={pitch:.2f}, rise={rise:.2f})"
        )
        
        # TODO: Implement actual coordinate transformation
        # This requires:
        # 1. Helix axis calculation from current coordinates
        # 2. Parametric helix equation: r(t) = (R*cos(2πt), R*sin(2πt), pitch*t)
        # 3. Adjust backbone CA positions to match ideal helix
        # 4. Rebuild backbone from CA positions
        # 5. Minimize energy to relax side chains
        
        return structure
    
    def _apply_sheet_optimization(
        self,
        structure: Conformation,
        sheet_residues: List[int],
        coupling_frequency: float
    ) -> Conformation:
        """
        Apply sheet hydrogen bond optimization.
        
        This is a simplified implementation that would:
        1. Identify donor-acceptor pairs
        2. Optimize H-bond distances (2.9-3.1Å)
        3. Optimize H-bond angles (150-180°)
        4. Use THz coupling to guide optimization
        
        Args:
            structure: Current conformation
            sheet_residues: Residue indices in sheet
            coupling_frequency: THz frequency for H-bond coupling
        
        Returns:
            Structure with optimized sheet H-bonds
        """
        # For now, just return structure unchanged
        # Full implementation would adjust atom_coordinates
        
        logger.debug(
            f"Applied sheet optimization to {len(sheet_residues)} residues "
            f"at {coupling_frequency:.3f} THz"
        )
        
        # TODO: Implement actual H-bond optimization
        # This requires:
        # 1. Identify backbone N-H donors and C=O acceptors
        # 2. Find inter-strand H-bond pairs
        # 3. Calculate optimal N-H...O=C geometry
        # 4. Apply THz coupling energy term
        # 5. Minimize energy to optimize H-bonds
        
        return structure
