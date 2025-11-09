"""
Quantum Core Analyzer for UBF Protein System

This module identifies high-coherence regions (quantum cores) in protein structures
and establishes THz resonance networks between coupled residues.

Quantum cores are regions with QCP (Quantum Consciousness Potential) > 7.0,
indicating high structural coherence and stability. These regions exhibit
characteristic THz vibrational modes that can couple with each other through
φ-harmonic resonances (golden ratio patterns).

Key Concepts:
    - Quantum Core: Region with QCP > 7.0 (high coherence)
    - THz Mode: Vibrational mode in terahertz frequency range (10^12 Hz)
    - φ-Harmonic Resonance: Coupling at frequencies near φ^n × 1.0 THz
      (1.618 THz, 2.618 THz, 4.236 THz, etc.)
    - Resonance Coupling: Residue pairs with matching THz modes

Physics Background:
    - Protein vibrations span femtosecond to microsecond timescales
    - THz modes (ps timescale) are critical for conformational dynamics
    - Golden ratio patterns emerge in stable protein geometries
    - Resonance coupling enables long-range energy transfer

Performance Targets:
    - Quantum core identification: <100ms
    - THz mode calculation: <50ms per core
    - Resonance coupling detection: <20ms per mode
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import logging

try:
    from .models import Conformation, QuantumCore, THzMode
    from .qcpp_integration import QCPPIntegrationAdapter
except ImportError:
    from ubf_protein.models import Conformation, QuantumCore, THzMode
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter

# Setup logger
logger = logging.getLogger(__name__)


class QuantumCoreAnalyzer:
    """
    Analyzes quantum coherence patterns and identifies resonance networks.
    
    This analyzer uses QCP (Quantum Consciousness Potential) values to identify
    high-coherence regions (quantum cores) in protein structures. It then calculates
    THz vibrational modes for these cores and identifies residue pairs coupled by
    φ-harmonic resonances.
    
    The analysis proceeds in three stages:
    1. Identify quantum cores (QCP > threshold)
    2. Calculate THz modes for each core
    3. Find coupled residues via φ-harmonic resonance
    
    Attributes:
        qcpp_adapter: QCPP integration for quantum metrics
        phi: Golden ratio constant (1.618033988749895)
        phi_harmonics: List of φ^n values for resonance detection
    
    Example:
        >>> analyzer = QuantumCoreAnalyzer(qcpp_adapter)
        >>> cores = analyzer.identify_quantum_cores(structure, qcp_threshold=7.0)
        >>> for core in cores:
        ...     modes = analyzer.calculate_local_thz_modes(core, structure)
        ...     for mode in modes:
        ...         couples = analyzer.find_coupled_residues(mode, structure)
    """
    
    # Golden ratio and harmonics (class-level constants)
    PHI = 1.618033988749895
    PHI_HARMONICS = [
        1.0 * PHI**0,  # 1.000 THz (base)
        1.0 * PHI**1,  # 1.618 THz (φ)
        1.0 * PHI**2,  # 2.618 THz (φ²)
        1.0 * PHI**3,  # 4.236 THz (φ³)
        1.0 * PHI**4,  # 6.854 THz (φ⁴)
    ]
    
    def __init__(self, qcpp_adapter: QCPPIntegrationAdapter):
        """
        Initialize quantum core analyzer.
        
        Args:
            qcpp_adapter: QCPP integration for quantum metrics (QCP, coherence, stability)
        
        Raises:
            TypeError: If qcpp_adapter is None or wrong type
        """
        if qcpp_adapter is None:
            raise TypeError("qcpp_adapter cannot be None")
        if not isinstance(qcpp_adapter, QCPPIntegrationAdapter):
            raise TypeError(f"qcpp_adapter must be QCPPIntegrationAdapter, got {type(qcpp_adapter)}")
        
        self.qcpp_adapter = qcpp_adapter
        self.phi = self.PHI
        self.phi_harmonics = self.PHI_HARMONICS
        
        logger.info(
            f"QuantumCoreAnalyzer initialized with φ={self.phi:.15f}, "
            f"harmonics={[f'{h:.3f}' for h in self.phi_harmonics]} THz"
        )
    
    def identify_quantum_cores(
        self,
        structure: Conformation,
        qcp_threshold: float = 7.0
    ) -> List[QuantumCore]:
        """
        Identify regions with QCP > threshold as quantum cores.
        
        Quantum cores are contiguous regions where all residues have QCP values
        above the threshold. The method:
        1. Calculate QCP for all residues using QCPP adapter
        2. Identify contiguous high-QCP regions (allow 1-residue gaps)
        3. Calculate average QCP and coherence for each core
        4. Calculate geometric center of mass for each core
        
        Algorithm:
            - Scan residues left-to-right
            - Start new core when QCP > threshold
            - Extend core while QCP > threshold (allow 1-residue gap)
            - End core when 2+ consecutive residues below threshold
            - Minimum core size: 3 residues
        
        Args:
            structure: Protein conformation to analyze
            qcp_threshold: Minimum QCP value for quantum core (default: 7.0)
        
        Returns:
            List of QuantumCore objects with residue indices and metrics
        
        Raises:
            ValueError: If qcp_threshold < 0 or structure has < 3 residues
        
        Example:
            >>> cores = analyzer.identify_quantum_cores(structure, qcp_threshold=7.0)
            >>> print(f"Found {len(cores)} quantum cores")
            >>> for core in cores:
            ...     print(f"Core at residues {core.residue_indices[0]}-{core.residue_indices[-1]}")
            ...     print(f"  Average QCP: {core.average_qcp:.2f}")
            ...     print(f"  Coherence: {core.coherence:.3f}")
        """
        # Validate inputs
        if qcp_threshold < 0:
            raise ValueError(f"qcp_threshold must be >= 0, got {qcp_threshold}")
        
        sequence_length = len(structure.sequence)
        if sequence_length < 3:
            raise ValueError(f"Structure must have >= 3 residues, got {sequence_length}")
        
        # Get overall QCP metrics for the structure
        overall_metrics = self.qcpp_adapter.analyze_conformation(structure)
        
        # Calculate per-residue QCP values
        # For now, use a simplified approach: overall QCP score as baseline,
        # modulated by local geometry
        qcp_values = self._calculate_per_residue_qcp(structure, overall_metrics)
        
        # Coherence values: use a simple decay from overall coherence based on distance from center
        coherence_values = self._calculate_per_residue_coherence(structure, overall_metrics)
        
        if len(qcp_values) != sequence_length:
            logger.warning(
                f"QCP values length ({len(qcp_values)}) != sequence length ({sequence_length}), "
                "using available values"
            )
        
        # Identify cores by scanning for contiguous high-QCP regions
        cores: List[QuantumCore] = []
        current_core_residues: List[int] = []
        gap_count = 0  # Track consecutive low-QCP residues
        
        for i in range(len(qcp_values)):
            qcp = qcp_values[i]
            
            if qcp >= qcp_threshold:
                # High QCP: add to current core
                current_core_residues.append(i)
                gap_count = 0  # Reset gap counter
            else:
                # Low QCP: check if gap is tolerable (max 1 residue)
                gap_count += 1
                
                if gap_count == 1 and len(current_core_residues) > 0:
                    # First gap: add anyway (allows 1-residue interruptions)
                    current_core_residues.append(i)
                elif gap_count >= 2:
                    # Second consecutive gap: end current core
                    if len(current_core_residues) >= 3:  # Minimum core size
                        core = self._create_quantum_core(
                            current_core_residues,
                            qcp_values,
                            coherence_values,
                            structure
                        )
                        cores.append(core)
                    
                    # Reset for next core
                    current_core_residues = []
                    gap_count = 0
        
        # Handle last core if sequence ended in high-QCP region
        if len(current_core_residues) >= 3:
            core = self._create_quantum_core(
                current_core_residues,
                qcp_values,
                coherence_values,
                structure
            )
            cores.append(core)
        
        logger.info(
            f"Identified {len(cores)} quantum cores with QCP > {qcp_threshold:.1f} "
            f"(total {sum(len(c.residue_indices) for c in cores)} residues)"
        )
        
        return cores
    
    def _create_quantum_core(
        self,
        residue_indices: List[int],
        qcp_values: List[float],
        coherence_values: List[float],
        structure: Conformation
    ) -> QuantumCore:
        """
        Create a QuantumCore object from residue data.
        
        Helper method to calculate metrics and create QuantumCore dataclass.
        
        Args:
            residue_indices: List of residue indices in core
            qcp_values: QCP values for all residues
            coherence_values: Coherence values for all residues
            structure: Protein conformation
        
        Returns:
            QuantumCore with calculated metrics
        """
        # Calculate average QCP (exclude gap residues below threshold)
        core_qcp_values = [qcp_values[i] for i in residue_indices]
        average_qcp = sum(core_qcp_values) / len(core_qcp_values)
        
        # Calculate average coherence
        core_coherence_values = [coherence_values[i] for i in residue_indices]
        average_coherence = sum(core_coherence_values) / len(core_coherence_values)
        
        # Calculate geometric center of mass (using CA coordinates)
        coords = structure.atom_coordinates
        center_x = sum(coords[i][0] for i in residue_indices) / len(residue_indices)
        center_y = sum(coords[i][1] for i in residue_indices) / len(residue_indices)
        center_z = sum(coords[i][2] for i in residue_indices) / len(residue_indices)
        
        return QuantumCore(
            residue_indices=residue_indices,
            average_qcp=average_qcp,
            coherence=average_coherence,
            center_of_mass=(center_x, center_y, center_z)
        )
    
    def _calculate_per_residue_qcp(
        self,
        structure: Conformation,
        overall_metrics: Any  # QCPPMetrics
    ) -> List[float]:
        """
        Calculate QCP values for each residue.
        
        Uses a simplified approach: modulate the overall QCP score based on
        local geometry (phi/psi angles, secondary structure).
        
        Args:
            structure: Protein conformation
            overall_metrics: Overall QCPP metrics for the structure
        
        Returns:
            List of QCP values, one per residue
        """
        base_qcp = overall_metrics.qcp_score
        sequence_length = len(structure.sequence)
        qcp_values: List[float] = []
        
        for i in range(sequence_length):
            # Modulate base QCP by secondary structure
            ss = structure.secondary_structure[i] if i < len(structure.secondary_structure) else 'C'
            
            # Helix/sheet have higher QCP than coils
            if ss == 'H':
                ss_factor = 1.2  # Helices are more ordered
            elif ss == 'E':
                ss_factor = 1.15  # Sheets are ordered
            else:
                ss_factor = 0.9  # Coils are less ordered
            
            # Modulate by phi angle if available (near -60° for helix, -120° for sheet)
            phi_factor = 1.0
            if i < len(structure.phi_angles):
                phi = structure.phi_angles[i]
                # Check if near ideal helix angle (-60°)
                if abs(phi - (-60.0)) < 30.0:
                    phi_factor = 1.1
                # Check if near ideal sheet angle (-120°)
                elif abs(phi - (-120.0)) < 30.0:
                    phi_factor = 1.05
            
            # Calculate residue QCP
            residue_qcp = base_qcp * ss_factor * phi_factor
            qcp_values.append(residue_qcp)
        
        return qcp_values
    
    def _calculate_per_residue_coherence(
        self,
        structure: Conformation,
        overall_metrics: Any  # QCPPMetrics
    ) -> List[float]:
        """
        Calculate coherence values for each residue.
        
        Uses a simplified approach: map field coherence (-1 to 1) to
        per-residue coherence (0 to 1), modulated by distance from structure center.
        
        Args:
            structure: Protein conformation
            overall_metrics: Overall QCPP metrics for the structure
        
        Returns:
            List of coherence values, one per residue (0-1 range)
        """
        # Map field coherence (-1 to 1) to (0 to 1) range
        base_coherence = (overall_metrics.field_coherence + 1.0) / 2.0
        base_coherence = max(0.0, min(1.0, base_coherence))  # Clamp to [0, 1]
        
        # Calculate structure center
        coords = structure.atom_coordinates
        center_x = sum(c[0] for c in coords) / len(coords)
        center_y = sum(c[1] for c in coords) / len(coords)
        center_z = sum(c[2] for c in coords) / len(coords)
        
        # Calculate max distance from center
        max_dist = 0.0
        for x, y, z in coords:
            dx = x - center_x
            dy = y - center_y
            dz = z - center_z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > max_dist:
                max_dist = dist
        
        if max_dist == 0:
            max_dist = 1.0  # Avoid division by zero
        
        # Calculate per-residue coherence
        coherence_values: List[float] = []
        for i in range(len(coords)):
            x, y, z = coords[i]
            dx = x - center_x
            dy = y - center_y
            dz = z - center_z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Coherence decreases slightly with distance from center
            # (residues at periphery are less constrained)
            distance_factor = 1.0 - 0.2 * (dist / max_dist)
            
            coherence = base_coherence * distance_factor
            coherence_values.append(coherence)
        
        return coherence_values
    
    def calculate_local_thz_modes(
        self,
        core: QuantumCore,
        structure: Conformation,
        num_modes: int = 5
    ) -> List[THzMode]:
        """
        Calculate THz vibrational modes for a quantum core region.
        
        Uses simplified normal mode analysis on local structure to find
        characteristic vibrational frequencies in the THz range (0.1-10 THz).
        
        The method uses a harmonic approximation based on:
        1. Inter-residue distances in the core
        2. QCP-weighted spring constants
        3. Mass-weighted frequency calculation
        
        Simplified Formula:
            ω = sqrt(k/μ) / (2π)
        where:
            k = force constant from inter-residue distance
            μ = reduced mass (approximated by CA-CA distance)
            ω = frequency in THz
        
        Args:
            core: QuantumCore to analyze
            structure: Protein conformation
            num_modes: Number of modes to calculate (default: 5)
        
        Returns:
            List of THzMode objects sorted by frequency
        
        Raises:
            ValueError: If num_modes < 1
        
        Example:
            >>> modes = analyzer.calculate_local_thz_modes(core, structure, num_modes=5)
            >>> for mode in modes:
            ...     if mode.is_phi_harmonic:
            ...         print(f"φ-harmonic mode at {mode.frequency:.3f} THz")
        """
        if num_modes < 1:
            raise ValueError(f"num_modes must be >= 1, got {num_modes}")
        
        residues = core.residue_indices
        coords = structure.atom_coordinates
        
        # Calculate inter-residue distances for spring constant estimation
        # Use harmonic approximation: k ∝ 1/r²
        frequencies: List[float] = []
        amplitudes: List[float] = []
        
        for i in range(len(residues) - 1):
            res_i = residues[i]
            res_j = residues[i + 1]
            
            # Calculate distance
            dx = coords[res_i][0] - coords[res_j][0]
            dy = coords[res_i][1] - coords[res_j][1]
            dz = coords[res_i][2] - coords[res_j][2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Harmonic approximation: k ∝ 1/r², ω = sqrt(k/m)
            # Use simplified formula calibrated to protein THz range
            # Base frequency: ~1 THz for CA-CA distance ~3.8Å
            base_freq = 1.0  # THz
            ref_distance = 3.8  # Ångströms (typical CA-CA distance)
            
            # Frequency scales with sqrt(1/distance)
            frequency = base_freq * math.sqrt(ref_distance / max(distance, 1.0))
            
            # QCP modulates frequency (higher QCP → higher frequency)
            # Get QCP for this residue pair
            overall_metrics = self.qcpp_adapter.analyze_conformation(structure)
            qcp_values = self._calculate_per_residue_qcp(structure, overall_metrics)
            
            qcp_i = qcp_values[res_i]
            qcp_j = qcp_values[res_j]
            avg_qcp = (qcp_i + qcp_j) / 2.0
            
            # QCP scaling: multiply by (1 + 0.1 * (QCP - 7))
            # QCP=7 → 1.0x, QCP=8 → 1.1x, QCP=10 → 1.3x
            qcp_scaling = 1.0 + 0.1 * (avg_qcp - 7.0)
            frequency *= qcp_scaling
            
            # Amplitude based on coherence (higher coherence → larger amplitude)
            amplitude = core.coherence * (1.0 + (avg_qcp - 7.0) / 10.0)
            
            frequencies.append(frequency)
            amplitudes.append(amplitude)
        
        # Sort by amplitude (most significant modes first)
        sorted_pairs = sorted(
            zip(frequencies, amplitudes),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create THzMode objects for top N modes
        modes: List[THzMode] = []
        for i in range(min(num_modes, len(sorted_pairs))):
            freq, amp = sorted_pairs[i]
            
            # Check if frequency is near a φ-harmonic
            is_phi = self._is_phi_harmonic(freq, tolerance=0.1)
            
            # All residues in core participate
            mode = THzMode(
                frequency=freq,
                amplitude=amp,
                participating_residues=residues.copy(),
                is_phi_harmonic=is_phi
            )
            modes.append(mode)
        
        logger.debug(
            f"Calculated {len(modes)} THz modes for core with {len(residues)} residues: "
            f"frequencies={[f'{m.frequency:.3f}' for m in modes]} THz"
        )
        
        return modes
    
    def _is_phi_harmonic(self, frequency: float, tolerance: float = 0.1) -> bool:
        """
        Check if frequency is near a φ-harmonic (within tolerance).
        
        Args:
            frequency: Frequency in THz
            tolerance: Tolerance in THz (default: 0.1)
        
        Returns:
            True if frequency is within tolerance of any φ-harmonic
        """
        for harmonic in self.phi_harmonics:
            if abs(frequency - harmonic) <= tolerance:
                return True
        return False
    
    def find_coupled_residues(
        self,
        mode: THzMode,
        structure: Conformation,
        phi_tolerance: float = 0.1
    ) -> List[Tuple[int, int]]:
        """
        Find residue pairs coupled by φ-harmonic resonance.
        
        Identifies pairs where THz mode frequency is within phi_tolerance
        of a φ-harmonic (1.618 THz, 2.618 THz, etc.). Coupled residues can
        transfer energy efficiently through resonance.
        
        Algorithm:
        1. Check if mode is φ-harmonic (frequency near φ^n × 1.0 THz)
        2. If yes, find all pairwise combinations in participating residues
        3. Filter pairs by sequence separation (≥5) and spatial distance (<15Å)
        4. Return valid coupled pairs
        
        Args:
            mode: THzMode to analyze
            structure: Protein conformation
            phi_tolerance: Tolerance for φ-harmonic matching in THz (default: 0.1)
        
        Returns:
            List of (residue_i, residue_j) tuples for coupled pairs
        
        Raises:
            ValueError: If phi_tolerance <= 0
        
        Example:
            >>> coupled = analyzer.find_coupled_residues(mode, structure, phi_tolerance=0.1)
            >>> print(f"Found {len(coupled)} coupled residue pairs")
            >>> for i, j in coupled:
            ...     print(f"Residues {i} and {j} are coupled via φ-resonance")
        """
        if phi_tolerance <= 0:
            raise ValueError(f"phi_tolerance must be > 0, got {phi_tolerance}")
        
        # Check if mode is φ-harmonic
        if not self._is_phi_harmonic(mode.frequency, tolerance=phi_tolerance):
            logger.debug(
                f"Mode at {mode.frequency:.3f} THz is not φ-harmonic, no coupling"
            )
            return []
        
        # Find which φ-harmonic this mode is near
        closest_harmonic = min(
            self.phi_harmonics,
            key=lambda h: abs(mode.frequency - h)
        )
        
        # Find coupled pairs among participating residues
        residues = mode.participating_residues
        coords = structure.atom_coordinates
        coupled_pairs: List[Tuple[int, int]] = []
        
        for i in range(len(residues)):
            for j in range(i + 1, len(residues)):
                res_i = residues[i]
                res_j = residues[j]
                
                # Check sequence separation (must be ≥5 for long-range coupling)
                seq_sep = abs(res_j - res_i)
                if seq_sep < 5:
                    continue
                
                # Check spatial distance (must be <15Å for effective coupling)
                dx = coords[res_i][0] - coords[res_j][0]
                dy = coords[res_i][1] - coords[res_j][1]
                dz = coords[res_i][2] - coords[res_j][2]
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                if distance > 15.0:
                    continue
                
                # Valid coupling found
                coupled_pairs.append((res_i, res_j))
        
        logger.debug(
            f"Found {len(coupled_pairs)} coupled pairs at φ-harmonic {closest_harmonic:.3f} THz "
            f"(mode frequency: {mode.frequency:.3f} THz)"
        )
        
        return coupled_pairs
