"""
Vibrational Analysis Module - Real Normal Mode Calculation

This module calculates actual vibrational normal modes for protein conformations,
converting eigenvalues to THz frequencies for determinism testing.

Pure Python implementation compatible with PyPy optimization.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass(frozen=True)
class VibrationalMode:
    """Immutable representation of a vibrational normal mode."""
    mode_number: int
    frequency_thz: float
    intensity: float
    eigenvector: Tuple[Tuple[float, ...], ...]  # Displacement vectors per atom (3D each)
    
    def __repr__(self) -> str:
        return f"Mode {self.mode_number}: {self.frequency_thz:.3f} THz (I={self.intensity:.3f})"


@dataclass(frozen=True)
class THzSpectrum:
    """Immutable THz spectrum with all vibrational modes."""
    modes: Tuple[VibrationalMode, ...]
    total_energy: float
    rmsd: float
    qcp_score: Optional[float] = None
    
    @property
    def frequencies(self) -> List[float]:
        """Extract frequency list."""
        return [mode.frequency_thz for mode in self.modes]
    
    @property
    def intensities(self) -> List[float]:
        """Extract intensity list."""
        return [mode.intensity for mode in self.modes]
    
    def get_peak_frequencies(self, threshold: float = 0.1) -> List[float]:
        """Get frequencies of peaks above intensity threshold."""
        return [mode.frequency_thz for mode in self.modes 
                if mode.intensity > threshold]


class VibrationalAnalyzer:
    """
    Calculate vibrational normal modes and THz spectra for protein conformations.
    
    Uses simplified elastic network model (ENM) for computational efficiency:
    - Harmonic springs between nearby residues (CA atoms)
    - Eigenvalue decomposition of mass-weighted Hessian
    - Conversion to THz frequencies via: ω = √(eigenvalue) / (2π × c)
    
    Pure Python implementation for PyPy compatibility.
    """
    
    # Physical constants
    C_LIGHT = 2.998e10  # Speed of light (cm/s)
    AMU_TO_KG = 1.66054e-27  # Atomic mass unit to kg
    KCAL_TO_JOULE = 4184.0  # kcal/mol to J/mol
    AVOGADRO = 6.022e23  # Avogadro's number
    
    # ENM parameters
    CUTOFF_DISTANCE = 10.0  # Angstroms - cutoff for spring connections
    SPRING_CONSTANT = 1.0  # kcal/(mol·Å²) - typical ENM value
    CA_MASS = 12.0  # Effective CA mass (atomic mass units)
    
    def __init__(self, cutoff: float = 10.0, spring_constant: float = 1.0):
        """
        Initialize vibrational analyzer.
        
        Args:
            cutoff: Distance cutoff for ENM springs (Angstroms)
            spring_constant: ENM spring constant (kcal/(mol·Å²))
        """
        self.cutoff = cutoff
        self.spring_constant = spring_constant
        
    def calculate_spectrum(
        self, 
        ca_coordinates: List[Tuple[float, float, float]],
        n_modes: int = 20,
        energy: float = 0.0,
        rmsd: float = 0.0,
        qcp_score: Optional[float] = None
    ) -> THzSpectrum:
        """
        Calculate full THz spectrum for a conformation.
        
        Args:
            ca_coordinates: List of (x, y, z) CA atom positions
            n_modes: Number of vibrational modes to calculate
            energy: Total energy of conformation (kcal/mol)
            rmsd: RMSD value if known
            qcp_score: QCP score if available
            
        Returns:
            THzSpectrum with all calculated modes
        """
        n_atoms = len(ca_coordinates)
        
        # Build Hessian matrix (3N × 3N)
        hessian = self._build_hessian(ca_coordinates)
        
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = self._diagonalize_hessian(hessian, n_atoms)
        
        # Convert to THz frequencies and calculate intensities
        modes = []
        for i in range(min(n_modes, len(eigenvalues))):
            freq_thz = self._eigenvalue_to_thz(eigenvalues[i])
            intensity = self._calculate_intensity(eigenvalues[i], eigenvectors[i], n_atoms)
            
            # Reshape eigenvector into per-atom displacements
            displacements = tuple(
                tuple(eigenvectors[i][3*j:3*j+3]) 
                for j in range(n_atoms)
            )
            
            mode = VibrationalMode(
                mode_number=i,
                frequency_thz=freq_thz,
                intensity=intensity,
                eigenvector=displacements
            )
            modes.append(mode)
        
        return THzSpectrum(
            modes=tuple(modes),
            total_energy=energy,
            rmsd=rmsd,
            qcp_score=qcp_score
        )
    
    def _build_hessian(self, ca_coords: List[Tuple[float, float, float]]) -> List[List[float]]:
        """
        Build Hessian matrix using elastic network model.
        
        The Hessian H is a 3N × 3N matrix where:
        - Diagonal blocks: sum of all spring connections to atom i
        - Off-diagonal blocks: spring connections between atoms i and j
        
        Args:
            ca_coords: CA atom coordinates
            
        Returns:
            Hessian matrix as list of lists (3N × 3N)
        """
        n_atoms = len(ca_coords)
        size = 3 * n_atoms
        
        # Initialize Hessian as list of lists (pure Python)
        hessian = [[0.0 for _ in range(size)] for _ in range(size)]
        
        # Build contact map and spring connections
        for i in range(n_atoms):
            xi, yi, zi = ca_coords[i]
            
            for j in range(i + 1, n_atoms):
                xj, yj, zj = ca_coords[j]
                
                # Calculate distance
                dx = xj - xi
                dy = yj - yi
                dz = zj - zi
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Only connect residues within cutoff
                if dist < self.cutoff and dist > 0.1:  # Avoid div by zero
                    # Spring constant for this pair
                    k = self.spring_constant / (dist * dist)
                    
                    # Calculate second derivatives (3×3 blocks)
                    # Off-diagonal block (i,j)
                    for alpha in range(3):
                        for beta in range(3):
                            d_vec = [dx, dy, dz]
                            value = -k * d_vec[alpha] * d_vec[beta] / (dist * dist)
                            
                            # Off-diagonal blocks
                            hessian[3*i + alpha][3*j + beta] = value
                            hessian[3*j + beta][3*i + alpha] = value
                            
                            # Diagonal blocks (negative of off-diagonal sum)
                            hessian[3*i + alpha][3*i + beta] -= value
                            hessian[3*j + alpha][3*j + beta] -= value
        
        return hessian
    
    def _diagonalize_hessian(
        self, 
        hessian: List[List[float]], 
        n_atoms: int
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Diagonalize Hessian to get eigenvalues and eigenvectors.
        
        Uses Jacobi iteration for symmetric eigenvalue problem.
        Pure Python implementation for PyPy compatibility.
        
        Args:
            hessian: Hessian matrix (3N × 3N)
            n_atoms: Number of atoms
            
        Returns:
            (eigenvalues, eigenvectors) sorted by eigenvalue magnitude
        """
        size = 3 * n_atoms
        
        # Copy hessian for in-place modification
        A = [row[:] for row in hessian]
        
        # Initialize eigenvectors as identity
        V = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        
        # Jacobi iteration (simplified for small systems)
        max_iterations = 50 * size
        tolerance = 1e-9
        
        for iteration in range(max_iterations):
            # Find largest off-diagonal element
            max_val = 0.0
            p, q = 0, 1
            
            for i in range(size):
                for j in range(i + 1, size):
                    if abs(A[i][j]) > max_val:
                        max_val = abs(A[i][j])
                        p, q = i, j
            
            # Convergence check
            if max_val < tolerance:
                break
            
            # Calculate rotation angle
            if abs(A[p][p] - A[q][q]) < 1e-12:
                theta = math.pi / 4.0
            else:
                theta = 0.5 * math.atan2(2.0 * A[p][q], A[q][q] - A[p][p])
            
            c = math.cos(theta)
            s = math.sin(theta)
            
            # Apply Givens rotation
            self._apply_givens_rotation(A, V, p, q, c, s, size)
        
        # Extract eigenvalues (diagonal of A)
        eigenvalues = [A[i][i] for i in range(size)]
        
        # Extract eigenvectors (columns of V)
        eigenvectors = [[V[i][j] for i in range(size)] for j in range(size)]
        
        # Sort by eigenvalue magnitude (ascending)
        pairs = list(zip(eigenvalues, eigenvectors))
        pairs.sort(key=lambda x: abs(x[0]))
        
        eigenvalues = [p[0] for p in pairs]
        eigenvectors = [p[1] for p in pairs]
        
        return eigenvalues, eigenvectors
    
    def _apply_givens_rotation(
        self, 
        A: List[List[float]], 
        V: List[List[float]], 
        p: int, 
        q: int, 
        c: float, 
        s: float,
        size: int
    ) -> None:
        """Apply Givens rotation to A and accumulate in V (in-place)."""
        # Rotate rows and columns of A
        for i in range(size):
            if i != p and i != q:
                temp_ip = c * A[i][p] - s * A[i][q]
                temp_iq = s * A[i][p] + c * A[i][q]
                A[i][p] = temp_ip
                A[p][i] = temp_ip
                A[i][q] = temp_iq
                A[q][i] = temp_iq
        
        # Update diagonal and (p,q) element
        app = A[p][p]
        aqq = A[q][q]
        apq = A[p][q]
        
        A[p][p] = c*c*app + s*s*aqq - 2.0*c*s*apq
        A[q][q] = s*s*app + c*c*aqq + 2.0*c*s*apq
        A[p][q] = 0.0
        A[q][p] = 0.0
        
        # Accumulate eigenvectors
        for i in range(size):
            temp_p = c * V[i][p] - s * V[i][q]
            temp_q = s * V[i][p] + c * V[i][q]
            V[i][p] = temp_p
            V[i][q] = temp_q
    
    def _eigenvalue_to_thz(self, eigenvalue: float) -> float:
        """
        Convert Hessian eigenvalue to THz frequency.
        
        Formula: ω (THz) = √(k/m) / (2π × c)
        where k is force constant, m is mass, c is speed of light
        
        Args:
            eigenvalue: Eigenvalue from Hessian (kcal/(mol·Å²))
            
        Returns:
            Frequency in THz
        """
        if eigenvalue <= 0:
            return 0.0
        
        # Convert units: kcal/(mol·Å²) → kg/s²
        # k = eigenvalue × (KCAL_TO_JOULE / AVOGADRO) × 10^20 (Å² to m²)
        k_si = eigenvalue * (self.KCAL_TO_JOULE / self.AVOGADRO) * 1e20
        
        # Mass in kg
        m_si = self.CA_MASS * self.AMU_TO_KG
        
        # Angular frequency (rad/s)
        omega = math.sqrt(k_si / m_si)
        
        # Convert to THz: ω/(2πc) where c in cm/s
        freq_hz = omega / (2.0 * math.pi)
        freq_thz = freq_hz / 1e12
        
        return freq_thz
    
    def _calculate_intensity(
        self, 
        eigenvalue: float, 
        eigenvector: List[float],
        n_atoms: int
    ) -> float:
        """
        Calculate approximate IR intensity for a vibrational mode.
        
        Simplified calculation based on:
        - Magnitude of displacement (larger displacement → higher intensity)
        - Eigenvalue magnitude (stronger modes → higher intensity)
        - Normalized to [0, 1] range
        
        Args:
            eigenvalue: Eigenvalue for this mode
            eigenvector: Eigenvector (3N components)
            n_atoms: Number of atoms
            
        Returns:
            Normalized intensity (0-1)
        """
        # Calculate displacement magnitude
        displacement_mag = 0.0
        for i in range(len(eigenvector)):
            displacement_mag += eigenvector[i] * eigenvector[i]
        displacement_mag = math.sqrt(displacement_mag)
        
        # Intensity proportional to eigenvalue and displacement
        intensity = abs(eigenvalue) * displacement_mag
        
        # Normalize (empirical factor)
        intensity = intensity / (n_atoms * self.spring_constant)
        
        # Clamp to [0, 1]
        intensity = max(0.0, min(1.0, intensity))
        
        return intensity
    
    def calculate_quick_spectrum(
        self,
        ca_coordinates: List[Tuple[float, float, float]],
        n_modes: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Quick spectrum calculation returning (frequency, intensity) pairs.
        
        Simplified interface for rapid testing.
        
        Args:
            ca_coordinates: CA atom positions
            n_modes: Number of modes to calculate
            
        Returns:
            List of (frequency_thz, intensity) tuples
        """
        spectrum = self.calculate_spectrum(ca_coordinates, n_modes=n_modes)
        return [(mode.frequency_thz, mode.intensity) for mode in spectrum.modes]


# Factory function for easy instantiation
def create_vibrational_analyzer(cutoff: float = 10.0, spring_constant: float = 1.0) -> VibrationalAnalyzer:
    """Create a VibrationalAnalyzer with specified parameters."""
    return VibrationalAnalyzer(cutoff=cutoff, spring_constant=spring_constant)
