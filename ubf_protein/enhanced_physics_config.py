"""
Configuration system for enhanced physics features in UBF protein folding.

This module provides centralized configuration management for all physics enhancements,
including feature toggles, parameter tuning, and size-based adaptation.
"""

import os
from dataclasses import dataclass, replace
from typing import Optional, List
from ubf_protein.models import DisulfideBond


@dataclass(frozen=True)
class EnhancedPhysicsConfig:
    """
    Comprehensive configuration for all enhanced physics features.
    
    This immutable configuration object controls:
    - Feature toggles (side-chains, solvent, entropic, refinement)
    - Disulfide bond constraints
    - Parameter tuning for each physics component
    - Size-based adaptation for different protein sizes
    
    Attributes:
        # Feature Toggles
        use_enhanced_energy: Enable enhanced energy calculator with all components
        enable_side_chains: Enable side-chain field interactions
        enable_solvent: Enable distance-dependent solvent screening
        enable_entropic: Enable entropic corrections (coherence + configurational)
        enable_refinement: Enable local refinement with gradient descent
        
        # Disulfide Bonds
        disulfide_bonds: List of disulfide bond constraints
        
        # Side-Chain Parameters
        sidechain_cutoff: Distance cutoff for side-chain interactions (Å)
        sidechain_steric_strength: Scaling factor for steric repulsion
        sidechain_hydrophobic_strength: Scaling factor for hydrophobic interactions
        sidechain_electrostatic_strength: Scaling factor for electrostatic interactions
        
        # Solvent Parameters
        solvent_screening_length: Characteristic length for dielectric screening (Å)
        solvent_buried_dielectric: Dielectric constant for buried residues
        solvent_surface_dielectric: Dielectric constant for surface residues
        solvent_neighbor_cutoff: Distance cutoff for neighbor counting (Å)
        
        # Entropic Parameters
        entropic_temperature: Temperature for free energy calculations (K)
        entropic_window_size: Number of conformations for configurational entropy
        entropic_variance_max: Maximum QCP variance for normalization
        
        # Refinement Parameters
        refinement_max_iterations: Maximum gradient descent iterations
        refinement_step_size: Initial step size for coordinate updates (Å)
        refinement_convergence_tolerance: Energy change threshold for convergence (kcal/mol)
        refinement_epsilon: Finite difference step for gradient calculation (Å)
        refinement_reduction_factor: Step size reduction factor on failure
        
        # Disulfide Parameters
        disulfide_spring_constant: Harmonic potential spring constant (kcal/mol/Ų)
        disulfide_target_distance: Target CA-CA distance for disulfide bonds (Å)
        disulfide_tolerance: Tolerance for constraint satisfaction (Å)
        
        # Adaptive Parameters
        stuck_window: Window size for detecting stuck agents
        stuck_threshold: Energy change threshold for stuck detection (kcal/mol)
        max_iterations: Maximum iterations for agent exploration
    """
    
    # Feature Toggles
    use_enhanced_energy: bool = False
    enable_side_chains: bool = True
    enable_solvent: bool = True
    enable_entropic: bool = True
    enable_refinement: bool = False
    
    # Disulfide Bonds
    disulfide_bonds: Optional[List[DisulfideBond]] = None
    
    # Side-Chain Parameters
    sidechain_cutoff: float = 15.0
    sidechain_steric_strength: float = 1.0
    sidechain_hydrophobic_strength: float = 1.0
    sidechain_electrostatic_strength: float = 1.0
    
    # Solvent Parameters
    solvent_screening_length: float = 3.0
    solvent_buried_dielectric: float = 4.0
    solvent_surface_dielectric: float = 80.0
    solvent_neighbor_cutoff: float = 8.0
    
    # Entropic Parameters
    entropic_temperature: float = 300.0
    entropic_window_size: int = 50
    entropic_variance_max: float = 10.0
    
    # Refinement Parameters
    refinement_max_iterations: int = 100
    refinement_step_size: float = 0.01
    refinement_convergence_tolerance: float = 0.001
    refinement_epsilon: float = 0.01
    refinement_reduction_factor: float = 0.5
    
    # Disulfide Parameters
    disulfide_spring_constant: float = 50.0
    disulfide_target_distance: float = 3.8
    disulfide_tolerance: float = 0.5
    
    # Adaptive Parameters
    stuck_window: int = 30
    stuck_threshold: float = 10.0
    max_iterations: int = 2000
    
    def __post_init__(self):
        """Ensure disulfide_bonds is an empty list if None."""
        if self.disulfide_bonds is None:
            object.__setattr__(self, 'disulfide_bonds', [])
    
    @classmethod
    def baseline(cls) -> 'EnhancedPhysicsConfig':
        """
        Create baseline configuration with all enhancements disabled.
        
        Returns:
            Configuration with use_enhanced_energy=False, suitable for comparison
        """
        return cls(
            use_enhanced_energy=False,
            enable_side_chains=False,
            enable_solvent=False,
            enable_entropic=False,
            enable_refinement=False
        )
    
    @classmethod
    def enhanced_default(cls, disulfide_bonds: Optional[List[DisulfideBond]] = None) -> 'EnhancedPhysicsConfig':
        """
        Create default enhanced configuration with all features enabled.
        
        Args:
            disulfide_bonds: Optional list of disulfide bond constraints
            
        Returns:
            Configuration with use_enhanced_energy=True and all sub-features enabled
        """
        return cls(
            use_enhanced_energy=True,
            enable_side_chains=True,
            enable_solvent=True,
            enable_entropic=True,
            enable_refinement=False,  # Refinement is optional (computationally expensive)
            disulfide_bonds=disulfide_bonds or []
        )
    
    @classmethod
    def for_small_protein(cls, num_residues: int, disulfide_bonds: Optional[List[DisulfideBond]] = None) -> 'EnhancedPhysicsConfig':
        """
        Create configuration optimized for small proteins (<50 residues).
        
        Uses shorter windows and fewer iterations for faster exploration.
        
        Args:
            num_residues: Number of residues in protein
            disulfide_bonds: Optional list of disulfide bond constraints
            
        Returns:
            Configuration optimized for small proteins
        """
        return cls(
            use_enhanced_energy=True,
            enable_side_chains=True,
            enable_solvent=True,
            enable_entropic=True,
            enable_refinement=False,
            disulfide_bonds=disulfide_bonds or [],
            stuck_window=20,
            stuck_threshold=5.0,
            max_iterations=1000,
            refinement_max_iterations=50
        )
    
    @classmethod
    def for_medium_protein(cls, num_residues: int, disulfide_bonds: Optional[List[DisulfideBond]] = None) -> 'EnhancedPhysicsConfig':
        """
        Create configuration optimized for medium proteins (50-150 residues).
        
        Uses balanced parameters for good accuracy and reasonable speed.
        
        Args:
            num_residues: Number of residues in protein
            disulfide_bonds: Optional list of disulfide bond constraints
            
        Returns:
            Configuration optimized for medium proteins
        """
        return cls(
            use_enhanced_energy=True,
            enable_side_chains=True,
            enable_solvent=True,
            enable_entropic=True,
            enable_refinement=False,
            disulfide_bonds=disulfide_bonds or [],
            stuck_window=30,
            stuck_threshold=10.0,
            max_iterations=2000,
            refinement_max_iterations=100
        )
    
    @classmethod
    def for_large_protein(cls, num_residues: int, disulfide_bonds: Optional[List[DisulfideBond]] = None) -> 'EnhancedPhysicsConfig':
        """
        Create configuration optimized for large proteins (>150 residues).
        
        Uses longer windows and more iterations to handle complexity.
        
        Args:
            num_residues: Number of residues in protein
            disulfide_bonds: Optional list of disulfide bond constraints
            
        Returns:
            Configuration optimized for large proteins
        """
        return cls(
            use_enhanced_energy=True,
            enable_side_chains=True,
            enable_solvent=True,
            enable_entropic=True,
            enable_refinement=False,
            disulfide_bonds=disulfide_bonds or [],
            stuck_window=40,
            stuck_threshold=15.0,
            max_iterations=5000,
            refinement_max_iterations=150
        )
    
    @classmethod
    def auto_adapt(cls, num_residues: int, disulfide_bonds: Optional[List[DisulfideBond]] = None) -> 'EnhancedPhysicsConfig':
        """
        Automatically create size-adapted configuration based on protein size.
        
        Args:
            num_residues: Number of residues in protein
            disulfide_bonds: Optional list of disulfide bond constraints
            
        Returns:
            Configuration optimized for the given protein size
        """
        if num_residues < 50:
            return cls.for_small_protein(num_residues, disulfide_bonds)
        elif num_residues < 150:
            return cls.for_medium_protein(num_residues, disulfide_bonds)
        else:
            return cls.for_large_protein(num_residues, disulfide_bonds)
    
    @classmethod
    def from_environment(cls, disulfide_bonds: Optional[List[DisulfideBond]] = None) -> 'EnhancedPhysicsConfig':
        """
        Create configuration from environment variables.
        
        Supported environment variables:
        - UBF_ENHANCED: Enable enhanced energy (true/false)
        - UBF_SIDECHAINS: Enable side-chain fields (true/false)
        - UBF_SOLVENT: Enable solvent screening (true/false)
        - UBF_ENTROPIC: Enable entropic corrections (true/false)
        - UBF_REFINEMENT: Enable local refinement (true/false)
        - UBF_SIDECHAIN_CUTOFF: Side-chain cutoff distance (float)
        - UBF_SOLVENT_SCREENING: Solvent screening length (float)
        - UBF_ENTROPIC_TEMP: Temperature for entropic calculations (float)
        - UBF_REFINEMENT_ITERATIONS: Max refinement iterations (int)
        - UBF_STUCK_WINDOW: Stuck detection window size (int)
        - UBF_STUCK_THRESHOLD: Stuck detection threshold (float)
        - UBF_MAX_ITERATIONS: Max agent iterations (int)
        
        Args:
            disulfide_bonds: Optional list of disulfide bond constraints
            
        Returns:
            Configuration parsed from environment variables
        """
        def parse_bool(value: str) -> bool:
            """Parse boolean from string."""
            return value.lower() in ('true', '1', 'yes', 'on')
        
        def get_env_bool(key: str, default: bool) -> bool:
            """Get boolean from environment variable."""
            value = os.environ.get(key)
            return parse_bool(value) if value else default
        
        def get_env_float(key: str, default: float) -> float:
            """Get float from environment variable."""
            value = os.environ.get(key)
            return float(value) if value else default
        
        def get_env_int(key: str, default: int) -> int:
            """Get integer from environment variable."""
            value = os.environ.get(key)
            return int(value) if value else default
        
        return cls(
            use_enhanced_energy=get_env_bool('UBF_ENHANCED', False),
            enable_side_chains=get_env_bool('UBF_SIDECHAINS', True),
            enable_solvent=get_env_bool('UBF_SOLVENT', True),
            enable_entropic=get_env_bool('UBF_ENTROPIC', True),
            enable_refinement=get_env_bool('UBF_REFINEMENT', False),
            disulfide_bonds=disulfide_bonds or [],
            sidechain_cutoff=get_env_float('UBF_SIDECHAIN_CUTOFF', 15.0),
            solvent_screening_length=get_env_float('UBF_SOLVENT_SCREENING', 3.0),
            entropic_temperature=get_env_float('UBF_ENTROPIC_TEMP', 300.0),
            refinement_max_iterations=get_env_int('UBF_REFINEMENT_ITERATIONS', 100),
            stuck_window=get_env_int('UBF_STUCK_WINDOW', 30),
            stuck_threshold=get_env_float('UBF_STUCK_THRESHOLD', 10.0),
            max_iterations=get_env_int('UBF_MAX_ITERATIONS', 2000)
        )
    
    def with_refinement(self, enable: bool = True, max_iterations: Optional[int] = None) -> 'EnhancedPhysicsConfig':
        """
        Create a new configuration with refinement enabled/disabled.
        
        Args:
            enable: Whether to enable refinement
            max_iterations: Optional custom max iterations for refinement
            
        Returns:
            New configuration with updated refinement settings
        """
        from typing import Any, Dict
        updates: Dict[str, Any] = {'enable_refinement': enable}
        if max_iterations is not None:
            updates['refinement_max_iterations'] = max_iterations
        return replace(self, **updates)
    
    def with_disulfide_bonds(self, bonds: List[DisulfideBond]) -> 'EnhancedPhysicsConfig':
        """
        Create a new configuration with specified disulfide bonds.
        
        Args:
            bonds: List of disulfide bond constraints
            
        Returns:
            New configuration with updated disulfide bonds
        """
        return replace(self, disulfide_bonds=bonds)
    
    def with_custom_parameters(self, **kwargs) -> 'EnhancedPhysicsConfig':
        """
        Create a new configuration with custom parameter overrides.
        
        Args:
            **kwargs: Parameters to override
            
        Returns:
            New configuration with updated parameters
        """
        return replace(self, **kwargs)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameters are out of valid range
        """
        if self.sidechain_cutoff <= 0:
            raise ValueError(f"sidechain_cutoff must be positive, got {self.sidechain_cutoff}")
        
        if self.solvent_screening_length <= 0:
            raise ValueError(f"solvent_screening_length must be positive, got {self.solvent_screening_length}")
        
        if self.solvent_buried_dielectric < 1:
            raise ValueError(f"solvent_buried_dielectric must be >= 1, got {self.solvent_buried_dielectric}")
        
        if self.solvent_surface_dielectric < self.solvent_buried_dielectric:
            raise ValueError(f"solvent_surface_dielectric ({self.solvent_surface_dielectric}) must be >= solvent_buried_dielectric ({self.solvent_buried_dielectric})")
        
        if self.entropic_temperature <= 0:
            raise ValueError(f"entropic_temperature must be positive, got {self.entropic_temperature}")
        
        if self.entropic_window_size < 2:
            raise ValueError(f"entropic_window_size must be >= 2, got {self.entropic_window_size}")
        
        if self.refinement_max_iterations < 1:
            raise ValueError(f"refinement_max_iterations must be >= 1, got {self.refinement_max_iterations}")
        
        if self.refinement_step_size <= 0:
            raise ValueError(f"refinement_step_size must be positive, got {self.refinement_step_size}")
        
        if self.refinement_epsilon <= 0:
            raise ValueError(f"refinement_epsilon must be positive, got {self.refinement_epsilon}")
        
        if not 0 < self.refinement_reduction_factor < 1:
            raise ValueError(f"refinement_reduction_factor must be in (0, 1), got {self.refinement_reduction_factor}")
        
        if self.disulfide_spring_constant <= 0:
            raise ValueError(f"disulfide_spring_constant must be positive, got {self.disulfide_spring_constant}")
        
        if self.disulfide_target_distance <= 0:
            raise ValueError(f"disulfide_target_distance must be positive, got {self.disulfide_target_distance}")
        
        if self.stuck_window < 1:
            raise ValueError(f"stuck_window must be >= 1, got {self.stuck_window}")
        
        if self.stuck_threshold < 0:
            raise ValueError(f"stuck_threshold must be non-negative, got {self.stuck_threshold}")
        
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'use_enhanced_energy': self.use_enhanced_energy,
            'enable_side_chains': self.enable_side_chains,
            'enable_solvent': self.enable_solvent,
            'enable_entropic': self.enable_entropic,
            'enable_refinement': self.enable_refinement,
            'num_disulfide_bonds': len(self.disulfide_bonds) if self.disulfide_bonds else 0,
            'sidechain_cutoff': self.sidechain_cutoff,
            'sidechain_steric_strength': self.sidechain_steric_strength,
            'sidechain_hydrophobic_strength': self.sidechain_hydrophobic_strength,
            'sidechain_electrostatic_strength': self.sidechain_electrostatic_strength,
            'solvent_screening_length': self.solvent_screening_length,
            'solvent_buried_dielectric': self.solvent_buried_dielectric,
            'solvent_surface_dielectric': self.solvent_surface_dielectric,
            'solvent_neighbor_cutoff': self.solvent_neighbor_cutoff,
            'entropic_temperature': self.entropic_temperature,
            'entropic_window_size': self.entropic_window_size,
            'entropic_variance_max': self.entropic_variance_max,
            'refinement_max_iterations': self.refinement_max_iterations,
            'refinement_step_size': self.refinement_step_size,
            'refinement_convergence_tolerance': self.refinement_convergence_tolerance,
            'refinement_epsilon': self.refinement_epsilon,
            'refinement_reduction_factor': self.refinement_reduction_factor,
            'disulfide_spring_constant': self.disulfide_spring_constant,
            'disulfide_target_distance': self.disulfide_target_distance,
            'disulfide_tolerance': self.disulfide_tolerance,
            'stuck_window': self.stuck_window,
            'stuck_threshold': self.stuck_threshold,
            'max_iterations': self.max_iterations
        }
    
    def summary(self) -> str:
        """
        Generate human-readable configuration summary.
        
        Returns:
            Multi-line string summarizing configuration
        """
        lines = ["Enhanced Physics Configuration:"]
        lines.append(f"  Enhanced Energy: {self.use_enhanced_energy}")
        
        if self.use_enhanced_energy:
            lines.append(f"    Side-Chains: {self.enable_side_chains}")
            lines.append(f"    Solvent: {self.enable_solvent}")
            lines.append(f"    Entropic: {self.enable_entropic}")
            lines.append(f"    Refinement: {self.enable_refinement}")
            bond_count = len(self.disulfide_bonds) if self.disulfide_bonds else 0
            lines.append(f"    Disulfide Bonds: {bond_count}")
        
        lines.append(f"  Agent Parameters:")
        lines.append(f"    Stuck Window: {self.stuck_window}")
        lines.append(f"    Stuck Threshold: {self.stuck_threshold}")
        lines.append(f"    Max Iterations: {self.max_iterations}")
        
        if self.enable_refinement:
            lines.append(f"  Refinement Parameters:")
            lines.append(f"    Max Iterations: {self.refinement_max_iterations}")
            lines.append(f"    Step Size: {self.refinement_step_size} Å")
            lines.append(f"    Convergence: {self.refinement_convergence_tolerance} kcal/mol")
        
        return "\n".join(lines)
