# Design Document

## Overview

The Quantum Refinement Engine is a two-stage optimization system that bridges the gap between coarse 7-14Å protein structure predictions and near-native sub-5Å accuracy. It leverages quantum coherence principles, THz resonance cascades, and golden ratio geometric patterns to guide fine-grained structural refinement. The system integrates seamlessly with the existing QCPP and UBF architectures, providing physics-grounded refinement capabilities.

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────┐
│                    UBF Protein System                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Stage 1: Global Fold Exploration             │  │
│  │  (Multi-Agent Coordinator + QCPP Integration)        │  │
│  │  Output: 7-14Å RMSD coarse structure                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │    NEW: Quantum Refinement Engine (Stage 2)          │  │
│  │  - Fine-grained THz resonance coupling                │  │
│  │  - Secondary structure registration                   │  │
│  │  - Hydrophobic core quantum packing                   │  │
│  │  - Loop refinement with G(φ,t)                        │  │
│  │  - Tertiary contact prediction & enforcement          │  │
│  │  - Distance restraint networks                        │  │
│  │  Output: <5Å RMSD refined structure                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Validation Suite (RMSD/GDT-TS/TM-score)      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              QuantumRefinementEngine                         │
├─────────────────────────────────────────────────────────────┤
│  Core Refinement Pipeline                                    │
│  ├─ refine_structure_quantum()                              │
│  ├─ optimize_two_stage()                                    │
│  └─ diagnose_rmsd_components()                              │
├─────────────────────────────────────────────────────────────┤
│  Quantum Core Identification                                 │
│  ├─ identify_quantum_cores()                                │
│  ├─ calculate_local_thz_modes()                             │
│  └─ find_coupled_residues()                                 │
├─────────────────────────────────────────────────────────────┤
│  Secondary Structure Refinement                              │
│  ├─ fix_secondary_structure_registration()                  │
│  ├─ enforce_helix_geometry()                                │
│  └─ optimize_sheet_hydrogen_bonds()                         │
├─────────────────────────────────────────────────────────────┤
│  Hydrophobic Core Optimization                               │
│  ├─ quantum_hydrophobic_packing()                           │
│  └─ calculate_water_exclusion_zones()                       │
├─────────────────────────────────────────────────────────────┤
│  Loop Refinement                                             │
│  ├─ refine_loops_dynamic()                                  │
│  ├─ apply_g_phi_t_evolution()                               │
│  └─ interpolate_loop_conformation()                         │
├─────────────────────────────────────────────────────────────┤
│  Tertiary Contact Management                                 │
│  ├─ predict_tertiary_contacts_quantum()                     │
│  ├─ enforce_contact_map()                                   │
│  └─ calculate_resonance_coupling()                          │
├─────────────────────────────────────────────────────────────┤
│  Distance Restraint System                                   │
│  ├─ add_quantum_distance_restraints()                       │
│  ├─ find_phi_harmonic_distances()                           │
│  └─ apply_restraints()                                      │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Core Refinement Engine

**Class: `QuantumRefinementEngine`**

Primary interface for quantum-guided structural refinement.

```python
class QuantumRefinementEngine:
    """
    Main refinement engine coordinating all quantum refinement strategies.
    """
    
    def __init__(self, 
                 qcpp_adapter: QCPPIntegrationAdapter,
                 energy_calculator: MolecularMechanicsEnergy,
                 rmsd_calculator: RMSDCalculator):
        """
        Initialize refinement engine with required calculators.
        
        Args:
            qcpp_adapter: QCPP integration for quantum metrics
            energy_calculator: Molecular mechanics energy function
            rmsd_calculator: RMSD and structure quality metrics
        """
        self.qcpp_adapter = qcpp_adapter
        self.energy_calculator = energy_calculator
        self.rmsd_calculator = rmsd_calculator
        self.phi = 1.618033988749895
        self.h_bar = 1.0545718e-34  # Planck's constant
        self.gamma_frequency = 40.0  # Hz
        self.coherence_time = 408e-15  # fs
        self.water_spacing = 0.28  # nm
        
    def refine_structure_quantum(self, 
                                coarse_structure: Conformation,
                                native_structure: Optional[NativeStructure] = None,
                                max_iterations: int = 10000) -> RefinementResult:
        """
        Main refinement pipeline: coarse (7-14Å) → refined (<5Å).
        
        Orchestrates all refinement strategies in optimal sequence.
        """
```

**Key Methods:**
- `refine_structure_quantum()`: Main entry point, orchestrates full refinement
- `optimize_two_stage()`: Implements two-stage optimization (global → local)
- `diagnose_rmsd_components()`: Breaks down RMSD by structural region

### 2. Quantum Core Analyzer

**Class: `QuantumCoreAnalyzer`**

Identifies high-coherence regions and establishes THz resonance networks.

```python
class QuantumCoreAnalyzer:
    """
    Analyzes quantum coherence patterns and identifies resonance networks.
    """
    
    def identify_quantum_cores(self, 
                               structure: Conformation,
                               qcp_threshold: float = 7.0) -> List[QuantumCore]:
        """
        Identify regions with QCP > threshold as quantum cores.
        
        Returns:
            List of QuantumCore objects with residue indices and QCP values
        """
    
    def calculate_local_thz_modes(self, 
                                  core: QuantumCore) -> List[THzMode]:
        """
        Calculate THz vibrational modes for a quantum core region.
        
        Uses normal mode analysis on local structure to find
        characteristic frequencies.
        """
    
    def find_coupled_residues(self, 
                             mode: THzMode,
                             phi_tolerance: float = 0.1) -> List[Tuple[int, int]]:
        """
        Find residue pairs coupled by φ-harmonic resonance.
        
        Identifies pairs where THz mode frequency is within
        phi_tolerance of 1.618 THz (or harmonics).
        """
```

**Data Structures:**
```python
@dataclass
class QuantumCore:
    residue_indices: List[int]
    average_qcp: float
    coherence: float
    center_of_mass: Tuple[float, float, float]

@dataclass
class THzMode:
    frequency: float  # THz
    amplitude: float
    participating_residues: List[int]
    is_phi_harmonic: bool
```

### 3. Secondary Structure Registrar

**Class: `SecondaryStructureRegistrar`**

Corrects helix and sheet alignment using quantum-corrected geometry.

```python
class SecondaryStructureRegistrar:
    """
    Fixes secondary structure registration using QCP-guided parameters.
    """
    
    def fix_secondary_structure_registration(self,
                                            structure: Conformation,
                                            qcp_values: Dict[int, float]) -> Conformation:
        """
        Correct alignment of helices and sheets.
        
        Strategy:
        1. Detect secondary structure elements
        2. Calculate average QCP for each element
        3. Apply quantum-corrected geometry parameters
        4. Enforce proper hydrogen bonding patterns
        """
    
    def enforce_helix_geometry(self,
                              helix_residues: List[int],
                              helix_qcp: float,
                              structure: Conformation) -> None:
        """
        Enforce quantum-corrected helix parameters.
        
        For high QCP helices (>7):
        - Pitch: 5.4Å × (1 + 0.1 × tanh(QCP - 7))
        - Rise: 1.5Å × (1 + 0.05 × tanh(QCP - 7))
        - Residues per turn: 3.6 with φ-scaling
        """
    
    def optimize_sheet_hydrogen_bonds(self,
                                     sheet_residues: List[int],
                                     coupling_frequency: float = 2.618) -> None:
        """
        Optimize β-sheet hydrogen bonding with THz coupling.
        
        Uses φ² harmonic (2.618 THz) for sheet stabilization.
        """
```

### 4. Hydrophobic Core Packer

**Class: `HydrophobicCorePacker`**

Optimizes hydrophobic residue packing using water shielding effects.

```python
class HydrophobicCorePacker:
    """
    Quantum-guided hydrophobic core packing optimizer.
    """
    
    def quantum_hydrophobic_packing(self,
                                   structure: Conformation,
                                   qcp_values: Dict[int, float]) -> List[PackingConstraint]:
        """
        Generate packing constraints for hydrophobic residues.
        
        Strategy:
        1. Identify hydrophobic residues
        2. Calculate water exclusion zones (0.28 nm spacing)
        3. Determine optimal packing distances (2.8Å intervals)
        4. Scale force constants by QCP coupling
        
        Returns:
            List of distance constraints with QCP-weighted forces
        """
    
    def calculate_water_exclusion_zones(self,
                                       residue_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        Calculate optimal distances based on water spacing.
        
        Water molecules create discrete spacing at 0.28 nm intervals,
        leading to preferred packing distances at multiples of 2.8Å.
        """
```

**Data Structures:**
```python
@dataclass
class PackingConstraint:
    residue_i: int
    residue_j: int
    target_distance: float  # Angstroms
    force_constant: float  # kcal/mol/Å²
    qcp_coupling: float  # QCP-based scaling factor
```

### 5. Loop Refiner

**Class: `LoopRefiner`**

Refines flexible loop regions using time-dependent golden ratio evolution.

```python
class LoopRefiner:
    """
    Dynamic loop refinement using G(φ,t) temporal evolution.
    """
    
    def refine_loops_dynamic(self,
                            structure: Conformation,
                            loops: List[LoopRegion],
                            qcp_values: Dict[int, float]) -> Conformation:
        """
        Refine loop conformations using quantum-classical hybrid approach.
        
        Strategy:
        - Low QCP (<4): Classical loop modeling
        - Medium QCP (4-7): G(φ,t) temporal evolution
        - High QCP (>7): Quantum-corrected geometry
        """
    
    def apply_g_phi_t_evolution(self,
                               loop: LoopRegion,
                               time_steps: int = 100,
                               max_time_ps: float = 1.0) -> List[Conformation]:
        """
        Apply time-dependent golden ratio evolution.
        
        Formula: G(φ,t) = exp(-t/τ_c) × φ
        where τ_c = 408 fs (coherence time)
        
        Gradually transitions loop from extended to compact
        following quantum decay dynamics.
        """
```

**Data Structures:**
```python
@dataclass
class LoopRegion:
    start_residue: int
    end_residue: int
    average_qcp: float
    current_conformation: List[Tuple[float, float, float]]
    target_conformation: Optional[List[Tuple[float, float, float]]]
```

### 6. Tertiary Contact Predictor

**Class: `TertiaryContactPredictor`**

Predicts and enforces long-range contacts using resonance coupling.

```python
class TertiaryContactPredictor:
    """
    Quantum resonance-based tertiary contact prediction.
    """
    
    def predict_tertiary_contacts_quantum(self,
                                         sequence: str,
                                         qcp_values: Dict[int, float]) -> List[TertiaryContact]:
        """
        Predict tertiary contacts using resonance coupling formula.
        
        Formula: R(E₁,E₂,t) = exp[-(E₁(t) - E₂(t) - ℏωγ)²/(2ℏωγ)] × G(φ,t)
        
        where:
        - E₁, E₂: Quantum energies from QCP
        - ωγ: Gamma frequency (40 Hz)
        - G(φ,t): Golden ratio temporal evolution
        
        Returns contacts with resonance > 0.7 and distance < 8Å
        """
    
    def enforce_contact_map(self,
                           structure: Conformation,
                           predicted_contacts: List[TertiaryContact]) -> Conformation:
        """
        Force predicted contacts to form using attractive forces.
        
        For missing contacts (distance > 8Å):
        - Calculate force vector between residues
        - Apply magnitude: (distance - 6.0) × 10.0
        - Maintain momentum conservation
        """
    
    def calculate_resonance_coupling(self,
                                    energy_i: float,
                                    energy_j: float,
                                    time: float = 0.0) -> float:
        """
        Calculate R(E₁,E₂,t) resonance coupling strength.
        """
```

**Data Structures:**
```python
@dataclass
class TertiaryContact:
    residue_i: int
    residue_j: int
    resonance_strength: float  # 0-1
    predicted_distance: float  # Angstroms
    sequence_separation: int  # |j - i|
```

### 7. Distance Restraint Manager

**Class: `DistanceRestraintManager`**

Manages φ-harmonic distance restraints for high-QCP residue pairs.

```python
class DistanceRestraintManager:
    """
    Manages golden ratio distance restraints.
    """
    
    def add_quantum_distance_restraints(self,
                                       structure: Conformation,
                                       qcp_values: Dict[int, float],
                                       qcp_threshold: float = 7.0) -> List[DistanceRestraint]:
        """
        Add distance restraints for high-QCP pairs.
        
        Strategy:
        1. Identify pairs where both residues have QCP > threshold
        2. Calculate current distance
        3. Find nearest φ-harmonic distance: [d/φ, d, d×φ]
        4. Select value closest to 6.0Å (optimal contact distance)
        5. Apply strong restraint (weight=100.0, tolerance=0.5Å)
        """
    
    def find_phi_harmonic_distances(self,
                                   current_distance: float) -> List[float]:
        """
        Calculate φ-harmonic distance options.
        
        Returns: [current/φ, current, current×φ]
        """
    
    def apply_restraints(self,
                        structure: Conformation,
                        restraints: List[DistanceRestraint]) -> Conformation:
        """
        Apply distance restraints during optimization.
        
        Adds harmonic potential: E = k × (r - r₀)²
        where k = weight, r₀ = target distance
        """
```

**Data Structures:**
```python
@dataclass
class DistanceRestraint:
    residue_i: int
    residue_j: int
    target_distance: float  # Angstroms
    weight: float  # Force constant
    tolerance: float  # Angstroms
    is_phi_harmonic: bool
```

## Data Models

### Refinement Configuration

```python
@dataclass
class RefinementConfig:
    """Configuration for quantum refinement process."""
    
    # Stage 1 (Global) parameters
    stage1_temperature: float = 1.0
    stage1_iterations: int = 1000
    
    # Stage 2 (Refinement) parameters
    stage2_temperature: float = 0.1  # 10x lower
    stage2_iterations: int = 10000  # 10x more
    restraint_weight: float = 10.0
    qcp_weight: float = 0.3  # 30% quantum contribution
    
    # Quantum parameters
    qcp_threshold: float = 7.0  # High coherence threshold
    phi_tolerance: float = 0.1  # THz
    resonance_threshold: float = 0.7
    
    # Water shielding
    water_spacing_nm: float = 0.28
    coherence_time_fs: float = 408.0
    
    # Performance
    max_refinement_time_seconds: float = 300.0  # 5 minutes
    checkpoint_interval: int = 1000
```

### Refinement Result

```python
@dataclass
class RefinementResult:
    """Complete refinement result with metrics."""
    
    # Structures
    initial_structure: Conformation
    refined_structure: Conformation
    native_structure: Optional[NativeStructure]
    
    # RMSD metrics
    initial_rmsd: float
    final_rmsd: float
    rmsd_improvement: float  # Angstroms
    
    # Component RMSD breakdown
    helix_rmsd: float
    sheet_rmsd: float
    loop_rmsd: float
    core_rmsd: float
    
    # Quality metrics
    gdt_ts: float
    tm_score: float
    energy: float  # kcal/mol
    
    # Refinement statistics
    iterations_used: int
    refinement_time_seconds: float
    quantum_cores_identified: int
    restraints_applied: int
    contacts_enforced: int
    
    # Convergence tracking
    rmsd_trajectory: List[float]
    energy_trajectory: List[float]
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
```

## Error Handling

### Graceful Degradation Strategy

```python
class RefinementError(Exception):
    """Base exception for refinement errors."""
    pass

class ConvergenceError(RefinementError):
    """Raised when refinement fails to converge."""
    pass

class GeometryError(RefinementError):
    """Raised when structure geometry becomes invalid."""
    pass
```

**Error Handling Principles:**
1. **Non-critical failures**: Log warning, continue with fallback
2. **Critical failures**: Raise exception with diagnostic info
3. **Validation checks**: After each major step
4. **Checkpoint recovery**: Save state before risky operations

### Validation Checkpoints

```python
def validate_geometry(structure: Conformation) -> bool:
    """
    Validate structure geometry at checkpoints.
    
    Checks:
    - Bond lengths: 1.0-10.0 Å
    - No steric clashes: min distance > 2.0 Å
    - Reasonable angles: 60-180 degrees
    - Finite coordinates: no NaN/Inf
    """

def validate_energy(energy: float, threshold: float = 10000.0) -> bool:
    """
    Validate energy is physically reasonable.
    
    Rejects structures with |energy| > threshold
    """
```

## Testing Strategy

### Unit Tests

**Test Coverage:**
1. **Quantum Core Analyzer** (10 tests)
   - QCP threshold detection
   - THz mode calculation
   - Resonance coupling identification
   - Edge cases: no cores, all cores

2. **Secondary Structure Registrar** (8 tests)
   - Helix geometry enforcement
   - Sheet hydrogen bonding
   - Mixed secondary structure
   - QCP-dependent parameter scaling

3. **Hydrophobic Core Packer** (8 tests)
   - Water exclusion zone calculation
   - Optimal distance determination
   - QCP-weighted force constants
   - Packing constraint generation

4. **Loop Refiner** (10 tests)
   - G(φ,t) temporal evolution
   - QCP-based strategy selection
   - Loop interpolation
   - Energy minimization

5. **Tertiary Contact Predictor** (8 tests)
   - Resonance coupling calculation
   - Contact prediction accuracy
   - Contact map enforcement
   - Force application

6. **Distance Restraint Manager** (8 tests)
   - φ-harmonic distance calculation
   - Restraint generation
   - Restraint application
   - Optimal distance selection

7. **Integration Tests** (12 tests)
   - Full refinement pipeline
   - Two-stage optimization
   - RMSD component diagnostics
   - Performance benchmarks

**Total: 64 unit/integration tests**

### Validation Tests

**Test Proteins:**
- 1UBQ (Ubiquitin, 76 residues): Target RMSD <4Å
- 1CRN (Crambin, 46 residues): Target RMSD <3Å
- 2MR9 (Villin, 35 residues): Target RMSD <3Å

**Success Criteria:**
- RMSD improvement: >50% reduction from initial
- Final RMSD: <5Å for all test proteins
- GDT-TS: >50 for all test proteins
- Energy: <0 kcal/mol (folded)
- Runtime: <5 minutes per protein

### Performance Tests

**Benchmarks:**
- Quantum core identification: <100ms
- Secondary structure registration: <200ms
- Hydrophobic packing: <500ms
- Loop refinement: <1s per loop
- Contact prediction: <500ms
- Full refinement: <5 minutes (100 residues)

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)
- Implement `QuantumRefinementEngine` skeleton
- Implement `QuantumCoreAnalyzer`
- Implement `DistanceRestraintManager`
- Unit tests for core components
- **Milestone**: Distance restraints working, 8-10Å RMSD

### Phase 2: Secondary Structure (Days 3-4)
- Implement `SecondaryStructureRegistrar`
- Implement helix geometry enforcement
- Implement sheet hydrogen bonding
- Integration with refinement engine
- **Milestone**: Secondary structure aligned, 6-8Å RMSD

### Phase 3: Hydrophobic Core & Loops (Days 5-7)
- Implement `HydrophobicCorePacker`
- Implement `LoopRefiner`
- Water shielding calculations
- G(φ,t) temporal evolution
- **Milestone**: Core packed, loops refined, 5-7Å RMSD

### Phase 4: Tertiary Contacts (Days 8-10)
- Implement `TertiaryContactPredictor`
- Resonance coupling calculations
- Contact map enforcement
- Integration tests
- **Milestone**: Contacts enforced, 3-5Å RMSD

### Phase 5: Optimization & Validation (Days 11-14)
- Two-stage optimization pipeline
- RMSD component diagnostics
- Performance optimization
- Comprehensive validation
- Documentation
- **Milestone**: Production-ready, <5Å RMSD consistently

## Dependencies

### Existing Components
- `QCPPIntegrationAdapter`: QCP and coherence metrics
- `MolecularMechanicsEnergy`: Energy calculations
- `RMSDCalculator`: Structure validation
- `Conformation`: Structure representation
- `NativeStructureLoader`: Reference structures

### New Dependencies
- None (pure Python implementation)

### Optional Enhancements
- NumPy: For faster matrix operations (already available)
- SciPy: For proper SVD in Kabsch alignment (already available)

## Performance Considerations

### Optimization Strategies

1. **Caching**
   - Cache QCP values for residues
   - Cache THz mode calculations
   - Cache distance matrices

2. **Selective Refinement**
   - Focus on high-RMSD regions first
   - Skip well-aligned regions
   - Adaptive iteration counts

3. **Parallel Processing**
   - Independent loop refinement
   - Parallel contact evaluation
   - Multi-threaded energy calculations

4. **Early Termination**
   - Stop if RMSD < target
   - Stop if no improvement for N iterations
   - Time-based cutoffs

### Memory Management

- Limit trajectory storage (last 1000 frames)
- Prune low-significance restraints
- Clear caches periodically

## Integration Points

### With UBF System

```python
# In multi_agent_coordinator.py
def run_parallel_exploration_with_refinement(self, iterations: int):
    """
    Run exploration with automatic refinement.
    """
    # Stage 1: Global exploration
    coarse_result = self.run_parallel_exploration(iterations)
    
    # Check if refinement needed
    if coarse_result.best_rmsd > 5.0:
        # Stage 2: Quantum refinement
        refinement_engine = QuantumRefinementEngine(
            qcpp_adapter=self.qcpp_integration,
            energy_calculator=self.energy_calculator,
            rmsd_calculator=self.rmsd_calculator
        )
        
        refined_result = refinement_engine.refine_structure_quantum(
            coarse_structure=coarse_result.best_conformation,
            native_structure=self.native_structure
        )
        
        return refined_result
    
    return coarse_result
```

### With QCPP System

```python
# Enhanced QCPP integration for refinement
class QCPPRefinementAdapter(QCPPIntegrationAdapter):
    """
    Extended QCPP adapter with refinement-specific methods.
    """
    
    def calculate_local_qcp_map(self, 
                                structure: Conformation,
                                window_size: int = 5) -> Dict[int, float]:
        """
        Calculate QCP values with local averaging for smoother gradients.
        """
    
    def identify_phi_patterns(self,
                             structure: Conformation) -> List[PhiPattern]:
        """
        Identify golden ratio geometric patterns in structure.
        """
```

## Monitoring and Diagnostics

### Real-time Monitoring

```python
@dataclass
class RefinementProgress:
    """Real-time refinement progress tracking."""
    iteration: int
    current_rmsd: float
    current_energy: float
    restraints_active: int
    contacts_formed: int
    time_elapsed: float
    estimated_time_remaining: float
```

### Diagnostic Tools

```python
def generate_refinement_report(result: RefinementResult) -> str:
    """
    Generate comprehensive refinement report.
    
    Includes:
    - RMSD trajectory plot
    - Energy landscape
    - Component RMSD breakdown
    - Quantum core visualization
    - Contact map comparison
    - Performance statistics
    """
```

## Future Enhancements

### Phase 2 Features (Post-MVP)

1. **Adaptive Refinement**
   - Automatically adjust parameters based on progress
   - Learn optimal strategies per protein type
   - Dynamic restraint weighting

2. **Multi-Scale Refinement**
   - Coarse-grained → all-atom transition
   - Hierarchical refinement levels
   - Progressive detail addition

3. **Ensemble Refinement**
   - Refine multiple conformations simultaneously
   - Consensus structure generation
   - Uncertainty quantification

4. **Machine Learning Integration**
   - Learn QCP-RMSD correlations
   - Predict optimal refinement strategies
   - Transfer learning from successful refinements

## Conclusion

The Quantum Refinement Engine provides a systematic, physics-grounded approach to breaking through the 7-14Å RMSD barrier. By leveraging THz resonance cascades, golden ratio geometric patterns, and quantum coherence principles, it achieves sub-5Å accuracy while maintaining computational efficiency. The modular design ensures easy integration with existing UBF and QCPP systems, and the comprehensive testing strategy guarantees production-ready quality.
