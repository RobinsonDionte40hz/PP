# Design Document

## Overview

This design document specifies the architecture for enhancing the UBF Protein System with disulfide bond modeling, side-chain field representations, solvent corrections, and entropic contributions. The enhancements integrate seamlessly with the existing SOLID architecture while maintaining PyPy compatibility and performance targets.

### Design Goals

1. **Maintain SOLID Architecture**: All new components follow interface-driven design
2. **PyPy Compatibility**: Pure Python implementation without NumPy/SciPy dependencies in core paths
3. **Performance**: <50ms energy calculation for 300-residue proteins
4. **Backward Compatibility**: Existing code continues to work without modifications
5. **Incremental Adoption**: Features can be enabled independently via configuration

### Expected Impact

- **Crambin (1CRN)**: RMSD improvement from 10Å → 4-5Å, Energy -199 → -320 to -360 kcal/mol
- **SSI (3SSI)**: RMSD improvement to 2-3Å, Energy -427 → -450 to -480 kcal/mol
- **Large Proteins (1PRN)**: Energy +955 → -600 to -800 kcal/mol (currently fails)

## Architecture

### Component Hierarchy

```
Enhanced Physics Layer
├── DisulfideDetector (NEW)
├── SideChainFieldCalculator (NEW)
├── SolventFieldCorrection (NEW)
├── EntropicCalculator (NEW)
├── EnhancedEnergyCalculator (NEW)
└── LocalRefinement (NEW)

Integration Points
├── StructuralValidator (MODIFIED)
├── MaplessMoveGenerator (MODIFIED)
├── MultiAgentCoordinator (MODIFIED)
└── TestFramework (MODIFIED)
```

### Data Flow

```
PDB File → DisulfideDetector → [DisulfideBond list]
                                        ↓
Sequence → SideChainFieldCalculator → [SideChainField list]
                                        ↓
Conformation → EnhancedEnergyCalculator → Total Energy
                ├── Base Energy (existing)
                ├── Side-Chain Interactions
                ├── Disulfide Constraints
                └── Entropic Corrections
                                        ↓
                        LocalRefinement → Refined Conformation
```


## Components and Interfaces

### 1. DisulfideDetector

**Purpose**: Detect and represent disulfide bonds from PDB files or predict from sequence.

**Interface**:
```python
class IDisulfideDetector(ABC):
    @abstractmethod
    def detect_from_pdb(self, pdb_file: str) -> List[DisulfideBond]:
        """Parse SSBOND records from PDB file."""
        pass
    
    @abstractmethod
    def predict_from_sequence(self, sequence: str) -> List[DisulfideBond]:
        """Predict likely disulfide bonds from cysteine positions."""
        pass
```

**Data Model**:
```python
@dataclass(frozen=True)
class DisulfideBond:
    residue_i: int          # First cysteine index
    residue_j: int          # Second cysteine index
    distance: float = 3.8   # Target CA-CA distance (Å)
    tolerance: float = 1.0  # Acceptable deviation (Å)
```

**Implementation Details**:
- Parse SSBOND records: `SSBOND   1 CYS A   6    CYS A  127`
- Extract residue indices from columns 17-21 and 31-35
- For prediction: pair cysteines by sequence proximity (simple heuristic)
- Return empty list if no cysteines or no bonds detected

### 2. SideChainFieldCalculator

**Purpose**: Represent amino acid side-chains as scalar fields with physical properties.

**Interface**:
```python
class ISideChainFieldCalculator(ABC):
    @abstractmethod
    def create_field(self, residue_type: str) -> SideChainField:
        """Create field from amino acid type."""
        pass
    
    @abstractmethod
    def calculate_field_interaction(
        self, field_i: SideChainField, field_j: SideChainField, 
        ca_distance: float
    ) -> float:
        """Calculate interaction energy between two fields."""
        pass
```

**Data Model**:
```python
@dataclass(frozen=True)
class SideChainField:
    residue_type: str
    hydrophobicity: float   # -2.53 to +1.38 (Kyte-Doolittle scale)
    volume: float           # Å³ (effective side-chain volume)
    charge: float           # -1, 0, +1
    radius: float           # Effective field radius (Å)
    field_sigma: float = 2.0  # Gaussian decay width (Å)
```

**Amino Acid Properties Table**:
- 20 standard amino acids with experimentally derived values
- Hydrophobicity: Kyte-Doolittle scale
- Volume: From crystallographic data
- Charge: Ionization state at pH 7.0
- Radius: Calculated from volume as sphere

**Interaction Energy Components**:
1. **Steric Repulsion**: `E = 10.0 * (r_overlap)²` when fields overlap
2. **Hydrophobic Effect**: `E = -2.0 * |h_i * h_j| * exp(-r²/2σ²)` for like pairs
3. **Electrostatic**: `E = 332 * q_i * q_j / (ε * r)` with effective dielectric


### 3. SolventFieldCorrection

**Purpose**: Model solvent screening through distance-dependent dielectric.

**Interface**:
```python
class ISolventFieldCorrection(ABC):
    @abstractmethod
    def calculate_effective_dielectric(
        self, distance: float, burial_factor: float
    ) -> float:
        """Calculate distance and burial-dependent dielectric."""
        pass
    
    @abstractmethod
    def calculate_burial_factor(
        self, conformation: Conformation, residue_index: int
    ) -> float:
        """Calculate how buried a residue is (0=surface, 1=core)."""
        pass
```

**Dielectric Model**:
```
ε(r, burial) = ε_buried + (ε_water - ε_buried) * (1 - burial) * (1 - exp(-r/λ))

Where:
- ε_water = 80.0 (bulk water)
- ε_buried = 4.0 (protein interior)
- λ = 3.0 Å (screening length)
```

**Burial Calculation**:
- Count neighbors within 8.0 Å cutoff
- Normalize: 12 neighbors = fully buried (burial = 1.0)
- Surface residues: 0-4 neighbors (burial = 0.0-0.33)
- Core residues: 10-12+ neighbors (burial = 0.83-1.0)

### 4. EntropicCalculator

**Purpose**: Calculate entropic contributions from coherence field variance and configurational diversity.

**Interface**:
```python
class IEntropicCalculator(ABC):
    @abstractmethod
    def calculate_coherence_entropy(
        self, qcp_values: np.ndarray, temperature: float = 300.0
    ) -> float:
        """Calculate entropy from QCP field variance."""
        pass
    
    @abstractmethod
    def calculate_configurational_entropy(
        self, conformation: Conformation, 
        previous_conformations: List[Conformation],
        window_size: int = 50
    ) -> float:
        """Calculate entropy from structural diversity."""
        pass
```

**Coherence Entropy Model**:
```
S_coherence = k_B * variance_normalized * 10.0
ΔG = -T * S

Where:
- k_B = 0.001987 kcal/(mol·K)
- variance_normalized = min(1.0, var(QCP) / 10.0)
- High variance → high entropy → favorable at high T
```

**Configurational Entropy Model**:
```
S_config = k_B * ln(1 + RMSD_avg)
ΔG = -T * S

Where:
- RMSD_avg = average RMSD over last 50 conformations
- High diversity → high entropy
```


### 5. EnhancedEnergyCalculator

**Purpose**: Unified energy function combining all physics enhancements.

**Interface**:
```python
class IEnhancedEnergyCalculator(IPhysicsCalculator):
    @abstractmethod
    def calculate_total_energy(
        self, conformation: Conformation,
        qcp_values: Optional[np.ndarray] = None,
        previous_conformations: Optional[List[Conformation]] = None
    ) -> float:
        """Calculate total energy with all enhancements."""
        pass
    
    @abstractmethod
    def calculate_with_breakdown(
        self, conformation: Conformation
    ) -> Dict[str, float]:
        """Return energy breakdown for debugging."""
        pass
```

**Energy Composition**:
```
E_total = E_base + E_sidechain + E_disulfide + E_entropic

Where:
- E_base: Existing molecular mechanics (bond, angle, dihedral, vdw, elec, hbond)
- E_sidechain: Side-chain field interactions with solvent correction
- E_disulfide: Harmonic constraint energy for S-S bonds
- E_entropic: Coherence + configurational entropy contributions
```

**Implementation Strategy**:
- Initialize side-chain fields once during construction
- Cache burial factors for residues (recalculate only when conformation changes significantly)
- Apply 15.0 Å cutoff for side-chain interactions
- Skip interactions for residues within 3 positions in sequence
- Use neighbor lists for O(N) scaling instead of O(N²) where possible

**Performance Optimization**:
- Lazy evaluation: only calculate components when enabled
- Caching: burial factors, neighbor lists, field strengths
- Early termination: skip calculations for distant residue pairs
- Target: <50ms for 300 residues

### 6. LocalRefinement

**Purpose**: Perform gradient descent energy minimization on conformations.

**Interface**:
```python
class ILocalRefinement(ABC):
    @abstractmethod
    def refine_conformation(
        self, conformation: Conformation,
        max_steps: int = 100,
        step_size: float = 0.01,
        tolerance: float = 0.001
    ) -> Conformation:
        """Perform local energy minimization."""
        pass
```

**Algorithm**:
1. Calculate numerical gradient using central differences (ε = 0.01 Å)
2. Update coordinates: `x_new = x_old - α * ∇E`
3. Validate geometry after each step
4. Reduce step size by 0.5 if geometry becomes invalid or energy increases
5. Terminate when `|ΔE| < tolerance` or `steps > max_steps`

**Gradient Calculation**:
```python
∇E[i,dim] = (E(x + ε) - E(x - ε)) / (2ε)

For each residue i and dimension (x,y,z)
```

**Geometry Validation**:
- Check bond lengths: 3.0 Å < d < 4.5 Å
- Check for steric clashes: no CA-CA < 2.5 Å
- Reject moves that violate constraints


## Data Models

### DisulfideBond
```python
@dataclass(frozen=True)
class DisulfideBond:
    """Immutable representation of a disulfide bond."""
    residue_i: int
    residue_j: int
    distance: float = 3.8
    tolerance: float = 1.0
    
    def is_satisfied(self, ca_distance: float) -> bool:
        """Check if bond constraint is satisfied."""
        return abs(ca_distance - self.distance) <= self.tolerance
```

### SideChainField
```python
@dataclass(frozen=True)
class SideChainField:
    """Immutable scalar field representation of side-chain."""
    residue_type: str
    hydrophobicity: float
    volume: float
    charge: float
    radius: float
    field_sigma: float = 2.0
    
    def field_strength_at(self, distance: float) -> float:
        """Gaussian decay: exp(-r²/2σ²)"""
        return math.exp(-distance**2 / (2 * self.field_sigma**2))
```

### EnhancedEnergyComponents
```python
@dataclass
class EnhancedEnergyComponents:
    """Breakdown of energy contributions for analysis."""
    base_energy: float
    sidechain_energy: float
    disulfide_energy: float
    coherence_entropy: float
    configurational_entropy: float
    total_energy: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            'base': self.base_energy,
            'sidechain': self.sidechain_energy,
            'disulfide': self.disulfide_energy,
            'coherence_entropy': self.coherence_entropy,
            'config_entropy': self.configurational_entropy,
            'total': self.total_energy
        }
```

## Integration with Existing Components

### StructuralValidator (Modified)

**New Method**:
```python
def validate_disulfide_bonds(
    self, conformation: Conformation,
    disulfide_bonds: List[DisulfideBond]
) -> Tuple[bool, List[str]]:
    """
    Validate disulfide bond constraints.
    
    Returns:
        (is_valid, violation_messages)
    """
    violations = []
    for bond in disulfide_bonds:
        ca_i = conformation.coordinates[bond.residue_i]
        ca_j = conformation.coordinates[bond.residue_j]
        distance = np.linalg.norm(ca_i - ca_j)
        
        if not bond.is_satisfied(distance):
            violations.append(
                f"S-S bond {bond.residue_i}-{bond.residue_j}: "
                f"{distance:.2f}Å (target {bond.distance}±{bond.tolerance}Å)"
            )
    
    return len(violations) == 0, violations
```

**Integration Point**: Called during `validate_conformation()` if disulfide bonds exist.


### MaplessMoveGenerator (Modified)

**New Move Type**:
```python
class MoveType(Enum):
    # ... existing move types ...
    DISULFIDE_CONSTRAINT = "disulfide_constraint"
```

**New Method**:
```python
def _generate_disulfide_moves(
    self, conformation: Conformation,
    disulfide_bonds: List[DisulfideBond]
) -> List[ConformationalMove]:
    """
    Generate moves that pull cysteines closer to satisfy S-S bonds.
    
    Strategy:
    - For each unsatisfied bond (distance > target + tolerance)
    - Calculate direction vector from residue_i to residue_j
    - Generate move that translates residue_i toward residue_j
    - Step size: 0.5 Å (conservative to maintain stability)
    """
    moves = []
    for bond in disulfide_bonds:
        ca_i = conformation.coordinates[bond.residue_i]
        ca_j = conformation.coordinates[bond.residue_j]
        current_dist = np.linalg.norm(ca_i - ca_j)
        
        if current_dist > bond.distance + bond.tolerance:
            direction = (ca_j - ca_i) / current_dist
            move_vector = direction * 0.5  # 0.5 Å step
            
            moves.append(ConformationalMove(
                move_type=MoveType.DISULFIDE_CONSTRAINT,
                residue_index=bond.residue_i,
                parameters={
                    'vector': move_vector,
                    'bond': bond,
                    'current_distance': current_dist
                }
            ))
    
    return moves
```

**Integration Point**: Called in `generate_moves()` after base moves if disulfide bonds exist.

### MultiAgentCoordinator (Modified)

**Constructor Update**:
```python
def __init__(
    self,
    protein_sequence: str,
    qcpp_integration: Optional[QCPPIntegrationAdapter] = None,
    disulfide_bonds: Optional[List[DisulfideBond]] = None,  # NEW
    enable_sidechain_fields: bool = False,  # NEW
    enable_solvent_correction: bool = False,  # NEW
    enable_entropic_correction: bool = False,  # NEW
    enable_local_refinement: bool = False,  # NEW
    **kwargs
):
    self.disulfide_bonds = disulfide_bonds or []
    self.enable_sidechain_fields = enable_sidechain_fields
    # ... initialize enhanced calculators if enabled ...
```

**Energy Calculator Selection**:
```python
def _create_energy_calculator(self) -> IPhysicsCalculator:
    """Create appropriate energy calculator based on enabled features."""
    if (self.enable_sidechain_fields or 
        self.enable_solvent_correction or 
        self.enable_entropic_correction or 
        len(self.disulfide_bonds) > 0):
        return EnhancedEnergyCalculator(
            sequence=self.protein_sequence,
            disulfide_bonds=self.disulfide_bonds,
            enable_sidechain=self.enable_sidechain_fields,
            enable_solvent=self.enable_solvent_correction,
            enable_entropy=self.enable_entropic_correction
        )
    else:
        return MolecularMechanicsEnergy()  # Existing calculator
```


### Test Framework (Modified)

**test_protein.py Updates**:
```python
def run_protein_test(
    sequence: str,
    pdb_file: Optional[Path] = None,
    use_disulfides: bool = True,  # NEW
    use_sidechains: bool = False,  # NEW
    use_solvent: bool = False,  # NEW
    use_entropy: bool = False,  # NEW
    use_refinement: bool = False,  # NEW
    **kwargs
):
    # Detect disulfide bonds if PDB available
    disulfide_bonds = []
    if pdb_file and use_disulfides:
        detector = DisulfideDetector()
        disulfide_bonds = detector.detect_from_pdb(str(pdb_file))
        if disulfide_bonds:
            print(f"✓ Detected {len(disulfide_bonds)} disulfide bonds:")
            for bond in disulfide_bonds:
                print(f"  - CYS {bond.residue_i} ↔ CYS {bond.residue_j}")
    
    # Create coordinator with enhancements
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        disulfide_bonds=disulfide_bonds,
        enable_sidechain_fields=use_sidechains,
        enable_solvent_correction=use_solvent,
        enable_entropic_correction=use_entropy,
        enable_local_refinement=use_refinement,
        **kwargs
    )
    
    # ... rest of test logic ...
```

**Command-Line Arguments**:
```bash
python test_protein.py --pdb 1CRN \
    --use-disulfides \
    --use-sidechains \
    --use-solvent \
    --use-entropy \
    --use-refinement
```

## Error Handling

### Validation Failures

**Disulfide Bond Violations**:
- Log warning with specific bond and distance
- Continue exploration (don't crash)
- Track violation count in metrics

**Invalid Geometry After Refinement**:
- Reduce step size by 0.5
- Retry up to 5 times
- If still invalid, return original conformation

**Missing PDB Data**:
- If SSBOND records missing, fall back to sequence prediction
- If sequence has no cysteines, return empty bond list
- Log info message about detection method used

### Performance Degradation

**Energy Calculation Timeout**:
- If calculation exceeds 100ms, log warning
- Disable most expensive component (side-chains) for next iteration
- Continue with reduced physics model

**Memory Pressure**:
- If memory usage exceeds 200MB per agent, reduce cache sizes
- Clear old burial factor cache entries
- Log memory usage statistics


## Testing Strategy

### Unit Tests

**DisulfideDetector Tests** (`test_disulfide_detector.py`):
- Test SSBOND parsing with 0, 1, 3 bonds
- Test sequence prediction with 0, 2, 4, 6 cysteines
- Test invalid PDB format handling
- Test bond constraint satisfaction checking

**SideChainFieldCalculator Tests** (`test_sidechain_fields.py`):
- Test field creation for all 20 amino acids
- Test hydrophobic-hydrophobic attraction
- Test hydrophobic-hydrophilic repulsion
- Test electrostatic interactions (positive-negative, like charges)
- Test steric repulsion at close distances
- Test field strength decay with distance

**SolventFieldCorrection Tests** (`test_solvent_correction.py`):
- Test burial factor calculation (surface, intermediate, core)
- Test dielectric constant calculation at various distances
- Test combined distance and burial effects
- Test edge cases (0 neighbors, 20+ neighbors)

**EntropicCalculator Tests** (`test_entropic_calculator.py`):
- Test coherence entropy with low/high variance QCP values
- Test configurational entropy with diverse/similar conformations
- Test temperature dependence
- Test edge cases (no previous conformations, single conformation)

**EnhancedEnergyCalculator Tests** (`test_enhanced_energy.py`):
- Test energy calculation with each component enabled/disabled
- Test energy breakdown reporting
- Test performance with 50, 100, 300 residue proteins
- Test caching behavior
- Test numerical stability with extreme conformations

**LocalRefinement Tests** (`test_local_refinement.py`):
- Test convergence on simple test cases
- Test step size reduction on invalid geometry
- Test maximum iteration limit
- Test gradient calculation accuracy
- Test performance (should complete in <5s for 100 residues)

### Integration Tests

**End-to-End Disulfide Test** (`test_disulfide_integration.py`):
```python
def test_crambin_with_disulfides():
    """Test Crambin (1CRN) with 3 disulfide bonds."""
    # Load PDB and detect bonds
    detector = DisulfideDetector()
    bonds = detector.detect_from_pdb("pdb_cache/pdb1crn.ent")
    assert len(bonds) == 3
    
    # Run simulation
    coordinator = MultiAgentCoordinator(
        protein_sequence=CRAMBIN_SEQUENCE,
        disulfide_bonds=bonds,
        enable_sidechain_fields=True
    )
    coordinator.initialize_agents(count=10)
    results = coordinator.run_parallel_exploration(iterations=500)
    
    # Verify improvement
    assert results.best_energy < -250  # Better than -199 baseline
    assert results.best_rmsd < 8.0     # Better than 10Å baseline
    
    # Verify bonds satisfied
    validator = StructuralValidator()
    is_valid, violations = validator.validate_disulfide_bonds(
        results.best_conformation, bonds
    )
    assert is_valid, f"Disulfide violations: {violations}"
```

**Progressive Enhancement Test** (`test_progressive_enhancements.py`):
```python
def test_progressive_improvements():
    """Test that each enhancement improves results."""
    sequence = CRAMBIN_SEQUENCE
    pdb_file = "pdb_cache/pdb1crn.ent"
    
    # Baseline
    result_baseline = run_test(sequence, pdb_file, 
                               use_disulfides=False, use_sidechains=False)
    
    # + Disulfides
    result_disulfides = run_test(sequence, pdb_file,
                                 use_disulfides=True, use_sidechains=False)
    assert result_disulfides.energy < result_baseline.energy
    
    # + Side-chains
    result_sidechains = run_test(sequence, pdb_file,
                                 use_disulfides=True, use_sidechains=True)
    assert result_sidechains.energy < result_disulfides.energy
    
    # + Solvent
    result_solvent = run_test(sequence, pdb_file,
                              use_disulfides=True, use_sidechains=True,
                              use_solvent=True)
    assert result_solvent.energy < result_sidechains.energy
```

### Validation Tests

**Known Protein Structures** (`test_known_structures.py`):
- Crambin (1CRN): 46 residues, 3 S-S bonds → Target RMSD <5Å
- SSI (3SSI): 113 residues, S-S bonds → Target RMSD <3Å
- Lysozyme (1LYZ): 129 residues, 4 S-S bonds → Target RMSD <4Å
- Compare energy and RMSD with/without enhancements

**Performance Benchmarks** (`test_performance.py`):
- Energy calculation: <50ms for 300 residues
- Refinement: <5s for 100 residues
- Memory: <100MB per agent with enhancements
- Throughput: 10 agents × 1000 iterations < 5 minutes


## Configuration

### Feature Flags

**AdaptiveConfig Extension**:
```python
@dataclass
class EnhancedPhysicsConfig:
    """Configuration for physics enhancements."""
    
    # Feature toggles
    enable_disulfides: bool = True
    enable_sidechain_fields: bool = False
    enable_solvent_correction: bool = False
    enable_entropic_correction: bool = False
    enable_local_refinement: bool = False
    
    # Disulfide parameters
    disulfide_spring_constant: float = 50.0  # kcal/mol/Å²
    disulfide_target_distance: float = 3.8   # Å
    disulfide_tolerance: float = 1.0         # Å
    
    # Side-chain parameters
    sidechain_cutoff: float = 15.0           # Å
    sidechain_field_sigma: float = 2.0       # Å
    hydrophobic_strength: float = 2.0        # kcal/mol
    steric_strength: float = 10.0            # kcal/mol
    
    # Solvent parameters
    dielectric_water: float = 80.0
    dielectric_buried: float = 4.0
    screening_length: float = 3.0            # Å
    burial_cutoff: float = 8.0               # Å
    burial_max_neighbors: int = 12
    
    # Entropy parameters
    entropy_temperature: float = 300.0       # K
    entropy_window_size: int = 50
    coherence_variance_max: float = 10.0
    
    # Refinement parameters
    refinement_max_steps: int = 100
    refinement_step_size: float = 0.01       # Å
    refinement_tolerance: float = 0.001      # kcal/mol
    refinement_gradient_epsilon: float = 0.01  # Å
```

### Size-Based Adaptation

**Small Proteins (<50 residues)**:
- Enable all enhancements by default
- Aggressive refinement (200 steps)
- Tight disulfide tolerance (0.5 Å)

**Medium Proteins (50-150 residues)**:
- Enable disulfides and side-chains
- Standard refinement (100 steps)
- Standard tolerance (1.0 Å)

**Large Proteins (>150 residues)**:
- Enable disulfides only
- Light refinement (50 steps)
- Relaxed tolerance (1.5 Å)
- Prioritize performance over accuracy

### Environment Variables

```bash
# Enable all enhancements
export UBF_ENABLE_ENHANCED_PHYSICS=true

# Enable specific features
export UBF_ENABLE_DISULFIDES=true
export UBF_ENABLE_SIDECHAINS=true
export UBF_ENABLE_SOLVENT=true
export UBF_ENABLE_ENTROPY=true
export UBF_ENABLE_REFINEMENT=true

# Performance tuning
export UBF_SIDECHAIN_CUTOFF=12.0
export UBF_REFINEMENT_MAX_STEPS=50
```

## Performance Considerations

### Computational Complexity

**Without Enhancements**: O(N²) for base energy
**With Side-Chains**: O(N²) for pairwise interactions (same asymptotic complexity)
**With Refinement**: O(N² × steps) for gradient descent

### Optimization Strategies

1. **Neighbor Lists**: Update only when conformation changes significantly (RMSD > 1.0 Å)
2. **Cutoff Distances**: 15.0 Å for side-chains, 12.0 Å for electrostatics
3. **Lazy Evaluation**: Only calculate enabled components
4. **Caching**: Burial factors, field strengths, dielectric constants
5. **Early Termination**: Skip distant pairs, converge refinement early

### Memory Usage

**Per Agent**:
- Base system: ~30 MB
- Side-chain fields: +5 MB (20 fields × 250 KB)
- Burial cache: +2 MB (300 residues × 8 bytes)
- Refinement gradient: +7 MB (300 residues × 3 dims × 8 bytes)
- **Total**: ~44 MB per agent (within 100 MB target)

### Parallelization

- All components are thread-safe (immutable data models)
- Multi-agent exploration parallelizes naturally
- Energy calculations can be parallelized across conformations
- Refinement is sequential but fast (<5s)


## Implementation Phases

### Phase 1: Foundation (Disulfide Detection)
**Duration**: 1 week  
**Files**: `disulfide_detector.py`, `test_disulfide_detector.py`  
**Deliverables**:
- DisulfideBond data model
- DisulfideDetector with PDB parsing
- Sequence-based prediction
- Unit tests (15 tests)

### Phase 2: Side-Chain Fields
**Duration**: 1.5 weeks  
**Files**: `sidechain_fields.py`, `test_sidechain_fields.py`  
**Deliverables**:
- SideChainField data model
- Amino acid property database (20 types)
- Field interaction calculator
- Unit tests (25 tests)

### Phase 3: Solvent Correction
**Duration**: 1 week  
**Files**: `solvent_field.py`, `test_solvent_correction.py`  
**Deliverables**:
- Distance-dependent dielectric
- Burial factor calculation
- Integration with side-chain calculator
- Unit tests (15 tests)

### Phase 4: Entropic Corrections
**Duration**: 1 week  
**Files**: `entropic_correction.py`, `test_entropic_calculator.py`  
**Deliverables**:
- Coherence entropy calculator
- Configurational entropy calculator
- Temperature dependence
- Unit tests (12 tests)

### Phase 5: Enhanced Energy Calculator
**Duration**: 1.5 weeks  
**Files**: `enhanced_energy_calculator.py`, `test_enhanced_energy.py`  
**Deliverables**:
- Unified energy function
- Component breakdown
- Performance optimization
- Unit tests (20 tests)

### Phase 6: Local Refinement
**Duration**: 1 week  
**Files**: `local_refinement.py`, `test_local_refinement.py`  
**Deliverables**:
- Gradient descent optimizer
- Numerical gradient calculation
- Geometry validation
- Unit tests (15 tests)

### Phase 7: Integration
**Duration**: 1.5 weeks  
**Files**: Modified existing files  
**Deliverables**:
- StructuralValidator updates
- MaplessMoveGenerator updates
- MultiAgentCoordinator updates
- Test framework updates
- Integration tests (10 tests)

### Phase 8: Validation & Optimization
**Duration**: 1.5 weeks  
**Files**: Validation test suite  
**Deliverables**:
- Known structure validation (Crambin, SSI, Lysozyme)
- Performance benchmarks
- Documentation updates
- Bug fixes and optimization

**Total Duration**: 10 weeks  
**Total Tests**: 112+ tests  
**Total Code**: ~2,500 lines

## Migration Path

### Backward Compatibility

**Existing Code**: Continues to work without modifications
```python
# Old code still works
coordinator = MultiAgentCoordinator(protein_sequence="ACDEFGH")
```

**Opt-In Enhancements**: Enable features explicitly
```python
# New code with enhancements
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    disulfide_bonds=bonds,
    enable_sidechain_fields=True
)
```

### Gradual Adoption

1. **Week 1-2**: Deploy disulfide detection (minimal risk)
2. **Week 3-5**: Deploy side-chain fields (test on small proteins)
3. **Week 6-7**: Deploy solvent correction (validate energy scales)
4. **Week 8-9**: Deploy entropy and refinement (full validation)
5. **Week 10**: Production deployment with monitoring

### Rollback Strategy

- Feature flags allow instant disable
- Separate modules minimize coupling
- Existing energy calculator remains available
- Checkpoints allow resuming with different settings

## Success Metrics

### Accuracy Improvements

- **Crambin (1CRN)**: RMSD <5Å (from 10Å), Energy <-320 kcal/mol (from -199)
- **SSI (3SSI)**: RMSD <3Å, Energy <-450 kcal/mol (from -427)
- **Lysozyme (1LYZ)**: RMSD <4Å, Energy <-500 kcal/mol
- **Large Proteins**: Negative energy (currently positive for 1PRN)

### Performance Targets

- Energy calculation: <50ms for 300 residues ✓
- Refinement: <5s for 100 residues ✓
- Memory: <100MB per agent ✓
- Throughput: No degradation vs baseline ✓

### Code Quality

- Test coverage: >90% ✓
- All tests passing ✓
- Documentation complete ✓
- PyPy compatible ✓

## Risks and Mitigations

### Risk 1: Performance Degradation
**Mitigation**: Extensive benchmarking, caching, cutoffs, lazy evaluation

### Risk 2: Numerical Instability
**Mitigation**: Careful parameter tuning, gradient clipping, geometry validation

### Risk 3: Integration Complexity
**Mitigation**: Interface-driven design, incremental integration, comprehensive testing

### Risk 4: Accuracy Not Improved
**Mitigation**: Progressive validation, parameter tuning, fallback to baseline

## Future Extensions

### Post-MVP Enhancements

1. **All-Atom Modeling**: Explicit side-chain atoms instead of fields
2. **Explicit Solvent**: Water molecules in active site
3. **Metal Coordination**: Zinc fingers, iron-sulfur clusters
4. **Ligand Binding**: Small molecule docking
5. **Machine Learning**: Train on successful conformations

### Research Directions

1. **Quantum Effects**: Integrate QCPP more deeply with side-chain fields
2. **Consciousness-Physics Coupling**: Map consciousness to quantum coherence
3. **Multi-Scale Modeling**: Coarse-grain → all-atom refinement
4. **Ensemble Generation**: Multiple conformations for dynamics

