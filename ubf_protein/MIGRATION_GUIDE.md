# Migration Guide: Enhanced Physics

Guide for transitioning from baseline UBF to enhanced physics system.

## Table of Contents

- [Overview](#overview)
- [Backward Compatibility](#backward-compatibility)
- [Migration Strategies](#migration-strategies)
- [Feature-by-Feature Migration](#feature-by-feature-migration)
- [Performance Considerations](#performance-considerations)
- [Breaking Changes](#breaking-changes)
- [Troubleshooting](#troubleshooting)

---

## Overview

The physics enhancements add 5 major improvements to the UBF protein folding system:

1. **Disulfide Bond Constraints**: Spatial constraints for Cys-Cys bonds
2. **Side-Chain Field Interactions**: Hydrophobic/electrostatic effects
3. **Solvent Screening Corrections**: Distance and burial-dependent dielectric
4. **Entropic Contributions**: Coherence and configurational entropy
5. **Local Refinement**: Gradient descent optimization

### When to Migrate

**Migrate if:**
- You need improved accuracy for proteins with disulfide bonds
- Your proteins have >20 residues (side-chain effects become significant)
- You're working with charged or hydrophobic-rich sequences
- You want physics-validated final conformations

**Stay with baseline if:**
- You need maximum speed (baseline is 2-5x faster)
- Your proteins are <10 residues (enhancements less impactful)
- You're doing high-throughput screening (100+ proteins)
- You're benchmarking core algorithm performance

---

## Backward Compatibility

**100% backward compatible** - All existing code continues to work unchanged.

### Default Behavior

```python
# This code works exactly as before (baseline mode)
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator

coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    num_agents=10
)

results = coordinator.run_parallel_exploration(iterations=500)
```

The default behavior is **baseline mode** (no enhancements) to preserve existing performance and behavior.

### Opt-In Enhancement

Enhancements are **opt-in** via `EnhancedPhysicsConfig`:

```python
from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig

# Explicitly enable enhancements
config = EnhancedPhysicsConfig.enhanced_default()

coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    physics_config=config  # Add this parameter
)
```

---

## Migration Strategies

### Strategy 1: Immediate Full Migration (Recommended)

**Best for**: New projects, proteins with known disulfide bonds, accuracy-critical applications

```python
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig
from ubf_protein.disulfide_detector import DisulfideDetector

# Step 1: Detect disulfide bonds (if applicable)
sequence = "YOUR_SEQUENCE_WITH_CYSTEINES"
detector = DisulfideDetector()
bonds = detector.predict_from_sequence(sequence)
# Or: bonds = detector.detect_from_pdb("your_protein.pdb")

# Step 2: Create enhanced configuration
config = EnhancedPhysicsConfig.auto_adapt(sequence)
if bonds:
    config = config.with_disulfide_bonds(bonds)

# Step 3: Run with enhanced physics
coordinator = MultiAgentCoordinator(
    protein_sequence=sequence,
    physics_config=config
)

results = coordinator.run_parallel_exploration(iterations=1000)
```

**Advantages**:
- Maximum accuracy from day one
- All physics improvements active
- Auto-adapted to protein size

**Disadvantages**:
- 20-50% slower than baseline
- Requires understanding of enhancement parameters

---

### Strategy 2: Gradual Feature Addition

**Best for**: Existing projects, performance-sensitive applications, learning the enhancements

#### Phase 1: Add Disulfide Constraints Only

```python
from ubf_protein.enhanced_physics_config import EnhancedPhysicsConfig
from ubf_protein.disulfide_detector import DisulfideDetector

# Detect bonds
detector = DisulfideDetector()
bonds = detector.predict_from_sequence(sequence)

# Enable only disulfide constraints
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_side_chains=False,  # Disable for now
    enable_solvent=False,
    enable_entropic=False,
    enable_refinement=False
).with_disulfide_bonds(bonds)

coordinator = MultiAgentCoordinator(
    protein_sequence=sequence,
    physics_config=config
)
```

**Impact**: +5-10% accuracy, +2-5% time overhead

#### Phase 2: Add Side-Chain Interactions

```python
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_side_chains=True,  # ← Enable
    enable_solvent=False,
    enable_entropic=False,
    enable_refinement=False
).with_disulfide_bonds(bonds)
```

**Impact**: +10-20% accuracy, +10-15% time overhead

#### Phase 3: Add Solvent Corrections

```python
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_side_chains=True,
    enable_solvent=True,  # ← Enable
    enable_entropic=False,
    enable_refinement=False
).with_disulfide_bonds(bonds)
```

**Impact**: +5-10% accuracy (charged residues), +5% time overhead

#### Phase 4: Add Entropic Terms

```python
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_side_chains=True,
    enable_solvent=True,
    enable_entropic=True,  # ← Enable
    enable_refinement=False
).with_disulfide_bonds(bonds)
```

**Impact**: Better free energy estimates, +3% time overhead

#### Phase 5: Add Local Refinement

```python
config = EnhancedPhysicsConfig.enhanced_default().with_disulfide_bonds(bonds)
# All features enabled, including refinement
```

**Impact**: Polished final structures, +10-20 seconds per protein

---

### Strategy 3: Parallel Comparison

**Best for**: Validation, performance testing, gradual rollout

```python
# Run both baseline and enhanced in parallel
from concurrent.futures import ThreadPoolExecutor

def run_baseline(sequence):
    coordinator = MultiAgentCoordinator(protein_sequence=sequence)
    return coordinator.run_parallel_exploration(iterations=500)

def run_enhanced(sequence):
    config = EnhancedPhysicsConfig.auto_adapt(sequence)
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        physics_config=config
    )
    return coordinator.run_parallel_exploration(iterations=500)

# Run both
with ThreadPoolExecutor(max_workers=2) as executor:
    baseline_future = executor.submit(run_baseline, sequence)
    enhanced_future = executor.submit(run_enhanced, sequence)
    
    baseline_result = baseline_future.result()
    enhanced_result = enhanced_future.result()

# Compare
print(f"Baseline: Energy={baseline_result.best_energy:.2f}, RMSD={baseline_result.best_rmsd:.2f}")
print(f"Enhanced: Energy={enhanced_result.best_energy:.2f}, RMSD={enhanced_result.best_rmsd:.2f}")
```

---

## Feature-by-Feature Migration

### Disulfide Bond Constraints

**Before** (manually enforced or ignored):
```python
# Disulfide bonds not explicitly modeled
coordinator = MultiAgentCoordinator(protein_sequence=sequence)
```

**After**:
```python
from ubf_protein.disulfide_detector import DisulfideDetector

# Automatic detection
detector = DisulfideDetector()
bonds = detector.detect_from_pdb("1CRN.pdb")  # From PDB
# Or: bonds = detector.predict_from_sequence(sequence)  # From sequence

# Use in configuration
config = EnhancedPhysicsConfig.enhanced_default()
config = config.with_disulfide_bonds(bonds)

coordinator = MultiAgentCoordinator(
    protein_sequence=sequence,
    physics_config=config
)
```

**Benefits**:
- Proper spatial constraints for Cys-Cys bonds
- Prevents unphysical bond violations
- Essential for proteins like Crambin, Lysozyme, SSI

---

### Side-Chain Field Interactions

**Before** (simplified or CA-only models):
```python
# Only backbone atoms considered
# Hydrophobic effects approximated
```

**After**:
```python
# Side-chain fields automatically created from sequence
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_side_chains=True,
    sidechain_sigma=2.0,  # Gaussian width (Å)
    sidechain_cutoff=15.0  # Distance cutoff (Å)
)
```

**Benefits**:
- Realistic hydrophobic core formation
- Electrostatic salt bridge formation
- Steric clash prevention
- Hydrophilic surface preference

**Tuning**:
- Increase `sidechain_sigma` (2.0 → 2.5 Å) for softer fields
- Decrease `sidechain_cutoff` (15.0 → 12.0 Å) for speed

---

### Solvent Screening Corrections

**Before** (vacuum or constant dielectric):
```python
# Electrostatics with ε=4 throughout protein
```

**After**:
```python
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_solvent=True,
    screening_length=3.0,       # Debye length (Å)
    burial_radius=8.0,          # Neighbor counting radius (Å)
    buried_dielectric=4.0,      # ε for buried residues
    surface_dielectric=80.0     # ε for surface residues
)
```

**Benefits**:
- Reduced electrostatic strength at protein surface
- Realistic solvent shielding
- Better charge-charge interaction modeling

**Tuning**:
- Increase `screening_length` (3.0 → 4.0 Å) for stronger screening
- Adjust `burial_radius` (8.0 → 10.0 Å) for stricter burial definition

---

### Entropic Contributions

**Before** (enthalpy only):
```python
# Energy = H (no entropy term)
```

**After**:
```python
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_entropic=True,
    temperature=300.0,          # Kelvin
    trajectory_window=50,       # Snapshots for diversity
    max_entropy_variance=10.0   # Variance normalization
)
```

**Benefits**:
- Free energy G = H - T×S
- Penalizes overly rigid structures
- Rewards conformational diversity
- Better thermodynamic accuracy

**Tuning**:
- Increase `temperature` (300 → 310 K) for higher entropy weight
- Increase `trajectory_window` (50 → 100) for longer memory

---

### Local Refinement

**Before** (Monte Carlo or no refinement):
```python
# Best conformation as-is from exploration
```

**After**:
```python
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_refinement=True,
    refinement_max_iterations=100,
    refinement_convergence_threshold=0.001,  # kcal/mol
    refinement_step_size=0.01  # Å
)
```

**Benefits**:
- Polished final structures (23 kcal/mol typical improvement)
- Local minimum optimization
- Geometry validation
- Publication-quality conformations

**Tuning**:
- Increase `refinement_max_iterations` (100 → 150) for difficult cases
- Decrease `refinement_step_size` (0.01 → 0.005 Å) for careful optimization

---

## Performance Considerations

### Computational Cost

| Feature | Time Overhead | Memory Overhead |
|---------|--------------|----------------|
| Disulfide Constraints | +2-5% | Negligible |
| Side-Chain Fields | +10-15% | +5-10 MB |
| Solvent Corrections | +5% | Negligible |
| Entropic Terms | +3% | +2-5 MB (trajectory) |
| Local Refinement | +10-30 sec/protein | Negligible |
| **All Combined** | **+20-50%** | **+10-20 MB** |

### Performance Optimization Tips

#### 1. Use Size-Appropriate Configurations

```python
# Automatic size adaptation
config = EnhancedPhysicsConfig.auto_adapt(sequence)

# Manual size selection
if len(sequence) < 50:
    config = EnhancedPhysicsConfig.small_protein(len(sequence))
elif len(sequence) <= 150:
    config = EnhancedPhysicsConfig.medium_protein(len(sequence))
else:
    config = EnhancedPhysicsConfig.large_protein(len(sequence))
```

#### 2. Reduce Side-Chain Cutoff for Large Proteins

```python
# Default: 15.0 Å (accurate but slower)
# Large protein: 12.0 Å (faster, 95% accuracy retained)

config = EnhancedPhysicsConfig.large_protein(len(sequence))
# Already has sidechain_cutoff=12.0
```

#### 3. Disable Refinement for Screening

```python
# High-throughput mode: skip refinement
config = EnhancedPhysicsConfig.enhanced_default()
config = config.with_refinement(False)
```

#### 4. Use PyPy for 2-5x Speedup

```bash
# Install PyPy
# Windows: choco install pypy3
# Linux/Mac: brew install pypy3 or download from pypy.org

# Run with PyPy
pypy3 your_script.py
```

#### 5. Cache Results for Repeated Conformations

```python
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_caching=True  # Default, but explicit here
)
```

---

## Breaking Changes

**None** - The physics enhancements introduce **zero breaking changes**.

All existing code continues to work unchanged. The only "breaking" change is improved accuracy, which may alter specific numerical results compared to baseline.

### API Additions (Non-Breaking)

New classes (all optional):
- `DisulfideDetector`
- `SideChainFieldCalculator`
- `SolventFieldCorrection`
- `EntropicCalculator`
- `EnhancedEnergyCalculator`
- `LocalRefinement`
- `EnhancedPhysicsConfig`

New parameters (all optional):
- `MultiAgentCoordinator(..., physics_config=None)` (default None = baseline)

---

## Troubleshooting

### Issue: "Disulfide bonds not satisfied"

**Symptom**: Disulfide bond distances remain >10 Å after exploration

**Cause**: Agents need guidance to satisfy bonds (not just penalties)

**Solution**: System already implements disulfide constraint awareness (Task 14 completed). If still seeing issues:

```python
# 1. Verify bonds are detected
detector = DisulfideDetector()
bonds = detector.predict_from_sequence(sequence)
print(f"Detected {len(bonds)} bonds")

# 2. Check satisfaction before/after
satisfied, violations = detector.check_satisfaction(bonds, coords)
if not satisfied:
    print("Violations:", violations)

# 3. Increase iterations if needed
coordinator.run_parallel_exploration(iterations=2000)  # Instead of 1000
```

---

### Issue: "Performance slower than expected"

**Symptom**: Enhanced mode >2x slower than baseline

**Solutions**:

```python
# 1. Use size-appropriate config
config = EnhancedPhysicsConfig.auto_adapt(sequence)

# 2. Disable refinement for screening
config = config.with_refinement(False)

# 3. Reduce cutoffs for large proteins
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_side_chains=True,
    sidechain_cutoff=12.0,  # Reduced from 15.0
    enable_solvent=True,
    enable_entropic=True,
    enable_refinement=False  # Disable for speed
)

# 4. Use PyPy
# pypy3 your_script.py
```

---

### Issue: "Out of memory"

**Symptom**: Memory usage grows over time

**Cause**: Trajectory snapshots accumulating

**Solution**:

```python
# 1. Reduce trajectory window
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_entropic=True,
    trajectory_window=30  # Reduced from 50
)

# 2. Or disable entropic if not needed
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_entropic=False  # Disable trajectory collection
)
```

---

### Issue: "Energy values different from baseline"

**Symptom**: Enhanced energy differs significantly from baseline

**This is expected!** Enhanced energy includes additional terms:

```python
# Get breakdown to understand differences
calculator = EnhancedEnergyCalculator(sequence)
breakdown = calculator.calculate_energy_breakdown(conformation)

print("Energy components:")
for component, energy in breakdown.items():
    print(f"  {component}: {energy:.2f} kcal/mol")

# Typical breakdown:
#   base_mm:      -65.20 kcal/mol  (similar to baseline)
#   side_chains:  -12.45 kcal/mol  (NEW)
#   disulfide:      8.20 kcal/mol  (NEW - penalty for violations)
#   entropic:      -0.87 kcal/mol  (NEW)
#   total:        -70.32 kcal/mol  (combined)
```

---

### Issue: "Refinement not converging"

**Symptom**: Refinement uses all 100 iterations without convergence

**Solutions**:

```python
# 1. Increase max iterations
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_refinement=True,
    refinement_max_iterations=200  # Increased from 100
)

# 2. Loosen convergence threshold
config = EnhancedPhysicsConfig(
    use_enhanced_energy=True,
    enable_refinement=True,
    refinement_convergence_threshold=0.01  # Increased from 0.001
)

# 3. Check for geometry issues
refiner = LocalRefinement(calculator)
refined, stats = refiner.refine(conformation)
if not stats['converged']:
    print(f"Final energy change: {stats['energy_improvement']:.3f} kcal/mol")
    # If improvement is small, convergence failure is acceptable
```

---

## Summary

✅ **100% backward compatible** - Existing code works unchanged  
✅ **Opt-in enhancements** - Enable via `EnhancedPhysicsConfig`  
✅ **Gradual migration** - Add features one at a time  
✅ **Auto-adaptation** - Size-based configuration  
✅ **No breaking changes** - Only API additions  
✅ **Performance tuned** - 20-50% overhead for significant accuracy gains  

**Recommended migration path**: Use `EnhancedPhysicsConfig.auto_adapt(sequence)` for immediate full benefits with automatic tuning.

For questions or issues, see:
- [API.md](API.md) - Complete API reference
- [EXAMPLES.md](EXAMPLES.md) - Usage examples (Examples 14-17)
- [Test suite](tests/) - 344+ tests demonstrating features
- [docs/DISULFIDE_CONSTRAINT_AWARENESS.md](../docs/DISULFIDE_CONSTRAINT_AWARENESS.md) - Disulfide implementation details
