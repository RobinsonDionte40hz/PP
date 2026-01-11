# Secondary Structure Detection Algorithm

## Overview

This document describes the Chou-Fasman-based secondary structure detection algorithm implemented in `ubf_protein/persistent_channel_memory.py`. The algorithm predicts helix and sheet regions from amino acid sequence alone, providing structural guidance to the folding agents.

## Problem Statement

The original issue: **"we can't get our agents out of conformational space or impedance space"**

The agents knew WHERE they were in energy space (via Memory/Mediator systems) but didn't know WHICH ANGLES to target. The detection algorithm provides a **structural blueprint** - target phi/psi angles for each residue based on predicted secondary structure.

## Algorithm: Classic Chou-Fasman with Enhancements

### Propensity Values

Each amino acid has propensity values from the Chou-Fasman literature:

| Property | Description | Threshold |
|----------|-------------|-----------|
| P_alpha (helix_prop) | Helix propensity | ≥1.03 = former |
| P_beta (sheet_prop) | Sheet propensity | ≥1.05 = former |
| P_turn (turn_prop) | Turn propensity | ≥1.2 = turn-prone |

**Key dual-formers** (high in both helix and sheet):
- Leucine (L): P_alpha=1.21, P_beta=1.30
- Phenylalanine (F): P_alpha=1.13, P_beta=1.38
- Valine (V): P_alpha=1.06, P_beta=1.70

### Helix Detection (`_detect_helix_regions`)

```
Nucleation Rule: 4+ out of 6 consecutive residues with P_alpha ≥ 1.03
Breaker: Proline (P) anywhere in window
Extension: Continue while P_alpha ≥ 1.0
Minimum Length: 4 residues
```

**Process:**
1. Scan sequence for nucleation sites (4/6 formers, no proline)
2. Extend each nucleation site left/right while favorable
3. Merge overlapping regions
4. Keep only regions with avg P_alpha ≥ 1.0 and length ≥ 4

### Sheet Detection (`_detect_sheet_regions`)

```
Nucleation Rule: 3+ out of 5 consecutive residues with P_beta ≥ 1.05
Alternative: Alternating hydrophobic pattern (i, i+2, i+4) with VILFY residues
Extension: Continue while P_beta ≥ 1.0
Minimum Length: 3 residues
```

**Process:**
1. Scan for nucleation sites (3/5 formers)
2. Detect alternating hydrophobic patterns (strong sheet indicator)
3. Extend and merge regions
4. Keep only regions with avg P_beta ≥ 1.0 and length ≥ 3

### Global Bias (`_calculate_global_bias`)

Analyzes overall sequence composition to determine if protein is likely helix-rich or sheet-rich:

```python
helix_score = (EALM fraction) * 2 - (proline fraction) * 3
sheet_score = (VIY fraction) * 2 + (alternating pattern fraction) * 5

Returns: 'helix', 'sheet', or 'neutral'
```

**Factors:**
- Strong helix formers: E, A, L, M
- Strong sheet formers: V, I, Y
- Proline: helix breaker (negative for helix)
- Alternating hydrophobic: strong sheet indicator

### Conflict Resolution (`_resolve_conflicts`)

When helix and sheet regions overlap:

1. Count helix formers (P_alpha ≥ 1.03) in overlap
2. Count sheet formers (P_beta ≥ 1.05) in overlap
3. Winner takes the overlapping residues:
   - More helix formers → helix wins
   - More sheet formers → sheet wins
   - Tie → use global bias (sheet if bias='sheet', else helix)

4. Rebuild regions from resolved assignments
5. Enforce minimum lengths (helix: 4, sheet: 3)

## Target Angles

Detected regions are assigned canonical backbone angles:

| Structure | Channel | φ (phi) | ψ (psi) |
|-----------|---------|---------|---------|
| Helix | 10 Hz | -60° | -45° |
| Sheet | 7 Hz | -120° | +120° |
| Turn | 12 Hz | -60° | +30° |
| Coil | 12 Hz | -60° | +30° |

## Test Results

### Validation Proteins

| Protein | Expected | Predicted | Global Bias |
|---------|----------|-----------|-------------|
| Villin (1VII) | H~100% | H=72%, E=0% | helix ✓ |
| SSI Inhibitor (3SSI) | E~45% | H=6%, E=52% | sheet ✓ |
| Ubiquitin (1UBQ) | H~20%, E~40% | H=34%, E=43% | sheet |
| WW Domain | E~60% | H=35%, E=18% | sheet |
| GB1 | mixed | H=38%, E=36% | helix |

### Accuracy Analysis

**Strengths:**
- Correctly identifies helix-dominated proteins (Villin)
- Correctly identifies sheet-dominated proteins (3SSI)
- Global bias improves tie-breaking for ambiguous residues

**Limitations:**
- Dual-formers (L, F, V) are inherently ambiguous
- Cannot capture tertiary contact effects (e.g., WW domain tryptophan sandwich)
- ~55-65% accuracy typical for Chou-Fasman (matches literature)

## Integration with Folding System

### Blueprint Generation Flow

```
Sequence → BlueprintGenerator.generate_blueprint()
         ↓
    1. Calculate global_bias
    2. Assign initial residue targets
    3. Detect helix regions
    4. Detect sheet regions  
    5. Resolve conflicts
    6. Refine targets with detected regions
         ↓
    StructuralBlueprint (persistent memory)
```

### Agent Integration

The `PersistentChannelMemory` class provides:

```python
# Get target angles for a residue
phi, psi = blueprint.get_target_angles(residue_idx)

# Get region type
region = blueprint.get_region_type(residue_idx)  # 'helix', 'sheet', or 'coil'

# Get move bias toward target
bias = memory.get_move_bias(residue_idx, current_phi, current_psi)
```

## Files Modified

### `ubf_protein/persistent_channel_memory.py`

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `AMINO_ACID_PROPERTIES` | Chou-Fasman propensities for all 21 amino acids |
| `StructuralBlueprint` | Dataclass holding targets, regions, and global_bias |
| `BlueprintGenerator` | Main class for sequence → structure prediction |
| `_detect_helix_regions()` | Classic Chou-Fasman helix nucleation/extension |
| `_detect_sheet_regions()` | Sheet detection with alternating pattern bonus |
| `_calculate_global_bias()` | Sequence composition analysis |
| `_resolve_conflicts()` | Former-count based conflict resolution |

## Usage Example

```python
from ubf_protein.persistent_channel_memory import BlueprintGenerator

# Generate blueprint from sequence
gen = BlueprintGenerator()
blueprint = gen.generate_blueprint("MLSDEDFKAVFGMTRSAFANLPLWKQQHLKKEKGLF")

# Access predictions
print(f"Helix regions: {blueprint.helix_regions}")
print(f"Sheet regions: {blueprint.sheet_regions}")
print(f"Global bias: {blueprint.global_bias}")

# Get target for specific residue
target = blueprint.get_target(10)
print(f"Residue 10: target_ss={target.target_ss}, phi={target.target_phi}, psi={target.target_psi}")
```

## Future Improvements

1. **Position-specific scoring**: Weight N/C-terminal residues differently
2. **Helix capping**: Detect N-cap and C-cap motifs
3. **Beta-sheet pairing**: Predict strand-strand interactions
4. **Machine learning hybrid**: Use CF as features for a simple classifier
5. **Homology hints**: If similar sequence known, use as bias

## References

- Chou, P.Y. & Fasman, G.D. (1974). Prediction of protein conformation. Biochemistry, 13(2), 222-245.
- Chou, P.Y. & Fasman, G.D. (1978). Empirical predictions of protein conformation. Ann. Rev. Biochem., 47, 251-276.

## Version History

| Date | Changes |
|------|---------|
| 2026-01-11 | Initial Chou-Fasman implementation with global bias and conflict resolution |
