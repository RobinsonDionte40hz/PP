# Disulfide Bond Modeling Implementation Plan

## Overview
Add disulfide bond constraints to improve folding of small proteins (Crambin, SSI, etc.)

## Current Gap
- System only models CA-CA distances
- No cysteine-cysteine disulfide bridges
- Small proteins with S-S bonds perform poorly

## Implementation Strategy

### Phase 1: Detection & Parsing

**File:** `ubf_protein/disulfide_detector.py`

```python
from typing import List, Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class DisulfideBond:
    """Represents a disulfide bond between two cysteines."""
    residue_i: int  # First cysteine residue index
    residue_j: int  # Second cysteine residue index
    distance: float = 5.5  # Target S-S distance (Å)

class DisulfideDetector:
    """Detect disulfide bonds from PDB structure or predict from sequence."""
    
    def detect_from_pdb(self, pdb_file: str) -> List[DisulfideBond]:
        """Parse SSBOND records from PDB file."""
        bonds = []
        with open(pdb_file) as f:
            for line in f:
                if line.startswith('SSBOND'):
                    # SSBOND record format:
                    # SSBOND   1 CYS A   6    CYS A  127
                    res1 = int(line[17:21].strip())
                    res2 = int(line[31:35].strip())
                    bonds.append(DisulfideBond(res1, res2))
        return bonds
    
    def predict_from_sequence(self, sequence: str) -> List[DisulfideBond]:
        """Predict likely disulfide bonds from cysteine positions."""
        cysteines = [i for i, aa in enumerate(sequence) if aa == 'C']
        
        if len(cysteines) < 2:
            return []
        
        # Simple heuristic: pair cysteines by proximity in sequence
        # More sophisticated: use ML model or pattern matching
        bonds = []
        for i in range(0, len(cysteines)-1, 2):
            bonds.append(DisulfideBond(cysteines[i], cysteines[i+1]))
        
        return bonds
```

### Phase 2: Constraint Integration

**File:** `ubf_protein/structural_validation.py` (modify existing)

Add disulfide bond validation:

```python
def validate_disulfide_bonds(
    conformation: Conformation,
    disulfide_bonds: List[DisulfideBond],
    tolerance: float = 1.0  # Allow ±1.0 Å
) -> bool:
    """Check if disulfide bonds are satisfied."""
    for bond in disulfide_bonds:
        ca_i = conformation.coordinates[bond.residue_i]
        ca_j = conformation.coordinates[bond.residue_j]
        distance = np.linalg.norm(ca_i - ca_j)
        
        # S-S bond should be ~5.5 Å at CA level (CB is 3.8 Å from CA)
        # So CA-CA distance ~3.8 Å when bonded
        if abs(distance - 3.8) > tolerance:
            return False
    
    return True
```

### Phase 3: Move Generation with S-S Constraints

**File:** `ubf_protein/mapless_moves.py` (modify)

Bias moves to satisfy disulfide constraints:

```python
class MaplessMoveGenerator:
    def __init__(self, disulfide_bonds: List[DisulfideBond] = None):
        self.disulfide_bonds = disulfide_bonds or []
    
    def generate_moves(self, conformation: Conformation) -> List[Move]:
        moves = self._generate_base_moves(conformation)
        
        # If disulfide bonds exist, add constraint-satisfying moves
        if self.disulfide_bonds:
            moves.extend(self._generate_disulfide_moves(conformation))
        
        return moves
    
    def _generate_disulfide_moves(self, conformation: Conformation) -> List[Move]:
        """Generate moves that pull cysteines closer together."""
        moves = []
        
        for bond in self.disulfide_bonds:
            ca_i = conformation.coordinates[bond.residue_i]
            ca_j = conformation.coordinates[bond.residue_j]
            current_dist = np.linalg.norm(ca_i - ca_j)
            target_dist = 3.8  # CA-CA distance for S-S bond
            
            if current_dist > target_dist + 1.0:
                # Generate move that pulls them closer
                direction = (ca_j - ca_i) / current_dist
                move_vector = direction * 0.5  # Small step
                
                moves.append(Move(
                    residue_index=bond.residue_i,
                    move_type=MoveType.CONSTRAINED,
                    parameters={'vector': move_vector, 'bond': bond}
                ))
        
        return moves
```

### Phase 4: Energy Function Modification

**File:** `ubf_protein/energy_calculator.py` (modify)

Add disulfide bond energy term:

```python
def calculate_energy_with_disulfides(
    conformation: Conformation,
    disulfide_bonds: List[DisulfideBond]
) -> float:
    """Calculate energy including disulfide bond penalty."""
    
    base_energy = self.calculate_base_energy(conformation)
    
    # Add disulfide bond constraint energy
    ss_energy = 0.0
    for bond in disulfide_bonds:
        ca_i = conformation.coordinates[bond.residue_i]
        ca_j = conformation.coordinates[bond.residue_j]
        distance = np.linalg.norm(ca_i - ca_j)
        
        # Harmonic potential: k * (r - r0)^2
        k = 50.0  # Spring constant (kcal/mol/Å²)
        r0 = 3.8  # Target distance
        ss_energy += k * (distance - r0)**2
    
    return base_energy + ss_energy
```

### Phase 5: Integration with Test Tool

**File:** `test_protein.py` (modify)

Auto-detect and use disulfide bonds:

```python
from ubf_protein.disulfide_detector import DisulfideDetector

def run_protein_test(sequence: str, pdb_file: Path = None, ...):
    # ... existing code ...
    
    # Detect disulfide bonds if PDB available
    disulfide_bonds = []
    if pdb_file:
        detector = DisulfideDetector()
        disulfide_bonds = detector.detect_from_pdb(str(pdb_file))
        if disulfide_bonds:
            print(f"✓ Detected {len(disulfide_bonds)} disulfide bonds")
    
    # Pass to coordinator
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        qcpp_integration=qcpp_adapter,
        disulfide_bonds=disulfide_bonds  # NEW
    )
```

---

## Expected Impact

### Crambin (1CRN) - 46 residues, 3 S-S bonds
- **Before:** Energy -199 kcal/mol, RMSD 10 Å
- **After (expected):** Energy -250 to -280 kcal/mol, RMSD 6-8 Å

### SSI Inhibitor (3SSI) - 113 residues, disulfide bonds
- **Before:** Energy -427 kcal/mol (good, but could improve)
- **After (expected):** Energy -450 to -480 kcal/mol, RMSD 2-3 Å

---

## Testing Plan

```python
# Test 1: Crambin with S-S bonds
python test_protein.py --pdb 1CRN --use-disulfides

# Test 2: SSI with S-S bonds
python test_protein.py --pdb 3SSI --use-disulfides

# Test 3: Batch test small proteins
python batch_test_proteins.py --ids 1CRN 2MR9 1VII --use-disulfides
```

---

## Implementation Timeline

- **Week 1:** DisulfideDetector + parsing (Phase 1)
- **Week 2:** Constraint validation (Phase 2)
- **Week 3:** Move generation with constraints (Phase 3)
- **Week 4:** Energy function modification (Phase 4)
- **Week 5:** Integration + testing (Phase 5)

---

## Files to Create/Modify

**New Files:**
- `ubf_protein/disulfide_detector.py` (150 lines)
- `ubf_protein/tests/test_disulfide_bonds.py` (200 lines)

**Modified Files:**
- `ubf_protein/structural_validation.py` (+50 lines)
- `ubf_protein/mapless_moves.py` (+80 lines)
- `ubf_protein/energy_calculator.py` (+40 lines)
- `ubf_protein/multi_agent_coordinator.py` (+20 lines)
- `test_protein.py` (+30 lines)

**Total:** ~570 lines of code

---

## Alternative: Side-Chain Modeling First?

Disulfide modeling requires side-chain positions (CB atoms at minimum). Consider implementing side-chain modeling first, then disulfide bonds become easier.

---

# Extended Implementation Plan: Side-Chain Fields & Physics Enhancements

## Phase 6: Side-Chain Modeling (Field-Based Approach)

### Concept: Side-Chain Fields Around CA Nodes
Instead of explicit all-atom modeling, represent side-chains as **scalar fields** centered at CA positions.

**File:** `ubf_protein/sidechain_fields.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SideChainField:
    """Scalar field representing side-chain properties around CA."""
    residue_type: str           # Amino acid type
    hydrophobicity: float       # -1 (hydrophilic) to +1 (hydrophobic)
    volume: float               # Å³ (effective side-chain volume)
    charge: float               # -1, 0, +1
    radius: float               # Effective field radius (Å)
    
    # Field decay function: strength(r) = max_strength * exp(-r²/2σ²)
    field_sigma: float = 2.0    # Field width (Å)

class SideChainFieldCalculator:
    """Calculate side-chain field contributions to energy."""
    
    # Amino acid properties
    AA_PROPERTIES = {
        'A': {'hydro': 0.62, 'vol': 88.6,  'charge': 0},   # Alanine
        'C': {'hydro': 0.29, 'vol': 108.5, 'charge': 0},   # Cysteine
        'D': {'hydro': -0.90, 'vol': 111.1, 'charge': -1}, # Aspartate
        'E': {'hydro': -0.74, 'vol': 138.4, 'charge': -1}, # Glutamate
        'F': {'hydro': 1.19, 'vol': 189.9, 'charge': 0},   # Phenylalanine
        'G': {'hydro': 0.48, 'vol': 60.1,  'charge': 0},   # Glycine
        'H': {'hydro': -0.40, 'vol': 153.2, 'charge': 0.1},# Histidine
        'I': {'hydro': 1.38, 'vol': 166.7, 'charge': 0},   # Isoleucine
        'K': {'hydro': -1.50, 'vol': 168.6, 'charge': +1}, # Lysine
        'L': {'hydro': 1.06, 'vol': 166.7, 'charge': 0},   # Leucine
        'M': {'hydro': 0.64, 'vol': 162.9, 'charge': 0},   # Methionine
        'N': {'hydro': -0.78, 'vol': 114.1, 'charge': 0},  # Asparagine
        'P': {'hydro': 0.12, 'vol': 112.7, 'charge': 0},   # Proline
        'Q': {'hydro': -0.85, 'vol': 143.8, 'charge': 0},  # Glutamine
        'R': {'hydro': -2.53, 'vol': 173.4, 'charge': +1}, # Arginine
        'S': {'hydro': -0.18, 'vol': 89.0,  'charge': 0},  # Serine
        'T': {'hydro': -0.05, 'vol': 116.1, 'charge': 0},  # Threonine
        'V': {'hydro': 1.08, 'vol': 140.0, 'charge': 0},   # Valine
        'W': {'hydro': 0.81, 'vol': 227.8, 'charge': 0},   # Tryptophan
        'Y': {'hydro': 0.26, 'vol': 193.6, 'charge': 0},   # Tyrosine
    }
    
    def create_field(self, residue_type: str) -> SideChainField:
        """Create field from amino acid type."""
        props = self.AA_PROPERTIES.get(residue_type, self.AA_PROPERTIES['A'])
        radius = (props['vol'] / (4/3 * np.pi))**(1/3)  # Effective radius
        
        return SideChainField(
            residue_type=residue_type,
            hydrophobicity=props['hydro'],
            volume=props['vol'],
            charge=props['charge'],
            radius=radius,
            field_sigma=2.0
        )
    
    def calculate_field_strength(self, field: SideChainField, distance: float) -> float:
        """Calculate field strength at distance r from CA."""
        # Gaussian decay
        return np.exp(-distance**2 / (2 * field.field_sigma**2))
    
    def calculate_field_interaction(
        self,
        field_i: SideChainField,
        field_j: SideChainField,
        ca_distance: float
    ) -> float:
        """Calculate interaction energy between two side-chain fields."""
        
        # 1. Steric repulsion (volume overlap)
        overlap_dist = field_i.radius + field_j.radius
        if ca_distance < overlap_dist:
            steric_energy = 10.0 * (overlap_dist - ca_distance)**2
        else:
            steric_energy = 0.0
        
        # 2. Hydrophobic attraction (like attracts like)
        hydro_i = field_i.hydrophobicity
        hydro_j = field_j.hydrophobicity
        
        if hydro_i * hydro_j > 0:  # Same sign
            # Attractive for hydrophobic pairs
            hydro_energy = -2.0 * abs(hydro_i * hydro_j) * self.calculate_field_strength(field_i, ca_distance)
        else:
            # Repulsive for hydrophobic-hydrophilic pairs
            hydro_energy = 1.0 * abs(hydro_i * hydro_j) * self.calculate_field_strength(field_i, ca_distance)
        
        # 3. Electrostatic interaction
        if field_i.charge != 0 and field_j.charge != 0:
            # Coulomb-like: k * q1*q2 / (εr)
            k = 332.0  # kcal*Å/mol (electrostatic constant)
            epsilon_eff = 20.0  # Effective dielectric (will be improved below)
            elec_energy = k * field_i.charge * field_j.charge / (epsilon_eff * ca_distance)
        else:
            elec_energy = 0.0
        
        return steric_energy + hydro_energy + elec_energy
```

---

## Phase 7: Solvent-Field Correction (Dielectric Response)

**File:** `ubf_protein/solvent_field.py`

```python
class SolventFieldCorrection:
    """Model solvent screening using distance-dependent dielectric."""
    
    def calculate_effective_dielectric(
        self,
        distance: float,
        burial_factor: float = 0.0  # 0 = exposed, 1 = buried
    ) -> float:
        """
        Distance-dependent dielectric with burial correction.
        
        ε(r, burial) = ε_buried + (ε_water - ε_buried) * (1 - burial) * (1 - exp(-r/λ))
        
        Where:
        - ε_water = 80 (bulk water)
        - ε_buried = 4 (protein interior)
        - λ = 3.0 Å (screening length)
        """
        epsilon_water = 80.0
        epsilon_buried = 4.0
        screening_length = 3.0
        
        # Sigmoidal transition from buried to water-screened
        screening_factor = 1.0 - np.exp(-distance / screening_length)
        
        epsilon_eff = (
            epsilon_buried +
            (epsilon_water - epsilon_buried) * (1.0 - burial_factor) * screening_factor
        )
        
        return epsilon_eff
    
    def calculate_burial_factor(
        self,
        conformation: Conformation,
        residue_index: int,
        neighbor_distance: float = 8.0
    ) -> float:
        """
        Calculate how buried a residue is (0 = surface, 1 = core).
        Based on number of neighbors within cutoff distance.
        """
        ca_pos = conformation.coordinates[residue_index]
        neighbor_count = 0
        
        for j, ca_j in enumerate(conformation.coordinates):
            if j == residue_index:
                continue
            distance = np.linalg.norm(ca_pos - ca_j)
            if distance < neighbor_distance:
                neighbor_count += 1
        
        # Normalize: ~12 neighbors = fully buried
        max_neighbors = 12
        burial = min(1.0, neighbor_count / max_neighbors)
        
        return burial
```

---

## Phase 8: Entropic Correction (Coherence Field Variance)

**File:** `ubf_protein/entropic_correction.py`

```python
class EntropicCorrection:
    """Calculate entropic contribution from coherence field variance."""
    
    def calculate_coherence_entropy(
        self,
        qcp_values: np.ndarray,
        temperature: float = 300.0  # Kelvin
    ) -> float:
        """
        S_coherence = -k_B * Σ p_i * ln(p_i)
        
        Where p_i derived from QCP field variance.
        High variance = high entropy (disordered)
        Low variance = low entropy (ordered)
        """
        k_B = 0.001987  # kcal/(mol·K) (Boltzmann constant)
        
        # Calculate field variance
        qcp_variance = np.var(qcp_values)
        
        # Map variance to effective entropy
        # High variance → high entropy (favorable at high T)
        # Low variance → low entropy (favorable at low T)
        
        # Normalize variance
        variance_normalized = min(1.0, qcp_variance / 10.0)
        
        # Entropy contribution (simplified)
        # S = k_B * ln(Ω), where Ω ~ exp(variance)
        entropy = k_B * variance_normalized * 10.0  # Effective multiplicity
        
        # Free energy contribution: -T*S
        entropic_energy = -temperature * entropy
        
        return entropic_energy
    
    def calculate_configurational_entropy(
        self,
        conformation: Conformation,
        previous_conformations: List[Conformation],
        window_size: int = 50
    ) -> float:
        """
        Estimate configurational entropy from structural diversity.
        More diverse conformations = higher entropy.
        """
        if len(previous_conformations) < 2:
            return 0.0
        
        # Sample recent conformations
        recent = previous_conformations[-window_size:]
        
        # Calculate RMSD matrix
        rmsd_values = []
        for conf in recent:
            rmsd = self._calculate_rmsd(conformation, conf)
            rmsd_values.append(rmsd)
        
        # High average RMSD = high diversity = high entropy
        avg_rmsd = np.mean(rmsd_values)
        
        # Map to entropy: S ~ k_B * ln(1 + RMSD)
        k_B = 0.001987
        entropy = k_B * np.log(1.0 + avg_rmsd)
        
        return -300.0 * entropy  # -T*S at 300K
```

---

## Phase 9: Integrated Energy Function

**File:** `ubf_protein/enhanced_energy_calculator.py`

```python
class EnhancedEnergyCalculator:
    """Complete energy function with all enhancements."""
    
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.sidechain_calc = SideChainFieldCalculator()
        self.solvent_calc = SolventFieldCorrection()
        self.entropy_calc = EntropicCorrection()
        
        # Create side-chain fields for all residues
        self.fields = [
            self.sidechain_calc.create_field(aa) 
            for aa in sequence
        ]
    
    def calculate_total_energy(
        self,
        conformation: Conformation,
        qcp_values: np.ndarray,
        previous_conformations: List[Conformation] = None
    ) -> float:
        """
        E_total = E_base + E_sidechain + E_disulfide + E_entropic
        """
        
        # 1. Base energy (existing)
        E_base = self.calculate_base_energy(conformation)
        
        # 2. Side-chain field interactions
        E_sidechain = self._calculate_sidechain_energy(conformation)
        
        # 3. Disulfide bonds (if applicable)
        E_disulfide = self._calculate_disulfide_energy(conformation)
        
        # 4. Entropic corrections
        E_entropic = 0.0
        if qcp_values is not None:
            E_entropic += self.entropy_calc.calculate_coherence_entropy(qcp_values)
        if previous_conformations:
            E_entropic += self.entropy_calc.calculate_configurational_entropy(
                conformation, previous_conformations
            )
        
        return E_base + E_sidechain + E_disulfide + E_entropic
    
    def _calculate_sidechain_energy(self, conformation: Conformation) -> float:
        """Calculate all side-chain field interactions with solvent correction."""
        total_energy = 0.0
        n_residues = len(self.sequence)
        
        for i in range(n_residues):
            # Calculate burial for solvent correction
            burial_i = self.solvent_calc.calculate_burial_factor(conformation, i)
            
            for j in range(i+1, n_residues):
                if abs(j - i) < 3:  # Skip near neighbors
                    continue
                
                ca_i = conformation.coordinates[i]
                ca_j = conformation.coordinates[j]
                distance = np.linalg.norm(ca_i - ca_j)
                
                if distance > 15.0:  # Cutoff
                    continue
                
                # Get effective dielectric
                burial_j = self.solvent_calc.calculate_burial_factor(conformation, j)
                avg_burial = (burial_i + burial_j) / 2.0
                epsilon_eff = self.solvent_calc.calculate_effective_dielectric(
                    distance, avg_burial
                )
                
                # Calculate field interaction with solvent screening
                field_energy = self.sidechain_calc.calculate_field_interaction(
                    self.fields[i], self.fields[j], distance
                )
                
                # Apply solvent screening to electrostatics
                # (already included in calculate_field_interaction with epsilon_eff)
                
                total_energy += field_energy
        
        return total_energy
```

---

## Phase 10: Local Refinement (Gradient Descent)

**File:** `ubf_protein/local_refinement.py`

```python
class LocalRefinement:
    """Perform local energy minimization on conformations."""
    
    def __init__(self, energy_calculator: EnhancedEnergyCalculator):
        self.energy_calc = energy_calculator
    
    def refine_conformation(
        self,
        conformation: Conformation,
        max_steps: int = 100,
        step_size: float = 0.01,
        tolerance: float = 0.001
    ) -> Conformation:
        """
        Gradient descent refinement.
        Adjusts CA positions to minimize energy.
        """
        current_conf = conformation
        current_energy = self.energy_calc.calculate_total_energy(current_conf, None)
        
        for step in range(max_steps):
            # Calculate gradient (numerical)
            gradient = self._calculate_gradient(current_conf)
            
            # Update coordinates
            new_coords = current_conf.coordinates - step_size * gradient
            new_conf = Conformation(
                sequence=current_conf.sequence,
                coordinates=new_coords,
                energy=0.0  # Will be recalculated
            )
            
            # Check bond lengths after update
            if not self._validate_geometry(new_conf):
                step_size *= 0.5  # Reduce step size
                continue
            
            # Calculate new energy
            new_energy = self.energy_calc.calculate_total_energy(new_conf, None)
            
            # Check convergence
            if abs(new_energy - current_energy) < tolerance:
                break
            
            if new_energy < current_energy:
                current_conf = new_conf
                current_energy = new_energy
            else:
                step_size *= 0.5  # Reduce step size
        
        return current_conf
    
    def _calculate_gradient(self, conformation: Conformation) -> np.ndarray:
        """Calculate energy gradient numerically."""
        gradient = np.zeros_like(conformation.coordinates)
        epsilon = 0.01  # Finite difference step
        
        for i in range(len(conformation.coordinates)):
            for dim in range(3):  # x, y, z
                # Forward difference
                coords_plus = conformation.coordinates.copy()
                coords_plus[i, dim] += epsilon
                conf_plus = Conformation(
                    sequence=conformation.sequence,
                    coordinates=coords_plus,
                    energy=0.0
                )
                energy_plus = self.energy_calc.calculate_total_energy(conf_plus, None)
                
                # Backward difference
                coords_minus = conformation.coordinates.copy()
                coords_minus[i, dim] -= epsilon
                conf_minus = Conformation(
                    sequence=conformation.sequence,
                    coordinates=coords_minus,
                    energy=0.0
                )
                energy_minus = self.energy_calc.calculate_total_energy(conf_minus, None)
                
                # Central difference
                gradient[i, dim] = (energy_plus - energy_minus) / (2 * epsilon)
        
        return gradient
```

---

## Expected Impact with All Enhancements

### Crambin (1CRN) - 46 residues, 3 S-S bonds
- **Current:** Energy -199 kcal/mol, RMSD 10 Å ⚠️
- **With Disulfides:** Energy -250 to -280 kcal/mol, RMSD 7-8 Å
- **+ Side-chains:** Energy -280 to -320 kcal/mol, RMSD 6-7 Å
- **+ Solvent:** Energy -300 to -340 kcal/mol, RMSD 5-6 Å
- **+ Entropy + Refinement:** Energy -320 to -360 kcal/mol, RMSD 4-5 Å ✅

### 1PRN (Proteinase A) - 290 residues
- **Current:** Energy +955 kcal/mol ⚠️ (FAILED)
- **With Enhancements:** Energy -600 to -800 kcal/mol, RMSD 4-5 Å ✅

---

## Implementation Priority

1. **Side-Chain Fields** (Phase 6) - Foundation for everything else
2. **Solvent Correction** (Phase 7) - Major physics improvement
3. **Disulfide Bonds** (Phases 1-5) - Critical for small proteins
4. **Entropic Correction** (Phase 8) - Improves sampling
5. **Local Refinement** (Phase 10) - Final polish

**Estimated Total:** ~2,000 lines of code, 8-10 weeks implementation

---

## Testing Strategy

```bash
# Progressive testing
python test_protein.py --pdb 1CRN --use-sidechains
python test_protein.py --pdb 1CRN --use-sidechains --use-solvent
python test_protein.py --pdb 1CRN --use-sidechains --use-solvent --use-disulfides
python test_protein.py --pdb 1CRN --use-sidechains --use-solvent --use-disulfides --refine
```
