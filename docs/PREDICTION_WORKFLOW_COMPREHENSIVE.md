# EmergentFolds Prediction Workflow - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Complete Prediction Workflow](#complete-prediction-workflow)
4. [The Multi-Agent Parameter Map](#the-multi-agent-parameter-map)
5. [Energy Function Details](#energy-function-details)
6. [QCPP Integration](#qcpp-integration)
7. [Screening Workflows](#screening-workflows)
8. [Hierarchical Folding](#hierarchical-folding)
9. [Quantum Refinement Engine](#quantum-refinement-engine)
10. [Checkpointing and State Management](#checkpointing-and-state-management)
11. [Validation and Quality Metrics](#validation-and-quality-metrics)
12. [Configuration Reference](#configuration-reference)

---

## Overview

EmergentFolds uses a **multi-agent physics-based simulation** approach to predict protein structures from amino acid sequences. Unlike deep learning methods, this system explores conformational space using autonomous agents guided by:

- **Molecular mechanics energy functions** (AMBER-like force field)
- **Quantum coherence principles** (QCPP integration)
- **Collective learning** through shared memory pools
- **Adaptive exploration strategies** based on 2D parameter space

### Key Innovation: Conformational Space Search

The system achieves O(1) move generation without spatial pathfinding by using **capability-based filtering**. Each conformation declares what moves are feasible, eliminating the need for N² distance calculations during move selection. This enables efficient searching of the vast conformational space proteins can occupy.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PredictionRunner (Entry Point)                       │
│                    Single Source of Truth for All Predictions                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐    ┌──────────────────────┐    ┌────────────────┐  │
│  │  MultiAgentCoordinator │◄──►│    SharedMemoryPool    │◄──►│  Checkpoints   │  │
│  │   (Orchestration)       │    │   (Collective Learning)│    │  (Persistence) │  │
│  └───────────┬─────────┘    └──────────────────────┘    └────────────────┘  │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Protein Agents (Parallel Exploration)              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │    │
│  │  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent N  │  │  ...   │  │    │
│  │  │ Cautious │  │ Balanced │  │Aggressive│  │ Diverse  │  │        │  │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┘  │    │
│  └───────┼─────────────┼─────────────┼─────────────┼────────────────────┘    │
│          │             │             │             │                         │
│          ▼             ▼             ▼             ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Core Calculation Systems                        │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │    │
│  │  │  MolecularMech   │  │ QCPPIntegration │  │  MaplessMoveGen      │ │    │
│  │  │  EnergyFunction  │  │   (Physics)      │  │  (O(1) Moves)        │ │    │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Optional Enhancements                         │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │    │
│  │  │ Hierarchical    │  │    Quantum      │  │  Geometric Attractor │ │    │
│  │  │   Folding       │  │   Refinement    │  │     Analysis         │ │    │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Prediction Workflow

### Step-by-Step Process

A prediction flows through these stages:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: INITIALIZATION                                                  │
│  ════════════════════════════════════════════════════════════════════════│
│                                                                          │
│  1.1 Create PredictionConfig                                             │
│      • Parse sequence, validate amino acids                              │
│      • Auto-configure agents/iterations based on sequence length         │
│      • Set QCPP configuration preset                                     │
│                                                                          │
│  1.2 Initialize QCPP Predictor                                           │
│      • Load QuantumCoherenceProteinPredictor                             │
│      • Create QCPPIntegrationAdapter with caching                        │
│      • Geometric scorer if targeting Platonic solid                      │
│                                                                          │
│  1.3 Load Native Structure (if available)                                │
│      • Fetch from RCSB PDB or load local file                            │
│      • Extract Cα coordinates for RMSD validation                        │
│      • Initialize RMSDCalculator with Kabsch alignment                   │
│                                                                          │
│  1.4 Create MultiAgentCoordinator                                        │
│      • Configure checkpointing, hierarchical folding                     │
│      • Initialize SharedMemoryPool for collective learning               │
│      • Optional: Initialize Mediator Agents                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: AGENT INITIALIZATION                                           │
│  ════════════════════════════════════════════════════════════════════════│
│                                                                          │
│  2.1 Agent Population Creation                                           │
│      • Diversity: distributed across cautious, balanced, aggressive      │
│      • Each agent gets unique exploration parameters                    │
│        - Aggressiveness (frequency): 3-15 (dimensionless)                │
│        - Consistency (coherence): 0.2-1.0 (dimensionless)                │
│                                                                          │
│  2.2 Per-Agent Initialization                                            │
│      • ConsciousnessState with exploration parameters                    │
│      • BehavioralState (5D search strategy derived from 2D params)       │
│      • MemorySystem for experience storage                               │
│      • LocalMinimaDetector for stuck detection                           │
│      • StructuralValidation for geometry checking                        │
│      • Generate initial random conformation (extended chain)             │
│                                                                          │
│  2.3 Initial Conformation                                                │
│      • Linear chain with ~3.8Å CA-CA spacing                             │
│      • Random phi/psi angles within allowed regions                      │
│      • Energy calculated for starting point                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: PARALLEL EXPLORATION (Main Loop)                               │
│  ════════════════════════════════════════════════════════════════════════│
│                                                                          │
│  FOR each iteration (1 to max_iterations):                               │
│                                                                          │
│    3.1 Parallel Agent Execution                                          │
│        • Each agent executes explore_step() concurrently                 │
│        • Thread pool executor manages parallel execution                 │
│                                                                          │
│    3.2 Move Generation (O(1) Conformational Space Search)               │
│        • Generate feasible moves based on capabilities                   │
│        • Move types: BACKBONE_ROTATION, HELIX_FORMATION,                 │
│          SHEET_FORMATION, HYDROPHOBIC_COLLAPSE, PIVOT_ROTATION, etc.     │
│                                                                          │
│    3.3 Move Evaluation (5-Factor Composite)                              │
│        ┌─────────────────────────────────────────────────────────┐       │
│        │  Factor 1: Physical Score (Energy Change)               │       │
│        │  Factor 2: Quantum Alignment (QCPP Metrics)             │       │
│        │  Factor 3: Behavioral Fit (Match to search strategy)    │       │
│        │  Factor 4: Historical Success (Memory influence)        │       │
│        │  Factor 5: Goal Proximity (Native-like features)        │       │
│        └─────────────────────────────────────────────────────────┘       │
│                                                                          │
│    3.4 Move Execution                                                    │
│        • Apply selected move to conformation                             │
│        • Update atom coordinates, phi/psi angles                         │
│        • Calculate new energy with MolecularMechanicsEnergy              │
│                                                                          │
│    3.5 QCPP Analysis (every N iterations)                                │
│        • Calculate QCP score, field coherence, stability                 │
│        • Cache results for performance (40-85% hit rate)                 │
│        • Update agent parameters based on QCPP feedback                  │
│                                                                          │
│    3.6 Memory Storage                                                    │
│        • Store significant outcomes (energy change > threshold)          │
│        • Calculate significance based on energy, RMSD, QCPP metrics      │
│        • Prune old/low-significance memories to maintain performance     │
│                                                                          │
│    3.7 Collective Learning                                               │
│        • Exchange discoveries via SharedMemoryPool                       │
│        • Broadcast breakthroughs (large energy decreases)                │
│        • Other agents can learn from shared experiences                  │
│                                                                          │
│    3.8 Local Minima Detection & Escape                                   │
│        • Detect stuck state (energy plateau)                             │
│        • Apply escape strategies: temperature boost, random jump         │
│        • Update exploration parameters to increase exploration           │
│                                                                          │
│    3.9 Parameter Update                                                  │
│        • Update aggressiveness/consistency based on outcome              │
│        • Regenerate behavioral state if change > threshold               │
│                                                                          │
│    3.10 Checkpoint (if enabled, every N iterations)                      │
│        • Save full state: agents, memories, best conformation            │
│        • Rotate old checkpoints to maintain disk space                   │
│                                                                          │
│    3.11 Progress Callback                                                │
│        • Emit progress update for UI/monitoring                          │
│        • Include: iteration, energy, RMSD, conformations explored        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: QUANTUM REFINEMENT (Optional)                                  │
│  ════════════════════════════════════════════════════════════════════════│
│                                                                          │
│  If enable_refinement=True and native structure available:               │
│                                                                          │
│    4.1 Quantum Core Identification                                       │
│        • Identify most stable residues via QCP analysis                  │
│        • Build THz resonance cascade model                               │
│                                                                          │
│    4.2 Secondary Structure Registration                                  │
│        • Align helices/sheets to golden ratio patterns                   │
│        • Apply φ (1.618...) geometric constraints                        │
│                                                                          │
│    4.3 Hydrophobic Core Packing                                          │
│        • Optimize hydrophobic residue clustering                         │
│        • Apply water shielding principles                                │
│                                                                          │
│    4.4 Loop Refinement                                                   │
│        • Refine flexible loop regions                                    │
│        • Apply distance restraints from contact prediction               │
│                                                                          │
│    4.5 Final Optimization                                                │
│        • Two-stage: global fold → local refinement                       │
│        • Target: <5Å RMSD from native                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: ANALYSIS & RESULTS                                             │
│  ════════════════════════════════════════════════════════════════════════│
│                                                                          │
│    5.1 Best Conformation Selection                                       │
│        • Select lowest energy conformation from all agents               │
│        • May differ from lowest RMSD if no native available              │
│                                                                          │
│    5.2 RMSD Calculation (if native available)                            │
│        • Kabsch alignment for optimal superposition                      │
│        • Calculate: RMSD, GDT-TS, TM-score                               │
│                                                                          │
│    5.3 Geometric Attractor Analysis                                      │
│        • Golden ratio pattern detection                                  │
│        • Platonic solid similarity scoring                               │
│        • Symmetry metrics (rotational, local, asphericity)               │
│                                                                          │
│    5.4 QCPP Statistics                                                   │
│        • Total analyses, cache hit rate                                  │
│        • Average calculation time                                        │
│        • Performance recommendations                                     │
│                                                                          │
│    5.5 Build PredictionResults                                           │
│        • All metrics, coordinates, configuration                         │
│        • Quality assessment: excellent/good/acceptable/poor              │
│        • Export: PDB file, JSON results, trajectory                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The Multi-Agent Parameter Map

### The 2D Exploration Parameter Space

The core innovation is a **2D parameter space** that controls agent behavior:

```
                        EXPLORATION PARAMETER SPACE
    
    High (1.0)  ┌─────────────────────────────────────────────────┐
                │                                                  │
                │    CAUTIOUS AGENTS              CONSERVATIVE    │
                │    • High precision              • Stable        │
    Consistency │    • Local refinement            • Risk-averse   │
    (Coherence) │    • Detailed exploration        • Native-like   │
                │                                                  │
    Medium      │────────────────────────────────────────────────│
    (0.6)       │                                                  │
                │            BALANCED AGENTS                       │
                │    • Moderate exploration                        │
                │    • Adaptive strategy                           │
                │    • General-purpose                             │
                │                                                  │
    Low (0.2)   │────────────────────────────────────────────────│
                │                                                  │
                │    EXPLORATORY                   AGGRESSIVE     │
                │    • Random jumps                • High risk     │
                │    • Escape minima               • Fast search   │
                │    • Diverse sampling            • Bold moves    │
                │                                                  │
                └─────────────────────────────────────────────────┘
                Low (3)          Medium (9)           High (15)
                            Aggressiveness (Frequency)
```

### 2D to 5D Transformation

The 2D parameters are transformed into a 5D search strategy using proprietary transformation functions:

```python
# Input: 2D Parameters
aggressiveness = f(...)  # Normalized exploration aggressiveness
consistency = f(...)     # Normalized behavioral consistency

# Output: 5D Search Strategy (Proprietary Transformation)
exploration_energy    = transform_energy(aggressiveness)      # How much to explore
structural_focus      = transform_focus(consistency)          # Precision of moves
hydrophobic_drive     = transform_hydrophobic(...)            # Core formation tendency
risk_tolerance        = transform_risk(...)                   # Willingness for bold moves
native_state_ambition = transform_ambition(...)               # Goal-directedness

# Note: Specific transformation functions are proprietary
```

### Agent Diversity Distribution

Agents are distributed across behavioral profiles using a proprietary ratio optimized through extensive testing:

```
                 AGENT POPULATION DIVERSITY
    
    ┌─────────────────────────────────────────────────┐
    │                                                  │
    │   CAUTIOUS              BALANCED                │
    │   ┌─────────┐          ┌─────────┐              │
    │   │ High    │          │ Medium  │              │
    │   │ Coherence│          │ Both    │              │
    │   │ Low-Med │          │ Params  │              │
    │   │ Frequency│          │         │              │
    │   └─────────┘          └─────────┘              │
    │                                                  │
    │   AGGRESSIVE                                     │
    │   ┌─────────┐                                    │
    │   │ High    │                                    │
    │   │ Frequency│                                    │
    │   │ Varied  │                                    │
    │   │ Coherence│                                    │
    │   └─────────┘                                    │
    │                                                  │
    │   Distribution ratios are proprietary            │
    └─────────────────────────────────────────────────┘
```

---

## Energy Function Details

### The 6-Term AMBER-Like Force Field

```
E_total = E_bond + E_angle + E_dihedral + E_vdw + E_electrostatic + E_hbond + E_compact
```

#### 1. Bond Stretching Energy

```
E_bond = Σ k_b(r - r₀)²

Parameters:
  k_b = 10.0 kcal/mol/Å²  (force constant)
  r₀ = 3.8 Å              (CA-CA equilibrium distance)

Expected: ~-5 kcal/mol per bond when near equilibrium
```

#### 2. Angle Bending Energy

```
E_angle = Σ k_θ(θ - θ₀)²

Parameters:
  k_θ = 5.0 kcal/mol/rad²  (force constant)
  θ₀ = 1.91 rad (~110°)    (CA-CA-CA equilibrium angle)

Expected: ~-3 kcal/mol per angle when optimal
```

#### 3. Dihedral (Torsional) Energy

```
E_dihedral = Σ V_n/2 [1 + cos(nφ - γ)]

Parameters:
  V_n = 0.5 kcal/mol  (barrier height)
  n = 3               (periodicity)
  γ = 0               (phase)

Expected: ~-0.3 kcal/mol per dihedral in favorable conformation
```

#### 4. Van der Waals Energy (Lennard-Jones 12-6)

```
E_vdw = Σ ε[(r_min/r)¹² - 2(r_min/r)⁶]

Parameters:
  ε = 0.2 kcal/mol   (well depth)
  r_min = 3.8 Å       (equilibrium distance)
  cutoff = 12.0 Å

Expected: Favorable when atoms at optimal distance
```

#### 5. Electrostatic Energy (Coulomb)

```
E_elec = Σ (q_i × q_j) / (4πε₀εᵣr_ij)

Parameters:
  Coulomb constant = 332.0637 kcal·Å/(mol·e²)
  ε_r = distance-dependent (protein interior ~4)
  charges: alternating ±0.2e (simplified backbone)
  cutoff = 12.0 Å
```

#### 6. Hydrogen Bond Energy (10-12 Potential)

```
E_hbond = Σ C/r¹² - D/r¹⁰

Parameters:
  C, D = optimized coefficients (proprietary)
  cutoff = standard H-bond distance

Requirements:
  - Helix: both residues 'H' and |i-j| ≤ 4
  - Sheet: both residues 'E'
  - Sweet spot: typical H-bond geometry
```

#### 7. Compactness Bonus (Hydrophobic Collapse)

```
E_compact = f(Rg/Rg_ideal)

Where:
  Rg = radius of gyration
  Rg_ideal = scales with protein size

If Rg > Rg_ideal (extended):
  E = penalty function (proprietary scaling)
  
If Rg < Rg_ideal (compact):
  E = bonus function (proprietary scaling)

Critical for driving protein folding!
```

### Expected Energy Ranges

| Structure Type | Energy Range | Notes |
|---------------|--------------|-------|
| Extended chain | +500 to +2000 | Unfavorable (compactness penalty) |
| Partially folded | 0 to +100 | Transition state |
| Well-folded | -50 to -200 | Favorable (stable) |
| Native-like | -100 to -300 | Optimal |

---

## QCPP Integration

### What is QCPP?

**Quantum Coherence Protein Predictor** provides physics-based stability analysis using:

- **QCP (Quantum Consciousness Potential)** scores
- **Field coherence** metrics
- **THz resonance** analysis
- **Golden ratio (φ)** pattern matching

### The QCP Formula

QCP (Quantum Consciousness Potential) scores are calculated using a proprietary formula that integrates:

- **Structural hierarchy**: Different weights for coil, helix, sheet, and specialized structures
- **Local environment**: Neighbor relationships and packing density
- **Residue properties**: Hydrophobicity and chemical characteristics  
- **Golden ratio scaling**: φ-based harmonic patterns

The formula produces scores typically in the range of 3-12, where higher values indicate greater structural stability and quantum coherence alignment.

```
QCP = f(structure_level, neighbors, residue_properties, φ)

# Specific formula and coefficients are proprietary
# Higher QCP → more stable, better folded regions
```

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      QCPP Integration Flow                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────┐        ┌──────────────────┐      ┌────────────────┐  │
│  │  Conformation │───────►│  QCPPIntegration │─────►│   QCPPMetrics  │  │
│  │  (Coordinates)│        │     Adapter      │      │ (Cached Result)│  │
│  └───────────────┘        └────────┬─────────┘      └────────────────┘  │
│                                    │                                     │
│                                    ▼                                     │
│                           ┌────────────────────┐                        │
│                           │   Hash-Based Cache │                        │
│                           │  (40-85% hit rate) │                        │
│                           └────────────────────┘                        │
│                                    │                                     │
│                                    ▼                                     │
│                           ┌────────────────────┐                        │
│                           │  QCPP Calculations │                        │
│                           │  • QCP Score       │                        │
│                           │  • Field Coherence │                        │
│                           │  • Stability Score │                        │
│                           │  • Phi Match Score │                        │
│                           │  • Geometric Sim   │                        │
│                           └────────────────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### QCPPMetrics Data Structure

```python
@dataclass(frozen=True)
class QCPPMetrics:
    qcp_score: float           # 0-20 (typically 3-8)
    field_coherence: float     # -1 to 1 (normalized)
    stability_score: float     # ≥0 (higher = more stable)
    phi_match_score: float     # 0-1 (golden ratio matching)
    calculation_time_ms: float # Performance tracking
    geometric_similarity: float # 0-1 (Platonic solid match)
```

### How QCPP Guides Exploration

1. **Move Evaluation**: QCPP metrics influence the "quantum alignment" factor in 5-factor evaluation
2. **Parameter Adjustment**: High stability → increase consistency; low stability → increase aggressiveness
3. **Memory Significance**: QCPP metrics contribute to memory significance scoring
4. **Phi Pattern Rewards**: Golden ratio geometries receive energy bonuses

---

## Screening Workflows

### Aggregation Risk Screening

Unlike structure prediction (finding THE native fold), screening answers:
> "Will this sequence fold stably, or aggregate?"

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGGREGATION SCREENING WORKFLOW                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Protein Sequence                                                 │
│                                                                          │
│  STEP 1: Pre-Screen (FAST - no simulation)                              │
│  ─────────────────────────────────────────                              │
│    • Hydrophobic ratio (>50% = risk)                                    │
│    • Charged residue ratio (<10% = risk)                                │
│    • Aggregation-prone patterns (VVVV, IIII, FFFF)                      │
│    • Hydrophobic stretches (>6 consecutive = risk)                      │
│                                                                          │
│  STEP 2: Quick Folding Simulation (MODERATE)                            │
│  ──────────────────────────────────────────                             │
│    • 50-200 iterations (vs 500-2000 for full prediction)                │
│    • 2-5 agents (vs 10-50 for full prediction)                          │
│    • Track: energy, secondary structure, compactness                    │
│                                                                          │
│  STEP 3: Risk Assessment                                                 │
│  ──────────────────────                                                 │
│    Metrics (all 0-1, higher = better):                                  │
│    ┌────────────────────────────────────────────────────────────┐       │
│    │  energy_score      = f(final_energy)                       │       │
│    │  structure_score   = secondary_structure_pct / 100         │       │
│    │  hydrophobic_score = f(hydrophobic_clustering)             │       │
│    │  convergence_score = f(iterations_to_stable)               │       │
│    │  compactness_score = f(radius_of_gyration)                 │       │
│    │                                                             │       │
│    │  aggregation_score = weighted_average(above)               │       │
│    └────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  OUTPUT: Risk Classification                                             │
│  ─────────────────────────                                              │
│    • LOW: Likely to fold stably (aggregation_score > 0.7)               │
│    • MODERATE: Some concerns (0.4 < score < 0.7)                        │
│    • HIGH: Likely to aggregate (0.2 < score < 0.4)                      │
│    • CRITICAL: Almost certainly will aggregate (score < 0.2)            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Screening Configuration Presets

```python
# Ultra-fast (sacrifice accuracy for speed)
ScreeningConfig.fast()      # 50 iterations, 2 agents

# Balanced (recommended)
ScreeningConfig.balanced()  # 100 iterations, 3 agents

# Thorough (higher accuracy)
ScreeningConfig.thorough()  # 200 iterations, 5 agents, QCPP enabled
```

### Batch Screening Example

```python
from ubf_protein.aggregation_screening import AggregationScreener

screener = AggregationScreener()

sequences = [
    "ACDEFGHIKLMNPQRSTVWY",  # Diverse
    "VVVVVVVVVVVVVVVVVVVV",  # Poly-valine (aggregation-prone)
    "EKEKKEKEKEKEKEKEKEKE",  # Charged (soluble)
]

results = screener.screen_batch(sequences)

for seq, result in zip(sequences, results):
    print(f"{seq[:10]}... → {result.risk_level.value}")
    if result.risk_factors:
        print(f"  Concerns: {', '.join(result.risk_factors)}")
```

---

## Hierarchical Folding

### Progressive Search Space Confinement

Real proteins fold hierarchically:
1. Secondary structure forms first
2. Tertiary contacts lock in
3. Fine-tuning of final structure

The hierarchical folding system mimics this:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL FOLDING PHASES                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1: GLOBAL EXPLORATION (0-30% progress)                           │
│  ──────────────────────────────────────────                             │
│    • Wide search, no constraints                                        │
│    • Form initial secondary structure                                   │
│    • All moves allowed, high aggressiveness                             │
│                                                                          │
│            ▼ Transition based on secondary structure formation           │
│                                                                          │
│  PHASE 2: LOCAL EXPLORATION (30-60% progress)                           │
│  ─────────────────────────────────────────                              │
│    • Anchor stable secondary structure (soft locks)                     │
│    • Focus on tertiary contact formation                                │
│    • Reduce large conformational jumps                                  │
│                                                                          │
│            ▼ Transition based on anchoring progress                      │
│                                                                          │
│  PHASE 3: FINE REFINEMENT (60-100% progress)                            │
│  ────────────────────────────────────────                               │
│    • Lock most secondary structure (hard locks)                         │
│    • Small-scale optimization only                                      │
│    • Fine-tune loop regions                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Residue Anchoring System

```
ANCHOR STRENGTH LEVELS:

  NONE    → No constraint, free to move
  SOFT    → Prefer to keep, can override with good energy
  MEDIUM  → Resist change, need significant improvement to move
  HARD    → Strongly constrained, only minor adjustments
  LOCKED  → Fixed, no further changes allowed

ANCHORING CRITERIA:
  • Secondary structure detected (helix 'H' or sheet 'E')
  • Phi/psi angles in allowed regions
  • Confidence > 0.8 (persistence across iterations)
```

---

## Quantum Refinement Engine

### Two-Stage Optimization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM REFINEMENT ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Coarse Structure (7-14Å RMSD from native)                       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     STAGE 1: QUANTUM CORE                         │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                    │   │
│  │  1.1 Quantum Core Identification                                  │   │
│  │      • Identify high-QCP residues (most stable regions)           │   │
│  │      • Build THz resonance cascade model                          │   │
│  │      • Map quantum coherence network                              │   │
│  │                                                                    │   │
│  │  1.2 Secondary Structure Registration                             │   │
│  │      • Align helices to φ-harmonic spacing                        │   │
│  │      • Register sheets with golden ratio patterns                 │   │
│  │      • Apply G(φ,t) field constraints                             │   │
│  │                                                                    │   │
│  │  1.3 Hydrophobic Core Quantum Packing                             │   │
│  │      • Cluster hydrophobic residues                               │   │
│  │      • Apply water shielding principles                           │   │
│  │      • Optimize van der Waals contacts                            │   │
│  │                                                                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│                              ▼                                           │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   STAGE 2: LOCAL REFINEMENT                       │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                    │   │
│  │  2.1 Loop Refinement                                              │   │
│  │      • Identify flexible loop regions                             │   │
│  │      • Apply distance restraints                                  │   │
│  │      • Minimize loop energy while respecting restraints           │   │
│  │                                                                    │   │
│  │  2.2 Tertiary Contact Enforcement                                 │   │
│  │      • Predict expected contacts from sequence                    │   │
│  │      • Apply distance restraint network                           │   │
│  │      • Iteratively minimize violations                            │   │
│  │                                                                    │   │
│  │  2.3 Final Energy Minimization                                    │   │
│  │      • Combined energy + QCPP optimization                        │   │
│  │      • Golden ratio geometry fine-tuning                          │   │
│  │      • Convergence check                                          │   │
│  │                                                                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  OUTPUT: Refined Structure (<5Å RMSD target)                            │
│                                                                          │
│  Physics Constants:                                                      │
│    φ = 1.618...            (Golden ratio)                              │
│    Additional quantum parameters are proprietary                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Checkpointing and State Management

### What Gets Saved

```python
checkpoint_data = {
    "version": "1.0.0",
    "timestamp": 1733500000,
    "iteration": 500,
    
    "agents": [
        {
            "agent_id": "agent_0",
            "consciousness": {
                "frequency": 9.5,
                "coherence": 0.65
            },
            "behavioral": {
                "exploration_energy": 0.54,
                "structural_focus": 0.65,
                # ... 5D state
            },
            "conformation": {
                "coordinates": [[x, y, z], ...],
                "phi_angles": [...],
                "psi_angles": [...],
                "secondary_structure": "CCCHHHHHHCCEEEEECC"
            },
            "memories": [...],
            "best_energy": -125.4,
            "best_rmsd": 4.8
        },
        # ... more agents
    ],
    
    "shared_pool": {
        "significant_memories": [...],
        "broadcast_count": 23
    },
    
    "metadata": {
        "sequence": "MQIFVKTLTGKTITLEVEPS...",
        "native_pdb": "1UBQ",
        "total_conformations": 10000
    },
    
    "integrity_hash": "a1b2c3d4..."  # SHA256 for validation
}
```

### Checkpoint Rotation

```
checkpoints/
├── checkpoint_iter500_1733500000.json   # Most recent
├── checkpoint_iter400_1733499500.json
├── checkpoint_iter300_1733499000.json
├── checkpoint_iter200_1733498500.json
└── checkpoint_iter100_1733498000.json   # Oldest (will be deleted next)

Max checkpoints: 5 (configurable)
Auto-save interval: Every 50 iterations (configurable)
```

### Resume from Checkpoint

```python
from ubf_protein.checkpoint import CheckpointManager
from ubf_protein.protein_agent import ProteinAgent

manager = CheckpointManager("checkpoints")

# Load checkpoint
checkpoint_data = manager.load_checkpoint("checkpoint_iter500.json")

# Restore agents and shared pool
agents, shared_pool, iteration = manager.restore_agents(
    checkpoint_data, 
    ProteinAgent
)

# Continue exploration from iteration 500
coordinator.resume_exploration(
    agents=agents,
    shared_pool=shared_pool,
    start_iteration=iteration
)
```

---

## Validation and Quality Metrics

### RMSD (Root Mean Square Deviation)

```
RMSD = sqrt(Σ(r_pred - r_native)² / N)

With Kabsch alignment for optimal superposition:
1. Center both structures at origin
2. Calculate covariance matrix H
3. SVD: H = UΣV^T
4. Rotation matrix: R = VU^T
5. Apply rotation, calculate RMSD
```

### GDT-TS (Global Distance Test - Total Score)

```
GDT-TS = (GDT_P1 + GDT_P2 + GDT_P4 + GDT_P8) / 4

Where GDT_Pn = % of residues within n Å of native position

Quality thresholds:
  > 80%: Excellent (near-experimental quality)
  > 65%: Good (useful for most applications)
  > 50%: Acceptable (rough fold correct)
  < 50%: Poor (may have wrong fold)
```

### TM-score (Template Modeling Score)

```
TM-score = max[Σ(1 / (1 + (d_i/d_0)²)) / L_native]

Where:
  d_i = distance between aligned residues
  d_0 = 1.24 × ∛(L_native - 15) - 1.8  (length-dependent)

Quality thresholds:
  > 0.5: Same fold (statistically significant)
  < 0.17: Random relationship
```

### Quality Assessment Summary

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| RMSD | <2.0 Å | <4.0 Å | <6.0 Å | >6.0 Å |
| GDT-TS | >80% | >65% | >50% | <50% |
| TM-score | >0.7 | >0.5 | >0.4 | <0.4 |

---

## Configuration Reference

### PredictionConfig Options

```python
@dataclass
class PredictionConfig:
    # Required
    sequence: str                     # Amino acid sequence (uppercase)
    
    # Native structure (for validation)
    native_pdb: Optional[str] = None  # PDB ID (e.g., "1UBQ")
    pdb_file_path: Optional[str] = None  # Local PDB file
    
    # Agent configuration (None = auto-configure)
    agents: Optional[int] = None      # Number of agents
    iterations: Optional[int] = None  # Iterations per agent
    diversity: str = "balanced"       # Agent diversity profile
    
    # QCPP configuration
    qcpp_config: str = "default"      # 'default', 'high_performance', 'high_accuracy', 'none'
    qcpp_frequency: int = 20          # Analyze every N iterations
    cache_size: int = 10000           # QCPP cache size
    
    # Advanced features
    enable_refinement: bool = False   # Quantum refinement
    enable_mediators: bool = False    # Mediator agents
    mediator_count: int = 2
    target_geometry: str = "none"     # Platonic solid target
    
    # Hierarchical folding
    enable_hierarchical_folding: bool = False
    hierarchical_aggressive: bool = False
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 50
    
    # Output
    output_dir: Optional[str] = None
    save_pdb: bool = True
    save_trajectory: bool = True
```

### Auto-Configuration by Protein Size

```python
def get_optimal_settings(sequence_length: int) -> Dict:
    if sequence_length < 50:      # Small
        return {"agents": 15, "iterations": 300}
    elif sequence_length < 100:   # Medium
        return {"agents": 20, "iterations": 200}
    elif sequence_length < 150:   # Large
        return {"agents": 30, "iterations": 250}
    else:                         # Very large
        return {"agents": 50, "iterations": 300}
```

### QCPP Configuration Presets

| Preset | Analysis Frequency | Cache Size | Use Case |
|--------|-------------------|------------|----------|
| `default` | Every iteration | 1,000 | Balanced |
| `high_performance` | Every 5 iterations | 5,000 | Speed priority |
| `high_accuracy` | Every iteration | 10,000 | Quality priority |
| `none` | Disabled | N/A | No QCPP |

---

## Usage Examples

### Basic Prediction

```python
from ubf_protein.prediction_runner import PredictionRunner, PredictionConfig

config = PredictionConfig(
    sequence="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQD...",
    native_pdb="1UBQ",  # Optional: for validation
)

runner = PredictionRunner(config)
results = runner.run()

print(f"Best energy: {results.best_energy:.2f} kcal/mol")
print(f"Best RMSD: {results.best_rmsd:.2f} Å")
print(f"Quality: {results.validation_quality}")
```

### With Progress Monitoring

```python
def on_progress(update):
    print(f"[{update.progress_percentage:.0f}%] "
          f"Energy: {update.best_energy:.1f}, "
          f"RMSD: {update.best_rmsd:.2f}Å")

results = runner.run(progress_callback=on_progress)
```

### Full Configuration

```python
config = PredictionConfig(
    sequence="MQIFVKTLTGKTITLEVEPS...",
    native_pdb="1UBQ",
    
    # More agents for better exploration
    agents=30,
    iterations=500,
    
    # Enable all features
    qcpp_config="high_accuracy",
    enable_refinement=True,
    enable_hierarchical_folding=True,
    
    # Checkpointing
    enable_checkpointing=True,
    checkpoint_interval=50,
)

runner = PredictionRunner(config)
results = runner.run(progress_callback=on_progress)
```

---

## Appendix: Move Types

| Move Type | Description | Target Residues |
|-----------|-------------|-----------------|
| `BACKBONE_ROTATION` | Rotate backbone phi/psi angles | 1-5 random |
| `SIDECHAIN_ADJUST` | Adjust side chain conformation | 1-5 random |
| `HELIX_FORMATION` | Bias toward helical angles | 6 consecutive |
| `SHEET_FORMATION` | Bias toward sheet angles | 6 consecutive |
| `TURN_FORMATION` | Create turn/loop structure | 3-4 consecutive |
| `HYDROPHOBIC_COLLAPSE` | Move hydrophobic residues together | Variable |
| `ENERGY_MINIMIZATION` | Local energy optimization | All |
| `PIVOT_ROTATION` | Rotate segment around pivot point | Half of chain |

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `prediction_runner.py` | **Entry point** - Use this for all predictions |
| `multi_agent_coordinator.py` | Orchestrates parallel agent exploration |
| `protein_agent.py` | Individual agent implementation |
| `energy_function.py` | 6-term molecular mechanics energy |
| `qcpp_integration.py` | QCPP adapter with caching |
| `mapless_moves.py` | O(1) conformational space move generation |
| `memory_system.py` | Experience storage and collective learning |
| `consciousness.py` | 2D exploration parameters |
| `behavioral_state.py` | 5D search strategy derivation |
| `aggregation_screening.py` | Fast risk screening |
| `hierarchical_folding.py` | Progressive search confinement |
| `quantum_refinement_engine.py` | Two-stage refinement |
| `rmsd_calculator.py` | Structure validation metrics |
| `checkpoint.py` | State persistence |
| `geometric_attractor.py` | Golden ratio pattern analysis |

---

*Document generated for EmergentFolds v1.0*
*Last updated: December 2025*
