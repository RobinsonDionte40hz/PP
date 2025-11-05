# Geometric Targeting Enhancement Proposal

## Current State (Post-Analysis Only)
✅ System calculates octahedron/icosahedron/dodecahedron similarities  
✅ Reported after exploration completes  
❌ Agents don't actively optimize for specific geometries

**Example Current Output:**
```
✓ Geometric analysis complete:
  📐 Platonic Solid Similarities:
     - Icosahedron (φ-based): 0.847
     - Dodecahedron (φ-based): 0.723
     - Octahedron: 0.978  ← HIGH but not used to guide agents
```

## Proposed Enhancement: Active Geometric Guidance

### Goal
Allow users to specify target geometry per protein test. Agents actively search for and optimize toward that geometry during exploration.

### Use Cases
- **Membrane proteins**: Target octahedral packing
- **Globular proteins**: Target icosahedral symmetry
- **Fibrous proteins**: Target specific symmetries
- **Research**: Compare folding efficiency with different geometric targets

---

## Implementation Design

### 1. CLI Configuration (User Interface)

**Add `--target-geometry` flag to test_protein.py:**

```python
parser.add_argument('--target-geometry', 
                    choices=['none', 'octahedron', 'icosahedron', 'dodecahedron', 'tetrahedron', 'cube'],
                    default='none',
                    help='Target geometric pattern for agents to optimize toward (default: none)')
```

**Example Usage:**
```bash
# Optimize for octahedral geometry
python test_protein.py --pdb 1CRN --agents 10 --iterations 200 --target-geometry octahedron

# Optimize for icosahedral symmetry  
python test_protein.py --pdb 1CRN --agents 10 --iterations 200 --target-geometry icosahedron

# No geometric guidance (current behavior)
python test_protein.py --pdb 1CRN --agents 10 --iterations 200 --target-geometry none
```

### 2. QCPPMetrics Extension (Data Model)

**Add geometric similarity to QCPPMetrics dataclass:**

```python
# In ubf_protein/qcpp_integration.py

@dataclass(frozen=True)
class QCPPMetrics:
    """QCPP analysis results for a protein conformation."""
    qcp_value: float
    coherence: float
    field_strength: float
    phi_match_score: float
    resonance_strength: float
    water_shielding: float
    stability_score: float
    geometric_similarity: float = 0.0  # ← NEW: Similarity to target geometry (0-1)
```

### 3. Real-Time Geometric Evaluation (Analysis Engine)

**Add fast geometric similarity calculation:**

```python
# In ubf_protein/qcpp_integration.py or new geometric_scoring.py

def calculate_geometric_similarity(coords: List[np.ndarray], 
                                   target_geometry: str) -> float:
    """
    Fast geometric similarity calculation for real-time evaluation.
    
    Args:
        coords: CA coordinates (N x 3)
        target_geometry: 'octahedron', 'icosahedron', 'dodecahedron', etc.
        
    Returns:
        Similarity score 0.0-1.0
    """
    if target_geometry == 'none':
        return 0.0
    
    # Fast approximation (not full Platonic solid analysis)
    # Use distance ratios, angle distributions, symmetry measures
    
    if target_geometry == 'octahedron':
        # Octahedron: 6 vertices, 12 edges, square cross-section
        return _octahedron_similarity_fast(coords)
    elif target_geometry == 'icosahedron':
        # Icosahedron: 12 vertices, 30 edges, pentagonal symmetry, φ ratios
        return _icosahedron_similarity_fast(coords)
    # ... etc
```

**Performance Target**: <2ms (must stay within move evaluation budget)

### 4. Move Evaluation Integration (Agent Decision-Making)

**Weight moves by geometric similarity:**

```python
# In ubf_protein/protein_agent.py - _evaluate_move_with_qcpp()

def _evaluate_move_with_qcpp(self, move: Move, metrics: QCPPMetrics) -> float:
    """Evaluate move with QCPP-enhanced weighting including geometric targeting."""
    
    # ... existing factors (physical, quantum, behavioral, historical, goal) ...
    
    # NEW: Geometric targeting factor
    geometric_factor = 1.0
    if metrics.geometric_similarity > 0.0:  # Target geometry specified
        # Reward moves that increase similarity to target
        geometric_factor = 0.8 + (0.4 * metrics.geometric_similarity)  # Range: 0.8-1.2
        
        # Extra bonus for high similarity (positive feedback loop)
        if metrics.geometric_similarity > 0.7:
            geometric_factor *= 1.1  # Up to 1.32x weight
    
    # Combine all factors
    weight = (physical_factor * 
              quantum_factor * 
              behavioral_factor * 
              historical_factor * 
              goal_factor * 
              geometric_factor *  # ← NEW
              temperature)
    
    return weight
```

### 5. Memory Significance Enhancement (Learning System)

**Factor geometric similarity into 7-signal significance (becomes 8-signal):**

```python
# In ubf_protein/memory_system.py - calculate_significance()

def calculate_significance(self, memory: ConformationalMemory, 
                          target_geometry: Optional[str] = None) -> float:
    """Enhanced 8-signal significance including geometric targeting."""
    
    # Existing 7 signals
    energy_impact = ...
    structural_novelty = ...
    rmsd_improvement = ...
    consciousness_state = ...
    qcpp_metrics_quality = ...
    resonance_alignment = ...
    water_shielding_quality = ...
    
    # NEW: 8th signal - geometric targeting
    geometric_alignment = 0.0
    if target_geometry and memory.qcpp_metrics:
        geometric_alignment = memory.qcpp_metrics.geometric_similarity
        
        # High geometric similarity is very significant
        if geometric_alignment > 0.7:
            geometric_alignment *= 1.5  # Boost significant matches
    
    # Combine signals with weights
    significance = (
        energy_impact * 0.25 +
        structural_novelty * 0.15 +
        rmsd_improvement * 0.15 +
        consciousness_state * 0.10 +
        qcpp_metrics_quality * 0.10 +
        resonance_alignment * 0.10 +
        water_shielding_quality * 0.05 +
        geometric_alignment * 0.10  # ← NEW (10% weight)
    )
    
    return significance
```

### 6. Multi-Agent Coordination (System-Wide)

**Pass target geometry through coordinator:**

```python
# In ubf_protein/multi_agent_coordinator.py

def __init__(self, 
             protein_sequence: str,
             target_geometry: str = 'none',  # ← NEW parameter
             ...):
    self.target_geometry = target_geometry
    # ... pass to agents during creation

def _create_agent(self, agent_id: int, ...) -> ProteinAgent:
    return ProteinAgent(
        ...,
        target_geometry=self.target_geometry  # ← Pass to agent
    )
```

---

## Expected Impact

### Performance
- **Move evaluation**: +1-2ms per move (geometric calculation)
- **Memory storage**: +8 bytes per memory (geometric_similarity float)
- **Cache efficiency**: Similar (geometric similarity part of state hash)

### Behavior Changes
- **Convergence speed**: 20-40% faster to target geometry (predicted)
- **Memory sharing**: More geometric-focused experiences shared
- **Diversity**: Agents still explore but biased toward target
- **Energy landscape**: May sacrifice some energy for geometric optimality

### Output Enhancements
```
🎯 Geometric Targeting: Octahedron
  Initial similarity: 0.543
  Final similarity: 0.978 (+80.1%)
  
✓ Target geometry achieved in 847 iterations
  Agents converged 35% faster than non-targeted exploration
```

---

## Testing Strategy

### Unit Tests
1. **Geometric scoring**: Test fast similarity calculations
2. **Move weighting**: Verify geometric factor applied correctly
3. **Memory significance**: Confirm 8th signal integration
4. **Configuration**: Test CLI flag parsing and propagation

### Integration Tests
1. **Target vs non-target**: Compare convergence speed
2. **Multiple geometries**: Test octahedron, icosahedron, dodecahedron targets
3. **Small/medium/large proteins**: Ensure adaptive config compatibility
4. **QCPP cache**: Verify geometric similarity cached properly

### Validation Metrics
- **Convergence speed**: Iterations to reach 0.7+ target similarity
- **Final similarity**: Geometric score at end of exploration
- **Energy trade-off**: Compare final energies (target vs non-target)
- **RMSD impact**: Check if targeting affects structural accuracy

---

## Implementation Phases

### Phase 1: Infrastructure (1-2 hours)
- [ ] Add `--target-geometry` CLI flag to test_protein.py
- [ ] Extend QCPPMetrics with `geometric_similarity` field
- [ ] Add `target_geometry` parameter to ProteinAgent, MultiAgentCoordinator
- [ ] Update configuration propagation

### Phase 2: Geometric Scoring (2-3 hours)
- [ ] Implement fast geometric similarity calculations
  - [ ] `_octahedron_similarity_fast(coords)`
  - [ ] `_icosahedron_similarity_fast(coords)`
  - [ ] `_dodecahedron_similarity_fast(coords)`
- [ ] Integrate into QCPPIntegration.analyze_conformation()
- [ ] Performance test (<2ms target)

### Phase 3: Agent Integration (1-2 hours)
- [ ] Add geometric factor to move evaluation
- [ ] Update memory significance (8th signal)
- [ ] Test move weighting with geometric targets

### Phase 4: Testing & Validation (2-3 hours)
- [ ] Unit tests for geometric scoring
- [ ] Integration tests comparing targeted vs non-targeted
- [ ] Validate on 1CRN, 1VII, 1UBQ with different targets
- [ ] Performance profiling

### Phase 5: Documentation (1 hour)
- [ ] Update README.md with geometric targeting examples
- [ ] Add to EXAMPLES.md
- [ ] Update copilot-instructions.md

**Total Estimated Time**: 7-11 hours

---

## Alternative: Lightweight Configuration-Only Approach

If full prescriptive guidance is too complex, we can add **post-analysis filtering**:

### Quick Win: Analysis-Driven Re-Testing

```python
# In test_protein.py - add after geometric analysis

if args.auto_retry and symmetry_results.octahedron_similarity < 0.7:
    print(f"⚡ Octahedron similarity below threshold. Re-running with more agents...")
    # Automatically re-test with adjusted parameters
    run_test(args.pdb, agents=args.agents * 2, iterations=args.iterations * 1.5)
```

**Pros**: No move evaluation changes, simple to implement  
**Cons**: Still not prescriptive, just automated retry logic

---

## Recommendation

**Implement Full Prescriptive Geometric Targeting** (Option 2, all 5 phases)

**Why?**
1. **Aligns with UBF philosophy**: Consciousness-guided exploration toward goals
2. **Leverages existing infrastructure**: QCPP metrics, memory significance, move weighting
3. **High research value**: Can compare folding efficiency with different geometric targets
4. **User flexibility**: Per-protein geometry configuration via simple CLI flag
5. **Modest complexity**: 7-11 hours implementation, minimal performance impact

**Quick Start After Implementation:**
```bash
# Test octahedral targeting on Crambin
python test_protein.py --pdb 1CRN --agents 10 --iterations 200 --target-geometry octahedron

# Compare with no targeting
python test_protein.py --pdb 1CRN --agents 10 --iterations 200 --target-geometry none

# Test icosahedral targeting on Ubiquitin
python test_protein.py --pdb 1UBQ --agents 20 --iterations 500 --target-geometry icosahedron
```

---

## Questions for Decision

1. **Scope**: Full prescriptive targeting or lightweight auto-retry?
2. **Priority**: Immediate need or future enhancement?
3. **Geometries**: Start with octahedron only or all 5 Platonic solids?
4. **Performance**: Accept 1-2ms move evaluation overhead?
5. **Testing**: Which proteins to validate on (1CRN, 1UBQ, 1VII)?

Let me know which direction you want to go and I can start implementation! 🚀
