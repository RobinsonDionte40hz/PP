# Deep Mechanism Analysis - Investigation Summary

## Purpose
This analysis addresses three follow-up questions about the inverse scaling mechanism:

1. **Exploration Diversity**: Do large proteins explore more unique conformations?
2. **Conformational Mixing**: Do large proteins make bigger conformational jumps?
3. **Consciousness Adaptation**: Does protein size drive more behavioral complexity?

## What We're Measuring

### 1. Exploration Diversity
- **Unique Conformations**: Count of distinct φ/ψ angle patterns visited
- **Diversity Ratio**: unique_conformations / total_conformations
- **Redundancy Ratio**: 1 - diversity_ratio

**Hypothesis**: If large proteins have smoother landscapes (fewer local minima), they should:
- Visit MORE unique conformations (less revisiting)
- Have HIGHER diversity ratios
- Show POSITIVE correlation: Size vs Diversity

### 2. Conformational Mixing  
- **Transition Fraction**: % of residues changing φ/ψ significantly between steps
- **Mixing Events**: Count of large conformational jumps (>50% residues changed)
- **Mixing Rate**: mixing_events / total_steps

**Hypothesis**: If smooth landscapes enable bigger moves, large proteins should:
- Make LARGER conformational transitions
- Have HIGHER mixing rates
- Show POSITIVE correlation: Size vs Mixing

### 3. Consciousness Adaptation
- **Frequency/Coherence Trajectories**: Path through consciousness space over time
- **Trajectory Complexity**: Total path length in (frequency, coherence) 2D space
- **Path Length Per Step**: Average consciousness coordinate change per iteration

**Hypothesis**: If large proteins drive more behavioral adaptation, they should:
- Show MORE consciousness coordinate changes
- Have HIGHER trajectory complexity
- Show POSITIVE correlation: Size vs Consciousness

## Expected Outcomes

### Scenario A: All Three Correlate Positively with Size
**Interpretation**: Large proteins enable:
- More diverse exploration (fewer revisits)
- Bigger conformational jumps (better mixing)
- More behavioral adaptation (dynamic consciousness)
→ **Smooth landscapes drive exploratory behavior**

### Scenario B: Only Diversity Correlates
**Interpretation**: 
- Large proteins visit more conformations BUT
- Don't make bigger individual jumps
- Don't require more behavioral adaptation
→ **Smooth landscapes reduce redundancy, not change dynamics**

### Scenario C: Only Mixing Correlates
**Interpretation**:
- Large proteins make bigger jumps BUT
- Don't necessarily visit more unique states
- May revisit similar regions via different paths
→ **Smooth landscapes enable bold moves, but convergence still occurs**

### Scenario D: Only Consciousness Correlates
**Interpretation**:
- Large proteins drive more behavioral changes BUT
- Don't necessarily explore more or jump further
- Consciousness adapts to landscape, not just size
→ **Behavioral complexity scales with protein size independently**

## Running Analysis

```bash
python scripts/experiments/analyze_deep_mechanisms.py
```

**Duration**: ~10 minutes (5 proteins × 500 iterations × 10 agents)

**Output**:
- `results/deep_mechanism/1VII_deep_analysis.json`
- `results/deep_mechanism/1CRN_deep_analysis.json`
- `results/deep_mechanism/1UBQ_deep_analysis.json`
- `results/deep_mechanism/1LYZ_deep_analysis.json`
- `results/deep_mechanism/1MBN_deep_analysis.json`
- `results/deep_mechanism/comparative_deep_analysis.json`

## Metrics Collected

### Per Protein:
- Total conformations sampled
- Unique conformations visited
- Diversity ratio
- Mean transition fraction
- Mixing events count
- Mixing rate
- Mean frequency/coherence
- Frequency/coherence ranges
- Trajectory complexity
- Best energy achieved

### Correlations Computed:
- Size vs Diversity (r, p-value)
- Size vs Mixing (r, p-value)
- Size vs Consciousness (r, p-value)

## Integration with Publication

If strong correlations are found (|r| > 0.7, p < 0.05), we can:

1. **Add to Results section**: 
   - "Mechanistic Deep Dive: Exploration Dynamics"
   - Report correlations and interpret mechanism

2. **Create Figure 2**: 
   - 3-panel plot (Size vs Diversity/Mixing/Consciousness)
   - Similar style to Figure 1 (landscape topology)

3. **Strengthen Discussion**:
   - Explain HOW smooth landscapes improve prediction
   - Connect topology to exploratory behavior
   - Distinguish between landscape structure and search dynamics

4. **Enhance Conclusions**:
   - "Smooth landscapes enable X, Y, Z behaviors"
   - Quantify behavioral advantages of large proteins
   - Guide algorithm optimization strategies

## Status

**RUNNING**: Deep mechanism analysis in progress
**Expected completion**: ~10 minutes
**Next**: Analyze correlations and update publication if significant

---

**Analysis started**: November 5, 2025
**Script**: `scripts/experiments/analyze_deep_mechanisms.py`
**Investigator**: AI-driven mechanistic inquiry
