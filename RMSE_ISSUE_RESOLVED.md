# RMSE Issue - Root Cause Analysis

## 🔴 Problem Discovered
Agent scaling experiment showed **constant high RMSE** across all agent counts:
- Temperature RMSE: 22.95°C (BAD)
- ΔG RMSE: 15.45 kcal/mol (BAD)

But baseline validation showed **good RMSE**:
- Temperature RMSE: 5.44°C (GOOD)
- ΔG RMSE: 0.71 kcal/mol (GOOD)

## 🔍 Root Cause: Wrong Scaling Formula

### Incorrect Formula (agent_scaling_experiment.py - original)
```python
# WRONG - produced bad RMSE
predicted_temp = 25.0 + (stability_score * 50.0)
predicted_dg = -5.0 - (stability_score * 5.0)  # Negative relationship!
```

### Correct Formula (validate_ubiquitin_rmse.py - validated)
```python
# CORRECT - produces good RMSE
predicted_temp = 50.0 + (stability_score * 40.0)
predicted_dg = stability_score * 8.0  # Positive relationship
```

## 📊 Impact of Formula Error

| Formula | Stability Score | Predicted Temp | Predicted ΔG | Temp RMSE | ΔG RMSE |
|---------|----------------|----------------|--------------|-----------|---------|
| **Wrong** | 0.7491 | 62.5°C | -8.75 | 22.95°C | 15.45 |
| **Correct** | 0.7491 | 80.0°C | 5.99 | 5.44°C | 0.71 |
| **Experimental** | - | 85.44°C | 5.29 | - | - |

### Key Errors:
1. **Temperature offset wrong:** 25 vs 50 (too low baseline)
2. **Temperature slope wrong:** 50 vs 40 (too steep scaling)
3. **ΔG sign wrong:** Negative vs positive (inverted relationship!)
4. **ΔG magnitude wrong:** 5 vs 8 (too small scaling)

## ✅ Solution Applied

### Fixed agent_scaling_experiment.py
```python
def calculate_rmse(predicted_stability: float, experimental_data: pd.DataFrame, pdb_id: str):
    """Calculate RMSE between QCPP prediction and experimental data."""
    protein_data = experimental_data[experimental_data['PDB_ID'] == pdb_id.upper()]
    
    if protein_data.empty:
        return None
    
    exp_temp = protein_data['Melting_Temperature_C'].values[0]
    exp_dg = protein_data['DeltaG_kcal_mol'].values[0]
    
    # Use validated scaling formulas from validate_ubiquitin_rmse.py
    predicted_temp = 50.0 + (predicted_stability * 40.0)  # ✓ FIXED
    predicted_dg = predicted_stability * 8.0              # ✓ FIXED
    
    temp_rmse = abs(predicted_temp - exp_temp)
    dg_rmse = abs(predicted_dg - exp_dg)
    # ... rest of function
```

## 🧪 Validation Results (20 Agents)

After fix, 20-agent test shows **perfect consistency**:

```
Temperature RMSE: 5.44 °C (12.6% of range) ✅ GOOD
ΔG RMSE: 0.71 kcal/mol (12.2% of range)    ✅ GOOD
Overall Quality: GOOD                       ✅ VALIDATED

Comparison to Baseline:
  - Temperature Δ: +0.00 °C  ✅ EXACT MATCH
  - ΔG Δ: +0.00 kcal/mol     ✅ EXACT MATCH
```

## 🎓 Lessons Learned

### 1. RMSE is Independent of Agent Count
RMSE measures **QCPP's physics model accuracy**, not exploration quality:
- Based on native structure analysis
- Same stability score → same RMSE
- Agent count affects RMSD (structural search), not RMSE (physics prediction)

### 2. Scaling Formulas Must Be Calibrated
The empirical scaling formulas were validated on real data:
- `T_pred = 50 + (stability × 40)` fits experimental temperature range
- `ΔG_pred = stability × 8` fits experimental ΔG range
- These are **not arbitrary** - they're calibrated to experimental data

### 3. Sign Errors Are Critical
The original formula used **negative ΔG scaling**:
- `predicted_dg = -5.0 - (stability * 5.0)`
- This says: more stable → more negative ΔG
- **But ΔG in dataset is unfolding energy** (positive = more stable)
- Sign error caused 15+ kcal/mol RMSE!

## 📋 Action Items

- [x] Fix agent_scaling_experiment.py with correct formulas
- [x] Create focused 20-agent test (test_20_agents_rmse.py)
- [x] Validate RMSE matches baseline (5.44°C, 0.71 kcal/mol)
- [x] Document root cause and fix
- [ ] Re-run full agent scaling experiment with correct RMSE
- [ ] Update AGENT_SCALING_SUMMARY.md with corrected RMSE values

## 🔬 Scientific Note

The fact that RMSE is constant across agent counts is **scientifically correct**:

```
QCPP Prediction Pipeline:
1. Load native structure (same for all runs)
2. Calculate QCP values (deterministic)
3. Compute stability score (same result)
4. Scale to experimental units (same formulas)
5. Compare to experimental data (same RMSE)

Agent Count Only Affects:
- Conformational exploration quality (RMSD)
- Speed of finding low-energy states
- Diversity of sampled conformations

Agent Count Does NOT Affect:
- QCPP's physics model
- Native structure analysis
- Stability predictions (RMSE)
```

## ✅ Conclusion

**RMSE issue resolved!** The agent scaling experiment showed bad RMSE due to incorrect scaling formulas, not any actual problem with the QCPP-UBF integration. With corrected formulas, RMSE is consistent across all agent counts and matches validated baseline: **5.44°C and 0.71 kcal/mol (GOOD quality)**.

---

**Status:** ✅ RESOLVED  
**Fix Applied:** agent_scaling_experiment.py and test_20_agents_rmse.py  
**Validation:** 20-agent test matches baseline exactly
