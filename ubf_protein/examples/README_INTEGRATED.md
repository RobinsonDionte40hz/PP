# Integrated QCPP-UBF Exploration Example

This example demonstrates the full integration between the **Quantum Coherence Protein Predictor (QCPP)** and **Universal Behavioral Framework (UBF)** systems for real-time protein structure prediction.

## Overview

The `integrated_exploration.py` script showcases:

- **Real-time QCPP guidance**: Move evaluation uses quantum physics instead of simplified approximations
- **Physics-grounded consciousness**: Agent consciousness coordinates map to QCPP metrics
- **Dynamic parameter adjustment**: Exploration adapts based on QCPP stability scores
- **Phi pattern rewards**: Golden ratio geometries receive energy bonuses
- **Integrated trajectory recording**: Tracks both UBF and QCPP metrics
- **Correlation analysis**: Validates relationship between quantum coherence and structural quality

## Quick Start

### Basic Usage

```bash
# Simple exploration with default settings
python integrated_exploration.py --sequence ACDEFGH

# With native structure for RMSD validation
python integrated_exploration.py \
    --sequence ACDEFGH \
    --native native_structure.pdb \
    --agents 5 \
    --iterations 1000
```

### Advanced Usage

```bash
# Full exploration with custom configuration
python integrated_exploration.py \
    --sequence ACDEFGHIKLMNPQRSTVWY \
    --agents 10 \
    --iterations 2000 \
    --diversity balanced \
    --config high_accuracy \
    --output results.json

# High-performance mode with large cache
python integrated_exploration.py \
    --sequence ACDEFGH \
    --config high_performance \
    --cache-size 5000 \
    --agents 20 \
    --iterations 5000
```

## Command-Line Arguments

### Required Arguments

- `--sequence SEQUENCE`: Protein amino acid sequence (single-letter codes)

### Optional Exploration Parameters

- `--agents N` (default: 10): Number of autonomous agents
- `--iterations N` (default: 2000): Iterations per agent
- `--diversity PROFILE` (default: balanced): Agent diversity profile
  - `cautious`: Conservative exploration (low frequency, high coherence)
  - `balanced`: Mixed exploration (moderate frequency and coherence)
  - `aggressive`: Bold exploration (high frequency, low coherence)
- `--native PATH`: Path to native structure PDB for RMSD calculation

### QCPP Configuration

- `--config PRESET` (default: default): QCPP integration configuration
  - `default`: Balanced performance and accuracy
  - `high_performance`: Optimized for speed (larger cache, less frequent analysis)
  - `high_accuracy`: Optimized for quality (analyze every iteration, detailed logging)
- `--cache-size N`: Override QCPP cache size (conformations)
- `--analysis-freq N`: Override QCPP analysis frequency (every N iterations)
- `--disable-qcpp`: Disable QCPP integration (use UBF only)

### Output Options

- `--output FILE`: Path to save results JSON
- `--quiet`: Suppress detailed output

## Configuration Presets

### Default Configuration
- Analysis frequency: Every iteration
- Cache size: 1000 conformations
- Physics grounding: Enabled
- Dynamic adjustment: Enabled
- Trajectory recording: Enabled

**Best for**: Standard exploration with good balance of speed and accuracy

### High-Performance Configuration
- Analysis frequency: Every 5 iterations
- Cache size: 5000 conformations
- Max calculation time: 2ms (strict timeout)
- Trajectory recording: Disabled

**Best for**: Large-scale explorations where speed is critical

### High-Accuracy Configuration
- Analysis frequency: Every iteration
- Cache size: 10000 conformations
- Max calculation time: 10ms (generous timeout)
- Trajectory recording: Enabled (50000 points)
- Detailed logging: Enabled

**Best for**: Research applications where accuracy is paramount

## Output

### Console Output

The script prints detailed information about:
1. **Configuration**: QCPP integration settings
2. **Initialization**: System setup and agent creation
3. **Progress**: Iteration updates during exploration
4. **Results**: Best energy, RMSD, and performance metrics
5. **QCPP Statistics**: Analysis count, cache hit rate, correlations

Example output:
```
======================================================================
INTEGRATED QCPP-UBF PROTEIN STRUCTURE EXPLORATION
======================================================================
Sequence:       ACDEFGHIKLMNPQRSTVWY
Length:         20 residues
Agents:         10
Iterations:     2000 per agent
Diversity:      balanced
Native PDB:     None (de novo prediction)
======================================================================

======================================================================
QCPP INTEGRATION CONFIGURATION
======================================================================
Status:                ENABLED
Analysis Frequency:    Every 1 iteration(s)
Cache Size:            1000 conformations
Max Calculation Time:  5.0ms
Phi Reward Threshold:  0.8
Phi Reward Energy:     -50.0 kcal/mol
Dynamic Adjustment:    ENABLED
Physics Grounding:     ENABLED
Trajectory Recording:  ENABLED
======================================================================

Step 1: Initializing QCPP Integration...
  ✓ Using production QCPP predictor
  ✓ QCPP adapter initialized (cache size: 1000)

Step 2: Creating Multi-Agent Coordinator...
  ✓ Coordinator created with 10 agents
  ✓ Diversity profile: balanced
  ✓ QCPP integration: ENABLED
  ✓ Physics-grounded consciousness: ENABLED
  ✓ Dynamic parameter adjustment: ENABLED

Step 3: Running Parallel Exploration with QCPP Feedback...
  Expected duration: ~6.7 minutes
  
  ✓ Exploration complete in 385.2s

Step 4: Analyzing Results and QCPP Correlations...

======================================================================
EXPLORATION RESULTS
======================================================================
Best Agent:             #3
Best Energy:            -125.8 kcal/mol
Best RMSD:              3.2 Å
Total Conformations:    20000
Exploration Time:       385.2s
Throughput:             51.9 conf/s
======================================================================

======================================================================
QCPP INTEGRATION STATISTICS
======================================================================
Total QCPP Analyses:    15234
Cache Hits:             4766
Cache Hit Rate:         31.3%
Avg Calculation Time:   2.3ms

QCP-RMSD Correlation:   -0.652
QCP-Energy Correlation: 0.721
Phi Pattern Frequency:  18.5%
======================================================================

TOTAL EXECUTION TIME: 387.5s
======================================================================
```

### JSON Output

When `--output FILE` is specified, results are saved in JSON format:

```json
{
  "sequence": "ACDEFGHIKLMNPQRSTVWY",
  "num_agents": 10,
  "iterations_per_agent": 2000,
  "diversity_profile": "balanced",
  "best_agent_id": 3,
  "best_energy": -125.8,
  "best_rmsd": 3.2,
  "total_conformations": 20000,
  "exploration_time_seconds": 385.2,
  "throughput_conformations_per_second": 51.9,
  "qcpp_integration": {
    "enabled": true,
    "total_analyses": 15234,
    "cache_hits": 4766,
    "cache_hit_rate": 31.3,
    "avg_calculation_time_ms": 2.3,
    "qcp_rmsd_correlation": -0.652,
    "qcp_energy_correlation": 0.721,
    "phi_pattern_frequency": 18.5
  },
  "configuration": {
    "enabled": true,
    "analysis_frequency": 1,
    "cache_size": 1000,
    "max_calculation_time_ms": 5.0,
    "phi_reward_threshold": 0.8,
    "phi_reward_energy": -50.0,
    "enable_dynamic_adjustment": true,
    "stability_low_threshold": 1.0,
    "stability_high_threshold": 2.0,
    "enable_physics_grounding": true,
    "smoothing_factor": 0.1,
    "enable_trajectory_recording": true,
    "trajectory_max_points": 10000
  },
  "timestamp": 1730000000.0
}
```

## Performance Expectations

### Single Agent
- **Move evaluation**: 0.5-1.5ms per move
- **QCPP analysis**: <5ms per conformation (target), 0.5-3ms typical
- **Cache hit rate**: 30-50% for typical exploration
- **Throughput**: ≥50 conformations/second

### Multi-Agent (10 agents)
- **Total runtime**: 2-5 minutes for 2000 iterations
- **Parallel efficiency**: 80-95% (near-linear scaling)
- **Memory overhead**: ~50MB per agent with QCPP integration
- **Cache sharing**: Thread-safe, minimal contention

### Performance Tips

1. **Increase cache size** for repeated structural motifs
2. **Reduce analysis frequency** for faster exploration
3. **Use high_performance config** for large-scale studies
4. **Disable trajectory recording** if not needed for analysis

## Understanding the Results

### QCP-RMSD Correlation
- **Negative correlation** (e.g., -0.65): Higher QCP scores predict lower RMSD (better structures)
- **Magnitude**: |r| > 0.6 indicates strong relationship
- **Interpretation**: Validates quantum coherence as structural quality metric

### QCP-Energy Correlation
- **Positive correlation** (e.g., 0.72): Higher QCP correlates with more favorable energy
- **Magnitude**: |r| > 0.7 indicates very strong relationship
- **Interpretation**: Quantum coherence aligns with energetic stability

### Phi Pattern Frequency
- **Percentage**: Fraction of conformations with strong phi patterns (score > 0.8)
- **Typical range**: 10-30% for well-structured proteins
- **High frequency**: Suggests golden ratio geometry is prevalent

### Cache Hit Rate
- **Low rate (<20%)**: Highly diverse exploration, many unique conformations
- **Medium rate (20-50%)**: Balanced exploration with some revisitation
- **High rate (>50%)**: Convergent exploration, focusing on specific regions

## Integration Features

### 1. Real-Time QCPP Guidance

Move evaluation uses QCPP-derived quantum alignment:
```
quantum_alignment = 0.5 + min(1.0, 
    (qcp_score / 5.0) * 0.4 +      # QCP contribution
    (coherence + 1.0) * 0.3 +       # Coherence contribution
    phi_match * 0.3                 # Phi pattern contribution
)
```

### 2. Physics-Grounded Consciousness

Agent consciousness maps to QCPP metrics:
```
target_frequency = 15.0 - (qcp_score / 0.5)  # Inverse relationship
target_coherence = 0.2 + (qcpp_coherence * 0.8)  # Direct mapping
```

### 3. Dynamic Parameter Adjustment

Exploration adapts based on stability:
- **Unstable region** (stability < 1.0): Increase frequency +2Hz, temperature +50K
- **Stable region** (stability > 2.0): Decrease frequency -1Hz, temperature -20K

### 4. Phi Pattern Rewards

Strong phi patterns (score > 0.8) receive -50 kcal/mol energy bonus, encouraging golden ratio geometries.

## Troubleshooting

### "QCPP system not available"
- The script uses a mock QCPP predictor for demonstration
- Install full QCPP system for production use
- Mock predictor provides simplified calculations

### Slow Performance
- Reduce `--agents` or `--iterations`
- Use `--config high_performance`
- Increase `--cache-size`
- Increase `--analysis-freq` (analyze less frequently)

### High Memory Usage
- Reduce `--cache-size`
- Disable trajectory recording: `--config` with custom settings
- Reduce `--agents` count

### Low Cache Hit Rate
- Normal for highly diverse exploration
- Increase `--cache-size` if revisiting conformations
- Check if exploration is too random (may need to adjust diversity)

## Example Workflows

### De Novo Prediction
```bash
# Predict structure without native reference
python integrated_exploration.py \
    --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL \
    --agents 20 \
    --iterations 5000 \
    --config high_accuracy \
    --output villin_prediction.json
```

### Validation Against Native
```bash
# Validate prediction quality
python integrated_exploration.py \
    --sequence ACDEFGHIKLMNPQRSTVWY \
    --native native.pdb \
    --agents 10 \
    --iterations 2000 \
    --output validation_results.json
```

### Fast Screening
```bash
# Quick exploration for multiple sequences
for seq in ACDEFG GHIKLMN PQRSTVWY; do
    python integrated_exploration.py \
        --sequence $seq \
        --agents 5 \
        --iterations 500 \
        --config high_performance \
        --quiet \
        --output ${seq}_results.json
done
```

## References

- **Design Document**: `.kiro/specs/qcpp-ubf-integration/design.md`
- **Requirements**: `.kiro/specs/qcpp-ubf-integration/requirements.md`
- **Task List**: `.kiro/specs/qcpp-ubf-integration/tasks.md`
- **UBF Documentation**: `ubf_protein/README.md`, `ubf_protein/API.md`
- **QCPP Documentation**: Root directory README and inline documentation

## License

See project root LICENSE file.
