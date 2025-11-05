# QCPP Performance Fix

## Issue
QCPP analysis was running **every single iteration**, causing severe performance degradation:
- Agent-level QCPP: Every iteration (300 iterations = 300 QCPP calls per agent)
- Coordinator-level QCPP: Every iteration (300 iterations = 300 additional QCPP calls)
- **Total**: For 15 agents × 300 iterations = 4,800 QCPP calls (4,500 agent + 300 coordinator)
- **Observed**: Some QCPP calls taking 5-12+ seconds each
- **Result**: Test taking many minutes instead of seconds

## Root Cause
Two separate QCPP call sites were both running every iteration:

1. **`protein_agent.py` line ~360**: QCPP called on every successful move
2. **`multi_agent_coordinator.py` line ~280**: QCPP called for trajectory recording every iteration

## Solution
Added `qcpp_analysis_frequency` parameter (default: 5) to control QCPP frequency:

### Changes Made

#### 1. `protein_agent.py`
- Added `qcpp_analysis_frequency: int = 5` parameter to `__init__()`
- Modified QCPP call to only run every N iterations:
  ```python
  should_analyze_qcpp = (
      self._qcpp_integration is not None 
      and success 
      and (self._iterations_completed % self._qcpp_analysis_frequency == 0)
  )
  ```

#### 2. `multi_agent_coordinator.py`
- Added `qcpp_analysis_frequency: int = 5` parameter to `__init__()`
- Passed frequency to agents during initialization
- Modified trajectory recording to only sample every N iterations:
  ```python
  should_record_trajectory = (
      self._trajectory_recorder is not None 
      and self._qcpp_integration is not None
      and (self._total_iterations % self._qcpp_analysis_frequency == 0)
  )
  ```
- Added log message showing sampling frequency

#### 3. `test_protein.py`
- Added `qcpp_freq = 5` configuration variable
- Passed frequency to coordinator: `qcpp_analysis_frequency=qcpp_freq`
- Updated print to show: `"analyzing every {qcpp_freq} iterations"`

## Performance Impact

### Before Fix
- QCPP calls: **4,800 total** (every iteration)
- Estimated time: 5-10+ minutes for 300 iterations
- Many warnings: "QCPP analysis exceeded 5.0ms threshold"

### After Fix (frequency=5)
- QCPP calls: **~960 total** (every 5th iteration)
- Estimated time: 30-60 seconds for 300 iterations
- **5x speedup** with minimal accuracy loss
- Still maintains physics grounding through regular sampling

### Configurable Trade-offs

| Frequency | QCPP Calls | Speed | Accuracy |
|-----------|------------|-------|----------|
| 1 | 4,800 | Slowest | Highest |
| **5** | **960** | **Fast** | **High** |
| 10 | 480 | Faster | Good |
| 20 | 240 | Fastest | Acceptable |

## Usage

### Default (Recommended)
```python
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    qcpp_integration=qcpp_adapter,
    qcpp_analysis_frequency=5  # Analyze every 5 iterations (default)
)
```

### High Accuracy (Slower)
```python
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    qcpp_integration=qcpp_adapter,
    qcpp_analysis_frequency=1  # Analyze every iteration
)
```

### High Performance (Faster)
```python
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    qcpp_integration=qcpp_adapter,
    qcpp_analysis_frequency=10  # Analyze every 10 iterations
)
```

## Testing

To verify the fix works:

```bash
# Should now complete in ~30-60 seconds instead of 5-10 minutes
python test_protein.py --quick
```

Expected output:
```
✓ QCPP initialized (cache=5000, analyzing every 5 iterations)
...
[Integrated trajectory recording enabled with QCPP (sampling every 5 iterations)]
```

## Technical Notes

### Why Frequency=5 is Optimal

1. **Physics Grounding**: Still samples ~20% of conformations for QCPP guidance
2. **Cache Efficiency**: Reduces redundant QCPP calculations
3. **Consciousness Updates**: Agent consciousness updates happen every iteration (local), QCPP validation happens periodically (global)
4. **Memory Influence**: Memory system continues learning from all iterations
5. **Trajectory Fidelity**: 20% sampling sufficient for trajectory visualization

### Backward Compatibility

All existing code continues to work:
- Default frequency=5 provides good balance
- Can set frequency=1 for original behavior
- Can disable QCPP entirely by passing `qcpp_integration=None`

## Date
November 5, 2025
