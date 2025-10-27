# Configuration Management Quick Reference

## Quick Start (30 seconds)

```python
from validation.campaign_config import CampaignConfigManager

manager = CampaignConfigManager()
config = manager.get_preset('default')
manager.validate(config)
# Use config with campaign
```

## Available Presets

| Preset | Proteins | Agents | Iterations | Best For |
|--------|----------|--------|------------|----------|
| **default** | 60 | 10 | 1000 | General validation |
| **fast** | 50 | 5 | 500 | Development/testing |
| **high_accuracy** | 75 | 20 | 2000 | Publications |
| **baseline** | 60 | 10 | 1000 | UBF-only (no QCPP) |
| **benchmark** | 30 | 10 | 1000 | Comparative analysis |
| **development** | 20 | 5 | 300 | Rapid prototyping |
| **production** | 60 | 12 | 1500 | Production runs |

## Common Tasks

### Load Preset
```python
config = manager.get_preset('high_accuracy')
```

### Load from File
```python
config = manager.load('./my_config.json')
```

### Override Settings
```python
config = manager.override(
    base_config,
    num_agents=15,
    random_seed=42
)
```

### Validate
```python
manager.validate(config)  # Raises ValidationError if invalid
```

### Save
```python
manager.save(config, './my_config.json')
```

### List Presets
```python
presets = manager.list_presets()
# ['default', 'fast', 'high_accuracy', ...]
```

## CLI Integration

```bash
# Use preset via CLI
python validation/run_validation_campaign.py \
    --config validation/configs/default_campaign.json \
    --batch

# Override from CLI
python validation/run_validation_campaign.py \
    --config validation/configs/high_accuracy_campaign.json \
    --batch \
    --agents 15 \
    --seed 123
```

## Parameter Ranges

| Parameter | Minimum | Typical | Maximum |
|-----------|---------|---------|---------|
| `target_protein_count` | 20 | 50-75 | 100+ |
| `max_parallel_tests` | 1 | 3-5 | 10 |
| `num_agents` | 5 | 10-12 | 20 |
| `iterations_per_agent` | 500 | 1000-1500 | 2000 |
| `quality_gate_threshold` | 0.50 | 0.60 | 0.70 |
| `failure_rmsd_threshold` | 5.0Å | 7-8Å | 10Å |

## Validation Errors vs Warnings

### Errors (will fail):
- `num_agents < 1`
- `iterations_per_agent < 100`
- `quality_gate_threshold` not in (0, 1]
- `timeout_multiplier <= 0`

### Warnings (will succeed):
- `target_protein_count < 20` (too small for statistics)
- `num_agents > 50` (excessive)
- `quality_gate_threshold < 0.4` (very lenient)
- Estimated memory > 8GB

## Preset Selection Guide

### For Development
```python
config = manager.get_preset('development')  # 20 proteins, fast
```

### For Testing Features
```python
config = manager.get_preset('fast')  # 50 proteins, quick
```

### For Production
```python
config = manager.get_preset('production')  # 60 proteins, robust
```

### For Publications
```python
config = manager.get_preset('high_accuracy')  # 75 proteins, thorough
config = manager.override(config, random_seed=42)  # Reproducible
```

### For Benchmarking
```python
config = manager.get_preset('benchmark')  # 30 proteins, paired tests
```

### For Ablation Studies
```python
config = manager.get_preset('baseline')  # UBF-only, no QCPP
```

## Complete Workflow Example

```python
from validation.campaign_config import CampaignConfigManager
from validation.large_scale_validation_campaign import LargeScaleValidationCampaign

# 1. Setup
manager = CampaignConfigManager()

# 2. Load and customize
config = manager.get_preset('high_accuracy')
config = manager.override(
    config,
    target_protein_count=60,
    num_agents=15,
    random_seed=2025,
    output_dir='./my_research_results'
)

# 3. Validate
manager.validate(config)

# 4. Save for reproducibility
manager.save(config, './my_research_config.json')

# 5. Run campaign
campaign = LargeScaleValidationCampaign(config=config)
results = campaign.run_campaign()

print(f"Success rate: {results.overall_success_rate:.1f}%")
print(f"Report: {results.final_report_path}")
```

## Common Patterns

### Reproducible Research
```python
config = manager.get_preset('high_accuracy')
config = manager.override(config, random_seed=42)
manager.save(config, './reproducible_config.json')
```

### Memory-Constrained Systems
```python
config = manager.get_preset('default')
config = manager.override(
    config,
    max_parallel_tests=2,  # Fewer concurrent
    num_agents=8           # Fewer agents
)
```

### Maximum Throughput
```python
config = manager.get_preset('fast')
config = manager.override(
    config,
    max_parallel_tests=8,  # High concurrency
    iterations_per_agent=300  # Quick iterations
)
```

### Strict Quality Control
```python
config = manager.get_preset('high_accuracy')
config = manager.override(
    config,
    quality_gate_threshold=0.70,  # 70% success required
    failure_rmsd_threshold=5.0    # Strict RMSD cutoff
)
```

## Error Handling

```python
from validation.campaign_config import ValidationError

try:
    config = manager.get_preset('high_accuracy')
    config = manager.override(config, num_agents=15)
    manager.validate(config)
except ValueError as e:
    print(f"Invalid preset or parameter: {e}")
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
```

## Non-Strict Validation

```python
# Don't raise exception, just log warnings
is_valid = manager.validate(config, strict=False)
if is_valid:
    print("Configuration valid (check logs for warnings)")
else:
    print("Configuration has errors (check logs)")
```

## YAML Support (Optional)

```bash
# Install PyYAML
pip install pyyaml
```

```python
# Save as YAML
manager.save(config, './my_config.yaml', format='yaml')

# Load YAML
config = manager.load('./my_config.yaml')
```

## See Also

- **Full Documentation**: `validation/configs/README.md`
- **Examples**: `validation/examples/config_management_examples.py`
- **API Reference**: `validation/campaign_config.py`
- **CLI Usage**: `validation/run_validation_campaign.py --help`
