# Campaign Configuration Files

This directory contains pre-configured campaign setups for different use cases. The configuration management system (`campaign_config.py`) provides validation, merging, and preset management.

## Configuration Management System

The validation framework includes a comprehensive configuration management system with:
- **Presets**: 7 pre-configured campaign types
- **Validation**: Comprehensive parameter validation with detailed error messages
- **File Support**: JSON and YAML configuration files
- **Merging**: Combine configurations from multiple sources
- **CLI Integration**: Seamless integration with `run_validation_campaign.py`
- **Python API**: Programmatic configuration management

### Quick Start with Python API

```python
from validation.campaign_config import CampaignConfigManager

# Initialize manager
manager = CampaignConfigManager()

# Load from preset
config = manager.get_preset('high_accuracy')

# Load from file
config = manager.load('./validation/configs/default_campaign.json')

# Override settings
config = manager.override(config, num_agents=15, random_seed=123)

# Validate
manager.validate(config)

# Save custom configuration
manager.save(config, './my_custom_config.json')
```

## Available Configurations

### `default_campaign.json`
**Purpose**: Standard validation campaign with balanced settings.

- **Target Proteins**: 60
- **QCPP Integration**: Enabled
- **Parallelism**: 3 concurrent tests
- **Agents**: 10 per protein
- **Iterations**: 1000 per agent
- **Quality Gate**: 60% success rate
- **Use Case**: General-purpose validation, production runs

### `fast_campaign.json`
**Purpose**: Quick validation for rapid iteration and testing.

- **Target Proteins**: 50
- **QCPP Integration**: Enabled
- **Parallelism**: 5 concurrent tests (higher throughput)
- **Agents**: 5 per protein (fewer agents)
- **Iterations**: 500 per agent (fewer iterations)
- **Quality Gate**: 55% success rate (more lenient)
- **Use Case**: Development, debugging, quick feasibility checks

### `high_accuracy_campaign.json`
**Purpose**: Maximum accuracy for publication-quality results.

- **Target Proteins**: 75 (maximum recommended)
- **QCPP Integration**: Enabled
- **Parallelism**: 2 concurrent tests (more resources per test)
- **Agents**: 20 per protein (extensive exploration)
- **Iterations**: 2000 per agent (thorough search)
- **Quality Gate**: 65% success rate (stricter)
- **RMSD Threshold**: 6.0 Å (stricter failure classification)
- **Random Seed**: 42 (reproducible)
- **Use Case**: Final validation, research publications, comprehensive analysis

### `baseline_campaign.json`
**Purpose**: Baseline comparison without QCPP integration.

- **Target Proteins**: 60
- **QCPP Integration**: **Disabled** (UBF-only mode)
- **Parallelism**: 3 concurrent tests
- **Agents**: 10 per protein
- **Iterations**: 1000 per agent
- **Quality Gate**: 60% success rate
- **Use Case**: Comparative analysis, ablation studies, quantifying QCPP impact

### `benchmark_campaign.json`
**Purpose**: Optimized for comparative benchmarking.

- **Target Proteins**: 30 (smaller set for paired comparison)
- **QCPP Integration**: Enabled
- **Parallelism**: 2 concurrent tests
- **Agents**: 10 per protein
- **Iterations**: 1000 per agent
- **Random Seed**: 42 (reproducible)
- **Use Case**: Running side-by-side baseline vs integrated benchmarks

## Usage

### Using a configuration file:

```bash
# Run with default configuration
python validation/run_validation_campaign.py --config validation/configs/default_campaign.json --batch

# Run high-accuracy campaign interactively
python validation/run_validation_campaign.py --config validation/configs/high_accuracy_campaign.json --interactive

# Run fast campaign for testing
python validation/run_validation_campaign.py --config validation/configs/fast_campaign.json --batch

# Run baseline campaign for comparison
python validation/run_validation_campaign.py --config validation/configs/baseline_campaign.json --batch
```

### Overriding configuration values:

Command-line arguments take precedence over configuration file values:

```bash
# Load config but override agent count
python validation/run_validation_campaign.py \
    --config validation/configs/default_campaign.json \
    --batch \
    --agents 15 \
    --parallel 5

# Load config but change output directory
python validation/run_validation_campaign.py \
    --config validation/configs/high_accuracy_campaign.json \
    --batch \
    --output ./my_custom_results
```

### Saving current configuration:

```bash
# Create a custom configuration from command-line args
python validation/run_validation_campaign.py \
    --batch \
    --proteins 50 \
    --agents 15 \
    --iterations 1500 \
    --parallel 4 \
    --seed 123 \
    --save-config ./validation/configs/my_custom.json
```

## Configuration Parameters

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| `target_protein_count` | int | Number of proteins to test | 50-75 recommended |
| `enable_qcpp` | bool | Enable QCPP integration | true/false |
| `max_parallel_tests` | int | Max concurrent executions | 1-10 |
| `num_agents` | int | Agents per protein | 5-20 typical |
| `iterations_per_agent` | int | Iterations per agent | 500-2000 typical |
| `checkpoint_interval` | int | Save checkpoint every N tests | 3-10 |
| `quality_gate_threshold` | float | Phase 1 success rate threshold | 0.5-0.7 |
| `failure_rmsd_threshold` | float | RMSD threshold for failure (Å) | 6.0-10.0 |
| `timeout_multiplier` | float | Timeout as multiple of expected time | 1.5-3.0 |
| `random_seed` | int/null | Random seed (null for random) | Any integer |
| `output_dir` | string | Output directory path | Any valid path |

## Performance Considerations

### Fast Configuration (fast_campaign.json)
- **Pros**: Fastest execution, ideal for testing
- **Cons**: Lower accuracy, may miss difficult structures
- **Runtime**: ~1-2 hours for 50 proteins

### Default Configuration (default_campaign.json)
- **Pros**: Balanced performance and accuracy
- **Cons**: Moderate runtime
- **Runtime**: ~3-4 hours for 60 proteins

### High-Accuracy Configuration (high_accuracy_campaign.json)
- **Pros**: Best accuracy, publication-quality
- **Cons**: Longest runtime, highest resource usage
- **Runtime**: ~8-12 hours for 75 proteins

## Custom Configuration Tips

1. **For Development**: Use `fast_campaign.json` with fewer proteins (30-40)
2. **For Production**: Use `default_campaign.json` or customize based on needs
3. **For Publication**: Use `high_accuracy_campaign.json` with reproducible seed
4. **For Benchmarking**: Use `benchmark_campaign.json` with matched protein sets
5. **For Ablation Studies**: Use `baseline_campaign.json` to test without QCPP

## Troubleshooting

**Memory Issues**: Reduce `num_agents` or `max_parallel_tests`
**Slow Performance**: Increase `max_parallel_tests` or reduce `iterations_per_agent`
**Low Success Rate**: Increase `num_agents`, `iterations_per_agent`, or lower `quality_gate_threshold`
**Quality Gate Failures**: Review failure analysis reports, adjust parameters based on patterns

## See Also

- [CLI Documentation](../run_validation_campaign.py) - Command-line interface usage
- [Campaign Design](../../../.kiro/specs/large-scale-protein-validation/design.md) - Architecture overview
- [Tasks](../../../.kiro/specs/large-scale-protein-validation/tasks.md) - Implementation progress
