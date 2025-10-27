"""
Example Usage of the Validation Campaign CLI

This script demonstrates various ways to use the command-line interface
for running large-scale protein validation campaigns.

For full documentation, run:
    python validation/run_validation_campaign.py --help
"""

# ============================================================================
# Example 1: Interactive Mode with Default Settings
# ============================================================================

# This runs in interactive mode, prompting for approval after each phase.
# Useful for monitoring progress and making decisions at each quality gate.

# Command:
# python validation/run_validation_campaign.py --interactive

# Output:
# - Phase-by-phase prompts
# - Real-time results display
# - Quality gate warnings
# - Option to stop at any point


# ============================================================================
# Example 2: Batch Mode with Custom Parameters
# ============================================================================

# Fully automated execution with custom parameters.
# Best for production runs when you know your configuration.

# Command:
# python validation/run_validation_campaign.py \
#     --batch \
#     --proteins 60 \
#     --agents 10 \
#     --iterations 1000 \
#     --parallel 3 \
#     --output ./my_campaign_results \
#     --log-level INFO \
#     --log-file ./campaign.log

# This will:
# - Test 60 proteins
# - Use 10 agents per protein
# - Run 1000 iterations per agent
# - Execute 3 tests in parallel
# - Save results to ./my_campaign_results
# - Log to both console and ./campaign.log


# ============================================================================
# Example 3: Using Configuration Files
# ============================================================================

# Load settings from pre-configured JSON file.
# Simplifies command line and enables reproducible runs.

# Command:
# python validation/run_validation_campaign.py \
#     --config validation/configs/default_campaign.json \
#     --batch

# Available configs:
# - default_campaign.json: Balanced settings
# - fast_campaign.json: Quick validation
# - high_accuracy_campaign.json: Publication quality
# - baseline_campaign.json: UBF-only (no QCPP)
# - benchmark_campaign.json: Comparative benchmarking


# ============================================================================
# Example 4: Overriding Configuration File Settings
# ============================================================================

# Load config but override specific parameters.
# Command-line args take precedence over config file.

# Command:
# python validation/run_validation_campaign.py \
#     --config validation/configs/high_accuracy_campaign.json \
#     --batch \
#     --agents 15 \
#     --parallel 4 \
#     --seed 123

# This loads high_accuracy_campaign.json but changes:
# - agents from 20 to 15
# - parallel from 2 to 4
# - seed from 42 to 123


# ============================================================================
# Example 5: Saving Configuration
# ============================================================================

# Create a custom config and save it for reuse.

# Command:
# python validation/run_validation_campaign.py \
#     --batch \
#     --proteins 50 \
#     --agents 12 \
#     --iterations 1500 \
#     --parallel 4 \
#     --quality-threshold 0.65 \
#     --seed 42 \
#     --output ./custom_results \
#     --save-config ./validation/configs/my_custom.json

# This creates my_custom.json with your settings (doesn't run campaign)


# ============================================================================
# Example 6: Resuming from Checkpoint
# ============================================================================

# Resume a campaign from a saved checkpoint.
# Useful if campaign was interrupted or you want to continue with same config.

# Command:
# python validation/run_validation_campaign.py \
#     --resume ./campaign_results/checkpoint_latest.json \
#     --batch

# Note: Currently loads configuration from checkpoint.
# Full state restoration (completed tests, progress) planned for future.


# ============================================================================
# Example 7: Comparative Benchmarking Mode
# ============================================================================

# Run both baseline (UBF-only) and integrated (UBF+QCPP) modes side-by-side.
# Generates statistical comparison report.

# Command:
# python validation/run_validation_campaign.py \
#     --benchmark \
#     --proteins 30 \
#     --agents 10 \
#     --iterations 1000 \
#     --parallel 2 \
#     --output ./benchmark_results

# This will:
# - Select 30 proteins
# - Run baseline tests (QCPP disabled)
# - Run integrated tests (QCPP enabled)
# - Calculate performance deltas
# - Perform statistical significance tests
# - Generate JSON and Markdown reports


# ============================================================================
# Example 8: High-Accuracy Publication Run
# ============================================================================

# Complete workflow for publication-quality validation.

# Step 1: Run high-accuracy campaign
# python validation/run_validation_campaign.py \
#     --config validation/configs/high_accuracy_campaign.json \
#     --batch \
#     --log-file ./publication_run.log

# Step 2: Review results
# - Check ./high_accuracy_results/final_report.md
# - Review statistical analysis
# - Check failure analysis for any issues

# Step 3: If needed, run additional analyses
# - Use failure_analyzer on specific proteins
# - Generate custom visualizations
# - Export data for external plotting tools


# ============================================================================
# Example 9: Development and Testing Workflow
# ============================================================================

# Fast iteration during development.

# Command:
# python validation/run_validation_campaign.py \
#     --config validation/configs/fast_campaign.json \
#     --batch \
#     --proteins 20 \
#     --parallel 5 \
#     --log-level DEBUG

# This gives:
# - Only 20 proteins (faster)
# - High parallelism (5 concurrent)
# - Fast agents/iterations (from config)
# - DEBUG logging for troubleshooting


# ============================================================================
# Example 10: Baseline Comparison Study
# ============================================================================

# Compare QCPP-enabled vs baseline performance.

# Step 1: Run integrated campaign
# python validation/run_validation_campaign.py \
#     --batch \
#     --proteins 60 \
#     --agents 10 \
#     --iterations 1000 \
#     --output ./integrated_results \
#     --seed 42

# Step 2: Run baseline campaign (same settings, no QCPP)
# python validation/run_validation_campaign.py \
#     --batch \
#     --proteins 60 \
#     --agents 10 \
#     --iterations 1000 \
#     --no-qcpp \
#     --output ./baseline_results \
#     --seed 42

# Step 3: Compare results manually or use benchmark mode
# python validation/run_validation_campaign.py \
#     --benchmark \
#     --proteins 60 \
#     --agents 10 \
#     --iterations 1000 \
#     --output ./comparison_results


# ============================================================================
# Tips and Best Practices
# ============================================================================

"""
1. INTERACTIVE VS BATCH MODE
   - Use --interactive for:
     * First runs with new parameters
     * Monitoring critical campaigns
     * Learning the system
   - Use --batch for:
     * Production runs
     * Automated pipelines
     * Unattended execution

2. LOGGING
   - Always use --log-file for production runs
   - Use --log-level DEBUG only during development
   - Log files help diagnose issues later

3. CONFIGURATION FILES
   - Use configs for reproducibility
   - Start with provided configs, customize as needed
   - Save successful configs with --save-config

4. CHECKPOINTING
   - Campaign auto-saves every N tests (default: 5)
   - Use --checkpoint-interval to adjust frequency
   - Resume with --resume if interrupted

5. RESOURCE MANAGEMENT
   - Monitor system resources
   - Adjust --parallel based on CPU/memory
   - Reduce --agents if memory is constrained

6. RANDOM SEEDS
   - Always use --seed for reproducible research
   - Omit seed for exploratory runs
   - Document seed in publications

7. QUALITY GATES
   - Default threshold is 60% (Phase 1)
   - Adjust with --quality-threshold if needed
   - Interactive mode allows manual override

8. OUTPUT ORGANIZATION
   - Use descriptive --output directories
   - Include date/experiment info in path
   - Keep baseline and integrated results separate

9. BENCHMARKING
   - Use smaller protein sets (20-30) for benchmarks
   - Always use same seed for paired comparisons
   - Review statistical significance carefully

10. TROUBLESHOOTING
    - Check logs first
    - Use DEBUG logging to diagnose issues
    - Review failure analysis reports
    - Start with fast config to test changes
"""

# ============================================================================
# Common Error Messages and Solutions
# ============================================================================

"""
ERROR: "Phase 1 failed quality gate"
SOLUTION: Review failure analysis report, consider:
  - Increasing --agents or --iterations
  - Adjusting protein selection criteria
  - Lowering --quality-threshold (with caution)

ERROR: "Out of memory"
SOLUTION: Reduce resource usage:
  - Decrease --parallel (fewer concurrent tests)
  - Decrease --agents (fewer agents per test)
  - Test fewer proteins at once

ERROR: "Checkpoint file not found"
SOLUTION: Ensure checkpoint exists:
  - Check --output directory for checkpoints
  - Verify checkpoint file path is correct
  - Campaign may not have reached checkpoint interval

ERROR: "Configuration file invalid"
SOLUTION: Validate JSON syntax:
  - Use JSON validator
  - Check for missing commas or brackets
  - Ensure all required fields present

ERROR: "PDB structure download failed"
SOLUTION: Check network and PDB availability:
  - Verify internet connection
  - RCSB PDB may be temporarily unavailable
  - Try again later or use cached structures
"""
