# Running a 10-Protein Test Campaign

## Quick Start

### Option 1: Simple Command (Recommended for Testing)
```bash
python run_10_protein_test.py
```

This will:
- Test 10 diverse proteins from the curated list
- Use QCPP-UBF integration
- Run 3 tests in parallel
- Each test: 10 agents × 1000 iterations
- Save results to `./campaign_10_proteins/`

### Option 2: Interactive Mode (Step-by-Step Control)
```bash
python run_10_protein_test.py --interactive
```

This will:
- Ask for approval after each phase
- Show progress and results before continuing
- Allow parameter adjustments between phases

### Option 3: Direct Campaign Runner
```bash
python validation/run_validation_campaign.py --config validation/configs/test_10_proteins.json
```

## Configuration Details

The 10-protein test uses this configuration (`validation/configs/test_10_proteins.json`):

```json
{
  "target_protein_count": 10,      // Test 10 proteins
  "enable_qcpp": true,              // Full QCPP integration
  "max_parallel_tests": 3,          // Run 3 proteins at once
  "num_agents": 10,                 // 10 agents per protein
  "iterations_per_agent": 1000,     // 1000 iterations each
  "checkpoint_interval": 5,         // Save every 5 proteins
  "quality_gate_threshold": 0.60,   // 60% success for Phase 1
  "failure_rmsd_threshold": 8.0,    // RMSD > 8Å is failure
  "timeout_multiplier": 2.0,        // 2x expected time
  "random_seed": 42,                // Reproducible results
  "output_dir": "./campaign_10_proteins"
}
```

## Expected Timeline

**Per Protein:**
- Small (<50 residues): 3-5 minutes
- Medium (50-150): 5-10 minutes  
- Large (>150): 10-20 minutes

**Total Campaign:**
- Sequential: ~60-120 minutes
- With 3 parallel: ~20-40 minutes

## Protein Selection

The campaign automatically selects 10 diverse proteins:
- 3-4 tiny proteins (<50 residues)
- 3-4 small proteins (50-100 residues)
- 2-3 medium proteins (100-200 residues)

Example proteins that might be selected:
- 1VII (Villin, 36 residues)
- 1CRN (Crambin, 46 residues)
- 1UBQ (Ubiquitin, 76 residues)
- 1LYZ (Lysozyme, 129 residues)
- etc.

## Phase Structure

### Phase 1: Initial Testing (3 proteins)
- Easiest proteins
- Quality gate: 60% success rate required
- If fails: Analyze and adjust parameters

### Phase 2: Expansion (3 proteins)
- Mixed difficulty
- Continue if Phase 1 passed

### Phase 3: Comprehensive (4 proteins)
- Diverse characteristics
- Final validation

## Output Files

All results saved to `./campaign_10_proteins/`:

```
campaign_10_proteins/
├── results/
│   ├── test_results/          # Individual protein results (JSON)
│   ├── reports/               # Phase reports (Markdown)
│   └── structures/            # Predicted structures (PDB)
├── logs/
│   └── campaign.log          # Execution log
├── checkpoints/
│   └── checkpoint_*.json     # Resume points
└── final_report.md           # Complete analysis
```

## Monitoring Progress

The campaign will show:
- Real-time progress bar
- Current protein being tested
- Success/failure status
- Running averages (RMSD, energy, GDT-TS)
- Estimated completion time

## Resume After Interruption

If the campaign is interrupted:
```bash
python run_10_protein_test.py --resume
```

This will:
- Load the last checkpoint
- Continue from where it stopped
- Preserve all previous results

## Customizing the Test

### Change Number of Proteins
Edit `validation/configs/test_10_proteins.json`:
```json
{
  "target_protein_count": 20,  // Change to 20 proteins
  ...
}
```

### Speed vs. Accuracy Trade-off

**Faster (for quick testing):**
```json
{
  "num_agents": 5,
  "iterations_per_agent": 500,
  "max_parallel_tests": 5
}
```

**Higher Accuracy (for publication):**
```json
{
  "num_agents": 20,
  "iterations_per_agent": 2000,
  "max_parallel_tests": 2
}
```

### Test Specific Proteins

Create a custom protein list file and modify the campaign code to use it.

## Comparing Results

After completion, compare with baseline:
```bash
python validation/run_validation_campaign.py \
    --config validation/configs/test_10_proteins.json \
    --benchmark
```

This will run both QCPP-integrated and baseline (UBF-only) versions for comparison.

## Troubleshooting

### Campaign Fails to Start
- Check Python environment: `python --version`
- Verify dependencies: `pip install -r requirements_qcpp.txt`
- Check disk space: Need ~1GB free

### Individual Test Failures
- Check logs in `campaign_10_proteins/logs/`
- Review failure analysis in phase reports
- Adjust parameters if needed

### Out of Memory
- Reduce `max_parallel_tests` to 1 or 2
- Reduce `num_agents` to 5
- Close other applications

### Too Slow
- Increase `max_parallel_tests` (if you have RAM)
- Reduce `iterations_per_agent` to 500
- Use `fast_campaign.json` config instead

## Next Steps

After successful 10-protein test:

1. **Analyze Results:**
   ```bash
   cat campaign_10_proteins/final_report.md
   ```

2. **Scale Up:**
   - Use `validation/configs/default_campaign.json` for 60 proteins
   - Use `validation/configs/high_accuracy_campaign.json` for publication

3. **Compare with Baseline:**
   - Run benchmark mode to compare QCPP vs non-QCPP

## Questions?

- See full documentation: `validation/configs/README.md`
- Check examples: `ubf_protein/examples/README_INTEGRATED.md`
- Review API: `ubf_protein/API.md`
