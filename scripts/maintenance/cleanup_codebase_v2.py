"""
Comprehensive Codebase Cleanup Script v2
Cleans up remaining files after initial cleanup
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

# Get the root directory
ROOT_DIR = Path(__file__).parent.resolve()

# Define cleanup operations
CLEANUP_OPERATIONS = {
    # Test files from root -> scripts/experiments
    "test_files": {
        "source_files": [
            "test_adaptive_detection.py",
            "test_forced_revisit.py",
            "test_geometric_attractors.py",
            "test_geometric_targeting.py",
            "test_hash_function.py",
            "test_registry_wiring.py",
            "test_thz_opt_in.py",
        ],
        "destination": "scripts/experiments",
        "description": "Test scripts from root"
    },
    
    # JSON result files
    "json_results": {
        "source_files": [
            "determinism_7res_20251105_034318.json",
            "phi_reanalysis_results.json",
            "test_determinism_quick.json",
            "test_results.json",
            "ubiquitin_adaptive.json",
            "ubiquitin_determinism.json",
            "validation_1UBQ.json",
            "validation_1VII.json",
        ],
        "destination": "results/test_results",
        "description": "JSON result files from root"
    },
    
    # Markdown documentation files
    "md_docs": {
        "source_files": [
            "COMPLETE_DATA_FLOW_ANALYSIS.md",
            "FINAL_ANALYSIS_SUMMARY.md",
            "geometric_attractor_analysis.md",
            "GEOMETRIC_INTEGRITY_RESEARCH_REPORT.md",
            "GEOMETRIC_TARGETING_IMPLEMENTATION.md",
            "GEOMETRIC_TARGETING_PROPOSAL.md",
            "GEOMETRIC_TARGETING_SUMMARY.md",
            "QCPP_OPTIMIZATION_COMPLETE.md",
            "QCPP_PERFORMANCE_FIX.md",
            "THz_OPT_IN_REFACTOR.md",
            "VALIDATION_GUIDE.md",
            "VALIDATION_IMPLEMENTATION_SUMMARY.md",
        ],
        "destination": "docs/analysis",
        "description": "Markdown documentation from root"
    },
    
    # Utility/analysis scripts
    "utility_scripts": {
        "source_files": [
            "analyze_predicted_structure.py",
            "compute_true_rmsd.py",
            "debug_energy_landscape.py",
            "quick_validation_test.py",
            "reanalyze_phi_on_predictions.py",
            "run_10_protein_test.py",
            "run_20_protein_phi_test.py",
            "run_field_guided_test.py",
            "validate_geometric_hypothesis.py",
            "visualize_results.py",
        ],
        "destination": "scripts/utilities",
        "description": "Utility and analysis scripts"
    },
    
    # Temp and generated files to delete
    "temp_files": {
        "source_files": [
            "temp.pdb",
            "geometric_hypothesis_results.png",
        ],
        "destination": None,  # None means delete
        "description": "Temporary files to delete"
    },
    
    # Old backup directory
    "old_backups": {
        "source_files": [
            "backup_20251027_181101",
        ],
        "destination": None,  # None means delete
        "description": "Old backup directories to remove"
    }
}

# Files to keep in root (don't move these)
ROOT_ESSENTIAL_FILES = {
    "run_analysis.py",  # Main QCPP entry point
    "test_protein.py",  # Main test entry point
    "setup.py",
    "pyrightconfig.json",
    "requirements_qcpp.txt",
    ".gitignore",
    "README.md",
    "PUBLICATION_DRAFT.md",  # Keep in root for visibility
    "CLEANUP_REPORT.md",
    "cleanup_workspace.py",
    "cleanup_codebase_v2.py",
    "install_biopython.bat",
    "run_validation_tests.bat",
    "setup_build_tools.bat",
}

def create_backup():
    """Create a backup before cleanup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT_DIR / f"backup_{timestamp}"
    
    print(f"\n📦 Creating backup at: {backup_dir}")
    
    # Only backup files we're about to move
    backup_files = []
    for operation in CLEANUP_OPERATIONS.values():
        if operation["destination"] is not None:  # Only backup files being moved
            backup_files.extend(operation["source_files"])
    
    if backup_files:
        backup_dir.mkdir(exist_ok=True)
        for file in backup_files:
            src = ROOT_DIR / file
            if src.exists():
                dst = backup_dir / file
                if src.is_file():
                    shutil.copy2(src, dst)
                elif src.is_dir():
                    shutil.copytree(src, dst)
                print(f"  ✓ Backed up: {file}")
    
    return backup_dir

def move_files(operation_name, config):
    """Move files according to configuration"""
    source_files = config["source_files"]
    destination = config["destination"]
    description = config["description"]
    
    print(f"\n📁 {description}")
    print(f"   Operation: {operation_name}")
    
    moved_count = 0
    skipped_count = 0
    
    for file_name in source_files:
        src_path = ROOT_DIR / file_name
        
        if not src_path.exists():
            print(f"  ⏭️  Skip (not found): {file_name}")
            skipped_count += 1
            continue
        
        if destination is None:
            # Delete the file/directory
            try:
                if src_path.is_file():
                    src_path.unlink()
                    print(f"  🗑️  Deleted file: {file_name}")
                elif src_path.is_dir():
                    shutil.rmtree(src_path)
                    print(f"  🗑️  Deleted directory: {file_name}")
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Error deleting {file_name}: {e}")
        else:
            # Move the file/directory
            dst_dir = ROOT_DIR / destination
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / file_name
            
            try:
                if dst_path.exists():
                    print(f"  ⚠️  Destination exists, skipping: {file_name}")
                    skipped_count += 1
                    continue
                
                shutil.move(str(src_path), str(dst_path))
                print(f"  ✓ Moved: {file_name} -> {destination}/")
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Error moving {file_name}: {e}")
    
    print(f"   Summary: {moved_count} processed, {skipped_count} skipped")
    return moved_count, skipped_count

def clean_pycache():
    """Remove all __pycache__ directories"""
    print(f"\n🧹 Cleaning __pycache__ directories...")
    count = 0
    for pycache_dir in ROOT_DIR.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            print(f"  ✓ Removed: {pycache_dir.relative_to(ROOT_DIR)}")
            count += 1
        except Exception as e:
            print(f"  ❌ Error removing {pycache_dir}: {e}")
    print(f"   Removed {count} __pycache__ directories")

def clean_old_checkpoints():
    """Clean up old checkpoint files, keeping only recent ones"""
    print(f"\n🔍 Analyzing checkpoint directories...")
    
    checkpoint_dirs = [
        ROOT_DIR / "checkpoints",
        ROOT_DIR / "ubf_protein" / "checkpoints",
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue
        
        print(f"\n  Directory: {checkpoint_dir.relative_to(ROOT_DIR)}")
        
        # Get all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
        
        if not checkpoint_files:
            print(f"    No checkpoint files found")
            continue
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Keep the 3 most recent, delete the rest
        keep_count = 3
        to_keep = checkpoint_files[:keep_count]
        to_delete = checkpoint_files[keep_count:]
        
        print(f"    Found {len(checkpoint_files)} checkpoints")
        print(f"    Keeping {len(to_keep)} most recent")
        
        if to_delete:
            print(f"    Deleting {len(to_delete)} old checkpoints:")
            for f in to_delete:
                try:
                    f.unlink()
                    print(f"      ✓ Deleted: {f.name}")
                except Exception as e:
                    print(f"      ❌ Error: {e}")

def update_gitignore():
    """Create or update .gitignore with comprehensive patterns"""
    gitignore_path = ROOT_DIR / ".gitignore"
    
    additional_patterns = [
        "",
        "# Cleanup v2 additions",
        "# Python cache",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        ".pytest_cache/",
        "",
        "# Virtual environments",
        "venv/",
        ".venv/",
        "myvenv/",
        "pypy_env/",
        "",
        "# Results and outputs",
        "results/**/*.json",
        "test_results.json",
        "validation_*.json",
        "ubiquitin_*.json",
        "*_results.json",
        "",
        "# Checkpoints (keep only recent)",
        "checkpoints/checkpoint_*.json",
        "ubf_protein/checkpoints/checkpoint_*.json",
        "",
        "# Temporary files",
        "*.pdb",
        "temp.*",
        "*.tmp",
        "",
        "# Backup directories",
        "backup_*/",
        "",
        "# Logs",
        "*.log",
        "",
        "# IDE/Editor",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "",
        "# OS",
        ".DS_Store",
        "Thumbs.db",
        "",
        "# Distribution",
        "*.egg-info/",
        "dist/",
        "build/",
    ]
    
    print(f"\n📝 Updating .gitignore...")
    
    # Read existing .gitignore
    existing_content = ""
    if gitignore_path.exists():
        existing_content = gitignore_path.read_text()
    
    # Check if we've already added our patterns
    if "# Cleanup v2 additions" not in existing_content:
        with open(gitignore_path, "a") as f:
            f.write("\n".join(additional_patterns))
        print(f"  ✓ Added {len(additional_patterns)} new patterns to .gitignore")
    else:
        print(f"  ⏭️  .gitignore already updated")

def generate_report(backup_dir, operations_summary):
    """Generate a cleanup report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Codebase Cleanup Report v2

**Date:** {timestamp}

**Backup Location:** `{backup_dir.name}/`

## Summary

This cleanup builds on the previous cleanup (2025-10-27) to further organize the codebase.

### Operations Performed

"""
    
    total_moved = sum(op[0] for op in operations_summary.values())
    total_skipped = sum(op[1] for op in operations_summary.values())
    
    for op_name, (moved, skipped) in operations_summary.items():
        config = CLEANUP_OPERATIONS[op_name]
        report += f"\n#### {config['description']}\n"
        report += f"- **Destination:** `{config['destination'] or 'DELETED'}`\n"
        report += f"- **Files processed:** {moved}\n"
        report += f"- **Files skipped:** {skipped}\n"
    
    report += f"""

### Overall Statistics

- **Total files moved/deleted:** {total_moved}
- **Total files skipped:** {total_skipped}
- **Backup created:** Yes
- **__pycache__ cleaned:** Yes
- **Old checkpoints cleaned:** Yes
- **.gitignore updated:** Yes

## Final Directory Structure

```
PP/
├── run_analysis.py          (QCPP entry point)
├── test_protein.py          (Main test entry point)
├── setup.py
├── pyrightconfig.json
├── requirements_qcpp.txt
│
├── src/                     (Core QCPP modules)
├── ubf_protein/             (UBF system - complete)
├── validation/              (Validation framework)
├── quantum_coherence_proteins/ (QCPP data)
│
├── scripts/
│   ├── experiments/         (Experimental test scripts)
│   └── utilities/           (Analysis and helper scripts)
│
├── results/
│   ├── test_results/        (All JSON results)
│   ├── experiments/
│   └── reports/
│
├── docs/
│   ├── analysis/            (Analysis reports and studies)
│   ├── completed_tasks/     (Historical task summaries)
│   ├── guides/              (User guides)
│   └── troubleshooting/     (Problem resolution docs)
│
├── data/                    (Datasets and inputs)
├── assets/images/           (Images and figures)
├── checkpoints/             (Checkpoints - keep recent only)
├── pdb_cache/               (Cached PDB structures)
├── campaign_10_proteins/    (Campaign results)
├── geometric_analysis/      (Geometric analysis results)
├── scaling_results/         (Scaling test results)
└── visualization_output/    (Visualization exports)
```

## Clean Codebase Guidelines

### Root Directory

Keep only:
- Entry point scripts (`run_analysis.py`, `test_protein.py`)
- Configuration files (`setup.py`, `pyrightconfig.json`, `requirements_qcpp.txt`)
- Essential documentation (`README.md`, `PUBLICATION_DRAFT.md`)
- Build scripts (`*.bat`)

### Testing

- Unit tests: In module-specific `tests/` directories
- Experimental scripts: In `scripts/experiments/`
- Validation suite: In `validation/`

### Results

- All `.json` results: In `results/test_results/`
- Experimental data: In `results/experiments/`
- Reports: In `results/reports/`

### Documentation

- Analysis documents: In `docs/analysis/`
- Guides: In `docs/guides/`
- Completed tasks: In `docs/completed_tasks/`
- Technical docs: In respective module directories

## Rollback Instructions

If you need to undo these changes:

1. Restore from backup:
   ```bash
   xcopy /E /I {backup_dir.name}\\* .
   ```

2. Or use git (if committed):
   ```bash
   git checkout HEAD -- .
   ```

## Next Steps

1. **Verify the cleanup:**
   ```bash
   python test_protein.py --pdb 1UBQ
   python run_analysis.py
   ```

2. **Run tests:**
   ```bash
   pytest ubf_protein/tests/ -v
   pytest validation/tests/ -v
   ```

3. **Check imports:**
   ```bash
   python -m py_compile scripts/experiments/*.py
   python -m py_compile scripts/utilities/*.py
   ```

4. **If everything works, remove old backup:**
   ```bash
   rmdir /S /Q backup_20251027_181101
   rmdir /S /Q {backup_dir.name}
   ```

## Maintenance

To keep the codebase clean:

1. **Results:** Always save to `results/test_results/`
2. **Experiments:** Put in `scripts/experiments/`
3. **Docs:** Categorize in `docs/` subdirectories
4. **Checkpoints:** Periodically clean old ones
5. **Cache:** Clean `__pycache__` regularly

---

*Generated by cleanup_codebase_v2.py*
"""
    
    report_path = ROOT_DIR / "CLEANUP_REPORT_V2.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"\n📄 Generated report: {report_path.name}")

def main():
    """Main cleanup function"""
    print("=" * 70)
    print("  CODEBASE CLEANUP v2")
    print("=" * 70)
    print(f"\nRoot directory: {ROOT_DIR}")
    
    # Create backup
    backup_dir = create_backup()
    
    # Perform cleanup operations
    operations_summary = {}
    
    for op_name, config in CLEANUP_OPERATIONS.items():
        moved, skipped = move_files(op_name, config)
        operations_summary[op_name] = (moved, skipped)
    
    # Clean __pycache__
    clean_pycache()
    
    # Clean old checkpoints
    clean_old_checkpoints()
    
    # Update .gitignore
    update_gitignore()
    
    # Generate report
    generate_report(backup_dir, operations_summary)
    
    print("\n" + "=" * 70)
    print("  ✅ CLEANUP COMPLETE!")
    print("=" * 70)
    print(f"\n📦 Backup saved to: {backup_dir}")
    print(f"📄 Report saved to: CLEANUP_REPORT_V2.md")
    print(f"\n⚠️  IMPORTANT: Test your code before deleting the backup!")
    print(f"   Run: python test_protein.py --pdb 1UBQ")
    print(f"   Run: python run_analysis.py")

if __name__ == "__main__":
    main()
