"""
Codebase Cleanup Script v3 - Third Major Cleanup
Focuses on removing accumulated test results, old markdown docs, and organizing remaining files.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def create_backup(workspace_root):
    """Create timestamped backup of current state"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = workspace_root / f"backup_{timestamp}"
    
    print(f"\n📦 Creating backup: {backup_dir}")
    
    # Files and directories to backup (important ones only)
    to_backup = [
        # Root files
        "*.json", "*.log", "*.md", "*.py", 
        # Important directories
        "src", "ubf_protein", "validation", "quantum_coherence_proteins",
        "data", "results", "docs", "scripts"
    ]
    
    backup_dir.mkdir(exist_ok=True)
    
    # Copy files
    for pattern in to_backup:
        if "*" in pattern:
            for file in workspace_root.glob(pattern):
                if file.is_file():
                    shutil.copy2(file, backup_dir / file.name)
        else:
            src = workspace_root / pattern
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, backup_dir / pattern, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, backup_dir / src.name)
    
    print(f"✅ Backup created: {backup_dir}")
    return backup_dir

def clean_pycache(workspace_root):
    """Remove all __pycache__ directories"""
    print("\n🧹 Cleaning __pycache__ directories...")
    count = 0
    for pycache in workspace_root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            count += 1
    print(f"✅ Removed {count} __pycache__ directories")

def clean_old_checkpoints(workspace_root):
    """Keep only the 5 most recent checkpoints"""
    checkpoint_dir = workspace_root / "checkpoints"
    if not checkpoint_dir.exists():
        return
    
    print("\n🧹 Cleaning old checkpoints...")
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
    
    kept = checkpoints[:5]
    removed = checkpoints[5:]
    
    for cp in removed:
        cp.unlink()
    
    print(f"✅ Kept {len(kept)} recent checkpoints, removed {len(removed)} old ones")

def organize_root_files(workspace_root):
    """Organize files in root directory"""
    
    print("\n📁 Organizing root directory files...")
    
    operations = {
        # JSON test results -> results/test_results
        "json_results": {
            "pattern": "*.json",
            "destination": "results/test_results",
            "exclude": ["pyrightconfig.json"],  # Keep this in root
            "files": []
        },
        
        # Log files -> results/logs
        "logs": {
            "pattern": "*.log",
            "destination": "results/logs",
            "exclude": [],
            "files": []
        },
        
        # Markdown documentation -> docs/archived
        "markdown_docs": {
            "pattern": "*.md",
            "destination": "docs/archived",
            "exclude": [
                "PUBLICATION_DRAFT.md",  # Keep in root
                "README.md",  # If it exists
            ],
            "files": []
        },
        
        # Python utility scripts -> scripts/archived
        "python_scripts": {
            "pattern": "*.py",
            "destination": "scripts/archived",
            "exclude": [
                "run_analysis.py",  # QCPP entry point
                "test_protein.py",  # Main test entry
                "setup.py",  # Setup script
                "cleanup_codebase_v3.py",  # This script
                "cleanup_codebase_v2.py",  # Previous cleanup
                "cleanup_workspace.py",  # Previous cleanup
            ],
            "files": []
        },
    }
    
    # Collect files to move
    for op_name, op in operations.items():
        for file in workspace_root.glob(op["pattern"]):
            if file.is_file() and file.name not in op["exclude"]:
                op["files"].append(file)
    
    # Execute moves
    total_moved = 0
    total_skipped = 0
    
    for op_name, op in operations.items():
        dest_dir = workspace_root / op["destination"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        moved = 0
        skipped = 0
        
        for file in op["files"]:
            dest_file = dest_dir / file.name
            
            # Skip if file already exists in destination
            if dest_file.exists():
                print(f"  ⏭️  Skipped (exists): {file.name}")
                skipped += 1
                continue
            
            try:
                shutil.move(str(file), str(dest_file))
                print(f"  ✅ Moved: {file.name} -> {op['destination']}")
                moved += 1
            except Exception as e:
                print(f"  ❌ Error moving {file.name}: {e}")
                skipped += 1
        
        print(f"\n  {op_name}: {moved} moved, {skipped} skipped")
        total_moved += moved
        total_skipped += skipped
    
    return total_moved, total_skipped

def remove_old_backups(workspace_root):
    """Remove old backup directories (keep latest 2)"""
    print("\n🧹 Cleaning old backup directories...")
    
    backups = sorted(workspace_root.glob("backup_*"), 
                    key=lambda p: p.stat().st_mtime, reverse=True)
    
    kept = backups[:2]  # Keep latest 2 backups
    removed = backups[2:]
    
    for backup in removed:
        if backup.is_dir():
            try:
                shutil.rmtree(backup)
                print(f"  ✅ Removed: {backup.name}")
            except Exception as e:
                print(f"  ❌ Error removing {backup.name}: {e}")
    
    print(f"✅ Kept {len(kept)} recent backups, removed {len(removed)} old ones")

def clean_test_checkpoints(workspace_root):
    """Clean test_checkpoints directory"""
    test_checkpoint_dir = workspace_root / "test_checkpoints"
    if test_checkpoint_dir.exists() and test_checkpoint_dir.is_dir():
        print("\n🧹 Cleaning test_checkpoints...")
        try:
            shutil.rmtree(test_checkpoint_dir)
            test_checkpoint_dir.mkdir()
            print("✅ Cleaned test_checkpoints directory")
        except Exception as e:
            print(f"❌ Error cleaning test_checkpoints: {e}")

def update_gitignore(workspace_root):
    """Update .gitignore with comprehensive patterns"""
    gitignore = workspace_root / ".gitignore"
    
    patterns = [
        "# Python",
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        "*.so",
        ".Python",
        "build/",
        "develop-eggs/",
        "dist/",
        "downloads/",
        "eggs/",
        ".eggs/",
        "lib/",
        "lib64/",
        "parts/",
        "sdist/",
        "var/",
        "wheels/",
        "*.egg-info/",
        ".installed.cfg",
        "*.egg",
        "",
        "# Virtual environments",
        "venv/",
        "ENV/",
        "env/",
        "myvenv/",
        ".venv/",
        "",
        "# IDEs",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "*~",
        "",
        "# Project specific",
        "checkpoints/*.json",
        "test_checkpoints/",
        "pdb_cache/",
        "visualization_output/",
        "*.log",
        "backup_*/",
        ".kiro/",
        "",
        "# Results (keep structure, ignore data)",
        "results/test_results/*.json",
        "results/experiments/*.json",
        "results/logs/*.log",
        "",
        "# Campaign data",
        "campaign_*/",
        "scaling_results/",
        "",
        "# Keep important files",
        "!.gitignore",
        "!README.md",
        "!PUBLICATION_DRAFT.md",
    ]
    
    print("\n📝 Updating .gitignore...")
    
    # Read existing content
    existing = set()
    if gitignore.exists():
        with open(gitignore, 'r') as f:
            existing = set(line.strip() for line in f if line.strip())
    
    # Merge with new patterns
    all_patterns = existing.union(set(patterns))
    
    # Write back
    with open(gitignore, 'w') as f:
        f.write('\n'.join(sorted(all_patterns, key=lambda x: (x.startswith('#'), x))))
        f.write('\n')
    
    print("✅ Updated .gitignore")

def generate_report(workspace_root, backup_dir, stats):
    """Generate cleanup report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Codebase Cleanup Report v3

**Date:** {timestamp}

**Backup Location:** `{backup_dir.name}/`

## Summary

This is the third major cleanup, focusing on accumulated test results, logs, and documentation files.

### Operations Performed

#### Root Directory Organization
- **JSON results moved:** {stats.get('json_moved', 0)}
- **Log files moved:** {stats.get('logs_moved', 0)}
- **Markdown docs moved:** {stats.get('md_moved', 0)}
- **Python scripts moved:** {stats.get('py_moved', 0)}
- **Total files moved:** {stats.get('total_moved', 0)}
- **Total files skipped:** {stats.get('total_skipped', 0)}

#### Cleanup Tasks
- **__pycache__ cleaned:** Yes
- **Old checkpoints cleaned:** Yes (kept 5 most recent)
- **Old backups removed:** Yes (kept 2 most recent)
- **test_checkpoints cleaned:** Yes
- **.gitignore updated:** Yes

## Current Clean Structure

```
PP/
├── run_analysis.py          (QCPP entry point)
├── test_protein.py          (Main test entry point)
├── setup.py                 (Setup configuration)
├── pyrightconfig.json       (Type checker config)
├── requirements_qcpp.txt    (QCPP dependencies)
├── PUBLICATION_DRAFT.md     (Main publication document)
│
├── src/                     (Core QCPP modules)
├── ubf_protein/             (UBF system - complete)
├── validation/              (Validation framework)
├── quantum_coherence_proteins/ (QCPP data)
│
├── scripts/
│   ├── experiments/         (Experimental test scripts)
│   ├── utilities/           (Analysis and helper scripts)
│   └── archived/            (Old/unused scripts)
│
├── results/
│   ├── test_results/        (All JSON test results)
│   ├── experiments/         (Experimental data)
│   ├── reports/             (Generated reports)
│   └── logs/                (Log files)
│
├── docs/
│   ├── analysis/            (Analysis reports)
│   ├── completed_tasks/     (Task summaries)
│   ├── guides/              (User guides)
│   ├── troubleshooting/     (Problem resolution)
│   └── archived/            (Old documentation)
│
├── data/                    (Datasets and inputs)
├── assets/images/           (Images and figures)
├── checkpoints/             (Active checkpoints - recent only)
├── pdb_cache/               (Cached PDB structures)
└── backup_YYYYMMDD_HHMMSS/  (Latest 2 backups only)
```

## Files Moved to Archive

### JSON Results
All `.json` files from root moved to `results/test_results/`

### Log Files
All `.log` files from root moved to `results/logs/`

### Markdown Documentation
All `.md` files (except PUBLICATION_DRAFT.md) moved to `docs/archived/`

### Python Scripts
All `.py` files (except entry points and setup) moved to `scripts/archived/`

## Maintenance Guidelines

### Root Directory Policy
Keep ONLY:
- Entry points: `run_analysis.py`, `test_protein.py`
- Configuration: `setup.py`, `pyrightconfig.json`, `requirements_qcpp.txt`
- Essential docs: `PUBLICATION_DRAFT.md`, `README.md`
- Build scripts: `*.bat`

### Results Policy
- Always save test results to `results/test_results/`
- Save logs to `results/logs/`
- Clean up old results periodically

### Checkpoint Policy
- Keep only 5 most recent checkpoints
- Clean up automatically on each run

### Backup Policy
- Keep only 2 most recent backups
- Create backup before major changes
- Remove old backups after verification

## Verification Steps

1. **Test QCPP system:**
   ```bash
   python run_analysis.py
   ```

2. **Test UBF system:**
   ```bash
   python test_protein.py --pdb 1UBQ
   ```

3. **Run unit tests:**
   ```bash
   pytest ubf_protein/tests/ -v
   pytest validation/tests/ -v
   ```

4. **Verify imports:**
   ```bash
   python -c "from src.protein_predictor import QuantumCoherenceProteinPredictor"
   python -c "from ubf_protein.protein_agent import ProteinAgent"
   ```

## Rollback Instructions

If you need to undo these changes:

1. **Restore from backup:**
   ```bash
   xcopy /E /I {backup_dir.name}\\* .
   ```

2. **Or use git (if committed):**
   ```bash
   git checkout HEAD -- .
   ```

## Next Steps

1. Verify all tests pass
2. Commit cleaned codebase to git
3. Remove old backups after confirmation:
   ```bash
   rmdir /S /Q backup_20251105_164219
   rmdir /S /Q backup_20251105_164254
   ```

---

*Generated by cleanup_codebase_v3.py*
*Third Major Cleanup - {timestamp}*
"""
    
    report_file = workspace_root / "CLEANUP_REPORT_V3.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved: {report_file}")
    return report_file

def main():
    """Main cleanup process"""
    workspace_root = Path(__file__).parent
    
    print("=" * 70)
    print("🧹 CODEBASE CLEANUP v3 - THIRD MAJOR CLEANUP")
    print("=" * 70)
    print(f"\nWorkspace: {workspace_root}")
    print("\nThis will:")
    print("  1. Create a backup")
    print("  2. Move test results to results/test_results/")
    print("  3. Move log files to results/logs/")
    print("  4. Move old markdown docs to docs/archived/")
    print("  5. Move old python scripts to scripts/archived/")
    print("  6. Clean __pycache__ directories")
    print("  7. Clean old checkpoints (keep 5)")
    print("  8. Clean old backups (keep 2)")
    print("  9. Clean test_checkpoints directory")
    print(" 10. Update .gitignore")
    print(" 11. Generate cleanup report")
    
    response = input("\n❓ Proceed with cleanup? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Cleanup cancelled")
        return
    
    # Create backup first
    backup_dir = create_backup(workspace_root)
    
    # Perform cleanup operations
    stats = {}
    
    # Organize root files
    total_moved, total_skipped = organize_root_files(workspace_root)
    stats['total_moved'] = total_moved
    stats['total_skipped'] = total_skipped
    
    # Clean various directories
    clean_pycache(workspace_root)
    clean_old_checkpoints(workspace_root)
    remove_old_backups(workspace_root)
    clean_test_checkpoints(workspace_root)
    update_gitignore(workspace_root)
    
    # Generate report
    report_file = generate_report(workspace_root, backup_dir, stats)
    
    print("\n" + "=" * 70)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 70)
    print(f"\n📦 Backup: {backup_dir}")
    print(f"📄 Report: {report_file}")
    print("\n🔍 Next steps:")
    print("  1. Review the cleanup report")
    print("  2. Run tests to verify everything works")
    print("  3. Delete old backups if satisfied")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
