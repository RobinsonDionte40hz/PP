#!/usr/bin/env python3
"""
Workspace Cleanup and Organization Script

This script organizes the cluttered root directory into a clean structure:
- Moves test scripts to scripts/experiments/
- Moves utility scripts to scripts/utilities/
- Moves results to results/
- Moves documentation to docs/
- Updates all import paths automatically
- Creates backup before changes
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import re

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKBLUE}→ {msg}{Colors.ENDC}")


class WorkspaceCleanup:
    """Manages workspace cleanup and organization."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.changes_made = []
        
        # Files to keep in root (essential entry points and configs)
        self.keep_in_root = {
            'test_protein.py',
            'run_analysis.py',
            'setup.py',
            'pyrightconfig.json',
            'requirements_qcpp.txt',
            'install_biopython.bat',
            'setup_build_tools.bat',
            '.gitignore',
            'README.md',
            'LICENSE',
            'cleanup_workspace.py'  # This script
        }
        
        # Define file categorization
        self.file_mappings = {
            # Experiment/Test Scripts -> scripts/experiments/
            'scripts/experiments': [
                'agent_scaling_experiment.py',
                'batch_test_proteins.py',
                'compare_predictions.py',
                'compare_qcpp_ubf.py',
                'diagnose_rejection_rate.py',
                'test_parallel_agents.py',
                'test_parallel_extended.py',
                'test_task11.py',
                'test_task12_performance.py',
                'test_20_agents_rmse.py',
                'validate_qcpp_ubf_integration.py',
                'validate_ubiquitin_rmse.py',
                'run_task9_validation.py',
                'quick_start_integrated.py',
                'quick_test_integration.py',
            ],
            
            # Utility Scripts -> scripts/utilities/
            'scripts/utilities': [
                'analyze_memories.py',
                'download_large_proteins.py',
                'generate_protein_report.py',
                'plot_agent_scaling.py',
                'profile_energy.py',
                'profile_rmsd.py',
                'show_results.py',
                'metrics_utils.py',
            ],
            
            # Core modules that should stay accessible -> src/
            'src': [
                'protein_predictor.py',
                'qc_pipeline.py',
                'quantum_utils.py',
                'simple_quantum_dssp.py',
                'stability_calculator.py',
                'stability_predictor.py',
                'models.py',
                'model_refinement.py',
                'validation.py',
                'validation_framework.py',
            ],
            
            # JSON Results -> results/test_results/
            'results/test_results': [
                'test_1UBQ_results.json',
                'test_1CRN_results.json',
                'test_1LYZ_results.json',
                'test_1TIM_results.json',
                'test_2MR9_results.json',
                'test_3CLN_results.json',
                'test_3SSI_results.json',
                'crambin_final_comprehensive_test.json',
                'crambin_fixed_test.json',
                'crambin_high_test.json',
                'diagnostic_results.json',
                'task9_validation_results.json',
                'test_all_fixes.json',
                'test_auto_report.json',
                'test_enhanced_output.json',
                'test_fixes.json',
                'test_integration_finale.json',
                'ubf_test_results.json',
                'ubiquitin_multi_agent.json',
                'ubiquitin_parallel_2000iter.json',
                'ubiquitin_qcpp_integration_test.json',
                'ubiquitin_rmse_validation.json',
                'ubiquitin_single_agent.json',
                'ubiquitin_validation_test.json',
                'ubiquitin_with_native.json',
                'test_20_agents_results.json',
            ],
            
            # Experiment Results -> results/experiments/
            'results/experiments': [
                'agent_scaling_results.json',
            ],
            
            # TXT Reports -> results/reports/
            'results/reports': [
                'ubiquitin_multi_agent_REPORT.txt',
                'ubiquitin_single_agent_REPORT.txt',
            ],
            
            # Documentation - Task Completion -> docs/completed_tasks/
            'docs/completed_tasks': [
                'TASK_9_COMPLETE.md',
                'TASK_9_QUALITY_CONTROL_COMPLETE.md',
                'TASK_11_COMPLETE.md',
                'TASK_11_SUMMARY.md',
                'TASK_13_COMPLETE.md',
                'TASK_14_COMPLETE.md',
                'PROJECT_COMPLETE_SUMMARY.md',
                'OPTIMIZATION_COMPLETE_SUMMARY.md',
                'QCPP_UBF_INTEGRATION_COMPLETE.md',
            ],
            
            # Documentation - Analysis Reports -> docs/analysis/
            'docs/analysis': [
                'AGENT_SCALING_ANALYSIS.md',
                'AGENT_SCALING_QUICK_REFERENCE.md',
                'AGENT_SCALING_SUMMARY.md',
                'COMPREHENSIVE_TEST_RESULTS.md',
                'COMPUTATIONAL_CAPACITY_ANALYSIS.md',
                'REAL_PROTEIN_VALIDATION_RESULTS.md',
                'TESTING_SUMMARY.md',
                'QCPP_UBF_COMPARISON.md',
                'QCPP_UBF_SYNERGY_VALIDATED.md',
            ],
            
            # Documentation - Technical Guides -> docs/guides/
            'docs/guides': [
                'QCPP_VALIDATION_GUIDE.md',
                'UBF_VALIDATION_GUIDE.md',
                'EASY_PROTEIN_TESTING.md',
            ],
            
            # Documentation - Issue Analysis -> docs/troubleshooting/
            'docs/troubleshooting': [
                'BOND_THRESHOLD_FIX.md',
                'EXPLORATION_FIXES_SUMMARY.md',
                'RMSE_EXPLAINED.md',
                'RMSE_ISSUE_RESOLVED.md',
                'ROOT_CAUSE_ANALYSIS.md',
                'THRESHOLD_TUNING_ANALYSIS.md',
            ],
            
            # Data files -> data/
            'data': [
                'experimental_stability.csv',
            ],
            
            # Images -> assets/images/
            'assets/images': [
                'qcpp_ubf_comparison.png',
            ],
        }
    
    def create_backup(self):
        """Create backup of current state."""
        print_header("Creating Backup")
        
        # Only backup files we're going to move
        all_files_to_move = set()
        for files in self.file_mappings.values():
            all_files_to_move.update(files)
        
        self.backup_dir.mkdir(exist_ok=True)
        backed_up = 0
        
        for filename in all_files_to_move:
            src = self.root_dir / filename
            if src.exists() and src.is_file():
                dst = self.backup_dir / filename
                shutil.copy2(src, dst)
                backed_up += 1
        
        print_success(f"Backed up {backed_up} files to: {self.backup_dir}")
        return True
    
    def create_directory_structure(self):
        """Create new directory structure."""
        print_header("Creating Directory Structure")
        
        dirs_created = []
        for target_dir in self.file_mappings.keys():
            dir_path = self.root_dir / target_dir
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                dirs_created.append(target_dir)
                print_success(f"Created: {target_dir}/")
        
        if not dirs_created:
            print_info("All directories already exist")
        
        # Create __init__.py files for Python packages
        for dir_name in ['src', 'scripts/experiments', 'scripts/utilities']:
            init_file = self.root_dir / dir_name / '__init__.py'
            if not init_file.exists():
                init_file.touch()
                print_success(f"Created: {dir_name}/__init__.py")
    
    def move_files(self):
        """Move files to their new locations."""
        print_header("Moving Files")
        
        moved = 0
        skipped = 0
        
        for target_dir, filenames in self.file_mappings.items():
            for filename in filenames:
                src = self.root_dir / filename
                dst = self.root_dir / target_dir / filename
                
                if not src.exists():
                    print_warning(f"File not found: {filename}")
                    skipped += 1
                    continue
                
                if src.is_file():
                    # Move file
                    shutil.move(str(src), str(dst))
                    self.changes_made.append({
                        'file': filename,
                        'from': str(src.relative_to(self.root_dir)),
                        'to': str(dst.relative_to(self.root_dir))
                    })
                    print_success(f"Moved: {filename} → {target_dir}/")
                    moved += 1
        
        print_info(f"\nMoved {moved} files, skipped {skipped} missing files")
    
    def update_imports_in_file(self, file_path: Path):
        """Update import statements in a Python file."""
        if not file_path.suffix == '.py':
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # Import patterns to update
            import_replacements = {
                # Old root imports -> new src imports
                r'from protein_predictor import': 'from src.protein_predictor import',
                r'import protein_predictor': 'import src.protein_predictor',
                r'from qc_pipeline import': 'from src.qc_pipeline import',
                r'import qc_pipeline': 'import src.qc_pipeline',
                r'from quantum_utils import': 'from src.quantum_utils import',
                r'import quantum_utils': 'import src.quantum_utils',
                r'from stability_calculator import': 'from src.stability_calculator import',
                r'import stability_calculator': 'import src.stability_calculator',
                r'from stability_predictor import': 'from src.stability_predictor import',
                r'import stability_predictor': 'import src.stability_predictor',
                r'from simple_quantum_dssp import': 'from src.simple_quantum_dssp import',
                r'import simple_quantum_dssp': 'import src.simple_quantum_dssp',
                r'from models import': 'from src.models import',
                r'import models([^.]|$)': r'import src.models\1',
                r'from model_refinement import': 'from src.model_refinement import',
                r'import model_refinement': 'import src.model_refinement',
                r'from validation import': 'from src.validation import',
                r'from validation_framework import': 'from src.validation_framework import',
                r'from metrics_utils import': 'from scripts.utilities.metrics_utils import',
                r'import metrics_utils': 'import scripts.utilities.metrics_utils',
            }
            
            for old_pattern, new_import in import_replacements.items():
                if re.search(old_pattern, content):
                    content = re.sub(old_pattern, new_import, content)
                    modified = True
            
            # Update sys.path.insert for ubf_protein
            # Change from parent directory reference to explicit path
            old_path_pattern = r'sys\.path\.insert\(0,\s*str\(Path\(__file__\)\.parent\s*/\s*"ubf_protein"\)\)'
            new_path = 'sys.path.insert(0, str(Path(__file__).parent.parent / "ubf_protein"))'
            if re.search(old_path_pattern, content):
                content = re.sub(old_path_pattern, new_path, content)
                modified = True
            
            # Update relative paths for scripts moved to subdirectories
            if 'scripts/experiments' in str(file_path) or 'scripts/utilities' in str(file_path):
                # Need to go up two levels now instead of one
                content = re.sub(
                    r'Path\(__file__\)\.parent\s*/',
                    'Path(__file__).parent.parent.parent / ',
                    content
                )
                modified = True
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
        except Exception as e:
            print_warning(f"Error updating imports in {file_path.name}: {e}")
        
        return False
    
    def fix_all_imports(self):
        """Fix imports in all Python files."""
        print_header("Fixing Import Statements")
        
        fixed = 0
        
        # Fix imports in root scripts
        for script in ['test_protein.py', 'run_analysis.py']:
            script_path = self.root_dir / script
            if script_path.exists():
                if self.update_imports_in_file(script_path):
                    print_success(f"Updated imports: {script}")
                    fixed += 1
        
        # Fix imports in moved scripts
        for target_dir in ['scripts/experiments', 'scripts/utilities', 'src']:
            dir_path = self.root_dir / target_dir
            if dir_path.exists():
                for py_file in dir_path.glob('*.py'):
                    if py_file.name != '__init__.py':
                        if self.update_imports_in_file(py_file):
                            print_success(f"Updated imports: {target_dir}/{py_file.name}")
                            fixed += 1
        
        # Fix imports in ubf_protein (if they reference moved files)
        ubf_dir = self.root_dir / 'ubf_protein'
        if ubf_dir.exists():
            for py_file in ubf_dir.glob('*.py'):
                if self.update_imports_in_file(py_file):
                    print_success(f"Updated imports: ubf_protein/{py_file.name}")
                    fixed += 1
        
        # Fix imports in validation (if they reference moved files)
        val_dir = self.root_dir / 'validation'
        if val_dir.exists():
            for py_file in val_dir.glob('*.py'):
                if self.update_imports_in_file(py_file):
                    print_success(f"Updated imports: validation/{py_file.name}")
                    fixed += 1
        
        print_info(f"\nFixed imports in {fixed} files")
    
    def create_readme_files(self):
        """Create README files for new directories."""
        print_header("Creating README Files")
        
        readmes = {
            'scripts/experiments/README.md': """# Experiment Scripts

Test and experiment scripts for validating QCPP-UBF integration.

## Contents

- `agent_scaling_experiment.py` - Test agent scaling behavior
- `compare_qcpp_ubf.py` - Compare QCPP and UBF predictions
- `batch_test_proteins.py` - Batch testing multiple proteins
- `validate_qcpp_ubf_integration.py` - Integration validation tests

## Usage

Run from project root:
```bash
python scripts/experiments/agent_scaling_experiment.py
```

Or use absolute imports:
```python
from scripts.experiments import agent_scaling_experiment
```
""",
            
            'scripts/utilities/README.md': """# Utility Scripts

Helper scripts for analysis and visualization.

## Contents

- `analyze_memories.py` - Analyze agent memory patterns
- `plot_agent_scaling.py` - Visualize scaling results
- `generate_protein_report.py` - Generate analysis reports
- `profile_energy.py` - Profile energy calculations
- `profile_rmsd.py` - Profile RMSD calculations

## Usage

Run from project root:
```bash
python scripts/utilities/plot_agent_scaling.py
```
""",
            
            'src/README.md': """# Source Modules

Core modules for QCPP protein prediction.

## Contents

- `protein_predictor.py` - Main QCPP predictor class
- `quantum_utils.py` - Quantum coherence utilities
- `stability_calculator.py` - Stability calculations
- `models.py` - Data models

## Usage

Import from anywhere in the project:
```python
from src.protein_predictor import QuantumCoherenceProteinPredictor
from src.quantum_utils import calculate_phi_angle
```
""",
            
            'results/README.md': """# Results Directory

Stores all test results, reports, and experimental data.

## Structure

- `test_results/` - Individual protein test results (JSON)
- `experiments/` - Experiment-specific results
- `reports/` - Generated text reports

## Note

Results files are not tracked in git. See `.gitignore`.
""",
            
            'docs/README.md': """# Documentation

Comprehensive documentation for the QCPP-UBF protein prediction platform.

## Structure

- `completed_tasks/` - Task completion reports
- `analysis/` - Analysis and validation reports
- `guides/` - User guides and tutorials
- `troubleshooting/` - Issue resolution documentation

## Key Documents

- [QCPP Validation Guide](guides/QCPP_VALIDATION_GUIDE.md)
- [UBF Validation Guide](guides/UBF_VALIDATION_GUIDE.md)
- [Project Summary](completed_tasks/PROJECT_COMPLETE_SUMMARY.md)
""",
        }
        
        for readme_path, content in readmes.items():
            full_path = self.root_dir / readme_path
            if not full_path.exists():
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print_success(f"Created: {readme_path}")
    
    def update_gitignore(self):
        """Update .gitignore for new structure."""
        print_header("Updating .gitignore")
        
        gitignore_path = self.root_dir / '.gitignore'
        
        new_entries = [
            '\n# Results and temporary files',
            'results/',
            'backup_*/',
            '*.pyc',
            '__pycache__/',
            '.pytest_cache/',
            '*.egg-info/',
            'temp.pdb',
        ]
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                current_content = f.read()
            
            # Add new entries if not present
            with open(gitignore_path, 'a') as f:
                for entry in new_entries:
                    if entry not in current_content:
                        f.write(entry + '\n')
            
            print_success("Updated .gitignore")
        else:
            with open(gitignore_path, 'w') as f:
                f.write('\n'.join(new_entries))
            print_success("Created .gitignore")
    
    def generate_migration_report(self):
        """Generate report of all changes made."""
        print_header("Generating Migration Report")
        
        report_path = self.root_dir / 'CLEANUP_REPORT.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Workspace Cleanup Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Backup Location:** `{self.backup_dir.name}/`\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Files moved: {len(self.changes_made)}\n")
            f.write(f"- Backup created: Yes\n")
            f.write(f"- Imports updated: Yes\n\n")
            
            f.write("## New Directory Structure\n\n")
            f.write("```\n")
            f.write("PP/\n")
            f.write("├── test_protein.py          (main entry point)\n")
            f.write("├── run_analysis.py          (QCPP entry point)\n")
            f.write("├── setup.py\n")
            f.write("│\n")
            f.write("├── src/                     (core modules)\n")
            f.write("├── scripts/\n")
            f.write("│   ├── experiments/         (test scripts)\n")
            f.write("│   └── utilities/           (helper scripts)\n")
            f.write("├── results/\n")
            f.write("│   ├── test_results/        (JSON results)\n")
            f.write("│   ├── experiments/\n")
            f.write("│   └── reports/\n")
            f.write("├── docs/\n")
            f.write("│   ├── completed_tasks/\n")
            f.write("│   ├── analysis/\n")
            f.write("│   ├── guides/\n")
            f.write("│   └── troubleshooting/\n")
            f.write("├── data/                    (datasets)\n")
            f.write("├── assets/images/\n")
            f.write("│\n")
            f.write("├── ubf_protein/             (unchanged)\n")
            f.write("├── validation/              (unchanged)\n")
            f.write("└── quantum_coherence_proteins/ (unchanged)\n")
            f.write("```\n\n")
            
            f.write("## Files Moved\n\n")
            for change in sorted(self.changes_made, key=lambda x: x['to']):
                f.write(f"- `{change['file']}`\n")
                f.write(f"  - From: `{change['from']}`\n")
                f.write(f"  - To: `{change['to']}`\n\n")
            
            f.write("## Import Changes\n\n")
            f.write("All import statements have been automatically updated:\n\n")
            f.write("**Before:**\n")
            f.write("```python\n")
            f.write("from protein_predictor import QuantumCoherenceProteinPredictor\n")
            f.write("from quantum_utils import calculate_phi_angle\n")
            f.write("```\n\n")
            f.write("**After:**\n")
            f.write("```python\n")
            f.write("from src.protein_predictor import QuantumCoherenceProteinPredictor\n")
            f.write("from src.quantum_utils import calculate_phi_angle\n")
            f.write("```\n\n")
            
            f.write("## Rollback Instructions\n\n")
            f.write("If you need to undo these changes:\n\n")
            f.write("1. Delete the new directories:\n")
            f.write("   ```bash\n")
            f.write("   rm -rf src/ scripts/ results/ docs/ data/ assets/\n")
            f.write("   ```\n\n")
            f.write("2. Restore from backup:\n")
            f.write("   ```bash\n")
            f.write(f"   cp -r {self.backup_dir.name}/* .\n")
            f.write("   ```\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Test the main entry points:\n")
            f.write("   ```bash\n")
            f.write("   python test_protein.py --pdb 1UBQ\n")
            f.write("   python run_analysis.py\n")
            f.write("   ```\n\n")
            f.write("2. Run tests to verify imports:\n")
            f.write("   ```bash\n")
            f.write("   pytest ubf_protein/tests/\n")
            f.write("   pytest validation/tests/\n")
            f.write("   ```\n\n")
            f.write("3. If everything works, you can delete the backup:\n")
            f.write("   ```bash\n")
            f.write(f"   rm -rf {self.backup_dir.name}/\n")
            f.write("   ```\n")
        
        print_success(f"Report saved to: CLEANUP_REPORT.md")
    
    def run_cleanup(self, auto_confirm=False):
        """Execute full cleanup process."""
        print_header("WORKSPACE CLEANUP")
        print_info("This will organize your workspace and update all imports")
        print_info(f"Root directory: {self.root_dir}")
        
        # Confirm
        if not auto_confirm:
            response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
            if response != 'yes':
                print_warning("Cleanup cancelled")
                return False
        else:
            print_success("Auto-confirmed - proceeding with cleanup")
        
        try:
            # Step 1: Backup
            self.create_backup()
            
            # Step 2: Create structure
            self.create_directory_structure()
            
            # Step 3: Move files
            self.move_files()
            
            # Step 4: Fix imports
            self.fix_all_imports()
            
            # Step 5: Create READMEs
            self.create_readme_files()
            
            # Step 6: Update gitignore
            self.update_gitignore()
            
            # Step 7: Generate report
            self.generate_migration_report()
            
            print_header("CLEANUP COMPLETE!")
            print_success(f"✓ Organized {len(self.changes_made)} files")
            print_success(f"✓ Backup created at: {self.backup_dir}")
            print_success(f"✓ All imports updated")
            print_success(f"✓ Report saved to: CLEANUP_REPORT.md")
            
            print("\n" + Colors.BOLD + "Next Steps:" + Colors.ENDC)
            print("1. Test your main scripts:")
            print("   python test_protein.py --pdb 1UBQ")
            print("   python run_analysis.py")
            print("\n2. Run tests:")
            print("   pytest ubf_protein/tests/")
            print("   pytest validation/tests/")
            print("\n3. If everything works, delete backup folder")
            
            return True
            
        except Exception as e:
            print_warning(f"\nError during cleanup: {e}")
            print_warning("Your files are safe in the backup directory")
            return False


def main():
    """Main entry point."""
    import sys
    auto_confirm = '--auto' in sys.argv or '-y' in sys.argv
    
    cleanup = WorkspaceCleanup(os.getcwd())
    success = cleanup.run_cleanup(auto_confirm=auto_confirm)
    
    if not success:
        print("\n" + Colors.FAIL + "Cleanup failed or cancelled" + Colors.ENDC)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
