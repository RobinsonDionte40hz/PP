"""
DocumentationGenerator - Automated research documentation and figure generation.

This module provides comprehensive documentation generation capabilities for protein
structure prediction validation campaigns, including:
- Phase summary reports in Markdown format
- Publication-ready figure specifications
- Methods section generation with exact protocol descriptions
- Supplementary data table generation
- Multi-format export for plotting tools (CSV, Excel, JSON)

Author: Large-Scale Validation Framework
Date: October 26, 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json
import csv


@dataclass(frozen=True)
class ResearchReport:
    """
    Comprehensive research report structure.
    
    Attributes:
        title: Report title
        methodology: Detailed methodology description
        results_summary: Summary of validation results
        statistical_analysis: Statistical analysis section
        figures: List of figure file paths
        tables: List of table file paths
        conclusions: List of conclusion statements
        generated_timestamp: Report generation timestamp
    """
    title: str
    methodology: str
    results_summary: str
    statistical_analysis: str
    figures: List[str]
    tables: List[str]
    conclusions: List[str]
    generated_timestamp: datetime


class DocumentationGenerator:
    """
    Automated research documentation and figure generation.
    
    Generates publication-ready documentation including phase reports, methods sections,
    supplementary tables, and data exports in multiple formats for analysis and plotting.
    
    Example:
        >>> generator = DocumentationGenerator()
        >>> report = generator.generate_phase_report(phase, results)
        >>> print(report)  # Markdown formatted report
        >>> 
        >>> figures = generator.generate_publication_figures(results)
        >>> tables = generator.generate_supplementary_tables(results)
    """
    
    def __init__(self, output_dir: str = "./documentation"):
        """
        Initialize the DocumentationGenerator.
        
        Args:
            output_dir: Base directory for all generated documentation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "exports").mkdir(exist_ok=True)
    
    def generate_phase_report(
        self,
        phase: Dict,
        results: List[Dict],
        statistical_summary: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive phase summary report in Markdown format.
        
        Creates a detailed report including phase information, success metrics,
        statistical summaries, and individual test results.
        
        Args:
            phase: Phase dictionary with phase_number, name, protein_count
            results: List of validation reports for this phase
            statistical_summary: Optional statistical analysis results
        
        Returns:
            Markdown formatted report string
        
        Example:
            >>> phase = {'phase_number': 1, 'name': 'Initial Validation', 'protein_count': 10}
            >>> report = generator.generate_phase_report(phase, results)
            >>> with open('phase1_report.md', 'w') as f:
            ...     f.write(report)
        """
        phase_num = phase.get('phase_number', 1)
        phase_name = phase.get('name', f'Phase {phase_num}')
        
        # Calculate metrics
        total_tests = len(results)
        if total_tests == 0:
            return f"# {phase_name}\n\nNo results available.\n"
        
        # Success metrics (RMSD < 5.0Å as threshold)
        success_threshold = 5.0
        successes = sum(1 for r in results if r.get('rmsd', float('inf')) < success_threshold)
        success_rate = (successes / total_tests) * 100
        
        # Calculate averages
        avg_rmsd = sum(r.get('rmsd', 0.0) for r in results) / total_tests
        avg_gdt_ts = sum(r.get('gdt_ts', 0.0) for r in results) / total_tests
        avg_tm_score = sum(r.get('tm_score', 0.0) for r in results) / total_tests
        avg_energy = sum(r.get('energy', r.get('final_energy', 0.0)) for r in results) / total_tests
        
        # Build report
        report = f"""# {phase_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Total Tests:** {total_tests}
- **Successes:** {successes} ({success_rate:.1f}%)
- **Failures:** {total_tests - successes} ({100 - success_rate:.1f}%)

## Performance Metrics

### Summary Statistics

| Metric | Average | Best | Worst |
|--------|---------|------|-------|
| RMSD (Å) | {avg_rmsd:.2f} | {min(r.get('rmsd', 0) for r in results):.2f} | {max(r.get('rmsd', 0) for r in results):.2f} |
| GDT-TS | {avg_gdt_ts:.1f} | {max(r.get('gdt_ts', 0) for r in results):.1f} | {min(r.get('gdt_ts', 0) for r in results):.1f} |
| TM-score | {avg_tm_score:.3f} | {max(r.get('tm_score', 0) for r in results):.3f} | {min(r.get('tm_score', 0) for r in results):.3f} |
| Energy (kcal/mol) | {avg_energy:.1f} | {min(r.get('energy', r.get('final_energy', 0)) for r in results):.1f} | {max(r.get('energy', r.get('final_energy', 0)) for r in results):.1f} |

"""
        
        # Add statistical analysis if provided
        if statistical_summary:
            report += "\n### Statistical Analysis\n\n"
            if 'correlations' in statistical_summary:
                corr = statistical_summary['correlations']
                report += f"**Correlations:**\n"
                report += f"- Size vs RMSD: {corr.get('size_vs_rmsd', 0):.3f}\n"
                report += f"- Resolution vs RMSD: {corr.get('resolution_vs_rmsd', 0):.3f}\n\n"
        
        # Add individual results
        report += "## Individual Test Results\n\n"
        report += "| PDB ID | RMSD (Å) | GDT-TS | TM-score | Energy | Status |\n"
        report += "|--------|----------|--------|----------|--------|--------|\n"
        
        for result in sorted(results, key=lambda x: x.get('rmsd', 0)):
            pdb_id = result.get('pdb_id', 'UNKNOWN')
            rmsd = result.get('rmsd', 0.0)
            gdt_ts = result.get('gdt_ts', 0.0)
            tm_score = result.get('tm_score', 0.0)
            energy = result.get('energy', result.get('final_energy', 0.0))
            status = "✅ Success" if rmsd < success_threshold else "❌ Failure"
            
            report += f"| {pdb_id} | {rmsd:.2f} | {gdt_ts:.1f} | {tm_score:.3f} | {energy:.1f} | {status} |\n"
        
        # Add conclusions
        report += "\n## Conclusions\n\n"
        
        if success_rate >= 60:
            report += f"✅ **Quality gate PASSED** (success rate {success_rate:.1f}% ≥ 60%)\n\n"
            report += "The phase demonstrates acceptable prediction accuracy. Ready to proceed to next phase.\n"
        else:
            report += f"⚠️ **Quality gate FAILED** (success rate {success_rate:.1f}% < 60%)\n\n"
            report += "Parameter adjustments recommended before proceeding:\n"
            report += "- Review failure patterns using FailureAnalyzer\n"
            report += "- Consider increasing iterations or agent count\n"
            report += "- Enable QCPP integration if not already active\n"
        
        return report
    
    def generate_publication_figures(
        self,
        results: List[Dict],
        output_subdir: str = "figures"
    ) -> List[str]:
        """
        Generate publication-ready figure specifications.
        
        Creates JSON specifications for various plots that can be rendered using
        matplotlib, seaborn, or other plotting libraries.
        
        Args:
            results: List of validation reports
            output_subdir: Subdirectory within output_dir for figures
        
        Returns:
            List of file paths to generated figure specifications
        
        Example:
            >>> figure_specs = generator.generate_publication_figures(results)
            >>> print(f"Generated {len(figure_specs)} figure specifications")
        """
        figures_dir = self.output_dir / output_subdir
        figures_dir.mkdir(exist_ok=True)
        
        generated_files = []
        
        # Figure 1: RMSD distribution histogram
        rmsd_hist_spec = {
            'figure_type': 'histogram',
            'title': 'Distribution of RMSD Values',
            'data': [r.get('rmsd', 0.0) for r in results],
            'x_label': 'RMSD (Å)',
            'y_label': 'Count',
            'bins': 20,
            'color': 'skyblue',
            'vertical_lines': [
                {'x': 5.0, 'label': 'Success Threshold', 'color': 'green', 'linestyle': '--'},
                {'x': 8.0, 'label': 'Severe Failure', 'color': 'red', 'linestyle': '--'}
            ]
        }
        
        rmsd_file = figures_dir / "rmsd_distribution.json"
        with open(rmsd_file, 'w') as f:
            json.dump(rmsd_hist_spec, f, indent=2)
        generated_files.append(str(rmsd_file))
        
        # Figure 2: GDT-TS vs RMSD scatter plot
        scatter_spec = {
            'figure_type': 'scatter',
            'title': 'GDT-TS vs RMSD',
            'x_data': [r.get('rmsd', 0.0) for r in results],
            'y_data': [r.get('gdt_ts', 0.0) for r in results],
            'x_label': 'RMSD (Å)',
            'y_label': 'GDT-TS Score',
            'point_labels': [r.get('pdb_id', '') for r in results],
            'color_by': 'protein_length',
            'colormap': 'viridis'
        }
        
        scatter_file = figures_dir / "gdt_vs_rmsd.json"
        with open(scatter_file, 'w') as f:
            json.dump(scatter_spec, f, indent=2)
        generated_files.append(str(scatter_file))
        
        # Figure 3: Size category comparison box plot
        size_box_spec = {
            'figure_type': 'boxplot',
            'title': 'RMSD by Protein Size Category',
            'data_by_category': self._group_by_category(results, 'size_category', 'rmsd'),
            'x_label': 'Size Category',
            'y_label': 'RMSD (Å)',
            'colors': ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        }
        
        box_file = figures_dir / "rmsd_by_size.json"
        with open(box_file, 'w') as f:
            json.dump(size_box_spec, f, indent=2)
        generated_files.append(str(box_file))
        
        # Figure 4: Energy vs RMSD correlation
        energy_scatter_spec = {
            'figure_type': 'scatter',
            'title': 'Energy vs RMSD Correlation',
            'x_data': [r.get('rmsd', 0.0) for r in results],
            'y_data': [r.get('energy', r.get('final_energy', 0.0)) for r in results],
            'x_label': 'RMSD (Å)',
            'y_label': 'Energy (kcal/mol)',
            'show_trendline': True,
            'highlight_regions': [
                {'x_min': 0, 'x_max': 5.0, 'y_min': -100, 'y_max': 0, 'label': 'Success Zone', 'alpha': 0.1, 'color': 'green'}
            ]
        }
        
        energy_file = figures_dir / "energy_vs_rmsd.json"
        with open(energy_file, 'w') as f:
            json.dump(energy_scatter_spec, f, indent=2)
        generated_files.append(str(energy_file))
        
        return generated_files
    
    def generate_methods_section(
        self,
        metadata: Dict
    ) -> str:
        """
        Generate detailed methods section with exact protocol description.
        
        Creates publication-ready methods section documenting software versions,
        parameters, and experimental protocol.
        
        Args:
            metadata: Test run metadata with configuration details
        
        Returns:
            Markdown formatted methods section
        
        Example:
            >>> metadata = {
            ...     'ubf_version': '1.0.0',
            ...     'qcpp_enabled': True,
            ...     'num_agents': 10,
            ...     'iterations_per_agent': 1000
            ... }
            >>> methods = generator.generate_methods_section(metadata)
        """
        methods = f"""## Methods

### Protein Structure Prediction

**Software:** Universal Behavioral Framework (UBF) Protein System v{metadata.get('ubf_version', '1.0.0')}

**Configuration:**
- Agent population: {metadata.get('num_agents', 10)} autonomous agents
- Iterations per agent: {metadata.get('iterations_per_agent', 1000)}
- Exploration strategy: {metadata.get('diversity_profile', 'balanced')}
- QCPP integration: {'Enabled' if metadata.get('qcpp_enabled', False) else 'Disabled'}

**Behavioral Parameters:**
- Exploration energy: {metadata.get('exploration_energy', 0.5):.2f}
- Structural focus: {metadata.get('structural_focus', 0.7):.2f}
- Risk tolerance: {metadata.get('risk_tolerance', 0.3):.2f}
- Native state ambition: {metadata.get('native_state_ambition', 0.6):.2f}

**Energy Function:**
- Bond stretch penalty: {metadata.get('bond_weight', 10.0):.1f}
- Angle bend penalty: {metadata.get('angle_weight', 5.0):.1f}
- Dihedral torsion: {metadata.get('dihedral_weight', 2.0):.1f}
- Van der Waals: {metadata.get('vdw_weight', 1.0):.1f}
- Electrostatics: {metadata.get('electrostatic_weight', 1.0):.1f}
- Hydrogen bonds: {metadata.get('hbond_weight', 2.0):.1f}

### Validation Protocol

**Metrics:**
- Root Mean Square Deviation (RMSD): Cα atoms aligned to native structure
- Global Distance Test Total Score (GDT-TS): 1%, 2%, 4%, 8% distance cutoffs
- TM-score: Template Modeling score for fold similarity
- Final energy: Molecular mechanics energy in kcal/mol

**Success Criteria:**
- RMSD < 5.0 Å: Acceptable structure prediction
- GDT-TS > 50: Correct overall fold
- TM-score > 0.5: Same fold family

**Quality Gates:**
- Phase 1: ≥60% success rate required to proceed
- Each phase must demonstrate consistent or improving performance

### Statistical Analysis

Pearson correlation coefficients were calculated to assess relationships between protein characteristics (size, resolution, secondary structure content) and prediction accuracy metrics. One-way ANOVA was performed to compare performance across protein size categories. 95% confidence intervals were calculated for all mean metrics using t-distribution for sample sizes <30 and normal distribution otherwise.

### Computational Environment

- **Operating System:** {metadata.get('os_version', 'Windows 10/11')}
- **Python Version:** {metadata.get('python_version', '3.8+')}
- **Processor:** {metadata.get('cpu_model', 'Multi-core processor')}
- **RAM:** {metadata.get('ram_gb', 16)} GB
- **Execution Time:** {metadata.get('total_runtime_hours', 'N/A')} hours (wall-clock)

### Reproducibility

All experiments were conducted with fixed random seeds (seed={metadata.get('random_seed', 42)}) to ensure reproducibility. Complete configuration files and execution logs are available in supplementary materials. Source code and analysis scripts are available at [repository URL].

"""
        return methods
    
    def generate_supplementary_tables(
        self,
        results: List[Dict],
        output_subdir: str = "tables"
    ) -> str:
        """
        Generate supplementary data tables in CSV format.
        
        Creates comprehensive tables with all validation results suitable for
        supplementary materials or further analysis.
        
        Args:
            results: List of validation reports
            output_subdir: Subdirectory within output_dir for tables
        
        Returns:
            Path to generated table file
        
        Example:
            >>> table_path = generator.generate_supplementary_tables(results)
            >>> print(f"Supplementary table: {table_path}")
        """
        tables_dir = self.output_dir / output_subdir
        tables_dir.mkdir(exist_ok=True)
        
        output_file = tables_dir / "supplementary_table_1.csv"
        
        # Define columns
        fieldnames = [
            'pdb_id', 'protein_length', 'size_category', 'structural_class',
            'resolution', 'helix_fraction', 'sheet_fraction',
            'rmsd', 'gdt_ts', 'tm_score', 'final_energy',
            'num_agents', 'iterations', 'success'
        ]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for result in results:
                # Add success field
                result_copy = result.copy()
                result_copy['success'] = 'Yes' if result.get('rmsd', float('inf')) < 5.0 else 'No'
                result_copy['final_energy'] = result.get('energy', result.get('final_energy', 0.0))
                
                writer.writerow(result_copy)
        
        return str(output_file)
    
    def export_for_plotting_tools(
        self,
        results: List[Dict],
        formats: List[str] = ['csv', 'json'],
        output_subdir: str = "exports"
    ) -> Dict[str, str]:
        """
        Export data in multiple formats for external plotting tools.
        
        Exports validation results in various formats suitable for analysis
        in R, Python, Excel, Prism, or other statistical software.
        
        Args:
            results: List of validation reports
            formats: List of export formats ('csv', 'json', 'excel')
            output_subdir: Subdirectory within output_dir for exports
        
        Returns:
            Dictionary mapping format names to file paths
        
        Example:
            >>> exports = generator.export_for_plotting_tools(results, ['csv', 'json'])
            >>> print(f"CSV export: {exports['csv']}")
            >>> print(f"JSON export: {exports['json']}")
        """
        exports_dir = self.output_dir / output_subdir
        exports_dir.mkdir(exist_ok=True)
        
        exported_files = {}
        
        # CSV export
        if 'csv' in formats:
            csv_file = exports_dir / "validation_results.csv"
            
            fieldnames = ['pdb_id', 'rmsd', 'gdt_ts', 'tm_score', 'energy', 
                         'protein_length', 'size_category', 'resolution']
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                
                for result in results:
                    row = result.copy()
                    row['energy'] = result.get('energy', result.get('final_energy', 0.0))
                    writer.writerow(row)
            
            exported_files['csv'] = str(csv_file)
        
        # JSON export
        if 'json' in formats:
            json_file = exports_dir / "validation_results.json"
            
            # Create clean export data
            export_data = []
            for result in results:
                clean_result = {
                    'pdb_id': result.get('pdb_id'),
                    'rmsd': result.get('rmsd'),
                    'gdt_ts': result.get('gdt_ts'),
                    'tm_score': result.get('tm_score'),
                    'energy': result.get('energy', result.get('final_energy', 0.0)),
                    'protein_length': result.get('protein_length'),
                    'size_category': result.get('size_category'),
                    'resolution': result.get('resolution')
                }
                export_data.append(clean_result)
            
            with open(json_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            exported_files['json'] = str(json_file)
        
        # Excel-compatible CSV (with additional formatting hints)
        if 'excel' in formats:
            excel_file = exports_dir / "validation_results_excel.csv"
            
            with open(excel_file, 'w', newline='') as f:
                fieldnames = ['PDB ID', 'RMSD (Å)', 'GDT-TS', 'TM-score', 
                             'Energy (kcal/mol)', 'Protein Length', 'Size Category', 
                             'Resolution (Å)']
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                
                for result in results:
                    writer.writerow([
                        result.get('pdb_id', ''),
                        f"{result.get('rmsd', 0.0):.2f}",
                        f"{result.get('gdt_ts', 0.0):.1f}",
                        f"{result.get('tm_score', 0.0):.3f}",
                        f"{result.get('energy', result.get('final_energy', 0.0)):.1f}",
                        result.get('protein_length', 0),
                        result.get('size_category', ''),
                        f"{result.get('resolution', 0.0):.2f}"
                    ])
            
            exported_files['excel'] = str(excel_file)
        
        return exported_files
    
    def create_research_report(
        self,
        title: str,
        phase: Dict,
        results: List[Dict],
        metadata: Dict,
        statistical_summary: Optional[Dict] = None
    ) -> ResearchReport:
        """
        Create comprehensive research report with all components.
        
        Generates a complete ResearchReport dataclass instance with all sections
        and exports.
        
        Args:
            title: Report title
            phase: Phase information
            results: Validation results
            metadata: Test run metadata
            statistical_summary: Optional statistical analysis
        
        Returns:
            ResearchReport instance with all components
        
        Example:
            >>> report = generator.create_research_report(
            ...     "Phase 1 Validation Results",
            ...     phase_dict,
            ...     results_list,
            ...     metadata_dict
            ... )
            >>> print(report.title)
            >>> print(f"Generated {len(report.figures)} figures")
        """
        # Generate all components
        methodology = self.generate_methods_section(metadata)
        results_summary = self.generate_phase_report(phase, results, statistical_summary)
        
        # Generate figures and tables
        figures = self.generate_publication_figures(results)
        tables = [self.generate_supplementary_tables(results)]
        
        # Generate conclusions
        total_tests = len(results)
        successes = sum(1 for r in results if r.get('rmsd', float('inf')) < 5.0)
        success_rate = (successes / total_tests * 100) if total_tests > 0 else 0
        
        conclusions = []
        if success_rate >= 60:
            conclusions.append(f"Phase achieved {success_rate:.1f}% success rate, meeting quality gate threshold.")
            conclusions.append("System demonstrates acceptable prediction accuracy for this protein set.")
        else:
            conclusions.append(f"Phase achieved {success_rate:.1f}% success rate, below quality gate threshold.")
            conclusions.append("Parameter adjustments required before proceeding to next phase.")
        
        # Statistical analysis text
        stat_analysis = ""
        if statistical_summary and 'correlations' in statistical_summary:
            corr = statistical_summary['correlations']
            stat_analysis = f"""Statistical analysis revealed correlations between protein characteristics and accuracy:
- Protein size vs RMSD: r = {corr.get('size_vs_rmsd', 0):.3f}
- Resolution vs RMSD: r = {corr.get('resolution_vs_rmsd', 0):.3f}
"""
        
        return ResearchReport(
            title=title,
            methodology=methodology,
            results_summary=results_summary,
            statistical_analysis=stat_analysis,
            figures=figures,
            tables=tables,
            conclusions=conclusions,
            generated_timestamp=datetime.now()
        )
    
    # Helper methods
    
    def _group_by_category(
        self,
        results: List[Dict],
        category_field: str,
        value_field: str
    ) -> Dict[str, List[float]]:
        """Group numeric values by category."""
        grouped = {}
        for result in results:
            category = result.get(category_field, 'unknown')
            value = result.get(value_field, 0.0)
            
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(value)
        
        return grouped
