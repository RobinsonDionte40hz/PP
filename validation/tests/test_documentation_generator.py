"""
Test suite for DocumentationGenerator.

Comprehensive tests covering all report generation, export, and documentation features.

Author: Large-Scale Validation Framework  
Date: October 26, 2025
"""

import pytest
from pathlib import Path
import json
import csv
from datetime import datetime
from validation.documentation_generator import (
    DocumentationGenerator,
    ResearchReport
)


# Fixtures

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def generator(temp_dir):
    """Create DocumentationGenerator instance."""
    return DocumentationGenerator(output_dir=str(temp_dir))


@pytest.fixture
def sample_results():
    """Sample validation results."""
    return [
        {
            'pdb_id': '1UBQ',
            'protein_length': 76,
            'size_category': 'small',
            'structural_class': 'mainly-beta',
            'resolution': 1.8,
            'helix_fraction': 0.2,
            'sheet_fraction': 0.5,
            'rmsd': 3.2,
            'gdt_ts': 65.5,
            'tm_score': 0.72,
            'energy': -45.3,
            'num_agents': 10,
            'iterations': 1000
        },
        {
            'pdb_id': '1CRN',
            'protein_length': 46,
            'size_category': 'small',
            'structural_class': 'mainly-alpha',
            'resolution': 1.5,
            'helix_fraction': 0.6,
            'sheet_fraction': 0.1,
            'rmsd': 4.8,
            'gdt_ts': 58.2,
            'tm_score': 0.65,
            'energy': -32.1,
            'num_agents': 10,
            'iterations': 1000
        },
        {
            'pdb_id': '2MR9',
            'protein_length': 35,
            'size_category': 'tiny',
            'structural_class': 'mainly-alpha',
            'resolution': 2.0,
            'helix_fraction': 0.7,
            'sheet_fraction': 0.0,
            'rmsd': 6.5,
            'gdt_ts': 42.0,
            'tm_score': 0.48,
            'energy': -18.5,
            'num_agents': 10,
            'iterations': 1000
        }
    ]


@pytest.fixture
def sample_phase():
    """Sample phase dictionary."""
    return {
        'phase_number': 1,
        'name': 'Initial Validation',
        'protein_count': 3,
        'success_threshold': 60
    }


@pytest.fixture
def sample_metadata():
    """Sample test run metadata."""
    return {
        'ubf_version': '1.0.0',
        'qcpp_enabled': True,
        'num_agents': 10,
        'iterations_per_agent': 1000,
        'diversity_profile': 'balanced',
        'exploration_energy': 0.5,
        'structural_focus': 0.7,
        'risk_tolerance': 0.3,
        'native_state_ambition': 0.6,
        'bond_weight': 10.0,
        'angle_weight': 5.0,
        'dihedral_weight': 2.0,
        'vdw_weight': 1.0,
        'electrostatic_weight': 1.0,
        'hbond_weight': 2.0,
        'os_version': 'Windows 11',
        'python_version': '3.14.0',
        'cpu_model': 'Intel i7',
        'ram_gb': 16,
        'total_runtime_hours': 2.5,
        'random_seed': 42
    }


# Test DocumentationGenerator initialization

def test_generator_initialization(temp_dir):
    """Test DocumentationGenerator creates required directories."""
    generator = DocumentationGenerator(output_dir=str(temp_dir))
    
    assert generator.output_dir == Path(temp_dir)
    assert (temp_dir / "reports").exists()
    assert (temp_dir / "figures").exists()
    assert (temp_dir / "tables").exists()
    assert (temp_dir / "exports").exists()


def test_generator_default_directory():
    """Test default output directory."""
    generator = DocumentationGenerator()
    assert generator.output_dir == Path("./documentation")


# Test generate_phase_report

def test_generate_phase_report_basic(generator, sample_phase, sample_results):
    """Test basic phase report generation."""
    report = generator.generate_phase_report(sample_phase, sample_results)
    
    assert isinstance(report, str)
    assert "Initial Validation" in report
    assert "**Total Tests:** 3" in report
    assert "1UBQ" in report
    assert "RMSD" in report


def test_generate_phase_report_empty_results(generator, sample_phase):
    """Test phase report with no results."""
    report = generator.generate_phase_report(sample_phase, [])
    
    assert "No results available" in report


def test_generate_phase_report_success_metrics(generator, sample_phase, sample_results):
    """Test success rate calculation."""
    report = generator.generate_phase_report(sample_phase, sample_results)
    
    # 2 out of 3 pass (RMSD < 5.0)
    assert "Successes: 2" in report or "66.7%" in report


def test_generate_phase_report_quality_gate_pass(generator, sample_phase):
    """Test quality gate passing message."""
    # All successes
    results = [
        {'pdb_id': f'TEST{i}', 'rmsd': 3.0, 'gdt_ts': 70.0, 'tm_score': 0.7, 'energy': -40.0}
        for i in range(3)
    ]
    
    report = generator.generate_phase_report(sample_phase, results)
    assert "Quality gate PASSED" in report or "✅" in report


def test_generate_phase_report_quality_gate_fail(generator, sample_phase):
    """Test quality gate failing message."""
    # All failures
    results = [
        {'pdb_id': f'TEST{i}', 'rmsd': 8.0, 'gdt_ts': 30.0, 'tm_score': 0.3, 'energy': 10.0}
        for i in range(3)
    ]
    
    report = generator.generate_phase_report(sample_phase, results)
    assert "Quality gate FAILED" in report or "⚠️" in report


def test_generate_phase_report_with_statistics(generator, sample_phase, sample_results):
    """Test phase report with statistical summary."""
    stats = {
        'correlations': {
            'size_vs_rmsd': 0.65,
            'resolution_vs_rmsd': -0.42
        }
    }
    
    report = generator.generate_phase_report(sample_phase, sample_results, stats)
    
    assert "Statistical Analysis" in report
    assert "0.65" in report or "0.650" in report
    assert "-0.42" in report or "-0.420" in report


def test_generate_phase_report_table_formatting(generator, sample_phase, sample_results):
    """Test report contains properly formatted tables."""
    report = generator.generate_phase_report(sample_phase, sample_results)
    
    # Check for Markdown table headers
    assert "| Metric |" in report
    assert "| PDB ID |" in report
    assert "|--------|" in report


# Test generate_publication_figures

def test_generate_publication_figures_basic(generator, sample_results):
    """Test basic figure generation."""
    figures = generator.generate_publication_figures(sample_results)
    
    assert isinstance(figures, list)
    assert len(figures) == 4  # 4 figure types


def test_generate_publication_figures_files_created(generator, sample_results, temp_dir):
    """Test figure specification files are created."""
    figures = generator.generate_publication_figures(sample_results)
    
    for fig_path in figures:
        assert Path(fig_path).exists()
        assert fig_path.endswith('.json')


def test_generate_publication_figures_rmsd_histogram(generator, sample_results, temp_dir):
    """Test RMSD histogram specification."""
    figures = generator.generate_publication_figures(sample_results)
    
    # Find RMSD distribution figure
    rmsd_fig = next(f for f in figures if 'rmsd_distribution' in f)
    
    with open(rmsd_fig, 'r') as f:
        spec = json.load(f)
    
    assert spec['figure_type'] == 'histogram'
    assert 'RMSD' in spec['title']
    assert len(spec['data']) == 3
    assert spec['bins'] == 20


def test_generate_publication_figures_scatter_plot(generator, sample_results, temp_dir):
    """Test GDT-TS vs RMSD scatter plot."""
    figures = generator.generate_publication_figures(sample_results)
    
    scatter_fig = next(f for f in figures if 'gdt_vs_rmsd' in f)
    
    with open(scatter_fig, 'r') as f:
        spec = json.load(f)
    
    assert spec['figure_type'] == 'scatter'
    assert len(spec['x_data']) == 3
    assert len(spec['y_data']) == 3
    assert len(spec['point_labels']) == 3


def test_generate_publication_figures_box_plot(generator, sample_results, temp_dir):
    """Test size category box plot."""
    figures = generator.generate_publication_figures(sample_results)
    
    box_fig = next(f for f in figures if 'rmsd_by_size' in f)
    
    with open(box_fig, 'r') as f:
        spec = json.load(f)
    
    assert spec['figure_type'] == 'boxplot'
    assert 'data_by_category' in spec
    assert 'small' in spec['data_by_category']


def test_generate_publication_figures_energy_correlation(generator, sample_results, temp_dir):
    """Test energy vs RMSD correlation plot."""
    figures = generator.generate_publication_figures(sample_results)
    
    energy_fig = next(f for f in figures if 'energy_vs_rmsd' in f)
    
    with open(energy_fig, 'r') as f:
        spec = json.load(f)
    
    assert spec['figure_type'] == 'scatter'
    assert spec['show_trendline'] is True
    assert 'highlight_regions' in spec


# Test generate_methods_section

def test_generate_methods_section_basic(generator, sample_metadata):
    """Test basic methods section generation."""
    methods = generator.generate_methods_section(sample_metadata)
    
    assert isinstance(methods, str)
    assert "## Methods" in methods
    assert "Protein Structure Prediction" in methods
    assert "Validation Protocol" in methods


def test_generate_methods_section_software_version(generator, sample_metadata):
    """Test software version included."""
    methods = generator.generate_methods_section(sample_metadata)
    
    assert "v1.0.0" in methods
    assert "UBF" in methods


def test_generate_methods_section_configuration(generator, sample_metadata):
    """Test configuration parameters included."""
    methods = generator.generate_methods_section(sample_metadata)
    
    assert "10 autonomous agents" in methods
    assert "1000" in methods  # iterations
    assert "balanced" in methods


def test_generate_methods_section_qcpp_enabled(generator, sample_metadata):
    """Test QCPP integration status."""
    methods = generator.generate_methods_section(sample_metadata)
    assert "QCPP integration: Enabled" in methods
    
    # Test disabled
    sample_metadata['qcpp_enabled'] = False
    methods = generator.generate_methods_section(sample_metadata)
    assert "QCPP integration: Disabled" in methods


def test_generate_methods_section_behavioral_params(generator, sample_metadata):
    """Test behavioral parameters included."""
    methods = generator.generate_methods_section(sample_metadata)
    
    assert "Exploration energy: 0.50" in methods or "0.5" in methods
    assert "Structural focus: 0.70" in methods or "0.7" in methods


def test_generate_methods_section_energy_function(generator, sample_metadata):
    """Test energy function parameters included."""
    methods = generator.generate_methods_section(sample_metadata)
    
    assert "Bond stretch penalty: 10.0" in methods
    assert "Van der Waals: 1.0" in methods


def test_generate_methods_section_success_criteria(generator, sample_metadata):
    """Test success criteria documented."""
    methods = generator.generate_methods_section(sample_metadata)
    
    assert "RMSD < 5.0" in methods
    assert "GDT-TS > 50" in methods
    assert "TM-score > 0.5" in methods


# Test generate_supplementary_tables

def test_generate_supplementary_tables_basic(generator, sample_results, temp_dir):
    """Test basic supplementary table generation."""
    table_path = generator.generate_supplementary_tables(sample_results)
    
    assert Path(table_path).exists()
    assert table_path.endswith('.csv')


def test_generate_supplementary_tables_content(generator, sample_results, temp_dir):
    """Test table contains all data."""
    table_path = generator.generate_supplementary_tables(sample_results)
    
    with open(table_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 3
    assert rows[0]['pdb_id'] == '1UBQ'
    assert float(rows[0]['rmsd']) == 3.2


def test_generate_supplementary_tables_success_field(generator, sample_results, temp_dir):
    """Test success field added correctly."""
    table_path = generator.generate_supplementary_tables(sample_results)
    
    with open(table_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 1UBQ and 1CRN should be success (RMSD < 5.0)
    assert rows[0]['success'] == 'Yes'
    assert rows[1]['success'] == 'Yes'
    # 2MR9 should be failure (RMSD = 6.5)
    assert rows[2]['success'] == 'No'


# Test export_for_plotting_tools

def test_export_plotting_tools_csv(generator, sample_results, temp_dir):
    """Test CSV export."""
    exports = generator.export_for_plotting_tools(sample_results, formats=['csv'])
    
    assert 'csv' in exports
    assert Path(exports['csv']).exists()


def test_export_plotting_tools_json(generator, sample_results, temp_dir):
    """Test JSON export."""
    exports = generator.export_for_plotting_tools(sample_results, formats=['json'])
    
    assert 'json' in exports
    assert Path(exports['json']).exists()
    
    # Verify JSON structure
    with open(exports['json'], 'r') as f:
        data = json.load(f)
    
    assert len(data) == 3
    assert data[0]['pdb_id'] == '1UBQ'


def test_export_plotting_tools_excel(generator, sample_results, temp_dir):
    """Test Excel-compatible CSV export."""
    exports = generator.export_for_plotting_tools(sample_results, formats=['excel'])
    
    assert 'excel' in exports
    assert Path(exports['excel']).exists()


def test_export_plotting_tools_multiple_formats(generator, sample_results, temp_dir):
    """Test multiple format export."""
    exports = generator.export_for_plotting_tools(
        sample_results,
        formats=['csv', 'json', 'excel']
    )
    
    assert len(exports) == 3
    assert 'csv' in exports
    assert 'json' in exports
    assert 'excel' in exports


def test_export_plotting_tools_csv_content(generator, sample_results, temp_dir):
    """Test CSV export contains correct data."""
    exports = generator.export_for_plotting_tools(sample_results, formats=['csv'])
    
    with open(exports['csv'], 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 3
    assert rows[0]['pdb_id'] == '1UBQ'
    assert float(rows[0]['rmsd']) == 3.2


# Test create_research_report

def test_create_research_report_basic(generator, sample_phase, sample_results, sample_metadata):
    """Test complete research report creation."""
    report = generator.create_research_report(
        "Test Report",
        sample_phase,
        sample_results,
        sample_metadata
    )
    
    assert isinstance(report, ResearchReport)
    assert report.title == "Test Report"
    assert len(report.figures) == 4
    assert len(report.tables) == 1


def test_create_research_report_timestamp(generator, sample_phase, sample_results, sample_metadata):
    """Test report has timestamp."""
    report = generator.create_research_report(
        "Test Report",
        sample_phase,
        sample_results,
        sample_metadata
    )
    
    assert isinstance(report.generated_timestamp, datetime)


def test_create_research_report_with_statistics(generator, sample_phase, sample_results, sample_metadata):
    """Test report with statistical summary."""
    stats = {
        'correlations': {
            'size_vs_rmsd': 0.55,
            'resolution_vs_rmsd': -0.32
        }
    }
    
    report = generator.create_research_report(
        "Test Report",
        sample_phase,
        sample_results,
        sample_metadata,
        statistical_summary=stats
    )
    
    assert "0.55" in report.statistical_analysis or "0.550" in report.statistical_analysis


def test_create_research_report_conclusions(generator, sample_phase, sample_results, sample_metadata):
    """Test report conclusions generated."""
    report = generator.create_research_report(
        "Test Report",
        sample_phase,
        sample_results,
        sample_metadata
    )
    
    assert len(report.conclusions) > 0
    assert any("success rate" in c.lower() for c in report.conclusions)


# Test ResearchReport dataclass

def test_research_report_immutable():
    """Test ResearchReport is immutable."""
    report = ResearchReport(
        title="Test",
        methodology="Methods",
        results_summary="Results",
        statistical_analysis="Stats",
        figures=[],
        tables=[],
        conclusions=[],
        generated_timestamp=datetime.now()
    )
    
    with pytest.raises(AttributeError):
        report.title = "New Title"  # type: ignore[misc]


# Test helper methods

def test_group_by_category(generator, sample_results):
    """Test grouping helper method."""
    grouped = generator._group_by_category(sample_results, 'size_category', 'rmsd')
    
    assert 'small' in grouped
    assert 'tiny' in grouped
    assert len(grouped['small']) == 2
    assert len(grouped['tiny']) == 1
