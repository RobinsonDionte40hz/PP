"""
Example demonstrations of DocumentationGenerator.

Shows how to generate research documentation, publication figures,
methods sections, supplementary tables, and multi-format exports.

Author: Large-Scale Validation Framework
Date: October 26, 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.documentation_generator import DocumentationGenerator
import json


def example_1_phase_report():
    """Example 1: Generate comprehensive phase summary report."""
    print("\n" + "="*70)
    print("Example 1: Generate Phase Summary Report")
    print("="*70)
    
    generator = DocumentationGenerator(output_dir="./example_docs")
    
    # Sample phase info
    phase = {
        'phase_number': 1,
        'name': 'Initial Validation - Small Proteins',
        'protein_count': 5,
        'success_threshold': 60
    }
    
    # Sample validation results
    results = [
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
        },
        {
            'pdb_id': '1VII',
            'protein_length': 36,
            'size_category': 'tiny',
            'structural_class': 'mixed',
            'resolution': 1.0,
            'helix_fraction': 0.4,
            'sheet_fraction': 0.3,
            'rmsd': 3.8,
            'gdt_ts': 68.5,
            'tm_score': 0.78,
            'energy': -28.9,
            'num_agents': 10,
            'iterations': 1000
        },
        {
            'pdb_id': '1LYZ',
            'protein_length': 129,
            'size_category': 'medium',
            'structural_class': 'mainly-alpha',
            'resolution': 1.9,
            'helix_fraction': 0.5,
            'sheet_fraction': 0.2,
            'rmsd': 4.2,
            'gdt_ts': 62.0,
            'tm_score': 0.68,
            'energy': -55.7,
            'num_agents': 10,
            'iterations': 1000
        }
    ]
    
    # Generate phase report
    report = generator.generate_phase_report(phase, results)
    
    # Save to file
    output_file = generator.output_dir / "reports" / "phase1_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ Phase report generated: {output_file}")
    print("\nReport preview (first 500 characters):")
    print("-" * 70)
    print(report[:500] + "...")


def example_2_publication_figures():
    """Example 2: Generate publication-ready figure specifications."""
    print("\n" + "="*70)
    print("Example 2: Generate Publication Figures")
    print("="*70)
    
    generator = DocumentationGenerator(output_dir="./example_docs")
    
    results = [
        {'pdb_id': '1UBQ', 'rmsd': 3.2, 'gdt_ts': 65.5, 'tm_score': 0.72, 
         'energy': -45.3, 'protein_length': 76, 'size_category': 'small'},
        {'pdb_id': '1CRN', 'rmsd': 4.8, 'gdt_ts': 58.2, 'tm_score': 0.65,
         'energy': -32.1, 'protein_length': 46, 'size_category': 'small'},
        {'pdb_id': '2MR9', 'rmsd': 6.5, 'gdt_ts': 42.0, 'tm_score': 0.48,
         'energy': -18.5, 'protein_length': 35, 'size_category': 'tiny'},
        {'pdb_id': '1VII', 'rmsd': 3.8, 'gdt_ts': 68.5, 'tm_score': 0.78,
         'energy': -28.9, 'protein_length': 36, 'size_category': 'tiny'},
        {'pdb_id': '1LYZ', 'rmsd': 4.2, 'gdt_ts': 62.0, 'tm_score': 0.68,
         'energy': -55.7, 'protein_length': 129, 'size_category': 'medium'}
    ]
    
    # Generate figure specifications
    figures = generator.generate_publication_figures(results)
    
    print(f"\n✅ Generated {len(figures)} figure specifications:")
    for i, fig_path in enumerate(figures, 1):
        print(f"  {i}. {Path(fig_path).name}")
        
        # Show first figure spec
        if i == 1:
            with open(fig_path, 'r') as f:
                spec = json.load(f)
            print(f"\n  Preview of {Path(fig_path).name}:")
            print(f"    Type: {spec['figure_type']}")
            print(f"    Title: {spec['title']}")
            print(f"    Data points: {len(spec.get('data', []))}")


def example_3_methods_section():
    """Example 3: Generate publication methods section."""
    print("\n" + "="*70)
    print("Example 3: Generate Methods Section")
    print("="*70)
    
    generator = DocumentationGenerator(output_dir="./example_docs")
    
    # Test run metadata
    metadata = {
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
        'cpu_model': 'Intel Core i7-12700K',
        'ram_gb': 32,
        'total_runtime_hours': 2.5,
        'random_seed': 42
    }
    
    # Generate methods section
    methods = generator.generate_methods_section(metadata)
    
    # Save to file
    output_file = generator.output_dir / "reports" / "methods_section.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(methods)
    
    print(f"\n✅ Methods section generated: {output_file}")
    print("\nMethods preview (first 600 characters):")
    print("-" * 70)
    print(methods[:600] + "...")


def example_4_supplementary_tables():
    """Example 4: Generate supplementary data tables."""
    print("\n" + "="*70)
    print("Example 4: Generate Supplementary Tables")
    print("="*70)
    
    generator = DocumentationGenerator(output_dir="./example_docs")
    
    results = [
        {
            'pdb_id': '1UBQ', 'protein_length': 76, 'size_category': 'small',
            'structural_class': 'mainly-beta', 'resolution': 1.8,
            'helix_fraction': 0.2, 'sheet_fraction': 0.5,
            'rmsd': 3.2, 'gdt_ts': 65.5, 'tm_score': 0.72,
            'energy': -45.3, 'num_agents': 10, 'iterations': 1000
        },
        {
            'pdb_id': '1CRN', 'protein_length': 46, 'size_category': 'small',
            'structural_class': 'mainly-alpha', 'resolution': 1.5,
            'helix_fraction': 0.6, 'sheet_fraction': 0.1,
            'rmsd': 4.8, 'gdt_ts': 58.2, 'tm_score': 0.65,
            'energy': -32.1, 'num_agents': 10, 'iterations': 1000
        },
        {
            'pdb_id': '2MR9', 'protein_length': 35, 'size_category': 'tiny',
            'structural_class': 'mainly-alpha', 'resolution': 2.0,
            'helix_fraction': 0.7, 'sheet_fraction': 0.0,
            'rmsd': 6.5, 'gdt_ts': 42.0, 'tm_score': 0.48,
            'energy': -18.5, 'num_agents': 10, 'iterations': 1000
        }
    ]
    
    # Generate supplementary table
    table_path = generator.generate_supplementary_tables(results)
    
    print(f"\n✅ Supplementary table generated: {table_path}")
    print("\nTable preview (first 3 rows):")
    print("-" * 70)
    
    # Read and display first few rows
    import csv
    with open(table_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == 0:
                print("Headers:", ", ".join(row.keys()))
            print(f"Row {i+1}: PDB={row['pdb_id']}, RMSD={row['rmsd']}, "
                  f"GDT-TS={row['gdt_ts']}, Success={row['success']}")
            if i >= 2:
                break


def example_5_export_formats():
    """Example 5: Export data in multiple formats."""
    print("\n" + "="*70)
    print("Example 5: Export Data for Plotting Tools")
    print("="*70)
    
    generator = DocumentationGenerator(output_dir="./example_docs")
    
    results = [
        {
            'pdb_id': '1UBQ', 'rmsd': 3.2, 'gdt_ts': 65.5, 'tm_score': 0.72,
            'energy': -45.3, 'protein_length': 76, 'size_category': 'small',
            'resolution': 1.8
        },
        {
            'pdb_id': '1CRN', 'rmsd': 4.8, 'gdt_ts': 58.2, 'tm_score': 0.65,
            'energy': -32.1, 'protein_length': 46, 'size_category': 'small',
            'resolution': 1.5
        },
        {
            'pdb_id': '2MR9', 'rmsd': 6.5, 'gdt_ts': 42.0, 'tm_score': 0.48,
            'energy': -18.5, 'protein_length': 35, 'size_category': 'tiny',
            'resolution': 2.0
        }
    ]
    
    # Export in multiple formats
    exports = generator.export_for_plotting_tools(
        results,
        formats=['csv', 'json', 'excel']
    )
    
    print(f"\n✅ Exported data in {len(exports)} formats:")
    for format_name, file_path in exports.items():
        print(f"  {format_name.upper()}: {file_path}")
        print(f"    Size: {Path(file_path).stat().st_size} bytes")


def example_6_complete_research_report():
    """Example 6: Generate complete research report."""
    print("\n" + "="*70)
    print("Example 6: Generate Complete Research Report")
    print("="*70)
    
    generator = DocumentationGenerator(output_dir="./example_docs")
    
    # Phase info
    phase = {
        'phase_number': 1,
        'name': 'Phase 1: Small Protein Validation',
        'protein_count': 5
    }
    
    # Results
    results = [
        {
            'pdb_id': '1UBQ', 'protein_length': 76, 'size_category': 'small',
            'structural_class': 'mainly-beta', 'resolution': 1.8,
            'helix_fraction': 0.2, 'sheet_fraction': 0.5,
            'rmsd': 3.2, 'gdt_ts': 65.5, 'tm_score': 0.72,
            'energy': -45.3, 'num_agents': 10, 'iterations': 1000
        },
        {
            'pdb_id': '1CRN', 'protein_length': 46, 'size_category': 'small',
            'structural_class': 'mainly-alpha', 'resolution': 1.5,
            'helix_fraction': 0.6, 'sheet_fraction': 0.1,
            'rmsd': 4.8, 'gdt_ts': 58.2, 'tm_score': 0.65,
            'energy': -32.1, 'num_agents': 10, 'iterations': 1000
        },
        {
            'pdb_id': '2MR9', 'protein_length': 35, 'size_category': 'tiny',
            'structural_class': 'mainly-alpha', 'resolution': 2.0,
            'helix_fraction': 0.7, 'sheet_fraction': 0.0,
            'rmsd': 6.5, 'gdt_ts': 42.0, 'tm_score': 0.48,
            'energy': -18.5, 'num_agents': 10, 'iterations': 1000
        }
    ]
    
    # Metadata
    metadata = {
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
        'ram_gb': 32,
        'total_runtime_hours': 1.5,
        'random_seed': 42
    }
    
    # Statistical summary
    statistical_summary = {
        'correlations': {
            'size_vs_rmsd': 0.45,
            'resolution_vs_rmsd': -0.32
        }
    }
    
    # Create complete research report
    report = generator.create_research_report(
        title="Phase 1 Validation Results: Small Protein Predictions",
        phase=phase,
        results=results,
        metadata=metadata,
        statistical_summary=statistical_summary
    )
    
    print(f"\n✅ Complete research report created")
    print(f"\nReport details:")
    print(f"  Title: {report.title}")
    print(f"  Generated: {report.generated_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Figures: {len(report.figures)} files")
    print(f"  Tables: {len(report.tables)} files")
    print(f"  Conclusions: {len(report.conclusions)} statements")
    print(f"\nFirst conclusion:")
    print(f"  '{report.conclusions[0]}'")


def example_7_phase_comparison():
    """Example 7: Generate reports for multiple phases."""
    print("\n" + "="*70)
    print("Example 7: Multi-Phase Report Generation")
    print("="*70)
    
    generator = DocumentationGenerator(output_dir="./example_docs")
    
    # Define multiple phases with results
    phases = [
        {
            'phase': {'phase_number': 1, 'name': 'Phase 1: Tiny Proteins'},
            'results': [
                {'pdb_id': f'TEST{i}', 'rmsd': 3.5 + i*0.5, 'gdt_ts': 65 - i*2,
                 'tm_score': 0.7 - i*0.05, 'energy': -40 + i*5,
                 'protein_length': 35, 'size_category': 'tiny'}
                for i in range(3)
            ]
        },
        {
            'phase': {'phase_number': 2, 'name': 'Phase 2: Small Proteins'},
            'results': [
                {'pdb_id': f'TEST{i+3}', 'rmsd': 4.0 + i*0.5, 'gdt_ts': 60 - i*2,
                 'tm_score': 0.65 - i*0.05, 'energy': -35 + i*5,
                 'protein_length': 50, 'size_category': 'small'}
                for i in range(3)
            ]
        }
    ]
    
    print(f"\n✅ Generating reports for {len(phases)} phases:")
    
    for phase_data in phases:
        phase = phase_data['phase']
        results = phase_data['results']
        
        # Generate report
        report = generator.generate_phase_report(phase, results)
        
        # Calculate success rate
        successes = sum(1 for r in results if r['rmsd'] < 5.0)
        success_rate = (successes / len(results)) * 100
        
        print(f"\n  Phase {phase['phase_number']}: {phase['name']}")
        print(f"    Tests: {len(results)}")
        print(f"    Success rate: {success_rate:.1f}%")
        print(f"    Status: {'✅ PASSED' if success_rate >= 60 else '❌ FAILED'}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("DocumentationGenerator Examples")
    print("="*70)
    print("\nThese examples demonstrate automated research documentation generation.")
    print("All outputs will be saved to ./example_docs/")
    
    try:
        example_1_phase_report()
        example_2_publication_figures()
        example_3_methods_section()
        example_4_supplementary_tables()
        example_5_export_formats()
        example_6_complete_research_report()
        example_7_phase_comparison()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
        print("\nGenerated files are in ./example_docs/")
        print("  - reports/: Markdown reports and methods sections")
        print("  - figures/: JSON figure specifications")
        print("  - tables/: CSV supplementary tables")
        print("  - exports/: Multi-format data exports")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
