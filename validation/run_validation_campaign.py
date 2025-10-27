"""
Command-Line Interface for Large-Scale Validation Campaign

This script provides a CLI for executing validation campaigns with various
configuration options and execution modes.

Features:
- Interactive mode: Step through phases with user approval
- Batch mode: Fully automated execution
- Resume functionality: Continue from checkpoint
- Configuration via arguments or JSON file
- Real-time progress display
- Comprehensive logging

Usage Examples:
    # Interactive mode with default configuration
    python run_validation_campaign.py --interactive
    
    # Batch mode with custom parameters
    python run_validation_campaign.py --batch --proteins 60 --agents 10 --parallel 3
    
    # Resume from checkpoint
    python run_validation_campaign.py --resume ./campaign_results/checkpoint_latest.json
    
    # Load configuration from file
    python run_validation_campaign.py --config my_campaign.json --batch
    
    # Comparative benchmarking mode
    python run_validation_campaign.py --benchmark --proteins 30 --output ./benchmark_results
"""

import argparse
import json
import sys
import os
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.large_scale_validation_campaign import (
    LargeScaleValidationCampaign,
    CampaignConfig,
    CampaignResults
)
from validation.comparative_benchmarking import ComparativeBenchmark


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for campaign execution.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path for persistent logs
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    
    # Setup file handler if requested
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers
    )


# ============================================================================
# Configuration Management
# ============================================================================

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load campaign configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path_obj, 'r') as f:
        config = json.load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    return config


def create_config_from_args(args: argparse.Namespace) -> CampaignConfig:
    """
    Create CampaignConfig from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        CampaignConfig instance
    """
    # Start with defaults
    config_dict = {}
    
    # Override with file config if provided
    if args.config:
        config_dict = load_config_from_file(args.config)
    
    # Override with command-line arguments (highest priority)
    if args.proteins is not None:
        config_dict['target_protein_count'] = args.proteins
    if args.no_qcpp:
        config_dict['enable_qcpp'] = False
    elif 'enable_qcpp' not in config_dict:
        config_dict['enable_qcpp'] = True
    if args.parallel is not None:
        config_dict['max_parallel_tests'] = args.parallel
    if args.agents is not None:
        config_dict['num_agents'] = args.agents
    if args.iterations is not None:
        config_dict['iterations_per_agent'] = args.iterations
    if args.checkpoint_interval is not None:
        config_dict['checkpoint_interval'] = args.checkpoint_interval
    if args.quality_threshold is not None:
        config_dict['quality_gate_threshold'] = args.quality_threshold
    if args.seed is not None:
        config_dict['random_seed'] = args.seed
    if args.output is not None:
        config_dict['output_dir'] = args.output
    
    return CampaignConfig(**config_dict)


def save_config_to_file(config: CampaignConfig, output_path: str) -> None:
    """
    Save campaign configuration to JSON file.
    
    Args:
        config: Campaign configuration to save
        output_path: Output file path
    """
    from dataclasses import asdict
    
    config_dict = asdict(config)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info(f"Saved configuration to {output_path}")


# ============================================================================
# Interactive Mode
# ============================================================================

def run_interactive_mode(campaign: LargeScaleValidationCampaign) -> CampaignResults:
    """
    Run campaign in interactive mode with phase-by-phase approval.
    
    Args:
        campaign: Initialized campaign instance
        
    Returns:
        Campaign results
    """
    print("\n" + "="*80)
    print("INTERACTIVE VALIDATION CAMPAIGN")
    print("="*80)
    
    # Setup campaign
    print("\n[Phase 0] Setting up campaign...")
    campaign.setup_campaign()
    print("✓ Campaign setup complete")
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  - Target proteins: {campaign.config.target_protein_count}")
    print(f"  - QCPP enabled: {campaign.config.enable_qcpp}")
    print(f"  - Max parallel: {campaign.config.max_parallel_tests}")
    print(f"  - Agents per protein: {campaign.config.num_agents}")
    print(f"  - Iterations per agent: {campaign.config.iterations_per_agent}")
    
    # Run phases interactively
    for phase_num in range(1, 5):
        print("\n" + "-"*80)
        print(f"[Phase {phase_num}] Ready to execute")
        
        # Get phase info
        assert campaign._phase_manager is not None
        phase = campaign._phase_manager.get_phase(phase_num)
        assert phase is not None
        print(f"  - Proteins in phase: {len(phase.proteins)}")
        print(f"  - Phase status: {phase.status.value}")
        
        # Ask for approval
        response = input(f"\nExecute Phase {phase_num}? (y/n/q): ").strip().lower()
        
        if response == 'q':
            print("\nCampaign cancelled by user")
            sys.exit(0)
        elif response != 'y':
            print(f"Skipping Phase {phase_num}")
            continue
        
        # Execute phase
        print(f"\n[Phase {phase_num}] Executing...")
        start_time = time.time()
        
        phase_results = campaign.run_phase(phase_num)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Phase {phase_num} complete in {elapsed:.1f}s")
        
        # Display results
        print(f"\nResults:")
        print(f"  - Proteins tested: {phase_results.proteins_tested}")
        print(f"  - Success rate: {phase_results.success_rate:.1f}%")
        print(f"  - Average RMSD: {phase_results.average_rmsd:.2f} Å")
        print(f"  - Average GDT-TS: {phase_results.average_gdt_ts:.1f}")
        print(f"  - Quality gate passed: {phase_results.quality_gate_passed}")
        
        # Check quality gate for Phase 1
        if phase_num == 1 and not phase_results.quality_gate_passed:
            print("\n⚠ WARNING: Phase 1 quality gate FAILED")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("\nCampaign stopped at quality gate")
                break
    
    # Get final results from status
    print("\n" + "="*80)
    print("CAMPAIGN COMPLETE")
    print("="*80)
    
    results = campaign.get_campaign_status()
    
    # Create simple results summary
    from validation.large_scale_validation_campaign import CampaignResults
    campaign_end = datetime.now()
    results_obj = CampaignResults(
        campaign_id=campaign.campaign_id,
        config=campaign.config,
        start_time=results['start_time'],
        end_time=campaign_end,
        total_proteins=results['proteins_tested'],
        phases_completed=results['phases_completed'],
        overall_success_rate=results['overall_success_rate'],
        validation_reports=[],
        phase_summaries=campaign._phase_results if campaign._phase_results else [],
        statistical_analysis_path="",
        failure_analysis_path="",
        final_report_path=str(campaign.output_dir / "interactive_report.md")
    )
    
    print(f"\nFinal Results:")
    print(f"  - Total proteins: {results_obj.total_proteins}")
    print(f"  - Phases completed: {results_obj.phases_completed}")
    print(f"  - Overall success rate: {results_obj.overall_success_rate:.1f}%")
    print(f"  - Final report: {results_obj.final_report_path}")
    
    return results_obj


# ============================================================================
# Batch Mode
# ============================================================================

def run_batch_mode(campaign: LargeScaleValidationCampaign) -> CampaignResults:
    """
    Run campaign in fully automated batch mode.
    
    Args:
        campaign: Initialized campaign instance
        
    Returns:
        Campaign results
    """
    logging.info("="*80)
    logging.info("BATCH VALIDATION CAMPAIGN")
    logging.info("="*80)
    
    # Run campaign
    start_time = time.time()
    results = campaign.run_campaign()
    elapsed = time.time() - start_time
    
    # Log results
    logging.info("="*80)
    logging.info("CAMPAIGN COMPLETE")
    logging.info("="*80)
    logging.info(f"Total runtime: {elapsed/60:.1f} minutes")
    logging.info(f"Total proteins: {results.total_proteins}")
    logging.info(f"Phases completed: {results.phases_completed}")
    logging.info(f"Overall success rate: {results.overall_success_rate:.1f}%")
    logging.info(f"Final report: {results.final_report_path}")
    
    return results


# ============================================================================
# Resume Mode
# ============================================================================

def run_resume_mode(checkpoint_path: str, interactive: bool = False) -> CampaignResults:
    """
    Resume campaign from checkpoint.
    
    Note: Currently simplified - creates new campaign with saved config.
    Full checkpoint/resume functionality to be implemented.
    
    Args:
        checkpoint_path: Path to checkpoint file
        interactive: Whether to run in interactive mode after resume
        
    Returns:
        Campaign results
    """
    logging.info(f"Loading campaign configuration from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with open(checkpoint_path_obj, 'r') as f:
        checkpoint_data = json.load(f)
    
    # Recreate config
    config_data = checkpoint_data.get('config', {})
    config = CampaignConfig(**config_data)
    
    # Create campaign with saved config
    campaign = LargeScaleValidationCampaign(config=config)
    
    logging.info("Campaign initialized with checkpoint configuration")
    logging.info("Note: Full state restoration not yet implemented - starting fresh campaign")
    
    # Continue execution
    if interactive:
        return run_interactive_mode(campaign)
    else:
        return run_batch_mode(campaign)


# ============================================================================
# Benchmark Mode
# ============================================================================

def run_benchmark_mode(args: argparse.Namespace) -> None:
    """
    Run comparative benchmarking between baseline and integrated modes.
    
    Args:
        args: Parsed command-line arguments
    """
    logging.info("="*80)
    logging.info("COMPARATIVE BENCHMARKING MODE")
    logging.info("="*80)
    
    # Create config
    config = create_config_from_args(args)
    
    # Create protein selector and select proteins for benchmarking
    from validation.protein_selector import ProteinSelector
    selector = ProteinSelector()
    proteins = selector.select_proteins(
        target_count=config.target_protein_count,
        size_distribution={'tiny': 0.3, 'small': 0.4, 'medium': 0.3}
    )
    
    # Create benchmark
    benchmark = ComparativeBenchmark(output_dir=config.output_dir)
    
    # Run benchmark
    logging.info(f"Running benchmark with {len(proteins)} proteins...")
    start_time = time.time()
    
    report = benchmark.run_benchmark(
        proteins=proteins,
        num_agents=config.num_agents,
        iterations=config.iterations_per_agent,
        max_parallel=config.max_parallel_tests
    )
    
    elapsed = time.time() - start_time
    
    # Export results
    json_path = Path(config.output_dir) / "benchmark_report.json"
    md_path = Path(config.output_dir) / "benchmark_report.md"
    
    benchmark.export_report(report, str(json_path))
    benchmark.generate_markdown_report(report, str(md_path))
    
    # Log summary
    logging.info("="*80)
    logging.info("BENCHMARK COMPLETE")
    logging.info("="*80)
    logging.info(f"Runtime: {elapsed/60:.1f} minutes")
    logging.info(f"Proteins tested: {len(report.baseline_results)}")
    
    # Get summary statistics from report
    summary = report.summary_statistics
    logging.info(f"Average RMSD improvement: {summary.get('avg_rmsd_improvement', 0):.2f} Å")
    logging.info(f"Average GDT-TS improvement: {summary.get('avg_gdt_ts_improvement', 0):.1f}")
    logging.info(f"Average energy improvement: {summary.get('avg_energy_improvement', 0):.1f} kcal/mol")
    
    # Check for RMSD statistical significance
    rmsd_test = next((t for t in report.statistical_tests if t.metric_name == 'rmsd'), None)
    if rmsd_test:
        logging.info(f"RMSD improvement statistically significant: {rmsd_test.significant} (p={rmsd_test.p_value:.4f})")
    
    logging.info(f"Reports saved to: {config.output_dir}")


# ============================================================================
# Argument Parser
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Large-Scale Protein Structure Validation Campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with defaults
  %(prog)s --interactive
  
  # Batch mode with custom parameters
  %(prog)s --batch --proteins 60 --agents 10 --parallel 3
  
  # Resume from checkpoint
  %(prog)s --resume ./campaign_results/checkpoint_latest.json
  
  # Load configuration from file
  %(prog)s --config campaign.json --batch
  
  # Comparative benchmarking
  %(prog)s --benchmark --proteins 30 --output ./benchmark_results
        """
    )
    
    # Execution mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode with phase-by-phase approval'
    )
    mode_group.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Run in fully automated batch mode'
    )
    mode_group.add_argument(
        '--resume', '-r',
        type=str,
        metavar='CHECKPOINT',
        help='Resume from checkpoint file'
    )
    mode_group.add_argument(
        '--benchmark',
        action='store_true',
        help='Run comparative benchmarking (baseline vs integrated)'
    )
    
    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        metavar='FILE',
        help='Load configuration from JSON file'
    )
    
    # Campaign parameters
    params_group = parser.add_argument_group('Campaign Parameters')
    params_group.add_argument(
        '--proteins', '-p',
        type=int,
        metavar='N',
        help='Target number of proteins to test (50-75)'
    )
    params_group.add_argument(
        '--no-qcpp',
        action='store_true',
        help='Disable QCPP integration (baseline mode)'
    )
    params_group.add_argument(
        '--parallel',
        type=int,
        metavar='N',
        help='Maximum number of parallel test executions'
    )
    params_group.add_argument(
        '--agents',
        type=int,
        metavar='N',
        help='Number of agents per protein prediction'
    )
    params_group.add_argument(
        '--iterations',
        type=int,
        metavar='N',
        help='Iterations per agent'
    )
    params_group.add_argument(
        '--checkpoint-interval',
        type=int,
        metavar='N',
        help='Save checkpoint every N completed tests'
    )
    params_group.add_argument(
        '--quality-threshold',
        type=float,
        metavar='RATE',
        help='Success rate threshold for Phase 1 quality gate (0-1)'
    )
    params_group.add_argument(
        '--seed',
        type=int,
        metavar='N',
        help='Random seed for reproducibility'
    )
    params_group.add_argument(
        '--output', '-o',
        type=str,
        metavar='DIR',
        help='Output directory for campaign results'
    )
    
    # Logging options
    log_group = parser.add_argument_group('Logging Options')
    log_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    log_group.add_argument(
        '--log-file',
        type=str,
        metavar='FILE',
        help='Log file path for persistent logs'
    )
    
    # Save configuration
    parser.add_argument(
        '--save-config',
        type=str,
        metavar='FILE',
        help='Save resolved configuration to JSON file and exit'
    )
    
    return parser


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """
    Main entry point for CLI.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Handle save-config mode
        if args.save_config:
            config = create_config_from_args(args)
            save_config_to_file(config, args.save_config)
            print(f"Configuration saved to {args.save_config}")
            return 0
        
        # Handle benchmark mode
        if args.benchmark:
            run_benchmark_mode(args)
            return 0
        
        # Handle resume mode
        if args.resume:
            interactive = args.interactive if hasattr(args, 'interactive') else False
            results = run_resume_mode(args.resume, interactive)
            print(f"\n✓ Campaign resumed and completed successfully")
            print(f"Final report: {results.final_report_path}")
            return 0
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Create campaign
        campaign = LargeScaleValidationCampaign(config=config)
        
        # Run campaign
        if args.interactive:
            results = run_interactive_mode(campaign)
        else:  # batch mode
            results = run_batch_mode(campaign)
        
        print(f"\n✓ Campaign completed successfully")
        print(f"Final report: {results.final_report_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logging.warning("\nCampaign interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Campaign failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
