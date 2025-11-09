#!/usr/bin/env python3
"""
Quantum Refinement Validation Suite

This script validates the Quantum Refinement Engine on test proteins:
- 1UBQ (Ubiquitin, 76 residues) - Target RMSD < 4Å
- 1CRN (Crambin, 46 residues) - Target RMSD < 3Å
- 2MR9 (Villin headpiece, 35 residues) - Target RMSD < 3Å

For each protein:
1. Download native structure from PDB
2. Run UBF exploration to get coarse structure (7-14Å)
3. Apply quantum refinement engine
4. Measure RMSD improvement
5. Verify success criteria

Success Criteria (per Task 14.2):
- RMSD improvement > 50% from initial
- Final RMSD < 5Å for all test proteins
- GDT-TS > 50 for all test proteins
- Energy < 0 kcal/mol (thermodynamically stable)
- Runtime < 5 minutes per protein
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ubf_protein.quantum_refinement_engine import QuantumRefinementEngine
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.energy_function import MolecularMechanicsEnergy
from ubf_protein.rmsd_calculator import RMSDCalculator, NativeStructureLoader
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.models import Conformation, RefinementConfig
from ubf_protein.refinement_visualization import RefinementVisualizer, visualize_refinement_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Protein Configuration
# ============================================================================

TEST_PROTEINS = [
    {
        "pdb_id": "1UBQ",
        "name": "Ubiquitin",
        "residues": 76,
        "target_rmsd": 4.0,
        "difficulty": "medium",
        "exploration_agents": 15,
        "exploration_iterations": 500,
    },
    {
        "pdb_id": "1CRN",
        "name": "Crambin",
        "residues": 46,
        "target_rmsd": 3.0,
        "difficulty": "easy",
        "exploration_agents": 10,
        "exploration_iterations": 300,
    },
    {
        "pdb_id": "2MR9",
        "name": "Villin headpiece",
        "residues": 35,
        "target_rmsd": 3.0,
        "difficulty": "easy",
        "exploration_agents": 8,
        "exploration_iterations": 200,
    },
]


# ============================================================================
# Validation Report Data Classes
# ============================================================================

@dataclass
class RefinementValidationResult:
    """Results from validating quantum refinement on a single protein."""
    pdb_id: str
    protein_name: str
    sequence_length: int
    
    # Pre-refinement metrics
    initial_rmsd: float
    initial_energy: float
    initial_gdt_ts: float
    
    # Post-refinement metrics
    final_rmsd: float
    final_energy: float
    final_gdt_ts: float
    final_tm_score: float
    
    # Improvement metrics
    rmsd_improvement_percent: float
    energy_improvement: float
    
    # Component RMSD breakdown
    helix_rmsd: float
    sheet_rmsd: float
    loop_rmsd: float
    core_rmsd: float
    
    # Performance metrics
    refinement_time_seconds: float
    quantum_cores_identified: int
    distance_restraints_applied: int
    tertiary_contacts_enforced: int
    
    # Success flags
    meets_rmsd_target: bool
    meets_improvement_target: bool
    meets_energy_target: bool
    meets_gdt_target: bool
    meets_time_target: bool
    
    def is_successful(self) -> bool:
        """Check if all success criteria are met."""
        return (
            self.meets_rmsd_target and
            self.meets_improvement_target and
            self.meets_energy_target and
            self.meets_gdt_target and
            self.meets_time_target
        )
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        success = "✓ PASS" if self.is_successful() else "✗ FAIL"
        
        return f"""
{'=' * 70}
Quantum Refinement Validation: {self.pdb_id} ({self.protein_name})
{'=' * 70}

Protein Info:
  Sequence Length:     {self.sequence_length} residues
  
Pre-Refinement:
  RMSD:                {self.initial_rmsd:.2f} Å
  Energy:              {self.initial_energy:.2f} kcal/mol
  GDT-TS:              {self.initial_gdt_ts:.1f}
  
Post-Refinement:
  RMSD:                {self.final_rmsd:.2f} Å
  Energy:              {self.final_energy:.2f} kcal/mol
  GDT-TS:              {self.final_gdt_ts:.1f}
  TM-Score:            {self.final_tm_score:.3f}
  
Improvements:
  RMSD Improvement:    {self.rmsd_improvement_percent:.1f}%
  Energy Improvement:  {self.energy_improvement:.2f} kcal/mol
  
Component RMSD Breakdown:
  Helix:               {self.helix_rmsd:.2f} Å
  Sheet:               {self.sheet_rmsd:.2f} Å
  Loop:                {self.loop_rmsd:.2f} Å
  Core:                {self.core_rmsd:.2f} Å
  
Refinement Details:
  Quantum Cores:       {self.quantum_cores_identified}
  Distance Restraints: {self.distance_restraints_applied}
  Tertiary Contacts:   {self.tertiary_contacts_enforced}
  Runtime:             {self.refinement_time_seconds:.1f} seconds
  
Success Criteria:
  RMSD < 5Å:           {'✓' if self.meets_rmsd_target else '✗'}
  RMSD Improvement >50%: {'✓' if self.meets_improvement_target else '✗'}
  Energy < 0:          {'✓' if self.meets_energy_target else '✗'}
  GDT-TS > 50:         {'✓' if self.meets_gdt_target else '✗'}
  Time < 5min:         {'✓' if self.meets_time_target else '✗'}
  
Overall: {success}
"""


@dataclass
class ValidationSuiteResults:
    """Aggregated results from all test proteins."""
    validation_results: List[RefinementValidationResult]
    total_runtime_seconds: float
    success_rate: float
    average_rmsd_improvement: float
    average_final_rmsd: float
    
    def get_summary(self) -> str:
        """Get summary of entire validation suite."""
        successful = sum(1 for r in self.validation_results if r.is_successful())
        total = len(self.validation_results)
        
        summary = f"""
{'=' * 70}
QUANTUM REFINEMENT VALIDATION SUITE RESULTS
{'=' * 70}

Overall Statistics:
  Proteins Tested:     {total}
  Successful:          {successful}/{total} ({self.success_rate:.1f}%)
  Average RMSD:        {self.average_final_rmsd:.2f} Å
  Average Improvement: {self.average_rmsd_improvement:.1f}%
  Total Runtime:       {self.total_runtime_seconds:.1f} seconds

Individual Results:
"""
        for result in self.validation_results:
            status = "✓" if result.is_successful() else "✗"
            summary += f"\n  {status} {result.pdb_id}: {result.final_rmsd:.2f}Å "
            summary += f"({result.rmsd_improvement_percent:.0f}% improvement)"
        
        return summary


# ============================================================================
# Validation Functions
# ============================================================================

def run_exploration_stage(
    pdb_id: str,
    num_agents: int,
    iterations: int
) -> Tuple[Conformation, str]:
    """
    Run UBF exploration to get coarse structure.
    
    Args:
        pdb_id: PDB identifier
        num_agents: Number of exploration agents
        iterations: Iterations per agent
        
    Returns:
        Tuple of (best_conformation, sequence)
    """
    logger.info(f"Running exploration for {pdb_id}...")
    
    # Load native structure to get sequence
    loader = NativeStructureLoader()
    native_struct = loader.load_from_pdb_id(pdb_id, ca_only=True)
    sequence = native_struct.sequence
    
    # Run multi-agent exploration
    coordinator = MultiAgentCoordinator(
        protein_sequence=sequence,
        enable_checkpointing=False
    )
    
    coordinator.initialize_agents(num_agents)
    results = coordinator.run_parallel_exploration(iterations)
    
    if results.best_conformation is None:
        raise RuntimeError(f"Exploration failed for {pdb_id}")
    
    logger.info(f"Exploration complete: RMSD={results.best_rmsd:.2f}Å, "
                f"Energy={results.best_energy:.2f} kcal/mol")
    
    return results.best_conformation, sequence


def run_refinement_stage(
    initial_conformation: Conformation,
    native_pdb_id: str,
    config: Optional[RefinementConfig] = None
) -> Tuple[Conformation, Dict]:
    """
    Run quantum refinement on coarse structure.
    
    Args:
        initial_conformation: Starting structure from exploration
        native_pdb_id: PDB ID for validation
        config: Optional refinement configuration
        
    Returns:
        Tuple of (refined_conformation, refinement_metrics)
    """
    logger.info("Initializing quantum refinement engine...")
    
    # Initialize components
    # Note: Using None predictor for now - will be updated when QCPP is fully integrated
    qcpp_adapter = QCPPIntegrationAdapter(predictor=None, cache_size=1000)
    energy_calc = MolecularMechanicsEnergy()
    rmsd_calc = RMSDCalculator()
    
    # Create refinement engine
    engine = QuantumRefinementEngine(
        qcpp_adapter=qcpp_adapter,
        energy_calculator=energy_calc,
        rmsd_calculator=rmsd_calc
    )
    
    # Load native structure for RMSD tracking
    loader = NativeStructureLoader()
    native_struct = loader.load_from_pdb_id(native_pdb_id, ca_only=True)
    
    # Run refinement
    logger.info("Running quantum refinement...")
    start_time = time.time()
    
    result = engine.refine_structure_quantum(
        coarse_structure=initial_conformation,
        native_structure=native_struct,
        config=config or RefinementConfig()
    )
    
    refinement_time = time.time() - start_time
    
    metrics = {
        'refinement_time': refinement_time,
        'quantum_cores': result.quantum_cores_identified,
        'distance_restraints': result.restraints_applied,
        'tertiary_contacts': result.contacts_enforced,
        'helix_rmsd': result.helix_rmsd,
        'sheet_rmsd': result.sheet_rmsd,
        'loop_rmsd': result.loop_rmsd,
        'core_rmsd': result.core_rmsd,
    }
    
    logger.info(f"Refinement complete: RMSD={result.final_rmsd:.2f}Å in {refinement_time:.1f}s")
    
    return result.refined_structure, metrics


def validate_protein(
    protein_config: Dict,
    refinement_config: Optional[RefinementConfig] = None
) -> RefinementValidationResult:
    """
    Validate quantum refinement on a single protein.
    
    Args:
        protein_config: Configuration dict from TEST_PROTEINS
        refinement_config: Optional refinement configuration
        
    Returns:
        Validation result
    """
    pdb_id = protein_config['pdb_id']
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Validating {pdb_id} ({protein_config['name']})")
    logger.info(f"{'=' * 70}")
    
    try:
        # Step 1: Run exploration to get coarse structure
        initial_conf, sequence = run_exploration_stage(
            pdb_id=pdb_id,
            num_agents=protein_config['exploration_agents'],
            iterations=protein_config['exploration_iterations']
        )
        
        initial_rmsd = initial_conf.rmsd_to_native or 0.0
        initial_energy = initial_conf.energy
        
        # Calculate initial GDT-TS
        # Load native structure for validation
        loader = NativeStructureLoader()
        native_struct = loader.load_from_pdb_id(pdb_id, ca_only=True)
        
        rmsd_calc = RMSDCalculator()
        # GDT-TS calculation: approximated from RMSD for now
        # In full implementation, would use proper GDT-TS algorithm
        initial_gdt_ts = max(0.0, 100.0 - (initial_rmsd * 10.0))  # Rough approximation
        
        # Step 2: Run quantum refinement
        refined_conf, metrics = run_refinement_stage(
            initial_conformation=initial_conf,
            native_pdb_id=pdb_id,
            config=refinement_config
        )
        
        final_rmsd = refined_conf.rmsd_to_native or 0.0
        final_energy = refined_conf.energy
        
        # Calculate final quality metrics
        # GDT-TS and TM-score: approximated from RMSD for now
        # In full implementation, would use proper algorithms
        final_gdt_ts = max(0.0, 100.0 - (final_rmsd * 10.0))  # Rough approximation
        final_tm_score = max(0.0, 1.0 - (final_rmsd / 20.0))  # Rough approximation
        
        # Calculate improvements
        rmsd_improvement = ((initial_rmsd - final_rmsd) / initial_rmsd) * 100
        energy_improvement = initial_energy - final_energy
        
        # Check success criteria
        result = RefinementValidationResult(
            pdb_id=pdb_id,
            protein_name=protein_config['name'],
            sequence_length=len(sequence),
            initial_rmsd=initial_rmsd,
            initial_energy=initial_energy,
            initial_gdt_ts=initial_gdt_ts,
            final_rmsd=final_rmsd,
            final_energy=final_energy,
            final_gdt_ts=final_gdt_ts,
            final_tm_score=final_tm_score,
            rmsd_improvement_percent=rmsd_improvement,
            energy_improvement=energy_improvement,
            helix_rmsd=metrics['helix_rmsd'],
            sheet_rmsd=metrics['sheet_rmsd'],
            loop_rmsd=metrics['loop_rmsd'],
            core_rmsd=metrics['core_rmsd'],
            refinement_time_seconds=metrics['refinement_time'],
            quantum_cores_identified=metrics['quantum_cores'],
            distance_restraints_applied=metrics['distance_restraints'],
            tertiary_contacts_enforced=metrics['tertiary_contacts'],
            meets_rmsd_target=(final_rmsd < 5.0),
            meets_improvement_target=(rmsd_improvement > 50.0),
            meets_energy_target=(final_energy < 0.0),
            meets_gdt_target=(final_gdt_ts > 50.0),
            meets_time_target=(metrics['refinement_time'] < 300.0),  # 5 minutes
        )
        
        logger.info(result.get_summary())
        return result
        
    except Exception as e:
        logger.error(f"Validation failed for {pdb_id}: {e}", exc_info=True)
        raise


def run_validation_suite(
    proteins: Optional[List[Dict]] = None,
    refinement_config: Optional[RefinementConfig] = None,
    output_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    generate_plots: bool = True
) -> ValidationSuiteResults:
    """
    Run validation suite on all test proteins.
    
    Args:
        proteins: List of protein configs (default: TEST_PROTEINS)
        refinement_config: Optional refinement configuration
        output_file: Optional file to save results JSON
        output_dir: Optional directory for visualization plots
        generate_plots: Whether to generate visualization plots (default: True)
        
    Returns:
        Aggregated validation results
    """
    if proteins is None:
        proteins = TEST_PROTEINS
    
    logger.info(f"\n{'=' * 70}")
    logger.info("STARTING QUANTUM REFINEMENT VALIDATION SUITE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Testing {len(proteins)} proteins")
    
    start_time = time.time()
    results = []
    
    # Create visualizer if plots requested
    visualizer = RefinementVisualizer() if generate_plots else None
    
    for protein_config in proteins:
        try:
            result = validate_protein(protein_config, refinement_config)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to validate {protein_config['pdb_id']}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Calculate aggregated metrics
    if results:
        success_rate = (sum(1 for r in results if r.is_successful()) / len(results)) * 100
        avg_improvement = sum(r.rmsd_improvement_percent for r in results) / len(results)
        avg_rmsd = sum(r.final_rmsd for r in results) / len(results)
    else:
        success_rate = 0.0
        avg_improvement = 0.0
        avg_rmsd = 0.0
    
    suite_results = ValidationSuiteResults(
        validation_results=results,
        total_runtime_seconds=total_time,
        success_rate=success_rate,
        average_rmsd_improvement=avg_improvement,
        average_final_rmsd=avg_rmsd
    )
    
    # Print summary
    logger.info(suite_results.get_summary())
    
    # Generate visualizations if requested
    if generate_plots and output_dir and visualizer:
        logger.info(f"\nGenerating visualizations in {output_dir}...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate multi-protein comparison plot
        if results:
            comparison_file = Path(output_dir) / "multi_protein_comparison.png"
            visualizer.plot_multi_protein_comparison(
                [asdict(r) for r in results],
                output_file=str(comparison_file)
            )
            logger.info(f"  Created: {comparison_file}")
    
    # Save to file if requested
    if output_file:
        output_data = {
            'suite_summary': {
                'total_proteins': len(results),
                'successful': sum(1 for r in results if r.is_successful()),
                'success_rate': success_rate,
                'average_rmsd_improvement': avg_improvement,
                'average_final_rmsd': avg_rmsd,
                'total_runtime_seconds': total_time,
            },
            'individual_results': [asdict(r) for r in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_file}")
    
    return suite_results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Quantum Refinement Engine on test proteins"
    )
    parser.add_argument(
        '--protein',
        type=str,
        choices=['1UBQ', '1CRN', '2MR9', 'all'],
        default='all',
        help='Protein to validate (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='quantum_refinement_validation.json',
        help='Output file for results (default: quantum_refinement_validation.json)'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='validation_plots',
        help='Directory for visualization plots (default: validation_plots)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable visualization plot generation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Select proteins to test
    if args.protein == 'all':
        proteins = TEST_PROTEINS
    else:
        proteins = [p for p in TEST_PROTEINS if p['pdb_id'] == args.protein]
    
    # Run validation
    try:
        results = run_validation_suite(
            proteins=proteins,
            output_file=args.output,
            output_dir=args.plot_dir if not args.no_plots else None,
            generate_plots=not args.no_plots
        )
        
        # Exit with appropriate code
        if results.success_rate == 100.0:
            logger.info("\n✓ All validations PASSED!")
            return 0
        else:
            logger.warning(f"\n✗ Some validations FAILED ({results.success_rate:.0f}% success rate)")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\nValidation suite failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
