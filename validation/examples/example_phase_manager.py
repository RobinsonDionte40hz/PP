"""
Example usage of PhaseManager for Large-Scale Validation

This script demonstrates how to:
1. Initialize phases with protein distribution
2. Execute phase workflow with quality gates
3. Generate phase summary reports
4. Handle parameter adjustments between phases
5. Export/import phase state
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from validation.protein_selector import ProteinSelector
from validation.phase_manager import PhaseManager, PhaseStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def simulate_test_results(phase, success_rate=70.0):
    """
    Simulate test results for a phase.
    
    Args:
        phase: Phase to simulate results for
        success_rate: Desired success rate (0-100)
    
    Returns:
        List of simulated test results
    """
    results = []
    
    for i, protein in enumerate(phase.proteins):
        # Determine if this test succeeds
        success = (i / len(phase.proteins)) < (success_rate / 100.0)
        
        if success:
            # Good results
            result = {
                'pdb_id': protein.pdb_id,
                'success': True,
                'rmsd': 2.0 + (i * 0.5),  # Gradually increasing
                'gdt_ts': 80.0 - (i * 2.0),  # Gradually decreasing
                'tm_score': 0.85 - (i * 0.02),
                'energy': -60.0 + (i * 2.0),
                'execution_time': 100.0 + (i * 10.0)
            }
        else:
            # Failed results
            result = {
                'pdb_id': protein.pdb_id,
                'success': False,
                'rmsd': 8.0,
                'gdt_ts': 30.0,
                'tm_score': 0.3,
                'energy': 10.0,
                'execution_time': 150.0
            }
        
        results.append(result)
    
    return results


def main():
    """Demonstrate PhaseManager usage."""
    
    # ========================================================================
    # Example 1: Initialize phases with protein distribution
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Example 1: Initialize phases")
    logger.info("=" * 70)
    
    # Select proteins
    selector = ProteinSelector()
    proteins = selector.select_proteins(target_count=60)
    
    logger.info(f"\nSelected {len(proteins)} proteins for validation campaign")
    
    # Create phase manager
    manager = PhaseManager(
        phase1_count=10,
        phase2_count=15,
        phase3_count=25,
        quality_gate_threshold=60.0
    )
    
    # Initialize phases
    phases = manager.initialize_phases(proteins)
    
    logger.info(f"\nInitialized {len(phases)} phases:")
    for phase_num, phase in phases.items():
        logger.info(f"  Phase {phase_num}: {phase.protein_count} proteins ({phase.status.value})")
        logger.info(f"    First 3: {[p.pdb_id for p in phase.proteins[:3]]}")
    
    # ========================================================================
    # Example 2: Execute Phase 1 with quality gate
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Execute Phase 1 with quality gate")
    logger.info("=" * 70)
    
    # Get Phase 1
    phase1 = manager.get_current_phase()
    logger.info(f"\nCurrent phase: {phase1.phase_number}")
    logger.info(f"Proteins to test: {phase1.protein_count}")
    
    # Start phase
    manager.start_phase(1)
    logger.info(f"Started Phase 1 at {phase1.start_time}")
    
    # Simulate running tests (in real scenario, would use ValidationSuite)
    logger.info("\nSimulating protein structure prediction tests...")
    results = simulate_test_results(phase1, success_rate=75.0)
    
    # Update phase with results
    manager.update_phase_results(phase1, results)
    
    logger.info(f"\nPhase 1 Results:")
    logger.info(f"  Success Rate: {phase1.success_rate:.1f}%")
    logger.info(f"  Average RMSD: {phase1.average_rmsd:.2f} Å")
    logger.info(f"  Average GDT-TS: {phase1.average_gdt_ts:.1f}")
    logger.info(f"  Average TM-Score: {phase1.average_tm_score:.3f}")
    logger.info(f"  Average Energy: {phase1.average_energy:.1f} kcal/mol")
    logger.info(f"  Failed Proteins: {len(phase1.failed_proteins)}")
    
    # Complete phase
    manager.complete_phase(1)
    logger.info(f"Completed Phase 1 at {phase1.end_time}")
    logger.info(f"Duration: {phase1.get_duration_seconds():.1f} seconds")
    
    # Check quality gate
    gate_result = manager.check_quality_gate(phase1)
    logger.info(f"\n{gate_result.get_summary()}")
    
    if gate_result.passed:
        logger.info("✓ Phase 1 passed quality gate - proceeding to Phase 2")
    else:
        logger.warning("✗ Phase 1 failed quality gate - review needed")
    
    # ========================================================================
    # Example 3: Generate phase summary report
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Generate phase summary report")
    logger.info("=" * 70)
    
    summary = manager.generate_phase_summary(phase1, results)
    
    logger.info(f"\nPhase Summary Report:")
    logger.info(f"  Phase: {summary.phase_number}")
    logger.info(f"  Proteins Tested: {summary.proteins_tested}")
    logger.info(f"  Success Rate: {summary.success_rate:.1f}%")
    logger.info(f"  Execution Time: {summary.execution_time_seconds:.1f} seconds")
    logger.info(f"\n  Top Performers: {summary.top_performers}")
    logger.info(f"  Worst Performers: {summary.worst_performers}")
    
    # Generate markdown report
    markdown = summary.to_markdown()
    logger.info("\nGenerated Markdown Report (first 500 chars):")
    logger.info(markdown[:500] + "...")
    
    # ========================================================================
    # Example 4: Parameter adjustment based on phase results
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Parameter adjustment recommendations")
    logger.info("=" * 70)
    
    adjustments = manager.allow_parameter_adjustment(phase1)
    
    logger.info("\nParameter Adjustment Recommendations:")
    for key, value in adjustments.items():
        logger.info(f"  {key}: {value}")
    
    # ========================================================================
    # Example 5: Advance to Phase 2
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Advance to Phase 2")
    logger.info("=" * 70)
    
    if gate_result.passed:
        advanced = manager.advance_to_next_phase()
        if advanced:
            phase2 = manager.get_current_phase()
            logger.info(f"\nAdvanced to Phase {phase2.phase_number}")
            logger.info(f"  Status: {phase2.status.value}")
            logger.info(f"  Proteins: {phase2.protein_count}")
            logger.info(f"  First 5: {[p.pdb_id for p in phase2.proteins[:5]]}")
    
    # ========================================================================
    # Example 6: Simulate Phase 2 with lower success rate
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 6: Phase 2 with quality gate failure")
    logger.info("=" * 70)
    
    phase2 = manager.get_current_phase()
    manager.start_phase(2)
    
    # Simulate worse performance in Phase 2
    results2 = simulate_test_results(phase2, success_rate=50.0)
    manager.update_phase_results(phase2, results2)
    manager.complete_phase(2)
    
    logger.info(f"\nPhase 2 Results:")
    logger.info(f"  Success Rate: {phase2.success_rate:.1f}%")
    logger.info(f"  Average RMSD: {phase2.average_rmsd:.2f} Å")
    logger.info(f"  Average GDT-TS: {phase2.average_gdt_ts:.1f}")
    
    # Check quality gate
    gate_result2 = manager.check_quality_gate(phase2)
    logger.info(f"\n{gate_result2.get_summary()}")
    
    if not gate_result2.passed:
        logger.warning("\n⚠ Phase 2 failed quality gate!")
        logger.warning("Recommended actions:")
        for rec in gate_result2.recommendations:
            logger.warning(f"  - {rec}")
        
        # Get parameter adjustments
        adjustments2 = manager.allow_parameter_adjustment(phase2)
        logger.warning("\nSuggested parameter adjustments:")
        for key, value in adjustments2.items():
            if '_recommendation' in key:
                logger.warning(f"  {value}")
    
    # ========================================================================
    # Example 7: Export and import phase state
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 7: Export and import phase state")
    logger.info("=" * 70)
    
    # Export current state
    export_path = 'validation/phase_state.json'
    manager.export_phases(export_path)
    logger.info(f"\nExported phase state to {export_path}")
    
    # Create new manager and load state
    new_manager = PhaseManager()
    new_manager.load_phases(export_path)
    
    logger.info(f"Loaded phase state from {export_path}")
    logger.info(f"  Current phase: {new_manager.current_phase_number}")
    logger.info(f"  Total phases: {len(new_manager.phases)}")
    
    # Verify loaded data
    loaded_phase1 = new_manager.get_phase(1)
    logger.info(f"\nVerifying Phase 1 data:")
    logger.info(f"  Status: {loaded_phase1.status.value}")
    logger.info(f"  Success Rate: {loaded_phase1.success_rate:.1f}%")
    logger.info(f"  Proteins: {loaded_phase1.protein_count}")
    
    # ========================================================================
    # Example 8: Complete workflow scenario
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 8: Complete 4-phase workflow")
    logger.info("=" * 70)
    
    # Create fresh manager for complete workflow
    workflow_manager = PhaseManager(
        phase1_count=5,
        phase2_count=8,
        phase3_count=12,
        quality_gate_threshold=60.0
    )
    
    workflow_proteins = selector.select_proteins(target_count=30)
    workflow_manager.initialize_phases(workflow_proteins)
    
    logger.info("\nExecuting 4-phase validation workflow:")
    
    for phase_num in range(1, 5):
        phase = workflow_manager.get_phase(phase_num)
        
        logger.info(f"\n--- Phase {phase_num} ---")
        logger.info(f"Proteins: {phase.protein_count}")
        
        # Start phase
        workflow_manager.start_phase(phase_num)
        
        # Simulate tests (gradually decreasing success rate)
        success_rate = 80.0 - (phase_num * 5.0)  # 80%, 75%, 70%, 65%
        results = simulate_test_results(phase, success_rate=success_rate)
        
        # Update and complete
        workflow_manager.update_phase_results(phase, results)
        workflow_manager.complete_phase(phase_num)
        
        # Check quality gate
        gate = workflow_manager.check_quality_gate(phase)
        
        logger.info(f"Success Rate: {phase.success_rate:.1f}%")
        logger.info(f"Quality Gate: {'✓ PASSED' if gate.passed else '✗ FAILED'}")
        
        # Advance to next phase (if not last)
        if phase_num < 4:
            workflow_manager.advance_to_next_phase()
    
    logger.info("\n" + "=" * 70)
    logger.info("Workflow Summary")
    logger.info("=" * 70)
    
    logger.info("\nAll phases completed:")
    for phase_num in range(1, 5):
        phase = workflow_manager.get_phase(phase_num)
        logger.info(f"  Phase {phase_num}: {phase.success_rate:.1f}% success, "
                   f"{phase.protein_count} proteins, "
                   f"{phase.get_duration_seconds():.1f}s duration")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    
    logger.info("""
The PhaseManager enables progressive testing with quality gates:

1. Organize proteins into 4 phases by difficulty
2. Track phase status and execution metrics
3. Check quality gates (60% success threshold)
4. Generate comprehensive phase summary reports
5. Recommend parameter adjustments based on results
6. Export/import phase state for reproducibility
7. Support iterative research workflow

Key features:
- Difficulty-based protein sorting
- Automated quality gate checking
- Parameter adjustment recommendations
- Comprehensive metrics tracking
- Phase transition management
- Checkpoint/resume support

Ready for Task 3: Implement ResultsRepository for centralized storage
""")


if __name__ == '__main__':
    main()
