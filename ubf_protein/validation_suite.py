"""
Validation Suite for UBF Protein Prediction System

This module provides comprehensive validation framework for testing UBF predictions
against known protein structures from the PDB database.

Includes:
- ValidationReport: Results from validating a single protein
- TestSuiteResults: Aggregated results from multiple proteins
- ComparisonReport: Comparison to baseline methods
- ValidationSuite: Main class for running validations
"""

import json
import time
import logging
import os
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from .models import Conformation
from .protein_agent import ProteinAgent
from .multi_agent_coordinator import MultiAgentCoordinator
from .rmsd_calculator import NativeStructureLoader, RMSDCalculator

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Validation Results
# ============================================================================

@dataclass
class ValidationReport:
    """Results from validating a single protein"""
    pdb_id: str
    sequence_length: int
    best_rmsd: float  # Ångströms
    best_energy: float  # kcal/mol
    gdt_ts_score: float  # 0-100
    tm_score: float  # 0-1
    runtime_seconds: float
    conformations_explored: int
    num_agents: int
    iterations_per_agent: int
    
    def is_successful(self) -> bool:
        """
        Determine if prediction is successful.
        
        Criteria:
        - RMSD < 5.0 Å (acceptable)
        - Energy < 0 (thermodynamically stable) 
        - GDT-TS > 50 (correct fold)
        
        Returns:
            True if prediction meets success criteria
        """
        return (self.best_rmsd < 5.0 and 
                self.best_energy < 0 and 
                self.gdt_ts_score > 50)
    
    def assess_quality(self) -> str:
        """
        Assess prediction quality based on RMSD and GDT-TS.
        
        Quality levels:
        - Excellent: RMSD < 2.0Å, GDT-TS > 80
        - Good: RMSD < 3.0Å, GDT-TS > 70
        - Acceptable: RMSD < 5.0Å, GDT-TS > 50
        - Poor: RMSD >= 5.0Å or GDT-TS <= 50
        
        Returns:
            Quality assessment string
        """
        if self.best_rmsd < 2.0 and self.gdt_ts_score > 80:
            return "excellent"
        elif self.best_rmsd < 3.0 and self.gdt_ts_score > 70:
            return "good"
        elif self.best_rmsd < 5.0 and self.gdt_ts_score > 50:
            return "acceptable"
        else:
            return "poor"
    
    def get_summary(self) -> str:
        """Get human-readable summary of validation results."""
        quality = self.assess_quality()
        success = "✓" if self.is_successful() else "✗"
        
        summary = f"""
Validation Report: {self.pdb_id}
{'=' * 60}
Sequence Length:       {self.sequence_length} residues
Best RMSD:            {self.best_rmsd:.2f} Å
Best Energy:          {self.best_energy:.2f} kcal/mol
GDT-TS Score:         {self.gdt_ts_score:.1f}
TM-Score:             {self.tm_score:.3f}
Quality Assessment:   {quality.upper()}
Successful:           {success}
Runtime:              {self.runtime_seconds:.1f} seconds
Conformations:        {self.conformations_explored}
Agents:               {self.num_agents}
Iterations/Agent:     {self.iterations_per_agent}
{'=' * 60}
"""
        return summary


@dataclass
class BaselineResult:
    """Results from a baseline method for comparison"""
    method_name: str  # "random_sampling" or "monte_carlo"
    best_rmsd: float
    best_energy: float
    runtime_seconds: float
    conformations_sampled: int


@dataclass
class ComparisonReport:
    """Comparison of UBF results to baseline methods"""
    pdb_id: str
    ubf_rmsd: float
    baselines: List[BaselineResult]
    
    def get_improvement_summary(self) -> Dict[str, float]:
        """
        Calculate improvement over each baseline method.
        
        Returns:
            Dictionary mapping method name to % improvement in RMSD
        """
        improvements = {}
        for baseline in self.baselines:
            if baseline.best_rmsd > 0:
                improvement_pct = ((baseline.best_rmsd - self.ubf_rmsd) / baseline.best_rmsd) * 100
                improvements[baseline.method_name] = improvement_pct
        return improvements


@dataclass
class TestSuiteResults:
    """Aggregated results from validating multiple proteins"""
    validation_reports: List[ValidationReport]
    total_runtime_seconds: float
    success_rate: float  # Percentage of successful predictions
    average_rmsd: float
    average_gdt_ts: float
    quality_distribution: Dict[str, int]  # Count of excellent/good/acceptable/poor
    
    def get_summary(self) -> str:
        """Get human-readable summary of test suite results."""
        summary = f"""
Test Suite Results
{'=' * 60}
Proteins Tested:      {len(self.validation_reports)}
Success Rate:         {self.success_rate:.1f}%
Average RMSD:         {self.average_rmsd:.2f} Å
Average GDT-TS:       {self.average_gdt_ts:.1f}
Total Runtime:        {self.total_runtime_seconds:.1f} seconds

Quality Distribution:
"""
        for quality, count in self.quality_distribution.items():
            percentage = (count / len(self.validation_reports)) * 100
            summary += f"  {quality.capitalize():12s} {count:2d} ({percentage:5.1f}%)\n"
        
        summary += "=" * 60 + "\n"
        return summary


# ============================================================================
# Validation Suite Main Class
# ============================================================================

class ValidationSuite:
    """
    Comprehensive validation framework for testing UBF predictions
    against known protein structures.
    
    Usage:
        suite = ValidationSuite()
        report = suite.validate_protein("1UBQ", num_agents=10, iterations=1000)
        results = suite.run_test_suite()
    """
    
    def __init__(self, test_set_path: Optional[str] = None, pdb_cache_dir: str = "./pdb_cache"):
        """
        Initialize validation suite.
        
        Args:
            test_set_path: Path to JSON file with test protein definitions.
                          If None, uses default validation_proteins.json in same directory.
            pdb_cache_dir: Directory for caching downloaded PDB files
        """
        # Set default test set path
        if test_set_path is None:
            current_dir = os.path.dirname(__file__)
            test_set_path = os.path.join(current_dir, "validation_proteins.json")
        
        self.test_set_path = test_set_path
        self.pdb_cache_dir = pdb_cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(pdb_cache_dir, exist_ok=True)
        
        # Initialize native structure loader
        self.native_loader = NativeStructureLoader(cache_dir=pdb_cache_dir)
        
        # Load test set
        self.test_proteins = self._load_test_set()
        
        logger.info(f"ValidationSuite initialized with {len(self.test_proteins)} test proteins")
    
    def _load_test_set(self) -> List[Dict]:
        """
        Load test protein definitions from JSON file.
        
        Returns:
            List of protein definitions
        """
        try:
            with open(self.test_set_path, 'r') as f:
                data = json.load(f)
                return data.get("validation_proteins", [])
        except FileNotFoundError:
            logger.warning(f"Test set file not found: {self.test_set_path}")
            logger.info("Using empty test set. Create validation_proteins.json to add test proteins.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing test set JSON: {e}")
            return []
    
    def validate_protein(self,
                        pdb_id: str,
                        num_agents: int = 10,
                        iterations: int = 1000,
                        use_multi_agent: bool = True) -> ValidationReport:
        """
        Run full validation on a single protein.
        
        Args:
            pdb_id: PDB identifier (e.g., "1UBQ")
            num_agents: Number of agents for multi-agent exploration
            iterations: Number of iterations per agent
            use_multi_agent: If True, use multi-agent coordinator; if False, single agent
        
        Returns:
            ValidationReport with all metrics
            
        Raises:
            ValueError: If protein cannot be loaded or validation fails
        """
        logger.info(f"Starting validation for {pdb_id}")
        start_time = time.time()
        
        try:
            # Load native structure
            logger.info(f"Loading native structure for {pdb_id}")
            native_struct_obj = self.native_loader.load_from_pdb_id(pdb_id, ca_only=True)
            
            # Convert NativeStructure to Conformation for compatibility
            native_structure = Conformation(
                conformation_id=f"native_{pdb_id}",
                sequence=native_struct_obj.sequence,
                atom_coordinates=native_struct_obj.ca_coords,
                energy=-100.0,  # Placeholder native energy
                rmsd_to_native=0.0,  # Native has 0 RMSD to itself
                secondary_structure=['C'] * len(native_struct_obj.sequence),
                phi_angles=[0.0] * len(native_struct_obj.sequence),
                psi_angles=[0.0] * len(native_struct_obj.sequence),
                available_move_types=[],
                structural_constraints={},
                native_structure_ref=pdb_id
            )
            
            # Extract sequence
            sequence = native_structure.sequence
            logger.info(f"Sequence length: {len(sequence)} residues")
            
            # Run exploration with native structure for RMSD validation
            if use_multi_agent:
                logger.info(f"Running multi-agent exploration ({num_agents} agents, {iterations} iterations each)")
                coordinator = MultiAgentCoordinator(
                    protein_sequence=sequence,
                    enable_checkpointing=False
                )
                
                # Initialize agents with native structure
                agents = []
                for i in range(num_agents):
                    agent = ProteinAgent(
                        protein_sequence=sequence,
                        native_structure=native_structure,
                        enable_visualization=False
                    )
                    agents.append(agent)
                coordinator._agents = agents
                
                results = coordinator.run_parallel_exploration(iterations)
                
                best_conformation = results.best_conformation
                conformations_explored = results.total_conformations_explored
                
            else:
                logger.info(f"Running single-agent exploration ({iterations} iterations)")
                agent = ProteinAgent(
                    protein_sequence=sequence,
                    native_structure=native_structure,
                    enable_visualization=False
                )
                
                # Run exploration
                for _ in range(iterations):
                    agent.explore_step()
                
                # Get best conformation
                metrics = agent.get_exploration_metrics()
                best_conformation = agent._current_conformation  # Access current best
                conformations_explored = metrics["conformations_explored"]
            
            # Extract metrics from best conformation
            if best_conformation is None:
                raise ValueError(f"No valid conformation found for {pdb_id}")
            
            best_rmsd = best_conformation.rmsd_to_native if best_conformation.rmsd_to_native is not None else float('inf')
            best_energy = best_conformation.energy
            gdt_ts = best_conformation.gdt_ts_score if best_conformation.gdt_ts_score is not None else 0.0
            tm_score = best_conformation.tm_score if best_conformation.tm_score is not None else 0.0
            
            runtime = time.time() - start_time
            
            # Create validation report
            report = ValidationReport(
                pdb_id=pdb_id,
                sequence_length=len(sequence),
                best_rmsd=best_rmsd,
                best_energy=best_energy,
                gdt_ts_score=gdt_ts,
                tm_score=tm_score,
                runtime_seconds=runtime,
                conformations_explored=int(conformations_explored),  # Convert to int
                num_agents=num_agents if use_multi_agent else 1,
                iterations_per_agent=iterations
            )
            
            logger.info(f"Validation complete for {pdb_id}: Quality = {report.assess_quality()}")
            return report
            
        except Exception as e:
            logger.error(f"Validation failed for {pdb_id}: {e}")
            raise ValueError(f"Failed to validate {pdb_id}: {e}")
    
    def run_test_suite(self,
                      num_agents: int = 10,
                      iterations: int = 1000) -> TestSuiteResults:
        """
        Run validation on entire test set.
        
        Args:
            num_agents: Number of agents per protein
            iterations: Number of iterations per agent
            
        Returns:
            TestSuiteResults with aggregated metrics
        """
        logger.info(f"Starting test suite with {len(self.test_proteins)} proteins")
        start_time = time.time()
        
        validation_reports = []
        
        for protein_def in self.test_proteins:
            pdb_id = protein_def.get("pdb_id")
            if not pdb_id:
                logger.warning(f"Skipping protein with missing pdb_id: {protein_def}")
                continue
            
            try:
                report = self.validate_protein(pdb_id, num_agents, iterations)
                validation_reports.append(report)
            except Exception as e:
                logger.error(f"Failed to validate {pdb_id}: {e}")
                # Continue with other proteins
        
        # Calculate aggregate metrics
        total_runtime = time.time() - start_time
        
        if not validation_reports:
            logger.warning("No successful validations in test suite")
            return TestSuiteResults(
                validation_reports=[],
                total_runtime_seconds=total_runtime,
                success_rate=0.0,
                average_rmsd=float('inf'),
                average_gdt_ts=0.0,
                quality_distribution={}
            )
        
        # Success rate
        successful_count = sum(1 for r in validation_reports if r.is_successful())
        success_rate = (successful_count / len(validation_reports)) * 100
        
        # Average metrics
        average_rmsd = sum(r.best_rmsd for r in validation_reports) / len(validation_reports)
        average_gdt_ts = sum(r.gdt_ts_score for r in validation_reports) / len(validation_reports)
        
        # Quality distribution
        quality_distribution = {
            "excellent": 0,
            "good": 0,
            "acceptable": 0,
            "poor": 0
        }
        for report in validation_reports:
            quality = report.assess_quality()
            quality_distribution[quality] += 1
        
        results = TestSuiteResults(
            validation_reports=validation_reports,
            total_runtime_seconds=total_runtime,
            success_rate=success_rate,
            average_rmsd=average_rmsd,
            average_gdt_ts=average_gdt_ts,
            quality_distribution=quality_distribution
        )
        
        logger.info(f"Test suite complete: {success_rate:.1f}% success rate")
        return results
    
    def compare_to_baseline(self,
                           pdb_id: str,
                           num_samples: int = 1000) -> ComparisonReport:
        """
        Compare UBF results to baseline methods.
        
        Baseline methods:
        1. Random Sampling: Generate random conformations and pick best
        2. Monte Carlo: Simple Metropolis Monte Carlo with random moves
        
        Args:
            pdb_id: PDB identifier
            num_samples: Number of samples for baseline methods
            
        Returns:
            ComparisonReport with comparison to baselines
        """
        logger.info(f"Running baseline comparison for {pdb_id}")
        
        # First run UBF validation
        ubf_report = self.validate_protein(pdb_id, num_agents=10, iterations=100)
        
        # Load native structure for baseline comparisons
        native_struct_obj = self.native_loader.load_from_pdb_id(pdb_id, ca_only=True)
        native_coords = native_struct_obj.ca_coords
        rmsd_calculator = RMSDCalculator(align_structures=True)
        
        baselines = []
        
        # Baseline 1: Random Sampling
        logger.info("Running random sampling baseline")
        random_start = time.time()
        best_random_rmsd = float('inf')
        
        for _ in range(num_samples):
            # Generate random conformation (simplified - just perturb coordinates)
            random_coords = [
                (x + random.gauss(0, 5), y + random.gauss(0, 5), z + random.gauss(0, 5))
                for x, y, z in native_coords
            ]
            
            # Calculate RMSD
            try:
                rmsd_result = rmsd_calculator.calculate_rmsd(
                    random_coords,
                    native_coords,
                    calculate_metrics=False
                )
                if rmsd_result.rmsd < best_random_rmsd:
                    best_random_rmsd = rmsd_result.rmsd
            except:
                pass
        
        random_runtime = time.time() - random_start
        
        baselines.append(BaselineResult(
            method_name="random_sampling",
            best_rmsd=best_random_rmsd,
            best_energy=0.0,  # Not calculated for random sampling
            runtime_seconds=random_runtime,
            conformations_sampled=num_samples
        ))
        
        # Baseline 2: Simple Monte Carlo
        logger.info("Running Monte Carlo baseline")
        mc_start = time.time()
        
        # Start from random conformation
        current_coords = [
            (x + random.gauss(0, 10), y + random.gauss(0, 10), z + random.gauss(0, 10))
            for x, y, z in native_coords
        ]
        
        try:
            current_rmsd = rmsd_calculator.calculate_rmsd(
                current_coords,
                native_coords,
                calculate_metrics=False
            ).rmsd
        except:
            current_rmsd = float('inf')
        
        best_mc_rmsd = current_rmsd
        temperature = 10.0
        
        for _ in range(num_samples):
            # Random perturbation
            new_coords = [
                (x + random.gauss(0, 1), y + random.gauss(0, 1), z + random.gauss(0, 1))
                for x, y, z in current_coords
            ]
            
            try:
                new_rmsd = rmsd_calculator.calculate_rmsd(
                    new_coords,
                    native_coords,
                    calculate_metrics=False
                ).rmsd
                
                # Metropolis criterion (minimizing RMSD)
                delta_rmsd = new_rmsd - current_rmsd
                if delta_rmsd < 0 or random.random() < pow(2.718, -delta_rmsd / temperature):
                    current_coords = new_coords
                    current_rmsd = new_rmsd
                    
                    if current_rmsd < best_mc_rmsd:
                        best_mc_rmsd = current_rmsd
            except:
                pass
        
        mc_runtime = time.time() - mc_start
        
        baselines.append(BaselineResult(
            method_name="monte_carlo",
            best_rmsd=best_mc_rmsd,
            best_energy=0.0,  # Not calculated for Monte Carlo
            runtime_seconds=mc_runtime,
            conformations_sampled=num_samples
        ))
        
        comparison = ComparisonReport(
            pdb_id=pdb_id,
            ubf_rmsd=ubf_report.best_rmsd,
            baselines=baselines
        )
        
        logger.info(f"Baseline comparison complete for {pdb_id}")
        return comparison
    
    def save_results(self, results: TestSuiteResults, output_path: str):
        """
        Save test suite results to JSON file.
        
        Args:
            results: TestSuiteResults to save
            output_path: Path to output JSON file
        """
        data = {
            "test_suite_results": {
                "total_runtime_seconds": results.total_runtime_seconds,
                "success_rate": results.success_rate,
                "average_rmsd": results.average_rmsd,
                "average_gdt_ts": results.average_gdt_ts,
                "quality_distribution": results.quality_distribution,
                "validation_reports": [asdict(r) for r in results.validation_reports]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
