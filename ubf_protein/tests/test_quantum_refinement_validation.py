"""
Unit tests for quantum refinement validation suite.

Tests validation script functionality without requiring actual PDB downloads
or full refinement runs (which would be integration tests).
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import validation components
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ubf_protein.validate_quantum_refinement import (
    TEST_PROTEINS,
    RefinementValidationResult,
    ValidationSuiteResults,
)


class TestRefinementValidationResult(unittest.TestCase):
    """Test RefinementValidationResult data class."""
    
    def test_successful_validation(self):
        """Test validation result that meets all success criteria."""
        result = RefinementValidationResult(
            pdb_id="1UBQ",
            protein_name="Ubiquitin",
            sequence_length=76,
            initial_rmsd=10.5,
            initial_energy=50.0,
            initial_gdt_ts=30.0,
            final_rmsd=3.8,
            final_energy=-45.0,
            final_gdt_ts=75.0,
            final_tm_score=0.82,
            rmsd_improvement_percent=63.8,  # (10.5-3.8)/10.5 * 100
            energy_improvement=95.0,
            helix_rmsd=2.5,
            sheet_rmsd=3.2,
            loop_rmsd=4.8,
            core_rmsd=2.9,
            refinement_time_seconds=145.2,
            quantum_cores_identified=8,
            distance_restraints_applied=24,
            tertiary_contacts_enforced=12,
            meets_rmsd_target=True,  # <5Å
            meets_improvement_target=True,  # >50%
            meets_energy_target=True,  # <0
            meets_gdt_target=True,  # >50
            meets_time_target=True,  # <300s
        )
        
        self.assertTrue(result.is_successful())
        summary = result.get_summary()
        self.assertIn("PASS", summary)
        self.assertIn("1UBQ", summary)
        self.assertIn("Ubiquitin", summary)
    
    def test_failed_validation_rmsd(self):
        """Test validation result that fails RMSD criteria."""
        result = RefinementValidationResult(
            pdb_id="1CRN",
            protein_name="Crambin",
            sequence_length=46,
            initial_rmsd=12.0,
            initial_energy=40.0,
            initial_gdt_ts=25.0,
            final_rmsd=6.5,  # >5Å - FAIL
            final_energy=-30.0,
            final_gdt_ts=65.0,
            final_tm_score=0.68,
            rmsd_improvement_percent=45.8,
            energy_improvement=70.0,
            helix_rmsd=5.0,
            sheet_rmsd=6.0,
            loop_rmsd=7.5,
            core_rmsd=5.8,
            refinement_time_seconds=120.5,
            quantum_cores_identified=6,
            distance_restraints_applied=18,
            tertiary_contacts_enforced=8,
            meets_rmsd_target=False,  # FAIL
            meets_improvement_target=False,  # <50%
            meets_energy_target=True,
            meets_gdt_target=True,
            meets_time_target=True,
        )
        
        self.assertFalse(result.is_successful())
        summary = result.get_summary()
        self.assertIn("FAIL", summary)
    
    def test_failed_validation_time(self):
        """Test validation result that exceeds time limit."""
        result = RefinementValidationResult(
            pdb_id="2MR9",
            protein_name="Villin",
            sequence_length=35,
            initial_rmsd=9.0,
            initial_energy=35.0,
            initial_gdt_ts=35.0,
            final_rmsd=3.2,
            final_energy=-25.0,
            final_gdt_ts=70.0,
            final_tm_score=0.75,
            rmsd_improvement_percent=64.4,
            energy_improvement=60.0,
            helix_rmsd=2.0,
            sheet_rmsd=2.8,
            loop_rmsd=4.2,
            core_rmsd=2.5,
            refinement_time_seconds=350.0,  # >300s - FAIL
            quantum_cores_identified=5,
            distance_restraints_applied=15,
            tertiary_contacts_enforced=7,
            meets_rmsd_target=True,
            meets_improvement_target=True,
            meets_energy_target=True,
            meets_gdt_target=True,
            meets_time_target=False,  # FAIL
        )
        
        self.assertFalse(result.is_successful())


class TestValidationSuiteResults(unittest.TestCase):
    """Test ValidationSuiteResults aggregation."""
    
    def test_suite_summary_all_pass(self):
        """Test suite results when all proteins pass."""
        result1 = self._create_passing_result("1UBQ", 3.8)
        result2 = self._create_passing_result("1CRN", 2.5)
        result3 = self._create_passing_result("2MR9", 2.8)
        
        suite_results = ValidationSuiteResults(
            validation_results=[result1, result2, result3],
            total_runtime_seconds=450.0,
            success_rate=100.0,
            average_rmsd_improvement=62.0,
            average_final_rmsd=3.03,  # (3.8+2.5+2.8)/3
        )
        
        summary = suite_results.get_summary()
        self.assertIn("3/3", summary)
        self.assertIn("100.0%", summary)
        self.assertIn("3.03", summary)
    
    def test_suite_summary_partial_pass(self):
        """Test suite results when some proteins fail."""
        result1 = self._create_passing_result("1UBQ", 3.8)
        result2 = self._create_failing_result("1CRN", 6.5)
        result3 = self._create_passing_result("2MR9", 2.8)
        
        suite_results = ValidationSuiteResults(
            validation_results=[result1, result2, result3],
            total_runtime_seconds=480.0,
            success_rate=66.7,
            average_rmsd_improvement=55.0,
            average_final_rmsd=4.37,
        )
        
        summary = suite_results.get_summary()
        self.assertIn("2/3", summary)
        self.assertIn("66.7%", summary)
    
    def _create_passing_result(self, pdb_id: str, final_rmsd: float) -> RefinementValidationResult:
        """Helper to create a passing validation result."""
        return RefinementValidationResult(
            pdb_id=pdb_id,
            protein_name="Test",
            sequence_length=50,
            initial_rmsd=10.0,
            initial_energy=50.0,
            initial_gdt_ts=30.0,
            final_rmsd=final_rmsd,
            final_energy=-40.0,
            final_gdt_ts=75.0,
            final_tm_score=0.80,
            rmsd_improvement_percent=60.0,
            energy_improvement=90.0,
            helix_rmsd=2.0,
            sheet_rmsd=3.0,
            loop_rmsd=4.0,
            core_rmsd=2.5,
            refinement_time_seconds=150.0,
            quantum_cores_identified=8,
            distance_restraints_applied=20,
            tertiary_contacts_enforced=10,
            meets_rmsd_target=True,
            meets_improvement_target=True,
            meets_energy_target=True,
            meets_gdt_target=True,
            meets_time_target=True,
        )
    
    def _create_failing_result(self, pdb_id: str, final_rmsd: float) -> RefinementValidationResult:
        """Helper to create a failing validation result."""
        return RefinementValidationResult(
            pdb_id=pdb_id,
            protein_name="Test",
            sequence_length=50,
            initial_rmsd=10.0,
            initial_energy=50.0,
            initial_gdt_ts=30.0,
            final_rmsd=final_rmsd,
            final_energy=-30.0,
            final_gdt_ts=55.0,
            final_tm_score=0.60,
            rmsd_improvement_percent=35.0,
            energy_improvement=80.0,
            helix_rmsd=5.0,
            sheet_rmsd=6.0,
            loop_rmsd=7.0,
            core_rmsd=5.5,
            refinement_time_seconds=200.0,
            quantum_cores_identified=5,
            distance_restraints_applied=15,
            tertiary_contacts_enforced=7,
            meets_rmsd_target=False,  # FAIL
            meets_improvement_target=False,  # FAIL
            meets_energy_target=True,
            meets_gdt_target=True,
            meets_time_target=True,
        )


class TestTestProteinConfiguration(unittest.TestCase):
    """Test TEST_PROTEINS configuration."""
    
    def test_all_proteins_configured(self):
        """Test that all required proteins are configured."""
        protein_ids = {p['pdb_id'] for p in TEST_PROTEINS}
        
        self.assertIn("1UBQ", protein_ids)
        self.assertIn("1CRN", protein_ids)
        self.assertIn("2MR9", protein_ids)
    
    def test_protein_configs_complete(self):
        """Test that all protein configs have required fields."""
        required_fields = {
            'pdb_id', 'name', 'residues', 'target_rmsd', 'difficulty',
            'exploration_agents', 'exploration_iterations'
        }
        
        for protein in TEST_PROTEINS:
            self.assertTrue(
                required_fields.issubset(protein.keys()),
                f"Protein {protein.get('pdb_id', 'UNKNOWN')} missing required fields"
            )
    
    def test_target_rmsd_values(self):
        """Test that target RMSD values are reasonable."""
        for protein in TEST_PROTEINS:
            target = protein['target_rmsd']
            self.assertGreater(target, 0.0)
            self.assertLess(target, 10.0)
            
            # Smaller proteins should have tighter RMSD targets
            if protein['residues'] < 40:
                self.assertLessEqual(target, 3.0)


if __name__ == '__main__':
    unittest.main()
