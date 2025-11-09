"""
Unit tests for refinement visualization module.

Tests visualization functionality without displaying plots.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

from ubf_protein.refinement_visualization import RefinementVisualizer, visualize_refinement_result
from ubf_protein.models import RefinementResult, Conformation


class TestRefinementVisualizer(unittest.TestCase):
    """Test cases for RefinementVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = RefinementVisualizer(figsize=(8, 6), dpi=100)
        
        # Create mock refinement result
        self.sequence = "ACDEFGHIKLMNPQRSTVWY"
        self.mock_conformation = Conformation(
            conformation_id="test",
            sequence=self.sequence,
            atom_coordinates=[(i*3.8, 0.0, 0.0) for i in range(len(self.sequence))],
            energy=-50.0,
            rmsd_to_native=3.5,
            secondary_structure=['H'] * len(self.sequence),
            phi_angles=[-60.0] * len(self.sequence),
            psi_angles=[-40.0] * len(self.sequence),
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )
        
        self.mock_result = RefinementResult(
            initial_structure=self.mock_conformation,
            refined_structure=self.mock_conformation,
            native_structure=None,
            initial_rmsd=10.5,
            final_rmsd=3.5,
            rmsd_improvement=7.0,
            helix_rmsd=2.5,
            sheet_rmsd=3.2,
            loop_rmsd=4.8,
            core_rmsd=2.9,
            gdt_ts=75.0,
            tm_score=0.82,
            energy=-50.0,
            iterations_used=1000,
            refinement_time_seconds=120.0,
            quantum_cores_identified=8,
            restraints_applied=24,
            contacts_enforced=12,
            rmsd_trajectory=[10.5, 9.0, 7.5, 6.0, 5.0, 4.0, 3.5],
            energy_trajectory=[20.0, 0.0, -20.0, -35.0, -45.0, -48.0, -50.0]
        )
    
    def test_visualizer_initialization(self):
        """Test visualizer initializes with correct parameters."""
        self.assertEqual(self.visualizer.figsize, (8, 6))
        self.assertEqual(self.visualizer.dpi, 100)
    
    def test_plot_rmsd_trajectory_creates_file(self):
        """Test RMSD trajectory plot creates output file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name
        
        try:
            self.visualizer.plot_rmsd_trajectory(
                self.mock_result,
                output_file=temp_file,
                show=False
            )
            
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 0)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_plot_energy_landscape_creates_file(self):
        """Test energy landscape plot creates output file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name
        
        try:
            self.visualizer.plot_energy_landscape(
                self.mock_result,
                output_file=temp_file,
                show=False
            )
            
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 0)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_plot_component_breakdown_creates_file(self):
        """Test component breakdown plot creates output file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name
        
        try:
            self.visualizer.plot_component_breakdown(
                self.mock_result,
                output_file=temp_file,
                show=False
            )
            
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 0)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_plot_quantum_cores_creates_file(self):
        """Test quantum cores plot creates output file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name
        
        try:
            self.visualizer.plot_quantum_cores(
                self.mock_result,
                sequence_length=len(self.sequence),
                output_file=temp_file,
                show=False
            )
            
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 0)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_plot_contact_map_creates_file(self):
        """Test contact map plot creates output file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name
        
        try:
            self.visualizer.plot_contact_map(
                self.mock_result,
                sequence_length=len(self.sequence),
                output_file=temp_file,
                show=False
            )
            
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 0)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_create_validation_report_creates_all_files(self):
        """Test validation report creates all visualization files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = self.visualizer.create_validation_report(
                self.mock_result,
                sequence_length=len(self.sequence),
                output_dir=temp_dir,
                pdb_id="1TEST"
            )
            
            # Check all expected files are created
            self.assertIn('rmsd_trajectory', files)
            self.assertIn('energy_landscape', files)
            self.assertIn('component_breakdown', files)
            self.assertIn('quantum_cores', files)
            self.assertIn('contact_map', files)
            
            # Check files exist
            for file_path in files.values():
                self.assertTrue(os.path.exists(file_path))
                self.assertGreater(os.path.getsize(file_path), 0)
    
    def test_plot_multi_protein_comparison(self):
        """Test multi-protein comparison plot."""
        validation_results = [
            {
                'pdb_id': '1UBQ',
                'initial_rmsd': 10.5,
                'final_rmsd': 3.8,
                'rmsd_improvement_percent': 63.8,
                'meets_rmsd_target': True,
                'meets_improvement_target': True,
                'meets_energy_target': True,
                'meets_gdt_target': True,
                'meets_time_target': True,
            },
            {
                'pdb_id': '1CRN',
                'initial_rmsd': 9.0,
                'final_rmsd': 2.5,
                'rmsd_improvement_percent': 72.2,
                'meets_rmsd_target': True,
                'meets_improvement_target': True,
                'meets_energy_target': True,
                'meets_gdt_target': True,
                'meets_time_target': True,
            },
            {
                'pdb_id': '2MR9',
                'initial_rmsd': 8.5,
                'final_rmsd': 2.8,
                'rmsd_improvement_percent': 67.1,
                'meets_rmsd_target': True,
                'meets_improvement_target': True,
                'meets_energy_target': True,
                'meets_gdt_target': True,
                'meets_time_target': True,
            },
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name
        
        try:
            self.visualizer.plot_multi_protein_comparison(
                validation_results,
                output_file=temp_file,
                show=False
            )
            
            self.assertTrue(os.path.exists(temp_file))
            self.assertGreater(os.path.getsize(temp_file), 0)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_convenience_function(self):
        """Test convenience function for quick visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = visualize_refinement_result(
                self.mock_result,
                sequence_length=len(self.sequence),
                output_dir=temp_dir,
                pdb_id="1TEST"
            )
            
            # Check all files created
            self.assertEqual(len(files), 5)
            for file_path in files.values():
                self.assertTrue(os.path.exists(file_path))
    
    def test_empty_trajectory_handling(self):
        """Test handling of empty trajectories."""
        result_with_empty = RefinementResult(
            initial_structure=self.mock_conformation,
            refined_structure=self.mock_conformation,
            native_structure=None,
            initial_rmsd=10.5,
            final_rmsd=3.5,
            rmsd_improvement=7.0,
            helix_rmsd=2.5,
            sheet_rmsd=3.2,
            loop_rmsd=4.8,
            core_rmsd=2.9,
            gdt_ts=75.0,
            tm_score=0.82,
            energy=-50.0,
            iterations_used=0,
            refinement_time_seconds=0.0,
            quantum_cores_identified=0,
            restraints_applied=0,
            contacts_enforced=0,
            rmsd_trajectory=[],
            energy_trajectory=[]
        )
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name
        
        try:
            # Should not crash with empty trajectories
            self.visualizer.plot_rmsd_trajectory(
                result_with_empty,
                output_file=temp_file,
                show=False
            )
            self.assertTrue(os.path.exists(temp_file))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()
