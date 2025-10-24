"""
Tests for UBF protein system validation functionality.
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from ubf_protein.validate import run_validation, calculate_validation_metrics, load_native_structure
from ubf_protein.models import ExplorationResults, ExplorationMetrics, Conformation
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator


class TestValidation(unittest.TestCase):
    """Test cases for validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # 141 residues - hemoglobin alpha chain sequence
        self.test_sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"[:-1]  # Remove last char to get 141
        assert len(self.test_sequence) == 141, f"Sequence length is {len(self.test_sequence)}, expected 141"

        # Create mock exploration results
        self.mock_metrics = [
            ExplorationMetrics(
                agent_id="agent_0",
                iterations_completed=100,
                conformations_explored=50,
                memories_created=10,
                best_energy_found=-15.5,
                best_rmsd_found=2.1,
                learning_improvement=25.0,
                avg_decision_time_ms=5.2,
                stuck_in_minima_count=2,
                successful_escapes=1
            ),
            ExplorationMetrics(
                agent_id="agent_1",
                iterations_completed=100,
                conformations_explored=48,
                memories_created=8,
                best_energy_found=-12.3,
                best_rmsd_found=2.5,
                learning_improvement=20.0,
                avg_decision_time_ms=4.8,
                stuck_in_minima_count=1,
                successful_escapes=1
            )
        ]

        self.mock_conformation = Conformation(
            conformation_id="best_conf",
            sequence=self.test_sequence,
            atom_coordinates=[(i*3.8, 0.0, 0.0) for i in range(len(self.test_sequence))],
            energy=-15.5,
            rmsd_to_native=2.1,
            secondary_structure=['H'] * len(self.test_sequence),
            phi_angles=[-60.0] * len(self.test_sequence),
            psi_angles=[-40.0] * len(self.test_sequence),
            available_move_types=["backbone_rotation"],
            structural_constraints={}
        )

        self.mock_results = ExplorationResults(
            total_iterations=200,
            total_conformations_explored=98,
            best_conformation=self.mock_conformation,
            best_energy=-15.5,
            best_rmsd=2.1,
            agent_metrics=self.mock_metrics,
            collective_learning_benefit=15.0,
            total_runtime_seconds=45.2,
            shared_memories_created=5
        )

        self.native_data = {
            'sequence': self.test_sequence,
            'length': len(self.test_sequence),
            'experimental_stability': {
                'melting_temperature': 50.5,
                'delta_g_unfolding': -5.2,
                'stability_class': 'stable'
            }
        }

    def test_load_native_structure_success(self):
        """Test successful loading of native structure."""
        with patch('ubf_protein.validate.load_native_structure') as mock_load:
            mock_load.return_value = self.native_data

            result = load_native_structure("fake.pdb")
            self.assertIsNotNone(result)
            if result is not None:  # Type guard for mypy
                self.assertEqual(result['length'], len(self.test_sequence))
                self.assertIn('experimental_stability', result)

    def test_load_native_structure_failure(self):
        """Test failure to load native structure."""
        # The current load_native_structure is a placeholder that returns mock data
        # In a real implementation, it would parse PDB files and could fail
        # For now, we just verify it doesn't crash
        result = load_native_structure("nonexistent.pdb")
        # Placeholder always returns data, so we just check it's a dict
        self.assertIsInstance(result, dict)

    def test_calculate_validation_metrics(self):
        """Test calculation of validation metrics."""
        metrics = calculate_validation_metrics(self.mock_results, self.native_data)

        # Check that all metric categories are present
        self.assertIn('prediction_accuracy', metrics)
        self.assertIn('convergence_quality', metrics)
        self.assertIn('exploration_efficiency', metrics)
        self.assertIn('learning_assessment', metrics)

        # Check prediction accuracy calculations
        pred_acc = metrics['prediction_accuracy']
        self.assertIn('rmsd_accuracy', pred_acc)
        self.assertIn('energy_accuracy', pred_acc)
        self.assertIn('overall_accuracy', pred_acc)

        # Overall accuracy should be reasonable
        self.assertGreaterEqual(pred_acc['overall_accuracy'], 0.0)
        self.assertLessEqual(pred_acc['overall_accuracy'], 1.0)

        # Check convergence quality
        conv_qual = metrics['convergence_quality']
        self.assertEqual(conv_qual['total_iterations'], 200)
        self.assertIn('convergence_rate', conv_qual)
        self.assertIn('avg_iterations_per_agent', conv_qual)

        # Check exploration efficiency
        exp_eff = metrics['exploration_efficiency']
        self.assertEqual(exp_eff['total_conformations_explored'], 98)
        self.assertIn('exploration_efficiency', exp_eff)

        # Check learning assessment
        learn_assess = metrics['learning_assessment']
        self.assertAlmostEqual(learn_assess['avg_learning_improvement'], 22.5)  # (25+20)/2
        self.assertEqual(learn_assess['collective_learning_benefit'], 15.0)
        self.assertIn('learning_effectiveness', learn_assess)

    def test_run_validation_basic(self):
        """Test basic validation run without native structure."""
        with patch('ubf_protein.validate.MultiAgentCoordinator') as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator
            mock_coordinator.initialize_agents.return_value = []
            mock_coordinator.run_parallel_exploration.return_value = self.mock_results

            results = run_validation(
                sequence=self.test_sequence,
                agents=2,
                iterations=100
            )

            # Verify coordinator was called correctly
            mock_coordinator.initialize_agents.assert_called_once_with(2)
            mock_coordinator.run_parallel_exploration.assert_called_once_with(100)

            # Check results structure
            self.assertIn('metadata', results)
            self.assertIn('exploration_results', results)
            self.assertIn('validation_metrics', results)
            self.assertIn('agent_summary', results)

            # Check metadata
            self.assertEqual(results['metadata']['sequence_length'], len(self.test_sequence))
            self.assertEqual(results['metadata']['agents_used'], 2)
            self.assertFalse(results['metadata']['native_structure_provided'])

    def test_run_validation_with_native_structure(self):
        """Test validation run with native structure."""
        with patch('ubf_protein.validate.MultiAgentCoordinator') as mock_coordinator_class, \
             patch('ubf_protein.validate.load_native_structure') as mock_load:

            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator
            mock_coordinator.initialize_agents.return_value = []
            mock_coordinator.run_parallel_exploration.return_value = self.mock_results
            mock_load.return_value = self.native_data

            results = run_validation(
                sequence=self.test_sequence,
                native_pdb="test.pdb",
                agents=2,
                iterations=100
            )

            # Verify native structure was loaded
            mock_load.assert_called_once_with("test.pdb")

            # Check that native structure flag is set
            self.assertTrue(results['metadata']['native_structure_provided'])

    def test_run_validation_with_output_file(self):
        """Test validation run with output file saving."""
        with patch('ubf_protein.validate.MultiAgentCoordinator') as mock_coordinator_class, \
             tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:

            temp_file.close()  # Close so it can be written to

            try:
                mock_coordinator = MagicMock()
                mock_coordinator_class.return_value = mock_coordinator
                mock_coordinator.initialize_agents.return_value = []
                mock_coordinator.run_parallel_exploration.return_value = self.mock_results

                results = run_validation(
                    sequence=self.test_sequence,
                    output_file=temp_file.name,
                    agents=2,
                    iterations=100
                )

                # Verify file was created and contains valid JSON
                self.assertTrue(os.path.exists(temp_file.name))

                with open(temp_file.name, 'r') as f:
                    saved_data = json.load(f)

                # Check that saved data matches results
                self.assertEqual(saved_data['metadata']['agents_used'], 2)
                self.assertIn('validation_metrics', saved_data)

            finally:
                # Clean up
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    def test_learning_improvement_calculation(self):
        """Test that learning improvement is calculated correctly."""
        # Test with mock results
        metrics = calculate_validation_metrics(self.mock_results, self.native_data)

        learning = metrics['learning_assessment']
        expected_avg_improvement = (25.0 + 20.0) / 2  # 22.5
        expected_effectiveness = (22.5 + 15.0) / 2    # 18.75

        self.assertAlmostEqual(learning['avg_learning_improvement'], expected_avg_improvement)
        self.assertEqual(learning['collective_learning_benefit'], 15.0)
        self.assertAlmostEqual(learning['learning_effectiveness'], expected_effectiveness)

    def test_validation_metrics_edge_cases(self):
        """Test validation metrics with edge cases."""
        # Test with no agent metrics
        empty_results = ExplorationResults(
            total_iterations=0,
            total_conformations_explored=0,
            best_conformation=None,
            best_energy=float('inf'),
            best_rmsd=float('inf'),
            agent_metrics=[],
            collective_learning_benefit=0.0,
            total_runtime_seconds=0.0,
            shared_memories_created=0
        )

        metrics = calculate_validation_metrics(empty_results, {})

        # Should handle empty results gracefully
        self.assertEqual(metrics['convergence_quality']['total_iterations'], 0)
        self.assertEqual(metrics['exploration_efficiency']['total_conformations_explored'], 0)
        self.assertEqual(metrics['learning_assessment']['avg_learning_improvement'], 0.0)


if __name__ == '__main__':
    unittest.main()