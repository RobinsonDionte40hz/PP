"""
Tests for Aggregation Screening Module.

Tests the aggregation risk scoring and batch screening capabilities.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ubf_protein.aggregation_screening import (
    AggregationScreener,
    AggregationMetrics,
    AggregationRisk,
    ScreeningConfig,
    quick_screen,
)


class TestAggregationRisk(unittest.TestCase):
    """Test the AggregationRisk enum."""
    
    def test_risk_levels(self):
        """Test all risk levels exist."""
        self.assertEqual(AggregationRisk.LOW.value, "low")
        self.assertEqual(AggregationRisk.MODERATE.value, "moderate")
        self.assertEqual(AggregationRisk.HIGH.value, "high")
        self.assertEqual(AggregationRisk.CRITICAL.value, "critical")


class TestScreeningConfig(unittest.TestCase):
    """Test screening configuration presets."""
    
    def test_fast_config(self):
        """Test fast screening config."""
        config = ScreeningConfig.fast()
        self.assertEqual(config.iterations, 50)
        self.assertEqual(config.agents, 2)
    
    def test_balanced_config(self):
        """Test balanced screening config."""
        config = ScreeningConfig.balanced()
        self.assertEqual(config.iterations, 100)
        self.assertEqual(config.agents, 3)
    
    def test_thorough_config(self):
        """Test thorough screening config."""
        config = ScreeningConfig.thorough()
        self.assertEqual(config.iterations, 200)
        self.assertEqual(config.agents, 5)
        self.assertTrue(config.enable_qcpp)


class TestSequenceComposition(unittest.TestCase):
    """Test sequence composition analysis."""
    
    def setUp(self):
        self.screener = AggregationScreener(ScreeningConfig.fast())
    
    def test_hydrophobic_ratio(self):
        """Test hydrophobic ratio calculation."""
        # All hydrophobic
        result = self.screener._analyze_sequence_composition("AVIL")
        self.assertEqual(result['hydrophobic_ratio'], 1.0)
        
        # No hydrophobic
        result = self.screener._analyze_sequence_composition("DEKR")
        self.assertEqual(result['hydrophobic_ratio'], 0.0)
        
        # Mixed
        result = self.screener._analyze_sequence_composition("AVILDEEE")
        self.assertAlmostEqual(result['hydrophobic_ratio'], 0.5, places=2)
    
    def test_charged_ratio(self):
        """Test charged residue ratio."""
        # All charged
        result = self.screener._analyze_sequence_composition("RKDE")
        self.assertEqual(result['charged_ratio'], 1.0)
        
        # No charged
        result = self.screener._analyze_sequence_composition("AVIL")
        self.assertEqual(result['charged_ratio'], 0.0)
    
    def test_pattern_detection(self):
        """Test aggregation-prone pattern detection."""
        # Poly-valine
        result = self.screener._analyze_sequence_composition("AAVVVVAA")
        self.assertIn('poly-valine', result['patterns_found'])
        
        # No patterns
        result = self.screener._analyze_sequence_composition("ACDEFGH")
        self.assertEqual(result['patterns_found'], [])
    
    def test_hydrophobic_stretch(self):
        """Test max hydrophobic stretch detection."""
        result = self.screener._analyze_sequence_composition("AAAAVVVVVAAAAA")
        # AVVVVV = 6 consecutive hydrophobic
        self.assertGreaterEqual(result['max_hydrophobic_stretch'], 5)


class TestScoring(unittest.TestCase):
    """Test individual scoring functions."""
    
    def setUp(self):
        self.screener = AggregationScreener(ScreeningConfig.balanced())
    
    def test_energy_scoring(self):
        """Test energy to score conversion."""
        # Very stable (good)
        score = self.screener._score_energy(-100.0)
        self.assertEqual(score, 1.0)
        
        # Unstable (bad)
        score = self.screener._score_energy(50.0)
        self.assertEqual(score, 0.0)
        
        # Middling
        score = self.screener._score_energy(-25.0)
        self.assertTrue(0.0 < score < 1.0)
    
    def test_structure_scoring(self):
        """Test structure % to score conversion."""
        # Well structured
        score = self.screener._score_structure(80.0)
        self.assertEqual(score, 1.0)
        
        # Poorly structured
        score = self.screener._score_structure(20.0)
        self.assertEqual(score, 0.0)
    
    def test_compactness_scoring(self):
        """Test compactness scoring."""
        # Compact (good)
        score = self.screener._score_compactness(8.0, 36)  # 36 res, Rg=8 is compact
        self.assertGreater(score, 0.5)
        
        # Extended (bad)
        score = self.screener._score_compactness(40.0, 36)  # 36 res, Rg=40 is extended
        self.assertLess(score, 0.3)


class TestRiskClassification(unittest.TestCase):
    """Test risk level classification."""
    
    def setUp(self):
        self.screener = AggregationScreener()
    
    def test_low_risk(self):
        """Test low risk classification."""
        risk = self.screener._classify_risk(0.8, [])
        self.assertEqual(risk, AggregationRisk.LOW)
    
    def test_moderate_risk(self):
        """Test moderate risk classification."""
        risk = self.screener._classify_risk(0.55, [])
        self.assertEqual(risk, AggregationRisk.MODERATE)
    
    def test_high_risk(self):
        """Test high risk classification."""
        risk = self.screener._classify_risk(0.35, [])
        self.assertEqual(risk, AggregationRisk.HIGH)
    
    def test_critical_risk(self):
        """Test critical risk classification."""
        risk = self.screener._classify_risk(0.2, [])
        self.assertEqual(risk, AggregationRisk.CRITICAL)
    
    def test_critical_factor_override(self):
        """Test that critical factors override score."""
        # High score but critical factor
        risk = self.screener._classify_risk(0.9, ['poly-valine'])
        self.assertEqual(risk, AggregationRisk.CRITICAL)


class TestAggregationMetrics(unittest.TestCase):
    """Test AggregationMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = AggregationMetrics(
            sequence="ACDEFGH",
            sequence_length=7,
            energy_score=0.8,
            structure_score=0.7,
            hydrophobic_score=0.6,
            convergence_score=0.9,
            compactness_score=0.7,
            final_energy=-50.0,
            secondary_structure_pct=65.0,
            hydrophobic_clustering=12.0,
            convergence_iterations=50,
            radius_of_gyration=8.0,
            aggregation_score=0.75,
            risk_level=AggregationRisk.LOW,
            risk_factors=[],
        )
        
        d = metrics.to_dict()
        self.assertEqual(d['sequence'], "ACDEFGH")
        self.assertEqual(d['risk_level'], "low")  # Enum converted to string
    
    def test_passes_screening(self):
        """Test passes_screening property."""
        # Low risk passes
        metrics = AggregationMetrics(
            sequence="A", sequence_length=1,
            energy_score=0.8, structure_score=0.8,
            hydrophobic_score=0.8, convergence_score=0.8,
            compactness_score=0.8, final_energy=-50.0,
            secondary_structure_pct=80.0, hydrophobic_clustering=10.0,
            convergence_iterations=50, radius_of_gyration=8.0,
            aggregation_score=0.8, risk_level=AggregationRisk.LOW,
            risk_factors=[],
        )
        self.assertTrue(metrics.passes_screening)
        
        # Critical risk fails
        metrics.risk_level = AggregationRisk.CRITICAL
        self.assertFalse(metrics.passes_screening)


class TestSingleSequenceScreening(unittest.TestCase):
    """Test single sequence screening (integration test)."""
    
    def setUp(self):
        # Use fast config for testing
        self.screener = AggregationScreener(ScreeningConfig.fast())
    
    def test_screen_simple_sequence(self):
        """Test screening a simple sequence."""
        result = self.screener.screen_sequence("ACDEFGHIKLMNPQRSTVWY")
        
        self.assertIsInstance(result, AggregationMetrics)
        self.assertEqual(result.sequence_length, 20)
        self.assertIn(result.risk_level, list(AggregationRisk))
        self.assertTrue(0.0 <= result.aggregation_score <= 1.0)
    
    def test_screen_aggregation_prone(self):
        """Test screening an aggregation-prone sequence."""
        # Poly-valine should be flagged
        result = self.screener.screen_sequence("VVVVVVVVVVVVVVVVVVVV")
        
        self.assertIn('poly-valine', result.risk_factors)
        self.assertEqual(result.risk_level, AggregationRisk.CRITICAL)
    
    def test_screen_mixed_sequence(self):
        """Test screening a mixed sequence."""
        # A reasonably balanced sequence
        result = self.screener.screen_sequence("MKTAYIAKQRQISFVKSH")
        
        self.assertIsInstance(result, AggregationMetrics)
        self.assertGreater(result.screening_time_ms, 0)


class TestBatchScreening(unittest.TestCase):
    """Test batch screening functionality."""
    
    def setUp(self):
        self.screener = AggregationScreener(ScreeningConfig.fast())
    
    def test_batch_screen_multiple(self):
        """Test screening multiple sequences."""
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",  # Mixed
            "VVVVVVVVVVVVVVVVVVVV",  # Aggregation-prone
            "RRRRKKKKEEEEDDDD",       # Charged
        ]
        
        results = self.screener.screen_batch(sequences)
        
        self.assertEqual(len(results), 3)
        # Results should be sorted by score (best first)
        self.assertGreaterEqual(results[0].aggregation_score, results[-1].aggregation_score)
    
    def test_batch_with_callback(self):
        """Test batch screening with progress callback."""
        sequences = ["ACDEFGH", "HIKLMNP"]
        progress_updates = []
        
        def callback(current, total, result):
            progress_updates.append((current, total))
        
        results = self.screener.screen_batch(sequences, progress_callback=callback)
        
        self.assertEqual(len(progress_updates), 2)
        self.assertEqual(progress_updates[0], (1, 2))
        self.assertEqual(progress_updates[1], (2, 2))


class TestExport(unittest.TestCase):
    """Test export functionality."""
    
    def setUp(self):
        self.screener = AggregationScreener(ScreeningConfig.fast())
        self.results = [
            AggregationMetrics(
                sequence="ACDEFGH", sequence_length=7,
                energy_score=0.8, structure_score=0.7,
                hydrophobic_score=0.6, convergence_score=0.9,
                compactness_score=0.7, final_energy=-50.0,
                secondary_structure_pct=65.0, hydrophobic_clustering=12.0,
                convergence_iterations=50, radius_of_gyration=8.0,
                aggregation_score=0.75, risk_level=AggregationRisk.LOW,
                risk_factors=[], screening_time_ms=100.0, iterations_used=50,
            ),
        ]
    
    def test_export_csv(self):
        """Test CSV export."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            result_path = self.screener.export_csv(self.results, filepath)
            self.assertTrue(os.path.exists(result_path))
            
            # Verify content
            with open(result_path, 'r') as f:
                content = f.read()
                self.assertIn('ACDEFGH', content)
                self.assertIn('low', content)
        finally:
            os.unlink(filepath)
    
    def test_export_json(self):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            result_path = self.screener.export_json(self.results, filepath)
            self.assertTrue(os.path.exists(result_path))
            
            # Verify content
            with open(result_path, 'r') as f:
                data = json.load(f)
                self.assertIn('results', data)
                self.assertIn('summary', data)
                self.assertEqual(data['summary']['total_sequences'], 1)
        finally:
            os.unlink(filepath)


class TestQuickScreen(unittest.TestCase):
    """Test the convenience quick_screen function."""
    
    def test_quick_screen(self):
        """Test quick screening function."""
        result = quick_screen("ACDEFGHIKLMNPQRSTVWY")
        
        self.assertIsInstance(result, AggregationMetrics)
        self.assertEqual(result.iterations_used, 50)  # Fast config


if __name__ == '__main__':
    unittest.main()
