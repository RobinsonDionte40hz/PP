"""
FailureAnalyzer - Detailed analysis of failed predictions for system improvement.

This module provides comprehensive failure analysis capabilities for protein structure
prediction validation campaigns, including:
- Failure classification by type and severity
- Common characteristic extraction among failed predictions
- Failure visualization generation (metadata)
- Energy trajectory analysis for local minima detection
- Parameter adjustment recommendations based on failure patterns

Author: Large-Scale Validation Framework
Date: October 26, 2025
"""

import statistics
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import json


@dataclass(frozen=True)
class FailureClassification:
    """
    Classification of a failed prediction.
    
    Attributes:
        pdb_id: Protein identifier
        failure_type: Type of failure (high_rmsd, poor_energy, low_gdt_ts, multiple)
        severity: Severity level (minor, moderate, severe)
        rmsd: Root mean square deviation value
        energy: Final energy value
        gdt_ts: GDT-TS score
    """
    pdb_id: str
    failure_type: str
    severity: str
    rmsd: float
    energy: float
    gdt_ts: float


@dataclass(frozen=True)
class FailurePatterns:
    """
    Common patterns among failed predictions.
    
    Attributes:
        common_size_category: Most common size category among failures (if any)
        common_structural_class: Most common structural class among failures (if any)
        average_secondary_structure_content: Average helix/sheet fractions
        common_issues: List of common failure types observed
    """
    common_size_category: Optional[str]
    common_structural_class: Optional[str]
    average_secondary_structure_content: Dict[str, float]
    common_issues: List[str]


@dataclass(frozen=True)
class TrajectoryAnalysis:
    """
    Analysis of energy trajectory during prediction.
    
    Attributes:
        stuck_in_local_minima: Whether system appears stuck in local minimum
        minima_count: Number of local minima detected
        escape_attempts: Number of attempted escapes from minima
        energy_variance: Variance in energy values
        convergence_achieved: Whether energy converged to stable value
    """
    stuck_in_local_minima: bool
    minima_count: int
    escape_attempts: int
    energy_variance: float
    convergence_achieved: bool


class FailureAnalyzer:
    """
    Detailed analysis of failed predictions for system improvement.
    
    Provides comprehensive failure analysis including classification, pattern detection,
    visualization metadata generation, trajectory analysis, and parameter adjustment
    recommendations.
    
    Failure Criteria:
        - RMSD > 8.0 Å: Severe failure (incorrect structure)
        - Energy > 0 kcal/mol: Unstable structure
        - GDT-TS < 30: Incorrect fold
        - RMSD > 5.0 Å: Moderate failure
        - GDT-TS < 50: Minor to moderate failure
    
    Example:
        >>> analyzer = FailureAnalyzer()
        >>> classification = analyzer.classify_failure(validation_report)
        >>> print(f"{classification.pdb_id}: {classification.failure_type} ({classification.severity})")
        >>> 
        >>> patterns = analyzer.extract_common_characteristics(failed_reports)
        >>> recommendations = analyzer.recommend_parameter_adjustments(patterns)
    """
    
    # Failure thresholds
    SEVERE_RMSD_THRESHOLD = 8.0      # Å
    MODERATE_RMSD_THRESHOLD = 5.0    # Å
    UNSTABLE_ENERGY_THRESHOLD = 0.0  # kcal/mol
    SEVERE_GDT_THRESHOLD = 30.0      # GDT-TS score
    MODERATE_GDT_THRESHOLD = 50.0    # GDT-TS score
    
    def __init__(self):
        """Initialize the FailureAnalyzer."""
        self._failure_cache: List[FailureClassification] = []
    
    def classify_failure(self, report: Dict) -> FailureClassification:
        """
        Classify a prediction failure by type and severity.
        
        Analyzes validation metrics to determine failure type (high RMSD, poor energy,
        low GDT-TS, or multiple issues) and severity level.
        
        Args:
            report: Validation report dictionary with rmsd, energy, gdt_ts fields
        
        Returns:
            FailureClassification with type and severity information
        
        Raises:
            ValueError: If report is missing required fields
        
        Example:
            >>> report = {'pdb_id': '1ABC', 'rmsd': 9.5, 'energy': 10.0, 'gdt_ts': 25.0}
            >>> classification = analyzer.classify_failure(report)
            >>> print(f"Type: {classification.failure_type}, Severity: {classification.severity}")
        """
        # Validate required fields
        required_fields = ['pdb_id', 'rmsd', 'gdt_ts']
        missing_fields = [f for f in required_fields if f not in report]
        if missing_fields:
            raise ValueError(f"Report missing required fields: {missing_fields}")
        
        pdb_id = report['pdb_id']
        rmsd = report['rmsd']
        energy = report.get('energy', report.get('final_energy', 0.0))
        gdt_ts = report['gdt_ts']
        
        # Identify failure types
        failure_types = []
        if rmsd > self.SEVERE_RMSD_THRESHOLD:
            failure_types.append('high_rmsd')
        if energy > self.UNSTABLE_ENERGY_THRESHOLD:
            failure_types.append('poor_energy')
        if gdt_ts < self.SEVERE_GDT_THRESHOLD:
            failure_types.append('low_gdt_ts')
        
        # Determine primary failure type
        if len(failure_types) >= 2:
            failure_type = 'multiple'
        elif failure_types:
            failure_type = failure_types[0]
        else:
            # Moderate failures
            if rmsd > self.MODERATE_RMSD_THRESHOLD:
                failure_type = 'high_rmsd'
            elif gdt_ts < self.MODERATE_GDT_THRESHOLD:
                failure_type = 'low_gdt_ts'
            else:
                failure_type = 'unknown'
        
        # Determine severity
        severity = self._determine_severity(rmsd, energy, gdt_ts)
        
        classification = FailureClassification(
            pdb_id=pdb_id,
            failure_type=failure_type,
            severity=severity,
            rmsd=rmsd,
            energy=energy,
            gdt_ts=gdt_ts
        )
        
        self._failure_cache.append(classification)
        return classification
    
    def extract_common_characteristics(
        self,
        failures: List[Dict]
    ) -> FailurePatterns:
        """
        Extract common characteristics among failed predictions.
        
        Analyzes a set of failed predictions to identify patterns in protein size,
        structural class, secondary structure content, and failure types.
        
        Args:
            failures: List of validation reports for failed predictions
        
        Returns:
            FailurePatterns with common characteristics
        
        Raises:
            ValueError: If failures list is empty
        
        Example:
            >>> patterns = analyzer.extract_common_characteristics(failed_reports)
            >>> print(f"Common size: {patterns.common_size_category}")
            >>> print(f"Common issues: {patterns.common_issues}")
        """
        if not failures:
            raise ValueError("Cannot extract patterns from empty failures list")
        
        # Extract size categories
        size_categories = [f.get('size_category', 'unknown') for f in failures]
        common_size = self._find_most_common(size_categories)
        
        # Extract structural classes
        structural_classes = [f.get('structural_class', 'unknown') for f in failures]
        common_structural = self._find_most_common(structural_classes)
        
        # Calculate average secondary structure content
        helix_fractions = [f.get('helix_fraction', 0.0) for f in failures]
        sheet_fractions = [f.get('sheet_fraction', 0.0) for f in failures]
        
        avg_ss_content = {
            'helix': statistics.mean(helix_fractions) if helix_fractions else 0.0,
            'sheet': statistics.mean(sheet_fractions) if sheet_fractions else 0.0,
            'coil': 1.0 - (statistics.mean(helix_fractions) + statistics.mean(sheet_fractions))
        }
        
        # Identify common issues
        common_issues = []
        
        # Check for high RMSD prevalence
        high_rmsd_count = sum(1 for f in failures if f.get('rmsd', 0) > self.MODERATE_RMSD_THRESHOLD)
        if high_rmsd_count / len(failures) > 0.5:
            common_issues.append(f"High RMSD in {high_rmsd_count}/{len(failures)} cases")
        
        # Check for poor energy prevalence
        poor_energy_count = sum(1 for f in failures 
                               if f.get('energy', f.get('final_energy', -100)) > self.UNSTABLE_ENERGY_THRESHOLD)
        if poor_energy_count / len(failures) > 0.3:
            common_issues.append(f"Poor energy in {poor_energy_count}/{len(failures)} cases")
        
        # Check for low GDT-TS prevalence
        low_gdt_count = sum(1 for f in failures if f.get('gdt_ts', 100) < self.MODERATE_GDT_THRESHOLD)
        if low_gdt_count / len(failures) > 0.5:
            common_issues.append(f"Low GDT-TS in {low_gdt_count}/{len(failures)} cases")
        
        # Check for large protein bias
        large_protein_count = sum(1 for f in failures if f.get('protein_length', 0) > 200)
        if large_protein_count / len(failures) > 0.6:
            common_issues.append(f"Bias toward large proteins ({large_protein_count}/{len(failures)})")
        
        return FailurePatterns(
            common_size_category=common_size if common_size != 'unknown' else None,
            common_structural_class=common_structural if common_structural != 'unknown' else None,
            average_secondary_structure_content=avg_ss_content,
            common_issues=common_issues
        )
    
    def generate_failure_visualizations(
        self,
        pdb_id: str,
        report: Dict,
        output_dir: str = "./failure_visualizations"
    ) -> List[str]:
        """
        Generate visualization metadata for failed prediction.
        
        Creates JSON files containing visualization specifications for comparing
        predicted and native structures, highlighting problematic regions.
        
        Args:
            pdb_id: Protein identifier
            report: Validation report with structure information
            output_dir: Directory to save visualization metadata
        
        Returns:
            List of file paths to generated visualization metadata files
        
        Note:
            This method creates metadata files describing visualizations.
            Actual 3D rendering requires PyMOL, Chimera, or similar tools.
        
        Example:
            >>> vis_files = analyzer.generate_failure_visualizations('1ABC', report)
            >>> print(f"Generated {len(vis_files)} visualization specs")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Generate structure comparison metadata
        comparison_spec = {
            'pdb_id': pdb_id,
            'visualization_type': 'structure_comparison',
            'native_structure': f"{pdb_id}_native.pdb",
            'predicted_structure': f"{pdb_id}_predicted.pdb",
            'rmsd': report.get('rmsd', 0.0),
            'alignment_method': 'superposition',
            'color_scheme': {
                'native': 'blue',
                'predicted': 'red',
                'high_deviation_regions': 'yellow'
            },
            'rmsd_threshold_highlighting': 3.0  # Highlight regions with >3Å deviation
        }
        
        comparison_file = output_path / f"{pdb_id}_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_spec, f, indent=2)
        generated_files.append(str(comparison_file))
        
        # Generate per-residue deviation plot metadata
        deviation_spec = {
            'pdb_id': pdb_id,
            'visualization_type': 'per_residue_deviation',
            'plot_type': 'line_plot',
            'x_axis': 'residue_number',
            'y_axis': 'rmsd_deviation',
            'threshold_lines': [
                {'value': 3.0, 'label': 'Acceptable', 'color': 'green'},
                {'value': 5.0, 'label': 'Moderate', 'color': 'orange'},
                {'value': 8.0, 'label': 'Severe', 'color': 'red'}
            ],
            'secondary_structure_annotation': True
        }
        
        deviation_file = output_path / f"{pdb_id}_per_residue_deviation.json"
        with open(deviation_file, 'w') as f:
            json.dump(deviation_spec, f, indent=2)
        generated_files.append(str(deviation_file))
        
        # Generate energy landscape visualization metadata
        if 'energy' in report or 'final_energy' in report:
            energy_spec = {
                'pdb_id': pdb_id,
                'visualization_type': 'energy_landscape',
                'final_energy': report.get('energy', report.get('final_energy', 0.0)),
                'plot_type': 'energy_trajectory',
                'annotations': [
                    {'type': 'final_energy', 'value': report.get('energy', report.get('final_energy', 0.0))},
                    {'type': 'stability_threshold', 'value': 0.0, 'label': 'Stable/Unstable boundary'}
                ]
            }
            
            energy_file = output_path / f"{pdb_id}_energy_landscape.json"
            with open(energy_file, 'w') as f:
                json.dump(energy_spec, f, indent=2)
            generated_files.append(str(energy_file))
        
        return generated_files
    
    def analyze_energy_trajectory(
        self,
        trajectory_data: Dict
    ) -> TrajectoryAnalysis:
        """
        Analyze energy trajectory for local minima and convergence.
        
        Examines energy values over time to detect local minima traps,
        escape attempts, and overall convergence behavior.
        
        Args:
            trajectory_data: Dictionary with 'energies' list and optional metadata
        
        Returns:
            TrajectoryAnalysis with minima detection and convergence information
        
        Raises:
            ValueError: If trajectory_data is missing required fields
        
        Example:
            >>> trajectory = {'energies': [-40.0, -45.0, -43.0, -46.0, -45.5]}
            >>> analysis = analyzer.analyze_energy_trajectory(trajectory)
            >>> print(f"Minima count: {analysis.minima_count}")
            >>> print(f"Converged: {analysis.convergence_achieved}")
        """
        if 'energies' not in trajectory_data or not trajectory_data['energies']:
            raise ValueError("Trajectory data must contain non-empty 'energies' list")
        
        energies = trajectory_data['energies']
        
        # Detect local minima
        minima_count = self._count_local_minima(energies)
        
        # Detect escape attempts (energy increases after minima)
        escape_attempts = self._count_escape_attempts(energies)
        
        # Calculate energy variance
        energy_variance = statistics.variance(energies) if len(energies) > 1 else 0.0
        
        # Check convergence (energy stabilizes in last 20% of trajectory)
        convergence_achieved = self._check_convergence(energies)
        
        # Determine if stuck in local minimum
        stuck_in_local_minima = self._is_stuck_in_minimum(energies, minima_count, escape_attempts)
        
        return TrajectoryAnalysis(
            stuck_in_local_minima=stuck_in_local_minima,
            minima_count=minima_count,
            escape_attempts=escape_attempts,
            energy_variance=energy_variance,
            convergence_achieved=convergence_achieved
        )
    
    def recommend_parameter_adjustments(
        self,
        patterns: FailurePatterns
    ) -> List[str]:
        """
        Recommend parameter adjustments based on failure patterns.
        
        Analyzes common failure patterns to suggest specific parameter changes
        that may improve prediction accuracy.
        
        Args:
            patterns: FailurePatterns from extract_common_characteristics()
        
        Returns:
            List of parameter adjustment recommendations
        
        Example:
            >>> patterns = analyzer.extract_common_characteristics(failures)
            >>> recommendations = analyzer.recommend_parameter_adjustments(patterns)
            >>> for rec in recommendations:
            ...     print(f"• {rec}")
        """
        recommendations = []
        
        # Check for size-specific issues
        if patterns.common_size_category:
            if patterns.common_size_category == 'large':
                recommendations.append(
                    "Increase iterations_per_agent for large proteins (try 2000-5000)"
                )
                recommendations.append(
                    "Consider using adaptive_config for automatic scaling"
                )
                recommendations.append(
                    "Increase stuck_threshold for large proteins (try 15.0-20.0)"
                )
            elif patterns.common_size_category == 'tiny':
                recommendations.append(
                    "Decrease iterations_per_agent for tiny proteins (try 500-1000)"
                )
                recommendations.append(
                    "Reduce stuck_threshold for tiny proteins (try 3.0-5.0)"
                )
        
        # Check for high RMSD issues
        if any('High RMSD' in issue for issue in patterns.common_issues):
            recommendations.append(
                "Enable QCPP integration for physics-guided exploration"
            )
            recommendations.append(
                "Increase exploration_energy in behavioral parameters"
            )
            recommendations.append(
                "Try increasing agent population (num_agents=20-50)"
            )
        
        # Check for energy issues
        if any('Poor energy' in issue for issue in patterns.common_issues):
            recommendations.append(
                "Strengthen energy_function weights (bond, angle, dihedral)"
            )
            recommendations.append(
                "Enable QAAP resonance in QCPP integration"
            )
            recommendations.append(
                "Increase structural_focus in behavioral parameters"
            )
        
        # Check for GDT-TS issues
        if any('Low GDT-TS' in issue for issue in patterns.common_issues):
            recommendations.append(
                "Increase native_state_ambition parameter (try 0.8-0.9)"
            )
            recommendations.append(
                "Enable native structure guidance if available"
            )
            recommendations.append(
                "Try balanced or aggressive diversity profiles"
            )
        
        # Check for large protein bias
        if any('large proteins' in issue for issue in patterns.common_issues):
            recommendations.append(
                "Use adaptive configuration to auto-scale parameters by size"
            )
            recommendations.append(
                "Increase computational resources (parallel agents, longer runs)"
            )
            recommendations.append(
                "Consider hierarchical folding approach for very large proteins"
            )
        
        # Check secondary structure content
        helix_frac = patterns.average_secondary_structure_content.get('helix', 0.0)
        sheet_frac = patterns.average_secondary_structure_content.get('sheet', 0.0)
        
        if helix_frac < 0.2 and sheet_frac < 0.2:
            recommendations.append(
                "High coil content: Increase flexibility with lower structural_focus"
            )
        elif helix_frac > 0.6:
            recommendations.append(
                "High helix content: Use DSSP-based secondary structure constraints"
            )
        elif sheet_frac > 0.6:
            recommendations.append(
                "High sheet content: Strengthen hydrogen bond terms in energy function"
            )
        
        # General recommendations if no specific patterns found
        if not recommendations:
            recommendations.append(
                "No specific patterns detected. Try general improvements:"
            )
            recommendations.append(
                "  - Increase agent population (num_agents=20-50)"
            )
            recommendations.append(
                "  - Enable QCPP integration for physics guidance"
            )
            recommendations.append(
                "  - Use adaptive_config for automatic parameter tuning"
            )
        
        return recommendations
    
    def export_failure_report(
        self,
        output_path: str,
        classifications: List[FailureClassification],
        patterns: Optional[FailurePatterns] = None,
        recommendations: Optional[List[str]] = None
    ):
        """
        Export comprehensive failure analysis report to JSON.
        
        Args:
            output_path: Path to output JSON file
            classifications: List of failure classifications
            patterns: Optional failure patterns
            recommendations: Optional parameter recommendations
        
        Example:
            >>> analyzer.export_failure_report(
            ...     'failure_report.json',
            ...     classifications,
            ...     patterns,
            ...     recommendations
            ... )
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'failure_count': len(classifications),
            'failures': [self._dataclass_to_dict(c) for c in classifications],
        }
        
        if patterns:
            report['patterns'] = self._dataclass_to_dict(patterns)
        
        if recommendations:
            report['recommendations'] = recommendations
        
        # Add summary statistics
        if classifications:
            report['summary'] = {
                'avg_rmsd': statistics.mean(c.rmsd for c in classifications),
                'avg_energy': statistics.mean(c.energy for c in classifications),
                'avg_gdt_ts': statistics.mean(c.gdt_ts for c in classifications),
                'severity_distribution': self._count_severities(classifications),
                'failure_type_distribution': self._count_failure_types(classifications)
            }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Private helper methods
    
    def _determine_severity(self, rmsd: float, energy: float, gdt_ts: float) -> str:
        """Determine failure severity based on metrics."""
        severe_criteria = [
            rmsd > self.SEVERE_RMSD_THRESHOLD,
            energy > self.UNSTABLE_ENERGY_THRESHOLD,
            gdt_ts < self.SEVERE_GDT_THRESHOLD
        ]
        
        if sum(severe_criteria) >= 2:
            return 'severe'
        elif any(severe_criteria):
            return 'moderate'
        else:
            return 'minor'
    
    def _find_most_common(self, items: List[str]) -> str:
        """Find most common item in list."""
        if not items:
            return 'unknown'
        
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def _count_local_minima(self, energies: List[float]) -> int:
        """Count local minima in energy trajectory."""
        if len(energies) < 3:
            return 0
        
        minima_count = 0
        for i in range(1, len(energies) - 1):
            if energies[i] < energies[i-1] and energies[i] < energies[i+1]:
                minima_count += 1
        
        return minima_count
    
    def _count_escape_attempts(self, energies: List[float]) -> int:
        """Count escape attempts from local minima."""
        if len(energies) < 4:
            return 0
        
        escape_count = 0
        for i in range(1, len(energies) - 2):
            # Minimum followed by increase then decrease
            if (energies[i] < energies[i-1] and 
                energies[i+1] > energies[i] and 
                energies[i+2] < energies[i+1]):
                escape_count += 1
        
        return escape_count
    
    def _check_convergence(self, energies: List[float]) -> bool:
        """Check if energy has converged (stabilized)."""
        if len(energies) < 10:
            return False
        
        # Check last 20% of trajectory
        window_size = max(5, len(energies) // 5)
        recent_energies = energies[-window_size:]
        
        # Converged if variance is low
        variance = statistics.variance(recent_energies) if len(recent_energies) > 1 else 0.0
        return variance < 1.0  # Low energy variance indicates convergence
    
    def _is_stuck_in_minimum(
        self,
        energies: List[float],
        minima_count: int,
        escape_attempts: int
    ) -> bool:
        """Determine if system is stuck in local minimum."""
        if len(energies) < 10:
            return False
        
        # Multiple minima with few successful escapes suggests stuck
        if minima_count > 3 and escape_attempts < minima_count // 2:
            return True
        
        # Long period without significant energy improvement
        recent_window = energies[-min(20, len(energies)):]
        if len(recent_window) > 5:
            energy_change = abs(recent_window[-1] - recent_window[0])
            if energy_change < 1.0:  # Very little change
                return True
        
        return False
    
    def _count_severities(self, classifications: List[FailureClassification]) -> Dict[str, int]:
        """Count failures by severity."""
        counts = {'minor': 0, 'moderate': 0, 'severe': 0}
        for c in classifications:
            counts[c.severity] = counts.get(c.severity, 0) + 1
        return counts
    
    def _count_failure_types(self, classifications: List[FailureClassification]) -> Dict[str, int]:
        """Count failures by type."""
        counts = {}
        for c in classifications:
            counts[c.failure_type] = counts.get(c.failure_type, 0) + 1
        return counts
    
    def _dataclass_to_dict(self, obj) -> Dict:
        """Convert dataclass instance to dictionary for JSON serialization."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = self._dataclass_to_dict(value)
                elif isinstance(value, dict):
                    result[field_name] = {
                        k: self._dataclass_to_dict(v) if hasattr(v, '__dataclass_fields__') else v
                        for k, v in value.items()
                    }
                else:
                    result[field_name] = value
            return result
        return obj
