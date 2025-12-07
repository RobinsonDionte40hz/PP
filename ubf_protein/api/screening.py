"""
Screening API - Public interface for aggregation screening.

This module exposes the screening functionality through a clean API,
hiding internal implementation details.
"""

from typing import Optional, Callable, List
import logging

from .interfaces import IScreener
from .schemas import (
    ScreeningConfig, 
    ScreeningResults, 
    AggregationRegion,
    AggregationRisk,
)

logger = logging.getLogger(__name__)


class AggregationScreener(IScreener):
    """
    Public interface for aggregation screening.
    
    Analyzes protein sequences for aggregation-prone regions.
    
    Usage:
        from ubf_protein.api import AggregationScreener, ScreeningConfig
        
        screener = AggregationScreener()
        results = screener.screen("MQIFVKTLTGK...")
        
        print(f"Risk level: {results.risk_level}")
        print(f"Passes screening: {results.passes_screening}")
    """
    
    def __init__(self):
        """Initialize the screener."""
        self._internal_screener = None
        
    def screen(
        self,
        sequence: str,
        config: Optional[ScreeningConfig] = None
    ) -> ScreeningResults:
        """
        Screen a protein sequence for aggregation-prone regions.
        
        Args:
            sequence: Amino acid sequence to screen
            config: Optional screening configuration
            
        Returns:
            ScreeningResults with identified regions and scores
        """
        config = config or ScreeningConfig()
        
        try:
            # Lazy import internal implementation
            from ..aggregation_screening import (
                AggregationScreener as InternalScreener,
                ScreeningConfig as InternalConfig,
                AggregationRisk as InternalRisk,
            )
            
            # Create internal config based on window_size/threshold
            # Map our simple config to internal config presets
            if config.window_size <= 5:
                internal_config = InternalConfig.fast()
            elif config.window_size >= 9:
                internal_config = InternalConfig.thorough()
            else:
                internal_config = InternalConfig.balanced()
            
            screener = InternalScreener(internal_config)
            
            # Run screening
            internal_results = screener.screen_sequence(sequence)
            
            # Convert to public schema
            return self._convert_results(internal_results)
            
        except ImportError as e:
            logger.warning(f"Internal screener not available ({e}), using fallback")
            return self._fallback_screen(sequence, config)
        except Exception as e:
            logger.error(f"Screening failed: {e}")
            raise RuntimeError(f"Screening failed: {e}") from e
    
    def batch_screen(
        self,
        sequences: List[str],
        config: Optional[ScreeningConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ScreeningResults]:
        """
        Screen multiple sequences.
        
        Args:
            sequences: List of sequences to screen
            config: Optional screening configuration
            progress_callback: Callback with (current, total) progress
            
        Returns:
            List of screening results, one per sequence
        """
        results = []
        total = len(sequences)
        
        for i, seq in enumerate(sequences):
            results.append(self.screen(seq, config))
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def _convert_results(self, internal_results) -> ScreeningResults:
        """Convert internal AggregationMetrics to public ScreeningResults."""
        from ..aggregation_screening import AggregationRisk as InternalRisk
        
        # Map internal risk level to public enum
        risk_map = {
            InternalRisk.LOW: AggregationRisk.LOW,
            InternalRisk.MODERATE: AggregationRisk.MODERATE,
            InternalRisk.HIGH: AggregationRisk.HIGH,
            InternalRisk.CRITICAL: AggregationRisk.CRITICAL,
        }
        
        return ScreeningResults(
            sequence=internal_results.sequence,
            sequence_length=internal_results.sequence_length,
            aggregation_score=internal_results.aggregation_score,
            energy_score=internal_results.energy_score,
            structure_score=internal_results.structure_score,
            hydrophobic_score=internal_results.hydrophobic_score,
            compactness_score=internal_results.compactness_score,
            risk_level=risk_map[internal_results.risk_level],
            risk_factors=internal_results.risk_factors,
            passes_screening=internal_results.passes_screening,
            final_energy=internal_results.final_energy,
            secondary_structure_pct=internal_results.secondary_structure_pct,
            radius_of_gyration=internal_results.radius_of_gyration,
            screening_time_ms=internal_results.screening_time_ms,
            regions=[],  # Internal results don't have regions list
            per_residue_scores=[],  # Could be added if needed
            recommendations=self._generate_recommendations_from_metrics(internal_results),
        )
    
    def _generate_recommendations_from_metrics(self, metrics) -> List[str]:
        """Generate recommendations based on screening metrics."""
        recommendations = []
        
        if metrics.passes_screening:
            recommendations.append(f"Sequence passes screening with {metrics.risk_level.value} risk.")
        else:
            recommendations.append(f"Sequence has {metrics.risk_level.value} aggregation risk.")
        
        for factor in metrics.risk_factors[:3]:  # Top 3 risk factors
            recommendations.append(f"Risk factor: {factor}")
        
        return recommendations
    
    def _fallback_screen(
        self, 
        sequence: str, 
        config: ScreeningConfig
    ) -> ScreeningResults:
        """
        Fallback screening when internal implementation unavailable.
        
        Uses simple heuristics for basic aggregation detection.
        """
        # Simple hydrophobic stretch detection
        hydrophobic = set('VILMFYW')
        window_size = config.window_size
        
        per_residue_scores = []
        regions = []
        
        for i in range(len(sequence)):
            # Calculate local hydrophobicity
            start = max(0, i - window_size // 2)
            end = min(len(sequence), i + window_size // 2 + 1)
            window = sequence[start:end]
            
            hydro_count = sum(1 for aa in window if aa in hydrophobic)
            score = hydro_count / len(window)
            per_residue_scores.append(score)
        
        # Find regions above threshold
        in_region = False
        region_start = 0
        
        for i, score in enumerate(per_residue_scores):
            if score >= config.threshold and not in_region:
                in_region = True
                region_start = i
            elif score < config.threshold and in_region:
                in_region = False
                if i - region_start >= 5:  # Minimum region size
                    avg_score = sum(per_residue_scores[region_start:i]) / (i - region_start)
                    regions.append(AggregationRegion(
                        start=region_start,
                        end=i,
                        sequence=sequence[region_start:i],
                        score=avg_score,
                        type='hydrophobic',
                    ))
        
        # Close any open region
        if in_region and len(sequence) - region_start >= 5:
            avg_score = sum(per_residue_scores[region_start:]) / (len(sequence) - region_start)
            regions.append(AggregationRegion(
                start=region_start,
                end=len(sequence),
                sequence=sequence[region_start:],
                score=avg_score,
                type='hydrophobic',
            ))
        
        # Calculate overall score (invert so higher = better)
        avg_hydrophobicity = sum(per_residue_scores) / len(per_residue_scores) if per_residue_scores else 0.0
        aggregation_score = 1.0 - avg_hydrophobicity
        
        # Determine risk level
        if aggregation_score >= 0.7:
            risk_level = AggregationRisk.LOW
        elif aggregation_score >= 0.5:
            risk_level = AggregationRisk.MODERATE
        elif aggregation_score >= 0.3:
            risk_level = AggregationRisk.HIGH
        else:
            risk_level = AggregationRisk.CRITICAL
        
        risk_factors = []
        if len(regions) > 0:
            risk_factors.append(f"Found {len(regions)} hydrophobic stretch(es)")
        if avg_hydrophobicity > 0.5:
            risk_factors.append("High overall hydrophobicity")
        
        return ScreeningResults(
            sequence=sequence,
            sequence_length=len(sequence),
            aggregation_score=aggregation_score,
            energy_score=0.5,  # Unknown in fallback
            structure_score=0.5,  # Unknown in fallback
            hydrophobic_score=1.0 - avg_hydrophobicity,
            compactness_score=0.5,  # Unknown in fallback
            risk_level=risk_level,
            risk_factors=risk_factors,
            passes_screening=risk_level in (AggregationRisk.LOW, AggregationRisk.MODERATE),
            final_energy=0.0,  # Unknown in fallback
            secondary_structure_pct=0.0,  # Unknown in fallback
            radius_of_gyration=0.0,  # Unknown in fallback
            screening_time_ms=0.0,
            regions=regions,
            per_residue_scores=per_residue_scores,
            recommendations=self._generate_recommendations(regions),
        )
    
    def _generate_recommendations(self, regions: List[AggregationRegion]) -> List[str]:
        """Generate recommendations based on identified regions."""
        recommendations = []
        
        if not regions:
            recommendations.append("No significant aggregation-prone regions detected.")
        else:
            recommendations.append(
                f"Found {len(regions)} potential aggregation-prone region(s)."
            )
            
            for i, region in enumerate(regions[:3]):  # Top 3 regions
                if region.score > 0.7:
                    recommendations.append(
                        f"Region {region.start+1}-{region.end}: High risk. "
                        f"Consider introducing charged residues (K, R, E, D)."
                    )
                elif region.score > 0.5:
                    recommendations.append(
                        f"Region {region.start+1}-{region.end}: Moderate risk. "
                        f"Monitor during expression."
                    )
        
        return recommendations
