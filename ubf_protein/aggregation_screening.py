"""
Aggregation Risk Screening Module for UBF Protein System.

This module provides fast screening of protein sequences for aggregation propensity.
Unlike structure prediction (which tries to find THE native fold), screening answers:
"Will this sequence fold stably, or is it likely to aggregate/clump?"

USE CASES:
- Screen 100s-1000s of sequences to filter out aggregation-prone candidates
- Therapeutic protein development (biologics need stable sequences)
- Peptide library screening
- Protein engineering - identify problematic mutations

Key Insight: We don't need the correct structure - we need to know if ANY stable
structure forms. High energy, poor structure formation, and exposed hydrophobic
residues all indicate aggregation risk.

Metrics used:
1. Energy stability (negative = stable, positive = unstable)
2. Secondary structure % (high = well-folded)
3. Hydrophobic clustering (core formation = good)
4. Convergence rate (fast = folds easily)
5. Radius of gyration (compact = folded)
"""

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json
import csv
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AggregationRisk(Enum):
    """Aggregation risk classification."""
    LOW = "low"           # Likely to fold stably
    MODERATE = "moderate" # Some concerns, may need optimization
    HIGH = "high"         # Likely to aggregate
    CRITICAL = "critical" # Almost certainly will aggregate


@dataclass
class ConformationSnapshot:
    """
    A unique conformation found during screening.
    
    Allows researchers to examine all structural states discovered,
    not just the best one.
    """
    conformation_id: int          # Index in discovery order
    energy: float                 # kcal/mol
    rmsd_from_best: float         # Å from lowest-energy structure
    radius_of_gyration: float     # Å
    secondary_structure_pct: float  # 0-100%
    coordinates: List[List[float]]  # [[x,y,z], ...] for each CA atom
    discovered_at_iteration: int
    agent_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/PDB export."""
        return {
            'id': self.conformation_id,
            'energy': self.energy,
            'rmsd_from_best': self.rmsd_from_best,
            'radius_of_gyration': self.radius_of_gyration,
            'secondary_structure_pct': self.secondary_structure_pct,
            'discovered_at_iteration': self.discovered_at_iteration,
            'agent_id': self.agent_id,
            'num_atoms': len(self.coordinates),
        }
    
    def to_pdb_string(self, sequence: str) -> str:
        """Export as PDB format string."""
        lines = [f"REMARK   Conformation {self.conformation_id}, Energy: {self.energy:.2f} kcal/mol"]
        lines.append(f"REMARK   RMSD from best: {self.rmsd_from_best:.2f} A, Rg: {self.radius_of_gyration:.2f} A")
        
        aa_3letter = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        }
        
        for i, (coord, aa) in enumerate(zip(self.coordinates, sequence)):
            res_name = aa_3letter.get(aa, 'UNK')
            x, y, z = coord
            lines.append(
                f"ATOM  {i+1:5d}  CA  {res_name} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
        lines.append("END")
        return '\n'.join(lines)


@dataclass
class AggregationMetrics:
    """
    Detailed aggregation risk metrics for a sequence.
    
    All scores are 0-1, where higher = better (lower risk).
    """
    sequence: str
    sequence_length: int
    
    # Core stability metrics (0-1, higher = better)
    energy_score: float           # Based on final energy
    structure_score: float        # Based on secondary structure %
    hydrophobic_score: float      # Based on core formation
    convergence_score: float      # Based on how easily it folds
    compactness_score: float      # Based on radius of gyration
    
    # Raw values for debugging
    final_energy: float           # kcal/mol
    secondary_structure_pct: float  # 0-100%
    hydrophobic_clustering: float # avg distance between hydrophobic residues
    convergence_iterations: int   # iterations to reach stable state
    radius_of_gyration: float     # Angstroms
    
    # Composite score and classification
    aggregation_score: float      # 0-1, higher = LESS likely to aggregate
    risk_level: AggregationRisk
    risk_factors: List[str] = field(default_factory=list)
    
    # Metadata
    screening_time_ms: float = 0.0
    iterations_used: int = 0
    
    # Unique conformations found (optional, populated when include_conformations=True)
    conformations: List[ConformationSnapshot] = field(default_factory=list)
    num_unique_conformations: int = 0
    
    def to_dict(self, include_coordinates: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['risk_level'] = self.risk_level.value
        
        # Convert conformations
        if self.conformations:
            if include_coordinates:
                result['conformations'] = [c.to_dict() | {'coordinates': c.coordinates} 
                                          for c in self.conformations]
            else:
                result['conformations'] = [c.to_dict() for c in self.conformations]
        else:
            result.pop('conformations', None)
        
        return result
    
    @property
    def passes_screening(self) -> bool:
        """Returns True if sequence passes basic screening (low/moderate risk)."""
        return self.risk_level in (AggregationRisk.LOW, AggregationRisk.MODERATE)
    
    def get_conformation_by_energy_rank(self, rank: int = 0) -> Optional[ConformationSnapshot]:
        """Get conformation by energy rank (0 = lowest energy)."""
        if not self.conformations:
            return None
        sorted_conf = sorted(self.conformations, key=lambda c: c.energy)
        if rank < len(sorted_conf):
            return sorted_conf[rank]
        return None
    
    def export_all_conformations_pdb(self, output_dir: str) -> List[str]:
        """Export all conformations as separate PDB files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        paths = []
        for conf in sorted(self.conformations, key=lambda c: c.energy):
            filename = f"conf_{conf.conformation_id:03d}_E{conf.energy:.1f}.pdb"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(conf.to_pdb_string(self.sequence))
            paths.append(filepath)
        return paths


@dataclass 
class ScreeningConfig:
    """Configuration for fast screening mode."""
    
    # Speed vs accuracy tradeoff
    iterations: int = 100         # Fewer iterations for speed
    agents: int = 3               # Fewer agents for speed
    
    # Thresholds for risk classification
    energy_threshold_good: float = -50.0    # kcal/mol, better than this = good
    energy_threshold_bad: float = 0.0       # kcal/mol, worse than this = bad
    structure_threshold_good: float = 60.0  # %, more structure = good
    structure_threshold_bad: float = 30.0   # %, less structure = bad
    
    # Enable/disable expensive calculations
    enable_qcpp: bool = False     # Disable for max speed
    
    @classmethod
    def fast(cls) -> 'ScreeningConfig':
        """Ultra-fast screening (sacrifices accuracy)."""
        return cls(iterations=50, agents=2)
    
    @classmethod
    def balanced(cls) -> 'ScreeningConfig':
        """Balanced speed and accuracy."""
        return cls(iterations=100, agents=3)
    
    @classmethod
    def thorough(cls) -> 'ScreeningConfig':
        """More thorough screening."""
        return cls(iterations=200, agents=5, enable_qcpp=True)


class AggregationScreener:
    """
    Fast aggregation risk screening for protein sequences.
    
    This class provides quick assessment of whether a protein sequence
    is likely to fold into a stable structure or aggregate.
    
    Example usage:
        screener = AggregationScreener()
        
        # Screen single sequence
        result = screener.screen_sequence("ACDEFGHIKLMNPQRSTVWY")
        print(f"Risk: {result.risk_level.value}")
        
        # Batch screening
        sequences = ["AAAAAAAAAA", "ACDEFGHIKL", ...]
        results = screener.screen_batch(sequences)
        
        # Export to CSV
        screener.export_csv(results, "screening_results.csv")
    """
    
    # Hydrophobic residues (Kyte-Doolittle scale)
    HYDROPHOBIC = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'}
    VERY_HYDROPHOBIC = {'V', 'I', 'L', 'F', 'W'}  # Most aggregation-prone
    
    # Charged residues (help solubility)
    CHARGED = {'R', 'K', 'D', 'E'}
    
    # Aggregation-prone patterns
    AGGREGATION_PATTERNS = [
        ('VVVV', 'poly-valine'),
        ('IIII', 'poly-isoleucine'),
        ('FFFF', 'poly-phenylalanine'),
        ('LLLL', 'poly-leucine'),
        ('AAAA', 'poly-alanine'),
        ('PPPP', 'poly-proline (can cause issues)'),
    ]
    
    def __init__(self, config: Optional[ScreeningConfig] = None):
        """
        Initialize the screener.
        
        Args:
            config: Screening configuration. Uses balanced config if None.
        """
        self.config = config or ScreeningConfig.balanced()
        self._coordinator = None
        self._energy_calculator = None
    
    def _get_coordinator(self, sequence: str):
        """Get or create coordinator for the sequence."""
        # Import here to avoid circular imports
        try:
            from .multi_agent_coordinator import MultiAgentCoordinator
            from .energy_function import MolecularMechanicsEnergy
        except ImportError:
            from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
            from ubf_protein.energy_function import MolecularMechanicsEnergy
        
        coordinator = MultiAgentCoordinator(
            protein_sequence=sequence,
            enable_checkpointing=False,  # No checkpoints for screening
        )
        # Initialize agents separately
        coordinator.initialize_agents(count=self.config.agents, diversity_profile="balanced")
        
        return coordinator
    
    def _get_energy_calculator(self):
        """Get energy calculator instance."""
        if self._energy_calculator is None:
            try:
                from .energy_function import MolecularMechanicsEnergy
            except ImportError:
                from ubf_protein.energy_function import MolecularMechanicsEnergy
            self._energy_calculator = MolecularMechanicsEnergy()
        return self._energy_calculator
    
    def _analyze_sequence_composition(self, sequence: str) -> Dict[str, Any]:
        """
        Analyze sequence composition for aggregation risk factors.
        
        This is a FAST pre-screen before running any simulations.
        """
        n = len(sequence)
        
        # Count residue types
        hydrophobic_count = sum(1 for aa in sequence if aa in self.HYDROPHOBIC)
        very_hydrophobic_count = sum(1 for aa in sequence if aa in self.VERY_HYDROPHOBIC)
        charged_count = sum(1 for aa in sequence if aa in self.CHARGED)
        
        # Calculate ratios
        hydrophobic_ratio = hydrophobic_count / n if n > 0 else 0
        charged_ratio = charged_count / n if n > 0 else 0
        
        # Check for aggregation-prone patterns
        patterns_found = []
        for pattern, name in self.AGGREGATION_PATTERNS:
            if pattern in sequence:
                patterns_found.append(name)
        
        # Check for hydrophobic stretches
        max_hydrophobic_stretch = 0
        current_stretch = 0
        for aa in sequence:
            if aa in self.HYDROPHOBIC:
                current_stretch += 1
                max_hydrophobic_stretch = max(max_hydrophobic_stretch, current_stretch)
            else:
                current_stretch = 0
        
        return {
            'hydrophobic_ratio': hydrophobic_ratio,
            'charged_ratio': charged_ratio,
            'patterns_found': patterns_found,
            'max_hydrophobic_stretch': max_hydrophobic_stretch,
            'very_hydrophobic_count': very_hydrophobic_count,
        }
    
    def _calculate_radius_of_gyration(self, coords: List[Tuple[float, float, float]]) -> float:
        """Calculate radius of gyration from coordinates."""
        if not coords or len(coords) < 2:
            return float('inf')
        
        n = len(coords)
        center_x = sum(c[0] for c in coords) / n
        center_y = sum(c[1] for c in coords) / n
        center_z = sum(c[2] for c in coords) / n
        
        rg_sq = sum((c[0] - center_x)**2 + (c[1] - center_y)**2 + (c[2] - center_z)**2 
                    for c in coords) / n
        return math.sqrt(rg_sq)
    
    def _calculate_hydrophobic_clustering(
        self, 
        sequence: str, 
        coords: List[Tuple[float, float, float]]
    ) -> float:
        """
        Calculate average distance between hydrophobic residues.
        Lower distance = better clustering = lower aggregation risk.
        """
        hydrophobic_indices = [i for i, aa in enumerate(sequence) if aa in self.HYDROPHOBIC]
        
        if len(hydrophobic_indices) < 2:
            return 0.0  # Not enough to measure
        
        total_distance = 0.0
        pair_count = 0
        
        for i, idx1 in enumerate(hydrophobic_indices):
            for idx2 in hydrophobic_indices[i+1:]:
                if idx1 < len(coords) and idx2 < len(coords):
                    c1, c2 = coords[idx1], coords[idx2]
                    dist = math.sqrt(
                        (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2
                    )
                    total_distance += dist
                    pair_count += 1
        
        if pair_count == 0:
            return float('inf')
        
        return total_distance / pair_count
    
    def _estimate_secondary_structure(
        self, 
        coords: List[Tuple[float, float, float]]
    ) -> float:
        """
        Estimate secondary structure percentage from coordinates.
        Uses distance patterns to detect helix/sheet formation.
        """
        if len(coords) < 4:
            return 0.0
        
        structured_count = 0
        
        # Check for helical patterns (i, i+3 distance ~5Å)
        for i in range(len(coords) - 3):
            c1, c2 = coords[i], coords[i + 3]
            dist = math.sqrt(
                (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2
            )
            if 4.5 < dist < 6.5:  # Helical range
                structured_count += 1
        
        # Check for sheet patterns (i, i+2 distance ~6-7Å)
        for i in range(len(coords) - 2):
            c1, c2 = coords[i], coords[i + 2]
            dist = math.sqrt(
                (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2
            )
            if 6.0 < dist < 8.0:  # Extended/sheet range
                structured_count += 0.5  # Partial credit
        
        # Normalize to percentage
        max_possible = len(coords) - 3 + (len(coords) - 2) * 0.5
        if max_possible <= 0:
            return 0.0
        
        return min(100.0, (structured_count / max_possible) * 100.0)
    
    def _score_energy(self, energy: float) -> float:
        """Convert energy to 0-1 score (higher = better/more stable)."""
        # Good: < -50 kcal/mol → score ~1.0
        # Bad: > 0 kcal/mol → score ~0.0
        good = self.config.energy_threshold_good
        bad = self.config.energy_threshold_bad
        
        if energy <= good:
            return 1.0
        elif energy >= bad:
            return 0.0
        else:
            return (bad - energy) / (bad - good)
    
    def _score_structure(self, structure_pct: float) -> float:
        """Convert structure % to 0-1 score (higher = better)."""
        good = self.config.structure_threshold_good
        bad = self.config.structure_threshold_bad
        
        if structure_pct >= good:
            return 1.0
        elif structure_pct <= bad:
            return 0.0
        else:
            return (structure_pct - bad) / (good - bad)
    
    def _score_hydrophobic(self, avg_distance: float, seq_length: int) -> float:
        """Convert hydrophobic clustering distance to 0-1 score."""
        # Expected distance for well-folded: ~8-12Å
        # Extended/bad: > 20Å
        ideal_distance = 10.0
        bad_distance = 25.0
        
        if avg_distance <= ideal_distance:
            return 1.0
        elif avg_distance >= bad_distance:
            return 0.0
        else:
            return (bad_distance - avg_distance) / (bad_distance - ideal_distance)
    
    def _score_compactness(self, rg: float, seq_length: int) -> float:
        """Convert radius of gyration to 0-1 score."""
        # Ideal Rg ≈ 3 × N^(1/3) Å
        ideal_rg = 3.0 * (seq_length ** (1.0/3.0))
        
        # Extended is ~2-3x ideal
        extended_rg = ideal_rg * 2.5
        
        if rg <= ideal_rg:
            return 1.0
        elif rg >= extended_rg:
            return 0.0
        else:
            return (extended_rg - rg) / (extended_rg - ideal_rg)
    
    def _classify_risk(
        self, 
        score: float, 
        risk_factors: List[str]
    ) -> AggregationRisk:
        """Classify risk level based on score and factors."""
        # Critical risk factors override score
        critical_factors = [
            'poly-valine', 'poly-isoleucine', 'poly-phenylalanine',
            'extremely_hydrophobic', 'no_charged_residues'
        ]
        
        for factor in critical_factors:
            if factor in risk_factors:
                return AggregationRisk.CRITICAL
        
        # Score-based classification
        if score >= 0.7:
            return AggregationRisk.LOW
        elif score >= 0.5:
            return AggregationRisk.MODERATE
        elif score >= 0.3:
            return AggregationRisk.HIGH
        else:
            return AggregationRisk.CRITICAL
    
    def screen_sequence(
        self, 
        sequence: str,
        include_conformations: bool = False,
        rmsd_threshold: float = 2.0,
    ) -> AggregationMetrics:
        """
        Screen a single sequence for aggregation risk.
        
        Args:
            sequence: Protein sequence (amino acids)
            include_conformations: If True, collect all unique conformations found
            rmsd_threshold: RMSD threshold (Å) for considering conformations unique
            
        Returns:
            AggregationMetrics with risk assessment and optionally all unique structures
        """
        start_time = time.time()
        sequence = sequence.strip().upper()
        n = len(sequence)
        
        # Fast pre-screen based on composition
        composition = self._analyze_sequence_composition(sequence)
        risk_factors = composition['patterns_found'].copy()
        
        # Flag extreme compositions
        if composition['hydrophobic_ratio'] > 0.6:
            risk_factors.append('extremely_hydrophobic')
        if composition['charged_ratio'] < 0.05 and n > 10:
            risk_factors.append('no_charged_residues')
        if composition['max_hydrophobic_stretch'] > 8:
            risk_factors.append(f"long_hydrophobic_stretch_{composition['max_hydrophobic_stretch']}")
        
        # Run short simulation
        coordinator = self._get_coordinator(sequence)
        
        # Storage for unique conformations
        unique_conformations: List[ConformationSnapshot] = []
        best_coords = []
        
        try:
            # Run exploration
            results = coordinator.run_parallel_exploration(
                iterations=self.config.iterations
            )
            
            # Get best conformation
            best_conf = results.best_conformation
            best_coords = best_conf.atom_coordinates if best_conf else []
            final_energy = best_conf.energy if best_conf else 1000.0
            
            # Collect unique conformations if requested
            if include_conformations and hasattr(results, 'all_conformations'):
                unique_conformations = self._collect_unique_conformations(
                    results.all_conformations,
                    sequence,
                    best_coords,
                    rmsd_threshold
                )
            elif include_conformations:
                # Fallback: collect from agents' best conformations
                unique_conformations = self._collect_from_agents(
                    coordinator,
                    sequence,
                    best_coords,
                    rmsd_threshold
                )
            
            # Calculate metrics
            rg = self._calculate_radius_of_gyration(best_coords)
            hydro_dist = self._calculate_hydrophobic_clustering(sequence, best_coords)
            structure_pct = self._estimate_secondary_structure(best_coords)
            
            # Score each metric (0-1, higher = better)
            energy_score = self._score_energy(final_energy)
            structure_score = self._score_structure(structure_pct)
            hydrophobic_score = self._score_hydrophobic(hydro_dist, n)
            compactness_score = self._score_compactness(rg, n)
            
            # Convergence score (did it find something quickly?)
            convergence_iterations = results.total_iterations
            convergence_score = min(1.0, 50.0 / max(1, convergence_iterations / self.config.agents))
            
        except Exception as e:
            logger.error(f"Screening failed for sequence: {e}")
            # Fallback to composition-only assessment
            energy_score = 0.0
            structure_score = 0.0
            hydrophobic_score = 0.5
            compactness_score = 0.0
            convergence_score = 0.0
            final_energy = 1000.0
            structure_pct = 0.0
            hydro_dist = float('inf')
            rg = float('inf')
            convergence_iterations = self.config.iterations
            risk_factors.append('simulation_failed')
        
        # Composite aggregation score
        # Weights: energy (30%), structure (25%), hydrophobic (20%), compactness (15%), convergence (10%)
        aggregation_score = (
            0.30 * energy_score +
            0.25 * structure_score +
            0.20 * hydrophobic_score +
            0.15 * compactness_score +
            0.10 * convergence_score
        )
        
        # Add risk factors based on scores
        if energy_score < 0.3:
            risk_factors.append('unstable_energy')
        if structure_score < 0.3:
            risk_factors.append('poor_structure_formation')
        if hydrophobic_score < 0.3:
            risk_factors.append('exposed_hydrophobic_residues')
        if compactness_score < 0.3:
            risk_factors.append('extended_conformation')
        
        # Classify risk
        risk_level = self._classify_risk(aggregation_score, risk_factors)
        
        screening_time_ms = (time.time() - start_time) * 1000
        
        return AggregationMetrics(
            sequence=sequence,
            sequence_length=n,
            energy_score=energy_score,
            structure_score=structure_score,
            hydrophobic_score=hydrophobic_score,
            convergence_score=convergence_score,
            compactness_score=compactness_score,
            final_energy=final_energy,
            secondary_structure_pct=structure_pct,
            hydrophobic_clustering=hydro_dist,
            convergence_iterations=convergence_iterations,
            radius_of_gyration=rg,
            aggregation_score=aggregation_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            screening_time_ms=screening_time_ms,
            iterations_used=self.config.iterations,
            conformations=unique_conformations,
            num_unique_conformations=len(unique_conformations),
        )
    
    def _collect_unique_conformations(
        self,
        all_conformations: List[Any],
        sequence: str,
        best_coords: List[List[float]],
        rmsd_threshold: float
    ) -> List[ConformationSnapshot]:
        """Collect unique conformations from exploration results."""
        unique: List[ConformationSnapshot] = []
        seen_coords: List[List[List[float]]] = []
        
        for i, conf in enumerate(all_conformations):
            coords = conf.atom_coordinates if hasattr(conf, 'atom_coordinates') else []
            if not coords:
                continue
            
            # Check if this is unique (different from all seen)
            is_unique = True
            for seen in seen_coords:
                rmsd = self._calculate_rmsd(coords, seen)
                if rmsd < rmsd_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                seen_coords.append(coords)
                rmsd_from_best = self._calculate_rmsd(coords, best_coords) if best_coords else 0.0
                
                snapshot = ConformationSnapshot(
                    conformation_id=len(unique),
                    energy=conf.energy if hasattr(conf, 'energy') else 0.0,
                    rmsd_from_best=rmsd_from_best,
                    radius_of_gyration=self._calculate_radius_of_gyration(coords),
                    secondary_structure_pct=self._estimate_secondary_structure(coords),
                    coordinates=coords,
                    discovered_at_iteration=getattr(conf, 'iteration', i),
                    agent_id=getattr(conf, 'agent_id', 0),
                )
                unique.append(snapshot)
        
        # Sort by energy (lowest first)
        return sorted(unique, key=lambda c: c.energy)
    
    def _collect_from_agents(
        self,
        coordinator,
        sequence: str,
        best_coords: List[List[float]],
        rmsd_threshold: float
    ) -> List[ConformationSnapshot]:
        """Collect unique conformations from agents' best AND current states."""
        unique: List[ConformationSnapshot] = []
        seen_coords: List[List[List[float]]] = []
        
        # Get agents from coordinator
        agents = getattr(coordinator, '_agents', [])
        
        for agent_idx, agent in enumerate(agents):
            # Try to get best conformation from agent (method or attribute)
            best = None
            if hasattr(agent, 'get_best_conformation'):
                best = agent.get_best_conformation()
            elif hasattr(agent, '_best_conformation'):
                best = agent._best_conformation
            elif hasattr(agent, 'best_conformation'):
                best = agent.best_conformation
            
            if not best:
                continue
            
            coords = best.atom_coordinates if hasattr(best, 'atom_coordinates') else []
            if not coords:
                continue
            
            # Check uniqueness
            is_unique = True
            for seen in seen_coords:
                rmsd = self._calculate_rmsd(coords, seen)
                if rmsd < rmsd_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                seen_coords.append(coords)
                rmsd_from_best = self._calculate_rmsd(coords, best_coords) if best_coords else 0.0
                
                snapshot = ConformationSnapshot(
                    conformation_id=len(unique),
                    energy=best.energy if hasattr(best, 'energy') else 0.0,
                    rmsd_from_best=rmsd_from_best,
                    radius_of_gyration=self._calculate_radius_of_gyration(coords),
                    secondary_structure_pct=self._estimate_secondary_structure(coords),
                    coordinates=coords,
                    discovered_at_iteration=getattr(best, 'iteration', 0),
                    agent_id=agent_idx,
                )
                unique.append(snapshot)
            
            # Also try to get current conformation (might differ from best)
            current = None
            if hasattr(agent, '_current_conformation'):
                current = agent._current_conformation
            elif hasattr(agent, 'current_conformation'):
                current = agent.current_conformation
            
            if current and current != best:
                curr_coords = current.atom_coordinates if hasattr(current, 'atom_coordinates') else []
                if curr_coords:
                    is_curr_unique = True
                    for seen in seen_coords:
                        if self._calculate_rmsd(curr_coords, seen) < rmsd_threshold:
                            is_curr_unique = False
                            break
                    
                    if is_curr_unique:
                        seen_coords.append(curr_coords)
                        rmsd_from_best = self._calculate_rmsd(curr_coords, best_coords) if best_coords else 0.0
                        
                        snapshot = ConformationSnapshot(
                            conformation_id=len(unique),
                            energy=current.energy if hasattr(current, 'energy') else 0.0,
                            rmsd_from_best=rmsd_from_best,
                            radius_of_gyration=self._calculate_radius_of_gyration(curr_coords),
                            secondary_structure_pct=self._estimate_secondary_structure(curr_coords),
                            coordinates=curr_coords,
                            discovered_at_iteration=getattr(current, 'iteration', 0),
                            agent_id=agent_idx,
                        )
                        unique.append(snapshot)
        
        return sorted(unique, key=lambda c: c.energy)
    
    def _calculate_rmsd(
        self,
        coords1: List[List[float]],
        coords2: List[List[float]]
    ) -> float:
        """Calculate RMSD between two coordinate sets."""
        if len(coords1) != len(coords2) or not coords1:
            return float('inf')
        
        total = 0.0
        for c1, c2 in zip(coords1, coords2):
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            dz = c1[2] - c2[2]
            total += dx*dx + dy*dy + dz*dz
        
        return (total / len(coords1)) ** 0.5
    
    def screen_batch(
        self, 
        sequences: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[AggregationMetrics]:
        """
        Screen multiple sequences for aggregation risk.
        
        Args:
            sequences: List of protein sequences
            progress_callback: Optional callback(current, total, result)
            
        Returns:
            List of AggregationMetrics, sorted by aggregation_score (best first)
        """
        results = []
        total = len(sequences)
        
        for i, seq in enumerate(sequences):
            try:
                result = self.screen_sequence(seq)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total, result)
                    
            except Exception as e:
                logger.error(f"Failed to screen sequence {i}: {e}")
                # Create a failed result
                results.append(AggregationMetrics(
                    sequence=seq,
                    sequence_length=len(seq),
                    energy_score=0.0,
                    structure_score=0.0,
                    hydrophobic_score=0.0,
                    convergence_score=0.0,
                    compactness_score=0.0,
                    final_energy=1000.0,
                    secondary_structure_pct=0.0,
                    hydrophobic_clustering=float('inf'),
                    convergence_iterations=0,
                    radius_of_gyration=float('inf'),
                    aggregation_score=0.0,
                    risk_level=AggregationRisk.CRITICAL,
                    risk_factors=['screening_failed', str(e)],
                ))
        
        # Sort by score (best/lowest risk first)
        results.sort(key=lambda x: -x.aggregation_score)
        
        return results
    
    def export_csv(
        self, 
        results: List[AggregationMetrics], 
        filepath: str
    ) -> str:
        """
        Export screening results to CSV file.
        
        Args:
            results: List of screening results
            filepath: Output CSV file path
            
        Returns:
            Path to created file
        """
        filepath = Path(filepath)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'rank',
                'sequence',
                'length',
                'aggregation_score',
                'risk_level',
                'energy_score',
                'structure_score',
                'hydrophobic_score',
                'compactness_score',
                'final_energy_kcal_mol',
                'secondary_structure_pct',
                'radius_of_gyration_A',
                'risk_factors',
                'passes_screening',
                'screening_time_ms',
            ])
            
            # Data rows
            for rank, result in enumerate(results, 1):
                writer.writerow([
                    rank,
                    result.sequence,
                    result.sequence_length,
                    f"{result.aggregation_score:.3f}",
                    result.risk_level.value,
                    f"{result.energy_score:.3f}",
                    f"{result.structure_score:.3f}",
                    f"{result.hydrophobic_score:.3f}",
                    f"{result.compactness_score:.3f}",
                    f"{result.final_energy:.2f}",
                    f"{result.secondary_structure_pct:.1f}",
                    f"{result.radius_of_gyration:.2f}",
                    '; '.join(result.risk_factors),
                    'YES' if result.passes_screening else 'NO',
                    f"{result.screening_time_ms:.1f}",
                ])
        
        logger.info(f"Exported {len(results)} results to {filepath}")
        return str(filepath)
    
    def export_json(
        self, 
        results: List[AggregationMetrics], 
        filepath: str
    ) -> str:
        """
        Export screening results to JSON file.
        
        Args:
            results: List of screening results
            filepath: Output JSON file path
            
        Returns:
            Path to created file
        """
        filepath = Path(filepath)
        
        output = {
            'screening_date': datetime.now().isoformat(),
            'config': {
                'iterations': self.config.iterations,
                'agents': self.config.agents,
            },
            'summary': {
                'total_sequences': len(results),
                'passed': sum(1 for r in results if r.passes_screening),
                'failed': sum(1 for r in results if not r.passes_screening),
                'by_risk_level': {
                    'low': sum(1 for r in results if r.risk_level == AggregationRisk.LOW),
                    'moderate': sum(1 for r in results if r.risk_level == AggregationRisk.MODERATE),
                    'high': sum(1 for r in results if r.risk_level == AggregationRisk.HIGH),
                    'critical': sum(1 for r in results if r.risk_level == AggregationRisk.CRITICAL),
                },
            },
            'results': [r.to_dict() for r in results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Exported {len(results)} results to {filepath}")
        return str(filepath)
    
    def print_summary(self, results: List[AggregationMetrics]) -> None:
        """Print a summary of screening results to console."""
        total = len(results)
        passed = sum(1 for r in results if r.passes_screening)
        
        print("\n" + "="*60)
        print("AGGREGATION SCREENING SUMMARY")
        print("="*60)
        print(f"Total sequences screened: {total}")
        print(f"Passed screening: {passed} ({100*passed/total:.1f}%)")
        print(f"Failed screening: {total-passed} ({100*(total-passed)/total:.1f}%)")
        print()
        
        by_risk = {}
        for r in results:
            by_risk[r.risk_level.value] = by_risk.get(r.risk_level.value, 0) + 1
        
        print("By Risk Level:")
        for level in ['low', 'moderate', 'high', 'critical']:
            count = by_risk.get(level, 0)
            bar = '█' * int(20 * count / total) if total > 0 else ''
            print(f"  {level.upper():10} {count:4} {bar}")
        
        print()
        print("Top 5 Best Sequences (lowest aggregation risk):")
        for i, r in enumerate(results[:5], 1):
            print(f"  {i}. Score: {r.aggregation_score:.3f} | {r.sequence[:30]}...")
        
        print()
        print("Bottom 5 Worst Sequences (highest aggregation risk):")
        for i, r in enumerate(results[-5:], 1):
            print(f"  {i}. Score: {r.aggregation_score:.3f} | Risk: {r.risk_level.value}")
            if r.risk_factors:
                print(f"      Factors: {', '.join(r.risk_factors[:3])}")
        
        print("="*60)


# Convenience function for quick screening
def quick_screen(sequence: str) -> AggregationMetrics:
    """
    Quick single-sequence screening with default settings.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        AggregationMetrics result
    """
    screener = AggregationScreener(ScreeningConfig.fast())
    return screener.screen_sequence(sequence)


# Import time for the module
import time
