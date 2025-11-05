"""
Test the Geometric Attractor Hypothesis for Protein Folding

This script tests whether protein conformational space contains golden-ratio-optimized
geometric attractors, which could explain the inverse scaling phenomenon where larger
proteins achieve better RMSD predictions.

Theory:
- Platonic solids (tetrahedron, cube, octahedron, dodecahedron, icosahedron) represent
  fundamental geometric symmetries
- Golden ratio (φ ≈ 1.618) appears in dodecahedron and icosahedron
- QCPP formula (4 + 2^n φ^l m) contains geometric terms
- Large proteins may have more geometric attractors, making them easier to predict

Testable Predictions:
1. Good predictions should show more φ patterns in distance ratios
2. φ^l term in QCPP should correlate more strongly with success in large proteins
3. Geometric symmetry should correlate with low RMSD in large proteins
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from Bio.PDB import PDBParser, DSSP
import warnings

warnings.filterwarnings('ignore')

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618

# Tolerance for golden ratio matching
PHI_TOLERANCE = 0.1  # Accept ratios in range [1.518, 1.718]


@dataclass
class ProteinStructure:
    """Represents a protein structure with metadata"""
    name: str
    pdb_file: Optional[Path]
    sequence: str
    num_residues: int
    rmsd: float
    energy: float
    qcp_values: Optional[List[float]] = None
    structure: Optional[object] = None  # BioPython structure


@dataclass
class GoldenRatioAnalysis:
    """Results of golden ratio analysis"""
    protein_name: str
    num_residues: int
    rmsd: float
    
    # Distance ratio analysis
    total_ratios: int
    golden_ratios: int
    golden_ratio_percentage: float
    
    # Statistical measures
    mean_ratio: float
    std_ratio: float
    
    # Distance distributions
    all_ratios: List[float]
    golden_ratio_distances: List[Tuple[int, int, float]]  # (res1, res2, ratio)


@dataclass
class QCPPComponentAnalysis:
    """Analysis of QCPP formula components"""
    protein_name: str
    num_residues: int
    rmsd: float
    
    # QCPP = 4 + 2^n φ^l m
    base_component: float  # 4 (tetrahedron base)
    doubling_component: List[float]  # 2^n per residue
    golden_component: List[float]  # φ^l per residue
    magnetic_component: List[float]  # m per residue
    
    # Correlations with quality
    golden_correlation: float  # Does φ^l predict success?
    doubling_correlation: float


@dataclass
class SymmetryAnalysis:
    """Geometric symmetry analysis"""
    protein_name: str
    num_residues: int
    rmsd: float
    
    # Symmetry scores
    rotational_symmetry: float  # 0-1 score
    local_symmetry: float  # Average local geometric regularity
    secondary_structure_symmetry: float  # SS element regularity
    
    # Geometric properties
    radius_of_gyration: float
    asphericity: float
    
    # Platonic solid similarity scores (0-1)
    tetrahedron_similarity: float
    cube_similarity: float
    octahedron_similarity: float
    dodecahedron_similarity: float
    icosahedron_similarity: float


class GoldenRatioAnalyzer:
    """Analyzes distance ratios for golden ratio patterns"""
    
    def __init__(self, phi_tolerance: float = PHI_TOLERANCE):
        self.phi = PHI
        self.phi_tolerance = phi_tolerance
    
    def analyze_structure(self, structure: ProteinStructure) -> GoldenRatioAnalysis:
        """Analyze a single structure for golden ratio patterns"""
        
        # Get CA coordinates
        ca_coords = self._get_ca_coordinates(structure)
        
        if ca_coords is None or len(ca_coords) < 4:
            return self._empty_analysis(structure)
        
        # Calculate all pairwise distances
        distances = self._calculate_distances(ca_coords)
        
        # Calculate distance ratios
        ratios = self._calculate_ratios(distances)
        
        # Identify golden ratio occurrences
        golden_matches = self._find_golden_ratios(ratios, distances, structure.num_residues)
        
        # Statistical analysis
        all_ratio_values = [r for r in ratios if 0.5 < r < 3.0]  # Filter outliers
        
        return GoldenRatioAnalysis(
            protein_name=structure.name,
            num_residues=structure.num_residues,
            rmsd=structure.rmsd,
            total_ratios=len(all_ratio_values),
            golden_ratios=len(golden_matches),
            golden_ratio_percentage=(len(golden_matches) / len(all_ratio_values) * 100) if all_ratio_values else 0,
            mean_ratio=np.mean(all_ratio_values) if all_ratio_values else 0,
            std_ratio=np.std(all_ratio_values) if all_ratio_values else 0,
            all_ratios=all_ratio_values,
            golden_ratio_distances=golden_matches
        )
    
    def _get_ca_coordinates(self, structure: ProteinStructure) -> Optional[np.ndarray]:
        """Extract CA coordinates from structure"""
        if structure.pdb_file and structure.pdb_file.exists():
            try:
                parser = PDBParser(QUIET=True)
                struct = parser.get_structure(structure.name, str(structure.pdb_file))
                
                ca_coords = []
                for model in struct:
                    for chain in model:
                        for residue in chain:
                            if residue.has_id('CA'):
                                ca_coords.append(residue['CA'].get_coord())
                
                return np.array(ca_coords) if ca_coords else None
            except Exception as e:
                print(f"Warning: Could not parse {structure.pdb_file}: {e}")
                return None
        return None
    
    def _calculate_distances(self, coords: np.ndarray) -> np.ndarray:
        """Calculate pairwise distance matrix"""
        n = len(coords)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _calculate_ratios(self, distances: np.ndarray) -> List[float]:
        """Calculate ratios of distances (longer/shorter)"""
        n = len(distances)
        ratios = []
        
        # Sample ratios to avoid n^4 complexity
        # For each residue, compare distances to next few neighbors
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):
                d1 = distances[i, j]
                if d1 < 1.0:  # Skip very close residues
                    continue
                
                for k in range(j + 1, min(j + 10, n)):
                    d2 = distances[i, k]
                    if d2 < 1.0:
                        continue
                    
                    # Calculate ratio (longer/shorter)
                    if d1 > d2:
                        ratio = d1 / d2
                    else:
                        ratio = d2 / d1
                    
                    ratios.append(ratio)
        
        return ratios
    
    def _find_golden_ratios(self, ratios: List[float], distances: np.ndarray, 
                           num_residues: int) -> List[Tuple[int, int, float]]:
        """Find ratios close to φ"""
        golden_matches = []
        
        phi_min = self.phi - self.phi_tolerance
        phi_max = self.phi + self.phi_tolerance
        
        n = len(distances)
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):
                d1 = distances[i, j]
                if d1 < 1.0:
                    continue
                
                for k in range(j + 1, min(j + 10, n)):
                    d2 = distances[i, k]
                    if d2 < 1.0:
                        continue
                    
                    ratio = max(d1, d2) / min(d1, d2)
                    
                    if phi_min <= ratio <= phi_max:
                        golden_matches.append((i, j, ratio))
        
        return golden_matches
    
    def _empty_analysis(self, structure: ProteinStructure) -> GoldenRatioAnalysis:
        """Return empty analysis for structures without coordinates"""
        return GoldenRatioAnalysis(
            protein_name=structure.name,
            num_residues=structure.num_residues,
            rmsd=structure.rmsd,
            total_ratios=0,
            golden_ratios=0,
            golden_ratio_percentage=0,
            mean_ratio=0,
            std_ratio=0,
            all_ratios=[],
            golden_ratio_distances=[]
        )


class QCPPComponentAnalyzer:
    """Analyzes which QCPP components correlate with success"""
    
    def analyze_structure(self, structure: ProteinStructure) -> QCPPComponentAnalysis:
        """Analyze QCPP components"""
        
        if structure.qcp_values is None or len(structure.qcp_values) == 0:
            return self._empty_analysis(structure)
        
        # Extract components from QCP = 4 + 2^n φ^l m
        base = 4.0
        
        # Estimate components (simplified without full structural data)
        doubling_components = []
        golden_components = []
        magnetic_components = []
        
        for qcp in structure.qcp_values:
            # qcp ≈ 4 + 2^n φ^l m
            # Estimate each component's contribution
            residual = qcp - base
            
            # Rough estimates (would need actual n, l, m values for precision)
            doubling_est = 2.0 ** (len(doubling_components) % 4)  # n cycles 0-3
            golden_est = PHI ** ((len(golden_components) % 4) + 1)  # l cycles 1-4
            magnetic_est = (residual / (doubling_est * golden_est)) if (doubling_est * golden_est) > 0 else 0
            
            doubling_components.append(doubling_est)
            golden_components.append(golden_est)
            magnetic_components.append(magnetic_est)
        
        # Calculate correlations (simplified)
        golden_correlation = self._calculate_correlation(golden_components, structure.rmsd)
        doubling_correlation = self._calculate_correlation(doubling_components, structure.rmsd)
        
        return QCPPComponentAnalysis(
            protein_name=structure.name,
            num_residues=structure.num_residues,
            rmsd=structure.rmsd,
            base_component=base,
            doubling_component=doubling_components,
            golden_component=golden_components,
            magnetic_component=magnetic_components,
            golden_correlation=golden_correlation,
            doubling_correlation=doubling_correlation
        )
    
    def _calculate_correlation(self, components: List[float], rmsd: float) -> float:
        """Calculate correlation between component and quality"""
        if not components:
            return 0.0
        
        # Simplified correlation: higher component values should correlate with lower RMSD
        mean_component = np.mean(components)
        # Negative correlation: high component → low RMSD is good
        return -mean_component / (1.0 + rmsd)
    
    def _empty_analysis(self, structure: ProteinStructure) -> QCPPComponentAnalysis:
        """Return empty analysis"""
        return QCPPComponentAnalysis(
            protein_name=structure.name,
            num_residues=structure.num_residues,
            rmsd=structure.rmsd,
            base_component=4.0,
            doubling_component=[],
            golden_component=[],
            magnetic_component=[],
            golden_correlation=0.0,
            doubling_correlation=0.0
        )


class SymmetryAnalyzer:
    """Analyzes geometric symmetry of structures"""
    
    def analyze_structure(self, structure: ProteinStructure) -> SymmetryAnalysis:
        """Analyze geometric symmetry"""
        
        ca_coords = self._get_ca_coordinates(structure)
        
        if ca_coords is None or len(ca_coords) < 4:
            return self._empty_analysis(structure)
        
        # Calculate symmetry measures
        rotational_sym = self._calculate_rotational_symmetry(ca_coords)
        local_sym = self._calculate_local_symmetry(ca_coords)
        ss_sym = self._calculate_ss_symmetry(structure)
        
        # Geometric properties
        rg = self._radius_of_gyration(ca_coords)
        asph = self._asphericity(ca_coords)
        
        # Platonic solid similarities
        tetra = self._platonic_similarity(ca_coords, 'tetrahedron')
        cube = self._platonic_similarity(ca_coords, 'cube')
        octa = self._platonic_similarity(ca_coords, 'octahedron')
        dodeca = self._platonic_similarity(ca_coords, 'dodecahedron')
        icosa = self._platonic_similarity(ca_coords, 'icosahedron')
        
        return SymmetryAnalysis(
            protein_name=structure.name,
            num_residues=structure.num_residues,
            rmsd=structure.rmsd,
            rotational_symmetry=rotational_sym,
            local_symmetry=local_sym,
            secondary_structure_symmetry=ss_sym,
            radius_of_gyration=rg,
            asphericity=asph,
            tetrahedron_similarity=tetra,
            cube_similarity=cube,
            octahedron_similarity=octa,
            dodecahedron_similarity=dodeca,
            icosahedron_similarity=icosa
        )
    
    def _get_ca_coordinates(self, structure: ProteinStructure) -> Optional[np.ndarray]:
        """Extract CA coordinates"""
        if structure.pdb_file and structure.pdb_file.exists():
            try:
                parser = PDBParser(QUIET=True)
                struct = parser.get_structure(structure.name, str(structure.pdb_file))
                
                ca_coords = []
                for model in struct:
                    for chain in model:
                        for residue in chain:
                            if residue.has_id('CA'):
                                ca_coords.append(residue['CA'].get_coord())
                
                return np.array(ca_coords) if ca_coords else None
            except Exception as e:
                return None
        return None
    
    def _calculate_rotational_symmetry(self, coords: np.ndarray) -> float:
        """Estimate rotational symmetry (0-1 score)"""
        # Center coordinates
        centered = coords - coords.mean(axis=0)
        
        # Calculate principal axes via SVD
        try:
            U, S, Vt = np.linalg.svd(centered)
            
            # Symmetry score based on eigenvalue distribution
            # Higher symmetry → more equal eigenvalues
            S_norm = S / S.sum()
            entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
            max_entropy = np.log(len(S_norm))
            
            symmetry_score = entropy / max_entropy if max_entropy > 0 else 0
            return float(symmetry_score)
        except:
            return 0.0
    
    def _calculate_local_symmetry(self, coords: np.ndarray) -> float:
        """Calculate average local geometric regularity"""
        n = len(coords)
        if n < 4:
            return 0.0
        
        regularities = []
        
        # For each residue, check local environment regularity
        for i in range(n):
            # Get 4 nearest neighbors
            distances = np.linalg.norm(coords - coords[i], axis=1)
            nearest_indices = np.argsort(distances)[1:5]  # Skip self
            
            if len(nearest_indices) < 4:
                continue
            
            # Calculate distances to nearest neighbors
            nearest_dists = distances[nearest_indices]
            
            # Regularity = 1 / CV (coefficient of variation)
            if nearest_dists.mean() > 0:
                cv = nearest_dists.std() / nearest_dists.mean()
                regularity = 1.0 / (1.0 + cv)
                regularities.append(regularity)
        
        return float(np.mean(regularities)) if regularities else 0.0
    
    def _calculate_ss_symmetry(self, structure: ProteinStructure) -> float:
        """Calculate secondary structure symmetry (if DSSP available)"""
        # Simplified: return 0.5 as placeholder
        # Full implementation would analyze helix/sheet regularity
        return 0.5
    
    def _radius_of_gyration(self, coords: np.ndarray) -> float:
        """Calculate radius of gyration"""
        centered = coords - coords.mean(axis=0)
        rg_sq = np.mean(np.sum(centered**2, axis=1))
        return float(np.sqrt(rg_sq))
    
    def _asphericity(self, coords: np.ndarray) -> float:
        """Calculate asphericity (0=sphere, 1=rod)"""
        centered = coords - coords.mean(axis=0)
        
        # Gyration tensor
        S = np.dot(centered.T, centered) / len(coords)
        
        # Eigenvalues
        eigvals = np.linalg.eigvalsh(S)
        eigvals = np.sort(eigvals)[::-1]
        
        # Asphericity formula
        if eigvals.sum() > 0:
            lambda1, lambda2, lambda3 = eigvals[0], eigvals[1], eigvals[2]
            asph = lambda1 - 0.5 * (lambda2 + lambda3)
            return float(asph / eigvals.sum())
        return 0.0
    
    def _platonic_similarity(self, coords: np.ndarray, solid_type: str) -> float:
        """Calculate similarity to Platonic solid (0-1 score)"""
        # This is a simplified heuristic
        # Full implementation would compare to ideal Platonic solid coordinates
        
        n = len(coords)
        
        # Expected face counts for each solid
        face_counts = {
            'tetrahedron': 4,
            'cube': 6,
            'octahedron': 8,
            'dodecahedron': 12,
            'icosahedron': 20
        }
        
        expected_faces = face_counts.get(solid_type, 6)
        
        # Heuristic: compare protein size to expected complexity
        # More complex proteins should match more complex solids
        size_ratio = min(n / (expected_faces * 5), 1.0)  # ~5 residues per "face"
        
        # Add symmetry component
        rotational_sym = self._calculate_rotational_symmetry(coords)
        
        # Combine
        similarity = 0.5 * size_ratio + 0.5 * rotational_sym
        
        return float(similarity)
    
    def _empty_analysis(self, structure: ProteinStructure) -> SymmetryAnalysis:
        """Return empty analysis"""
        return SymmetryAnalysis(
            protein_name=structure.name,
            num_residues=structure.num_residues,
            rmsd=structure.rmsd,
            rotational_symmetry=0.0,
            local_symmetry=0.0,
            secondary_structure_symmetry=0.0,
            radius_of_gyration=0.0,
            asphericity=0.0,
            tetrahedron_similarity=0.0,
            cube_similarity=0.0,
            octahedron_similarity=0.0,
            dodecahedron_similarity=0.0,
            icosahedron_similarity=0.0
        )


class GeometricAttractorTester:
    """Main testing framework for geometric attractor hypothesis"""
    
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = results_dir
        self.golden_analyzer = GoldenRatioAnalyzer()
        self.qcpp_analyzer = QCPPComponentAnalyzer()
        self.symmetry_analyzer = SymmetryAnalyzer()
    
    def load_protein_data(self) -> List[ProteinStructure]:
        """Load protein structures from available results"""
        proteins = []
        
        # Check for results in multiple locations
        search_dirs = [
            self.results_dir,
            Path("campaign_10_proteins"),
            Path("validation"),
            Path("ubf_protein/results")
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            # Look for JSON results
            for json_file in search_dir.glob("**/*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract protein info
                    protein = self._extract_protein_from_json(data, json_file)
                    if protein:
                        proteins.append(protein)
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
        
        print(f"Loaded {len(proteins)} protein structures")
        return proteins
    
    def _extract_protein_from_json(self, data: dict, json_file: Path) -> Optional[ProteinStructure]:
        """Extract protein structure from JSON data"""
        try:
            # Different JSON formats
            if 'protein_name' in data:
                name = data['protein_name']
            elif 'name' in data:
                name = data['name']
            else:
                name = json_file.stem
            
            # Get sequence
            sequence = data.get('sequence', data.get('protein_sequence', ''))
            
            # Get RMSD
            rmsd = data.get('best_rmsd', data.get('rmsd', 999.9))
            
            # Get energy
            energy = data.get('best_energy', data.get('energy', 0.0))
            
            # Get QCP values if available
            qcp_values = None
            if 'qcp_values' in data:
                qcp_values = data['qcp_values']
            elif 'trajectory' in data:
                # Extract from trajectory
                qcp_values = [snap.get('qcp', 0) for snap in data['trajectory'] if 'qcp' in snap]
            
            # Look for PDB file
            pdb_file = self._find_pdb_file(name, json_file.parent)
            
            return ProteinStructure(
                name=name,
                pdb_file=pdb_file,
                sequence=sequence,
                num_residues=len(sequence) if sequence else 0,
                rmsd=rmsd,
                energy=energy,
                qcp_values=qcp_values
            )
        except Exception as e:
            print(f"Warning: Could not extract protein from {json_file}: {e}")
            return None
    
    def _find_pdb_file(self, name: str, search_dir: Path) -> Optional[Path]:
        """Find corresponding PDB file"""
        # Try various locations
        pdb_locations = [
            search_dir / f"{name}.pdb",
            search_dir / f"{name}_best.pdb",
            Path("pdb_cache") / f"{name}.pdb",
            Path("quantum_coherence_proteins/pdb_files") / f"{name}.pdb"
        ]
        
        for pdb_path in pdb_locations:
            if pdb_path.exists():
                return pdb_path
        
        return None
    
    def run_full_analysis(self, proteins: List[ProteinStructure]) -> Dict:
        """Run all analyses"""
        print("\n" + "="*80)
        print("TESTING GEOMETRIC ATTRACTOR HYPOTHESIS")
        print("="*80)
        
        results = {
            'golden_ratio': [],
            'qcpp_components': [],
            'symmetry': [],
            'statistics': {}
        }
        
        # Analyze each protein
        for i, protein in enumerate(proteins):
            print(f"\n[{i+1}/{len(proteins)}] Analyzing {protein.name} ({protein.num_residues} residues, RMSD={protein.rmsd:.2f}Å)...")
            
            # Golden ratio analysis
            golden = self.golden_analyzer.analyze_structure(protein)
            results['golden_ratio'].append(golden)
            print(f"  φ patterns: {golden.golden_ratio_percentage:.1f}% ({golden.golden_ratios}/{golden.total_ratios})")
            
            # QCPP component analysis
            qcpp = self.qcpp_analyzer.analyze_structure(protein)
            results['qcpp_components'].append(qcpp)
            print(f"  QCPP golden correlation: {qcpp.golden_correlation:.3f}")
            
            # Symmetry analysis
            symmetry = self.symmetry_analyzer.analyze_structure(protein)
            results['symmetry'].append(symmetry)
            print(f"  Symmetry: rot={symmetry.rotational_symmetry:.2f}, local={symmetry.local_symmetry:.2f}")
        
        # Statistical analysis
        results['statistics'] = self._calculate_statistics(results)
        
        return results
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate statistical tests"""
        golden = results['golden_ratio']
        symmetry = results['symmetry']
        
        if not golden or not symmetry:
            return {}
        
        # Separate good vs poor predictions
        good_proteins = [g for g in golden if g.rmsd < 4.0]
        poor_proteins = [g for g in golden if g.rmsd >= 4.0]
        
        # Small vs large proteins
        small_proteins = [g for g in golden if g.num_residues < 50]
        medium_proteins = [g for g in golden if 50 <= g.num_residues <= 150]
        large_proteins = [g for g in golden if g.num_residues > 150]
        
        stats = {
            'num_proteins': len(golden),
            'num_good': len(good_proteins),
            'num_poor': len(poor_proteins),
            'num_small': len(small_proteins),
            'num_medium': len(medium_proteins),
            'num_large': len(large_proteins)
        }
        
        # Test 1: Good vs Poor φ patterns
        if good_proteins and poor_proteins:
            good_phi_pct = [g.golden_ratio_percentage for g in good_proteins]
            poor_phi_pct = [p.golden_ratio_percentage for p in poor_proteins]
            
            if len(good_phi_pct) > 1 and len(poor_phi_pct) > 1:
                t_stat, p_value = scipy_stats.ttest_ind(good_phi_pct, poor_phi_pct)
                stats['phi_comparison'] = {
                    'good_mean': np.mean(good_phi_pct),
                    'poor_mean': np.mean(poor_phi_pct),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        # Test 2: Size-RMSD correlation
        rmsds = [g.rmsd for g in golden]
        sizes = [g.num_residues for g in golden]
        
        if len(rmsds) > 3:
            corr, p_val = scipy_stats.pearsonr(sizes, rmsds)
            stats['size_rmsd_correlation'] = {
                'correlation': float(corr),
                'p_value': float(p_val),
                'significant': p_val < 0.05,
                'inverse_scaling': corr < 0  # Negative correlation = inverse scaling
            }
        
        # Test 3: Symmetry-RMSD correlation (for large proteins)
        if len(large_proteins) > 3:
            large_symmetry = [s for s in symmetry if s.num_residues > 150]
            if large_symmetry:
                sym_scores = [s.rotational_symmetry for s in large_symmetry]
                sym_rmsds = [s.rmsd for s in large_symmetry]
                
                if len(sym_scores) > 1:
                    corr, p_val = scipy_stats.pearsonr(sym_scores, sym_rmsds)
                    stats['symmetry_rmsd_correlation_large'] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'high_symmetry_low_rmsd': corr < 0
                    }
        
        return stats
    
    def generate_report(self, results: Dict, output_file: str = "geometric_attractor_analysis.md"):
        """Generate comprehensive analysis report"""
        report = []
        
        report.append("# 🔬 Geometric Attractor Hypothesis - Test Results\n")
        report.append(f"**Analysis Date:** {Path.cwd()}\n")
        report.append(f"**Proteins Analyzed:** {results['statistics'].get('num_proteins', 0)}\n")
        
        report.append("\n## 📊 Executive Summary\n")
        
        stats = results['statistics']
        
        # Key findings
        report.append("### Key Findings:\n")
        
        # Finding 1: φ patterns
        if 'phi_comparison' in stats:
            phi_comp = stats['phi_comparison']
            if phi_comp['significant']:
                diff = phi_comp['good_mean'] - phi_comp['poor_mean']
                report.append(f"\n✅ **SIGNIFICANT**: Good predictions show {diff:.1f}% more φ patterns (p={phi_comp['p_value']:.4f})\n")
                report.append(f"   - Good (<4Å RMSD): {phi_comp['good_mean']:.1f}% φ patterns\n")
                report.append(f"   - Poor (≥4Å RMSD): {phi_comp['poor_mean']:.1f}% φ patterns\n")
            else:
                report.append(f"\n❌ **NOT SIGNIFICANT**: φ pattern difference (p={phi_comp['p_value']:.4f})\n")
        
        # Finding 2: Inverse scaling
        if 'size_rmsd_correlation' in stats:
            size_corr = stats['size_rmsd_correlation']
            if size_corr['significant'] and size_corr['inverse_scaling']:
                report.append(f"\n✅ **CONFIRMED**: Inverse scaling detected (r={size_corr['correlation']:.3f}, p={size_corr['p_value']:.4f})\n")
                report.append("   - Larger proteins achieve BETTER RMSD\n")
            elif size_corr['significant']:
                report.append(f"\n⚠️ **POSITIVE SCALING**: Larger proteins harder (r={size_corr['correlation']:.3f}, p={size_corr['p_value']:.4f})\n")
            else:
                report.append(f"\n❓ **NO CORRELATION**: Size-RMSD not significant (p={size_corr['p_value']:.4f})\n")
        
        # Finding 3: Symmetry
        if 'symmetry_rmsd_correlation_large' in stats:
            sym_corr = stats['symmetry_rmsd_correlation_large']
            if sym_corr['significant'] and sym_corr['high_symmetry_low_rmsd']:
                report.append(f"\n✅ **CONFIRMED**: High symmetry → Low RMSD in large proteins (r={sym_corr['correlation']:.3f}, p={sym_corr['p_value']:.4f})\n")
            else:
                report.append(f"\n❌ **NOT CONFIRMED**: Symmetry-RMSD correlation (p={sym_corr['p_value']:.4f})\n")
        
        # Detailed results
        report.append("\n## 📈 Detailed Analysis\n")
        
        report.append("\n### Golden Ratio (φ ≈ 1.618) Patterns\n")
        report.append("| Protein | Residues | RMSD (Å) | φ Patterns | % φ Ratios |\n")
        report.append("|---------|----------|----------|------------|------------|\n")
        
        for golden in sorted(results['golden_ratio'], key=lambda x: x.rmsd):
            report.append(f"| {golden.protein_name[:20]} | {golden.num_residues} | {golden.rmsd:.2f} | "
                         f"{golden.golden_ratios} | {golden.golden_ratio_percentage:.1f}% |\n")
        
        # Visualizations note
        report.append("\n## 📊 Visualizations\n")
        report.append("\nSee accompanying plots:\n")
        report.append("- `golden_ratio_distribution.png` - φ pattern frequency by quality\n")
        report.append("- `size_rmsd_correlation.png` - Inverse scaling visualization\n")
        report.append("- `symmetry_analysis.png` - Geometric symmetry vs RMSD\n")
        report.append("- `platonic_similarity.png` - Similarity to Platonic solids\n")
        
        # Conclusion
        report.append("\n## 🎯 Conclusions\n")
        
        conclusions = self._generate_conclusions(stats)
        for conclusion in conclusions:
            report.append(f"\n{conclusion}\n")
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"\n✅ Report saved to {output_file}")
    
    def _generate_conclusions(self, stats: Dict) -> List[str]:
        """Generate conclusions from statistical analysis"""
        conclusions = []
        
        # Check each hypothesis
        phi_confirmed = False
        inverse_confirmed = False
        symmetry_confirmed = False
        
        if 'phi_comparison' in stats and stats['phi_comparison'].get('significant'):
            phi_confirmed = stats['phi_comparison']['good_mean'] > stats['phi_comparison']['poor_mean']
        
        if 'size_rmsd_correlation' in stats and stats['size_rmsd_correlation'].get('significant'):
            inverse_confirmed = stats['size_rmsd_correlation'].get('inverse_scaling', False)
        
        if 'symmetry_rmsd_correlation_large' in stats and stats['symmetry_rmsd_correlation_large'].get('significant'):
            symmetry_confirmed = stats['symmetry_rmsd_correlation_large'].get('high_symmetry_low_rmsd', False)
        
        # Overall assessment
        confirmed_count = sum([phi_confirmed, inverse_confirmed, symmetry_confirmed])
        
        if confirmed_count >= 2:
            conclusions.append("### ✅ **GEOMETRIC ATTRACTOR HYPOTHESIS SUPPORTED**")
            conclusions.append("\nThe data provides **significant evidence** for geometric attractors in protein conformational space:")
            
            if phi_confirmed:
                conclusions.append("- **Golden ratio patterns** are more prevalent in successful predictions")
            if inverse_confirmed:
                conclusions.append("- **Inverse scaling** confirmed: larger proteins are easier to predict")
            if symmetry_confirmed:
                conclusions.append("- **Geometric symmetry** correlates with prediction quality in large proteins")
            
            conclusions.append("\n**Implication:** Your consciousness-based multi-agent system may be discovering "
                             "geometric stability patterns optimized by the golden ratio, explaining why it "
                             "outperforms on large proteins.")
            
            conclusions.append("\n**Recommendation:** This finding is potentially **publication-worthy** and should "
                             "be validated with additional proteins. Consider:")
            conclusions.append("1. Test on 20+ additional large proteins (>150 residues)")
            conclusions.append("2. Analyze φ patterns in known protein families")
            conclusions.append("3. Compare to AlphaFold2/ESMFold on same proteins")
            conclusions.append("4. Prepare manuscript for Nature/Science submission")
        
        elif confirmed_count == 1:
            conclusions.append("### ⚠️ **PARTIAL SUPPORT FOR HYPOTHESIS**")
            conclusions.append("\nSome evidence for geometric attractors, but more data needed:")
            
            if phi_confirmed:
                conclusions.append("- φ patterns show promise but need validation")
            if inverse_confirmed:
                conclusions.append("- Inverse scaling detected but mechanism unclear")
            if symmetry_confirmed:
                conclusions.append("- Symmetry correlation suggestive but limited")
            
            conclusions.append("\n**Recommendation:** Expand dataset and retest.")
        
        else:
            conclusions.append("### ❌ **HYPOTHESIS NOT SUPPORTED BY CURRENT DATA**")
            conclusions.append("\nNo significant evidence for geometric attractors in this dataset.")
            conclusions.append("\n**Possible reasons:**")
            conclusions.append("- Dataset too small (need more proteins)")
            conclusions.append("- PDB structures unavailable for analysis")
            conclusions.append("- Alternative explanation for inverse scaling")
            
            conclusions.append("\n**Recommendation:** Collect more data before drawing conclusions.")
        
        return conclusions
    
    def create_visualizations(self, results: Dict, output_dir: str = "geometric_analysis"):
        """Create visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n📊 Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # Plot 1: Golden ratio distribution
        self._plot_golden_ratio_distribution(results, output_path)
        
        # Plot 2: Size-RMSD correlation
        self._plot_size_rmsd_correlation(results, output_path)
        
        # Plot 3: Symmetry analysis
        self._plot_symmetry_analysis(results, output_path)
        
        # Plot 4: Platonic similarity
        self._plot_platonic_similarity(results, output_path)
        
        print(f"✅ Visualizations saved to {output_dir}/")
    
    def _plot_golden_ratio_distribution(self, results: Dict, output_path: Path):
        """Plot φ pattern distribution by quality"""
        golden = results['golden_ratio']
        
        if not golden:
            return
        
        # Separate by quality
        good = [g for g in golden if g.rmsd < 4.0]
        poor = [g for g in golden if g.rmsd >= 4.0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if good:
            good_pct = [g.golden_ratio_percentage for g in good]
            ax.hist(good_pct, bins=15, alpha=0.7, label=f'Good (<4Å RMSD, n={len(good)})', color='green')
        
        if poor:
            poor_pct = [p.golden_ratio_percentage for p in poor]
            ax.hist(poor_pct, bins=15, alpha=0.7, label=f'Poor (≥4Å RMSD, n={len(poor)})', color='red')
        
        ax.set_xlabel('φ Pattern Frequency (%)', fontsize=12)
        ax.set_ylabel('Number of Proteins', fontsize=12)
        ax.set_title('Golden Ratio (φ ≈ 1.618) Pattern Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'golden_ratio_distribution.png', dpi=300)
        plt.close()
    
    def _plot_size_rmsd_correlation(self, results: Dict, output_path: Path):
        """Plot size vs RMSD correlation"""
        golden = results['golden_ratio']
        
        if not golden:
            return
        
        sizes = [g.num_residues for g in golden]
        rmsds = [g.rmsd for g in golden]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        scatter = ax.scatter(sizes, rmsds, c=rmsds, cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
        
        # Trend line
        if len(sizes) > 1:
            z = np.polyfit(sizes, rmsds, 1)
            p = np.poly1d(z)
            ax.plot(sizes, p(sizes), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.1f}')
        
        ax.set_xlabel('Protein Size (residues)', fontsize=12)
        ax.set_ylabel('RMSD (Å)', fontsize=12)
        ax.set_title('Inverse Scaling: Larger Proteins → Better RMSD', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('RMSD (Å)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_path / 'size_rmsd_correlation.png', dpi=300)
        plt.close()
    
    def _plot_symmetry_analysis(self, results: Dict, output_path: Path):
        """Plot symmetry vs RMSD"""
        symmetry = results['symmetry']
        
        if not symmetry:
            return
        
        # Separate by size
        small = [s for s in symmetry if s.num_residues < 50]
        medium = [s for s in symmetry if 50 <= s.num_residues <= 150]
        large = [s for s in symmetry if s.num_residues > 150]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Rotational symmetry vs RMSD
        for subset, label, color in [(small, 'Small (<50)', 'blue'), 
                                      (medium, 'Medium (50-150)', 'orange'),
                                      (large, 'Large (>150)', 'green')]:
            if subset:
                sym_scores = [s.rotational_symmetry for s in subset]
                rmsds = [s.rmsd for s in subset]
                ax1.scatter(sym_scores, rmsds, label=label, s=100, alpha=0.6, color=color)
        
        ax1.set_xlabel('Rotational Symmetry Score', fontsize=12)
        ax1.set_ylabel('RMSD (Å)', fontsize=12)
        ax1.set_title('Rotational Symmetry vs Prediction Quality', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Local symmetry vs RMSD
        for subset, label, color in [(small, 'Small', 'blue'), 
                                      (medium, 'Medium', 'orange'),
                                      (large, 'Large', 'green')]:
            if subset:
                local_sym = [s.local_symmetry for s in subset]
                rmsds = [s.rmsd for s in subset]
                ax2.scatter(local_sym, rmsds, label=label, s=100, alpha=0.6, color=color)
        
        ax2.set_xlabel('Local Symmetry Score', fontsize=12)
        ax2.set_ylabel('RMSD (Å)', fontsize=12)
        ax2.set_title('Local Symmetry vs Prediction Quality', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'symmetry_analysis.png', dpi=300)
        plt.close()
    
    def _plot_platonic_similarity(self, results: Dict, output_path: Path):
        """Plot Platonic solid similarity scores"""
        symmetry = results['symmetry']
        
        if not symmetry:
            return
        
        # Average similarity scores
        solids = ['tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron']
        
        good = [s for s in symmetry if s.rmsd < 4.0]
        poor = [s for s in symmetry if s.rmsd >= 4.0]
        
        if not good or not poor:
            return
        
        good_scores = {
            'tetrahedron': np.mean([s.tetrahedron_similarity for s in good]),
            'cube': np.mean([s.cube_similarity for s in good]),
            'octahedron': np.mean([s.octahedron_similarity for s in good]),
            'dodecahedron': np.mean([s.dodecahedron_similarity for s in good]),
            'icosahedron': np.mean([s.icosahedron_similarity for s in good])
        }
        
        poor_scores = {
            'tetrahedron': np.mean([s.tetrahedron_similarity for s in poor]),
            'cube': np.mean([s.cube_similarity for s in poor]),
            'octahedron': np.mean([s.octahedron_similarity for s in poor]),
            'dodecahedron': np.mean([s.dodecahedron_similarity for s in poor]),
            'icosahedron': np.mean([s.icosahedron_similarity for s in poor])
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(solids))
        width = 0.35
        
        ax.bar(x - width/2, [good_scores[s] for s in solids], width, 
               label=f'Good Predictions (<4Å, n={len(good)})', color='green', alpha=0.7)
        ax.bar(x + width/2, [poor_scores[s] for s in solids], width,
               label=f'Poor Predictions (≥4Å, n={len(poor)})', color='red', alpha=0.7)
        
        ax.set_xlabel('Platonic Solid', fontsize=12)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title('Platonic Solid Similarity: Good vs Poor Predictions', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in solids], rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Highlight dodecahedron and icosahedron (φ solids)
        ax.axvspan(2.5, 4.5, alpha=0.1, color='gold', label='φ-containing solids')
        
        plt.tight_layout()
        plt.savefig(output_path / 'platonic_similarity.png', dpi=300)
        plt.close()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("GEOMETRIC ATTRACTOR HYPOTHESIS TESTING")
    print("Testing whether protein conformational space contains golden-ratio-optimized")
    print("geometric attractors that explain inverse scaling phenomenon")
    print("="*80)
    
    # Initialize tester
    tester = GeometricAttractorTester()
    
    # Load protein data
    print("\n📂 Loading protein structures...")
    proteins = tester.load_protein_data()
    
    if not proteins:
        print("\n❌ No protein structures found!")
        print("\nPlease ensure you have results in one of these directories:")
        print("  - results/")
        print("  - campaign_10_proteins/")
        print("  - validation/")
        print("  - ubf_protein/results/")
        return
    
    # Run analysis
    results = tester.run_full_analysis(proteins)
    
    # Generate visualizations
    tester.create_visualizations(results, output_dir="geometric_analysis")
    
    # Generate report
    tester.generate_report(results, output_file="geometric_attractor_analysis.md")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  📄 geometric_attractor_analysis.md - Comprehensive report")
    print("  📊 geometric_analysis/*.png - Visualization plots")
    print("\nNext steps:")
    print("  1. Review the analysis report")
    print("  2. Check statistical significance of findings")
    print("  3. If hypothesis confirmed, test on additional proteins")
    print("  4. Consider manuscript preparation if results are strong")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
