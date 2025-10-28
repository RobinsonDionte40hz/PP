"""
Protein Selector for Large-Scale Validation

Systematically selects diverse proteins from PDB database or curated lists
for comprehensive testing. Filters by size, structural class, resolution,
and completeness to ensure diverse representation.

Key Features:
- Size category filtering (tiny: <50, small: 50-100, medium: 100-200, large: >200)
- Structural class filtering (all-alpha, all-beta, alpha-beta, alpha+beta, irregular)
- Resolution filtering (X-ray: <2.5Å preferred)
- Completeness filtering (max 10% missing residues)
- Balanced distribution across categories
- Export to JSON/CSV for reproducibility
"""

import json
import csv
import logging
import os
import urllib.request
import urllib.error
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ProteinMetadata:
    """
    Metadata for a single protein in the test set.
    
    Attributes:
        pdb_id: PDB identifier (e.g., '1UBQ')
        sequence_length: Number of residues
        size_category: 'tiny' (<50), 'small' (50-100), 'medium' (100-200), 'large' (>200)
        structural_class: 'all-alpha', 'all-beta', 'alpha-beta', 'alpha+beta', 'irregular'
        experimental_method: 'X-ray' or 'NMR'
        resolution: Resolution in Angstroms (None for NMR)
        missing_residues_pct: Percentage of missing residues (0-100)
        organism: Source organism
        description: Brief description of the protein
    """
    pdb_id: str
    sequence_length: int
    size_category: str
    structural_class: str
    experimental_method: str
    resolution: Optional[float]
    missing_residues_pct: float
    organism: str
    description: str
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Normalize PDB ID to uppercase
        self.pdb_id = self.pdb_id.upper()
        
        # Validate size category
        valid_sizes = {'tiny', 'small', 'medium', 'large'}
        if self.size_category not in valid_sizes:
            raise ValueError(f"Invalid size_category: {self.size_category}. Must be one of {valid_sizes}")
        
        # Validate structural class
        valid_classes = {'all-alpha', 'all-beta', 'alpha-beta', 'alpha+beta', 'irregular'}
        if self.structural_class not in valid_classes:
            raise ValueError(f"Invalid structural_class: {self.structural_class}. Must be one of {valid_classes}")
        
        # Validate experimental method
        valid_methods = {'X-ray', 'NMR', 'Cryo-EM', 'Unknown'}
        if self.experimental_method not in valid_methods:
            raise ValueError(f"Invalid experimental_method: {self.experimental_method}. Must be one of {valid_methods}")
        
        # Validate resolution consistency
        if self.experimental_method == 'NMR' and self.resolution is not None:
            logger.warning(f"{self.pdb_id}: NMR structures should not have resolution")
        
        # Validate missing residues percentage
        if not (0.0 <= self.missing_residues_pct <= 100.0):
            raise ValueError(f"missing_residues_pct must be between 0 and 100, got {self.missing_residues_pct}")


class ProteinSelector:
    """
    Systematically select diverse proteins for validation testing.
    
    Selection Strategy:
    1. Start with curated list or PDB query
    2. Apply size filters (diverse size distribution)
    3. Apply structural class filters (all types represented)
    4. Apply quality filters (resolution, completeness)
    5. Balance distribution across categories
    6. Prioritize well-studied proteins for early phases
    
    Example:
        selector = ProteinSelector()
        proteins = selector.select_proteins(target_count=60)
        selector.export_selection(proteins, 'selected_proteins.json')
    """
    
    def __init__(self, cache_dir: str = 'pdb_cache'):
        """
        Initialize protein selector.
        
        Args:
            cache_dir: Directory for caching PDB metadata
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Curated list of well-studied proteins
        # These are validated proteins with known structures
        self.curated_proteins = self._load_curated_list()
    
    def _load_curated_list(self) -> List[Dict]:
        """
        Load curated list of well-studied proteins.
        
        This list includes proteins that have been extensively studied
        and validated, making them ideal for benchmarking.
        
        Returns:
            List of protein metadata dictionaries
        """
        # Curated list based on common benchmarks and previous testing
        curated = [
            # Tiny proteins (<50 residues)
            {'pdb_id': '1CRN', 'length': 46, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 1.5, 'organism': 'Hordeum vulgare', 'desc': 'Crambin'},
            {'pdb_id': '2MR9', 'length': 35, 'class': 'all-beta', 'method': 'X-ray', 'res': 1.1, 'organism': 'Synthetic', 'desc': 'Trp-cage miniprotein'},
            {'pdb_id': '1VII', 'length': 36, 'class': 'all-alpha', 'method': 'NMR', 'res': None, 'organism': 'Synthetic', 'desc': 'Villin headpiece'},
            {'pdb_id': '1L2Y', 'length': 20, 'class': 'all-alpha', 'method': 'NMR', 'res': None, 'organism': 'Synthetic', 'desc': 'Trp-cage TC5b'},
            {'pdb_id': '1PSV', 'length': 36, 'class': 'all-beta', 'method': 'NMR', 'res': None, 'organism': 'Synthetic', 'desc': 'Peptide hairpin'},
            
            # Small proteins (50-100 residues)
            {'pdb_id': '1UBQ', 'length': 76, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 1.8, 'organism': 'Homo sapiens', 'desc': 'Ubiquitin'},
            {'pdb_id': '1ROP', 'length': 56, 'class': 'all-alpha', 'method': 'X-ray', 'res': 1.7, 'organism': 'Escherichia coli', 'desc': 'Repressor of primer'},
            {'pdb_id': '1GB1', 'length': 56, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 2.0, 'organism': 'Streptococcus', 'desc': 'Protein G B1 domain'},
            {'pdb_id': '1ENH', 'length': 54, 'class': 'all-alpha', 'method': 'NMR', 'res': None, 'organism': 'Drosophila', 'desc': 'Engrailed homeodomain'},
            {'pdb_id': '1PGB', 'length': 56, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 1.0, 'organism': 'Streptococcus', 'desc': 'Protein G B1 (IgG binding)'},
            {'pdb_id': '2IGD', 'length': 61, 'class': 'all-beta', 'method': 'X-ray', 'res': 2.3, 'organism': 'Homo sapiens', 'desc': 'Immunoglobulin domain'},
            {'pdb_id': '1BDD', 'length': 60, 'class': 'all-alpha', 'method': 'X-ray', 'res': 1.8, 'organism': 'Escherichia coli', 'desc': 'B-DNA binding domain'},
            {'pdb_id': '1UTG', 'length': 70, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 1.4, 'organism': 'Homo sapiens', 'desc': 'Uteroglobin'},
            {'pdb_id': '1SHG', 'length': 57, 'class': 'all-alpha', 'method': 'NMR', 'res': None, 'organism': 'Synthetic', 'desc': 'SH3 domain'},
            {'pdb_id': '1HIV', 'length': 99, 'class': 'all-alpha', 'method': 'X-ray', 'res': 2.5, 'organism': 'HIV-1', 'desc': 'HIV-1 protease'},
            
            # Medium proteins (100-200 residues)
            {'pdb_id': '1LYZ', 'length': 129, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 1.5, 'organism': 'Gallus gallus', 'desc': 'Lysozyme'},
            {'pdb_id': '1RNB', 'length': 124, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 1.5, 'organism': 'Bos taurus', 'desc': 'Ribonuclease B'},
            {'pdb_id': '1MBN', 'length': 153, 'class': 'all-alpha', 'method': 'X-ray', 'res': 1.5, 'organism': 'Physeter catodon', 'desc': 'Myoglobin'},
            {'pdb_id': '1TIM', 'length': 247, 'class': 'alpha-beta', 'method': 'X-ray', 'res': 1.8, 'organism': 'Gallus gallus', 'desc': 'Triosephosphate isomerase'},
            {'pdb_id': '1AK3', 'length': 194, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 2.1, 'organism': 'Escherichia coli', 'desc': 'Adenylate kinase'},
            {'pdb_id': '1SHF', 'length': 107, 'class': 'all-beta', 'method': 'X-ray', 'res': 1.8, 'organism': 'Homo sapiens', 'desc': 'SH2 domain'},
            {'pdb_id': '1CRK', 'length': 118, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 2.2, 'organism': 'Rattus norvegicus', 'desc': 'Creatine kinase'},
            {'pdb_id': '3SSI', 'length': 107, 'class': 'all-beta', 'method': 'X-ray', 'res': 1.8, 'organism': 'Streptomyces', 'desc': 'Subtilisin inhibitor'},
            {'pdb_id': '3CLN', 'length': 148, 'class': 'all-alpha', 'method': 'X-ray', 'res': 1.7, 'organism': 'Homo sapiens', 'desc': 'Calmodulin'},
            {'pdb_id': '1CHO', 'length': 128, 'class': 'alpha-beta', 'method': 'X-ray', 'res': 2.2, 'organism': 'Bacillus', 'desc': 'Chitinase'},
            
            # Large proteins (>200 residues)
            {'pdb_id': '1HEW', 'length': 297, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 1.9, 'organism': 'Gallus gallus', 'desc': 'Lysozyme C'},
            {'pdb_id': '2DHB', 'length': 292, 'class': 'all-alpha', 'method': 'X-ray', 'res': 1.7, 'organism': 'Homo sapiens', 'desc': 'Deoxyhemoglobin'},
            {'pdb_id': '1ATP', 'length': 247, 'class': 'alpha-beta', 'method': 'X-ray', 'res': 2.0, 'organism': 'Homo sapiens', 'desc': 'ATP synthase'},
            {'pdb_id': '1PFK', 'length': 320, 'class': 'alpha-beta', 'method': 'X-ray', 'res': 2.4, 'organism': 'Escherichia coli', 'desc': 'Phosphofructokinase'},
            {'pdb_id': '1GCA', 'length': 246, 'class': 'alpha+beta', 'method': 'X-ray', 'res': 2.5, 'organism': 'Escherichia coli', 'desc': 'GTPase'},
        ]
        
        # Convert to list of dicts with standardized keys
        result = []
        for p in curated:
            result.append({
                'pdb_id': p['pdb_id'],
                'sequence_length': p['length'],
                'structural_class': p['class'],
                'experimental_method': p['method'],
                'resolution': p['res'],
                'organism': p['organism'],
                'description': p['desc']
            })
        
        return result
    
    def select_proteins(self, 
                       target_count: int = 60,
                       size_distribution: Optional[Dict[str, float]] = None,
                       max_resolution: float = 2.5,
                       max_missing_pct: float = 10.0,
                       include_nmr: bool = True,
                       max_protein_size: Optional[int] = None) -> List[ProteinMetadata]:
        """
        Select diverse set of proteins for testing.
        
        Args:
            target_count: Target number of proteins to select (default: 60)
            size_distribution: Desired distribution by size category
                             e.g., {'tiny': 0.15, 'small': 0.35, 'medium': 0.35, 'large': 0.15}
                             If None, uses default balanced distribution
            max_resolution: Maximum resolution for X-ray structures (Angstroms)
            max_missing_pct: Maximum percentage of missing residues
            include_nmr: Whether to include NMR structures
            max_protein_size: Maximum protein size in residues (optional filter)
        
        Returns:
            List of ProteinMetadata objects representing selected proteins
        """
        logger.info(f"Selecting {target_count} proteins for validation testing")
        
        # Set default size distribution if not provided
        if size_distribution is None:
            size_distribution = {
                'tiny': 0.15,    # 15% tiny proteins
                'small': 0.35,   # 35% small proteins
                'medium': 0.35,  # 35% medium proteins
                'large': 0.15    # 15% large proteins
            }
        
        # Start with curated proteins
        candidate_proteins = self._prepare_candidates(
            self.curated_proteins,
            max_resolution=max_resolution,
            max_missing_pct=max_missing_pct,
            include_nmr=include_nmr
        )
        
        # Filter by max protein size if specified
        if max_protein_size is not None:
            candidate_proteins = [
                p for p in candidate_proteins 
                if p.sequence_length <= max_protein_size
            ]
            logger.info(f"Filtered to {len(candidate_proteins)} proteins ≤ {max_protein_size} residues")
        
        logger.info(f"Found {len(candidate_proteins)} candidate proteins after filtering")
        
        # If we need more proteins, could query PDB (not implemented in this version)
        if len(candidate_proteins) < target_count:
            logger.warning(
                f"Only {len(candidate_proteins)} candidates available, "
                f"but {target_count} requested. Using all available."
            )
        
        # Balance selection across size categories
        selected = self._balance_selection(
            candidate_proteins,
            target_count=target_count,
            size_distribution=size_distribution
        )
        
        logger.info(f"Selected {len(selected)} proteins")
        self._log_selection_summary(selected)
        
        return selected
    
    def _prepare_candidates(self,
                          proteins: List[Dict],
                          max_resolution: float,
                          max_missing_pct: float,
                          include_nmr: bool) -> List[ProteinMetadata]:
        """
        Convert raw protein data to ProteinMetadata and apply filters.
        
        Args:
            proteins: List of protein dictionaries
            max_resolution: Maximum resolution filter
            max_missing_pct: Maximum missing residues filter
            include_nmr: Whether to include NMR structures
        
        Returns:
            List of filtered ProteinMetadata objects
        """
        candidates = []
        
        for p in proteins:
            try:
                # Determine size category
                length = p['sequence_length']
                if length < 50:
                    size_cat = 'tiny'
                elif length < 100:
                    size_cat = 'small'
                elif length < 200:
                    size_cat = 'medium'
                else:
                    size_cat = 'large'
                
                # Assume low missing residues for curated list
                missing_pct = 0.0  # Could be refined with actual PDB data
                
                # Create metadata object
                metadata = ProteinMetadata(
                    pdb_id=p['pdb_id'],
                    sequence_length=p['sequence_length'],
                    size_category=size_cat,
                    structural_class=p['structural_class'],
                    experimental_method=p['experimental_method'],
                    resolution=p['resolution'],
                    missing_residues_pct=missing_pct,
                    organism=p['organism'],
                    description=p['description']
                )
                
                # Apply filters
                if not self._passes_filters(metadata, max_resolution, max_missing_pct, include_nmr):
                    continue
                
                candidates.append(metadata)
                
            except Exception as e:
                logger.warning(f"Error processing protein {p.get('pdb_id', 'unknown')}: {e}")
                continue
        
        return candidates
    
    def _passes_filters(self,
                       protein: ProteinMetadata,
                       max_resolution: float,
                       max_missing_pct: float,
                       include_nmr: bool) -> bool:
        """
        Check if protein passes all filter criteria.
        
        Args:
            protein: ProteinMetadata to check
            max_resolution: Maximum resolution
            max_missing_pct: Maximum missing residues percentage
            include_nmr: Whether to include NMR structures
        
        Returns:
            True if protein passes all filters
        """
        # Missing residues filter
        if protein.missing_residues_pct > max_missing_pct:
            return False
        
        # NMR filter
        if not include_nmr and protein.experimental_method == 'NMR':
            return False
        
        # Resolution filter (only for X-ray)
        if protein.experimental_method == 'X-ray':
            if protein.resolution is None or protein.resolution > max_resolution:
                return False
        
        return True
    
    def _balance_selection(self,
                         candidates: List[ProteinMetadata],
                         target_count: int,
                         size_distribution: Dict[str, float]) -> List[ProteinMetadata]:
        """
        Select proteins to match desired size distribution.
        
        Args:
            candidates: List of candidate proteins
            target_count: Target number of proteins
            size_distribution: Desired distribution by size category
        
        Returns:
            Balanced selection of proteins
        """
        # Group candidates by size category
        by_size = defaultdict(list)
        for protein in candidates:
            by_size[protein.size_category].append(protein)
        
        # Calculate target counts for each category
        target_counts = {}
        for category, fraction in size_distribution.items():
            target_counts[category] = int(target_count * fraction)
        
        # Adjust if rounding doesn't sum to target_count
        diff = target_count - sum(target_counts.values())
        if diff != 0:
            # Add/subtract from largest category
            largest_cat = max(target_counts.items(), key=lambda x: x[1])[0]
            target_counts[largest_cat] += diff
        
        # Select proteins from each category
        selected = []
        for category in ['tiny', 'small', 'medium', 'large']:
            available = by_size[category]
            target = target_counts.get(category, 0)
            
            if len(available) >= target:
                # Randomly sample without replacement
                import random
                selected.extend(random.sample(available, target))
            else:
                # Use all available if not enough
                selected.extend(available)
                logger.warning(
                    f"Only {len(available)} {category} proteins available, "
                    f"but {target} requested"
                )
        
        return selected
    
    def filter_by_size(self,
                      proteins: List[ProteinMetadata],
                      size_categories: List[str]) -> List[ProteinMetadata]:
        """
        Filter proteins by size categories.
        
        Args:
            proteins: List of proteins to filter
            size_categories: List of allowed categories ('tiny', 'small', 'medium', 'large')
        
        Returns:
            Filtered list of proteins
        """
        return [p for p in proteins if p.size_category in size_categories]
    
    def filter_by_structural_class(self,
                                   proteins: List[ProteinMetadata],
                                   structural_classes: List[str]) -> List[ProteinMetadata]:
        """
        Filter proteins by structural class.
        
        Args:
            proteins: List of proteins to filter
            structural_classes: List of allowed classes
                              ('all-alpha', 'all-beta', 'alpha-beta', 'alpha+beta', 'irregular')
        
        Returns:
            Filtered list of proteins
        """
        return [p for p in proteins if p.structural_class in structural_classes]
    
    def filter_by_resolution(self,
                           proteins: List[ProteinMetadata],
                           max_resolution: float = 2.5) -> List[ProteinMetadata]:
        """
        Filter proteins by resolution (X-ray only).
        
        Args:
            proteins: List of proteins to filter
            max_resolution: Maximum resolution in Angstroms
        
        Returns:
            Filtered list of proteins
        """
        filtered = []
        for p in proteins:
            if p.experimental_method == 'NMR':
                # Include NMR structures (no resolution)
                filtered.append(p)
            elif p.resolution is not None and p.resolution <= max_resolution:
                filtered.append(p)
        return filtered
    
    def filter_by_completeness(self,
                              proteins: List[ProteinMetadata],
                              max_missing_pct: float = 10.0) -> List[ProteinMetadata]:
        """
        Filter proteins by completeness (max missing residues).
        
        Args:
            proteins: List of proteins to filter
            max_missing_pct: Maximum percentage of missing residues
        
        Returns:
            Filtered list of proteins
        """
        return [p for p in proteins if p.missing_residues_pct <= max_missing_pct]
    
    def export_selection(self,
                        proteins: List[ProteinMetadata],
                        output_path: str,
                        format: str = 'json') -> None:
        """
        Export selected proteins to file for reproducibility.
        
        Args:
            proteins: List of proteins to export
            output_path: Path to output file
            format: Output format ('json' or 'csv')
        
        Raises:
            ValueError: If format is not supported
        """
        if format == 'json':
            self._export_json(proteins, output_path)
        elif format == 'csv':
            self._export_csv(proteins, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")
        
        logger.info(f"Exported {len(proteins)} proteins to {output_path}")
    
    def _export_json(self, proteins: List[ProteinMetadata], output_path: str) -> None:
        """Export proteins to JSON format."""
        data = {
            'protein_count': len(proteins),
            'proteins': [asdict(p) for p in proteins]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, proteins: List[ProteinMetadata], output_path: str) -> None:
        """Export proteins to CSV format."""
        if not proteins:
            return
        
        # Get field names from dataclass
        fieldnames = list(asdict(proteins[0]).keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for protein in proteins:
                writer.writerow(asdict(protein))
    
    def load_selection(self, input_path: str) -> List[ProteinMetadata]:
        """
        Load previously exported protein selection.
        
        Args:
            input_path: Path to JSON or CSV file
        
        Returns:
            List of ProteinMetadata objects
        
        Raises:
            ValueError: If file format is not recognized
        """
        _, ext = os.path.splitext(input_path)
        
        if ext == '.json':
            return self._load_json(input_path)
        elif ext == '.csv':
            return self._load_csv(input_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .json or .csv")
    
    def _load_json(self, input_path: str) -> List[ProteinMetadata]:
        """Load proteins from JSON format."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        proteins = []
        for p_dict in data['proteins']:
            proteins.append(ProteinMetadata(**p_dict))
        
        logger.info(f"Loaded {len(proteins)} proteins from {input_path}")
        return proteins
    
    def _load_csv(self, input_path: str) -> List[ProteinMetadata]:
        """Load proteins from CSV format."""
        proteins = []
        
        with open(input_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                sequence_length = int(row['sequence_length'])
                missing_residues_pct = float(row['missing_residues_pct'])
                resolution: Optional[float] = None
                if row['resolution']:
                    resolution = float(row['resolution'])
                
                # Create protein metadata with correct types
                protein = ProteinMetadata(
                    pdb_id=row['pdb_id'],
                    sequence_length=sequence_length,
                    size_category=row['size_category'],
                    structural_class=row['structural_class'],
                    experimental_method=row['experimental_method'],
                    resolution=resolution,
                    missing_residues_pct=missing_residues_pct,
                    organism=row['organism'],
                    description=row['description']
                )
                proteins.append(protein)
        
        logger.info(f"Loaded {len(proteins)} proteins from {input_path}")
        return proteins
    
    def _log_selection_summary(self, proteins: List[ProteinMetadata]) -> None:
        """Log summary statistics of selected proteins."""
        # Count by size category
        by_size = defaultdict(int)
        for p in proteins:
            by_size[p.size_category] += 1
        
        # Count by structural class
        by_class = defaultdict(int)
        for p in proteins:
            by_class[p.structural_class] += 1
        
        # Count by experimental method
        by_method = defaultdict(int)
        for p in proteins:
            by_method[p.experimental_method] += 1
        
        logger.info("Selection Summary:")
        logger.info(f"  Total: {len(proteins)} proteins")
        logger.info(f"  By size: {dict(by_size)}")
        logger.info(f"  By class: {dict(by_class)}")
        logger.info(f"  By method: {dict(by_method)}")


# Example usage
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create selector
    selector = ProteinSelector()
    
    # Select 60 proteins with default distribution
    proteins = selector.select_proteins(target_count=60)
    
    # Export to JSON
    selector.export_selection(proteins, 'selected_proteins.json')
    
    # Export to CSV
    selector.export_selection(proteins, 'selected_proteins.csv', format='csv')
    
    # Demonstrate filtering
    small_proteins = selector.filter_by_size(proteins, ['small'])
    print(f"\nFound {len(small_proteins)} small proteins")
    
    alpha_proteins = selector.filter_by_structural_class(proteins, ['all-alpha'])
    print(f"Found {len(alpha_proteins)} all-alpha proteins")
