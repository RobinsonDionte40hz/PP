"""
Unit tests for ProteinSelector

Tests:
- ProteinMetadata validation
- Protein selection with various parameters
- Filtering by size, structural class, resolution, completeness
- Balanced distribution across categories
- Export/import functionality (JSON, CSV)
- Edge cases and error handling
"""

import pytest
import json
import csv
import os
import tempfile
from pathlib import Path
from typing import List

from validation.protein_selector import ProteinSelector, ProteinMetadata


class TestProteinMetadata:
    """Test ProteinMetadata dataclass and validation."""
    
    def test_valid_protein_metadata(self):
        """Test creating valid ProteinMetadata."""
        protein = ProteinMetadata(
            pdb_id='1ubq',  # Should be normalized to uppercase
            sequence_length=76,
            size_category='small',
            structural_class='alpha+beta',
            experimental_method='X-ray',
            resolution=1.8,
            missing_residues_pct=0.0,
            organism='Homo sapiens',
            description='Ubiquitin'
        )
        
        assert protein.pdb_id == '1UBQ'  # Normalized to uppercase
        assert protein.sequence_length == 76
        assert protein.size_category == 'small'
    
    def test_invalid_size_category(self):
        """Test that invalid size category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid size_category"):
            ProteinMetadata(
                pdb_id='1UBQ',
                sequence_length=76,
                size_category='invalid',  # Invalid
                structural_class='alpha+beta',
                experimental_method='X-ray',
                resolution=1.8,
                missing_residues_pct=0.0,
                organism='Homo sapiens',
                description='Ubiquitin'
            )
    
    def test_invalid_structural_class(self):
        """Test that invalid structural class raises ValueError."""
        with pytest.raises(ValueError, match="Invalid structural_class"):
            ProteinMetadata(
                pdb_id='1UBQ',
                sequence_length=76,
                size_category='small',
                structural_class='invalid',  # Invalid
                experimental_method='X-ray',
                resolution=1.8,
                missing_residues_pct=0.0,
                organism='Homo sapiens',
                description='Ubiquitin'
            )
    
    def test_invalid_experimental_method(self):
        """Test that invalid experimental method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid experimental_method"):
            ProteinMetadata(
                pdb_id='1UBQ',
                sequence_length=76,
                size_category='small',
                structural_class='alpha+beta',
                experimental_method='invalid',  # Invalid
                resolution=1.8,
                missing_residues_pct=0.0,
                organism='Homo sapiens',
                description='Ubiquitin'
            )
    
    def test_invalid_missing_residues_pct(self):
        """Test that invalid missing residues percentage raises ValueError."""
        with pytest.raises(ValueError, match="missing_residues_pct must be between 0 and 100"):
            ProteinMetadata(
                pdb_id='1UBQ',
                sequence_length=76,
                size_category='small',
                structural_class='alpha+beta',
                experimental_method='X-ray',
                resolution=1.8,
                missing_residues_pct=150.0,  # Invalid (>100)
                organism='Homo sapiens',
                description='Ubiquitin'
            )
    
    def test_nmr_structure_without_resolution(self):
        """Test that NMR structures can have None resolution."""
        protein = ProteinMetadata(
            pdb_id='1VII',
            sequence_length=36,
            size_category='tiny',
            structural_class='all-alpha',
            experimental_method='NMR',
            resolution=None,  # Should be None for NMR
            missing_residues_pct=0.0,
            organism='Synthetic',
            description='Villin headpiece'
        )
        
        assert protein.resolution is None
        assert protein.experimental_method == 'NMR'


class TestProteinSelector:
    """Test ProteinSelector functionality."""
    
    @pytest.fixture
    def selector(self):
        """Create ProteinSelector instance for testing."""
        return ProteinSelector(cache_dir='test_cache')
    
    @pytest.fixture
    def cleanup(self):
        """Cleanup test cache directory after tests."""
        yield
        if os.path.exists('test_cache'):
            import shutil
            shutil.rmtree('test_cache')
    
    def test_selector_initialization(self, selector, cleanup):
        """Test that selector initializes correctly."""
        assert selector is not None
        assert os.path.exists(selector.cache_dir)
        assert len(selector.curated_proteins) > 0
    
    def test_select_proteins_default(self, selector, cleanup):
        """Test selecting proteins with default parameters."""
        proteins = selector.select_proteins(target_count=30)
        
        # Should return proteins
        assert len(proteins) > 0
        assert all(isinstance(p, ProteinMetadata) for p in proteins)
    
    def test_select_proteins_custom_distribution(self, selector, cleanup):
        """Test selecting proteins with custom size distribution."""
        distribution = {
            'tiny': 0.25,
            'small': 0.25,
            'medium': 0.25,
            'large': 0.25
        }
        
        proteins = selector.select_proteins(
            target_count=20,
            size_distribution=distribution
        )
        
        assert len(proteins) > 0
        
        # Check that we have proteins from different size categories
        size_categories = {p.size_category for p in proteins}
        assert len(size_categories) > 1
    
    def test_select_proteins_xray_only(self, selector, cleanup):
        """Test selecting only X-ray structures."""
        proteins = selector.select_proteins(
            target_count=20,
            include_nmr=False
        )
        
        # All should be X-ray
        for protein in proteins:
            assert protein.experimental_method == 'X-ray'
    
    def test_select_proteins_with_resolution_filter(self, selector, cleanup):
        """Test selecting proteins with strict resolution filter."""
        proteins = selector.select_proteins(
            target_count=20,
            max_resolution=2.0  # Strict filter
        )
        
        # All X-ray structures should have resolution <= 2.0
        for protein in proteins:
            if protein.experimental_method == 'X-ray':
                assert protein.resolution is not None
                assert protein.resolution <= 2.0
    
    def test_filter_by_size(self, selector, cleanup):
        """Test filtering by size categories."""
        # Get all proteins
        all_proteins = selector.select_proteins(target_count=30)
        
        # Filter for small proteins only
        small_proteins = selector.filter_by_size(all_proteins, ['small'])
        
        # All should be small
        assert all(p.size_category == 'small' for p in small_proteins)
        
        # Filter for tiny and small
        tiny_small = selector.filter_by_size(all_proteins, ['tiny', 'small'])
        assert all(p.size_category in ['tiny', 'small'] for p in tiny_small)
    
    def test_filter_by_structural_class(self, selector, cleanup):
        """Test filtering by structural class."""
        all_proteins = selector.select_proteins(target_count=30)
        
        # Filter for all-alpha
        alpha_proteins = selector.filter_by_structural_class(
            all_proteins,
            ['all-alpha']
        )
        
        assert all(p.structural_class == 'all-alpha' for p in alpha_proteins)
    
    def test_filter_by_resolution(self, selector, cleanup):
        """Test filtering by resolution."""
        all_proteins = selector.select_proteins(target_count=30)
        
        # Filter for high-resolution structures
        high_res = selector.filter_by_resolution(all_proteins, max_resolution=1.8)
        
        for protein in high_res:
            if protein.experimental_method == 'X-ray':
                assert protein.resolution is not None
                assert protein.resolution <= 1.8
    
    def test_filter_by_completeness(self, selector, cleanup):
        """Test filtering by completeness."""
        all_proteins = selector.select_proteins(target_count=30)
        
        # Filter for complete structures
        complete = selector.filter_by_completeness(all_proteins, max_missing_pct=5.0)
        
        assert all(p.missing_residues_pct <= 5.0 for p in complete)
    
    def test_export_import_json(self, selector, cleanup):
        """Test exporting and importing proteins in JSON format."""
        # Select proteins
        proteins = selector.select_proteins(target_count=10)
        
        # Export to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            selector.export_selection(proteins, json_path, format='json')
            
            # Verify file exists
            assert os.path.exists(json_path)
            
            # Load back
            loaded_proteins = selector.load_selection(json_path)
            
            # Verify same proteins
            assert len(loaded_proteins) == len(proteins)
            for original, loaded in zip(proteins, loaded_proteins):
                assert original.pdb_id == loaded.pdb_id
                assert original.sequence_length == loaded.sequence_length
                assert original.size_category == loaded.size_category
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
    
    def test_export_import_csv(self, selector, cleanup):
        """Test exporting and importing proteins in CSV format."""
        # Select proteins
        proteins = selector.select_proteins(target_count=10)
        
        # Export to CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            selector.export_selection(proteins, csv_path, format='csv')
            
            # Verify file exists
            assert os.path.exists(csv_path)
            
            # Load back
            loaded_proteins = selector.load_selection(csv_path)
            
            # Verify same proteins
            assert len(loaded_proteins) == len(proteins)
            for original, loaded in zip(proteins, loaded_proteins):
                assert original.pdb_id == loaded.pdb_id
                assert original.sequence_length == loaded.sequence_length
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_export_invalid_format(self, selector, cleanup):
        """Test that exporting with invalid format raises ValueError."""
        proteins = selector.select_proteins(target_count=5)
        
        with pytest.raises(ValueError, match="Unsupported format"):
            selector.export_selection(proteins, 'test.txt', format='xml')
    
    def test_load_invalid_format(self, selector, cleanup):
        """Test that loading from invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            selector.load_selection('test.txt')
    
    def test_balanced_distribution(self, selector, cleanup):
        """Test that selection achieves balanced distribution."""
        distribution = {
            'tiny': 0.20,
            'small': 0.30,
            'medium': 0.30,
            'large': 0.20
        }
        
        proteins = selector.select_proteins(
            target_count=20,
            size_distribution=distribution
        )
        
        # Count by size
        size_counts = {}
        for protein in proteins:
            size_counts[protein.size_category] = size_counts.get(protein.size_category, 0) + 1
        
        # Check that distribution is approximately correct (within 1 protein)
        total = len(proteins)
        for category, expected_fraction in distribution.items():
            if category in size_counts:
                actual_fraction = size_counts[category] / total
                # Allow some variance
                assert abs(actual_fraction - expected_fraction) < 0.15
    
    def test_curated_list_quality(self, selector, cleanup):
        """Test that curated list has expected quality."""
        curated = selector.curated_proteins
        
        # Should have proteins from different sizes
        lengths = [p['sequence_length'] for p in curated]
        assert min(lengths) < 50  # Has tiny proteins
        assert max(lengths) > 200  # Has large proteins
        
        # Should have different structural classes
        classes = {p['structural_class'] for p in curated}
        assert len(classes) >= 3  # At least 3 different classes
        
        # Should have both X-ray and NMR
        methods = {p['experimental_method'] for p in curated}
        assert 'X-ray' in methods
        assert 'NMR' in methods


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def selector(self):
        return ProteinSelector(cache_dir='test_cache_edge')
    
    @pytest.fixture
    def cleanup(self):
        yield
        if os.path.exists('test_cache_edge'):
            import shutil
            shutil.rmtree('test_cache_edge')
    
    def test_select_more_than_available(self, selector, cleanup):
        """Test selecting more proteins than available."""
        # Try to select way more than curated list
        proteins = selector.select_proteins(target_count=1000)
        
        # Should return all available proteins
        assert len(proteins) > 0
        assert len(proteins) <= len(selector.curated_proteins)
    
    def test_empty_filter_results(self, selector, cleanup):
        """Test filtering that results in empty list."""
        all_proteins = selector.select_proteins(target_count=20)
        
        # Filter with impossible criteria
        filtered = selector.filter_by_resolution(all_proteins, max_resolution=0.5)
        
        # Should return empty list (very few structures <0.5Å)
        assert isinstance(filtered, list)
    
    def test_export_empty_list(self, selector, cleanup):
        """Test exporting empty protein list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            selector.export_selection([], json_path, format='json')
            
            # Should create file
            assert os.path.exists(json_path)
            
            # Load back
            loaded = selector.load_selection(json_path)
            assert len(loaded) == 0
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
