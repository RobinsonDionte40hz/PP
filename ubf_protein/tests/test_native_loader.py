"""
Unit tests for Native Structure Loader

Tests verify:
- Loading local PDB files
- Extracting Cα coordinates
- Extracting sequence
- Handling missing residues
- Error handling
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ubf_protein.rmsd_calculator import NativeStructureLoader, NativeStructure


class TestNativeStructureLoader:
    """Test native structure loading from PDB files."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_pdb = os.path.join(self.test_dir, "test_structure.pdb")
        self.loader = NativeStructureLoader(cache_dir="./test_pdb_cache")
    
    def teardown_method(self):
        """Clean up test files."""
        # Clean up cache directory if it exists
        import shutil
        if os.path.exists("./test_pdb_cache"):
            shutil.rmtree("./test_pdb_cache")
    
    def test_load_from_local_file(self):
        """Test loading structure from local PDB file."""
        if not os.path.exists(self.test_pdb):
            pytest.skip("Test PDB file not found")
        
        structure = self.loader.load_from_file(self.test_pdb, ca_only=True)
        
        assert structure is not None
        assert isinstance(structure, NativeStructure)
        assert structure.pdb_id == "test_structure"
        assert len(structure.ca_coords) > 0
    
    def test_extract_ca_coordinates(self):
        """Test extracting Cα coordinates from PDB file."""
        if not os.path.exists(self.test_pdb):
            pytest.skip("Test PDB file not found")
        
        structure = self.loader.load_from_file(self.test_pdb, ca_only=True)
        
        # Test PDB has 3 residues (ALA, CYS, ASP)
        assert len(structure.ca_coords) == 3
        
        # Check that coordinates are tuples of floats
        for coord in structure.ca_coords:
            assert isinstance(coord, tuple)
            assert len(coord) == 3
            assert all(isinstance(x, float) for x in coord)
    
    def test_extract_sequence(self):
        """Test extracting sequence from PDB file."""
        if not os.path.exists(self.test_pdb):
            pytest.skip("Test PDB file not found")
        
        structure = self.loader.load_from_file(self.test_pdb, ca_only=True)
        
        # Test PDB has sequence ACD (Ala-Cys-Asp)
        assert structure.sequence == "ACD"
        assert len(structure.sequence) == 3
    
    def test_n_residues_matches_coordinates(self):
        """Test that n_residues matches number of CA coordinates."""
        if not os.path.exists(self.test_pdb):
            pytest.skip("Test PDB file not found")
        
        structure = self.loader.load_from_file(self.test_pdb, ca_only=True)
        
        assert structure.n_residues == len(structure.ca_coords)
        assert structure.n_residues == len(structure.sequence)
    
    def test_missing_residues_detection(self):
        """Test detection of missing residues in structure."""
        if not os.path.exists(self.test_pdb):
            pytest.skip("Test PDB file not found")
        
        structure = self.loader.load_from_file(self.test_pdb, ca_only=True)
        
        # Test PDB has no missing residues (continuous 1,2,3)
        assert structure.missing_residues == []
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_from_file("nonexistent_file.pdb")
    
    def test_invalid_pdb_id_error(self):
        """Test error handling for invalid PDB ID."""
        with pytest.raises(ValueError, match="Invalid PDB ID"):
            self.loader.load_from_pdb_id("INVALID")  # Too long
    
    def test_get_sequence_from_pdb(self):
        """Test extracting sequence without loading full structure."""
        if not os.path.exists(self.test_pdb):
            pytest.skip("Test PDB file not found")
        
        sequence = self.loader.get_sequence_from_pdb(self.test_pdb)
        
        assert sequence == "ACD"
    
    def test_ca_only_vs_all_atom(self):
        """Test difference between CA-only and all-atom loading."""
        if not os.path.exists(self.test_pdb):
            pytest.skip("Test PDB file not found")
        
        structure_ca = self.loader.load_from_file(self.test_pdb, ca_only=True)
        structure_all = self.loader.load_from_file(self.test_pdb, ca_only=False)
        
        # CA-only should have 3 atoms (one per residue)
        assert len(structure_ca.ca_coords) == 3
        assert structure_ca.all_atom_coords is None
        
        # All-atom should have more atoms
        assert len(structure_all.ca_coords) == 3
        assert structure_all.all_atom_coords is not None
        assert len(structure_all.all_atom_coords) > len(structure_all.ca_coords)


class TestNativeStructure:
    """Test NativeStructure dataclass."""
    
    def test_native_structure_creation(self):
        """Test creating NativeStructure with all fields."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        
        structure = NativeStructure(
            pdb_id="TEST",
            sequence="ACE",
            ca_coords=coords,
            all_atom_coords=None,
            missing_residues=[],
            n_residues=3
        )
        
        assert structure.pdb_id == "TEST"
        assert structure.sequence == "ACE"
        assert len(structure.ca_coords) == 3
        assert structure.n_residues == 3
    
    def test_native_structure_post_init(self):
        """Test that post_init sets n_residues from ca_coords."""
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        structure = NativeStructure(
            pdb_id="TEST",
            sequence="AC",
            ca_coords=coords
        )
        
        # n_residues should be set automatically
        assert structure.n_residues == 2
        # missing_residues should default to empty list
        assert structure.missing_residues == []


class TestPDBDownload:
    """Test PDB download functionality (requires network)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = NativeStructureLoader(cache_dir="./test_pdb_cache")
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists("./test_pdb_cache"):
            shutil.rmtree("./test_pdb_cache")
    
    @pytest.mark.skip(reason="Network test - run manually if needed")
    def test_download_from_rcsb(self):
        """Test downloading PDB file from RCSB (requires network)."""
        # This test is skipped by default to avoid network dependency
        # Run manually with: pytest test_native_loader.py::TestPDBDownload::test_download_from_rcsb -v -s
        
        structure = self.loader.load_from_pdb_id("1UBQ", ca_only=True)
        
        assert structure is not None
        assert structure.pdb_id == "1UBQ"
        assert len(structure.ca_coords) > 0
        assert len(structure.sequence) > 0
        
        # Ubiquitin has 76 residues
        assert structure.n_residues == 76
    
    @pytest.mark.skip(reason="Network test - run manually if needed")
    def test_cached_file_reuse(self):
        """Test that cached files are reused."""
        # First load (downloads)
        structure1 = self.loader.load_from_pdb_id("1UBQ", ca_only=True)
        
        # Second load (uses cache)
        structure2 = self.loader.load_from_pdb_id("1UBQ", ca_only=True)
        
        assert structure1.pdb_id == structure2.pdb_id
        assert structure1.sequence == structure2.sequence
        assert len(structure1.ca_coords) == len(structure2.ca_coords)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
