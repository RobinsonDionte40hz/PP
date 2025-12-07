"""
Result Exporters - Public API for exporting prediction results.

Provides clean interfaces for exporting results to various formats
without exposing internal implementation details.
"""

from typing import Optional
from pathlib import Path
import json
import logging

from .interfaces import IResultsExporter
from .schemas import PredictionResults

logger = logging.getLogger(__name__)


class PDBExporter(IResultsExporter):
    """
    Export prediction results to PDB format.
    
    Usage:
        from ubf_protein.api import PDBExporter, PredictionResults
        
        exporter = PDBExporter()
        path = exporter.export(results, "output.pdb")
        
        # Or get as string
        pdb_string = exporter.export_string(results)
    """
    
    def export(self, results: PredictionResults, output_path: str) -> str:
        """
        Export results to a PDB file.
        
        Args:
            results: Prediction results to export
            output_path: Path for the output file
            
        Returns:
            Path to the created file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        pdb_string = self.export_string(results)
        path.write_text(pdb_string)
        
        logger.info(f"Exported PDB to {path}")
        return str(path)
    
    def export_string(self, results: PredictionResults) -> str:
        """
        Export results to a PDB string.
        
        Args:
            results: Prediction results to export
            
        Returns:
            PDB format string
        """
        # If results already have a PDB string, return it
        if results.pdb_string:
            return results.pdb_string
        
        # Otherwise, generate from coordinates
        return self._generate_pdb(results)
    
    def _generate_pdb(self, results: PredictionResults) -> str:
        """Generate PDB from coordinates."""
        lines = []
        
        # Header
        lines.append(f"HEADER    PROTEIN STRUCTURE PREDICTION")
        lines.append(f"TITLE     EmergentFolds Prediction")
        lines.append(f"REMARK   1 SEQUENCE: {results.sequence[:60]}...")
        
        if results.metrics:
            if results.metrics.rmsd is not None:
                lines.append(f"REMARK   2 RMSD: {results.metrics.rmsd:.2f} A")
            if results.metrics.energy_total is not None:
                lines.append(f"REMARK   3 ENERGY: {results.metrics.energy_total:.2f}")
        
        # Atoms (CA only from coordinates)
        aa_3letter = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        }
        
        for i, (aa, coord) in enumerate(zip(results.sequence, results.coordinates)):
            res_name = aa_3letter.get(aa, 'UNK')
            x, y, z = coord
            
            atom_line = (
                f"ATOM  {i+1:5d}  CA  {res_name:3s} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            lines.append(atom_line)
        
        lines.append("END")
        
        return '\n'.join(lines)
    
    @property
    def file_extension(self) -> str:
        return ".pdb"
    
    @property
    def mime_type(self) -> str:
        return "chemical/x-pdb"


class JSONExporter(IResultsExporter):
    """
    Export prediction results to JSON format.
    
    Includes all metrics, coordinates, and metadata for programmatic use.
    
    Usage:
        from ubf_protein.api import JSONExporter, PredictionResults
        
        exporter = JSONExporter()
        path = exporter.export(results, "output.json")
        
        # Or get as string
        json_string = exporter.export_string(results)
    """
    
    def __init__(self, pretty: bool = True):
        """
        Initialize JSON exporter.
        
        Args:
            pretty: Whether to format JSON with indentation
        """
        self._pretty = pretty
    
    def export(self, results: PredictionResults, output_path: str) -> str:
        """
        Export results to a JSON file.
        
        Args:
            results: Prediction results to export
            output_path: Path for the output file
            
        Returns:
            Path to the created file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        json_string = self.export_string(results)
        path.write_text(json_string)
        
        logger.info(f"Exported JSON to {path}")
        return str(path)
    
    def export_string(self, results: PredictionResults) -> str:
        """
        Export results to a JSON string.
        
        Args:
            results: Prediction results to export
            
        Returns:
            JSON format string
        """
        data = results.to_dict()
        
        if self._pretty:
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data)
    
    @property
    def file_extension(self) -> str:
        return ".json"
    
    @property
    def mime_type(self) -> str:
        return "application/json"


class CIFExporter(IResultsExporter):
    """
    Export prediction results to mmCIF format.
    
    mmCIF is the modern standard format for macromolecular structures.
    """
    
    def export(self, results: PredictionResults, output_path: str) -> str:
        """Export results to a mmCIF file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        cif_string = self.export_string(results)
        path.write_text(cif_string)
        
        logger.info(f"Exported mmCIF to {path}")
        return str(path)
    
    def export_string(self, results: PredictionResults) -> str:
        """Export results to a mmCIF string."""
        lines = []
        
        # Header
        lines.append("data_emergentfolds_prediction")
        lines.append("#")
        lines.append("_entry.id PREDICTION")
        lines.append("#")
        
        # Entity
        lines.append("_entity.id 1")
        lines.append("_entity.type polymer")
        lines.append("_entity.pdbx_description 'Predicted protein structure'")
        lines.append("#")
        
        # Atom site header
        lines.append("loop_")
        lines.append("_atom_site.group_PDB")
        lines.append("_atom_site.id")
        lines.append("_atom_site.type_symbol")
        lines.append("_atom_site.label_atom_id")
        lines.append("_atom_site.label_comp_id")
        lines.append("_atom_site.label_asym_id")
        lines.append("_atom_site.label_seq_id")
        lines.append("_atom_site.Cartn_x")
        lines.append("_atom_site.Cartn_y")
        lines.append("_atom_site.Cartn_z")
        
        aa_3letter = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        }
        
        for i, (aa, coord) in enumerate(zip(results.sequence, results.coordinates)):
            res_name = aa_3letter.get(aa, 'UNK')
            x, y, z = coord
            lines.append(
                f"ATOM {i+1} C CA {res_name} A {i+1} {x:.3f} {y:.3f} {z:.3f}"
            )
        
        lines.append("#")
        
        return '\n'.join(lines)
    
    @property
    def file_extension(self) -> str:
        return ".cif"
    
    @property
    def mime_type(self) -> str:
        return "chemical/x-cif"
