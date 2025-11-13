"""
Integration wrapper for the existing PP (Protein Prediction) system.
This module provides a Python interface to run predictions using the existing
test_protein.py and systematic_protein_testing.py scripts.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PPWrapper:
    """Wrapper for the PP system commands."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the PP wrapper.
        
        Args:
            project_root: Path to the PP project root. If None, uses parent directory.
        """
        if project_root is None:
            # Assume we're in backend/app/integrations, go up 3 levels
            self.project_root = Path(__file__).parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.test_protein_script = self.project_root / "test_protein.py"
        self.systematic_testing_script = self.project_root / "systematic_protein_testing.py"
        
        logger.info(f"PP Wrapper initialized with project root: {self.project_root}")
    
    def run_single_prediction(
        self,
        sequence: str,
        iterations: int = 1000,
        agents: int = 10,
        native_pdb: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run a single protein prediction using test_protein.py.
        
        Args:
            sequence: Protein sequence
            iterations: Number of iterations
            agents: Number of agents
            native_pdb: Optional PDB ID for validation
            output_dir: Optional output directory
            
        Returns:
            Dictionary with prediction results
        """
        cmd = [
            "python",
            str(self.test_protein_script),
            "--sequence", sequence,
            "--iterations", str(iterations),
            "--agents", str(agents),
        ]
        
        if native_pdb:
            cmd.extend(["--native", native_pdb])
        
        if output_dir:
            cmd.extend(["--output", str(output_dir)])
        
        logger.info(f"Running prediction command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Prediction failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout,
                }
            
            logger.info("Prediction completed successfully")
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Prediction timed out")
            return {
                "success": False,
                "error": "Prediction timed out after 1 hour",
            }
        except Exception as e:
            logger.error(f"Prediction failed with exception: {str(e)}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def run_campaign(
        self,
        protein_list: list[str],
        iterations: int = 500,
        agents: int = 10,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run a systematic campaign using systematic_protein_testing.py.
        
        Args:
            protein_list: List of PDB IDs or sequences
            iterations: Number of iterations per protein
            agents: Number of agents
            output_dir: Optional output directory
            
        Returns:
            Dictionary with campaign results
        """
        # Create a temporary protein list file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(protein_list, f)
            protein_list_file = f.name
        
        cmd = [
            "python",
            str(self.systematic_testing_script),
            "--proteins", protein_list_file,
            "--iterations", str(iterations),
            "--agents", str(agents),
        ]
        
        if output_dir:
            cmd.extend(["--output", str(output_dir)])
        
        logger.info(f"Running campaign command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout for campaigns
            )
            
            # Clean up temp file
            os.unlink(protein_list_file)
            
            if result.returncode != 0:
                logger.error(f"Campaign failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout,
                }
            
            logger.info("Campaign completed successfully")
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Campaign timed out")
            os.unlink(protein_list_file)
            return {
                "success": False,
                "error": "Campaign timed out after 2 hours",
            }
        except Exception as e:
            logger.error(f"Campaign failed with exception: {str(e)}")
            if os.path.exists(protein_list_file):
                os.unlink(protein_list_file)
            return {
                "success": False,
                "error": str(e),
            }
