"""
File manager for PP system files (PDB, checkpoints, results, visualizations).
"""

import shutil
import logging
from pathlib import Path
from typing import Optional, List
from app.config import settings

logger = logging.getLogger(__name__)

class FileManager:
    """Manages files for the PP system."""
    
    def __init__(self):
        """Initialize file manager with paths from settings."""
        self.results_dir = Path(settings.PP_RESULTS_DIR)
        self.checkpoints_dir = Path(settings.PP_CHECKPOINTS_DIR)
        self.pdb_cache_dir = Path(settings.PP_PDB_CACHE_DIR)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.results_dir, self.checkpoints_dir, self.pdb_cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def get_result_files(self, prediction_id: str) -> List[Path]:
        """
        Get all result files for a prediction.
        
        Args:
            prediction_id: Prediction identifier
            
        Returns:
            List of result file paths
        """
        pattern = f"*{prediction_id}*"
        files = list(self.results_dir.glob(pattern))
        logger.info(f"Found {len(files)} result files for prediction {prediction_id}")
        return files
    
    def get_checkpoint_files(self, prediction_id: str) -> List[Path]:
        """
        Get all checkpoint files for a prediction.
        
        Args:
            prediction_id: Prediction identifier
            
        Returns:
            List of checkpoint file paths
        """
        pattern = f"*{prediction_id}*"
        files = list(self.checkpoints_dir.glob(pattern))
        logger.info(f"Found {len(files)} checkpoint files for prediction {prediction_id}")
        return files
    
    def get_pdb_file(self, pdb_id: str) -> Optional[Path]:
        """
        Get PDB file from cache.
        
        Args:
            pdb_id: PDB identifier
            
        Returns:
            Path to PDB file or None if not found
        """
        # Try common PDB file naming patterns
        patterns = [
            f"pdb{pdb_id.lower()}.ent",
            f"{pdb_id.lower()}.pdb",
            f"{pdb_id.upper()}.pdb",
        ]
        
        for pattern in patterns:
            pdb_file = self.pdb_cache_dir / pattern
            if pdb_file.exists():
                logger.info(f"Found PDB file: {pdb_file}")
                return pdb_file
        
        logger.warning(f"PDB file not found for {pdb_id}")
        return None
    
    def save_uploaded_pdb(self, pdb_content: bytes, filename: str) -> Path:
        """
        Save an uploaded PDB file.
        
        Args:
            pdb_content: PDB file content
            filename: Filename to save as
            
        Returns:
            Path to saved file
        """
        file_path = self.pdb_cache_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(pdb_content)
        
        logger.info(f"Saved uploaded PDB file: {file_path}")
        return file_path
    
    def delete_prediction_files(self, prediction_id: str) -> bool:
        """
        Delete all files associated with a prediction.
        
        Args:
            prediction_id: Prediction identifier
            
        Returns:
            True if successful
        """
        try:
            # Delete result files
            for file in self.get_result_files(prediction_id):
                file.unlink()
                logger.info(f"Deleted result file: {file}")
            
            # Delete checkpoint files
            for file in self.get_checkpoint_files(prediction_id):
                file.unlink()
                logger.info(f"Deleted checkpoint file: {file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting files for prediction {prediction_id}: {str(e)}")
            return False
    
    def copy_file(self, source: Path, destination: Path) -> bool:
        """
        Copy a file.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful
        """
        try:
            shutil.copy2(source, destination)
            logger.info(f"Copied file from {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            return False
    
    def get_disk_usage(self) -> dict:
        """
        Get disk usage statistics.
        
        Returns:
            Dictionary with disk usage info
        """
        usage = {}
        
        for name, directory in [
            ('results', self.results_dir),
            ('checkpoints', self.checkpoints_dir),
            ('pdb_cache', self.pdb_cache_dir),
        ]:
            total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            file_count = len(list(directory.rglob('*')))
            
            usage[name] = {
                'size_bytes': total_size,
                'size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
            }
        
        logger.info(f"Disk usage: {usage}")
        return usage
