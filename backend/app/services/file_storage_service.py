"""
File storage service - manages file system operations for work sessions
"""
import logging
import shutil
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FileStorageService:
    """Service for managing file system operations for work sessions"""
    
    def __init__(self, base_path: str = "user_data"):
        """
        Initialize file storage service
        
        Args:
            base_path: Base directory for all user data (default: "user_data")
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileStorageService initialized with base_path: {self.base_path.absolute()}")
    
    def get_user_directory(self, user_id: str) -> Path:
        """
        Get user's root directory
        
        Args:
            user_id: User's key_id from authentication
            
        Returns:
            Path to user's root directory
        """
        return self.base_path / user_id
    
    def get_session_directory(self, user_id: str, session_id: str) -> Path:
        """
        Get session directory path
        
        Args:
            user_id: User's key_id
            session_id: Work session ID
            
        Returns:
            Path to session directory
        """
        return self.base_path / user_id / "sessions" / session_id
    
    def get_prediction_directory(self, user_id: str, session_id: str, prediction_id: str) -> Path:
        """
        Get prediction directory path
        
        Args:
            user_id: User's key_id
            session_id: Work session ID
            prediction_id: Prediction ID
            
        Returns:
            Path to prediction directory
        """
        return self.base_path / user_id / "sessions" / session_id / prediction_id
    
    def create_session_directory(self, user_id: str, session_id: str) -> Path:
        """
        Create directory structure for a new session
        
        Args:
            user_id: User's key_id
            session_id: Work session ID
            
        Returns:
            Path to created session directory
            
        Raises:
            OSError: If directory creation fails
        """
        session_dir = self.get_session_directory(user_id, session_id)
        
        try:
            # Use exist_ok=True to handle race conditions
            session_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created session directory: {session_dir}")
            return session_dir
        except OSError as e:
            logger.error(f"Failed to create session directory {session_dir}: {e}")
            raise
    
    def delete_session_directory(self, user_id: str, session_id: str) -> bool:
        """
        Delete session directory and all contents with safety checks
        
        Args:
            user_id: User's key_id
            session_id: Work session ID
            
        Returns:
            True if deletion succeeded, False if directory doesn't exist
            
        Raises:
            OSError: If deletion fails
        """
        session_dir = self.get_session_directory(user_id, session_id)
        
        if not session_dir.exists():
            logger.warning(f"Session directory does not exist: {session_dir}")
            return False
        
        # Safety check: ensure path is within base_path
        if not str(session_dir.absolute()).startswith(str(self.base_path.absolute())):
            logger.error(f"Security violation: attempted to delete directory outside base_path: {session_dir}")
            raise ValueError(f"Invalid directory path: {session_dir}")
        
        try:
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session directory: {session_dir}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete session directory {session_dir}: {e}")
            raise
    
    def save_prediction_artifacts(
        self, 
        user_id: str, 
        session_id: str, 
        prediction_id: str, 
        artifacts: Dict[str, Any]
    ) -> bool:
        """
        Save prediction result files with atomic writes
        
        Expected artifact keys:
        - results: Dict - Prediction results (saved as results.json)
        - trajectory: Dict - Agent trajectory (saved as trajectory.json)
        - structure: str - PDB structure content (saved as structure.pdb)
        - visualization: bytes - Visualization image (saved as visualization.png)
        
        Args:
            user_id: User's key_id
            session_id: Work session ID
            prediction_id: Prediction ID
            artifacts: Dictionary of artifact data
            
        Returns:
            True if all files saved successfully
            
        Raises:
            OSError: If file writing fails
            KeyError: If required artifact is missing
        """
        prediction_dir = self.get_prediction_directory(user_id, session_id, prediction_id)
        
        try:
            # Create prediction directory
            prediction_dir.mkdir(parents=True, exist_ok=True)
            
            # Define artifact mappings: {artifact_key: (filename, write_mode)}
            artifact_files = {
                'results': ('results.json', 'w'),
                'trajectory': ('trajectory.json', 'w'),
                'structure': ('structure.pdb', 'w'),
                'visualization': ('visualization.png', 'wb'),
            }
            
            # Write each artifact with atomic operations
            for artifact_key, (filename, mode) in artifact_files.items():
                if artifact_key not in artifacts:
                    logger.warning(f"Missing artifact '{artifact_key}' for prediction {prediction_id}")
                    continue
                
                file_path = prediction_dir / filename
                temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
                
                try:
                    # Write to temporary file
                    with open(temp_path, mode) as f:
                        data = artifacts[artifact_key]
                        
                        if mode == 'w':
                            # JSON or text data
                            if isinstance(data, (dict, list)):
                                json.dump(data, f, indent=2)
                            else:
                                f.write(str(data))
                        else:
                            # Binary data
                            f.write(data)
                    
                    # Atomic rename
                    temp_path.replace(file_path)
                    logger.debug(f"Saved artifact {filename} for prediction {prediction_id}")
                    
                except Exception as e:
                    # Clean up temp file on failure
                    if temp_path.exists():
                        temp_path.unlink()
                    raise
            
            logger.info(f"Saved all artifacts for prediction {prediction_id}")
            return True
            
        except OSError as e:
            logger.error(f"Failed to save artifacts for prediction {prediction_id}: {e}")
            raise
    
    def create_zip_archive(self, user_id: str, session_id: str, output_path: Path) -> Path:
        """
        Create ZIP archive of session directory
        
        Args:
            user_id: User's key_id
            session_id: Work session ID
            output_path: Path for output ZIP file
            
        Returns:
            Path to created ZIP file
            
        Raises:
            OSError: If archive creation fails
            FileNotFoundError: If session directory doesn't exist
        """
        session_dir = self.get_session_directory(user_id, session_id)
        
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        try:
            # Create ZIP file with compression
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through session directory
                for file_path in session_dir.rglob('*'):
                    if file_path.is_file():
                        # Calculate relative path from session directory
                        arcname = file_path.relative_to(session_dir)
                        zipf.write(file_path, arcname)
                        logger.debug(f"Added to archive: {arcname}")
            
            logger.info(f"Created ZIP archive for session {session_id}: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create ZIP archive for session {session_id}: {e}")
            # Clean up partial ZIP file
            if output_path.exists():
                output_path.unlink()
            raise
    
    def get_session_size(self, user_id: str, session_id: str) -> int:
        """
        Calculate total size of session directory in bytes
        
        Args:
            user_id: User's key_id
            session_id: Work session ID
            
        Returns:
            Total size in bytes, or 0 if directory doesn't exist
        """
        session_dir = self.get_session_directory(user_id, session_id)
        
        if not session_dir.exists():
            return 0
        
        try:
            total_size = 0
            for file_path in session_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            logger.debug(f"Session {session_id} size: {total_size} bytes")
            return total_size
            
        except OSError as e:
            logger.error(f"Failed to calculate size for session {session_id}: {e}")
            return 0
    
    def create_session_metadata(
        self,
        user_id: str,
        session_id: str,
        session_name: str,
        created_at: datetime,
        last_active_at: datetime,
        predictions: Optional[list] = None
    ) -> bool:
        """
        Create metadata.json file for a session
        
        Args:
            user_id: User's key_id
            session_id: Work session ID
            session_name: Session name
            created_at: Creation timestamp
            last_active_at: Last activity timestamp
            predictions: Optional list of prediction metadata
            
        Returns:
            True if metadata saved successfully
        """
        session_dir = self.get_session_directory(user_id, session_id)
        metadata_path = session_dir / "metadata.json"
        
        try:
            metadata = {
                "session_id": session_id,
                "name": session_name,
                "user_id": user_id,
                "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
                "last_active_at": last_active_at.isoformat() if isinstance(last_active_at, datetime) else last_active_at,
                "predictions": predictions or []
            }
            
            # Atomic write
            temp_path = metadata_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            temp_path.replace(metadata_path)
            
            logger.info(f"Created metadata for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create metadata for session {session_id}: {e}")
            return False
