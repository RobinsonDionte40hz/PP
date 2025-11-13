"""
Result parser for PP system outputs.
This module parses JSON results from the PP system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ResultParser:
    """Parser for PP system result files."""
    
    @staticmethod
    def parse_prediction_result(result_file: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a prediction result JSON file.
        
        Args:
            result_file: Path to the result JSON file
            
        Returns:
            Parsed result dictionary or None if parsing fails
        """
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Successfully parsed result file: {result_file}")
            return data
            
        except FileNotFoundError:
            logger.error(f"Result file not found: {result_file}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in result file {result_file}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing result file {result_file}: {str(e)}")
            return None
    
    @staticmethod
    def parse_checkpoint(checkpoint_file: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a checkpoint JSON file.
        
        Args:
            checkpoint_file: Path to the checkpoint JSON file
            
        Returns:
            Parsed checkpoint dictionary or None if parsing fails
        """
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Successfully parsed checkpoint file: {checkpoint_file}")
            return data
            
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {checkpoint_file}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in checkpoint file {checkpoint_file}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing checkpoint file {checkpoint_file}: {str(e)}")
            return None
    
    @staticmethod
    def extract_metrics(result_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract key metrics from a result dictionary.
        
        Args:
            result_data: Parsed result dictionary
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        try:
            # Extract common metrics (adjust based on actual PP output format)
            if 'final_rmsd' in result_data:
                metrics['rmsd'] = result_data['final_rmsd']
            
            if 'final_energy' in result_data:
                metrics['energy'] = result_data['final_energy']
            
            if 'gdt_ts' in result_data:
                metrics['gdt_ts'] = result_data['gdt_ts']
            
            if 'tm_score' in result_data:
                metrics['tm_score'] = result_data['tm_score']
            
            if 'iterations' in result_data:
                metrics['iterations'] = result_data['iterations']
            
            logger.info(f"Extracted metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {str(e)}")
            return {}
    
    @staticmethod
    def parse_campaign_report(report_file: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a campaign report JSON file.
        
        Args:
            report_file: Path to the campaign report JSON file
            
        Returns:
            Parsed report dictionary or None if parsing fails
        """
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Successfully parsed campaign report: {report_file}")
            return data
            
        except FileNotFoundError:
            logger.error(f"Campaign report file not found: {report_file}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in campaign report {report_file}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing campaign report {report_file}: {str(e)}")
            return None
