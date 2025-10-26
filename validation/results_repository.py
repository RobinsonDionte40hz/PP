"""
Results Repository for Large-Scale Validation

Centralized storage system for all test results, metadata, predicted structures,
and execution logs. Supports both JSON database and Markdown documentation formats.

Key Features:
- Store ValidationReport with full metadata tracking
- Append to COMPREHENSIVE_TEST_RESULTS.md with standardized formatting
- Save predicted structures in PDB format with timestamps
- Capture execution logs and warnings/errors
- Query and retrieve stored results with flexible filters
- Track software versions, configurations, and random seeds for reproducibility
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class TestRunMetadata:
    """
    Comprehensive metadata for a single test run to ensure reproducibility.
    
    Attributes:
        pdb_id: PDB identifier
        timestamp: Execution timestamp (ISO 8601 format)
        software_version: Version of UBF/QCPP platform
        python_version: Python interpreter version
        num_agents: Number of agents in multi-agent system
        iterations_per_agent: Iterations per agent
        qcpp_enabled: Whether QCPP integration was enabled
        random_seed: Random seed for reproducibility
        adaptive_config: Adaptive configuration parameters
        execution_parameters: Additional execution parameters
        warnings: List of warnings during execution
        errors: List of errors during execution
        execution_time_seconds: Total execution time
        native_pdb_path: Path to native structure file
        predicted_pdb_path: Path to predicted structure file
    """
    pdb_id: str
    timestamp: str
    software_version: str
    python_version: str
    num_agents: int
    iterations_per_agent: int
    qcpp_enabled: bool
    random_seed: Optional[int]
    adaptive_config: Dict[str, Any]
    execution_parameters: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    native_pdb_path: Optional[str] = None
    predicted_pdb_path: Optional[str] = None


@dataclass
class StoredValidationReport:
    """
    Validation report with metadata for storage and retrieval.
    
    Combines ValidationReport data with TestRunMetadata for complete tracking.
    """
    pdb_id: str
    metadata: TestRunMetadata
    validation_metrics: Dict[str, Any]
    structure_quality: Dict[str, Any]
    additional_data: Dict[str, Any] = field(default_factory=dict)


class ResultsRepository:
    """
    Centralized storage for all validation campaign results.
    
    Manages storage of:
    - Validation reports (JSON database)
    - Markdown documentation (COMPREHENSIVE_TEST_RESULTS.md)
    - Predicted structures (PDB format)
    - Execution logs
    - Metadata for reproducibility
    
    Storage Structure:
        results/
        ├── validation_database.json        # Machine-readable results
        ├── COMPREHENSIVE_TEST_RESULTS.md  # Human-readable results
        ├── logs/
        │   └── {pdb_id}_{timestamp}.log
        ├── structures/
        │   └── {pdb_id}_predicted_{timestamp}.pdb
        └── metadata/
            └── {pdb_id}_metadata_{timestamp}.json
    """
    
    def __init__(self, base_dir: str = "results"):
        """
        Initialize ResultsRepository with base directory.
        
        Args:
            base_dir: Base directory for all results storage
        """
        self.base_dir = Path(base_dir)
        self.logs_dir = self.base_dir / "logs"
        self.structures_dir = self.base_dir / "structures"
        self.metadata_dir = self.base_dir / "metadata"
        self.database_file = self.base_dir / "validation_database.json"
        self.markdown_file = self.base_dir / "COMPREHENSIVE_TEST_RESULTS.md"
        
        self._create_directories()
        self._initialize_database()
        self._initialize_markdown()
        
        logger.info(f"ResultsRepository initialized at {self.base_dir}")
    
    def _create_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.structures_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        logger.debug(f"Created directory structure at {self.base_dir}")
    
    def _initialize_database(self) -> None:
        """Initialize JSON database if it doesn't exist."""
        if not self.database_file.exists():
            initial_data = {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "results": []
            }
            with open(self.database_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
            logger.info(f"Initialized database at {self.database_file}")
    
    def _initialize_markdown(self) -> None:
        """Initialize Markdown file with header if it doesn't exist."""
        if not self.markdown_file.exists():
            header = """# Comprehensive Validation Test Results

This document contains all validation test results for the Large-Scale Protein Validation Campaign.

## Overview

- **Created**: {created}
- **Platform**: QCPP-UBF Integrated Protein Structure Prediction
- **Purpose**: Systematic validation across diverse protein test set

## Test Results

---

""".format(created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            with open(self.markdown_file, 'w') as f:
                f.write(header)
            logger.info(f"Initialized markdown file at {self.markdown_file}")
    
    def store_result(self, 
                     pdb_id: str,
                     validation_metrics: Dict[str, Any],
                     metadata: TestRunMetadata,
                     structure_quality: Optional[Dict[str, Any]] = None,
                     additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Store complete validation result with all metadata.
        
        Performs three storage operations:
        1. Append to JSON database
        2. Append to Markdown documentation
        3. Save metadata JSON file
        
        Args:
            pdb_id: PDB identifier
            validation_metrics: Validation metrics (RMSD, GDT-TS, TM-score, energy)
            metadata: Test run metadata for reproducibility
            structure_quality: Optional structure quality metrics
            additional_data: Optional additional data
        
        Returns:
            Result ID (pdb_id + timestamp)
        """
        result_id = f"{pdb_id}_{metadata.timestamp}"
        
        # Create stored report
        report = StoredValidationReport(
            pdb_id=pdb_id,
            metadata=metadata,
            validation_metrics=validation_metrics,
            structure_quality=structure_quality or {},
            additional_data=additional_data or {}
        )
        
        # Save to JSON database
        self._save_to_json_database(report)
        
        # Append to Markdown
        self._append_to_markdown(report)
        
        # Save metadata file
        self._save_metadata_file(report)
        
        logger.info(f"Stored result for {pdb_id} with ID {result_id}")
        return result_id
    
    def _save_to_json_database(self, report: StoredValidationReport) -> None:
        """
        Append result to JSON database.
        
        Args:
            report: Stored validation report
        """
        try:
            # Read existing database
            with open(self.database_file, 'r') as f:
                database = json.load(f)
            
            # Append new result
            result_dict = {
                "pdb_id": report.pdb_id,
                "timestamp": report.metadata.timestamp,
                "metadata": asdict(report.metadata),
                "validation_metrics": report.validation_metrics,
                "structure_quality": report.structure_quality,
                "additional_data": report.additional_data
            }
            database["results"].append(result_dict)
            
            # Write back to file
            with open(self.database_file, 'w') as f:
                json.dump(database, f, indent=2)
            
            logger.debug(f"Saved {report.pdb_id} to JSON database")
            
        except Exception as e:
            logger.error(f"Failed to save to JSON database: {e}")
            raise
    
    def _append_to_markdown(self, report: StoredValidationReport) -> None:
        """
        Append result to Markdown documentation with standardized formatting.
        
        Args:
            report: Stored validation report
        """
        try:
            # Format metrics with appropriate precision
            metrics = report.validation_metrics
            rmsd = metrics.get('rmsd', 'N/A')
            gdt_ts = metrics.get('gdt_ts', 'N/A')
            tm_score = metrics.get('tm_score', 'N/A')
            energy = metrics.get('final_energy', 'N/A')
            
            # Format numbers
            if isinstance(rmsd, (int, float)):
                rmsd = f"{rmsd:.2f} Å"
            if isinstance(gdt_ts, (int, float)):
                gdt_ts = f"{gdt_ts:.1f}"
            if isinstance(tm_score, (int, float)):
                tm_score = f"{tm_score:.3f}"
            if isinstance(energy, (int, float)):
                energy = f"{energy:.2f} kcal/mol"
            
            # Determine quality assessment
            quality = self._assess_quality(report.validation_metrics)
            
            # Create markdown section
            md_section = f"""
### {report.pdb_id} - {datetime.fromisoformat(report.metadata.timestamp).strftime('%Y-%m-%d %H:%M:%S')}

**Validation Metrics:**
- RMSD: {rmsd}
- GDT-TS: {gdt_ts}
- TM-score: {tm_score}
- Final Energy: {energy}

**Quality Assessment:** {quality}

**Configuration:**
- Agents: {report.metadata.num_agents}
- Iterations/Agent: {report.metadata.iterations_per_agent}
- QCPP Enabled: {report.metadata.qcpp_enabled}
- Random Seed: {report.metadata.random_seed}
- Execution Time: {report.metadata.execution_time_seconds:.1f}s

**Files:**
- Native: `{report.metadata.native_pdb_path or 'N/A'}`
- Predicted: `{report.metadata.predicted_pdb_path or 'N/A'}`

"""
            
            # Add warnings if present
            if report.metadata.warnings:
                md_section += "**Warnings:**\n"
                for warning in report.metadata.warnings:
                    md_section += f"- {warning}\n"
                md_section += "\n"
            
            # Add errors if present
            if report.metadata.errors:
                md_section += "**Errors:**\n"
                for error in report.metadata.errors:
                    md_section += f"- {error}\n"
                md_section += "\n"
            
            md_section += "---\n\n"
            
            # Append to file
            with open(self.markdown_file, 'a', encoding='utf-8') as f:
                f.write(md_section)
            
            logger.debug(f"Appended {report.pdb_id} to Markdown documentation")
            
        except Exception as e:
            logger.error(f"Failed to append to Markdown: {e}")
            raise
    
    def _assess_quality(self, metrics: Dict[str, Any]) -> str:
        """
        Assess overall quality based on validation metrics.
        
        Args:
            metrics: Validation metrics
        
        Returns:
            Quality assessment string (Excellent/Good/Acceptable/Poor)
        """
        rmsd = metrics.get('rmsd', float('inf'))
        gdt_ts = metrics.get('gdt_ts', 0)
        tm_score = metrics.get('tm_score', 0)
        
        # RMSD-based assessment
        if rmsd < 2.0:
            rmsd_quality = "Excellent"
        elif rmsd < 4.0:
            rmsd_quality = "Good"
        elif rmsd < 5.0:
            rmsd_quality = "Acceptable"
        else:
            rmsd_quality = "Poor"
        
        # GDT-TS-based assessment
        if gdt_ts >= 80:
            gdt_quality = "Excellent"
        elif gdt_ts >= 65:
            gdt_quality = "Good"
        elif gdt_ts >= 50:
            gdt_quality = "Acceptable"
        else:
            gdt_quality = "Poor"
        
        # TM-score-based assessment
        if tm_score > 0.8:
            tm_quality = "Excellent"
        elif tm_score > 0.6:
            tm_quality = "Good"
        elif tm_score > 0.5:
            tm_quality = "Acceptable"
        else:
            tm_quality = "Poor"
        
        return f"{rmsd_quality} (RMSD: {rmsd_quality}, GDT-TS: {gdt_quality}, TM-score: {tm_quality})"
    
    def _save_metadata_file(self, report: StoredValidationReport) -> None:
        """
        Save detailed metadata to separate JSON file.
        
        Args:
            report: Stored validation report
        """
        try:
            metadata_filename = f"{report.pdb_id}_metadata_{report.metadata.timestamp.replace(':', '-')}.json"
            metadata_path = self.metadata_dir / metadata_filename
            
            metadata_dict = {
                "pdb_id": report.pdb_id,
                "metadata": asdict(report.metadata),
                "validation_metrics": report.validation_metrics,
                "structure_quality": report.structure_quality,
                "additional_data": report.additional_data
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            logger.debug(f"Saved metadata file: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save metadata file: {e}")
            # Non-critical failure - log but don't raise
    
    def save_predicted_structure(self, 
                                 pdb_id: str,
                                 structure_content: str,
                                 timestamp: Optional[str] = None) -> str:
        """
        Save predicted structure in PDB format with timestamp.
        
        Args:
            pdb_id: PDB identifier
            structure_content: PDB format structure content
            timestamp: Optional timestamp (uses current time if None)
        
        Returns:
            Path to saved PDB file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            # Convert ISO format to filename-safe format
            timestamp = timestamp.replace(':', '-').replace('T', '_')
        
        filename = f"{pdb_id}_predicted_{timestamp}.pdb"
        filepath = self.structures_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                f.write(structure_content)
            
            logger.info(f"Saved predicted structure: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save predicted structure: {e}")
            raise
    
    def save_execution_log(self,
                          pdb_id: str,
                          log_content: str,
                          timestamp: Optional[str] = None) -> str:
        """
        Save execution log for a test run.
        
        Args:
            pdb_id: PDB identifier
            log_content: Log content
            timestamp: Optional timestamp (uses current time if None)
        
        Returns:
            Path to saved log file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            # Convert ISO format to filename-safe format
            timestamp = timestamp.replace(':', '-').replace('T', '_')
        
        filename = f"{pdb_id}_{timestamp}.log"
        filepath = self.logs_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                f.write(log_content)
            
            logger.debug(f"Saved execution log: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save execution log: {e}")
            # Non-critical failure - log but don't raise
            return ""
    
    def get_all_results(self) -> List[StoredValidationReport]:
        """
        Retrieve all stored validation results.
        
        Returns:
            List of all StoredValidationReport objects
        """
        try:
            with open(self.database_file, 'r') as f:
                database = json.load(f)
            
            reports = []
            for result in database.get("results", []):
                report = StoredValidationReport(
                    pdb_id=result["pdb_id"],
                    metadata=TestRunMetadata(**result["metadata"]),
                    validation_metrics=result["validation_metrics"],
                    structure_quality=result.get("structure_quality", {}),
                    additional_data=result.get("additional_data", {})
                )
                reports.append(report)
            
            logger.info(f"Retrieved {len(reports)} results from database")
            return reports
            
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            return []
    
    def query_results(self, filters: Optional[Dict[str, Any]] = None) -> List[StoredValidationReport]:
        """
        Query stored results with flexible filters.
        
        Supported filters:
        - pdb_id: Exact or list of PDB IDs
        - min_rmsd, max_rmsd: RMSD range
        - min_gdt_ts, max_gdt_ts: GDT-TS range
        - qcpp_enabled: Boolean for QCPP integration
        - min_timestamp, max_timestamp: Date range
        
        Args:
            filters: Dictionary of filter criteria
        
        Returns:
            List of matching StoredValidationReport objects
        """
        all_results = self.get_all_results()
        
        if not filters:
            return all_results
        
        filtered = []
        for result in all_results:
            if self._matches_filters(result, filters):
                filtered.append(result)
        
        logger.info(f"Query returned {len(filtered)} results (from {len(all_results)} total)")
        return filtered
    
    def _matches_filters(self, result: StoredValidationReport, filters: Dict[str, Any]) -> bool:
        """
        Check if result matches all filter criteria.
        
        Args:
            result: Stored validation report
            filters: Filter criteria
        
        Returns:
            True if result matches all filters
        """
        # PDB ID filter
        if 'pdb_id' in filters:
            pdb_filter = filters['pdb_id']
            if isinstance(pdb_filter, str):
                if result.pdb_id != pdb_filter:
                    return False
            elif isinstance(pdb_filter, list):
                if result.pdb_id not in pdb_filter:
                    return False
        
        # RMSD range filter
        rmsd = result.validation_metrics.get('rmsd')
        if rmsd is not None:
            if 'min_rmsd' in filters and rmsd < filters['min_rmsd']:
                return False
            if 'max_rmsd' in filters and rmsd > filters['max_rmsd']:
                return False
        
        # GDT-TS range filter
        gdt_ts = result.validation_metrics.get('gdt_ts')
        if gdt_ts is not None:
            if 'min_gdt_ts' in filters and gdt_ts < filters['min_gdt_ts']:
                return False
            if 'max_gdt_ts' in filters and gdt_ts > filters['max_gdt_ts']:
                return False
        
        # TM-score range filter
        tm_score = result.validation_metrics.get('tm_score')
        if tm_score is not None:
            if 'min_tm_score' in filters and tm_score < filters['min_tm_score']:
                return False
            if 'max_tm_score' in filters and tm_score > filters['max_tm_score']:
                return False
        
        # QCPP enabled filter
        if 'qcpp_enabled' in filters:
            if result.metadata.qcpp_enabled != filters['qcpp_enabled']:
                return False
        
        # Timestamp range filter
        timestamp = result.metadata.timestamp
        if 'min_timestamp' in filters and timestamp < filters['min_timestamp']:
            return False
        if 'max_timestamp' in filters and timestamp > filters['max_timestamp']:
            return False
        
        return True
    
    def get_result_by_id(self, result_id: str) -> Optional[StoredValidationReport]:
        """
        Retrieve a specific result by ID (pdb_id_timestamp).
        
        Args:
            result_id: Result ID in format "pdb_id_timestamp"
        
        Returns:
            StoredValidationReport or None if not found
        """
        parts = result_id.split('_', 1)
        if len(parts) != 2:
            logger.warning(f"Invalid result ID format: {result_id}")
            return None
        
        pdb_id, timestamp = parts
        
        results = self.query_results({'pdb_id': pdb_id})
        for result in results:
            if result.metadata.timestamp == timestamp:
                return result
        
        logger.warning(f"Result not found: {result_id}")
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for all stored results.
        
        Returns:
            Dictionary with summary statistics
        """
        all_results = self.get_all_results()
        
        if not all_results:
            return {
                "total_results": 0,
                "unique_proteins": 0,
                "average_rmsd": None,
                "average_gdt_ts": None,
                "average_tm_score": None,
                "average_energy": None
            }
        
        # Extract metrics (filter out None values and ensure numeric types)
        rmsds: List[float] = []
        gdt_tss: List[float] = []
        tm_scores: List[float] = []
        energies: List[float] = []
        
        for r in all_results:
            rmsd = r.validation_metrics.get('rmsd')
            if rmsd is not None and isinstance(rmsd, (int, float)):
                rmsds.append(float(rmsd))
            
            gdt_ts = r.validation_metrics.get('gdt_ts')
            if gdt_ts is not None and isinstance(gdt_ts, (int, float)):
                gdt_tss.append(float(gdt_ts))
            
            tm_score = r.validation_metrics.get('tm_score')
            if tm_score is not None and isinstance(tm_score, (int, float)):
                tm_scores.append(float(tm_score))
            
            energy = r.validation_metrics.get('final_energy')
            if energy is not None and isinstance(energy, (int, float)):
                energies.append(float(energy))
        
        unique_proteins = len(set(r.pdb_id for r in all_results))
        
        return {
            "total_results": len(all_results),
            "unique_proteins": unique_proteins,
            "average_rmsd": sum(rmsds) / len(rmsds) if rmsds else None,
            "average_gdt_ts": sum(gdt_tss) / len(gdt_tss) if gdt_tss else None,
            "average_tm_score": sum(tm_scores) / len(tm_scores) if tm_scores else None,
            "average_energy": sum(energies) / len(energies) if energies else None,
            "metrics_collected": {
                "rmsd_count": len(rmsds),
                "gdt_ts_count": len(gdt_tss),
                "tm_score_count": len(tm_scores),
                "energy_count": len(energies)
            }
        }
    
    def export_to_csv(self, output_path: str, filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Export results to CSV format for external analysis.
        
        Args:
            output_path: Path to output CSV file
            filters: Optional filters to apply before export
        """
        import csv
        
        results = self.query_results(filters)
        
        if not results:
            logger.warning("No results to export")
            return
        
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    "PDB_ID", "Timestamp", "RMSD", "GDT-TS", "TM-score", "Final_Energy",
                    "Num_Agents", "Iterations_Per_Agent", "QCPP_Enabled", "Random_Seed",
                    "Execution_Time_Seconds", "Warnings", "Errors"
                ]
                writer.writerow(header)
                
                # Write data
                for result in results:
                    row = [
                        result.pdb_id,
                        result.metadata.timestamp,
                        result.validation_metrics.get('rmsd', ''),
                        result.validation_metrics.get('gdt_ts', ''),
                        result.validation_metrics.get('tm_score', ''),
                        result.validation_metrics.get('final_energy', ''),
                        result.metadata.num_agents,
                        result.metadata.iterations_per_agent,
                        result.metadata.qcpp_enabled,
                        result.metadata.random_seed,
                        result.metadata.execution_time_seconds,
                        '; '.join(result.metadata.warnings),
                        '; '.join(result.metadata.errors)
                    ]
                    writer.writerow(row)
            
            logger.info(f"Exported {len(results)} results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise
