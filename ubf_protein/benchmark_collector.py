#!/usr/bin/env python3
"""
Benchmark Data Collector for bioRxiv Paper

Systematically collects prediction results for 50+ proteins to build
comprehensive benchmark dataset. Saves structured data for analysis
and figure generation.

Usage:
    from ubf_protein.benchmark_collector import BenchmarkCollector
    
    collector = BenchmarkCollector()
    collector.run_protein("1UBQ")
    collector.run_batch(["1UBQ", "1CRN", "1VII"])
    collector.generate_summary()
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from ubf_protein.api import (
    PredictionRunner,
    PredictionConfig,
    PredictionResults,
    get_optimal_settings
)


@dataclass
class ProteinBenchmark:
    """Single protein benchmark result."""
    # Identifiers
    pdb_id: str
    protein_name: str
    sequence: str
    sequence_length: int
    
    # Configuration
    agents: int
    iterations: int
    total_conformations: int
    enable_refinement: bool
    enable_mediators: bool
    qcpp_config: str
    
    # Performance metrics
    execution_time_seconds: float
    conformations_per_second: float
    
    # Prediction quality (structural)
    best_energy: float
    best_rmsd: Optional[float]
    gdt_ts_score: Optional[float]
    tm_score: Optional[float]
    validation_quality: Optional[str]
    
    # Prediction quality (quantum)
    mean_qcp: Optional[float]
    field_coherence: Optional[float]
    phi_match_percentage: Optional[float]
    
    # Energy decomposition
    energy_bond: Optional[float]
    energy_angle: Optional[float]
    energy_dihedral: Optional[float]
    energy_vdw: Optional[float]
    energy_electrostatic: Optional[float]
    energy_hbond: Optional[float]
    
    # Experimental data (if available)
    experimental_tm: Optional[float]  # Melting temperature
    experimental_deltag: Optional[float]  # ΔG unfolding
    
    # Metadata
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BenchmarkCollector:
    """
    Collects benchmark data for systematic protein testing.
    
    Saves results in structured format for:
    - Statistical analysis
    - Figure generation
    - Paper tables
    - Supplementary data
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark collector.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "individual").mkdir(exist_ok=True)
        (self.output_dir / "structures").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        
        self.results: List[ProteinBenchmark] = []
        
        # Load protein catalog
        self.protein_catalog = self._load_protein_catalog()
        
        print(f"📊 Benchmark Collector initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Protein catalog: {len(self.protein_catalog)} proteins")
    
    def _load_protein_catalog(self) -> Dict[str, Dict]:
        """Load known protein information."""
        return {
            "1UBQ": {"name": "Ubiquitin", "residues": 76, "fold": "α+β"},
            "1CRN": {"name": "Crambin", "residues": 46, "fold": "α"},
            "1LYZ": {"name": "Lysozyme", "residues": 129, "fold": "α+β"},
            "1VII": {"name": "Villin", "residues": 35, "fold": "α"},
            "2MR9": {"name": "BBL", "residues": 47, "fold": "α"},
            "1MBN": {"name": "Myoglobin", "residues": 153, "fold": "α"},
            "1AKI": {"name": "Ribonuclease A", "residues": 124, "fold": "α+β"},
            "3CLN": {"name": "Calmodulin", "residues": 148, "fold": "α"},
            "1HEN": {"name": "Hen Lysozyme", "residues": 129, "fold": "α+β"},
            "3SSI": {"name": "SSI Inhibitor", "residues": 113, "fold": "α+β"},
            "1L2Y": {"name": "Trp-cage", "residues": 20, "fold": "α"},
            "2GB1": {"name": "GB1 domain", "residues": 56, "fold": "α+β"},
            "1ENH": {"name": "Engrailed", "residues": 54, "fold": "α"},
            "1ROP": {"name": "Repressor of Primer", "residues": 63, "fold": "α"},
            "2CI2": {"name": "Chymotrypsin Inhibitor", "residues": 64, "fold": "α+β"},
        }
    
    def _load_experimental_data(self, pdb_id: str) -> Dict[str, Optional[float]]:
        """Load experimental stability data if available."""
        exp_file = Path("data/experimental_stability.csv")
        if not exp_file.exists():
            return {"tm": None, "deltag": None}
        
        try:
            df = pd.read_csv(exp_file)
            protein_data = df[df['PDB_ID'] == pdb_id.upper()]
            if protein_data.empty:
                return {"tm": None, "deltag": None}
            
            return {
                "tm": float(protein_data['Melting_Temperature_C'].values[0]),
                "deltag": float(protein_data['DeltaG_kcal_mol'].values[0])
            }
        except Exception:
            return {"tm": None, "deltag": None}
    
    def _load_sequence(self, pdb_id: str, pdb_file: Optional[Path] = None) -> str:
        """Load sequence from PDB file."""
        from Bio.PDB import PDBParser, PDBList
        from Bio.PDB.Polypeptide import aa3, aa1
        
        if pdb_file is None:
            # Try to find in cache
            cache_paths = [
                Path(f"pdb_cache/pdb{pdb_id.lower()}.ent"),
                Path(f"quantum_coherence_proteins/pdb_files/{pdb_id.lower()}.pdb"),
            ]
            for path in cache_paths:
                if path.exists():
                    pdb_file = path
                    break
        
        # If still not found, try to download
        if pdb_file is None or not pdb_file.exists():
            print(f"   📥 Downloading PDB file for {pdb_id}...")
            cache_dir = Path("pdb_cache")
            cache_dir.mkdir(exist_ok=True)
            
            try:
                pdbl = PDBList()
                pdbl.retrieve_pdb_file(pdb_id, pdir=str(cache_dir), file_format='pdb')
                pdb_file = cache_dir / f"pdb{pdb_id.lower()}.ent"
                
                if not pdb_file.exists():
                    raise FileNotFoundError(f"Download succeeded but file not found: {pdb_file}")
                    
                print(f"   ✓ Downloaded to {pdb_file}")
            except Exception as e:
                raise FileNotFoundError(f"PDB file not found for {pdb_id} and download failed: {e}")
        
        aa_map = dict(zip(aa3, aa1))
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_file))
        
        chain = list(structure.get_chains())[0]
        residues = list(chain.get_residues())
        
        sequence = ""
        for res in residues:
            if res.id[0] == ' ':
                resname = res.resname
                sequence += aa_map.get(resname, 'X')
        
        return sequence
    
    def run_protein(
        self,
        pdb_id: str,
        agents: Optional[int] = None,
        iterations: Optional[int] = None,
        enable_refinement: bool = True,
        enable_mediators: bool = True,
        qcpp_config: str = "default"
    ) -> Optional[ProteinBenchmark]:
        """
        Run prediction on a single protein and collect benchmark data.
        
        Args:
            pdb_id: PDB ID to test
            agents: Number of agents (None = auto)
            iterations: Iterations per agent (None = auto)
            enable_refinement: Enable quantum refinement
            enable_mediators: Enable mediator agents
            qcpp_config: QCPP configuration preset
            
        Returns:
            ProteinBenchmark object with results
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {pdb_id.upper()}")
        print(f"{'='*70}")
        
        try:
            # Load sequence
            sequence = self._load_sequence(pdb_id)
            print(f"✓ Sequence loaded: {len(sequence)} residues")
            
            # Get optimal settings if not specified
            if agents is None or iterations is None:
                settings = get_optimal_settings(len(sequence))
                agents = agents or settings.agents
                iterations = iterations or settings.iterations
            
            # Get protein info
            protein_info = self.protein_catalog.get(
                pdb_id.upper(),
                {"name": pdb_id.upper(), "residues": len(sequence), "fold": "unknown"}
            )
            
            # Load experimental data
            exp_data = self._load_experimental_data(pdb_id)
            
            print(f"📊 Configuration:")
            print(f"   - Name: {protein_info['name']}")
            print(f"   - Length: {len(sequence)} residues")
            print(f"   - Agents: {agents}")
            print(f"   - Iterations: {iterations}")
            print(f"   - Refinement: {enable_refinement}")
            print(f"   - Mediators: {enable_mediators}")
            print(f"   - QCPP: {qcpp_config}")
            
            # Create configuration
            config = PredictionConfig(
                sequence=sequence,
                native_pdb=pdb_id.upper(),
                native_pdb_path=self._find_pdb_file(pdb_id),
                agents=agents,
                iterations=iterations,
                enable_refinement=enable_refinement,
                enable_mediators=enable_mediators,
                qcpp_config=qcpp_config
            )
            
            # Run prediction
            print(f"\n🚀 Running prediction...")
            start_time = time.time()
            
            runner = PredictionRunner(config)
            results: PredictionResults = runner.run()
            
            execution_time = time.time() - start_time
            
            # Extract metrics from results
            metrics = results.metrics if results.metrics else {}
            
            # Get energy from trajectory or metadata
            best_energy = results.metadata.get('best_energy', 0.0)
            if results.trajectory:
                best_energy = min(results.trajectory)
            
            # Create benchmark record
            benchmark = ProteinBenchmark(
                # Identifiers
                pdb_id=pdb_id.upper(),
                protein_name=protein_info['name'],
                sequence=sequence,
                sequence_length=len(sequence),
                
                # Configuration
                agents=agents,
                iterations=iterations,
                total_conformations=agents * iterations,
                enable_refinement=enable_refinement,
                enable_mediators=enable_mediators,
                qcpp_config=qcpp_config,
                
                # Performance
                execution_time_seconds=execution_time,
                conformations_per_second=(agents * iterations) / execution_time,
                
                # Structural quality (from ValidationMetrics)
                best_energy=best_energy,
                best_rmsd=metrics.rmsd if isinstance(metrics, dict) else getattr(metrics, 'rmsd', None),
                gdt_ts_score=metrics.gdt_ts if isinstance(metrics, dict) else getattr(metrics, 'gdt_ts', None),
                tm_score=metrics.tm_score if isinstance(metrics, dict) else getattr(metrics, 'tm_score', None),
                validation_quality=None,  # Derive from RMSD
                
                # Quantum metrics (from metadata if available)
                mean_qcp=results.metadata.get('mean_qcp'),
                field_coherence=results.metadata.get('field_coherence'),
                phi_match_percentage=results.metadata.get('phi_match_percentage'),
                
                # Energy decomposition (from metadata if available)
                energy_bond=results.metadata.get('energy_bond'),
                energy_angle=results.metadata.get('energy_angle'),
                energy_dihedral=results.metadata.get('energy_dihedral'),
                energy_vdw=results.metadata.get('energy_vdw'),
                energy_electrostatic=results.metadata.get('energy_electrostatic'),
                energy_hbond=results.metadata.get('energy_hbond'),
                
                # Experimental
                experimental_tm=exp_data['tm'],
                experimental_deltag=exp_data['deltag'],
                
                # Metadata
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
            # Save results
            self._save_benchmark(benchmark, results)
            self.results.append(benchmark)
            
            print(f"\n✅ Benchmark complete!")
            print(f"   - Time: {execution_time:.1f}s")
            print(f"   - Energy: {benchmark.best_energy:.2f} kcal/mol")
            if benchmark.best_rmsd is not None:
                print(f"   - RMSD: {benchmark.best_rmsd:.2f} Å")
            
            return benchmark
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Create failed benchmark record
            benchmark = ProteinBenchmark(
                pdb_id=pdb_id.upper(),
                protein_name=self.protein_catalog.get(pdb_id.upper(), {}).get('name', pdb_id),
                sequence="",
                sequence_length=0,
                agents=agents or 0,
                iterations=iterations or 0,
                total_conformations=0,
                enable_refinement=enable_refinement,
                enable_mediators=enable_mediators,
                qcpp_config=qcpp_config,
                execution_time_seconds=0,
                conformations_per_second=0,
                best_energy=0.0,
                best_rmsd=None,
                gdt_ts_score=None,
                tm_score=None,
                validation_quality=None,
                mean_qcp=None,
                field_coherence=None,
                phi_match_percentage=None,
                energy_bond=None,
                energy_angle=None,
                energy_dihedral=None,
                energy_vdw=None,
                energy_electrostatic=None,
                energy_hbond=None,
                experimental_tm=None,
                experimental_deltag=None,
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
            
            self.results.append(benchmark)
            return None
    
    def _find_pdb_file(self, pdb_id: str) -> Optional[str]:
        """Find PDB file in cache directories."""
        cache_paths = [
            Path(f"pdb_cache/pdb{pdb_id.lower()}.ent"),
            Path(f"quantum_coherence_proteins/pdb_files/{pdb_id.lower()}.pdb"),
            Path(f"pdb_files/{pdb_id.lower()}.pdb"),
        ]
        
        for path in cache_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _save_benchmark(self, benchmark: ProteinBenchmark, results: PredictionResults):
        """Save benchmark data and predicted structure."""
        # Save individual result as JSON
        json_path = self.output_dir / "individual" / f"{benchmark.pdb_id}_benchmark.json"
        with open(json_path, 'w') as f:
            json.dump(benchmark.to_dict(), f, indent=2)
        
        # Save predicted structure as PDB
        if results.pdb_string:
            pdb_path = self.output_dir / "structures" / f"{benchmark.pdb_id}_predicted.pdb"
            with open(pdb_path, 'w') as f:
                f.write(results.pdb_string)
        
        print(f"   💾 Saved to: {json_path.name}")
    
    def run_batch(
        self,
        pdb_ids: List[str],
        **kwargs
    ) -> List[Optional[ProteinBenchmark]]:
        """
        Run benchmarks on multiple proteins.
        
        Args:
            pdb_ids: List of PDB IDs
            **kwargs: Arguments passed to run_protein()
            
        Returns:
            List of ProteinBenchmark results
        """
        print(f"\n{'='*70}")
        print(f"BATCH BENCHMARK: {len(pdb_ids)} proteins")
        print(f"{'='*70}\n")
        
        results = []
        for i, pdb_id in enumerate(pdb_ids, 1):
            print(f"\n[{i}/{len(pdb_ids)}] Processing {pdb_id}...")
            result = self.run_protein(pdb_id, **kwargs)
            results.append(result)
            
            # Save intermediate summary
            if i % 5 == 0:
                self.generate_summary()
        
        # Final summary
        self.generate_summary()
        
        return results
    
    def generate_summary(self):
        """Generate summary statistics and CSV export."""
        if not self.results:
            print("⚠️  No results to summarize")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Save complete dataset
        csv_path = self.output_dir / "summaries" / "complete_benchmark.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate summary statistics
        successful = df[df['success'] == True]
        
        summary = {
            "total_proteins": len(df),
            "successful": len(successful),
            "failed": len(df) - len(successful),
            "mean_execution_time": successful['execution_time_seconds'].mean(),
            "mean_conformations_per_second": successful['conformations_per_second'].mean(),
            "mean_energy": successful['best_energy'].mean(),
            "mean_rmsd": successful['best_rmsd'].mean() if 'best_rmsd' in successful else None,
            "proteins_by_size": {
                "small (<50)": len(successful[successful['sequence_length'] < 50]),
                "medium (50-150)": len(successful[(successful['sequence_length'] >= 50) & (successful['sequence_length'] < 150)]),
                "large (150+)": len(successful[successful['sequence_length'] >= 150]),
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = self.output_dir / "summaries" / "benchmark_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Total proteins: {summary['total_proteins']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Mean execution time: {summary['mean_execution_time']:.1f}s")
        print(f"Mean energy: {summary['mean_energy']:.2f} kcal/mol")
        if summary['mean_rmsd'] is not None:
            print(f"Mean RMSD: {summary['mean_rmsd']:.2f} Å")
        print(f"\n💾 Summary saved to: {summary_path}")
        print(f"📊 Complete data: {csv_path}")
    
    def get_50_protein_list(self) -> List[str]:
        """
        Get curated list of 50 proteins for comprehensive benchmark.
        
        Includes diversity in:
        - Size (20-200 residues)
        - Fold type (α, β, α+β, α/β)
        - Function (enzymes, structural, regulatory)
        """
        return [
            # Small (<50 residues)
            "1L2Y", "2GB1", "1VII", "1CRN", "2MR9", "1ENH", "1ROP", "2CI2",
            "1PRU", "1PGB", "1FSD", "1BDD", "2ACY", "1BPI",
            
            # Medium (50-100 residues)
            "1UBQ", "1SHG", "2PTN", "1IGD", "2RN2", "1PGB", "2ABD", "3SSI",
            "1BPI", "1LQ7", "1CTF", "2PTL", "1SRL", "1TEN",
            
            # Medium-Large (100-150 residues)
            "1AKI", "1LYZ", "1HEN", "3CLN", "2CGA", "1BVC", "1CSP", "2HQI",
            "1YCC", "1POA", "1COA", "1MBN",
            
            # Large (150-200 residues)
            "2LZM", "1TIM", "1FXI", "1CYO", "1HNG", "3CHY",
            
            # Very Large (200+ residues) - subset
            "1PRN", "1A68"
        ]
    
    def get_fast_50_protein_list(self) -> List[str]:
        """
        Get 50 proteins with reproducible pseudo-random selection.
        
        OPTIMIZED FOR LIMITED COMPUTING RESOURCES:
        - Only small (<50 residues) and medium (50-100 residues) proteins
        - No large proteins that require extensive computing power
        - Uses fixed random seed for reproducibility (same 50 every time)
        - Diverse selection across fold types
        - Fast benchmarking (~30-60 minutes total)
        
        Returns:
            List of 50 PDB IDs (25 small + 25 medium)
        """
        import random
        
        # Comprehensive protein pool - SMALL AND MEDIUM ONLY
        all_proteins = {
            # Small (<50 residues) - fast to predict (5-15 seconds each)
            "small": [
                "1L2Y", "2GB1", "1VII", "1CRN", "2MR9", "1ENH", "1ROP", "2CI2",
                "1PRU", "1PGB", "1FSD", "1BDD", "2ACY", "1BPI", "1WQC", "2JOF",
                "1E0M", "1PQX", "2KK7", "1YRF", "1MBA", "1PSV", "2ERL", "1AB1",
                "1LMB", "2ZTA", "1BRF", "1EDO", "2PDD", "1TIG"
            ],
            # Medium (50-100 residues) - moderate speed (15-30 seconds each)
            "medium": [
                "1UBQ", "1SHG", "2PTN", "1IGD", "2RN2", "2ABD", "3SSI",
                "1LQ7", "1CTF", "2PTL", "1SRL", "1TEN", "5UBQ", "2KOX",
                "1E8L", "3GB1", "1BNI", "2EVQ", "1MB6", "1HZ6", "2CI2",
                "1PPT", "2WRP", "1RIS", "1PIN", "1H8K", "2VB1", "1YRF",
                "3ICB", "1AB9"
            ]
        }
        
        # Fixed seed for reproducibility
        random.seed(42)
        
        # Select balanced set: 25 small + 25 medium = 50 total
        selected = []
        selected.extend(random.sample(all_proteins["small"], 25))   # 25 small (50%)
        selected.extend(random.sample(all_proteins["medium"], 25))  # 25 medium (50%)
        
        # Shuffle final list (with fixed seed)
        random.seed(42)
        random.shuffle(selected)
        
        return selected


def main():
    """Command-line interface for benchmark collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect benchmark data for bioRxiv paper")
    parser.add_argument("--protein", help="Single protein PDB ID")
    parser.add_argument("--batch", help="Comma-separated PDB IDs")
    parser.add_argument("--full-50", action="store_true", help="Run full 50-protein benchmark")
    parser.add_argument("--agents", type=int, help="Number of agents (None = auto)")
    parser.add_argument("--iterations", type=int, help="Iterations per agent (None = auto)")
    parser.add_argument("--no-refinement", action="store_true", help="Disable quantum refinement")
    parser.add_argument("--no-mediators", action="store_true", help="Disable mediator agents")
    parser.add_argument("--qcpp", default="default", help="QCPP config: none, default, high_accuracy")
    parser.add_argument("--output", default="benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    collector = BenchmarkCollector(output_dir=args.output)
    
    if args.protein:
        collector.run_protein(
            args.protein,
            agents=args.agents,
            iterations=args.iterations,
            enable_refinement=not args.no_refinement,
            enable_mediators=not args.no_mediators,
            qcpp_config=args.qcpp
        )
    elif args.batch:
        pdb_ids = [p.strip() for p in args.batch.split(',')]
        collector.run_batch(
            pdb_ids,
            agents=args.agents,
            iterations=args.iterations,
            enable_refinement=not args.no_refinement,
            enable_mediators=not args.no_mediators,
            qcpp_config=args.qcpp
        )
    elif args.full_50:
        pdb_ids = collector.get_50_protein_list()
        collector.run_batch(
            pdb_ids,
            agents=args.agents,
            iterations=args.iterations,
            enable_refinement=not args.no_refinement,
            enable_mediators=not args.no_mediators,
            qcpp_config=args.qcpp
        )
    else:
        print("Please specify --protein, --batch, or --full-50")
        parser.print_help()


if __name__ == "__main__":
    main()
