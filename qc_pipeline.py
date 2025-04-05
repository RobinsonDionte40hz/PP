import os
import numpy as np  # Fix numpy import
import pandas as pd
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, PDBList
import json

# Import your predictor and validator classes
from protein_predictor import QuantumCoherenceProteinPredictor
from validation_framework import QuantumProteinValidator


class QCProteinPipeline:
    """
    Pipeline for integrating the Quantum Coherence Protein Predictor with experimental validation.
    This pipeline connects prediction, simulation, and validation into a streamlined workflow.
    """
    
    def __init__(self, data_dir="protein_data"):
        """Initialize the pipeline."""
        self.data_dir = data_dir
        self.pdb_dir = os.path.join(data_dir, "pdb_files")
        self.results_dir = os.path.join(data_dir, "results")
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.pdb_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize components
        self.predictor = QuantumCoherenceProteinPredictor()
        self.predictor.pipeline = self
        
        # Initialize remaining components
        # Initialize quantum values
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.phi_harmonics = np.array([self.phi ** n for n in range(5)])
        

        
        # Initialize predictor with pipeline reference
        self.predictor = QuantumCoherenceProteinPredictor()
        self.predictor.pipeline = self
        
        # Initialize validator 
        self.validator = QuantumProteinValidator(predictor=self.predictor)
        self.validator.pipeline = self
        
        # Initialize data storage
        self.proteins = {}
        
        # No need for calculate_ss hook since we handle it in load_protein now

    def download_proteins(self, pdb_ids):
        """
        Download protein structures from the PDB.
        
        Parameters:
        -----------
        pdb_ids : list
            List of PDB IDs to download
        
        Returns:
        --------
        dict : Status of downloaded proteins
        """
        pdb_list = PDBList()
        status = {}
        
        for pdb_id in pdb_ids:
            try:
                pdb_file = pdb_list.retrieve_pdb_file(
                    pdb_code=pdb_id,
                    file_format="pdb",
                    pdir=self.pdb_dir
                )
                # Store the protein in self.proteins for later access
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(pdb_id, pdb_file)
                self.proteins[pdb_id] = {
                    "structure": structure,
                    "file_path": pdb_file
                }
                status[pdb_id] = {
                    "status": "success",
                    "file_path": pdb_file
                }
                print(f"Downloaded {pdb_id} to {pdb_file}")
            except Exception as e:
                status[pdb_id] = {
                    "status": "failed",
                    "error": str(e)
                }
                print(f"Failed to download {pdb_id}: {e}")
        
        return status
    
    def load_local_proteins(self, pdb_files):
        """
        Load protein structures from local PDB files.
        
        Parameters:
        -----------
        pdb_files : list
            List of paths to PDB files
        
        Returns:
        --------
        dict : Status of loaded proteins
        """
        status = {}
        
        for pdb_file in pdb_files:
            try:
                pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(pdb_id, pdb_file)
                self.proteins[pdb_id] = {
                    "structure": structure,
                    "file_path": pdb_file
                }
                status[pdb_id] = {
                    "status": "success",
                    "file_path": pdb_file
                }
                print(f"Loaded {pdb_id} from {pdb_file}")
            except Exception as e:
                status[pdb_id] = {
                    "status": "failed",
                    "error": str(e)
                }
                print(f"Failed to load {pdb_id}: {e}")
        
        return status
    
    def load_experimental_stability_data(self, csv_file=None):
        """
        Load real experimental stability data from CSV file.
        
        Parameters:
        -----------
        csv_file : str, optional
            Path to CSV file containing experimental data.
            If None, looks for 'experimental_stability.csv' in data_dir.
        
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing experimental stability data, or None if file not found
        """
        if csv_file is None:
            csv_file = os.path.join(self.data_dir, "experimental_stability.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Using simulated data instead.")
            return None
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded experimental data for {len(df)} proteins")
            return df
        except Exception as e:
            print(f"Error loading experimental data: {e}")
            return None
    
    def analyze_proteins(self, pdb_ids=None, chain_id='A'):
        """
        Run quantum coherence analysis on selected proteins.
        
        Parameters:
        -----------
        pdb_ids : list, optional
            List of PDB IDs to analyze. If None, analyzes all loaded proteins.
        chain_id : str, optional
            Chain ID to analyze
            
        Returns:
        --------
        dict : Analysis results
        """
        results = {}
        
        if pdb_ids is None:
            pdb_ids = list(self.proteins.keys())
        
        for pdb_id in pdb_ids:
            if pdb_id not in self.proteins:
                print(f"Protein {pdb_id} not loaded. Skipping analysis.")
                continue
            try:
                pdb_file = self.proteins[pdb_id]["file_path"]
                print(f"Analyzing {pdb_id}...")
                # Run prediction
                self.predictor.load_protein(pdb_file, chain_id)
                qcp_values = self.predictor.calculate_qcp()
                coherence = self.predictor.calculate_field_coherence()
                stability = self.predictor.predict_stability()
                phi_angles = self.predictor.analyze_phi_angles()
                thz_spectrum = self.predictor.predict_thz_spectrum()
                
                # Store results
                results[pdb_id] = {
                    "qcp_values": qcp_values.to_dict(orient="records"),
                    "coherence": coherence.to_dict(orient="records"),
                    "stability_score": stability,
                    "phi_angles": phi_angles.to_dict(orient="records") if not phi_angles.empty else [],
                    "thz_spectrum": thz_spectrum.to_dict(orient="records")
                }
                
                # Save results
                results_file = os.path.join(self.results_dir, f"{pdb_id}_analysis.json")
                with open(results_file, 'w') as f:
                    json.dump(results[pdb_id], f, indent=2)
                print(f"Analysis of {pdb_id} completed and saved to {results_file}")
            except Exception as e:
                print(f"Error analyzing {pdb_id}: {e}")
                results[pdb_id] = {"error": str(e)}
        
        return results
    
    def run_validation_pipeline(self, protein_ids, simulate_data=True):
        """
        Run the complete validation pipeline on a set of proteins.
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs to include in validation
        simulate_data : bool, optional
            Whether to use simulated experimental data (True) or real data (False)
        
        Returns:
        --------
        dict : Validation results
        """
        # Check if we have enough proteins for meaningful validation
        if len(protein_ids) < 2:
            print("WARNING: At least 2 proteins are recommended for meaningful validation.")
            print("The validation will proceed, but correlation analyses will be limited.")
        
        # Design THz experiment
        experiment_protocol = self.validator.design_thz_experiment(protein_ids)
        protocol_file = os.path.join(self.results_dir, "thz_experiment_protocol.json")
        with open(protocol_file, 'w') as f:
            json.dump(experiment_protocol, f, indent=2)
        
        # Generate simulated THz data (we always need this)
        print("Simulating THz experimental data...")
        thz_data = self.validator.simulate_thz_experiment(
            protein_ids,
            num_temp_points=6,
            add_noise=True,
            coherence_level='medium'
        )
        
        # Load experimental or simulated stability data
        if not simulate_data:
            print("Attempting to load real experimental stability data...")
            stability_data = self.load_experimental_stability_data()
            if stability_data is None:
                print("Falling back to simulated stability data")
                stability_data = self.validator.simulate_stability_data(
                    protein_ids,
                    correlation_with_coherence=0.8
                )
        else:
            print("Using simulated stability data...")
            stability_data = self.validator.simulate_stability_data(
                protein_ids,
                correlation_with_coherence=0.8
            )
        
        # Analyze THz spectra
        print("Analyzing THz spectra...")
        thz_analysis = self.validator.analyze_thz_spectra(thz_data)
        
        try:
            # Correlate with stability
            print("Correlating with stability data...")
            correlation_results = self.validator.correlate_with_stability(thz_analysis, stability_data)
            
            # Generate validation report
            print("Generating validation report...")
            validation_report = self.validator.generate_validation_report(thz_analysis, correlation_results)
            
            # Save validation results
            validation_file = os.path.join(self.results_dir, "validation_results.json")
            with open(validation_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            print(f"Validation complete. Results saved to {validation_file}")
        except Exception as e:
            print(f"Error during validation analysis: {e}")
            print("Continuing with available results...")
            correlation_results = {"error": str(e)}
            validation_report = {"error": str(e), "note": "Validation incomplete due to error"}
        
        return {
            "thz_data": thz_data,
            "stability_data": stability_data.to_dict(orient="records") if hasattr(stability_data, 'to_dict') else stability_data,
            "thz_analysis": thz_analysis,
            "correlation_results": correlation_results,
            "validation_report": validation_report
        }
    
    def visualize_results(self, protein_id, analysis_results=None):
        """
        Generate visualizations for protein analysis results.
        
        Parameters:
        -----------
        protein_id : str
            Protein ID to visualize
        analysis_results : dict, optional
            Analysis results. If None, loads from saved results.
            
        Returns:
        --------
        dict : Figure objects
        """
        if analysis_results is None:
            # Load saved analysis results
            results_file = os.path.join(self.results_dir, f"{protein_id}_analysis.json")
            if not os.path.exists(results_file):
                print(f"No saved results found for {protein_id}")
                return None
            with open(results_file, 'r') as f:
                analysis_results = json.load(f)
        
        figures = {}
        
        # Create plots directory
        plots_dir = os.path.join(self.results_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # 1. QCP Distribution
        try:
            qcp_df = pd.DataFrame(analysis_results["qcp_values"])
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(qcp_df["residue_id"], qcp_df["qcp"], 'b-')
            ax1.set_title(f'Quantum Consciousness Potential - {protein_id}')
            ax1.set_xlabel('Residue ID')
            ax1.set_ylabel('QCP Value')
            ax1.grid(True, alpha=0.3)
            
            fig1_path = os.path.join(plots_dir, f"{protein_id}_qcp.png")
            fig1.savefig(fig1_path)
            figures["qcp"] = fig1
        except Exception as e:
            print(f"Error generating QCP plot: {e}")
        
        # 2. Coherence Distribution
        try:
            coherence_df = pd.DataFrame(analysis_results["coherence"])
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(coherence_df["residue_id"], coherence_df["coherence"], 'r-')
            ax2.set_title(f'Field Coherence - {protein_id}')
            ax2.set_xlabel('Residue ID')
            ax2.set_ylabel('Coherence Value')
            ax2.grid(True, alpha=0.3)
            
            fig2_path = os.path.join(plots_dir, f"{protein_id}_coherence.png")
            fig2.savefig(fig2_path)
            figures["coherence"] = fig2
        except Exception as e:
            print(f"Error generating coherence plot: {e}")
        
        # 3. THz Spectrum
        try:
            thz_df = pd.DataFrame(analysis_results["thz_spectrum"])
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.bar(thz_df["frequency"], thz_df["intensity"], width=0.05, alpha=0.7)
            ax3.set_title(f'Predicted THz Spectrum - {protein_id}')
            ax3.set_xlabel('Frequency (THz)')
            ax3.set_ylabel('Intensity')
            
            # Add phi-based harmonics markers
            for i, freq in enumerate(self.phi_harmonics[:3]):
                ax3.axvline(x=freq, color='g', linestyle='--', 
                          label=f'φ^{i} harmonic' if i==0 else None)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            fig3_path = os.path.join(plots_dir, f"{protein_id}_thz_spectrum.png")
            fig3.savefig(fig3_path)
            figures["thz_spectrum"] = fig3
        except Exception as e:
            print(f"Error generating THz spectrum plot: {e}")
        
        # 4. Phi Angle Distribution
        try:
            if analysis_results["phi_angles"]:
                phi_df = pd.DataFrame(analysis_results["phi_angles"])
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.hist(phi_df["angle"], bins=36, range=(0, 360), alpha=0.7)
                ax4.set_title(f'Distribution of Inter-Residue Angles - {protein_id}')
                ax4.set_xlabel('Angle (degrees)')
                ax4.set_ylabel('Count')
                
                # Mark phi-based angles
                phi_angle_deg = (2 * np.pi / ((1 + np.sqrt(5)) / 2)) * 180 / np.pi  # ≈ 137.5 degrees
                ax4.axvline(x=phi_angle_deg, color='r', linestyle='--', 
                          label=f'φ angle (137.5°)')
                ax4.axvline(x=360-phi_angle_deg, color='g', linestyle='--', 
                          label=f'φ complement (222.5°)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                fig4_path = os.path.join(plots_dir, f"{protein_id}_phi_angles.png")
                fig4.savefig(fig4_path)
                figures["phi_angles"] = fig4
        except Exception as e:
            print(f"Error generating phi angle plot: {e}")
        
        print(f"Visualizations saved to {plots_dir}")
        return figures
    
    def compare_proteins(self, protein_ids, metric="stability_score"):
        """
        Compare multiple proteins based on a specific metric.
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs to compare
        metric : str, optional
            Metric to compare (e.g., "stability_score", "coherence")
            
        Returns:
        --------
        matplotlib.figure.Figure : Comparison figure
        """
        comparison_data = []
        
        for protein_id in protein_ids:
            results_file = os.path.join(self.results_dir, f"{protein_id}_analysis.json")
            if not os.path.exists(results_file):
                print(f"No saved results found for {protein_id}")
                continue
            
            with open(results_file, 'r') as f:
                analysis_results = json.load(f)
            
            if metric == "stability_score":
                comparison_data.append({
                    "protein_id": protein_id,
                    "value": analysis_results["stability_score"]
                })
            elif metric == "coherence":
                coherence_df = pd.DataFrame(analysis_results["coherence"])
                comparison_data.append({
                    "protein_id": protein_id,
                    "value": coherence_df["coherence"].mean()
                })
            elif metric == "qcp":
                qcp_df = pd.DataFrame(analysis_results["qcp_values"])
                comparison_data.append({
                    "protein_id": protein_id,
                    "value": qcp_df["qcp"].mean()
                })
            else:
                print(f"Metric {metric} not supported for comparison")
                return None
        
        if not comparison_data:
            print("No data available for comparison")
            return None
        
        # Create comparison plot
        comparison_df = pd.DataFrame(comparison_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(comparison_df["protein_id"], comparison_df["value"], alpha=0.7)
        ax.set_title(f'Protein Comparison - {metric}')
        ax.set_xlabel('Protein ID')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Save comparison plot
        plots_dir = os.path.join(self.results_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        comparison_path = os.path.join(plots_dir, f"protein_comparison_{metric}.png")
        fig.savefig(comparison_path)
        print(f"Comparison plot saved to {comparison_path}")
        
        return fig
    
    def run_complete_analysis(self, pdb_ids, chain_id='A', simulate_validation=True):
        """
        Run a complete analysis pipeline from start to finish.
        
        Parameters:
        -----------
        pdb_ids : list
            List of PDB IDs to analyze
        chain_id : str, optional
            Chain ID to analyze
        simulate_validation : bool, optional
            Whether to run validation with simulated data
        
        Returns:
        --------
        dict : Complete analysis results
        """
        print(f"Starting complete analysis for {len(pdb_ids)} proteins...")
        
        # Step 1: Download/load proteins
        print("\n=== Step 1: Downloading proteins ===")
        download_status = self.download_proteins(pdb_ids)
        
        # Step 2: Analyze proteins
        print("\n=== Step 2: Analyzing protein structure and quantum properties ===")
        analysis_results = self.analyze_proteins(pdb_ids, chain_id)
        
        # Step 3: Run validation pipeline
        print("\n=== Step 3: Running validation pipeline ===")
        if simulate_validation:
            validation_results = self.run_validation_pipeline(pdb_ids, simulate_data=True)
        else:
            validation_results = None
            print("Skipping validation step")
        
        # Step 4: Generate visualizations
        print("\n=== Step 4: Generating visualizations ===")
        visualization_results = {}
        for pdb_id in pdb_ids:
            try:
                visualization_results[pdb_id] = self.visualize_results(pdb_id, analysis_results.get(pdb_id))
            except Exception as e:
                print(f"Error generating visualizations for {pdb_id}: {e}")
        
        # Step 5: Compare proteins
        print("\n=== Step 5: Comparing proteins ===")
        if len(pdb_ids) > 1:
            comparison_fig = self.compare_proteins(pdb_ids, metric="stability_score")
        else:
            comparison_fig = None
            print("Skipping comparison (need at least 2 proteins)")
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved in {self.results_dir}")
        
        return {
            "download_status": download_status,
            "analysis_results": analysis_results,
            "validation_results": validation_results
        }


