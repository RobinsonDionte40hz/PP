import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import os
import json
import statsmodels.api as sm

class QuantumProteinValidator:
    """
    Experimental validation framework for the Quantum Coherence Protein Predictor.
    
    This class designs and analyzes THz spectroscopy experiments to validate
    quantum coherence metrics and correlates predictions with experimental stability data.
    """
    
    def __init__(self, predictor=None):
        """
        Initialize the validator with a predictor instance.
        
        Parameters:
        -----------
        predictor : QuantumCoherenceProteinPredictor, optional
            Instance of the protein predictor to validate
        """
        self.predictor = predictor
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
        
        # Storage for experimental data and validation results
        self.thz_experimental_data = {}
        self.stability_experimental_data = {}
        self.validation_results = {}
        
        # Define expected THz frequency ranges based on phi relationships
        self.base_frequency = 1.0  # THz
        self.phi_harmonics = [self.base_frequency * (self.phi ** n) for n in range(5)]
        
    def design_thz_experiment(self, protein_ids, output_file=None):
        """
        Design a THz spectroscopy experiment protocol for a set of proteins.
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs (e.g., PDB IDs) to include in the experiment
        output_file : str, optional
            File to save the experiment protocol
            
        Returns:
        --------
        dict : Experimental protocol specifications
        """
        # Define the experiment parameters
        protocol = {
            "experiment_type": "THz Time-Domain Spectroscopy",
            "proteins": protein_ids,
            "frequency_range": [0.1, 10.0],  # THz range to scan
            "temperature_points": [4, 20, 37, 50, 70, 90],  # Celsius
            "sample_preparation": {
                "concentration": "20 mg/mL",
                "buffer": "50 mM phosphate buffer, pH 7.4",
                "sample_state": "solution phase in quartz cuvette",
                "control_samples": ["buffer only", "denatured protein (95°C for 10 min)"]
            },
            "measurement_protocol": {
                "frequency_resolution": 0.01,  # THz
                "scans_per_sample": 100,
                "equilibration_time": 600,  # seconds
                "key_frequencies_to_monitor": self.phi_harmonics
            },
            "predicted_peaks": {}
        }
        
        # Add protein-specific predictions if predictor is available
        if self.predictor:
            for protein_id in protein_ids:
                # This would need the actual PDB file
                # predicted_thz = self.predictor.predict_thz_spectrum()
                
                # For demonstration, generate synthetic predictions
                predicted_peaks = []
                for harmonic in self.phi_harmonics:
                    # Add some protein-specific variation
                    protein_factor = sum([ord(c) for c in protein_id]) / 1000
                    adjusted_freq = harmonic * (1 + 0.02 * np.sin(protein_factor))
                    
                    predicted_peaks.append({
                        "frequency": round(adjusted_freq, 3),
                        "intensity_estimate": round(0.5 * (self.phi ** (self.phi_harmonics.index(harmonic))), 3),
                        "associated_with": f"φ^{self.phi_harmonics.index(harmonic)} harmonic"
                    })
                
                protocol["predicted_peaks"][protein_id] = predicted_peaks
        
        # Save the protocol if output file is specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(protocol, f, indent=2)
        
        return protocol
    
    def simulate_thz_experiment(self, protein_ids, num_temp_points=6, add_noise=True, coherence_level='medium'):
        """
        Simulate THz spectroscopy experimental data for testing the validation methods.
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs to simulate data for
        num_temp_points : int, optional
            Number of temperature points to simulate
        add_noise : bool, optional
            Whether to add realistic noise to the simulated data
        coherence_level : str, optional
            Level of quantum coherence to simulate ('low', 'medium', 'high')
            
        Returns:
        --------
        dict : Simulated experimental data
        """
        # Initialize storage for simulated data
        simulated_data = {}
        
        # Define coherence factors for different levels
        coherence_factors = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        }
        factor = coherence_factors.get(coherence_level, 1.0)
        
        # Define temperature points (in Celsius)
        temperatures = np.linspace(4, 90, num_temp_points)
        
        # Generate simulated THz data for each protein
        for protein_id in protein_ids:
            protein_data = {
                "metadata": {
                    "protein_id": protein_id,
                    "simulation_parameters": {
                        "coherence_level": coherence_level,
                        "noise_level": 0.1 if add_noise else 0
                    }
                },
                "temperature_data": {}
            }
            
            # Create a protein-specific factor for variation
            protein_factor = sum([ord(c) for c in protein_id]) / 1000
            
            # Generate data for each temperature point
            for temp in temperatures:
                # Temperature affects peak width and intensity
                temp_factor = 1.0 - (temp / 200)  # Higher temp = lower coherence
                
                # Create frequency range (0.1 to 10 THz)
                frequencies = np.linspace(0.1, 10, 1000)
                
                # Initialize absorption spectrum
                absorption = np.zeros_like(frequencies)
                
                # Add phi-based peaks with temperature-dependent properties
                for i, harmonic in enumerate(self.phi_harmonics):
                    # Calculate peak parameters
                    center = harmonic * (1 + 0.01 * np.sin(protein_factor))
                    amplitude = 0.8 * factor * (self.phi ** -i) * temp_factor
                    width = 0.05 + (0.01 * temp / 10)  # Peak broadens with temperature
                    
                    # Generate Gaussian peak
                    peak = amplitude * np.exp(-((frequencies - center) ** 2) / (2 * width ** 2))
                    absorption += peak
                
                # Add protein-specific background absorption that increases with frequency
                background = 0.02 * frequencies * protein_factor
                absorption += background
                
                # Add noise if requested
                if add_noise:
                    noise_level = 0.02 + (0.005 * temp / 20)  # Noise increases with temperature
                    noise = np.random.normal(0, noise_level, len(frequencies))
                    absorption += noise
                
                # Store the temperature data
                protein_data["temperature_data"][str(int(temp))] = {
                    "frequencies": frequencies.tolist(),
                    "absorption": absorption.tolist()
                }
            
            simulated_data[protein_id] = protein_data
        
        # Store the simulated data
        self.thz_experimental_data = simulated_data
        
        return simulated_data
    
    def simulate_stability_data(self, protein_ids, correlation_with_coherence=0.8):
        """
        Simulate protein stability experimental data to test correlation methods.
        
        Parameters:
        -----------
        protein_ids : list
            List of protein IDs to simulate data for
        correlation_with_coherence : float, optional
            Desired correlation between simulated coherence and stability
            
        Returns:
        --------
        pd.DataFrame : Simulated stability data
        """
        # Generate synthetic data for stability metrics
        stability_data = []
        
        for protein_id in protein_ids:
            # Use the protein ID to seed "inherent" stability
            base_stability = sum([ord(c) for c in protein_id]) % 50 + 30  # 30-80 range
            
            # Create synthetic coherence metric
            coherence = np.random.uniform(0.3, 0.9)
            
            # Generate stability metrics with correlation to coherence
            random_factor = np.random.normal(0, 0.2)
            stability_factor = correlation_with_coherence * coherence + (1 - correlation_with_coherence) * random_factor
            
            # Melting temperature (Tm): higher = more stable
            tm = base_stability + 20 * stability_factor
            
            # ΔG of unfolding: higher = more stable
            delta_g = 5 + 10 * stability_factor
            
            # Half-life at 37°C (hours): higher = more stable
            half_life = 24 * np.exp(2 * stability_factor)
            
            # pH sensitivity (pH units of stability range): higher = less sensitive
            ph_sensitivity = 2 + 3 * stability_factor
            
            # Denaturation midpoint concentration (M urea): higher = more stable
            denaturation_midpoint = 2 + 6 * stability_factor
            
            stability_data.append({
                "protein_id": protein_id,
                "predicted_coherence": coherence,
                "melting_temp_C": tm,
                "delta_G_unfolding_kcal_mol": delta_g,
                "half_life_hours_37C": half_life,
                "pH_stability_range": ph_sensitivity,
                "denaturation_midpoint_M_urea": denaturation_midpoint
            })
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(stability_data)
        
        # Store the simulated data
        self.stability_experimental_data = df
        
        return df
    
    def analyze_thz_spectra(self, experimental_data=None):
        """
        Analyze THz spectroscopy data to identify phi-based peaks.
        
        Parameters:
        -----------
        experimental_data : dict, optional
            THz spectroscopy data to analyze. If None, uses stored data.
            
        Returns:
        --------
        dict : Analysis results with identified peaks and phi correspondence
        """
        if experimental_data is None:
            if not self.thz_experimental_data:
                raise ValueError("No THz experimental data available for analysis")
            experimental_data = self.thz_experimental_data
        
        analysis_results = {}
        
        # Define the tolerance for peak matching to phi-harmonics (in THz)
        phi_peak_tolerance = 0.15
        
        for protein_id, protein_data in experimental_data.items():
            protein_results = {
                "identified_peaks": {},
                "phi_correspondence": {},
                "temperature_effects": {},
                "coherence_metric": {}
            }
            
            # Analyze each temperature point
            for temp, temp_data in protein_data["temperature_data"].items():
                frequencies = np.array(temp_data["frequencies"])
                absorption = np.array(temp_data["absorption"])
                
                # Find peaks in the spectrum
                # Use relative prominence to find significant peaks
                peak_indices, peak_properties = find_peaks(
                    absorption, 
                    height=0.05,
                    prominence=0.05,
                    width=3
                )
                
                # Extract peak information
                peaks = []
                for i, idx in enumerate(peak_indices):
                    peaks.append({
                        "frequency": frequencies[idx],
                        "amplitude": absorption[idx],
                        "width": peak_properties["widths"][i]
                    })
                
                # Match peaks to phi-harmonics
                phi_matched_peaks = {}
                for harmonic in self.phi_harmonics:
                    closest_peak = None
                    min_distance = phi_peak_tolerance
                    
                    for peak in peaks:
                        distance = abs(peak["frequency"] - harmonic)
                        if distance < min_distance:
                            min_distance = distance
                            closest_peak = peak
                    
                    if closest_peak:
                        phi_matched_peaks[f"phi^{self.phi_harmonics.index(harmonic)}"] = {
                            "expected_frequency": harmonic,
                            "actual_frequency": closest_peak["frequency"],
                            "amplitude": closest_peak["amplitude"],
                            "width": closest_peak["width"],
                            "match_quality": 1 - (min_distance / phi_peak_tolerance)
                        }
                
                # Calculate coherence metric from peak properties
                phi_match_scores = [p["match_quality"] for p in phi_matched_peaks.values()]
                peak_ratios = []
                
                # Calculate amplitude ratios between adjacent phi-harmonic peaks
                for i in range(len(self.phi_harmonics) - 1):
                    first_key = f"phi^{i}"
                    second_key = f"phi^{i+1}"
                    
                    if first_key in phi_matched_peaks and second_key in phi_matched_peaks:
                        ratio = phi_matched_peaks[first_key]["amplitude"] / max(
                            phi_matched_peaks[second_key]["amplitude"], 0.001)
                        expected_ratio = self.phi
                        ratio_match = min(ratio / expected_ratio, expected_ratio / ratio)
                        peak_ratios.append(ratio_match)
                
                # Calculate coherence metric
                if phi_match_scores and peak_ratios:
                    coherence = 0.6 * np.mean(phi_match_scores) + 0.4 * np.mean(peak_ratios)
                elif phi_match_scores:
                    coherence = np.mean(phi_match_scores)
                else:
                    coherence = 0
                    
                # Store results for this temperature
                protein_results["identified_peaks"][temp] = peaks
                protein_results["phi_correspondence"][temp] = phi_matched_peaks
                protein_results["coherence_metric"][temp] = coherence
            
            # Analyze temperature effects on coherence
            temps = sorted([float(t) for t in protein_results["coherence_metric"].keys()])
            coherence_values = [protein_results["coherence_metric"][str(int(t))] for t in temps]
            
            if len(temps) > 1:
                # Calculate coherence decay rate with temperature
                slope, intercept, r_value, p_value, std_err = stats.linregress(temps, coherence_values)
                
                protein_results["temperature_effects"] = {
                    "coherence_decay_rate": slope,
                    "correlation_coefficient": r_value,
                    "p_value": p_value,
                    "extrapolated_coherence_at_zero_C": intercept,
                    "half_coherence_temperature": -intercept/(2*slope) if slope < 0 else None
                }
            
            # Store the analysis results for this protein
            analysis_results[protein_id] = protein_results
        
        return analysis_results
    
    def calculate_prediction_accuracy(self, predictions, experimental):
        """
        Calculate RMSE and other accuracy metrics for predictions.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary containing predicted values for each protein
            {'protein_id': {'coherence': float, 'stability': float}}
        experimental : dict
            Dictionary containing experimental values
            {'protein_id': {'coherence': float, 'stability': float}}
            
        Returns:
        --------
        dict
            Dictionary containing accuracy metrics
        """
        # Collect matching protein data
        matching_proteins = set(predictions.keys()) & set(experimental.keys())
        if not matching_proteins:
            return {"error": "No matching proteins found"}
            
        # Calculate RMSE for each metric
        metrics = {
            'coherence': [],
            'stability': [],
            'temperature_sensitivity': []
        }
        
        for protein in matching_proteins:
            for metric in metrics.keys():
                if (metric in predictions[protein] and 
                    metric in experimental[protein]):
                    pred = predictions[protein][metric]
                    exp = experimental[protein][metric]
                    metrics[metric].append((pred - exp) ** 2)
        
        # Calculate RMSE for each metric
        accuracy_metrics = {}
        for metric, errors in metrics.items():
            if errors:
                rmse = np.sqrt(np.mean(errors))
                accuracy_metrics[f"{metric}_rmse"] = rmse
                
        return accuracy_metrics
    
    def correlate_with_stability(self, thz_analysis=None, stability_data=None):
        """
        Correlate THz spectroscopy analysis with experimental stability data.
        
        Parameters:
        -----------
        thz_analysis : dict, optional
            Results from THz spectroscopy analysis. If None, analyzes stored data.
        stability_data : pd.DataFrame, optional
            Experimental stability data. If None, uses stored data.
            
        Returns:
        --------
        dict : Correlation results between coherence metrics and stability
        """
        # Use stored data if not provided
        if thz_analysis is None:
            thz_analysis = self.analyze_thz_spectra()
        
        if stability_data is None:
            if hasattr(self.stability_experimental_data, 'empty') and self.stability_experimental_data.empty:
                raise ValueError("No stability data available for correlation analysis")
            stability_data = self.stability_experimental_data
        
        correlation_results = {
            "coherence_vs_stability": {},
            "phi_match_vs_stability": {},
            "temperature_sensitivity": {},
            "overall_correlation": {},
            "predictor_performance": {}
        }
        
        # Extract coherence metrics at room temperature (or closest available)
        coherence_data = []
        for protein_id, results in thz_analysis.items():
            # Find closest temperature to 25°C (room temperature)
            temps = [float(t) for t in results["coherence_metric"].keys()]
            closest_temp = min(temps, key=lambda x: abs(x - 25))
            closest_temp_str = str(int(closest_temp))
            
            # Get coherence at this temperature
            coherence = results["coherence_metric"][closest_temp_str]
            
            # Extract phi match scores
            phi_matches = {}
            if closest_temp_str in results["phi_correspondence"]:
                for harmonic, match_data in results["phi_correspondence"][closest_temp_str].items():
                    phi_matches[harmonic] = match_data["match_quality"]
            
            # Add temperature sensitivity
            temp_sensitivity = None
            if "temperature_effects" in results and "coherence_decay_rate" in results["temperature_effects"]:
                temp_sensitivity = results["temperature_effects"]["coherence_decay_rate"]
            
            coherence_data.append({
                "protein_id": protein_id,
                "thz_coherence": coherence,
                "temperature_sensitivity": temp_sensitivity,
                **phi_matches
            })
        
        # Convert to DataFrame
        coherence_df = pd.DataFrame(coherence_data)
        
        # Merge with stability data
        if not coherence_df.empty and not isinstance(stability_data, dict) and not stability_data.empty:
            merged_data = pd.merge(coherence_df, stability_data, on="protein_id")
            
            # Check if we have enough data points for correlation
            if len(merged_data) >= 2:
                # Calculate correlation between coherence and stability metrics
                stability_metrics = [
                    "melting_temp_C", 
                    "delta_G_unfolding_kcal_mol",
                    "half_life_hours_37C", 
                    "pH_stability_range",
                    "denaturation_midpoint_M_urea"
                ]
                
                for metric in stability_metrics:
                    if metric in merged_data.columns:
                        # Correlation with THz coherence
                        corr = merged_data["thz_coherence"].corr(merged_data[metric])
                        p_value = stats.pearsonr(merged_data["thz_coherence"], merged_data[metric])[1]
                        
                        correlation_results["coherence_vs_stability"][metric] = {
                            "correlation": corr,
                            "p_value": p_value
                        }
                        
                        # Correlation with temperature sensitivity (if available)
                        if "temperature_sensitivity" in merged_data.columns and not merged_data["temperature_sensitivity"].isna().all():
                            temp_corr = merged_data["temperature_sensitivity"].corr(merged_data[metric])
                            temp_p_value = stats.pearsonr(merged_data["temperature_sensitivity"], merged_data[metric])[1]
                            
                            correlation_results["temperature_sensitivity"][metric] = {
                                "correlation": temp_corr,
                                "p_value": temp_p_value
                            }
                        
                        # Correlation with phi match scores
                        phi_columns = [col for col in merged_data.columns if col.startswith("phi^")]
                        for phi_col in phi_columns:
                            phi_corr = merged_data[phi_col].corr(merged_data[metric])
                            phi_p_value = stats.pearsonr(merged_data[phi_col], merged_data[metric])[1]
                            
                            if phi_col not in correlation_results["phi_match_vs_stability"]:
                                correlation_results["phi_match_vs_stability"][phi_col] = {}
                            
                            correlation_results["phi_match_vs_stability"][phi_col][metric] = {
                                "correlation": phi_corr,
                                "p_value": phi_p_value
                            }
                
                # Calculate overall correlation score across all stability metrics
                all_corrs = [v["correlation"] for v in correlation_results["coherence_vs_stability"].values()]
                correlation_results["overall_correlation"] = {
                    "mean_correlation": np.mean(all_corrs),
                    "max_correlation": np.max(all_corrs),
                    "correlation_consistency": np.std(all_corrs)
                }
                
                # Evaluate predictor performance if predicted coherence is available
                if "predicted_coherence" in merged_data.columns:
                    pred_corr = merged_data["predicted_coherence"].corr(merged_data["thz_coherence"])
                    pred_p_value = stats.pearsonr(merged_data["predicted_coherence"], merged_data["thz_coherence"])[1]
                    
                    # Calculate correlation between predicted coherence and stability metrics
                    pred_stability_corrs = {}
                    for metric in stability_metrics:
                        if metric in merged_data.columns:
                            pred_stability_corr = merged_data["predicted_coherence"].corr(merged_data[metric])
                            pred_stability_p = stats.pearsonr(merged_data["predicted_coherence"], merged_data[metric])[1]
                            
                            pred_stability_corrs[metric] = {
                                "correlation": pred_stability_corr,
                                "p_value": pred_stability_p
                            }
                    
                    correlation_results["predictor_performance"] = {
                        "coherence_prediction_accuracy": pred_corr,
                        "coherence_prediction_p_value": pred_p_value,
                        "stability_predictions": pred_stability_corrs
                    }
                    
                    # Add RMSE calculations
                    rmse_metrics = {}
                    for metric in stability_metrics:
                        if metric in merged_data.columns:
                            pred = merged_data["predicted_coherence"]
                            actual = merged_data[metric]
                            rmse = np.sqrt(np.mean((pred - actual) ** 2))
                            rmse_metrics[f"{metric}_rmse"] = rmse
                    
                    correlation_results["predictor_performance"]["rmse_metrics"] = rmse_metrics
                    
                    # Get best performing metric by RMSE
                    best_rmse = min(rmse_metrics.values())
                    best_metric = [k for k,v in rmse_metrics.items() if v == best_rmse][0]
                    correlation_results["predictor_performance"]["best_rmse_metric"] = {
                        "metric": best_metric,
                        "rmse": best_rmse
                    }
            else:
                # Not enough data points for correlation
                correlation_results["note"] = f"Insufficient data points for correlation analysis (need at least 2, found {len(merged_data)})"
                print(f"WARNING: Insufficient data points for correlation analysis. Need at least 2 proteins, found {len(merged_data)}.")
                print("Skipping correlation calculations. Add more proteins for meaningful correlations.")
        else:
            correlation_results["note"] = "No data available for correlation analysis"
            print("WARNING: No data available for correlation analysis. Skipping correlations.")
        
        # Store the correlation results
        self.validation_results = correlation_results
        
        return correlation_results
    
    def design_validation_experiments(self, protein_set=None, num_proteins=10):
        """
        Design a comprehensive validation experiment protocol
        that tests both THz spectroscopy and stability predictions.
        
        Parameters:
        -----------
        protein_set : list, optional
            Specific proteins to include in the experiment
        num_proteins : int, optional
            Number of proteins to include if protein_set not provided
            
        Returns:
        --------
        dict : Complete experimental protocol
        """
        # Select proteins for validation
        if protein_set is None:
            # For demonstration, generate protein IDs
            # In a real implementation, would select diverse proteins from PDB
            letters = "ABCDEFGHIJKLMNOPQR"
            protein_set = [f"{letters[i%len(letters)]}{i+1}XY" for i in range(num_proteins)]
        
        # Create the validation protocol
        protocol = {
            "overview": {
                "title": "Quantum Coherence Protein Predictor Validation",
                "objective": "Validate phi-based quantum coherence model for protein stability prediction",
                "proteins": protein_set,
                "experimental_approaches": [
                    "THz spectroscopy", 
                    "DSC thermal stability",
                    "Chemical denaturation",
                    "Circular dichroism",
                    "Hydrogen-deuterium exchange"
                ]
            },
            "thz_spectroscopy": self.design_thz_experiment(protein_set),
            "stability_experiments": {
                "differential_scanning_calorimetry": {
                    "temperature_range": [20, 95],  # Celsius
                    "scan_rate": 1.0,  # °C/min
                    "sample_preparation": "Same as THz spectroscopy",
                    "key_metrics": ["Tm (°C)", "ΔH (kJ/mol)", "ΔCp (kJ/mol·K)"]
                },
                "chemical_denaturation": {
                    "denaturants": ["Urea", "GuHCl"],
                    "concentration_range": [0, 8],  # M
                    "sample_preparation": "Same as THz spectroscopy",
                    "key_metrics": ["ΔG (kJ/mol)", "m-value (kJ/mol·M)"]
                },
                "circular_dichroism": {
                    "wavelength_range": [190, 260],  # nm
                    "temperature_points": [20, 37, 50, 60, 70, 80, 90],  # Celsius
                    "scan_rate": 1.0,  # nm/s
                    "key_metrics": ["α-helix content (%)", "β-sheet content (%)", "Tm from CD (°C)"]
                },
                "hydrogen_exchange": {
                    "labeling_time": 300,  # seconds
                    "quench_conditions": "pH 2.5, 0°C",
                    "key_metrics": ["Protection factors", "EX2 rate constants (s^-1)"]
                }
            },
            "validation_criteria": {
                "experimental_correlations": [
                    "THz coherence vs. thermal stability",
                    "THz coherence vs. chemical stability",
                    "Peak amplitudes vs. stability metrics",
                    "Temperature dependence of THz spectra vs. thermal stability"
                ],
                "predictor_validation": [
                    "Agreement between QCP values from protein sequence/structure",
                    "Agreement between coherence metric from THz spectra", 
                    "Agreement between stability path from phi-harmonic ratios"
                ],
                "accuracy_metrics": [
                    "Root mean square error (RMSE) for stability prediction",
                    "Correlation coefficient (R) between predicted and measured stability",  
                    "Phi-pattern detection accuracy"
                ]
            },
            "working_hypothesis": "Proteins with stronger phi-based THz spectral patterns will demonstrate higher thermal and chemical stability",
            "success_criteria": {
                "strong_validation": "R = 0.7-0.9 between coherence metrics and stability",
                "moderate_validation": "R = 0.5-0.7 between coherence metrics and stability",
                "weak_validation": "R = 0.3-0.5 between coherence metrics and stability"
            }
        }
        
        return protocol

    def generate_validation_report(self, analysis_results=None, correlation_results=None):
        """Generate a comprehensive validation report.
        
        Parameters:
        -----------
        analysis_results : dict, optional
            Results from THz spectroscopy analysis
        correlation_results : dict, optional
            Results from correlation analysis
            
        Returns:
        --------
        dict : Validation report
        """
        # Use stored results if not provided
        if analysis_results is None:
            if hasattr(self, 'analysis_results'):
                analysis_results = self.analysis_results
            else:
                analysis_results = self.analyze_thz_spectra()
                
        if correlation_results is None:
            if hasattr(self, 'validation_results'):
                correlation_results = self.validation_results
            else:
                correlation_results = self.correlate_with_stability()
        
        # Generate the report
        report = {
            "summary": {
                "title": "Quantum Coherence Protein Predictor Validation Report",
                "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "num_proteins_analyzed": len(analysis_results) if analysis_results else 0,
                "key_findings": []
            },
            "thz_spectroscopy_findings": {
                "phi_harmonic_detection": {},
                "coherence_metrics": {},
                "temperature_effects": {}
            },
            "stability_correlations": {
                "overall_validation": {},
                "metric_specific_correlations": {},
                "predictor_performance": {}
            },
            "detailed_analysis": {
                "protein_specific_results": {}
            },
            "conclusion": {
                "validation_status": "",
                "recommendations": []
            }
        }

        # Process THz spectroscopy findings
        if analysis_results:
            # Count phi harmonic detections
            harmonic_counts = {f"phi^{i}": 0 for i in range(len(self.phi_harmonics))}
            coherence_values = []
            decay_rates = []
            
            for protein_id, results in analysis_results.items():
                # Track which harmonics were detected
                for temp, phi_matches in results["phi_correspondence"].items():
                    for harmonic in phi_matches:
                        harmonic_counts[harmonic] += 1
                
                # Collect coherence values at room temperature (or closest)
                temps = [float(t) for t in results["coherence_metric"].keys()]
                closest_temp = min(temps, key=lambda x: abs(x - 25))
                closest_temp_str = str(int(closest_temp))
                coherence = results["coherence_metric"][closest_temp_str]
                coherence_values.append(coherence)
                
                # Collect temperature decay rates
                if "temperature_effects" in results and "coherence_decay_rate" in results["temperature_effects"]:
                    decay_rates.append(results["temperature_effects"]["coherence_decay_rate"])
                
                # Store protein-specific results
                report["detailed_analysis"]["protein_specific_results"][protein_id] = {
                    "room_temp_coherence": coherence,
                    "phi_harmonics_detected": list(results["phi_correspondence"][closest_temp_str].keys()) if closest_temp_str in results["phi_correspondence"] else [],
                    "temperature_sensitivity": results["temperature_effects"]["coherence_decay_rate"] if "temperature_effects" in results and "coherence_decay_rate" in results["temperature_effects"] else None
                }
            
            # Summarize phi harmonic detection
            total_proteins = len(analysis_results)
            report["thz_spectroscopy_findings"]["phi_harmonic_detection"] = {
                harmonic: {
                    "count": count,
                    "percentage": (count / total_proteins) * 100
                }
                for harmonic, count in harmonic_counts.items()
            }
            
            # Summarize coherence metrics
            report["thz_spectroscopy_findings"]["coherence_metrics"] = {
                "mean_coherence": np.mean(coherence_values),
                "std_dev_coherence": np.std(coherence_values),
                "min_coherence": np.min(coherence_values),
                "max_coherence": np.max(coherence_values)
            }
            
            # Summarize temperature effects
            if decay_rates:
                report["thz_spectroscopy_findings"]["temperature_effects"] = {
                    "mean_decay_rate": np.mean(decay_rates),
                    "std_dev_decay_rate": np.std(decay_rates)
                }
        
        # Process stability correlations
        if correlation_results:
            # Overall validation status
            if "overall_correlation" in correlation_results:
                mean_corr = correlation_results["overall_correlation"]["mean_correlation"]
                report["stability_correlations"]["overall_validation"] = {
                    "mean_correlation": mean_corr,
                    "max_correlation": correlation_results["overall_correlation"]["max_correlation"],
                    "correlation_consistency": correlation_results["overall_correlation"]["correlation_consistency"]
                }
                
                # Determine validation status
                if mean_corr > 0.7:
                    validation_status = "Strong Validation"
                elif mean_corr > 0.5:
                    validation_status = "Moderate Validation"
                elif mean_corr > 0.3:
                    validation_status = "Weak Validation"
                else:
                    validation_status = "Insufficient Validation"
                
                report["conclusion"]["validation_status"] = validation_status
                
                # Add key findings based on correlation results
                if "coherence_vs_stability" in correlation_results:
                    for metric, result in correlation_results["coherence_vs_stability"].items():
                        if abs(result["correlation"]) > 0.5 and result["p_value"] < 0.05:
                            finding = f"THz coherence strongly correlates with {metric} (r={result['correlation']:.2f}, p={result['p_value']:.3f})"
                            report["summary"]["key_findings"].append(finding)
                
                # Evaluate predictor performance
                if "predictor_performance" in correlation_results:
                    performance = correlation_results["predictor_performance"]
                    report["stability_correlations"]["predictor_performance"] = {
                        "coherence_prediction_accuracy": performance.get("coherence_prediction_accuracy", None),
                        "stability_predictions": performance.get("stability_predictions", {})
                    }
                    
                    if "coherence_prediction_accuracy" in performance and performance["coherence_prediction_accuracy"] > 0.6:
                        finding = f"Predictor accurately estimates experimental coherence (r={performance['coherence_prediction_accuracy']:.2f})"
                        report["summary"]["key_findings"].append(finding)
                    
                    if "stability_predictions" in performance:
                        best_metric = None
                        best_corr = 0
                        
                        for metric, result in performance["stability_predictions"].items():
                            if abs(result["correlation"]) > abs(best_corr):
                                best_metric = metric
                                best_corr = result["correlation"]
                        
                        if best_metric and abs(best_corr) > 0.5:
                            finding = f"Predictor best estimates {best_metric} (r={best_corr:.2f})"
                            report["summary"]["key_findings"].append(finding)
            
            # Add recommendations based on findings
            if report["conclusion"]["validation_status"] in ["Strong Validation", "Moderate Validation"]:
                report["conclusion"]["recommendations"].append(
                    "Further refine the predictor by optimizing phi-harmonic detection in THz experiments"
                )
                report["conclusion"]["recommendations"].append(
                    "Extend analysis to a larger protein dataset to strengthen statistical power"
                )
                report["conclusion"]["recommendations"].append(
                    "Explore temperature-dependent coherence patterns for improved stability prediction"
                )
            else:
                report["conclusion"]["recommendations"].append(
                    "Revise coherence model to better capture experimental THz patterns"
                )
                report["conclusion"]["recommendations"].append(
                    "Investigate alternative phi-based metrics that may better correlate with stability"
                )
                report["conclusion"]["recommendations"].append(
                    "Consider additional quantum parameters beyond phi-harmonics in the model"
                )
        
        # Add RMSE analysis to report
        if correlation_results and "predictor_performance" in correlation_results:
            performance = correlation_results["predictor_performance"]
            
            if "rmse_metrics" in performance:
                report["stability_correlations"]["predictor_performance"]["rmse_analysis"] = {
                    "metrics": performance["rmse_metrics"],
                    "best_performing": performance.get("best_rmse_metric", {})
                }
                
                # Add RMSE-based findings
                best_rmse = performance.get("best_rmse_metric", {})
                if best_rmse:
                    finding = (f"Best RMSE achieved for {best_rmse['metric']} "
                               f"with RMSE = {best_rmse['rmse']:.3f}")
                    report["summary"]["key_findings"].append(finding)
                
                # Add recommendations based on RMSE
                if all(rmse > 0.5 for rmse in performance["rmse_metrics"].values()):
                    report["conclusion"]["recommendations"].append(
                        "Improve prediction accuracy by refining model parameters "
                        "to reduce RMSE across all metrics"
                    )
        
        return report
    
    def design_thz_sampling_time_series(self, protein_id, temperature_points=[20, 37, 60, 80], duration_hours=24, num_samples=12):
        """
        Design a time-series THz experiment to detect stability changes over time.
        
        Parameters:
        -----------
        protein_id : str
            Protein ID to design the experiment for
        temperature_points : list, optional
            List of temperatures (°C) to test
        duration_hours : int, optional
            Total duration of the experiment in hours
        num_samples : int, optional
            Number of time points to sample
            
        Returns:
        --------
        dict : Time-series experimental protocol
        """
        # Create logarithmically spaced time points (more samples early, fewer later)
        time_points = np.logspace(0, np.log10(duration_hours * 60), num_samples)  # in minutes
        
        protocol = {
            "protein_id": protein_id,
            "experiment_type": "Time-Series THz Spectroscopy",
            "objective": "Track coherence degradation over time to validate stability predictions",
            "temperature_points": temperature_points,
            "total_duration_hours": duration_hours,
            "sampling_schedule": [
                {
                    "time_point_minutes": round(t, 1),
                    "time_point_hours": round(t/60, 2),
                    "temperatures": temperature_points
                } for t in time_points
            ],
            "measurements": {
                "primary": "Full THz spectrum (0.1-10 THz)",
                "secondary": [
                    "CD at 222 nm (secondary structure)",
                    "Tryptophan fluorescence (tertiary structure)",
                    "Light scattering (aggregation)"
                ]
            },
            "analysis_plan": {
                "phi_harmonic_tracking": "Track amplitude of each phi-based harmonic over time",
                "coherence_decay_calculation": "Calculate decay constants at each temperature",
                "stability_correlation": "Compare decay rates to known stability metrics",
                "arrhenius_analysis": "Calculate activation energy of coherence decay"
            },
            "expected_outcomes": {
                "higher_stability_proteins": "Slower coherence decay over time",
                "temperature_dependence": "Arrhenius-like acceleration of decay with temperature",
                "structure_correlation": "Coherence decay should precede structural changes"
            }
        }
        
        return protocol
    
    def design_mutant_validation_experiment(self, protein_id, num_mutants=5):
        """
        Design an experiment using protein mutants with varying stabilities
        to validate quantum coherence predictions.
        
        Parameters:
        -----------
        protein_id : str
            Parent protein ID to design mutations from
        num_mutants : int, optional
            Number of mutants to design
            
        Returns:
        --------
        dict : Mutant validation protocol
        """
        # For a real implementation, this would use computational protein design
        # to create mutations with specific stability changes
        
        protocol = {
            "parent_protein": protein_id,
            "experiment_type": "Mutant Stability and Coherence Analysis",
            "objective": "Validate phi-coherence mechanism through systematic stability alterations",
            "designed_mutants": [
                {
                    "mutant_id": f"{protein_id}_M{i+1}",
                    "mutations": [f"Position {100 + i*10}, designed to {['increase', 'decrease'][i%2]} stability"],
                    "predicted_stability_change": f"{['Increase', 'Decrease'][i%2]} by {1 + i}°C in Tm",
                    "predicted_coherence_change": f"{['Increase', 'Decrease'][i%2]} in phi-harmonic strength"
                } for i in range(num_mutants)
            ],
            "experimental_methods": {
                "stability_characterization": [
                    "Differential scanning calorimetry (Tm)",
                    "Urea denaturation (ΔG)",
                    "Proteolysis resistance"
                ],
                "coherence_measurements": [
                    "THz spectroscopy (phi-harmonics)",
                    "Temperature-dependent coherence decay"
                ],
                "structural_validation": [
                    "Circular dichroism",
                    "NMR (if feasible)",
                    "X-ray crystallography (if feasible)"
                ]
            },
            "analysis_plan": {
                "correlation_analysis": "Compare stability changes to coherence changes",
                "structure-coherence relationship": "Identify structural features that modulate coherence",
                "phi-pattern sensitivity": "Determine how mutations affect specific phi-harmonics"
            },
            "validation_criteria": {
                "strong_support": "Linear relationship between stability changes and coherence changes (R > 0.8)",
                "moderate_support": "Consistent direction of change between stability and coherence (R > 0.6)",
                "weak_support": "Some correlation between stability and coherence (R > 0.4)"
            }
        }
        
        return protocol
    
    def design_inhibitor_challenge_experiment(self, protein_id, inhibitor_types=None):
        """
        Design an experiment to test how various inhibitors affect 
        quantum coherence patterns and correlate with functional changes.
        
        Parameters:
        -----------
        protein_id : str
            Protein ID to test
        inhibitor_types : list, optional
            Types of inhibitors to test
            
        Returns:
        --------
        dict : Inhibitor challenge protocol
        """
        if inhibitor_types is None:
            inhibitor_types = ["competitive", "allosteric", "covalent"]
        
        protocol = {
            "protein_id": protein_id,
            "experiment_type": "Inhibitor-Challenge THz Coherence Analysis",
            "objective": "Test how functional perturbation affects quantum coherence patterns",
            "inhibitor_panel": [
                {
                    "inhibitor_type": inhibitor_type,
                    "binding_site": f"{'Active' if inhibitor_type == 'competitive' else 'Allosteric'} site",
                    "concentration_range": "0.1× to 10× IC50",
                    "expected_functional_effect": "Dose-dependent inhibition",
                    "predicted_coherence_effect": "Disruption of specific phi-harmonics"
                } for inhibitor_type in inhibitor_types
            ],
            "experimental_methods": {
                "functional_assays": [
                    "Enzymatic activity",
                    "Binding assays",
                    "Conformational change measurements"
                ],
                "coherence_measurements": [
                    "THz spectroscopy with inhibitor titration",
                    "Temperature effects on inhibited protein",
                    "Time-dependent coherence changes after inhibition"
                ]
            },
            "analysis_plan": {
                "structure-function-coherence relationship": "Correlate functional inhibition with changes in specific phi-harmonics",
                "allosteric effects": "Compare coherence changes from different inhibitor types",
                "phi-harmonic specificity": "Identify which harmonics are most sensitive to functional state"
            },
            "validation_criteria": {
                "mechanism_support": "Inhibitors affect specific phi-harmonics related to their binding sites",
                "functional_correlation": "Degree of functional inhibition correlates with coherence changes",
                "selectivity": "Different inhibitor types produce distinct coherence change patterns"
            }
        }
        
        return protocol
    
    def integrate_experimental_data(self, thz_data, stability_data, structure_data=None):
        """
        Integrate multiple experimental datasets to provide a comprehensive
        validation of the quantum coherence model.
        
        Parameters:
        -----------
        thz_data : dict
            THz spectroscopy experimental data
        stability_data : pd.DataFrame
            Stability measurements data
        structure_data : dict, optional
            Structural information (e.g., from CD, NMR)
            
        Returns:
        --------
        dict : Integrated analysis results
        """
        # Analyze THz spectra
        thz_analysis = self.analyze_thz_spectra(thz_data)
        
        # Correlate with stability
        correlation_results = self.correlate_with_stability(thz_analysis, stability_data)
        
        # Prepare integrated results
        integrated_results = {
            "coherence_stability_correlation": correlation_results,
            "structure_relationships": {},
            "multi_variable_analysis": {},
            "validation_summary": {}
        }
        
        # Add structure relationships if data available
        if structure_data:
            structure_correlations = {}
            
            # Extract coherence metrics
            coherence_by_protein = {}
            for protein_id, results in thz_analysis.items():
                # Get coherence at room temperature (or closest)
                temps = [float(t) for t in results["coherence_metric"].keys()]
                closest_temp = min(temps, key=lambda x: abs(x - 25))
                closest_temp_str = str(int(closest_temp))
                
                coherence_by_protein[protein_id] = results["coherence_metric"][closest_temp_str]
            
            # Correlate with structural features
            if "secondary_structure" in structure_data:
                ss_correlations = {}
                
                for protein_id, ss_content in structure_data["secondary_structure"].items():
                    if protein_id in coherence_by_protein:
                        for ss_type, content in ss_content.items():
                            if ss_type not in ss_correlations:
                                ss_correlations[ss_type] = {"protein_ids": [], "coherence": [], "content": []}
                            
                            ss_correlations[ss_type]["protein_ids"].append(protein_id)
                            ss_correlations[ss_type]["coherence"].append(coherence_by_protein[protein_id])
                            ss_correlations[ss_type]["content"].append(content)
                
                # Calculate correlations
                for ss_type, data in ss_correlations.items():
                    if len(data["coherence"]) > 1:
                        corr, p_value = stats.pearsonr(data["coherence"], data["content"])
                        structure_correlations[f"secondary_structure_{ss_type}"] = {
                            "correlation": corr,
                            "p_value": p_value,
                            "sample_size": len(data["coherence"])
                        }
            
            integrated_results["structure_relationships"] = structure_correlations
            
            # Multi-variable analysis
            if len(stability_data) > 5 and structure_data:
                # Prepare data for multiple regression
                combined_data = []
                
                for protein_id in coherence_by_protein:
                    if protein_id in stability_data["protein_id"].values:
                        stability_row = stability_data[stability_data["protein_id"] == protein_id].iloc[0]
                        
                        data_row = {
                            "protein_id": protein_id,
                            "coherence": coherence_by_protein[protein_id]
                        }
                        
                        # Add stability metrics
                        for col in stability_data.columns:
                            if col != "protein_id":
                                data_row[col] = stability_row[col]
                        
                        # Add structural features if available
                        if "secondary_structure" in structure_data and protein_id in structure_data["secondary_structure"]:
                            for ss_type, content in structure_data["secondary_structure"][protein_id].items():
                                data_row[f"ss_{ss_type}"] = content
                        
                        combined_data.append(data_row)
                
                if combined_data:
                    # Convert to DataFrame
                    combined_df = pd.DataFrame(combined_data)
                    
                    # Select key stability metric for regression
                    if "melting_temp_C" in combined_df.columns:
                        target = "melting_temp_C"
                    elif "delta_G_unfolding_kcal_mol" in combined_df.columns:
                        target = "delta_G_unfolding_kcal_mol"
                    else:
                        target = [col for col in combined_df.columns if "protein_id" != col][0]
                    
                    # Predictors: coherence and structural features
                    predictors = ["coherence"]
                    predictors.extend([col for col in combined_df.columns if col.startswith("ss_")])
                    
                    # Simple multiple regression
                    X = combined_df[predictors]
                    y = combined_df[target]
                    
                    try:
                        # Add constant term
                        X = sm.add_constant(X)
                        
                        # Fit model
                        model = sm.OLS(y, X).fit()
                        
                        integrated_results["multi_variable_analysis"] = {
                            "target_variable": target,
                            "predictors": predictors,
                            "r_squared": model.rsquared,
                            "adjusted_r_squared": model.rsquared_adj,
                            "p_value": model.f_pvalue,
                            "coefficients": {
                                name: {"coef": coef, "p_value": p} 
                                for name, coef, p in zip(model.params.index, model.params, model.pvalues)
                            },
                            "coherence_contribution": {
                                "coefficient": model.params["coherence"],
                                "p_value": model.pvalues["coherence"],
                                "significance": "Significant" if model.pvalues["coherence"] < 0.05 else "Not significant"
                            }
                        }
                    except:
                        pass
        
        # Generate validation summary
        mean_corr = 0
        if "overall_correlation" in correlation_results and "mean_correlation" in correlation_results["overall_correlation"]:
            mean_corr = correlation_results["overall_correlation"]["mean_correlation"]
        
        if mean_corr > 0.7:
            validation_level = "Strong validation"
        elif mean_corr > 0.5:
            validation_level = "Moderate validation"
        elif mean_corr > 0.3:
            validation_level = "Weak validation"
        else:
            validation_level = "Insufficient validation"
        
        integrated_results["validation_summary"] = {
            "overall_validation_level": validation_level,
            "key_supporting_evidence": [],
            "improvement_areas": [],
            "next_steps": []
        }
        
        # Add supporting evidence
        if "coherence_vs_stability" in correlation_results:
            for metric, result in correlation_results["coherence_vs_stability"].items():
                if result["correlation"] > 0.6 and result["p_value"] < 0.05:
                    integrated_results["validation_summary"]["key_supporting_evidence"].append(
                        f"Strong correlation between coherence and {metric}: r = {result['correlation']:.2f}, p = {result['p_value']:.3f}"
                    )
        
        if "structure_relationships" in integrated_results:
            for feature, result in integrated_results["structure_relationships"].items():
                if abs(result["correlation"]) > 0.6 and result["p_value"] < 0.05:
                    integrated_results["validation_summary"]["key_supporting_evidence"].append(
                        f"Strong correlation between coherence and {feature}: r = {result['correlation']:.2f}, p = {result['p_value']:.3f}"
                    )
        
        # Add improvement areas
        if mean_corr < 0.7:
            integrated_results["validation_summary"]["improvement_areas"].append(
                "Strengthen coherence-stability correlation through improved THz spectroscopy methods"
            )
        
        if not structure_data:
            integrated_results["validation_summary"]["improvement_areas"].append(
                "Include structural data to better understand coherence-structure-stability relationships"
            )
        
        if "predictor_performance" in correlation_results and "coherence_prediction_accuracy" in correlation_results["predictor_performance"]:
            accuracy = correlation_results["predictor_performance"]["coherence_prediction_accuracy"]
            if accuracy < 0.7:
                integrated_results["validation_summary"]["improvement_areas"].append(
                    f"Improve predictor accuracy (current r = {accuracy:.2f}) through model refinement"
                )
        
        # Add next steps
        integrated_results["validation_summary"]["next_steps"] = [
            "Expand protein dataset to increase statistical power",
            "Test mutant series to validate causality between coherence and stability",
            "Explore time-resolved THz spectroscopy to capture dynamic coherence effects",
            "Correlate phi-harmonic patterns with specific structural features"
        ]
        
        return integrated_results