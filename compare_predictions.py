import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy import stats

def compare_predictions_with_experimental(results_dir="quantum_coherence_proteins/results", 
                                       exp_data_file="experimental_stability.csv"):
    """
    Compare QCP predictions with experimental stability data.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing analysis results
    exp_data_file : str
        Path to experimental data CSV file
        
    Returns:
    --------
    pd.DataFrame
        Comparison of predictions with experimental data
    """
    # Load experimental data with more flexible parsing
    try:
        exp_data = pd.read_csv(exp_data_file, 
                              sep=None,  # Auto-detect separator
                              engine='python',  # More flexible engine
                              on_bad_lines='warn')  # Warn about problematic lines
        
        required_columns = ["protein_id", "melting_temp_C", "delta_G_unfolding_kcal_mol"]
        if not all(col in exp_data.columns for col in required_columns):
            print(f"Warning: CSV file missing required columns. Expected: {required_columns}")
            print(f"Found columns: {exp_data.columns.tolist()}")
            return None
            
    except Exception as e:
        print(f"Error reading experimental data file: {e}")
        print("Please ensure the CSV file has the correct format:")
        print("protein_id,melting_temp_C,delta_G_unfolding_kcal_mol,...")
        return None

    # Collect predictions from analysis results
    predictions = []
    for protein_id in exp_data["protein_id"]:
        results_file = os.path.join(results_dir, f"{protein_id}_analysis.json")
        if not os.path.exists(results_file):
            print(f"No results found for {protein_id}")
            continue
            
        with open(results_file, 'r') as f:
            analysis = json.load(f)
            
        # Extract key metrics
        predictions.append({
            "protein_id": protein_id,
            "stability_score": analysis["stability_score"],
            "mean_qcp": np.mean([res["qcp"] for res in analysis["qcp_values"]]),
            "mean_coherence": np.mean([res["coherence"] for res in analysis["coherence"]])
        })
    
    if not predictions:
        print("No matching predictions found")
        return None
        
    # Convert to DataFrame and merge with experimental data
    pred_df = pd.DataFrame(predictions)
    comparison = pd.merge(pred_df, exp_data, on="protein_id")
    
    return comparison

if __name__ == "__main__":
    comparison_data = compare_predictions_with_experimental()
    if comparison_data is not None:
        print("\nComparison Data:")
        print(comparison_data)
        
        # Print correlations
        print("\nCorrelations with experimental data:")
        for pred_col in ["stability_score", "mean_qcp", "mean_coherence"]:
            for exp_col in ["melting_temp_C", "delta_G_unfolding_kcal_mol"]:
                corr = comparison_data[pred_col].corr(comparison_data[exp_col])
                print(f"{pred_col} vs {exp_col}: r = {corr:.3f}")