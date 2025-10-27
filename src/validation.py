import pandas as pd
import numpy as np
from scipy import stats

def normalize_scores(df):
    """Normalize stability-related scores"""
    score_columns = ['stability_score', 'melting_temp_C', 'delta_G_unfolding_kcal_mol']
    
    for col in score_columns:
        if col in df.columns:
            df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    return df

def calculate_correlations(df):
    """Calculate correlations between predicted and experimental values"""
    corr_data = {}
    
    if all(col in df.columns for col in ['stability_score_norm', 'melting_temp_C_norm']):
        r, p = stats.pearsonr(df['stability_score_norm'], df['melting_temp_C_norm'])
        corr_data['melting_temp'] = {'r': r, 'p': p}
    
    if all(col in df.columns for col in ['stability_score_norm', 'delta_G_unfolding_kcal_mol_norm']):
        r, p = stats.pearsonr(df['stability_score_norm'], df['delta_G_unfolding_kcal_mol_norm'])
        corr_data['delta_G'] = {'r': r, 'p': p}
    
    return corr_data
