import numpy as np
import pandas as pd
from .quantum_utils import resonance_coupling, calculate_phi_distance_match, normalize_spectrum

class StabilityPredictor:
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    def calculate_field_coherence(self):
        ca_coords = self.get_ca_coordinates()
        dist_matrix = self.calculate_distance_matrix(ca_coords)
        coherence_values = []
        
        for i, (coords, row) in enumerate(zip(ca_coords, dist_matrix)):
            qcp = self.qcp_values.iloc[i]['qcp']
            coherence = 0
            count = 0
            
            for j, dist in enumerate(row):
                if i != j and dist < 10.0:
                    if j < len(self.qcp_values):
                        neighbor_qcp = self.qcp_values.iloc[j]['qcp']
                        phi_match = calculate_phi_distance_match(dist, self.phi)
                        resonance = resonance_coupling(qcp, neighbor_qcp)
                        contribution = phi_match * resonance * np.exp(-dist / 15.0)
                        coherence += contribution
                        count += 1
            
            coherence_values.append(coherence / max(count, 1))
        
        self.coherence_metric = pd.DataFrame({
            'residue': range(len(coherence_values)),
            'coherence': coherence_values
        })

    def predict_stability(self):
        mean_qcp = self.qcp_values['qcp'].mean()
        coherence_values = self.coherence_metric['coherence'].values
        adjusted_coherence = coherence_values - np.min(coherence_values)
        mean_coherence = np.mean(adjusted_coherence)
        
        phi_angles = self.analyze_phi_angles()
        phi_score = phi_angles['phi_match_score'].mean() if len(phi_angles) > 0 else 0
        
        thz_spectrum = self.predict_thz_spectrum()
        harmonic_strength = self.calculate_harmonic_strength(thz_spectrum)
        
        stability = (0.25 * mean_qcp + 
                    0.35 * mean_coherence + 
                    0.15 * phi_score + 
                    0.25 * harmonic_strength)
        
        self.stability_score = stability
        return stability

    def calculate_harmonic_strength(self, thz_spectrum):
        phi_harmonics = [self.phi ** i for i in range(3)]
        harmonic_strength = 0
        
        for harmonic in phi_harmonics:
            closest_peak = thz_spectrum.iloc[
                (thz_spectrum['frequency'] - harmonic).abs().argsort()[:1]
            ]
            match_quality = np.exp(-((closest_peak['frequency'].values[0] - harmonic) ** 2) / 0.1)
            harmonic_strength += match_quality * closest_peak['intensity'].values[0]
        
        return harmonic_strength / len(phi_harmonics)
