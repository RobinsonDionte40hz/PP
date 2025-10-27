import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from .quantum_utils import resonance_coupling, calculate_phi_distance_match, calculate_angle_match

class StabilityCalculator:
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.base_energy = 4.0
        self.plank_reduced = 6.626e-34 / (2 * np.pi)
        
    def calculate_coherence(self, coords, qcp_values):
        """Calculate quantum coherence for given coordinates"""
        dist_matrix = squareform(pdist(coords))
        coherence_data = []
        
        for i, row in enumerate(dist_matrix):
            qcp = qcp_values[i]
            coherence = self._calculate_residue_coherence(i, row, qcp, qcp_values, coords)
            coherence_data.append(coherence)
            
        return pd.DataFrame(coherence_data)
    
    def _calculate_residue_coherence(self, idx, distances, qcp, qcp_values, coords):
        """Calculate coherence for a single residue"""
        coherence = 0
        count = 0
        
        for j, dist in enumerate(distances):
            if idx != j and dist < 12.0:
                phi_match = calculate_phi_distance_match(dist, self.phi)
                res = resonance_coupling(qcp, qcp_values[j])
                
                # Calculate angle match if possible
                angle_match = self._calculate_angle_match(idx, j, coords)
                
                contribution = (0.4 * phi_match + 0.3 * res + 0.3 * angle_match) * np.exp(-dist / 15.0)
                coherence += contribution
                count += 1
                
        return coherence / max(count, 1)
    
    def _calculate_angle_match(self, i, j, coords):
        """Calculate phi-based angle matching"""
        if i+1 < len(coords) and j+1 < len(coords):
            v1 = coords[i+1] - coords[i]
            v2 = coords[j+1] - coords[j]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return calculate_angle_match(angle, self.phi)
        return 0
