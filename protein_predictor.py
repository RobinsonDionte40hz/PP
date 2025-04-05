import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, Selection
from Bio.PDB.vectors import calc_dihedral
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import warnings
from simple_quantum_dssp import SimpleQuantumDSSP  
warnings.filterwarnings('ignore')

class QuantumCoherenceProteinPredictor:
    """
    A protein structure predictor based on quantum coherence principles,
    golden ratio patterns, and resonance dynamics.
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.pipeline = None
        self.structure = None
        self.chain = None
        self.residues = []
        
        # Initialize quantum constants
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
        self.phi_angle = 2 * np.pi / self.phi  # ≈ 137.5 degrees in radians
        self.phi_angle_deg = self.phi_angle * 180 / np.pi
        self.base_energy = 4.0
        self.phi_harmonics = np.array([self.phi ** n for n in range(5)])
        
        # Initialize DSSP calculator
        self.dssp_calculator = SimpleQuantumDSSP()
        
        # Analysis results
        self.qcp_values = None
        self.coherence_metric = None
        self.stability_score = None

        self.plank_reduced = 1.0545718e-34  # Added Planck's constant (ℏ)
        
    def load_protein(self, pdb_file, chain_id='A'):
        """Load protein structure and extract residues."""
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure('protein', pdb_file)
        self.chain = self.structure[0][chain_id]
        
        # Extract valid residues
        self.residues = []
        for res in self.chain:
            if res.id[0] == ' ' and res.has_id('CA'):  # Only standard amino acids with CA
                self.residues.append(res)
        
        # Calculate coords array for later use
        self.coords = np.array([res['CA'].get_coord() for res in self.residues])
        
        # Calculate secondary structure
        try:
            self.ss_assignments = self.dssp_calculator.calculate_secondary_structure(self.structure)
            print(f"Secondary structure calculated with SimpleQuantumDSSP")
        except Exception as e:
            print(f"DSSP calculation failed: {e}")
            print("Using simplified secondary structure")
            self.ss_assignments = None
            
        return self.structure
    
    def _calculate_simplified_ss(self):
        """Fallback simple secondary structure calculation."""
        ss = {}
        for model in self.structure:
            chain_ss = {}
            for chain in model:
                # Create 2D array of coordinates for distance calculation
                chain_coords = []
                for res in chain:
                    if res.has_id('CA'):
                        chain_coords.append(res['CA'].get_coord())
                chain_coords = np.array(chain_coords)
                
                if len(chain_coords) > 0:
                    # Calculate pairwise distances
                    distances = squareform(pdist(chain_coords))
                    
                    # Simple secondary structure assignment based on distances
                    residue_ss = ['C'] * len(chain_coords)  # Default to coil
                    
                    # Look for helical patterns (i,i+3 distances ~5Å)
                    for i in range(len(distances)-3):
                        if 4.5 < distances[i][i+3] < 5.5:
                            residue_ss[i:i+4] = ['H'] * 4
                            
                    chain_ss[chain.id] = ''.join(residue_ss)
                else:
                    chain_ss[chain.id] = ''
                    
            ss[model.id] = chain_ss
        return ss

    def calculate_qcp(self, n_levels=3):
        """
        Calculate Quantum Consciousness Potential (QCP) values for residues.
        QCP = 4 + (2^n × φ^l × m)
        
        Parameters:
        - n_levels: structural hierarchy levels to consider
        
        Returns:
        - DataFrame with QCP values for each residue
        """
        qcp_data = []

        for i, residue in enumerate(self.residues):
            if not residue.has_id('CA'):
                continue
                
            # Get residue information 
            res_id = residue.get_id()[1]
            res_name = residue.get_resname()
            
            # Get secondary structure assignment if available
            if self.ss_assignments is not None:
                try:
                    # Get the chain ID from the residue's parent
                    chain_id = residue.get_parent().id
                    # Get the model ID (usually 0)
                    model_id = 0
                    # Get the secondary structure string for this chain
                    chain_ss = self.ss_assignments[model_id][chain_id]
                    
                    # Find the correct position in the SS string
                    # This may require mapping the residue index to the SS string index
                    ss_idx = self.residues.index(residue)
                    if ss_idx < len(chain_ss):
                        ss_type = chain_ss[ss_idx]
                        
                        # Assign n based on structural hierarchy
                        if ss_type in ['H', 'G', 'I']:  # Helices
                            n = 1
                        elif ss_type in ['E', 'B']:  # Sheets
                            n = 2
                        elif ss_type in ['S']:  # Phi-based bends (our special case)
                            n = 3  # Highest quantum level for phi-based structures
                        elif ss_type in ['T']:  # Regular turns
                            n = 0
                        else:  # Coil
                            n = 0
                    else:
                        # Default if index is out of range
                        n = self._approximate_secondary_structure(i)
                except (KeyError, IndexError, ValueError):
                    # Fallback to approximation
                    n = self._approximate_secondary_structure(i)
            else:
                # Use approximation if no DSSP results
                n = self._approximate_secondary_structure(i)
            
            # Calculate l based on local geometry (bond network)
            # Simple approximation: count neighbor residues within 8Å
            neighbors = 0
            ca_pos = residue['CA'].get_coord()
            for other in self.residues:
                if other.get_id()[1] != res_id and other.has_id('CA'):
                    other_pos = other['CA'].get_coord()
                    dist = np.linalg.norm(ca_pos - other_pos)
                    if dist < 8.0:
                        neighbors += 1
            
            l = min(max(1, neighbors // 3), 3)  # Scale to 1-3
            
            # Calculate m based on environmental factors
            # Simple approximation: hydrophobicity index (Kyte-Doolittle scale)
            hydrophobicity = {
                'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
                'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
                'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
                'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
            }
            m = (hydrophobicity.get(res_name, 0) + 4.5) / 9.0  # Normalize to 0-1
            m = m * 2 - 1  # Scale to -1 to 1
            
            # Calculate QCP
            qcp = self.base_energy + (2**n * (self.phi**l) * m)
            
            qcp_data.append({
                'residue_id': res_id,
                'residue_name': res_name,
                'n': n,
                'l': l,
                'm': m,
                'qcp': qcp
            })
        
        self.qcp_values = pd.DataFrame(qcp_data)
        return self.qcp_values
    
    def calculate_field_coherence(self):
        """
        Calculate field coherence (C) for the protein.
        C = ∑(ψᵢ × e^(iφ)) × D(t)
        
        Simple approximation using spatial arrangement of residues.
        
        Returns:
        - Coherence score for each residue
        """
        if self.qcp_values is None:
            self.calculate_qcp()
            
        coherence_data = []
        ca_coords = []
        
        # Extract CA coordinates for all residues
        for res_idx, residue in enumerate(self.residues):
            if not residue.has_id('CA'):
                continue
            ca_coords.append(residue['CA'].get_coord())
        
        ca_coords = np.array(ca_coords)
        
        # Calculate pairwise distances
        dist_matrix = squareform(pdist(ca_coords))
        
        # Calculate field coherence for each residue
        for i, (coords, row) in enumerate(zip(ca_coords, dist_matrix)):
            # Skip if no QCP value for this residue
            if i >= len(self.qcp_values):
                continue
                
            # Get QCP for this residue
            qcp = self.qcp_values.iloc[i]['qcp']
            
            # Calculate coherence based on neighboring residues
            coherence = 0
            count = 0
            
            for j, dist in enumerate(row):
                if i != j and dist < 10.0:  # Consider neighbors within 10Å
                    if j < len(self.qcp_values):
                        neighbor_qcp = self.qcp_values.iloc[j]['qcp']
                        
                        # Phi-based phase factor (based on distance relative to phi)
                        phase = np.cos(2 * np.pi * dist / (3.8 * self.phi))
                        
                        # Resonance coupling strength
                        energy_diff = abs(qcp - neighbor_qcp)
                        resonance = np.exp(-(energy_diff - 0.1)**2 / (2 * 0.1))
                        
                        # Decoherence factor (simplistic model based on distance)
                        decoherence = np.exp(-dist / 10.0)
                        
                        # Contribution to coherence
                        coherence += phase * resonance * decoherence
                        count += 1
            
            # Normalize by number of neighbors
            if count > 0:
                coherence /= count
                
            # Store coherence data
            coherence_data.append({
                'residue_id': self.qcp_values.iloc[i]['residue_id'],
                'coherence': coherence
            })
        
        self.coherence_metric = pd.DataFrame(coherence_data)
        return self.coherence_metric
    
    def predict_stability(self):
        """
        Predict protein stability based on QCP and coherence metrics.
        Returns a stability score (higher = more stable).
        """
        if self.coherence_metric is None:
            self.calculate_field_coherence()
            
        # Calculate overall stability score
        mean_qcp = self.qcp_values['qcp'].mean()
        mean_coherence = self.coherence_metric['coherence'].mean()
        
        # Get phi-based structural features
        phi_angles = self.analyze_phi_angles()
        phi_score = phi_angles['phi_match_score'].mean() if len(phi_angles) > 0 else 0
        
        # Combine metrics
        stability = (0.4 * mean_qcp) + (0.4 * mean_coherence) + (0.2 * phi_score)
        
        self.stability_score = stability
        print(f"Predicted stability score: {stability:.2f}")
        return stability
    
    def analyze_phi_angles(self):
        """
        Analyze protein for golden ratio based angles.
        Looks for angles close to 137.5° and 222.5° (phi-based angles).
        
        Returns:
        - DataFrame with phi-based angle analysis
        """
        phi_data = []
        
        # Get alpha carbon coordinates
        ca_atoms = []
        for res in self.residues:
            if res.has_id('CA'):
                ca_atoms.append(res['CA'])
        
        # Check for triplets of residues that form phi-based angles
        target_angles = [self.phi_angle_deg, 360 - self.phi_angle_deg]  # 137.5° and 222.5°
        
        for i in range(len(ca_atoms) - 2):
            for j in range(i + 1, len(ca_atoms) - 1):
                for k in range(j + 1, len(ca_atoms)):
                    # Skip if residues are too close in sequence
                    if abs(ca_atoms[j].get_parent().get_id()[1] - ca_atoms[i].get_parent().get_id()[1]) < 3:
                        continue
                    if abs(ca_atoms[k].get_parent().get_id()[1] - ca_atoms[j].get_parent().get_id()[1]) < 3:
                        continue
                        
                    # Calculate angle between three CA atoms
                    v1 = ca_atoms[i].get_coord() - ca_atoms[j].get_coord()
                    v2 = ca_atoms[k].get_coord() - ca_atoms[j].get_coord()
                    
                    # Normalize vectors
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    
                    # Calculate angle in degrees
                    dot_product = np.dot(v1, v2)
                    dot_product = max(min(dot_product, 1.0), -1.0)  # Ensure within valid range
                    angle_rad = np.arccos(dot_product)
                    angle_deg = angle_rad * 180 / np.pi
                    
                    # Check if angle is close to target phi-based angles
                    best_match = min(abs(angle_deg - target_angles[0]), abs(angle_deg - target_angles[1]))
                    match_score = np.exp(-0.01 * best_match**2)  # Score based on closeness to target
                    
                    if match_score > 0.5:  # Only record significant matches
                        phi_data.append({
                            'residue_i': ca_atoms[i].get_parent().get_id()[1],
                            'residue_j': ca_atoms[j].get_parent().get_id()[1],
                            'residue_k': ca_atoms[k].get_parent().get_id()[1],
                            'angle': angle_deg,
                            'phi_match_score': match_score
                        })
        
        return pd.DataFrame(phi_data)
        
    def predict_thz_spectrum(self, n_modes=10):
        """
        Predict THz spectrum based on coherence patterns.
        This is a simplified model for demonstration.
        
        Returns:
        - DataFrame with predicted THz peaks
        """
        if self.qcp_values is None:
            self.calculate_qcp()
            
        # Get residue coordinates
        coords = []
        for res in self.residues:
            if res.has_id('CA'):
                coords.append(res['CA'].get_coord())
        
        coords = np.array(coords)
        
        # Simplified normal mode calculation (very approximate)
        # In a real implementation, this would use proper normal mode analysis
        thz_peaks = []
        
        # Base peak at 1 THz
        base_peak = 1.0  # THz
        
        # Generate phi-based harmonics
        for n in range(n_modes):
            # Frequency follows phi-based series
            freq = base_peak * (self.phi ** (n/2))
            
            # Intensity based on QCP distribution
            if n < len(self.qcp_values):
                intensity = np.mean(self.qcp_values['qcp'].values[:n+1]) / 10
            else:
                intensity = np.mean(self.qcp_values['qcp'].values) / 10
                
            # Dampening based on frequency
            damping = np.exp(-n/5)
            
            # Add some noise
            freq_noise = freq * (1 + np.random.normal(0, 0.01))
            intensity_noise = intensity * (1 + np.random.normal(0, 0.05))
            
            thz_peaks.append({
                'frequency': freq_noise,
                'intensity': intensity_noise * damping,
                'mode': n
            })
        
        return pd.DataFrame(thz_peaks)
    
    def resonance_coupling(self, energy1, energy2, gamma_freq=40):
        """
        Calculate resonance coupling strength between two energy states.
        R(E₁,E₂) = exp[-(E₁ - E₂ - ℏω_γ)²/(2ℏω_γ)]
        
        Parameters:
        - energy1, energy2: Energy states to compare
        - gamma_freq: Frequency of gamma oscillation (Hz)
        
        Returns:
        - Coupling strength (0-1)
        """
        # Convert gamma frequency to energy
        h_gamma = self.plank_reduced * gamma_freq
        
        # Calculate energy difference
        energy_diff = abs(energy1 - energy2)
        
        # Calculate coupling strength
        coupling = np.exp(-(energy_diff - h_gamma)**2 / (2 * h_gamma))
        
        return coupling
    
    def visualize_results(self):
        """Generate visualizations of the analysis results."""
        if self.qcp_values is None or self.coherence_metric is None:
            print("Please run calculations first")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot QCP values
        axes[0, 0].plot(self.qcp_values['residue_id'], self.qcp_values['qcp'], 'b-')
        axes[0, 0].set_title('Quantum Consciousness Potential (QCP)')
        axes[0, 0].set_xlabel('Residue ID')
        axes[0, 0].set_ylabel('QCP Value')
        
        # Plot coherence values
        axes[0, 1].plot(self.coherence_metric['residue_id'], self.coherence_metric['coherence'], 'r-')
        axes[0, 1].set_title('Field Coherence Metric')
        axes[0, 1].set_xlabel('Residue ID')
        axes[0, 1].set_ylabel('Coherence Value')
        
        # Plot predicted THz spectrum
        thz_data = self.predict_thz_spectrum()
        axes[1, 0].bar(thz_data['frequency'], thz_data['intensity'], width=0.05)
        axes[1, 0].set_title('Predicted THz Spectrum')
        axes[1, 0].set_xlabel('Frequency (THz)')
        axes[1, 0].set_ylabel('Intensity')
        
        # Mark phi-based frequencies
        for i, freq in enumerate(self.phi_harmonics[:3]):
            axes[1, 0].axvline(x=freq, color='g', linestyle='--', 
                              label=f'φ^{i} harmonic' if i==0 else None)
        
        axes[1, 0].legend()
        
        # Plot phi angle distribution
        phi_data = self.analyze_phi_angles()
        if len(phi_data) > 0:
            axes[1, 1].hist(phi_data['angle'], bins=36, range=(0, 360))
            axes[1, 1].set_title('Distribution of Inter-Residue Angles')
            axes[1, 1].set_xlabel('Angle (degrees)')
            axes[1, 1].set_ylabel('Count')
            
            # Mark phi-based angles
            axes[1, 1].axvline(x=self.phi_angle_deg, color='r', linestyle='--', 
                              label=f'φ angle (137.5°)')
            axes[1, 1].axvline(x=360-self.phi_angle_deg, color='g', linestyle='--', 
                              label=f'φ complement (222.5°)')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No phi angle data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            
        plt.tight_layout()
        return fig
    
    def run_full_analysis(self, pdb_file, chain_id='A'):
        """Run a complete analysis on a protein structure."""
        self.load_protein(pdb_file, chain_id)
        self.calculate_qcp()
        self.calculate_field_coherence()
        stability = self.predict_stability()
        fig = self.visualize_results()
        
        return {
            'stability': stability,
            'qcp_values': self.qcp_values,
            'coherence': self.coherence_metric,
            'figure': fig
        }


# Example usage
if __name__ == "__main__":
    predictor = QuantumCoherenceProteinPredictor()
    
    # Load a sample protein (would need a PDB file)
    # results = predictor.run_full_analysis("1ubq.pdb")  # Ubiquitin
    
    # For demonstration, create synthetic data
    print("Generating synthetic protein data for demonstration...")
    
    # Generate synthetic residue data
    n_residues = 100
    predictor.residues = [type('obj', (object,), {
        'get_id': lambda self=None: ('', i, ''),
        'get_resname': lambda self=None: np.random.choice(['ALA', 'LEU', 'VAL', 'ILE', 'PRO', 'PHE', 'TRP', 'MET']),
        'has_id': lambda x, self=None: True,
        '__getitem__': lambda x, self=None: type('atom', (object,), {
            'get_coord': lambda: np.random.randn(3) * 5,
            'get_vector': lambda: type('vector', (object,), {
                'norm': lambda: np.random.uniform(3.5, 4.0)
            }),
            'get_parent': lambda: type('obj', (object,), {'get_id': lambda: ('', i, '')})
        })
    }) for i in range(1, n_residues+1)]
    
    # Run calculations on synthetic data
    predictor.calculate_qcp()
    predictor.calculate_field_coherence()
    predictor.predict_stability()
    
    # Visualize results
    fig = predictor.visualize_results()
    plt.show()