import numpy as np

class SimpleQuantumDSSP:
    """
    A simplified DSSP implementation based on quantum coherence and phi-based patterns.
    
    This class identifies secondary structure elements (helices, sheets, turns) 
    based on geometric relationships that incorporate the golden ratio.
    """
    
    def __init__(self):
        """Initialize with quantum coherence parameters."""
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
        self.phi_angle_rad = 2 * np.pi / self.phi  # ≈ 137.5 degrees in radians
        self.phi_angle_deg = self.phi_angle_rad * 180 / np.pi
        
        # Secondary structure codes
        self.SS_CODES = {
            'H': 'alpha helix',
            'G': '3-10 helix',
            'I': 'pi helix',
            'E': 'extended strand',
            'B': 'isolated bridge',
            'T': 'turn',
            'S': 'bend',
            'C': 'coil'
        }
        
        # Phi-based distance thresholds
        self.helix_dist = 3.8  # Å (typical alpha-helix CA-CA distance)
        self.sheet_dist = self.helix_dist * self.phi  # ≈ 6.15 Å
        
    def calculate_secondary_structure(self, structure):
        """
        Calculate secondary structure for all chains in the structure.
        
        Parameters:
        -----------
        structure : Bio.PDB.Structure
            The protein structure to analyze
            
        Returns:
        --------
        dict : Secondary structure assignments for all residues
        """
        ss_assignments = {}
        
        for model in structure:
            model_ss = {}
            
            for chain in model:
                chain_ss = self._analyze_chain(chain)
                model_ss[chain.id] = chain_ss
                
            ss_assignments[model.id] = model_ss
            
        return ss_assignments
    
    def _analyze_chain(self, chain):
        """Analyze a single chain to determine secondary structure."""
        # Get CA atoms and coordinates
        ca_atoms = []
        residues = []
        
        for res in chain:
            if res.id[0] == ' ' and res.has_id('CA'):  # Standard residue with CA
                ca_atoms.append(res['CA'])
                residues.append(res)
        
        if len(ca_atoms) < 4:  # Need at least 4 residues for analysis
            return 'C' * len(ca_atoms)  # All coil
            
        # Get coordinates
        coords = np.array([atom.get_coord() for atom in ca_atoms])
        
        # Calculate distances between CA atoms
        n_res = len(coords)
        distances = np.zeros((n_res, n_res))
        
        for i in range(n_res):
            for j in range(i+1, n_res):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = distances[j, i] = dist
        
        # Calculate dihedral angles for groups of 4 consecutive CAs
        dihedrals = []
        for i in range(n_res - 3):
            atoms = ca_atoms[i:i+4]
            dihedral = self._calculate_dihedral(atoms)
            dihedrals.append(dihedral)
            
        # Initialize all residues as coil
        ss = ['C'] * n_res
        
        # Identify helices based on distances and dihedrals
        self._identify_helices(ss, distances, dihedrals)
        
        # Identify strands based on distances
        self._identify_strands(ss, distances, coords)
        
        # Identify turns and bends
        self._identify_turns(ss, distances, coords)
        
        # Apply phi-based refinement
        self._apply_phi_refinement(ss, distances, coords)
        
        return ''.join(ss)
    
    def _calculate_dihedral(self, atoms):
        """Calculate dihedral angle between 4 atoms."""
        coords = [atom.get_coord() for atom in atoms]
        vectors = [coords[i+1] - coords[i] for i in range(3)]
        
        # Calculate normal vectors to the planes
        normal1 = np.cross(vectors[0], vectors[1])
        normal2 = np.cross(vectors[1], vectors[2])
        
        # Normalize
        normal1 = normal1 / np.linalg.norm(normal1)
        normal2 = normal2 / np.linalg.norm(normal2)
        
        # Calculate angle
        cos_angle = np.dot(normal1, normal2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        
        angle = np.arccos(cos_angle)
        
        # Determine sign
        if np.dot(np.cross(normal1, normal2), vectors[1]) < 0:
            angle = -angle
            
        return angle * 180 / np.pi  # Convert to degrees
    
    def _identify_helices(self, ss, distances, dihedrals):
        """Identify alpha helices, 3-10 helices, and pi helices."""
        n_res = len(ss)
        
        # Alpha helices: i->i+4 distance ~ 6.2-6.5Å, dihedral ~ -50°
        for i in range(n_res - 4):
            # Check multiple criteria
            dist_i_i4 = distances[i, i+4]
            if 5.8 < dist_i_i4 < 6.8:  # CA(i) - CA(i+4) distance for alpha helix
                if i < len(dihedrals) and -65 < dihedrals[i] < -35:
                    # Mark potential helix
                    ss[i:i+5] = ['H'] * 5
        
        # 3-10 helices: i->i+3 distance ~ 5.8Å, tighter dihedral
        for i in range(n_res - 3):
            dist_i_i3 = distances[i, i+3]
            if 5.5 < dist_i_i3 < 6.1 and ss[i] != 'H':  # Not already in alpha helix
                if i < len(dihedrals) and -75 < dihedrals[i] < -45:
                    ss[i:i+4] = ['G'] * 4
        
        # Pi helices: i->i+5 distance ~ 7.0Å
        for i in range(n_res - 5):
            dist_i_i5 = distances[i, i+5]
            if 6.8 < dist_i_i5 < 7.2 and ss[i] != 'H' and ss[i] != 'G':
                ss[i:i+6] = ['I'] * 6
    
    def _identify_strands(self, ss, distances, coords):
        """Identify extended strands and isolated bridges."""
        n_res = len(ss)
        
        # Look for parallel/anti-parallel patterns
        for i in range(n_res - 2):
            if ss[i] != 'C':  # Skip if already assigned
                continue
                
            # Look for extended conformations (stretched)
            extended = True
            for j in range(i, min(i+3, n_res-1)):
                if j > 0:
                    dist = distances[j-1, j]
                    if dist < 3.7 or dist > 4.0:  # Expected CA-CA distance in strand
                        extended = False
                        break
            
            if extended:
                # Look for partners in beta sheets
                has_partner = False
                for j in range(n_res):
                    if abs(i - j) > 5:  # Not too close in sequence
                        dist = distances[i, j]
                        if 4.5 < dist < 5.5:  # Typical cross-strand distance
                            # Check if approximately parallel
                            if i+2 < n_res and j+2 < n_res:
                                v1 = coords[i+2] - coords[i]
                                v2 = coords[j+2] - coords[j]
                                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                                
                                # Parallel or anti-parallel
                                if angle < 30 or angle > 150:
                                    has_partner = True
                                    break
                
                # Assign based on findings
                if has_partner:
                    ss[i:i+3] = ['E'] * 3  # Extended strand in sheet
                else:
                    ss[i:i+3] = ['B'] * 3  # Isolated bridge
    
    def _identify_turns(self, ss, distances, coords):
        """Identify turns and bends."""
        n_res = len(ss)
        
        # Turns involve 3-4 residues making a directional change
        for i in range(n_res - 3):
            if ss[i] != 'C' and ss[i+3] != 'C':  # Skip if not coil
                continue
                
            # Calculate angle between vectors
            if i+3 < n_res:
                v1 = coords[i+1] - coords[i]
                v2 = coords[i+3] - coords[i+2]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                
                # Sharp turn
                if angle > 90:
                    ss[i+1:i+3] = ['T'] * 2  # Mark as turn
    
    def _apply_phi_refinement(self, ss, distances, coords):
        """Apply phi-based refinement to secondary structure assignments."""
        n_res = len(ss)
        
        # Look for phi-angle relationships (golden angle ~ 137.5°)
        for i in range(n_res - 2):
            for j in range(i+1, n_res-1):
                for k in range(j+1, n_res):
                    # Skip if sequence-adjacent
                    if j-i < 3 or k-j < 3:
                        continue
                        
                    # Calculate angle between three CA atoms
                    v1 = coords[i] - coords[j]
                    v2 = coords[k] - coords[j]
                    
                    # Normalize vectors
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    
                    # Calculate angle in degrees
                    cos_angle = np.dot(v1, v2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure within valid range
                    angle_deg = np.arccos(cos_angle) * 180 / np.pi
                    
                    # Check if angle is close to golden angle
                    phi_angle_match = min(abs(angle_deg - self.phi_angle_deg), 
                                        abs(angle_deg - (360 - self.phi_angle_deg)))
                    
                    # If matches phi angle (within 10°) and in coil, mark as special 'S' (bend)
                    if phi_angle_match < 10 and ss[j] == 'C':
                        ss[j] = 'S'  # Mark as phi-based bend
        
        # Enhanced stability for phi-adjacent structure elements
        for i in range(n_res):
            if ss[i] == 'S':  # Phi-based bend
                # Stabilize adjacent structural elements
                for offset in [-3, -2, -1, 1, 2, 3]:
                    idx = i + offset
                    if 0 <= idx < n_res:
                        if ss[idx] in ['H', 'E']:  # Helix or sheet
                            # This would be where we apply quantum coherence enhancement
                            # For now, we just maintain the structure type
                            pass