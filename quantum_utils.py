import numpy as np

def resonance_coupling(qcp1, qcp2, gamma_freq=40):
    """Calculate resonance coupling between two QCP values"""
    # Convert gamma frequency to energy equivalent
    h_gamma = 6.626e-34 * gamma_freq
    
    # Calculate energy difference
    energy_diff = abs(qcp1 - qcp2)
    target_diff = h_gamma
    
    return np.exp(-((energy_diff - target_diff)**2) / (2 * target_diff))

def calculate_phi_distance_match(dist, phi, tolerance=0.5):
    """Enhanced phi-based distance matching"""
    base_distance = 3.8  # Å
    phi_distances = [base_distance, base_distance * phi, base_distance * phi * phi]
    
    match_score = 0
    for phi_dist in phi_distances:
        match_score += np.exp(-((dist - phi_dist) ** 2) / (2 * tolerance))
    
    return match_score / len(phi_distances)

def normalize_spectrum(frequencies, intensities, baseline=0.1):
    """Normalize THz spectrum intensities with baseline"""
    norm_intensities = (intensities - np.min(intensities))
    if np.max(intensities) > np.min(intensities):
        norm_intensities = norm_intensities / (np.max(intensities) - np.min(intensities))
    norm_intensities = baseline + (1 - baseline) * norm_intensities
    return frequencies, norm_intensities

def calculate_angle_match(angle_rad, phi):
    """Calculate how well an angle matches phi-based angles"""
    phi_angle = 2 * np.pi / phi
    angle_match = np.exp(-((angle_rad - phi_angle) ** 2) / (2 * 0.3))
    return angle_match
