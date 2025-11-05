"""
Quick THz Analysis Test

Tests the vibrational analysis and signature matching on a small protein.
"""

import sys
import os

# Add ubf_protein directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
ubf_dir = os.path.dirname(script_dir)
sys.path.insert(0, ubf_dir)

from vibrational_analysis import create_vibrational_analyzer
from signature_analysis import create_signature_matcher, create_determinism_tester


def test_vibrational_analysis():
    """Test basic vibrational analysis."""
    print("=" * 70)
    print("TEST 1: Vibrational Analysis")
    print("=" * 70)
    
    # Create a simple test structure (small helix)
    ca_coords = [
        (0.0, 0.0, 0.0),
        (3.8, 0.0, 0.0),
        (7.6, 0.0, 0.0),
        (11.4, 0.0, 0.0),
        (15.2, 0.0, 0.0),
    ]
    
    analyzer = create_vibrational_analyzer()
    spectrum = analyzer.calculate_spectrum(ca_coords, n_modes=10)
    
    print(f"\n✅ Calculated THz spectrum for {len(ca_coords)} residues")
    print(f"   Number of modes: {len(spectrum.modes)}")
    print(f"\n   Top vibrational modes:")
    for i, mode in enumerate(spectrum.modes[:5]):
        print(f"   {i+1}. {mode.frequency_thz:.3f} THz (intensity={mode.intensity:.3f})")
    
    return spectrum


def test_signature_matching():
    """Test signature matching between two similar structures."""
    print("\n" + "=" * 70)
    print("TEST 2: Signature Matching")
    print("=" * 70)
    
    # Structure 1: Linear chain
    coords1 = [(i * 3.8, 0.0, 0.0) for i in range(7)]
    
    # Structure 2: Slightly perturbed chain
    coords2 = [(i * 3.8 + 0.1, 0.1, 0.0) for i in range(7)]
    
    analyzer = create_vibrational_analyzer()
    spectrum1 = analyzer.calculate_spectrum(coords1, n_modes=10)
    spectrum2 = analyzer.calculate_spectrum(coords2, n_modes=10)
    
    # Match signatures
    matcher = create_signature_matcher()
    match = matcher.match_signatures(
        spectrum1.frequencies, spectrum1.intensities,
        spectrum2.frequencies, spectrum2.intensities
    )
    
    print(f"\n✅ Matched two similar structures:")
    print(f"   Similarity score: {match.similarity_score:.3f}")
    print(f"   Frequency correlation: {match.frequency_correlation:.3f}")
    print(f"   Intensity correlation: {match.intensity_correlation:.3f}")
    print(f"   Matched peaks: {match.matched_peaks}/{match.total_peaks}")
    
    return match


def test_determinism_scoring():
    """Test determinism scoring with multiple trials."""
    print("\n" + "=" * 70)
    print("TEST 3: Determinism Scoring")
    print("=" * 70)
    
    # Simulate 10 trials with convergent results
    print("\n   Simulating 10 folding trials...")
    
    analyzer = create_vibrational_analyzer()
    all_frequencies = []
    all_intensities = []
    
    # 7 trials converge to similar structure
    base_coords = [(i * 3.8, 0.0, 0.0) for i in range(8)]
    for trial in range(7):
        # Add small random perturbations
        coords = [(x + trial * 0.05, y, z) for x, y, z in base_coords]
        spectrum = analyzer.calculate_spectrum(coords, n_modes=10)
        all_frequencies.append(spectrum.frequencies)
        all_intensities.append(spectrum.intensities)
    
    # 3 trials converge to different structure
    alt_coords = [(i * 4.0, 0.2, 0.0) for i in range(8)]
    for trial in range(3):
        coords = [(x + trial * 0.05, y, z) for x, y, z in alt_coords]
        spectrum = analyzer.calculate_spectrum(coords, n_modes=10)
        all_frequencies.append(spectrum.frequencies)
        all_intensities.append(spectrum.intensities)
    
    # Calculate determinism score
    tester = create_determinism_tester(similarity_threshold=0.7)
    score = tester.calculate_determinism_score(all_frequencies, all_intensities)
    
    print(f"\n✅ Determinism analysis:")
    print(f"   Trials: {score.n_trials}")
    print(f"   Clusters found: {score.n_clusters}")
    print(f"   Largest cluster: {score.largest_cluster_size} trials")
    print(f"   Convergence ratio: {score.convergence_ratio:.1%}")
    print(f"   Determinism score: {score.determinism_score:.3f}")
    print(f"\n   Interpretation: {score.interpret()}")
    
    return score


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("THz VIBRATIONAL ANALYSIS - QUICK TEST")
    print("=" * 70)
    
    try:
        # Test 1: Basic vibrational analysis
        spectrum = test_vibrational_analysis()
        
        # Test 2: Signature matching
        match = test_signature_matching()
        
        # Test 3: Determinism scoring
        score = test_determinism_scoring()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   - Vibrational modes calculated: {len(spectrum.modes)}")
        print(f"   - Signature similarity: {match.similarity_score:.3f}")
        print(f"   - Determinism score: {score.determinism_score:.3f}")
        print(f"\n🎯 Next step: Run full determinism test on real proteins")
        print(f"   python ubf_protein/experiments/test_folding_determinism.py --sequence MQIFVKTLTG --trials 100")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
