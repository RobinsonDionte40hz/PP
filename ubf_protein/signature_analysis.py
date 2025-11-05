"""
THz Signature Analysis - Folding Pathway Fingerprints

This module provides signature matching and clustering for THz spectra,
enabling determinism testing by comparing vibrational fingerprints across
multiple folding trials.

The key hypothesis: If folding is deterministic, all trials should converge
to the same THz signature at energy minima.
"""

from dataclasses import dataclass, replace
from typing import List, Tuple, Dict, Optional
import math


@dataclass(frozen=True)
class SignatureMatch:
    """Result of matching two THz signatures."""
    similarity_score: float  # 0-1, higher = more similar
    frequency_correlation: float  # How well frequencies align
    intensity_correlation: float  # How well intensities match
    matched_peaks: int  # Number of peaks successfully matched
    total_peaks: int  # Total peaks considered
    
    @property
    def match_percentage(self) -> float:
        """Percentage of peaks matched."""
        if self.total_peaks == 0:
            return 0.0
        return 100.0 * self.matched_peaks / self.total_peaks
    
    def __repr__(self) -> str:
        return (f"SignatureMatch(similarity={self.similarity_score:.3f}, "
                f"freq_corr={self.frequency_correlation:.3f}, "
                f"matched={self.matched_peaks}/{self.total_peaks})")


@dataclass(frozen=True)
class SignatureCluster:
    """A cluster of similar THz signatures."""
    cluster_id: int
    signatures: Tuple[int, ...]  # Indices of signatures in this cluster
    centroid_frequencies: Tuple[float, ...]  # Representative frequencies
    centroid_intensities: Tuple[float, ...]  # Representative intensities
    intra_cluster_similarity: float  # Average similarity within cluster
    
    @property
    def size(self) -> int:
        """Number of signatures in cluster."""
        return len(self.signatures)
    
    def __repr__(self) -> str:
        return (f"Cluster {self.cluster_id}: {self.size} signatures "
                f"(similarity={self.intra_cluster_similarity:.3f})")


@dataclass(frozen=True)
class DeterminismScore:
    """Quantifies determinism based on signature clustering."""
    n_trials: int
    n_clusters: int
    largest_cluster_size: int
    convergence_ratio: float  # Fraction in largest cluster
    average_intra_cluster_similarity: float
    determinism_score: float  # 0-1, higher = more deterministic
    
    def interpret(self) -> str:
        """Human-readable interpretation."""
        if self.determinism_score > 0.8:
            return "STRONG DETERMINISM: Folding pathway is highly deterministic"
        elif self.determinism_score > 0.6:
            return "MODERATE DETERMINISM: Multiple pathways converge to similar signatures"
        elif self.determinism_score > 0.4:
            return "WEAK DETERMINISM: Significant pathway diversity"
        else:
            return "STOCHASTIC: Highly variable folding pathways"
    
    def __repr__(self) -> str:
        return (f"DeterminismScore({self.determinism_score:.3f}): "
                f"{self.n_clusters} clusters from {self.n_trials} trials\n"
                f"{self.interpret()}")


class THzSignatureMatcher:
    """
    Match and compare THz signatures from different folding trials.
    
    Uses frequency-based alignment to account for:
    - Missing peaks (some modes may not appear in all conformations)
    - Frequency shifts (small variations due to structural differences)
    - Intensity variations (different mode amplitudes)
    """
    
    def __init__(self, frequency_tolerance: float = 0.5, intensity_weight: float = 0.3):
        """
        Initialize signature matcher.
        
        Args:
            frequency_tolerance: Max frequency difference for peak matching (THz)
            intensity_weight: Weight for intensity in similarity (0-1)
        """
        self.frequency_tolerance = frequency_tolerance
        self.intensity_weight = intensity_weight
        self.frequency_weight = 1.0 - intensity_weight
    
    def match_signatures(
        self,
        sig1_frequencies: List[float],
        sig1_intensities: List[float],
        sig2_frequencies: List[float],
        sig2_intensities: List[float]
    ) -> SignatureMatch:
        """
        Calculate similarity between two THz signatures.
        
        Algorithm:
        1. For each peak in sig1, find closest peak in sig2
        2. If within tolerance, calculate similarity
        3. Combine frequency and intensity matching
        
        Args:
            sig1_frequencies: Frequencies of first signature (THz)
            sig1_intensities: Intensities of first signature
            sig2_frequencies: Frequencies of second signature (THz)
            sig2_intensities: Intensities of second signature
            
        Returns:
            SignatureMatch with detailed comparison
        """
        if not sig1_frequencies or not sig2_frequencies:
            return SignatureMatch(
                similarity_score=0.0,
                frequency_correlation=0.0,
                intensity_correlation=0.0,
                matched_peaks=0,
                total_peaks=len(sig1_frequencies)
            )
        
        # Match peaks bidirectionally
        matches_1to2 = self._find_peak_matches(
            sig1_frequencies, sig1_intensities,
            sig2_frequencies, sig2_intensities
        )
        
        matches_2to1 = self._find_peak_matches(
            sig2_frequencies, sig2_intensities,
            sig1_frequencies, sig1_intensities
        )
        
        # Calculate metrics
        total_peaks = len(sig1_frequencies) + len(sig2_frequencies)
        matched_peaks = len(matches_1to2) + len(matches_2to1)
        
        # Frequency correlation
        if matches_1to2:
            freq_corr_1to2 = sum(m[2] for m in matches_1to2) / len(matches_1to2)
        else:
            freq_corr_1to2 = 0.0
        
        if matches_2to1:
            freq_corr_2to1 = sum(m[2] for m in matches_2to1) / len(matches_2to1)
        else:
            freq_corr_2to1 = 0.0
        
        frequency_correlation = (freq_corr_1to2 + freq_corr_2to1) / 2.0
        
        # Intensity correlation
        if matches_1to2:
            int_corr_1to2 = sum(m[3] for m in matches_1to2) / len(matches_1to2)
        else:
            int_corr_1to2 = 0.0
        
        if matches_2to1:
            int_corr_2to1 = sum(m[3] for m in matches_2to1) / len(matches_2to1)
        else:
            int_corr_2to1 = 0.0
        
        intensity_correlation = (int_corr_1to2 + int_corr_2to1) / 2.0
        
        # Overall similarity
        similarity_score = (
            self.frequency_weight * frequency_correlation +
            self.intensity_weight * intensity_correlation
        )
        
        return SignatureMatch(
            similarity_score=similarity_score,
            frequency_correlation=frequency_correlation,
            intensity_correlation=intensity_correlation,
            matched_peaks=matched_peaks // 2,  # Bidirectional, so divide by 2
            total_peaks=max(len(sig1_frequencies), len(sig2_frequencies))
        )
    
    def _find_peak_matches(
        self,
        freq1: List[float],
        int1: List[float],
        freq2: List[float],
        int2: List[float]
    ) -> List[Tuple[int, int, float, float]]:
        """
        Find matching peaks from freq1 to freq2.
        
        Returns:
            List of (idx1, idx2, freq_similarity, intensity_similarity)
        """
        matches = []
        
        for i, f1 in enumerate(freq1):
            # Find closest frequency in sig2
            best_j = -1
            best_freq_diff = float('inf')
            
            for j, f2 in enumerate(freq2):
                freq_diff = abs(f1 - f2)
                if freq_diff < best_freq_diff:
                    best_freq_diff = freq_diff
                    best_j = j
            
            # Check if within tolerance
            if best_j >= 0 and best_freq_diff < self.frequency_tolerance:
                # Calculate similarities
                freq_sim = math.exp(-best_freq_diff / self.frequency_tolerance)
                
                # Intensity similarity (1 - normalized difference)
                int_diff = abs(int1[i] - int2[best_j])
                max_int = max(int1[i], int2[best_j]) + 1e-9
                int_sim = 1.0 - (int_diff / max_int)
                
                matches.append((i, best_j, freq_sim, int_sim))
        
        return matches
    
    def match_to_native(
        self,
        predicted_frequencies: List[float],
        predicted_intensities: List[float],
        native_frequencies: List[float],
        native_intensities: List[float]
    ) -> SignatureMatch:
        """
        Match predicted spectrum to native structure spectrum.
        
        This is the "catch the protein lying" metric - if the predicted
        structure has the right THz signature, it's likely correct.
        
        Args:
            predicted_frequencies: Predicted THz peaks
            predicted_intensities: Predicted intensities
            native_frequencies: Native structure THz peaks
            native_intensities: Native intensities
            
        Returns:
            SignatureMatch comparing predicted vs native
        """
        return self.match_signatures(
            predicted_frequencies, predicted_intensities,
            native_frequencies, native_intensities
        )


class SignatureClusterer:
    """
    Cluster THz signatures to identify convergent folding pathways.
    
    Uses hierarchical agglomerative clustering based on signature similarity.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize clusterer.
        
        Args:
            similarity_threshold: Min similarity to merge clusters (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.matcher = THzSignatureMatcher()
    
    def cluster_signatures(
        self,
        all_frequencies: List[List[float]],
        all_intensities: List[List[float]]
    ) -> List[SignatureCluster]:
        """
        Cluster signatures from multiple trials.
        
        Args:
            all_frequencies: List of frequency lists (one per trial)
            all_intensities: List of intensity lists (one per trial)
            
        Returns:
            List of SignatureCluster objects
        """
        n_signatures = len(all_frequencies)
        
        if n_signatures == 0:
            return []
        
        # Calculate pairwise similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(
            all_frequencies, all_intensities
        )
        
        # Perform hierarchical clustering
        clusters = self._hierarchical_clustering(similarity_matrix, n_signatures)
        
        # Build cluster objects
        cluster_objects = []
        for cluster_id, sig_indices in enumerate(clusters):
            if sig_indices:
                # Calculate centroid
                centroid_freq, centroid_int = self._calculate_centroid(
                    [all_frequencies[i] for i in sig_indices],
                    [all_intensities[i] for i in sig_indices]
                )
                
                # Calculate intra-cluster similarity
                intra_sim = self._calculate_intra_cluster_similarity(
                    sig_indices, similarity_matrix
                )
                
                cluster_objects.append(SignatureCluster(
                    cluster_id=cluster_id,
                    signatures=tuple(sig_indices),
                    centroid_frequencies=tuple(centroid_freq),
                    centroid_intensities=tuple(centroid_int),
                    intra_cluster_similarity=intra_sim
                ))
        
        return cluster_objects
    
    def _calculate_similarity_matrix(
        self,
        all_frequencies: List[List[float]],
        all_intensities: List[List[float]]
    ) -> List[List[float]]:
        """Calculate n×n similarity matrix."""
        n = len(all_frequencies)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            matrix[i][i] = 1.0  # Perfect self-similarity
            for j in range(i + 1, n):
                match = self.matcher.match_signatures(
                    all_frequencies[i], all_intensities[i],
                    all_frequencies[j], all_intensities[j]
                )
                similarity = match.similarity_score
                matrix[i][j] = similarity
                matrix[j][i] = similarity
        
        return matrix
    
    def _hierarchical_clustering(
        self,
        similarity_matrix: List[List[float]],
        n_signatures: int
    ) -> List[List[int]]:
        """
        Perform agglomerative hierarchical clustering.
        
        Returns:
            List of clusters, each containing signature indices
        """
        # Start with each signature in its own cluster
        clusters = [[i] for i in range(n_signatures)]
        
        # Merge until no more clusters meet threshold
        while True:
            best_sim = 0.0
            best_i, best_j = -1, -1
            
            # Find most similar pair of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average linkage: average similarity between all pairs
                    avg_sim = 0.0
                    count = 0
                    for idx_i in clusters[i]:
                        for idx_j in clusters[j]:
                            avg_sim += similarity_matrix[idx_i][idx_j]
                            count += 1
                    avg_sim /= count
                    
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_i, best_j = i, j
            
            # Stop if best similarity below threshold
            if best_sim < self.similarity_threshold or best_i < 0:
                break
            
            # Merge clusters
            clusters[best_i].extend(clusters[best_j])
            del clusters[best_j]
        
        return clusters
    
    def _calculate_centroid(
        self,
        frequencies_list: List[List[float]],
        intensities_list: List[List[float]]
    ) -> Tuple[List[float], List[float]]:
        """Calculate centroid (average) of cluster signatures."""
        if not frequencies_list:
            return [], []
        
        # Collect all unique frequencies
        all_freqs = set()
        for freqs in frequencies_list:
            all_freqs.update(freqs)
        
        sorted_freqs = sorted(all_freqs)
        
        # For each frequency, average intensities from all signatures
        centroid_freq = []
        centroid_int = []
        
        for f in sorted_freqs:
            intensities = []
            for i, freqs in enumerate(frequencies_list):
                # Find closest frequency in this signature
                closest_idx = min(range(len(freqs)), 
                                key=lambda j: abs(freqs[j] - f))
                if abs(freqs[closest_idx] - f) < 0.5:  # Within tolerance
                    intensities.append(intensities_list[i][closest_idx])
            
            if intensities:
                centroid_freq.append(f)
                centroid_int.append(sum(intensities) / len(intensities))
        
        return centroid_freq, centroid_int
    
    def _calculate_intra_cluster_similarity(
        self,
        sig_indices: List[int],
        similarity_matrix: List[List[float]]
    ) -> float:
        """Calculate average similarity within a cluster."""
        if len(sig_indices) <= 1:
            return 1.0
        
        total_sim = 0.0
        count = 0
        
        for i in sig_indices:
            for j in sig_indices:
                if i < j:
                    total_sim += similarity_matrix[i][j]
                    count += 1
        
        return total_sim / count if count > 0 else 0.0


class DeterminismTester:
    """
    Test folding determinism hypothesis using THz signature clustering.
    
    Core hypothesis: If folding is deterministic, multiple independent trials
    should converge to the same THz signature at energy minima.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize determinism tester.
        
        Args:
            similarity_threshold: Clustering threshold for signature similarity
        """
        self.clusterer = SignatureClusterer(similarity_threshold)
    
    def calculate_determinism_score(
        self,
        all_frequencies: List[List[float]],
        all_intensities: List[List[float]]
    ) -> DeterminismScore:
        """
        Calculate determinism score from multiple trial signatures.
        
        Args:
            all_frequencies: THz frequencies from each trial
            all_intensities: THz intensities from each trial
            
        Returns:
            DeterminismScore with interpretation
        """
        n_trials = len(all_frequencies)
        
        if n_trials < 2:
            return DeterminismScore(
                n_trials=n_trials,
                n_clusters=1 if n_trials == 1 else 0,
                largest_cluster_size=n_trials,
                convergence_ratio=1.0 if n_trials == 1 else 0.0,
                average_intra_cluster_similarity=1.0,
                determinism_score=0.0  # Insufficient data
            )
        
        # Cluster signatures
        clusters = self.clusterer.cluster_signatures(all_frequencies, all_intensities)
        
        if not clusters:
            return DeterminismScore(
                n_trials=n_trials,
                n_clusters=0,
                largest_cluster_size=0,
                convergence_ratio=0.0,
                average_intra_cluster_similarity=0.0,
                determinism_score=0.0
            )
        
        # Calculate metrics
        n_clusters = len(clusters)
        largest_cluster = max(clusters, key=lambda c: c.size)
        largest_cluster_size = largest_cluster.size
        convergence_ratio = largest_cluster_size / n_trials
        
        # Average intra-cluster similarity
        avg_intra_sim = sum(c.intra_cluster_similarity for c in clusters) / n_clusters
        
        # Overall determinism score (0-1)
        # High score = few clusters, high convergence, high intra-cluster similarity
        cluster_score = 1.0 - ((n_clusters - 1) / n_trials)  # Fewer clusters = higher
        determinism_score = (
            0.5 * convergence_ratio +  # Weight on largest cluster
            0.3 * cluster_score +       # Weight on total clusters
            0.2 * avg_intra_sim         # Weight on cluster quality
        )
        
        return DeterminismScore(
            n_trials=n_trials,
            n_clusters=n_clusters,
            largest_cluster_size=largest_cluster_size,
            convergence_ratio=convergence_ratio,
            average_intra_cluster_similarity=avg_intra_sim,
            determinism_score=determinism_score
        )


# Factory functions
def create_signature_matcher(
    frequency_tolerance: float = 0.5,
    intensity_weight: float = 0.3
) -> THzSignatureMatcher:
    """Create a THzSignatureMatcher with specified parameters."""
    return THzSignatureMatcher(frequency_tolerance, intensity_weight)


def create_signature_clusterer(similarity_threshold: float = 0.7) -> SignatureClusterer:
    """Create a SignatureClusterer with specified threshold."""
    return SignatureClusterer(similarity_threshold)


def create_determinism_tester(similarity_threshold: float = 0.7) -> DeterminismTester:
    """Create a DeterminismTester with specified threshold."""
    return DeterminismTester(similarity_threshold)
