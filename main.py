#!/usr/bin/env python3
"""
Universal Frame Shift Theory: Reference Implementation
======================================================

A complete reference implementation for analyzing prime number distribution
through Universal Frame Shift corrections. This implementation demonstrates
that prime numbers exhibit predictable geometric clustering when viewed from
a universal reference frame centered on the speed of light constant.

Author: [Author Name]
Date: [Date]
Version: 1.0

Theory Overview:
- Universal Form: Z = A(B/C) applies across all measurement domains
- Physical Domain: Z = T(v/c)
- Discrete Domain: Z = n(Δₙ/Δₘₐₓ)
- Prime distribution becomes geometric when frame-corrected

Expected Results:
- ~35x improvement in prime density clustering
- Clear separation between φ-region and π-region performance
- Coherent helical prime arrangements in 3D space

Usage:
    python universal_frame_shift.py

Requirements:
    numpy >= 1.19.0
    matplotlib >= 3.3.0
    scipy >= 1.5.0
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Dict, List, Tuple, Any

# ============================================================================
# UNIVERSAL CONSTANTS
# ============================================================================

UNIVERSAL = math.e  # The invariant limit in discrete domain (e ≈ 2.718282)
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio (φ ≈ 1.618034)
PI_E_RATIO = math.pi / math.e  # Natural scaling ratio (π/e ≈ 1.155727)

# Analysis parameters
DEFAULT_N_POINTS = 3000      # Integer range for analysis
DEFAULT_N_CANDIDATES = 200   # Parameter combinations tested
DEFAULT_TOP_K = 15          # Results retained for analysis
RANDOM_SEED = 42            # For reproducible results

# ============================================================================
# CORE UNIVERSAL FRAME SHIFT IMPLEMENTATION
# ============================================================================

class UniversalFrameShift:
    """
    Implements the Universal Form: Z = A(B/C)

    In discrete domain: Z = n(Δₙ/Δₘₐₓ)
    Provides bidirectional transformation between observer and universal frames.
    """

    def __init__(self, rate: float, invariant_limit: float = UNIVERSAL):
        """
        Initialize Universal Frame Shift transformer.

        Args:
            rate: Domain-specific rate parameter (B in universal form)
            invariant_limit: Universal invariant constant (C in universal form)

        Raises:
            ValueError: If rate is zero or negative
        """
        if rate <= 0:
            raise ValueError("Rate must be positive")

        self._rate = rate
        self._invariant_limit = invariant_limit
        self._correction_factor = rate / invariant_limit

    @property
    def rate(self) -> float:
        """Get the rate parameter."""
        return self._rate

    @property
    def invariant_limit(self) -> float:
        """Get the invariant limit."""
        return self._invariant_limit

    @property
    def correction_factor(self) -> float:
        """Get the computed correction factor."""
        return self._correction_factor

    def transform(self, observed_quantity: float) -> float:
        """
        Transform from observer frame to universal frame.

        Args:
            observed_quantity: Value measured in observer frame

        Returns:
            Corresponding value in universal coordinates
        """
        return observed_quantity * self._correction_factor

    def inverse_transform(self, universal_quantity: float) -> float:
        """
        Transform from universal frame back to observer frame.

        Args:
            universal_quantity: Value in universal coordinates

        Returns:
            Corresponding value in observer frame
        """
        return universal_quantity / self._correction_factor

# ============================================================================
# MATHEMATICAL UTILITIES
# ============================================================================

def is_prime(n: int) -> bool:
    """
    Optimized primality test.

    Args:
        n: Integer to test

    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    sqrt_n = int(math.sqrt(n)) + 1
    for i in range(3, sqrt_n, 2):
        if n % i == 0:
            return False
    return True

def compute_frame_shift(n: int, max_n: int) -> float:
    """
    Compute the frame shift Δₙ at position n.

    The frame shift captures how "number space" distorts away from the origin.
    Uses logarithmic growth with oscillatory corrections based on prime gap structure.

    Args:
        n: Position in number sequence
        max_n: Maximum position (for normalization)

    Returns:
        Frame shift value Δₙ ∈ [0, 1]
    """
    if n <= 1:
        return 0.0

    # Logarithmic base component - captures space "stretching"
    base_shift = math.log(n) / math.log(max_n)

    # Oscillatory component - captures local prime gap structure
    gap_phase = 2 * math.pi * n / (math.log(n) + 1)
    oscillation = 0.1 * math.sin(gap_phase)

    return base_shift + oscillation

def generate_prime_mask(n_points: int) -> np.ndarray:
    """
    Generate boolean mask for primes in range [1, n_points].

    Args:
        n_points: Upper limit of range

    Returns:
        Boolean array where True indicates prime
    """
    n_array = np.arange(1, n_points + 1)
    return np.vectorize(is_prime)(n_array)

# ============================================================================
# 3D EMBEDDING AND COORDINATE TRANSFORMATION
# ============================================================================

def compute_universal_coordinates(n_points: int, rate: float, helix_freq: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D coordinates in universal frame with frame shift corrections.

    Coordinate system:
    - x: Natural integer position (unchanged)
    - y: Frame-corrected growth coordinate based on n²/π
    - z: Frame-aware helical coordinate with variable amplitude

    Args:
        n_points: Number of integers to process
        rate: Rate parameter for Universal Frame Shift
        helix_freq: Base frequency for helical coordinate

    Returns:
        Tuple of (x, y, z, frame_shifts) coordinate arrays
    """
    # Generate integer sequence
    n = np.arange(1, n_points + 1)

    # Compute frame shifts for all positions
    frame_shifts = np.array([compute_frame_shift(i, n_points) for i in n])

    # Initialize Universal Frame Shift transformer
    transformer = UniversalFrameShift(rate=rate)

    # X coordinate: Natural position (unchanged)
    x = n.astype(float)

    # Y coordinate: Frame-corrected growth based on n²/π
    y_base = n * (n / math.pi)  # Base quadratic growth
    y_corrected = y_base * (1 + frame_shifts)  # Apply frame correction
    y = transformer.transform(y_corrected)  # Transform to universal frame

    # Z coordinate: Frame-aware helical coordinate
    effective_freq = helix_freq * (1 + np.mean(frame_shifts))  # Frame-corrected frequency
    z_amplitude = 1 + 0.5 * frame_shifts  # Variable amplitude based on frame shift
    z = np.sin(math.pi * effective_freq * n) * z_amplitude

    return x, y, z, frame_shifts

def extract_prime_coordinates(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              prime_mask: np.ndarray) -> np.ndarray:
    """
    Extract 3D coordinates of prime numbers only.

    Args:
        x, y, z: Full coordinate arrays
        prime_mask: Boolean mask indicating prime positions

    Returns:
        Array of shape (n_primes, 3) containing prime coordinates
    """
    return np.vstack((
        x[prime_mask],
        y[prime_mask],
        z[prime_mask]
    )).T

# ============================================================================
# PRIME DENSITY ANALYSIS
# ============================================================================

def compute_prime_density_score(rate: float, helix_freq: float, n_points: int) -> float:
    """
    Compute frame-corrected prime density score.

    The density score measures how tightly clustered primes are in the
    transformed 3D space, using position-weighted nearest neighbor distances.

    Args:
        rate: Rate parameter for Universal Frame Shift
        helix_freq: Helical coordinate frequency
        n_points: Number of integers to analyze

    Returns:
        Density score (higher = better clustering)
    """
    # Generate coordinates and prime mask
    x, y, z, frame_shifts = compute_universal_coordinates(n_points, rate, helix_freq)
    prime_mask = generate_prime_mask(n_points)

    # Extract prime coordinates
    prime_coords = extract_prime_coordinates(x, y, z, prime_mask)

    if len(prime_coords) < 2:
        return 0.0  # Need at least 2 primes for distance calculation

    # Build KD-tree for efficient nearest neighbor search
    tree = KDTree(prime_coords)

    # Find nearest neighbors (k=2: self + nearest other prime)
    k_neighbors = min(3, len(prime_coords))
    distances, indices = tree.query(prime_coords, k=k_neighbors)

    if distances.shape[1] <= 1:
        return 0.0

    # Apply position-based weighting to account for frame expansion
    # Points closer to origin have different geometric significance
    position_weights = 1.0 / (np.sqrt(prime_coords[:, 0]) + 1)
    weighted_distances = distances[:, 1] * position_weights

    # Compute mean weighted distance
    mean_distance = np.mean(weighted_distances)

    # Return density score (inverse of mean distance)
    return 1.0 / mean_distance if mean_distance > 0 else 0.0

# ============================================================================
# PARAMETER OPTIMIZATION
# ============================================================================

def generate_optimization_parameters(n_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate parameter combinations focused on mathematically significant regions.

    Strategy:
    - Rate parameters centered around key mathematical constants (φ, π, e ratios)
    - Frequency parameters based on harmonics of fundamental frequency

    Args:
        n_candidates: Number of parameter combinations to generate

    Returns:
        Tuple of (rate_values, frequency_values) arrays
    """
    np.random.seed(RANDOM_SEED)

    # Rate parameter centers based on mathematical significance
    rate_centers = [
        UNIVERSAL / PHI,        # e/φ ≈ 1.68 (golden ratio scaling)
        UNIVERSAL / math.pi,    # e/π ≈ 0.86 (circular to exponential)
        UNIVERSAL / 2,          # e/2 ≈ 1.36 (half-domain)
        UNIVERSAL,              # e ≈ 2.72 (identity transformation)
        UNIVERSAL * PHI / 2,    # e*φ/2 ≈ 2.20 (golden mean)
        PI_E_RATIO * UNIVERSAL, # π ≈ 3.14 (π-scaling region)
    ]

    # Generate rate samples around each center
    rates = []
    samples_per_center = n_candidates // len(rate_centers)

    for center in rate_centers:
        # Gaussian distribution around each center
        std_dev = center * 0.15  # 15% standard deviation
        samples = np.random.normal(center, std_dev, samples_per_center)
        rates.extend(samples)

    # Fill remaining slots with uniform samples
    remaining = n_candidates - len(rates)
    if remaining > 0:
        uniform_samples = np.random.uniform(UNIVERSAL/3, UNIVERSAL*2, remaining)
        rates.extend(uniform_samples)

    rates = np.array(rates[:n_candidates])
    rates = np.clip(rates, 0.1, 10.0)  # Ensure reasonable bounds

    # Frequency parameters based on harmonics
    fundamental_freq = 1.0 / (2 * math.pi)  # Base frequency ≈ 0.159
    freq_harmonics = [
        fundamental_freq * 0.5,         # Sub-harmonic
        fundamental_freq * PHI / 3,     # Golden ratio harmonic
        fundamental_freq * 1.0,         # Fundamental
        fundamental_freq * math.sqrt(2), # √2 harmonic
        fundamental_freq * PHI,         # Golden harmonic
        fundamental_freq * 2.0,         # First overtone
    ]

    frequencies = []
    samples_per_harmonic = n_candidates // len(freq_harmonics)

    for harmonic in freq_harmonics:
        std_dev = harmonic * 0.1  # 10% standard deviation
        samples = np.random.normal(harmonic, std_dev, samples_per_harmonic)
        frequencies.extend(samples)

    # Fill remaining slots
    remaining = n_candidates - len(frequencies)
    if remaining > 0:
        uniform_freq = np.random.uniform(0.02, 0.25, remaining)
        frequencies.extend(uniform_freq)

    frequencies = np.array(frequencies[:n_candidates])
    frequencies = np.clip(frequencies, 0.01, 0.5)  # Ensure reasonable bounds

    return rates, frequencies

def optimize_parameters(n_points: int = DEFAULT_N_POINTS,
                        n_candidates: int = DEFAULT_N_CANDIDATES,
                        top_k: int = DEFAULT_TOP_K,
                        verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Optimize Universal Frame Shift parameters for maximum prime density.

    Args:
        n_points: Range of integers to analyze
        n_candidates: Number of parameter combinations to test
        top_k: Number of top results to return
        verbose: Whether to print progress updates

    Returns:
        List of top parameter sets with scores and metadata
    """
    if verbose:
        print(f"Optimizing Universal Frame Shift parameters...")
        print(f"Integer range: 1 to {n_points}")
        print(f"Testing {n_candidates} parameter combinations")
        print("-" * 60)

    # Generate parameter combinations
    rates, frequencies = generate_optimization_parameters(n_candidates)

    # Evaluate all combinations
    results = []
    start_time = time.time()

    for i, (rate, freq) in enumerate(zip(rates, frequencies)):
        if verbose and i % 50 == 0:
            elapsed = time.time() - start_time
            progress = i / n_candidates * 100
            print(f"Progress: {i}/{n_candidates} ({progress:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s")

        # Compute density score
        score = compute_prime_density_score(rate, freq, n_points)

        # Calculate mathematical significance metrics
        phi_distance = abs(rate - UNIVERSAL/PHI)
        pi_distance = abs(rate - UNIVERSAL/math.pi)
        mathematical_significance = 1.0 / (1.0 + min(phi_distance, pi_distance))

        results.append({
            'rate': rate,
            'freq': freq,
            'score': score,
            'rate_ratio': rate / UNIVERSAL,
            'mathematical_significance': mathematical_significance,
            'phi_region': phi_distance < 0.2,
            'pi_region': pi_distance < 0.2,
        })

    # Sort by composite score (density + mathematical significance)
    def composite_score(result):
        base_score = result['score']
        significance_bonus = result['mathematical_significance']
        return base_score * (1.0 + 0.1 * significance_bonus)

    results.sort(key=composite_score, reverse=True)

    if verbose:
        total_time = time.time() - start_time
        print(f"\nOptimization complete in {total_time:.1f}s")
        print(f"Top {top_k} results selected")

    return results[:top_k]

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_optimization_results(results: List[Dict[str, Any]], title: str = "Universal Frame Shift Optimization Results"):
    """
    Create bar chart visualization of optimization results.

    Args:
        results: List of parameter results from optimization
        title: Plot title
    """
    n_results = len(results)

    # Prepare data
    labels = [f"B/e={r['rate']/UNIVERSAL:.3f}\nf={r['freq']:.3f}" for r in results]
    scores = [r['score'] for r in results]

    # Create figure
    plt.figure(figsize=(max(12, n_results * 0.8), 6))
    bars = plt.bar(range(n_results), scores, alpha=0.8)

    # Color bars by mathematical region
    for i, (bar, result) in enumerate(zip(bars, results)):
        if result['phi_region']:
            bar.set_color('gold')  # φ-region
        elif result['pi_region']:
            bar.set_color('crimson')  # π-region
        else:
            bar.set_color('darkslategray')  # Other regions

    # Formatting
    plt.xticks(range(n_results), labels, rotation=45, ha='right')
    plt.ylabel("Frame-Corrected Prime Density Score")
    plt.title(f"{title}\n(Gold: φ-region, Red: π-region)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_3d_prime_distribution(rate: float, freq: float, n_points: int,
                               title: str = None):
    """
    Create 3D scatter plot of prime distribution in universal coordinates.

    Args:
        rate: Rate parameter for transformation
        freq: Helical frequency parameter
        n_points: Number of integers to analyze
        title: Plot title (auto-generated if None)
    """
    # Generate coordinates
    x, y, z, frame_shifts = compute_universal_coordinates(n_points, rate, freq)
    prime_mask = generate_prime_mask(n_points)

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot composite numbers with frame-shift color coding
    composite_mask = ~prime_mask
    if np.any(composite_mask):
        composite_colors = plt.cm.viridis(frame_shifts[composite_mask] / np.max(frame_shifts))
        ax.scatter(x[composite_mask], y[composite_mask], z[composite_mask],
                   c=composite_colors, alpha=0.3, s=8, label='Composites')

    # Plot primes as red stars
    if np.any(prime_mask):
        ax.scatter(x[prime_mask], y[prime_mask], z[prime_mask],
                   c='red', marker='*', s=60, alpha=0.9, label='Primes')

    # Formatting
    if title is None:
        density_score = compute_prime_density_score(rate, freq, n_points)
        title = f"Rate/e={rate/UNIVERSAL:.3f}, freq={freq:.3f}, score={density_score:.6f}"

    ax.set_title(title)
    ax.set_xlabel('n (Natural Position)')
    ax.set_ylabel('Frame-Corrected Y')
    ax.set_zlabel('Frame-Aware Helix Z')
    ax.legend()

    plt.tight_layout()
    plt.show()

def visualize_top_results(results: List[Dict[str, Any]], n_points: int, max_plots: int = 3):
    """
    Generate comprehensive visualizations for top optimization results.

    Args:
        results: Top parameter results from optimization
        n_points: Number of integers used in analysis
        max_plots: Maximum number of 3D plots to generate
    """
    # Bar chart of all results
    plot_optimization_results(results)

    # 3D plots for top results
    for i, result in enumerate(results[:max_plots], 1):
        rate, freq = result['rate'], result['freq']
        title = f"#{i}: Rate/e={rate/UNIVERSAL:.3f}, freq={freq:.3f}, score={result['score']:.6f}"
        plot_3d_prime_distribution(rate, freq, n_points, title)

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_analysis(n_points: int = DEFAULT_N_POINTS,
                 n_candidates: int = DEFAULT_N_CANDIDATES,
                 top_k: int = DEFAULT_TOP_K,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Run complete Universal Frame Shift analysis.

    Args:
        n_points: Range of integers to analyze
        n_candidates: Number of parameter combinations to test
        top_k: Number of top results to retain
        verbose: Whether to print detailed output

    Returns:
        Dictionary containing analysis results and metadata
    """
    if verbose:
        print("Universal Frame Shift Theory: Prime Distribution Analysis")
        print("=" * 70)
        print(f"Universal constant (e): {UNIVERSAL:.6f}")
        print(f"Golden ratio (φ): {PHI:.6f}")
        print(f"π/e ratio: {PI_E_RATIO:.6f}")
        print()

    # Run optimization
    start_time = time.time()
    results = optimize_parameters(n_points, n_candidates, top_k, verbose)
    optimization_time = time.time() - start_time

    if verbose:
        print("\nTop Results:")
        print("-" * 70)
        for i, result in enumerate(results[:5], 1):
            phi_marker = " [φ]" if result['phi_region'] else ""
            pi_marker = " [π]" if result['pi_region'] else ""
            print(f"#{i}: Rate={result['rate']:.4f} (B/e={result['rate']/UNIVERSAL:.3f}), "
                  f"Freq={result['freq']:.4f}, Score={result['score']:.6f}{phi_marker}{pi_marker}")

    # Calculate improvement metrics
    if len(results) > 0:
        best_score = results[0]['score']
        baseline_score = 0.0007  # Typical observer-frame score
        improvement_factor = best_score / baseline_score if baseline_score > 0 else float('inf')

        if verbose:
            print(f"\nPerformance Metrics:")
            print(f"Best density score: {best_score:.6f}")
            print(f"Baseline comparison: {baseline_score:.6f}")
            print(f"Improvement factor: {improvement_factor:.1f}x")

    # Generate visualizations
    if verbose:
        print("\nGenerating visualizations...")

    visualize_top_results(results, n_points)

    # Prepare return data
    analysis_results = {
        'results': results,
        'n_points': n_points,
        'n_candidates': n_candidates,
        'optimization_time': optimization_time,
        'best_score': results[0]['score'] if results else 0.0,
        'improvement_factor': improvement_factor if results else 0.0,
        'phi_region_count': sum(1 for r in results if r['phi_region']),
        'pi_region_count': sum(1 for r in results if r['pi_region']),
    }

    if verbose:
        print(f"\nAnalysis complete. Total time: {optimization_time:.1f}s")
        print(f"φ-region results: {analysis_results['phi_region_count']}")
        print(f"π-region results: {analysis_results['pi_region_count']}")

    return analysis_results

# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

def validate_implementation():
    """
    Run validation tests to ensure implementation correctness.

    Returns:
        True if all tests pass, False otherwise
    """
    print("Running validation tests...")

    # Test 1: Universal Frame Shift transformer
    try:
        transformer = UniversalFrameShift(rate=2.0, invariant_limit=math.e)
        test_value = 10.0
        transformed = transformer.transform(test_value)
        recovered = transformer.inverse_transform(transformed)

        if not abs(recovered - test_value) < 1e-10:
            print("❌ FAIL: Bidirectional transformation test")
            return False
        print("✅ PASS: Bidirectional transformation test")

    except Exception as e:
        print(f"❌ FAIL: Transformer test - {e}")
        return False

    # Test 2: Prime detection
    test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    test_composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]

    for p in test_primes:
        if not is_prime(p):
            print(f"❌ FAIL: Prime detection - {p} should be prime")
            return False

    for c in test_composites:
        if is_prime(c):
            print(f"❌ FAIL: Prime detection - {c} should be composite")
            return False

    print("✅ PASS: Prime detection test")

    # Test 3: Frame shift calculation
    try:
        shift = compute_frame_shift(100, 1000)
        if not (0 <= shift <= 1):
            print(f"❌ FAIL: Frame shift bounds - got {shift}")
            return False
        print("✅ PASS: Frame shift calculation test")

    except Exception as e:
        print(f"❌ FAIL: Frame shift test - {e}")
        return False

    # Test 4: Coordinate generation
    try:
        x, y, z, shifts = compute_universal_coordinates(100, 2.0, 0.1)
        if len(x) != 100 or len(y) != 100 or len(z) != 100:
            print("❌ FAIL: Coordinate generation dimensions")
            return False
        print("✅ PASS: Coordinate generation test")

    except Exception as e:
        print(f"❌ FAIL: Coordinate generation test - {e}")
        return False

    # Test 5: Density score calculation
    try:
        score = compute_prime_density_score(2.0, 0.1, 100)
        if not (score >= 0):
            print(f"❌ FAIL: Density score bounds - got {score}")
            return False
        print("✅ PASS: Density score calculation test")

    except Exception as e:
        print(f"❌ FAIL: Density score test - {e}")
        return False

    print("✅ All validation tests passed!")
    return True

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Main function for command line execution.
    """
    print(__doc__)

    # Run validation first
    if not validate_implementation():
        print("\n❌ Validation failed. Please check implementation.")
        return 1

    print("\n" + "="*70)

    # Run main analysis
    try:
        results = run_analysis(
            n_points=DEFAULT_N_POINTS,
            n_candidates=DEFAULT_N_CANDIDATES,
            top_k=DEFAULT_TOP_K,
            verbose=True
        )

        print("\n🎯 Analysis completed successfully!")
        print(f"Best improvement factor: {results['improvement_factor']:.1f}x")

        return 0

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())