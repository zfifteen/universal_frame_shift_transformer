import numpy as np
import matplotlib.pyplot as plt
from math import log, pi, sqrt
from scipy.spatial import KDTree  # For efficient nearest neighbors

# Tunable Parameters
N = 1000000  # Range of numbers to analyze (max n)
K = 0.3  # Exponent from prime_curve's optimal k-sweep for Z scaling
FREQ_BASE = 10 # Base frequency multiplier for sine wave
EARTH_OMEGA = 26  # Earth's rotational frequency (rad/s) for cosmic tuning
NEAREST_K = 7  # Number of nearest neighbors for local density (ratio-based)
ALPHA = 0.2  # Transparency for 3D scatter points
SIZE_3D = 5  # Size of 3D scatter points
SIZE_2D = 30  # Size of 2D scatter points/markers
ELEV = 20  # Elevation angle for 3D plot view
AZIM = -60  # Azimuth angle for 3D plot view
DENSITY_THRESHOLD = 168  # Number of top density points to consider (matches prime count for N=1000)

# Constants
C = 1 # e as universal anchor
PHI = (1 + sqrt(5)) / 2  # Golden ratio for resonance

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

def z_frame_shift(n, max_n):
    B = log(n + 1) / log(max_n) if n > 1 else 0
    return n * (B / C) * PHI ** K

def geometric_projection(numbers, max_n):
    n = np.array(numbers, dtype=float)
    x = n
    z_values = np.array([z_frame_shift(x_i, max_n) for x_i in n])  # Minimal var
    y = np.log(n + 1) * z_values
    z = np.sin((pi * FREQ_BASE + EARTH_OMEGA) * n) * (1 + z_values)
    coords = np.column_stack((x, y, z))

    # Derive local density via k-nearest (ratio-based, efficient)
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=NEAREST_K + 1)  # +1 for self
    density = 1.0 / np.mean(dists[:, 1:], axis=1)  # Avg inverse dist to nearest K
    return coords, density

def main():
    numbers = np.arange(1, N + 1)
    primes = np.array([n for n in numbers if is_prime(n)])

    coords, density = geometric_projection(numbers, N)

    # Filter top density
    threshold_indices = np.argsort(density)[-DENSITY_THRESHOLD:]
    prime_candidates = numbers[threshold_indices]

    fig = plt.figure(figsize=(16, 8))

    # 3D Plot
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=density, cmap='plasma', alpha=ALPHA, s=SIZE_3D)
    prime_coords = coords[np.isin(numbers, primes)]
    ax1.scatter(prime_coords[:, 0], prime_coords[:, 1], prime_coords[:, 2], c='red', marker='*', s=SIZE_2D, label='Primes')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Z-Log')
    ax1.set_zlabel('Z-Sine')
    ax1.set_title('Geometric Z-Projection of Primes (3D)')
    ax1.legend()
    ax1.view_init(elev=ELEV, azim=AZIM)
    fig.colorbar(sc, ax=ax1, shrink=0.5, aspect=5, label='Density')

    # 2D Plot
    ax2 = fig.add_subplot(122)
    ax2.plot(numbers, density, color='purple', alpha=0.7, label='Density Curve')
    ax2.scatter(primes, density[np.isin(numbers, primes)], c='red', marker='*', s=SIZE_2D, label='Primes')
    ax2.axhline(np.sort(density)[-DENSITY_THRESHOLD], color='gray', linestyle='--', label=f'Top {DENSITY_THRESHOLD} Threshold')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Derived Density')
    ax2.set_title('Density vs. n (2D)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, max(density) * 1.1)

    plt.tight_layout()
    plt.show()

    print(f"Found {len(prime_candidates)} prime candidates: {sorted(prime_candidates[:10])}")

if __name__ == "__main__":
    main()