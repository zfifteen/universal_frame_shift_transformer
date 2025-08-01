import numpy as np
import matplotlib.pyplot as plt
from math import log, pi, sqrt

# Constants: Z = A(B/C), C as invariant (speed of light analog)
C = 2.718281828459045  # e as universal anchor
PHI = (1 + sqrt(5)) / 2  # Golden ratio for resonance

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

def z_frame_shift(n, max_n, k=0.3):
    """Z = n * (B/C), B as derived frame shift, k from prime_curve's optimal"""
    B = log(n + 1) / log(max_n) if n > 1 else 0  # Derived frame shift
    return n * (B / C) * PHI ** k  # Z-normalized, golden-scaled

def geometric_projection(numbers, max_n):
    """Project numbers into 3D cylindrical space, derive density for primes"""
    n = np.array(numbers, dtype=float)
    z_values = np.array([z_frame_shift(x, max_n) for x in n])
    x = n  # Linear axis
    y = np.log(n + 1) * z_values  # Logarithmic scaling with Z
    z = np.sin(pi * 0.1 * n) * (1 + z_values)  # Oscillatory with Z-weight
    coords = np.array([x, y, z]).T

    # Derive density map (inverse distance, vectorized)
    distances = np.sqrt(((coords[:, np.newaxis] - coords) ** 2).sum(axis=2))
    np.fill_diagonal(distances, 1e-10)  # Avoid self-division
    density = np.sum(1.0 / (distances + 1e-10), axis=1) / len(coords)
    return coords, density

def main():
    N = 1000  # Small range for simplicity
    numbers = np.arange(1, N + 1)
    primes = np.array([n for n in numbers if is_prime(n)])

    # Project and derive density
    coords, density = geometric_projection(numbers, N)

    # Filter high-density points (top 168 for prime count)
    threshold_indices = np.argsort(density)[-len(primes):]  # Top 168
    prime_candidates = numbers[threshold_indices]

    # Visualize: 3D and 2D in subplots
    fig = plt.figure(figsize=(16, 8))

    # 3D Plot (cleaner view)
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=density, cmap='plasma', alpha=0.2, s=5)
    prime_coords = coords[np.isin(numbers, primes)]
    ax1.scatter(prime_coords[:, 0], prime_coords[:, 1], prime_coords[:, 2], c='red', marker='*', s=30, label='Primes')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Z-Log')
    ax1.set_zlabel('Z-Sine')
    ax1.set_title('Geometric Z-Projection of Primes (3D)')
    ax1.legend()
    ax1.view_init(elev=20, azim=-60)  # Side profile for helix clarity
    fig.colorbar(sc, ax=ax1, shrink=0.5, aspect=5, label='Density')

    # 2D Plot: n vs. Density with primes marked
    ax2 = fig.add_subplot(122)
    ax2.plot(numbers, density, color='purple', alpha=0.7, label='Density Curve')
    ax2.scatter(primes, density[np.isin(numbers, primes)], c='red', marker='*', s=30, label='Primes')
    ax2.axhline(np.sort(density)[-len(primes)], color='gray', linestyle='--', label=f'Top {len(primes)} Threshold')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Derived Density')
    ax2.set_title('Density vs. n (2D)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, max(density) * 1.1)  # Focus on relevant range

    plt.tight_layout()
    plt.show()

    print(f"Found {len(prime_candidates)} prime candidates: {sorted(prime_candidates[:10])}")

if __name__ == "__main__":
    main()