#!/usr/bin/env python3
"""
Flask Web Application for Geometric Z-Prime Visualization
========================================================

A web application that allows users to interactively adjust parameters
and visualize prime number distributions through geometric projections.

Based on the geometric_z_prime.py script, this Flask application provides
a user-friendly interface for parameter tuning and dynamic plot generation.

Author: Universal Frame Shift Transformer
Date: 2025
Version: 1.0
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
from math import log, pi, sqrt
from scipy.spatial import KDTree
from flask import Flask, render_template, request, jsonify
import traceback

app = Flask(__name__)

# Default tunable parameters (optimized for web performance)
DEFAULT_PARAMS = {
    'N': 10000,  # Reduced for web performance
    'K': 0.3,
    'FREQ_BASE': 10,
    'EARTH_OMEGA': 26,
    'NEAREST_K': 7,
    'ALPHA': 0.2,
    'SIZE_3D': 5,
    'SIZE_2D': 30,
    'ELEV': 20,
    'AZIM': -60,
    'DENSITY_THRESHOLD': 168
}

# Constants
C = 1  # Universal anchor
PHI = (1 + sqrt(5)) / 2  # Golden ratio for resonance


def is_prime(n):
    """
    Check if a number is prime.
    
    Args:
        n (int): Number to check
        
    Returns:
        bool: True if prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def z_frame_shift(n, max_n, k):
    """
    Calculate the Z frame shift for a given number.
    
    Args:
        n (int): The number
        max_n (int): Maximum number in range
        k (float): Exponent parameter
        
    Returns:
        float: Z frame shift value
    """
    if n <= 1:
        return 0.0
    B = log(n + 1) / log(max_n)
    return n * (B / C) * PHI ** k


def geometric_projection(numbers, max_n, k, freq_base, earth_omega, nearest_k):
    """
    Generate geometric projection coordinates and density for numbers.
    
    Args:
        numbers (array): Array of numbers to analyze
        max_n (int): Maximum number in range
        k (float): Exponent parameter
        freq_base (float): Base frequency multiplier
        earth_omega (float): Earth's rotational frequency
        nearest_k (int): Number of nearest neighbors
        
    Returns:
        tuple: (coordinates, density) arrays
    """
    n = np.array(numbers, dtype=float)
    x = n
    z_values = np.array([z_frame_shift(x_i, max_n, k) for x_i in n])
    y = np.log(n + 1) * z_values
    z = np.sin((pi * freq_base + earth_omega) * n) * (1 + z_values)
    coords = np.column_stack((x, y, z))

    # Calculate local density via k-nearest neighbors
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=nearest_k + 1)  # +1 for self
    density = 1.0 / np.mean(dists[:, 1:], axis=1)  # Average inverse distance
    
    return coords, density


def generate_plot(params):
    """
    Generate the geometric Z-projection plot with given parameters.
    
    Args:
        params (dict): Dictionary of tunable parameters
        
    Returns:
        str: Base64 encoded plot image, or None if error
    """
    try:
        # Extract parameters
        N = int(params.get('N', DEFAULT_PARAMS['N']))
        K = float(params.get('K', DEFAULT_PARAMS['K']))
        FREQ_BASE = float(params.get('FREQ_BASE', DEFAULT_PARAMS['FREQ_BASE']))
        EARTH_OMEGA = float(params.get('EARTH_OMEGA', DEFAULT_PARAMS['EARTH_OMEGA']))
        NEAREST_K = int(params.get('NEAREST_K', DEFAULT_PARAMS['NEAREST_K']))
        ALPHA = float(params.get('ALPHA', DEFAULT_PARAMS['ALPHA']))
        SIZE_3D = float(params.get('SIZE_3D', DEFAULT_PARAMS['SIZE_3D']))
        SIZE_2D = float(params.get('SIZE_2D', DEFAULT_PARAMS['SIZE_2D']))
        ELEV = float(params.get('ELEV', DEFAULT_PARAMS['ELEV']))
        AZIM = float(params.get('AZIM', DEFAULT_PARAMS['AZIM']))
        DENSITY_THRESHOLD = int(params.get('DENSITY_THRESHOLD', DEFAULT_PARAMS['DENSITY_THRESHOLD']))

        # Validate parameters
        if N <= 0 or N > 100000:  # Reduced max for web performance
            raise ValueError("N must be between 1 and 100,000")
        if NEAREST_K <= 0 or NEAREST_K > 20:
            raise ValueError("NEAREST_K must be between 1 and 20")
        if ALPHA < 0 or ALPHA > 1:
            raise ValueError("ALPHA must be between 0 and 1")
        if DENSITY_THRESHOLD <= 0:
            raise ValueError("DENSITY_THRESHOLD must be positive")

        # Generate data
        numbers = np.arange(1, N + 1)
        primes = np.array([n for n in numbers if is_prime(n)])
        
        coords, density = geometric_projection(
            numbers, N, K, FREQ_BASE, EARTH_OMEGA, NEAREST_K
        )

        # Filter top density points
        threshold_indices = np.argsort(density)[-DENSITY_THRESHOLD:]
        prime_candidates = numbers[threshold_indices]

        # Create figure
        fig = plt.figure(figsize=(16, 8))
        
        # 3D Plot
        ax1 = fig.add_subplot(121, projection='3d')
        sc = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                        c=density, cmap='plasma', alpha=ALPHA, s=SIZE_3D)
        
        # Add primes to 3D plot
        prime_coords = coords[np.isin(numbers, primes)]
        ax1.scatter(prime_coords[:, 0], prime_coords[:, 1], prime_coords[:, 2], 
                   c='red', marker='*', s=SIZE_2D, label='Primes')
        
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
        ax2.scatter(primes, density[np.isin(numbers, primes)], 
                   c='red', marker='*', s=SIZE_2D, label='Primes')
        ax2.axhline(np.sort(density)[-DENSITY_THRESHOLD], color='gray', 
                   linestyle='--', label=f'Top {DENSITY_THRESHOLD} Threshold')
        ax2.set_xlabel('n')
        ax2.set_ylabel('Derived Density')
        ax2.set_title('Density vs. n (2D)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, max(density) * 1.1)

        plt.tight_layout()

        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        # Calculate summary statistics
        stats = {
            'total_numbers': len(numbers),
            'total_primes': len(primes),
            'prime_candidates': len(prime_candidates),
            'prime_candidates_list': sorted(prime_candidates[:10].tolist()),
            'density_max': float(np.max(density)),
            'density_mean': float(np.mean(density))
        }

        return img_data, stats

    except Exception as e:
        print(f"Error generating plot: {e}")
        traceback.print_exc()
        return None, {'error': str(e)}


@app.route('/')
def index():
    """Main page with parameter form and plot display."""
    return render_template('index.html', params=DEFAULT_PARAMS)


@app.route('/generate', methods=['POST'])
def generate():
    """Generate plot with user-specified parameters."""
    try:
        # Get parameters from form
        params = {}
        for key in DEFAULT_PARAMS.keys():
            value = request.form.get(key)
            if value:
                params[key] = value

        # Generate plot
        img_data, stats = generate_plot(params)
        
        if img_data is None:
            return jsonify({'success': False, 'error': stats.get('error', 'Unknown error')})
        
        return jsonify({
            'success': True,
            'image': img_data,
            'stats': stats,
            'params': params
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)