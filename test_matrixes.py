import numpy as np
import matplotlib.pyplot as plt
from safe_trig_powers import (
    safe_sin, safe_sin2, safe_sin4, safe_sin6, safe_sin8,
    safe_cos, safe_cos2, safe_cos4, safe_cos6, safe_cos8
)
"""
MIT License

Copyright (c) 2025 Roman Szewczyk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
def test_with_scalar():
    """Test safe functions with scalar values"""
    print("Testing with scalar values...")
    
    # Test points near critical values
    sin_test_value = 0.0001  # Near 0 where sin(x) is small
    cos_test_value = np.pi/2 + 0.0001  # Near π/2 where cos(x) is small
    
    # Sin tests
    print("\nSine function tests near x=0:")
    print(f"Regular sin^2: {np.sin(sin_test_value)**2:.16e}")
    print(f"Safe sin2:     {safe_sin2(sin_test_value):.16e}")
    print(f"Difference:    {abs(np.sin(sin_test_value)**2 - safe_sin2(sin_test_value)):.16e}")
    
    print(f"\nRegular sin^4: {np.sin(sin_test_value)**4:.16e}")
    print(f"Safe sin4:     {safe_sin4(sin_test_value):.16e}")
    print(f"Difference:    {abs(np.sin(sin_test_value)**4 - safe_sin4(sin_test_value)):.16e}")
    
    # Cos tests
    print("\nCosine function tests near x=π/2:")
    print(f"Regular cos^2: {np.cos(cos_test_value)**2:.16e}")
    print(f"Safe cos2:     {safe_cos2(cos_test_value):.16e}")
    print(f"Difference:    {abs(np.cos(cos_test_value)**2 - safe_cos2(cos_test_value)):.16e}")
    
    print(f"\nRegular cos^4: {np.cos(cos_test_value)**4:.16e}")
    print(f"Safe cos4:     {safe_cos4(cos_test_value):.16e}")
    print(f"Difference:    {abs(np.cos(cos_test_value)**4 - safe_cos4(cos_test_value)):.16e}")

def test_with_arrays():
    """Test safe functions with arrays"""
    print("\nTesting with array values...")
    
    # Create test arrays
    sin_test_array = np.linspace(-0.001, 0.001, 5)  # Near 0
    cos_test_array = np.linspace(np.pi/2 - 0.001, np.pi/2 + 0.001, 5)  # Near π/2
    
    # Sin tests
    print("\nSine function array tests near x=0:")
    print(f"Input array:   {sin_test_array}")
    print(f"Regular sin^2: {np.power(np.sin(sin_test_array), 2)}")
    print(f"Safe sin2:     {safe_sin2(sin_test_array)}")
    print(f"Max difference: {np.max(np.abs(np.power(np.sin(sin_test_array), 2) - safe_sin2(sin_test_array))):.16e}")
    
    # Cos tests
    print("\nCosine function array tests near x=π/2:")
    print(f"Input array:   {cos_test_array}")
    print(f"Regular cos^2: {np.power(np.cos(cos_test_array), 2)}")
    print(f"Safe cos2:     {safe_cos2(cos_test_array)}")
    print(f"Max difference: {np.max(np.abs(np.power(np.cos(cos_test_array), 2) - safe_cos2(cos_test_array))):.16e}")

def test_with_matrices():
    """Test safe functions with matrices"""
    print("\nTesting with matrix values...")
    
    # Create test matrices
    sin_test_matrix = np.array([
        [0.0001, 0.0002],
        [0.0003, 0.0004]
    ])  # Values near 0
    
    cos_test_matrix = np.array([
        [np.pi/2 + 0.0001, np.pi/2 + 0.0002],
        [np.pi/2 + 0.0003, np.pi/2 + 0.0004]
    ])  # Values near π/2
    
    # Sin matrix tests
    print("\nSine function matrix tests near x=0:")
    print(f"Input matrix:\n{sin_test_matrix}")
    print(f"Regular sin^2:\n{np.power(np.sin(sin_test_matrix), 2)}")
    print(f"Safe sin2:\n{safe_sin2(sin_test_matrix)}")
    print(f"Max difference: {np.max(np.abs(np.power(np.sin(sin_test_matrix), 2) - safe_sin2(sin_test_matrix))):.16e}")
    
    # Cos matrix tests
    print("\nCosine function matrix tests near x=π/2:")
    print(f"Input matrix:\n{cos_test_matrix}")
    print(f"Regular cos^2:\n{np.power(np.cos(cos_test_matrix), 2)}")
    print(f"Safe cos2:\n{safe_cos2(cos_test_matrix)}")
    print(f"Max difference: {np.max(np.abs(np.power(np.cos(cos_test_matrix), 2) - safe_cos2(cos_test_matrix))):.16e}")

def plot_differences():
    """Plot the differences between regular and safe functions"""
    # Set up plots for sine around x=0
    sin_center = 0
    sin_range = np.linspace(sin_center - 0.005, sin_center + 0.005, 1000)
    
    # Set up plots for cosine around x=π/2
    cos_center = np.pi/2
    cos_range = np.linspace(cos_center - 0.005, cos_center + 0.005, 1000)
    
    # Powers to check
    powers = [1, 2, 4, 6, 8]
    
    # Function mapping for safe versions
    sin_funcs = {1: safe_sin, 2: safe_sin2, 4: safe_sin4, 6: safe_sin6, 8: safe_sin8}
    cos_funcs = {1: safe_cos, 2: safe_cos2, 4: safe_cos4, 6: safe_cos6, 8: safe_cos8}
    
    # Create figure for sin differences
    fig_sin, axes_sin = plt.subplots(len(powers), 1, figsize=(12, 15), sharex=True)
    fig_sin.suptitle('Differences between safe_sin functions and numpy.sin powers around x=0', fontsize=16)
    
    # Create figure for cos differences
    fig_cos, axes_cos = plt.subplots(len(powers), 1, figsize=(12, 15), sharex=True)
    fig_cos.suptitle('Differences between safe_cos functions and numpy.cos powers around x=π/2', fontsize=16)
    
    # Calculate and plot differences for sine
    for i, power in enumerate(powers):
        # Calculate sine values
        numpy_sin = np.power(np.sin(sin_range), power) if power > 1 else np.sin(sin_range)
        safe_sin_vals = sin_funcs[power](sin_range)
        sin_diff = safe_sin_vals - numpy_sin
        
        # Plot sine differences
        ax = axes_sin[i]
        ax.plot(sin_range, sin_diff, 'b-', linewidth=2)
        ax.axvline(x=sin_center, color='r', linestyle=':', label='x=0')
        
        if power == 1:
            ax.set_title(f'Difference: safe_sin(x) - numpy.sin(x)')
        else:
            ax.set_title(f'Difference: safe_sin{power}(x) - numpy.sin(x)^{power}')
        
        ax.set_ylabel('Difference')
        ax.grid(True)
        ax.legend([f'Difference (power {power})', 'x=0'])
        
        # Add text with max difference
        max_diff = np.max(np.abs(sin_diff))
        ax.text(0.02, 0.90, f'Max diff: {max_diff:.2e}', transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Calculate and plot differences for cosine
    for i, power in enumerate(powers):
        # Calculate cosine values
        numpy_cos = np.power(np.cos(cos_range), power) if power > 1 else np.cos(cos_range)
        safe_cos_vals = cos_funcs[power](cos_range)
        cos_diff = safe_cos_vals - numpy_cos
        
        # Plot cosine differences
        ax = axes_cos[i]
        ax.plot(cos_range, cos_diff, 'g-', linewidth=2)
        ax.axvline(x=cos_center, color='r', linestyle=':', label='x=π/2')
        
        if power == 1:
            ax.set_title(f'Difference: safe_cos(x) - numpy.cos(x)')
        else:
            ax.set_title(f'Difference: safe_cos{power}(x) - numpy.cos(x)^{power}')
        
        ax.set_ylabel('Difference')
        ax.grid(True)
        ax.legend([f'Difference (power {power})', 'x=π/2'])
        
        # Add text with max difference
        max_diff = np.max(np.abs(cos_diff))
        ax.text(0.02, 0.90, f'Max diff: {max_diff:.2e}', transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add common x-labels
    fig_sin.text(0.5, 0.04, 'Deviation from x=0', ha='center', fontsize=14)
    fig_cos.text(0.5, 0.04, 'Deviation from x=π/2', ha='center', fontsize=14)
    
    # Adjust layout
    plt.figure(fig_sin.number)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.figure(fig_cos.number)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    # Run all tests
    test_with_scalar()
    test_with_arrays()
    test_with_matrices()
    plot_differences()