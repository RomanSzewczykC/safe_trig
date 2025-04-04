import numpy as np
from tabulate import tabulate

# Import the functions from the uploaded code
from safe_trig_powers import (
    stable_mod_pi, stable_mod_half_pi, 
    safe_sin, safe_sin2, safe_sin4, safe_sin6, safe_sin8,
    safe_cos, safe_cos2, safe_cos4, safe_cos6, safe_cos8
)

def test_critical_points():
    """Test all functions at critical points and problematic values"""
    # Critical points to test (expressed as multiples of π)
    critical_points_pi = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5])
    
    # Convert to actual values in radians
    test_points = critical_points_pi * np.pi
    
    # Create tables to store results
    sin_results = []
    cos_results = []
    
    for i, x in enumerate(test_points):
        # Calculate reference values using NumPy
        np_sin = np.sin(x)
        np_cos = np.cos(x)
        
        # Calculate values using our safe functions
        s_sin = safe_sin(x)
        s_cos = safe_cos(x)
        
        # Calculate absolute differences
        sin_diff = abs(np_sin - s_sin)
        cos_diff = abs(np_cos - s_cos)
        
        # Add to results
        sin_results.append([
            f"{critical_points_pi[i]}π",
            f"{np_sin:.10f}",
            f"{s_sin:.10f}",
            f"{sin_diff:.10e}",
            "✓" if sin_diff < 1e-10 else "✗"
        ])
        
        cos_results.append([
            f"{critical_points_pi[i]}π",
            f"{np_cos:.10f}",
            f"{s_cos:.10f}",
            f"{cos_diff:.10e}",
            "✓" if cos_diff < 1e-10 else "✗"
        ])
    
    # Print results in tabular format
    print("SINE FUNCTION RESULTS:")
    print(tabulate(
        sin_results,
        headers=["x", "np.sin(x)", "safe_sin(x)", "Abs. Diff.", "Pass"],
        tablefmt="grid"
    ))
    
    print("\nCOSINE FUNCTION RESULTS:")
    print(tabulate(
        cos_results,
        headers=["x", "np.cos(x)", "safe_cos(x)", "Abs. Diff.", "Pass"],
        tablefmt="grid"
    ))

def test_additional_points():
    """Test specifically requested additional points"""
    # Additional points to test
    additional_points = [
        (4 * np.pi + np.pi/4, "4π+π/4"),
        (5 * np.pi + np.pi/4, "5π+π/4"),
        (-9 * np.pi + np.pi/4, "-9π+π/4"),
        (-10 * np.pi + np.pi/4, "-10π+π/4")
    ]
    
    print("\nTESTING ADDITIONAL POINTS:")
    
    # Create tables to store results
    results = []
    
    for x, label in additional_points:
        # Calculate reference values using NumPy
        np_sin = np.sin(x)
        np_cos = np.cos(x)
        np_sin2 = np_sin**2
        np_cos2 = np_cos**2
        
        # Calculate values using our safe functions
        s_sin = safe_sin(x)
        s_cos = safe_cos(x)
        s_sin2 = safe_sin2(x)
        s_cos2 = safe_cos2(x)
        
        # Calculate absolute differences
        sin_diff = abs(np_sin - s_sin)
        cos_diff = abs(np_cos - s_cos)
        sin2_diff = abs(np_sin2 - s_sin2)
        cos2_diff = abs(np_cos2 - s_cos2)
        
        # Add to results
        results.append([
            label,
            f"{np_sin:.8f}",
            f"{s_sin:.8f}",
            f"{sin_diff:.2e}",
            f"{np_cos:.8f}",
            f"{s_cos:.8f}",
            f"{cos_diff:.2e}",
            f"{sin2_diff:.2e}",
            f"{cos2_diff:.2e}",
            "✓" if max(sin_diff, cos_diff, sin2_diff, cos2_diff) < 1e-10 else "✗"
        ])
    
    # Print results in tabular format
    print(tabulate(
        results,
        headers=["Point", "np.sin(x)", "safe_sin(x)", "Sin Diff", "np.cos(x)", "safe_cos(x)", "Cos Diff", "Sin² Diff", "Cos² Diff", "Pass"],
        tablefmt="grid"
    ))

def test_wrapper_functions():
    """Test the wrapper functions that find closest multiples of π and π/2"""
    test_points_pi = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5]) * np.pi
    
    print("\nTESTING WRAPPER FUNCTIONS:")
    results = []
    
    for x in test_points_pi:
        # Calculate the closest multiple of π
        closest_pi = stable_mod_pi(x)
        
        # Calculate the closest odd multiple of π/2
        closest_half_pi = stable_mod_half_pi(x)
        
        # Add to results
        results.append([
            f"{x/np.pi:.2f}π",
            f"{closest_pi/np.pi:.2f}π",
            f"{closest_half_pi/np.pi:.2f}π"
        ])
    
    print(tabulate(
        results,
        headers=["x", "closest_pi", "closest_half_pi"],
        tablefmt="grid"
    ))

def test_with_vectors():
    """Test the functions with vector inputs."""
    # Vector of problematic points
    vec = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5]) * np.pi
    
    # NumPy reference
    np_sin_vec = np.sin(vec)
    np_cos_vec = np.cos(vec)
    
    # Our functions
    safe_sin_vec = safe_sin(vec)
    safe_cos_vec = safe_cos(vec)
    
    # Calculate differences
    sin_diff = np.abs(np_sin_vec - safe_sin_vec)
    cos_diff = np.abs(np_cos_vec - safe_cos_vec)
    
    print("\nVECTOR INPUT TESTS:")
    print(f"Maximum sin difference: {np.max(sin_diff):.10e}")
    print(f"Maximum cos difference: {np.max(cos_diff):.10e}")
    
    if np.max(sin_diff) < 1e-10 and np.max(cos_diff) < 1e-10:
        print("✓ Vector tests passed!")
    else:
        print("✗ Vector tests failed!")

def test_with_matrices():
    """Test the functions with matrix inputs."""
    # Create a 3x3 matrix of problematic points
    matrix = np.array([
        [-9.5, -7, -6],
        [-2, 0, 2],
        [6, 7, 9.5]
    ]) * np.pi
    
    # NumPy reference
    np_sin_mat = np.sin(matrix)
    np_cos_mat = np.cos(matrix)
    
    # Our functions
    safe_sin_mat = safe_sin(matrix)
    safe_cos_mat = safe_cos(matrix)
    
    # Calculate differences
    sin_diff = np.abs(np_sin_mat - safe_sin_mat)
    cos_diff = np.abs(np_cos_mat - safe_cos_mat)
    
    print("\nMATRIX INPUT TESTS:")
    print(f"Maximum sin difference: {np.max(sin_diff):.10e}")
    print(f"Maximum cos difference: {np.max(cos_diff):.10e}")
    
    if np.max(sin_diff) < 1e-10 and np.max(cos_diff) < 1e-10:
        print("✓ Matrix tests passed!")
    else:
        print("✗ Matrix tests failed!")

def test_small_offsets():
    """Test with very small offsets from critical points."""
    # Base points (specified critical points)
    base_points = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5]) * np.pi
    
    # Include π/2 critical points as well (for cosine)
    half_pi_points = np.array([-9.5, -7.5, -6.5, -1.5, -0.5, 0.5, 1.5, 6.5, 7.5, 9.5]) * np.pi
    
    # Very small offsets to test numerical stability
    offsets = np.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, -1e-12, -1e-10, -1e-8, -1e-6, -1e-4])
    
    print("\nSUMMARY: SMALL OFFSETS FROM CRITICAL POINTS")
    
    # Test sin around multiples of π
    max_sin_diff = 0
    max_cos_diff = 0
    failures = 0
    
    for base in base_points:
        for offset in offsets:
            x = base + offset
            
            # NumPy reference
            np_sin = np.sin(x)
            
            # Our function
            s_sin = safe_sin(x)
            
            # Calculate difference
            sin_diff = abs(np_sin - s_sin)
            
            # Track worst case
            if sin_diff > max_sin_diff:
                max_sin_diff = sin_diff
            
            # Count failures
            if sin_diff > 1e-10:
                failures += 1
    
    # Test cos around odd multiples of π/2
    for base in half_pi_points:
        for offset in offsets:
            x = base + offset
            
            # NumPy reference
            np_cos = np.cos(x)
            
            # Our function
            s_cos = safe_cos(x)
            
            # Calculate difference
            cos_diff = abs(np_cos - s_cos)
            
            # Track worst case
            if cos_diff > max_cos_diff:
                max_cos_diff = cos_diff
            
            # Count failures
            if cos_diff > 1e-10:
                failures += 1
    
    # Compile summary results
    summary_results = [
        ["Small Offsets Test", f"{max_sin_diff:.2e}", f"{max_cos_diff:.2e}", failures, "✓" if failures == 0 else "✗"]
    ]
    
    print(tabulate(
        summary_results,
        headers=["Test Case", "Max Sin Error", "Max Cos Error", "Failures", "Overall"],
        tablefmt="grid"
    ))

def test_power_functions():
    """Test the power functions (sin², sin⁴, etc.)"""
    # Critical points to test
    test_points = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5]) * np.pi
    
    print("\nTESTING POWER FUNCTIONS:")
    power_results = []
    
    for x in test_points:
        # NumPy reference
        np_sin = np.sin(x)
        np_sin2 = np_sin**2
        np_sin4 = np_sin**4
        np_sin6 = np_sin**6
        np_sin8 = np_sin**8
        
        # Our functions
        s_sin2 = safe_sin2(x)
        s_sin4 = safe_sin4(x)
        s_sin6 = safe_sin6(x)
        s_sin8 = safe_sin8(x)
        
        # Calculate differences
        sin2_diff = abs(np_sin2 - s_sin2)
        sin4_diff = abs(np_sin4 - s_sin4)
        sin6_diff = abs(np_sin6 - s_sin6)
        sin8_diff = abs(np_sin8 - s_sin8)
        
        # Add to results
        power_results.append([
            f"{x/np.pi:.2f}π",
            f"{sin2_diff:.1e}",
            f"{sin4_diff:.1e}",
            f"{sin6_diff:.1e}",
            f"{sin8_diff:.1e}"
        ])
    
    print(tabulate(
        power_results,
        headers=["x", "sin² Diff", "sin⁴ Diff", "sin⁶ Diff", "sin⁸ Diff"],
        tablefmt="grid"
    ))

    # Similar testing for cosine powers
    power_results = []
    
    for x in test_points:
        # NumPy reference
        np_cos = np.cos(x)
        np_cos2 = np_cos**2
        np_cos4 = np_cos**4
        np_cos6 = np_cos**6
        np_cos8 = np_cos**8
        
        # Our functions
        s_cos2 = safe_cos2(x)
        s_cos4 = safe_cos4(x)
        s_cos6 = safe_cos6(x)
        s_cos8 = safe_cos8(x)
        
        # Calculate differences
        cos2_diff = abs(np_cos2 - s_cos2)
        cos4_diff = abs(np_cos4 - s_cos4)
        cos6_diff = abs(np_cos6 - s_cos6)
        cos8_diff = abs(np_cos8 - s_cos8)
        
        # Add to results
        power_results.append([
            f"{x/np.pi:.2f}π",
            f"{cos2_diff:.1e}",
            f"{cos4_diff:.1e}",
            f"{cos6_diff:.1e}",
            f"{cos8_diff:.1e}"
        ])
    
    print(tabulate(
        power_results,
        headers=["x", "cos² Diff", "cos⁴ Diff", "cos⁶ Diff", "cos⁸ Diff"],
        tablefmt="grid"
    ))

def diagnose_issues():
    """Identify and diagnose any issues with the functions."""
    # Problem cases
    problem_points = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5]) * np.pi
    
    # Run detailed diagnostics
    print("\nDIAGNOSTIC INFORMATION:")
    
    for x in problem_points:
        print(f"\nDiagnostics for x = {x/np.pi:.2f}π:")
        
        # For sin function
        x_mod_sin = (x + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
        closest_pi = stable_mod_pi(x_mod_sin)
        delta_sin = x_mod_sin - closest_pi
        
        print(f"  sin diagnostics:")
        print(f"    x_mod = {x_mod_sin/np.pi:.4f}π")
        print(f"    closest_pi = {closest_pi/np.pi:.2f}π")
        print(f"    delta = {delta_sin/np.pi:.4f}π")
        print(f"    np.sin(x) = {np.sin(x):.10f}")
        print(f"    safe_sin(x) = {safe_sin(x):.10f}")
        
        # For cos function
        x_mod_cos = (x + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
        closest_half_pi = stable_mod_half_pi(x_mod_cos)
        delta_cos = x_mod_cos - closest_half_pi
        
        print(f"  cos diagnostics:")
        print(f"    x_mod = {x_mod_cos/np.pi:.4f}π")
        print(f"    closest_half_pi = {closest_half_pi/np.pi:.2f}π")
        print(f"    delta = {delta_cos/np.pi:.4f}π")
        print(f"    np.cos(x) = {np.cos(x):.10f}")
        print(f"    safe_cos(x) = {safe_cos(x):.10f}")

def suggest_fixes():
    """Suggest potential fixes for any issues found."""
    print("\nPOTENTIAL ISSUES AND FIXES:")
    
    # Test specific issue with -9.5π
    x = -9.5 * np.pi
    np_sin = np.sin(x)
    s_sin = safe_sin(x)
    sin_diff = abs(np_sin - s_sin)
    
    if sin_diff > 1e-10:
        print(f"Issue with safe_sin at x = -9.5π:")
        print(f"  np.sin(-9.5π) = {np_sin:.10f}")
        print(f"  safe_sin(-9.5π) = {s_sin:.10f}")
        print(f"  Difference: {sin_diff:.10e}")
        
        # Diagnostic info
        x_mod = (x + np.pi) % (2 * np.pi) - np.pi
        closest_pi = stable_mod_pi(x_mod)
        delta = x_mod - closest_pi
        near_critical = abs(delta) < 0.001
        
        print(f"  x_mod = {x_mod/np.pi:.4f}π")
        print(f"  closest_pi = {closest_pi/np.pi:.2f}π")
        print(f"  delta = {delta/np.pi:.4f}π")
        print(f"  near_critical = {near_critical}")
        
        # Suggested fix for sin
        print("  Potential fix: The 'near_critical' threshold may need adjustment")
    
    # Test specific issue with multiples of π
    test_points = np.array([-7, -6, -2, 0, 2, 6, 7]) * np.pi
    for x in test_points:
        np_sin = np.sin(x)
        s_sin = safe_sin(x)
        sin_diff = abs(np_sin - s_sin)
        
        if sin_diff > 1e-10:
            print(f"Issue with safe_sin at x = {x/np.pi:.0f}π:")
            print(f"  np.sin({x/np.pi:.0f}π) = {np_sin:.10f}")
            print(f"  safe_sin({x/np.pi:.0f}π) = {s_sin:.10f}")
            print(f"  Difference: {sin_diff:.10e}")

def run_summary_tests():
    """Run all tests and display only summary tables"""
    print("=" * 80)
    print("TRIGONOMETRIC STABILITY TEST SUMMARY")
    print("=" * 80)
    
    # Test results storage
    summary_results = []
    
    # Test critical points specified by user
    critical_points_pi = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5])
    test_points = critical_points_pi * np.pi
    
    # Additional test points
    additional_points = [
        4 * np.pi + np.pi/4,
        5 * np.pi + np.pi/4,
        -9 * np.pi + np.pi/4,
        -10 * np.pi + np.pi/4
    ]
    
    # Basic sin/cos test
    sin_failures = 0
    cos_failures = 0
    max_sin_diff = 0
    max_cos_diff = 0
    
    # Test critical points
    for x in test_points:
        np_sin = np.sin(x)
        np_cos = np.cos(x)
        s_sin = safe_sin(x)
        s_cos = safe_cos(x)
        
        sin_diff = abs(np_sin - s_sin)
        cos_diff = abs(np_cos - s_cos)
        
        max_sin_diff = max(max_sin_diff, sin_diff)
        max_cos_diff = max(max_cos_diff, cos_diff)
        
        if sin_diff > 1e-10:
            sin_failures += 1
        if cos_diff > 1e-10:
            cos_failures += 1
            
    # Test additional points
    for x in additional_points:
        np_sin = np.sin(x)
        np_cos = np.cos(x)
        s_sin = safe_sin(x)
        s_cos = safe_cos(x)
        
        sin_diff = abs(np_sin - s_sin)
        cos_diff = abs(np_cos - s_cos)
        
        max_sin_diff = max(max_sin_diff, sin_diff)
        max_cos_diff = max(max_cos_diff, cos_diff)
        
        if sin_diff > 1e-10:
            sin_failures += 1
        if cos_diff > 1e-10:
            cos_failures += 1
    
    summary_results.append([
        "Basic Functions",
        f"{max_sin_diff:.2e}",
        sin_failures,
        f"{max_cos_diff:.2e}",
        cos_failures,
        "✓" if sin_failures + cos_failures == 0 else "✗"
    ])
    
    # Vector input test
    vec = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5]) * np.pi
    np_sin_vec = np.sin(vec)
    np_cos_vec = np.cos(vec)
    safe_sin_vec = safe_sin(vec)
    safe_cos_vec = safe_cos(vec)
    
    sin_vec_diff = np.max(np.abs(np_sin_vec - safe_sin_vec))
    cos_vec_diff = np.max(np.abs(np_cos_vec - safe_cos_vec))
    
    summary_results.append([
        "Vector Input",
        f"{sin_vec_diff:.2e}",
        "0" if sin_vec_diff < 1e-10 else "1+",
        f"{cos_vec_diff:.2e}",
        "0" if cos_vec_diff < 1e-10 else "1+",
        "✓" if max(sin_vec_diff, cos_vec_diff) < 1e-10 else "✗"
    ])
    
    # Matrix input test
    matrix = np.array([
        [-9.5, -7, -6],
        [-2, 0, 2],
        [6, 7, 9.5]
    ]) * np.pi
    
    np_sin_mat = np.sin(matrix)
    np_cos_mat = np.cos(matrix)
    safe_sin_mat = safe_sin(matrix)
    safe_cos_mat = safe_cos(matrix)
    
    sin_mat_diff = np.max(np.abs(np_sin_mat - safe_sin_mat))
    cos_mat_diff = np.max(np.abs(np_cos_mat - safe_cos_mat))
    
    summary_results.append([
        "Matrix Input",
        f"{sin_mat_diff:.2e}",
        "0" if sin_mat_diff < 1e-10 else "1+",
        f"{cos_mat_diff:.2e}",
        "0" if cos_mat_diff < 1e-10 else "1+",
        "✓" if max(sin_mat_diff, cos_mat_diff) < 1e-10 else "✗"
    ])
    
    # Small offsets test
    base_points = np.array([-9.5, -7, -6, -2, 0, 2, 6, 7, 9.5]) * np.pi
    half_pi_points = np.array([-9.5, -7.5, -6.5, -1.5, -0.5, 0.5, 1.5, 6.5, 7.5, 9.5]) * np.pi
    offsets = np.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, -1e-12, -1e-10, -1e-8, -1e-6, -1e-4])
    
    max_sin_offset_diff = 0
    max_cos_offset_diff = 0
    offset_failures = 0
    
    for base in base_points:
        for offset in offsets:
            x = base + offset
            np_sin = np.sin(x)
            s_sin = safe_sin(x)
            sin_diff = abs(np_sin - s_sin)
            max_sin_offset_diff = max(max_sin_offset_diff, sin_diff)
            if sin_diff > 1e-10:
                offset_failures += 1
    
    for base in half_pi_points:
        for offset in offsets:
            x = base + offset
            np_cos = np.cos(x)
            s_cos = safe_cos(x)
            cos_diff = abs(np_cos - s_cos)
            max_cos_offset_diff = max(max_cos_offset_diff, cos_diff)
            if cos_diff > 1e-10:
                offset_failures += 1
    
    summary_results.append([
        "Small Offsets",
        f"{max_sin_offset_diff:.2e}",
        "N/A",
        f"{max_cos_offset_diff:.2e}",
        "N/A",
        "✓" if offset_failures == 0 else "✗"
    ])
    
    # Power functions test
    power_failures = 0
    max_sin_power_diff = np.zeros(4)  # For sin², sin⁴, sin⁶, sin⁸
    max_cos_power_diff = np.zeros(4)  # For cos², cos⁴, cos⁶, cos⁸
    
    for x in test_points:
        # Sine powers
        np_sin = np.sin(x)
        np_sin2 = np_sin**2
        np_sin4 = np_sin**4
        np_sin6 = np_sin**6
        np_sin8 = np_sin**8
        
        s_sin2 = safe_sin2(x)
        s_sin4 = safe_sin4(x)
        s_sin6 = safe_sin6(x)
        s_sin8 = safe_sin8(x)
        
        sin2_diff = abs(np_sin2 - s_sin2)
        sin4_diff = abs(np_sin4 - s_sin4)
        sin6_diff = abs(np_sin6 - s_sin6)
        sin8_diff = abs(np_sin8 - s_sin8)
        
        max_sin_power_diff[0] = max(max_sin_power_diff[0], sin2_diff)
        max_sin_power_diff[1] = max(max_sin_power_diff[1], sin4_diff)
        max_sin_power_diff[2] = max(max_sin_power_diff[2], sin6_diff)
        max_sin_power_diff[3] = max(max_sin_power_diff[3], sin8_diff)
        
        if sin2_diff > 1e-10 or sin4_diff > 1e-10 or sin6_diff > 1e-10 or sin8_diff > 1e-10:
            power_failures += 1
        
        # Cosine powers
        np_cos = np.cos(x)
        np_cos2 = np_cos**2
        np_cos4 = np_cos**4
        np_cos6 = np_cos**6
        np_cos8 = np_cos**8
        
        s_cos2 = safe_cos2(x)
        s_cos4 = safe_cos4(x)
        s_cos6 = safe_cos6(x)
        s_cos8 = safe_cos8(x)
        
        cos2_diff = abs(np_cos2 - s_cos2)
        cos4_diff = abs(np_cos4 - s_cos4)
        cos6_diff = abs(np_cos6 - s_cos6)
        cos8_diff = abs(np_cos8 - s_cos8)
        
        max_cos_power_diff[0] = max(max_cos_power_diff[0], cos2_diff)
        max_cos_power_diff[1] = max(max_cos_power_diff[1], cos4_diff)
        max_cos_power_diff[2] = max(max_cos_power_diff[2], cos6_diff)
        max_cos_power_diff[3] = max(max_cos_power_diff[3], cos8_diff)
        
        if cos2_diff > 1e-10 or cos4_diff > 1e-10 or cos6_diff > 1e-10 or cos8_diff > 1e-10:
            power_failures += 1
    
    summary_results.append([
        "Power Functions sin²",
        f"{max_sin_power_diff[0]:.2e}",
        "N/A",
        f"{max_cos_power_diff[0]:.2e}",
        "N/A",
        "✓" if max(max_sin_power_diff[0], max_cos_power_diff[0]) < 1e-10 else "✗"
    ])
    
    summary_results.append([
        "Power Functions sin⁴/cos⁴",
        f"{max_sin_power_diff[1]:.2e}",
        "N/A",
        f"{max_cos_power_diff[1]:.2e}",
        "N/A",
        "✓" if max(max_sin_power_diff[1], max_cos_power_diff[1]) < 1e-10 else "✗"
    ])
    
    summary_results.append([
        "Power Functions sin⁶/cos⁶",
        f"{max_sin_power_diff[2]:.2e}",
        "N/A",
        f"{max_cos_power_diff[2]:.2e}",
        "N/A",
        "✓" if max(max_sin_power_diff[2], max_cos_power_diff[2]) < 1e-10 else "✗"
    ])
    
    summary_results.append([
        "Power Functions sin⁸/cos⁸",
        f"{max_sin_power_diff[3]:.2e}",
        "N/A",
        f"{max_cos_power_diff[3]:.2e}",
        "N/A",
        "✓" if max(max_sin_power_diff[3], max_cos_power_diff[3]) < 1e-10 else "✗"
    ])
    
    # Print the final summary table
    print(tabulate(
        summary_results,
        headers=["Test Case", "Max Sin Error", "Sin Fails", "Max Cos Error", "Cos Fails", "Pass"],
        tablefmt="grid"
    ))

def test_additional_points_with_variations():
    """Test small variations around the additional points"""
    # Additional points to test
    base_points = [
        (4 * np.pi + np.pi/4, "4π+π/4"),
        (5 * np.pi + np.pi/4, "5π+π/4"),
        (-9 * np.pi + np.pi/4, "-9π+π/4"),
        (-10 * np.pi + np.pi/4, "-10π+π/4")
    ]
    
    # Small offsets to test
    offsets = np.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, -1e-12, -1e-10, -1e-8, -1e-6, -1e-4])
    
    print("\nTESTING SMALL VARIATIONS AROUND ADDITIONAL POINTS:")
    variation_results = []
    
    for base, label in base_points:
        max_sin_diff = 0
        max_cos_diff = 0
        max_sin2_diff = 0
        max_cos2_diff = 0
        
        for offset in offsets:
            x = base + offset
            
            # Calculate reference values
            np_sin = np.sin(x)
            np_cos = np.cos(x)
            np_sin2 = np_sin**2
            np_cos2 = np_cos**2
            
            # Calculate safe values
            s_sin = safe_sin(x)
            s_cos = safe_cos(x)
            s_sin2 = safe_sin2(x)
            s_cos2 = safe_cos2(x)
            
            # Calculate differences
            sin_diff = abs(np_sin - s_sin)
            cos_diff = abs(np_cos - s_cos)
            sin2_diff = abs(np_sin2 - s_sin2)
            cos2_diff = abs(np_cos2 - s_cos2)
            
            # Track maximum differences
            max_sin_diff = max(max_sin_diff, sin_diff)
            max_cos_diff = max(max_cos_diff, cos_diff)
            max_sin2_diff = max(max_sin2_diff, sin2_diff)
            max_cos2_diff = max(max_cos2_diff, cos2_diff)
        
        # Add to results
        variation_results.append([
            label,
            f"{max_sin_diff:.2e}",
            f"{max_cos_diff:.2e}",
            f"{max_sin2_diff:.2e}",
            f"{max_cos2_diff:.2e}",
            "✓" if max(max_sin_diff, max_cos_diff, max_sin2_diff, max_cos2_diff) < 1e-10 else "✗"
        ])
    
    # Print results in tabular format
    print(tabulate(
        variation_results,
        headers=["Point", "Max Sin Diff", "Max Cos Diff", "Max Sin² Diff", "Max Cos² Diff", "Pass"],
        tablefmt="grid"
    ))

if __name__ == "__main__":
    print("=" * 80)
    print("TRIGONOMETRIC STABILITY TEST SUITE")
    print("=" * 80)
    
    # Run individual tests
    test_critical_points()
    test_additional_points()
    test_wrapper_functions()
    test_with_vectors()
    test_with_matrices()
    test_small_offsets()
    test_power_functions()
    test_additional_points_with_variations()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Run summary
    run_summary_tests()