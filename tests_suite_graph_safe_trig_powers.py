import numpy as np
import time
import matplotlib.pyplot as plt
from safe_trig_powers import (
    safe_sin, safe_sin2, safe_sin4, safe_sin6, safe_sin8,
    safe_cos, safe_cos2, safe_cos4, safe_cos6, safe_cos8
)
import seaborn as sns
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
def test_sin_critical_points():
    """Test sin functions near critical points: 0, π, 2π, -2π and multiples by 3, 10."""
    print("\n========== Sin Functions Critical Points Testing ==========")
    
    # Generate critical points for sin
    base_points = [0, 1, 2, -2]  # Multiples of π
    multipliers = [1, 3, 10]
    critical_points = []
    
    for point in base_points:
        for mult in multipliers:
            critical_points.append(point * mult * np.pi)
    
    # Small perturbations
    perturbations = [1e-15, 1e-12, 1e-9, 1e-6]
    
    # Functions to test
    sin_funcs = [
        ("sin(x)", lambda x: np.sin(x), safe_sin),
        ("sin²(x)", lambda x: np.sin(x)**2, safe_sin2),
        ("sin⁴(x)", lambda x: np.sin(x)**4, safe_sin4),
        ("sin⁶(x)", lambda x: np.sin(x)**6, safe_sin6),
        ("sin⁸(x)", lambda x: np.sin(x)**8, safe_sin8)
    ]
    
    # Results summary
    summary = {}
    
    for point in critical_points:
        # Express point as multiple of π
        pi_multiple = point / np.pi
        point_key = f"{pi_multiple:.2f}π"
        summary[point_key] = {}
        
        for func_name, std_func, safe_func in sin_funcs:
            max_abs_diff = 0
            max_rel_diff = 0
            worst_perturb = 0
            
            for perturb in perturbations:
                test_x = point + perturb
                std_result = std_func(test_x)
                safe_result = safe_func(test_x)
                abs_diff = abs(std_result - safe_result)
                
                # Calculate relative difference (avoid div by 0)
                if abs(std_result) > 1e-15:
                    rel_diff = abs_diff / abs(std_result)
                else:
                    rel_diff = 0.0 if abs_diff < 1e-15 else float('inf')
                
                if abs_diff > max_abs_diff:
                    max_abs_diff = abs_diff
                    max_rel_diff = rel_diff
                    worst_perturb = perturb
            
            summary[point_key][func_name] = {
                "max_abs_diff": max_abs_diff,
                "max_rel_diff": max_rel_diff,
                "worst_perturb": worst_perturb
            }
    
    # Print summary
    print("\n--- Sin Critical Points Error Summary ---")
    for point, funcs in summary.items():
        print(f"\nPoint x = {point}:")
        for func_name, errors in funcs.items():
            print(f"  {func_name:<8}: Max Abs Error: {errors['max_abs_diff']:.2e}, "
                  f"Max Rel Error: {errors['max_rel_diff']:.2e} at ±{errors['worst_perturb']:.2e}")
    
    return summary


def test_cos_critical_points():
    """Test cos functions near critical points: π/2, -π/2, 3π/2, -3π/2 and multiples by 3, 10."""
    print("\n========== Cos Functions Critical Points Testing ==========")
    
    # Generate critical points for cos
    base_points = [0.5, -0.5, 1.5, -1.5]  # Multiples of π
    multipliers = [1, 3, 10]
    critical_points = []
    
    for point in base_points:
        for mult in multipliers:
            critical_points.append(point * mult * np.pi)
    
    # Small perturbations
    perturbations = [1e-15, 1e-12, 1e-9, 1e-6]
    
    # Functions to test
    cos_funcs = [
        ("cos(x)", lambda x: np.cos(x), safe_cos),
        ("cos²(x)", lambda x: np.cos(x)**2, safe_cos2),
        ("cos⁴(x)", lambda x: np.cos(x)**4, safe_cos4),
        ("cos⁶(x)", lambda x: np.cos(x)**6, safe_cos6),
        ("cos⁸(x)", lambda x: np.cos(x)**8, safe_cos8)
    ]
    
    # Results summary
    summary = {}
    
    for point in critical_points:
        # Express point as multiple of π
        pi_multiple = point / np.pi
        point_key = f"{pi_multiple:.2f}π"
        summary[point_key] = {}
        
        for func_name, std_func, safe_func in cos_funcs:
            max_abs_diff = 0
            max_rel_diff = 0
            worst_perturb = 0
            
            for perturb in perturbations:
                test_x = point + perturb
                std_result = std_func(test_x)
                safe_result = safe_func(test_x)
                abs_diff = abs(std_result - safe_result)
                
                # Calculate relative difference (avoid div by 0)
                if abs(std_result) > 1e-15:
                    rel_diff = abs_diff / abs(std_result)
                else:
                    rel_diff = 0.0 if abs_diff < 1e-15 else float('inf')
                
                if abs_diff > max_abs_diff:
                    max_abs_diff = abs_diff
                    max_rel_diff = rel_diff
                    worst_perturb = perturb
            
            summary[point_key][func_name] = {
                "max_abs_diff": max_abs_diff,
                "max_rel_diff": max_rel_diff,
                "worst_perturb": worst_perturb
            }
    
    # Print summary
    print("\n--- Cos Critical Points Error Summary ---")
    for point, funcs in summary.items():
        print(f"\nPoint x = {point}:")
        for func_name, errors in funcs.items():
            print(f"  {func_name:<8}: Max Abs Error: {errors['max_abs_diff']:.2e}, "
                  f"Max Rel Error: {errors['max_rel_diff']:.2e} at ±{errors['worst_perturb']:.2e}")
    
    return summary


def test_matrix_and_vector():
    """Test the functions with a 10x10 matrix and a vector of 20 elements."""
    print("\n========== Matrix and Vector Testing ==========")
    
    # Create matrix 10x10 and vector of 20 elements
    matrix_size = (10, 10)
    vector_size = 20
    
    # Create data with a mix of random points and critical points
    sin_critical = [0, np.pi, 2*np.pi, -2*np.pi]
    cos_critical = [np.pi/2, -np.pi/2, 3*np.pi/2, -3*np.pi/2]
    all_critical = sin_critical + cos_critical
    
    # Create matrix with random values
    matrix = np.random.uniform(-4*np.pi, 4*np.pi, matrix_size)
    
    # Insert some critical points in the diagonal
    for i in range(min(matrix_size)):
        if i < len(all_critical):
            matrix[i, i] = all_critical[i]
    
    # Create vector with random values and critical points
    vector = np.random.uniform(-4*np.pi, 4*np.pi, vector_size)
    for i in range(min(vector_size, len(all_critical))):
        vector[i] = all_critical[i]
    
    # Test functions
    funcs = [
        ("sin", np.sin, safe_sin),
        ("sin²", lambda x: np.sin(x)**2, safe_sin2),
        ("sin⁴", lambda x: np.sin(x)**4, safe_sin4),
        ("sin⁶", lambda x: np.sin(x)**6, safe_sin6),
        ("sin⁸", lambda x: np.sin(x)**8, safe_sin8),
        ("cos", np.cos, safe_cos),
        ("cos²", lambda x: np.cos(x)**2, safe_cos2),
        ("cos⁴", lambda x: np.cos(x)**4, safe_cos4),
        ("cos⁶", lambda x: np.cos(x)**6, safe_cos6),
        ("cos⁸", lambda x: np.cos(x)**8, safe_cos8)
    ]
    
    # Apply vectorized functions for matrix
    def apply_to_matrix(func, mat):
        """Apply a function to each element of a matrix."""
        result = np.zeros_like(mat, dtype=float)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                result[i, j] = func(mat[i, j])
        return result
    
    # Apply vectorized functions for vector
    def apply_to_vector(func, vec):
        """Apply a function to each element of a vector."""
        return np.array([func(x) for x in vec])
    
    # Test matrix
    print("\n--- Matrix 10x10 Results ---")
    matrix_results = {}
    
    for func_name, std_func, safe_func in funcs:
        # Standard numpy result
        std_matrix = apply_to_matrix(std_func, matrix)
        
        # Safe function result
        safe_matrix = apply_to_matrix(safe_func, matrix)
        
        # Calculate max difference
        max_diff = np.max(np.abs(std_matrix - safe_matrix))
        has_nan = np.isnan(std_matrix).any() or np.isnan(safe_matrix).any()
        has_inf = np.isinf(std_matrix).any() or np.isinf(safe_matrix).any()
        
        matrix_results[func_name] = {
            "max_diff": max_diff,
            "has_nan": has_nan,
            "has_inf": has_inf
        }
        
        print(f"  {func_name:<5}: Max Diff: {max_diff:.2e}, NaNs: {has_nan}, Infs: {has_inf}")
    
    # Test vector
    print("\n--- Vector (20 elements) Results ---")
    vector_results = {}
    
    for func_name, std_func, safe_func in funcs:
        # Standard numpy result
        std_vector = apply_to_vector(std_func, vector)
        
        # Safe function result
        safe_vector = apply_to_vector(safe_func, vector)
        
        # Calculate max difference
        max_diff = np.max(np.abs(std_vector - safe_vector))
        has_nan = np.isnan(std_vector).any() or np.isnan(safe_vector).any()
        has_inf = np.isinf(std_vector).any() or np.isinf(safe_vector).any()
        
        vector_results[func_name] = {
            "max_diff": max_diff,
            "has_nan": has_nan,
            "has_inf": has_inf
        }
        
        print(f"  {func_name:<5}: Max Diff: {max_diff:.2e}, NaNs: {has_nan}, Infs: {has_inf}")
    
    return matrix_results, vector_results


def benchmark_performance():
    """Benchmark the performance of standard numpy vs safe implementations."""
    print("\n========== Performance Benchmarking ==========")
    
    # Critical points for testing
    sin_critical = [0, np.pi, 2*np.pi, -2*np.pi]
    cos_critical = [np.pi/2, -np.pi/2, 3*np.pi/2, -3*np.pi/2]
    
    # Small perturbations to add to critical points
    perturbations = [1e-15, 1e-12, 1e-9, 1e-6]
    
    # Array sizes to test - start small and increase to maximum size
    sizes = [1000, 5000, 10000]
    num_iterations = 10  # Number of repetitions to get more accurate timing
    
    # Functions to test
    funcs = [
        ("sin", np.sin, lambda x: np.array([safe_sin(xi) for xi in x])),
        ("sin²", lambda x: np.sin(x)**2, lambda x: np.array([safe_sin2(xi) for xi in x])),
        ("sin⁴", lambda x: np.sin(x)**4, lambda x: np.array([safe_sin4(xi) for xi in x])),
        ("sin⁶", lambda x: np.sin(x)**6, lambda x: np.array([safe_sin6(xi) for xi in x])),
        ("sin⁸", lambda x: np.sin(x)**8, lambda x: np.array([safe_sin8(xi) for xi in x])),
        ("cos", np.cos, lambda x: np.array([safe_cos(xi) for xi in x])),
        ("cos²", lambda x: np.cos(x)**2, lambda x: np.array([safe_cos2(xi) for xi in x])),
        ("cos⁴", lambda x: np.cos(x)**4, lambda x: np.array([safe_cos4(xi) for xi in x])),
        ("cos⁶", lambda x: np.cos(x)**6, lambda x: np.array([safe_cos6(xi) for xi in x])),
        ("cos⁸", lambda x: np.cos(x)**8, lambda x: np.array([safe_cos8(xi) for xi in x]))
    ]
    
    # Results storage
    results = {size: {} for size in sizes}
    
    for size in sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Create test array with critical points and perturbations
        all_critical = np.concatenate([sin_critical, cos_critical])
        n_critical = min(len(all_critical), size // 10)
        
        # Select random critical points and add perturbations
        selected_critical = np.random.choice(all_critical, n_critical, replace=False)
        selected_perturbs = np.random.choice(perturbations, n_critical)
        critical_with_perturb = selected_critical + selected_perturbs
        
        # Fill the rest with random points
        random_points = np.random.uniform(-10*np.pi, 10*np.pi, size - n_critical)
        test_data = np.concatenate([critical_with_perturb, random_points])
        np.random.shuffle(test_data)
        
        for func_name, std_func, safe_func in funcs:
            # Run multiple iterations to get more accurate timing
            std_times = []
            safe_times = []
            
            for _ in range(num_iterations):
                # Benchmark standard implementation
                start_time = time.time()
                std_result = std_func(test_data)
                end_time = time.time()
                std_times.append(end_time - start_time)
                
                # Benchmark safe implementation
                start_time = time.time()
                safe_result = safe_func(test_data)
                end_time = time.time()
                safe_times.append(end_time - start_time)
            
            # Calculate average times
            avg_std_time = sum(std_times) / len(std_times)
            avg_safe_time = sum(safe_times) / len(safe_times)
            
            # Calculate the slowdown factor with protection against zero division
            if avg_std_time > 1e-10:  # Use a small threshold instead of exact zero
                slowdown = avg_safe_time / avg_std_time
            else:
                slowdown = float('inf')
            
            # Store results
            results[size][func_name] = {
                "std_time": avg_std_time,
                "safe_time": avg_safe_time,
                "slowdown": slowdown
            }
            
            print(f"  {func_name:<5}: Numpy: {avg_std_time:.6f}s, Safe: {avg_safe_time:.6f}s, "
                  f"Ratio: {slowdown:.2f}x slower")
    
    # Print overall summary
    print("\n=== Performance Summary ===")
    print(f"{'Function':<6} | {'Size':<7} | {'NumPy(s)':<10} | {'Safe(s)':<10} | {'Slowdown':<8}")
    print("-" * 50)
    
    for size in sizes:
        for func_name, data in results[size].items():
            print(f"{func_name:<6} | {size:<7d} | {data['std_time']:<10.6f} | {data['safe_time']:<10.6f} | {data['slowdown']:<8.2f}x")
    
    # Calculate and print average slowdown by function
    print("\n=== Average Slowdown by Function ===")
    avg_slowdowns = {}
    
    for func_name in [f[0] for f in funcs]:  # Get function names
        total_slowdown = 0
        count = 0
        
        for size in sizes:
            if func_name in results[size]:
                slowdown = results[size][func_name]["slowdown"]
                # Only count finite values
                if np.isfinite(slowdown):
                    total_slowdown += slowdown
                    count += 1
        
        if count > 0:
            avg_slowdowns[func_name] = total_slowdown / count
            print(f"{func_name:<6}: {avg_slowdowns[func_name]:.2f}x slower than NumPy")
    
    return results


def overall_summary(sin_results, cos_results, matrix_results, vector_results, perf_results):
    """Generate an overall summary of all test results."""
    print("\n========== OVERALL TEST SUMMARY ==========")
    
    # 1. Critical points accuracy summary
    print("\n--- Critical Points Accuracy ---")
    
    # Find the worst sin function and point
    worst_sin_func = None
    worst_sin_point = None
    max_sin_error = -1
    
    for point, funcs in sin_results.items():
        for func_name, errors in funcs.items():
            if errors['max_abs_diff'] > max_sin_error:
                max_sin_error = errors['max_abs_diff']
                worst_sin_func = func_name
                worst_sin_point = point
    
    # Find the worst cos function and point
    worst_cos_func = None
    worst_cos_point = None
    max_cos_error = -1
    
    for point, funcs in cos_results.items():
        for func_name, errors in funcs.items():
            if errors['max_abs_diff'] > max_cos_error:
                max_cos_error = errors['max_abs_diff']
                worst_cos_func = func_name
                worst_cos_point = point
    
    print(f"Highest sin function error: {max_sin_error:.2e} for {worst_sin_func} at x = {worst_sin_point}")
    print(f"Highest cos function error: {max_cos_error:.2e} for {worst_cos_func} at x = {worst_cos_point}")
    
    # 2. Matrix and vector test summary
    print("\n--- Matrix and Vector Tests ---")
    
    # Check for any NaNs or Infs
    matrix_has_issues = False
    vector_has_issues = False
    
    for func_name, results in matrix_results.items():
        if results['has_nan'] or results['has_inf']:
            matrix_has_issues = True
            print(f"Matrix issue detected in {func_name}: NaNs: {results['has_nan']}, Infs: {results['has_inf']}")
    
    for func_name, results in vector_results.items():
        if results['has_nan'] or results['has_inf']:
            vector_has_issues = True
            print(f"Vector issue detected in {func_name}: NaNs: {results['has_nan']}, Infs: {results['has_inf']}")
    
    if not matrix_has_issues:
        print("All matrix tests passed without NaNs or Infs")
    
    if not vector_has_issues:
        print("All vector tests passed without NaNs or Infs")
    
    # Find max differences
    max_matrix_diff = max([r['max_diff'] for r in matrix_results.values()])
    max_vector_diff = max([r['max_diff'] for r in vector_results.values()])
    
    print(f"Maximum matrix error: {max_matrix_diff:.2e}")
    print(f"Maximum vector error: {max_vector_diff:.2e}")
    
    # 3. Performance summary
    print("\n--- Performance Summary ---")
    
    # Calculate average slowdown for each function across all sizes
    all_sizes = sorted(perf_results.keys())
    max_size = all_sizes[-1]  # Largest size
    
    # Compile all function names
    all_func_names = set()
    for size in all_sizes:
        all_func_names.update(perf_results[size].keys())
    
    # Calculate average slowdowns across all sizes
    avg_slowdowns_by_func = {}
    for func_name in all_func_names:
        slowdowns = []
        for size in all_sizes:
            if func_name in perf_results[size]:
                slowdowns.append(perf_results[size][func_name]["slowdown"])
        if slowdowns:
            avg_slowdowns_by_func[func_name] = sum(slowdowns) / len(slowdowns)
    
    # Print summary for largest size
    print(f"\nPerformance at size {max_size}:")
    for func_name in sorted(perf_results[max_size].keys()):
        data = perf_results[max_size][func_name]
        print(f"  {func_name:<5}: NumPy: {data['std_time']:.6f}s, Safe: {data['safe_time']:.6f}s, "
              f"Slowdown: {data['slowdown']:.2f}x")
    
    # Find the fastest and slowest safe functions
    if avg_slowdowns_by_func:
        fastest_func = min(avg_slowdowns_by_func.items(), key=lambda x: x[1])
        slowest_func = max(avg_slowdowns_by_func.items(), key=lambda x: x[1])
        
        overall_avg = sum(avg_slowdowns_by_func.values()) / len(avg_slowdowns_by_func)
        
        print(f"\nOverall performance summary:")
        print(f"  Average slowdown factor: {overall_avg:.2f}x")
        print(f"  Most efficient function: {fastest_func[0]} ({fastest_func[1]:.2f}x slower)")
        print(f"  Least efficient function: {slowest_func[0]} ({slowest_func[1]:.2f}x slower)")
        
        # Group by function type (sin vs cos)
        sin_funcs = [f for f in avg_slowdowns_by_func.keys() if f.startswith('sin')]
        cos_funcs = [f for f in avg_slowdowns_by_func.keys() if f.startswith('cos')]
        
        if sin_funcs:
            sin_avg = sum(avg_slowdowns_by_func[f] for f in sin_funcs) / len(sin_funcs)
            print(f"  Average sin function slowdown: {sin_avg:.2f}x")
        
        if cos_funcs:
            cos_avg = sum(avg_slowdowns_by_func[f] for f in cos_funcs) / len(cos_funcs)
            print(f"  Average cos function slowdown: {cos_avg:.2f}x")
    else:
        print("No valid performance data available.")

def visualize_critical_points_errors(sin_results, cos_results):
    """Visualize errors at critical points for sin and cos functions."""
    # Extract data for visualization
    sin_data = []
    for point, funcs in sin_results.items():
        for func_name, errors in funcs.items():
            sin_data.append({
                'point': point,
                'function': func_name,
                'abs_error': errors['max_abs_diff'],
                'rel_error': errors['max_rel_diff']
            })
    
    cos_data = []
    for point, funcs in cos_results.items():
        for func_name, errors in funcs.items():
            cos_data.append({
                'point': point,
                'function': func_name,
                'abs_error': errors['max_abs_diff'],
                'rel_error': errors['max_rel_diff']
            })
    
    # Set up the figure for sin functions
    plt.figure(figsize=(15, 10))
    plt.suptitle('Errors at Critical Points', fontsize=16)
    
    # Plot sin absolute errors
    plt.subplot(2, 2, 1)
    data_to_plot = {}
    for item in sin_data:
        func = item['function']
        if func not in data_to_plot:
            data_to_plot[func] = []
        data_to_plot[func].append(item['abs_error'])
    
    for i, (func, errors) in enumerate(data_to_plot.items()):
        plt.semilogy(range(len(errors)), errors, 'o-', label=func)
    
    plt.title('Sin Functions - Absolute Errors')
    plt.xlabel('Critical Point Index')
    plt.ylabel('Absolute Error (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot cos absolute errors
    plt.subplot(2, 2, 2)
    data_to_plot = {}
    for item in cos_data:
        func = item['function']
        if func not in data_to_plot:
            data_to_plot[func] = []
        data_to_plot[func].append(item['abs_error'])
    
    for i, (func, errors) in enumerate(data_to_plot.items()):
        plt.semilogy(range(len(errors)), errors, 'o-', label=func)
    
    plt.title('Cos Functions - Absolute Errors')
    plt.xlabel('Critical Point Index')
    plt.ylabel('Absolute Error (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot sin by function type
    plt.subplot(2, 2, 3)
    funcs = ['sin(x)', 'sin²(x)', 'sin⁴(x)', 'sin⁶(x)', 'sin⁸(x)']
    avg_errors = []
    
    for func in funcs:
        errors = [item['abs_error'] for item in sin_data if item['function'] == func]
        avg_errors.append(np.mean(errors))
    
    plt.bar(funcs, avg_errors)
    plt.title('Average Absolute Error by Sin Function')
    plt.ylabel('Average Absolute Error')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    
    # Plot cos by function type
    plt.subplot(2, 2, 4)
    funcs = ['cos(x)', 'cos²(x)', 'cos⁴(x)', 'cos⁶(x)', 'cos⁸(x)']
    avg_errors = []
    
    for func in funcs:
        errors = [item['abs_error'] for item in cos_data if item['function'] == func]
        avg_errors.append(np.mean(errors))
    
    plt.bar(funcs, avg_errors)
    plt.title('Average Absolute Error by Cos Function')
    plt.ylabel('Average Absolute Error')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('critical_points_errors.png', dpi=300)
    plt.close()


def visualize_performance(perf_results):
    """Visualize performance comparison of standard vs safe implementations."""
    # Extract data for visualization
    all_sizes = sorted(perf_results.keys())
    all_funcs = set()
    for size_data in perf_results.values():
        all_funcs.update(size_data.keys())
    all_funcs = sorted(all_funcs)
    
    # Group by sin and cos
    sin_funcs = [f for f in all_funcs if f.startswith('sin')]
    cos_funcs = [f for f in all_funcs if f.startswith('cos')]
    
    # Set up the figure
    plt.figure(figsize=(15, 12))
    plt.suptitle('Performance Comparison', fontsize=16)
    
    # Plot slowdown factors by array size
    plt.subplot(1, 2, 1)
    for func in all_funcs:
        sizes = []
        slowdowns = []
        for size in all_sizes:
            if func in perf_results[size]:
                sizes.append(size)
                slowdowns.append(perf_results[size][func]['slowdown'])
        plt.plot(sizes, slowdowns, 'o-', label=func)
    
    plt.title('Slowdown Factor by Array Size')
    plt.xlabel('Array Size')
    plt.ylabel('Slowdown Factor (x times slower)')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    # # Plot average slowdown by function
    # plt.subplot(2, 2, 2)
    # avg_slowdowns = []
    # for func in all_funcs:
    #     slowdowns = []
    #     for size in all_sizes:
    #         if func in perf_results[size]:
    #             slowdowns.append(perf_results[size][func]['slowdown'])
    #     if slowdowns:
    #         avg_slowdowns.append(np.mean(slowdowns))
    #     else:
    #         avg_slowdowns.append(0)
    
    # plt.bar(all_funcs, avg_slowdowns)
    # plt.title('Average Slowdown by Function')
    # plt.ylabel('Average Slowdown Factor')
    # plt.xticks(rotation=45)
    # plt.grid(axis='y', linestyle='--')
    
    # Plot absolute times for largest size
    max_size = all_sizes[-1]
    plt.subplot(1, 2, 2)
    functions = []
    std_times = []
    safe_times = []
    
    for func in all_funcs:
        if func in perf_results[max_size]:
            functions.append(func)
            std_times.append(perf_results[max_size][func]['std_time'])
            safe_times.append(perf_results[max_size][func]['safe_time'])
    
    x = np.arange(len(functions))
    width = 0.35
    
    plt.bar(x - width/2, std_times, width, label='NumPy')
    plt.bar(x + width/2, safe_times, width, label='Safe')
    plt.title(f'Execution Time for Size {max_size}')
    plt.xlabel('Function')
    plt.ylabel('Time (seconds)')
    plt.xticks(x, functions, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    
    # Plot sin vs cos comparison
    # plt.subplot(2, 2, 4)
    
    # # Calculate average slowdown for sin and cos functions
    # sin_avg = [np.mean([perf_results[size][func]['slowdown'] 
    #                    for size in all_sizes if func in perf_results[size]]) 
    #           for func in sin_funcs]
    
    # cos_avg = [np.mean([perf_results[size][func]['slowdown'] 
    #                    for size in all_sizes if func in perf_results[size]]) 
    #           for func in cos_funcs]
    
    # # Group by power
    # powers = ['', '²', '⁴', '⁶', '⁸']
    # sin_by_power = []
    # cos_by_power = []
    
    # for i, power in enumerate(powers):
    #     sin_func = f'sin{power}'
    #     cos_func = f'cos{power}'
        
    #     sin_data = [np.mean([perf_results[size][sin_func]['slowdown'] 
    #                        for size in all_sizes if sin_func in perf_results[size]])
    #               if sin_func in sin_funcs else 0]
        
    #     cos_data = [np.mean([perf_results[size][cos_func]['slowdown'] 
    #                        for size in all_sizes if cos_func in perf_results[size]])
    #               if cos_func in cos_funcs else 0]
        
    #     sin_by_power.append(sin_data[0] if sin_data else 0)
    #     cos_by_power.append(cos_data[0] if cos_data else 0)
    
    # x = np.arange(len(powers))
    # width = 0.35
    
    # plt.bar(x - width/2, sin_by_power, width, label='sin')
    # plt.bar(x + width/2, cos_by_power, width, label='cos')
    # plt.title('Slowdown by Function Power')
    # plt.xlabel('Power')
    # plt.xticks(x, powers)
    # plt.ylabel('Average Slowdown Factor')
    # plt.legend()
    # plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('performance_comparison.png', dpi=300)
    plt.close()


def visualize_error_distribution(sin_results, cos_results, matrix_results, vector_results):
    """Visualize error distributions and compare matrix/vector results."""
    plt.figure(figsize=(15, 10))
    plt.suptitle('Error Distributions and Matrix/Vector Results', fontsize=16)
    
    # Extract all absolute errors for sin and cos
    sin_errors = []
    for point, funcs in sin_results.items():
        for func_name, errors in funcs.items():
            sin_errors.append({
                'function': func_name,
                'error': errors['max_abs_diff']
            })
    
    cos_errors = []
    for point, funcs in cos_results.items():
        for func_name, errors in funcs.items():
            cos_errors.append({
                'function': func_name,
                'error': errors['max_abs_diff']
            })
    
    # Plot sin error distribution
    plt.subplot(2, 2, 1)
    sin_func_errors = {}
    for error in sin_errors:
        if error['function'] not in sin_func_errors:
            sin_func_errors[error['function']] = []
        sin_func_errors[error['function']].append(error['error'])
    
    plt.boxplot([errors for errors in sin_func_errors.values()], 
               labels=sin_func_errors.keys())
    plt.title('Sin Functions Error Distribution')
    plt.ylabel('Absolute Error')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    
    # Plot cos error distribution
    plt.subplot(2, 2, 2)
    cos_func_errors = {}
    for error in cos_errors:
        if error['function'] not in cos_func_errors:
            cos_func_errors[error['function']] = []
        cos_func_errors[error['function']].append(error['error'])
    
    plt.boxplot([errors for errors in cos_func_errors.values()], 
               labels=cos_func_errors.keys())
    plt.title('Cos Functions Error Distribution')
    plt.ylabel('Absolute Error')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    
    # Plot matrix results
    plt.subplot(2, 2, 3)
    funcs = []
    matrix_diffs = []
    
    for func, data in matrix_results.items():
        funcs.append(func)
        matrix_diffs.append(data['max_diff'])
    
    plt.bar(funcs, matrix_diffs)
    plt.title('Matrix Test Maximum Differences')
    plt.ylabel('Maximum Absolute Difference')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    
    # Plot vector results
    plt.subplot(2, 2, 4)
    funcs = []
    vector_diffs = []
    
    for func, data in vector_results.items():
        funcs.append(func)
        vector_diffs.append(data['max_diff'])
    
    plt.bar(funcs, vector_diffs)
    plt.title('Vector Test Maximum Differences')
    plt.ylabel('Maximum Absolute Difference')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('error_distribution.png', dpi=300)
    plt.close()


def visualize_sin_cos_comparison():
    """Create a visualization showing the standard sin/cos functions vs. their powers."""
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    plt.suptitle('Sin and Cos Functions Behavior', fontsize=16)
    
    # Plot sin and powers
    plt.subplot(2, 2, 1)
    plt.plot(x, np.sin(x), label='sin(x)')
    plt.plot(x, np.sin(x)**2, label='sin²(x)')
    plt.plot(x, np.sin(x)**4, label='sin⁴(x)')
    plt.plot(x, np.sin(x)**6, label='sin⁶(x)')
    plt.plot(x, np.sin(x)**8, label='sin⁸(x)')
    
    plt.title('Sin Functions')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    # Plot cos and powers
    plt.subplot(2, 2, 2)
    plt.plot(x, np.cos(x), label='cos(x)')
    plt.plot(x, np.cos(x)**2, label='cos²(x)')
    plt.plot(x, np.cos(x)**4, label='cos⁴(x)')
    plt.plot(x, np.cos(x)**6, label='cos⁶(x)')
    plt.plot(x, np.cos(x)**8, label='cos⁸(x)')
    
    plt.title('Cos Functions')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    # Critical regions for sin
    plt.subplot(2, 2, 3)
    critical_x = np.linspace(-0.00001, 0.00001, 10000)
    
    plt.plot(critical_x, np.sin(critical_x), label='sin(x)')
    plt.plot(critical_x, np.sin(critical_x)**2, label='sin²(x)')
    plt.plot(critical_x, np.sin(critical_x)**4, label='sin⁴(x)')
    plt.plot(critical_x, np.sin(critical_x)**6, label='sin⁶(x)')
    plt.plot(critical_x, np.sin(critical_x)**8, label='sin⁸(x)')
    
    plt.ylim((-1e-11, 1e-11))
    plt.title('Sin Functions Near Critical Point (x = 0)')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    # Critical regions for cos
    plt.subplot(2, 2, 4)
    critical_x = np.linspace(np.pi/2 - 0.00001, np.pi/2 + 0.00001, 10000)
    
    plt.plot(critical_x, np.cos(critical_x), label='cos(x)')
    plt.plot(critical_x, np.cos(critical_x)**2, label='cos²(x)')
    plt.plot(critical_x, np.cos(critical_x)**4, label='cos⁴(x)')
    plt.plot(critical_x, np.cos(critical_x)**6, label='cos⁶(x)')
    plt.plot(critical_x, np.cos(critical_x)**8, label='cos⁸(x)')
    
    plt.ylim((-1e-11, 1e-11))
    plt.title('Cos Functions Near Critical Point (x = π/2)')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('sin_cos_behavior.png', dpi=300)
    plt.close()


def add_visualizations(test_sin_critical_points, test_cos_critical_points, 
                      test_matrix_and_vector, benchmark_performance):
    """
    Function to integrate with the existing test suite and generate visualizations.
    """
    # Run all tests to get results
    print("\n========== Generating Visualizations ==========")
    
    sin_results = test_sin_critical_points()
    cos_results = test_cos_critical_points()
    matrix_results, vector_results = test_matrix_and_vector()
    perf_results = benchmark_performance()
    
    # Generate all visualizations
    print("\nCreating visualization 1: Critical points errors")
    visualize_critical_points_errors(sin_results, cos_results)
    
    print("Creating visualization 2: Performance comparison")
    visualize_performance(perf_results)
    
    print("Creating visualization 3: Error distribution and matrix/vector results")
    visualize_error_distribution(sin_results, cos_results, matrix_results, vector_results)
    
    print("Creating visualization 4: Sin and Cos behavior")
    visualize_sin_cos_comparison()
    
    print("\nAll visualizations have been saved as PNG files.")
    return sin_results, cos_results, matrix_results, vector_results, perf_results


if __name__ == "__main__":
    print("===============================================")
    print("   Safe Trigonometric Functions Test Suite     ")
    print("===============================================")
    
    # Run all tests
    sin_results = test_sin_critical_points()
    cos_results = test_cos_critical_points()
    matrix_results, vector_results = test_matrix_and_vector()
    perf_results = benchmark_performance()
    
    # Generate overall summary
    overall_summary(sin_results, cos_results, matrix_results, vector_results, perf_results)
    
    print("\nText based test suite completed.")
    add_visualizations(test_sin_critical_points, test_cos_critical_points, 
        test_matrix_and_vector, benchmark_performance)
    print("\nGraphical presentation completed.")