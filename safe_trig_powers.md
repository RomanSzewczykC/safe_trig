# Safe Trigonometric Functions

This module provides numerically stable implementations of trigonometric functions (sine and cosine) and their powers. These implementations are designed to handle critical points with high precision, reducing errors caused by floating-point arithmetic near singularities.

## Functions

### `safe_sin(x, threshold=1e-3)`
**Description:**
Computes a numerically stable sine function with high precision around critical points.

**Parameters:**
- `x`: Input value or array of values (scalar or numpy array)
- `threshold`: Defines the region near critical points where higher precision approximation is applied (default: `1e-3`)

**Returns:**
- The computed sine values, maintaining numerical stability.

**Implementation Details:**
- The input is wrapped to the range \([-\pi, \pi]\)
- The closest multiple of \(\pi\) is found for stability
- A high-precision approximation is applied near \(0, \pi, -\pi\) to avoid floating-point errors
- Uses a minimax polynomial approximation optimized using Horner's method
- Supports both scalar inputs and matrix (numpy array) inputs

---

### `safe_cos(x, threshold=1e-3)`
**Description:**
Computes a numerically stable cosine function with high precision around critical points.

**Parameters:**
- `x`: Input value or array of values (scalar or numpy array)
- `threshold`: Defines the region near critical points where higher precision approximation is applied (default: `1e-3`)

**Returns:**
- The computed cosine values, maintaining numerical stability.

**Implementation Details:**
- The input is wrapped to the range \([-\pi, \pi]\)
- The closest odd multiple of \(\pi/2\) is found for stability
- A high-precision approximation is applied near \(\pi/2, 3\pi/2, -\pi/2\) to avoid floating-point errors
- Uses a minimax polynomial approximation optimized using Horner's method
- Supports both scalar inputs and matrix (numpy array) inputs

---

## Sine and Cosine Powers

### `safe_sin2(x, threshold=1e-3)`
Computes \(\sin^2(x)\) with enhanced stability near multiples of \(\pi\), where the function tends to zero. Supports both scalar inputs and matrix (numpy array) inputs.

### `safe_sin4(x, threshold=1e-3)`
Computes \(\sin^4(x)\), refining the stability approach used in `safe_sin2`. Supports both scalar inputs and matrix (numpy array) inputs.

### `safe_sin6(x, threshold=1e-3)`
Computes \(\sin^6(x)\), extending the accuracy improvements for even higher powers. Supports both scalar inputs and matrix (numpy array) inputs.

### `safe_sin8(x, threshold=1e-3)`
Computes \(\sin^8(x)\) with enhanced precision near critical points. Supports both scalar inputs and matrix (numpy array) inputs.

---

### `safe_cos2(x, threshold=1e-3)`
Computes \(\cos^2(x)\) with improved accuracy near odd multiples of \(\pi/2\). Supports both scalar inputs and matrix (numpy array) inputs.

### `safe_cos4(x, threshold=1e-3)`
Computes \(\cos^4(x)\), refining the precision of `safe_cos2`. Supports both scalar inputs and matrix (numpy array) inputs.

### `safe_cos6(x, threshold=1e-3)`
Computes \(\cos^6(x)\), ensuring numerical stability even for higher powers. Supports both scalar inputs and matrix (numpy array) inputs.

### `safe_cos8(x, threshold=1e-3)`
Computes \(\cos^8(x)\) while maintaining numerical robustness. Supports both scalar inputs and matrix (numpy array) inputs.

---

## Stability Mechanisms
1. **Modular Reduction**
   - Inputs are wrapped to a fundamental domain (\([-\pi, \pi]\) or \([0, 2\pi]\)) to prevent unnecessary precision loss.
2. **Critical Point Handling**
   - The functions detect when input values are near multiples of \(\pi\) or \(\pi/2\) and apply specialized approximations.
3. **Polynomial Approximation**
   - Uses Horner's method for evaluating polynomials, reducing numerical error and computational cost.
4. **Adaptive Precision**
   - Higher-order polynomial approximations are applied for inputs very close to critical points.
5. **Support for Scalars and Matrices**
   - All functions handle both individual scalar values and numpy arrays efficiently, making them suitable for vectorized computations in scientific applications.

These functions are useful in scientific computing and simulations where numerical stability is critical.

