# Safe Trigonometric Functions Testing Documentation

## Overview
This document details the tests conducted to compare safe trigonometric functions against standard NumPy implementations. Differences are analyzed both numerically and visually.

## Test Categories
### 1. Scalar Tests
- Tests are conducted on small values of sine near zero and cosine near \(\frac{\pi}{2}\).
- Example calculations:
  ```python
  sin_test_value = 0.0001
  cos_test_value = np.pi/2 + 0.0001
  ```
- Differences between standard and safe implementations are printed.

### 2. Array Tests
- Conducted on small value ranges:
  - Sine test array: `np.linspace(-0.001, 0.001, 5)`
  - Cosine test array: `np.linspace(np.pi/2 - 0.001, np.pi/2 + 0.001, 5)`
- Maximum differences between standard and safe versions are computed.

### 3. Matrix Tests
- Conducted on small 2x2 matrices:
  ```python
  sin_test_matrix = np.array([[0.0001, 0.0002], [0.0003, 0.0004]])
  cos_test_matrix = np.array([[np.pi/2 + 0.0001, np.pi/2 + 0.0002],
                              [np.pi/2 + 0.0003, np.pi/2 + 0.0004]])
  ```
- Maximum differences are printed.

## Plotting Differences
- Plots illustrate differences between safe trigonometric functions and NumPy equivalents.
- Evaluated over the ranges:
  ```python
  sin_range = np.linspace(-0.005, 0.005, 1000)
  cos_range = np.linspace(np.pi/2 - 0.005, np.pi/2 + 0.005, 1000)
  ```
- Safe implementations compared:
  - `safe_sin`, `safe_sin2`, `safe_sin4`, `safe_sin6`, `safe_sin8`
  - `safe_cos`, `safe_cos2`, `safe_cos4`, `safe_cos6`, `safe_cos8`
- Differences plotted with indicators for maximum deviation.

## Execution
Run the script to perform all tests and generate plots:
```bash
python test_matrixes.py
```

## Conclusion
This test suite evaluates safe trigonometric functions in different scenarios, highlighting numerical differences and verifying accuracy through visual plots.

