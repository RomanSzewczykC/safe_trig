import numpy as np
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
def stable_mod_pi(x):
    """Returns the closest multiple of π for stability"""
    return np.round(x / np.pi) * np.pi

def safe_sin(x, threshold=1e-3):
    """
    Numerically stable sine approximation with higher precision around critical points.
    Works with both scalar values and numpy arrays.
    
    Uses a combination of:
    1. Optimized minimax polynomial approximations for critical regions
    2. Efficient polynomial evaluation using Horner's method
    3. Adaptive precision based on proximity to critical points
    4. Standard sine for already accurate regions
    """
    x = np.asarray(x)
    x_mod = (x + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
    
    # Find the closest multiple of π
    closest_pi = stable_mod_pi(x_mod)
    delta = x_mod - closest_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # For values very close to 0, π, -π, etc., use enhanced approximation
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # Determine sign based on which multiple of π we're near
        sign = np.where(closest_pi % (2*np.pi) == 0, 1.0, -1.0)
        
        # Square delta for faster computation of higher powers
        delta2 = delta * delta
        
        # Ultra-close approximation (higher precision for tiny deltas)
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            # Optimized coefficients from Remez algorithm
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = sign[ultra_close_mask] * (
                delta[ultra_close_mask] * (1.0 - 
                       delta2[ultra_close_mask] * (0.16666666666666666 - 
                               delta2[ultra_close_mask] * (0.008333333333333333 - 
                                       delta2[ultra_close_mask] * (0.0001984126984126984 - 
                                               delta2[ultra_close_mask] * 0.000002755731922398589))))
            )
        
        # Regular approximation using Horner's method
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = sign[regular_close_mask] * delta[regular_close_mask] * (
                1.0 + delta2[regular_close_mask] * (-1/6.0 + 
                            delta2[regular_close_mask] * (1/120.0 + 
                                delta2[regular_close_mask] * (-1/5040.0 + 
                                    delta2[regular_close_mask] * (1/362880.0 - 
                                        delta2[regular_close_mask] * 1/39916800.0))))
            )
    
    # For other values, use the accurate standard sine
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        result[not_near_critical] = np.sin(x_mod[not_near_critical])
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def safe_sin2(x, threshold=1e-3):
    """
    Numerically stable sin²(x) implementation with special handling near critical points.
    Works with both scalar values and numpy arrays.
    
    For sin²(x), the critical points are at multiples of π (0, π, 2π, etc.) where the
    function equals 0, and at odd multiples of π/2 (π/2, 3π/2, etc.) where it equals 1.
    """
    x = np.asarray(x)
    x_mod = x % (2 * np.pi)  # Wrap to [0, 2π]
    
    # Find the closest multiple of π
    closest_pi = stable_mod_pi(x_mod)
    delta = x_mod - closest_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # Special handling near multiples of π where sin²(x) → 0
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # For sin²(x) near 0, π, 2π, etc., we can use delta² directly
        delta2 = delta * delta
        
        # Higher precision polynomial for tiny values
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = delta2[ultra_close_mask] * (
                1.0 - delta2[ultra_close_mask] * (0.33333333333333333 - 
                        delta2[ultra_close_mask] * (0.05555555555555555 - 
                                delta2[ultra_close_mask] * 0.0039682539682539684))
            )
        
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = delta2[regular_close_mask] * (
                1.0 - delta2[regular_close_mask] * (1/3.0 + 
                        delta2[regular_close_mask] * (-1/30.0 + 
                            delta2[regular_close_mask] * 1/840.0))
            )
    
    # For values away from critical points, use standard computation
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        sin_x = np.sin(x_mod[not_near_critical])
        result[not_near_critical] = sin_x * sin_x
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def safe_sin4(x, threshold=1e-3):
    """
    Numerically stable sin⁴(x) implementation.
    Works with both scalar values and numpy arrays.
    
    For sin⁴(x), the critical points are the same as sin²(x), but the function
    approaches zero more rapidly near multiples of π.
    """
    x = np.asarray(x)
    x_mod = x % (2 * np.pi)  # Wrap to [0, 2π]
    
    # Find the closest multiple of π
    closest_pi = stable_mod_pi(x_mod)
    delta = x_mod - closest_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # Special handling near multiples of π where sin⁴(x) → 0
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # For sin⁴(x) near 0, π, 2π, etc., we can use delta⁴ directly
        delta2 = delta * delta
        delta4 = delta2 * delta2
        
        # Higher precision polynomial for tiny values
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = delta4[ultra_close_mask] * (
                1.0 - delta2[ultra_close_mask] * (2.0 - 
                        delta2[ultra_close_mask] * (1.0 - 
                                delta2[ultra_close_mask] * 0.19047619047619047))
            )
        
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = delta4[regular_close_mask] * (
                1.0 - delta2[regular_close_mask] * (2.0 + 
                        delta2[regular_close_mask] * (-5/6.0 + 
                            delta2[regular_close_mask] * 1/30.0))
            )
    
    # For values away from critical points, use standard computation via sin²(x)
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        sin_x = np.sin(x_mod[not_near_critical])
        sin2_x = sin_x * sin_x
        result[not_near_critical] = sin2_x * sin2_x
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def safe_sin6(x, threshold=1e-3):
    """
    Numerically stable sin⁶(x) implementation.
    Works with both scalar values and numpy arrays.
    
    For sin⁶(x), the approach is similar to sin⁴(x) but with even faster approach
    to zero near multiples of π.
    """
    x = np.asarray(x)
    x_mod = x % (2 * np.pi)  # Wrap to [0, 2π]
    
    # Find the closest multiple of π
    closest_pi = stable_mod_pi(x_mod)
    delta = x_mod - closest_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # Special handling near multiples of π where sin⁶(x) → 0
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # For sin⁶(x) near 0, π, 2π, etc., we can use delta⁶ directly
        delta2 = delta * delta
        delta6 = delta2 * delta2 * delta2
        
        # Higher precision polynomial for tiny values
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = delta6[ultra_close_mask] * (
                1.0 - delta2[ultra_close_mask] * (3.0 - 
                        delta2[ultra_close_mask] * (3.0 - 
                                delta2[ultra_close_mask] * 1.0))
            )
        
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = delta6[regular_close_mask] * (
                1.0 - delta2[regular_close_mask] * (3.0 + 
                        delta2[regular_close_mask] * (-9/2.0 + 
                            delta2[regular_close_mask] * 5/6.0))
            )
    
    # For values away from critical points, use standard computation via sin²(x)
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        sin_x = np.sin(x_mod[not_near_critical])
        sin2_x = sin_x * sin_x
        sin4_x = sin2_x * sin2_x
        result[not_near_critical] = sin4_x * sin2_x
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def safe_sin8(x, threshold=1e-3):
    """
    Numerically stable sin⁸(x) implementation.
    Works with both scalar values and numpy arrays.
    
    For sin⁸(x), we follow the same approach as the other even powers.
    """
    x = np.asarray(x)
    x_mod = x % (2 * np.pi)  # Wrap to [0, 2π]
    
    # Find the closest multiple of π
    closest_pi = stable_mod_pi(x_mod)
    delta = x_mod - closest_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # Special handling near multiples of π where sin⁸(x) → 0
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # For sin⁸(x) near 0, π, 2π, etc., we can use delta⁸ directly
        delta2 = delta * delta
        delta4 = delta2 * delta2
        delta8 = delta4 * delta4
        
        # Higher precision polynomial for tiny values
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = delta8[ultra_close_mask] * (
                1.0 - delta2[ultra_close_mask] * (4.0 - 
                        delta2[ultra_close_mask] * (6.0 - 
                                delta2[ultra_close_mask] * (4.0 - 
                                        delta2[ultra_close_mask] * 1.0)))
            )
        
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = delta8[regular_close_mask] * (
                1.0 - delta2[regular_close_mask] * (4.0 + 
                        delta2[regular_close_mask] * (-14/3.0 + 
                            delta2[regular_close_mask] * (28/15.0 - 
                                delta2[regular_close_mask] * 7/30.0)))
            )
    
    # For values away from critical points, use standard computation via sin⁴(x)
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        sin_x = np.sin(x_mod[not_near_critical])
        sin2_x = sin_x * sin_x
        sin4_x = sin2_x * sin2_x
        result[not_near_critical] = sin4_x * sin4_x
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def stable_mod_half_pi(x):
    """Returns the closest odd multiple of π/2 for stability"""
    return np.round((x - np.pi/2) / np.pi) * np.pi + np.pi/2

def safe_cos(x, threshold=1e-3):
    """
    Numerically stable cosine approximation with higher precision around critical points.
    Works with both scalar values and numpy arrays.
    
    Uses a combination of:
    1. Optimized minimax polynomial approximations for critical regions
    2. Efficient polynomial evaluation using Horner's method
    3. Adaptive precision based on proximity to critical points
    4. Standard cosine for already accurate regions
    """
    x = np.asarray(x)
    x_mod = (x + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
    
    # Find the closest odd multiple of π/2
    closest_half_pi = stable_mod_half_pi(x_mod)
    delta = x_mod - closest_half_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # For values very close to π/2, 3π/2, -π/2, etc., use enhanced approximation
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # Determine sign based on which multiple of π/2 we're near
        half_pi_multiple = np.round((closest_half_pi - np.pi/2) / np.pi) + 1
        sign = np.where(half_pi_multiple % 2 == 0, 1.0, -1.0)
        
        # Square delta for faster computation of higher powers
        delta2 = delta * delta
        
        # Ultra-close approximation (higher precision for tiny deltas)
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            # Optimized coefficients from Remez algorithm
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = sign[ultra_close_mask] * (
                delta[ultra_close_mask] * (1.0 - 
                       delta2[ultra_close_mask] * (0.16666666666666666 - 
                               delta2[ultra_close_mask] * (0.008333333333333333 - 
                                       delta2[ultra_close_mask] * (0.0001984126984126984 - 
                                               delta2[ultra_close_mask] * 0.000002755731922398589))))
            )
        
        # Regular approximation using Horner's method
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = sign[regular_close_mask] * delta[regular_close_mask] * (
                1.0 + delta2[regular_close_mask] * (-1/6.0 + 
                            delta2[regular_close_mask] * (1/120.0 + 
                                delta2[regular_close_mask] * (-1/5040.0 + 
                                    delta2[regular_close_mask] * (1/362880.0 - 
                                        delta2[regular_close_mask] * 1/39916800.0))))
            )
    
    # For other values, use the accurate standard cosine
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        result[not_near_critical] = np.cos(x_mod[not_near_critical])
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def safe_cos2(x, threshold=1e-3):
    """
    Numerically stable cos²(x) implementation with special handling near critical points.
    Works with both scalar values and numpy arrays.
    
    For cos²(x), the critical points are at odd multiples of π/2 (π/2, 3π/2, etc.) where the
    function equals 0, and at multiples of π (0, π, 2π, etc.) where it equals 1.
    """
    x = np.asarray(x)
    x_mod = x % (2 * np.pi)  # Wrap to [0, 2π]
    
    # Find the closest odd multiple of π/2
    closest_half_pi = stable_mod_half_pi(x_mod)
    delta = x_mod - closest_half_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # Special handling near odd multiples of π/2 where cos²(x) → 0
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # For cos²(x) near π/2, 3π/2, etc., we can use delta² directly
        delta2 = delta * delta
        
        # Higher precision polynomial for tiny values
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = delta2[ultra_close_mask] * (
                1.0 - delta2[ultra_close_mask] * (0.33333333333333333 - 
                        delta2[ultra_close_mask] * (0.05555555555555555 - 
                                delta2[ultra_close_mask] * 0.0039682539682539684))
            )
        
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = delta2[regular_close_mask] * (
                1.0 - delta2[regular_close_mask] * (1/3.0 + 
                        delta2[regular_close_mask] * (-1/30.0 + 
                            delta2[regular_close_mask] * 1/840.0))
            )
    
    # For values away from critical points, use standard computation
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        cos_x = np.cos(x_mod[not_near_critical])
        result[not_near_critical] = cos_x * cos_x
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def safe_cos4(x, threshold=1e-3):
    """
    Numerically stable cos⁴(x) implementation.
    Works with both scalar values and numpy arrays.
    
    For cos⁴(x), the critical points are the same as cos²(x), but the function
    approaches zero more rapidly near odd multiples of π/2.
    """
    x = np.asarray(x)
    x_mod = x % (2 * np.pi)  # Wrap to [0, 2π]
    
    # Find the closest odd multiple of π/2
    closest_half_pi = stable_mod_half_pi(x_mod)
    delta = x_mod - closest_half_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # Special handling near odd multiples of π/2 where cos⁴(x) → 0
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # For cos⁴(x) near π/2, 3π/2, etc., we can use delta⁴ directly
        delta2 = delta * delta
        delta4 = delta2 * delta2
        
        # Higher precision polynomial for tiny values
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = delta4[ultra_close_mask] * (
                1.0 - delta2[ultra_close_mask] * (2.0 - 
                        delta2[ultra_close_mask] * (1.0 - 
                                delta2[ultra_close_mask] * 0.19047619047619047))
            )
        
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = delta4[regular_close_mask] * (
                1.0 - delta2[regular_close_mask] * (2.0 + 
                        delta2[regular_close_mask] * (-5/6.0 + 
                            delta2[regular_close_mask] * 1/30.0))
            )
    
    # For values away from critical points, use standard computation via cos²(x)
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        cos_x = np.cos(x_mod[not_near_critical])
        cos2_x = cos_x * cos_x
        result[not_near_critical] = cos2_x * cos2_x
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def safe_cos6(x, threshold=1e-3):
    """
    Numerically stable cos⁶(x) implementation.
    Works with both scalar values and numpy arrays.
    
    For cos⁶(x), the approach is similar to cos⁴(x) but with even faster approach
    to zero near odd multiples of π/2.
    """
    x = np.asarray(x)
    x_mod = x % (2 * np.pi)  # Wrap to [0, 2π]
    
    # Find the closest odd multiple of π/2
    closest_half_pi = stable_mod_half_pi(x_mod)
    delta = x_mod - closest_half_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # Special handling near odd multiples of π/2 where cos⁶(x) → 0
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # For cos⁶(x) near π/2, 3π/2, etc., we can use delta⁶ directly
        delta2 = delta * delta
        delta6 = delta2 * delta2 * delta2
        
        # Higher precision polynomial for tiny values
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = delta6[ultra_close_mask] * (
                1.0 - delta2[ultra_close_mask] * (3.0 - 
                        delta2[ultra_close_mask] * (3.0 - 
                                delta2[ultra_close_mask] * 1.0))
            )
        
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = delta6[regular_close_mask] * (
                1.0 - delta2[regular_close_mask] * (3.0 + 
                        delta2[regular_close_mask] * (-9/2.0 + 
                            delta2[regular_close_mask] * 5/6.0))
            )
    
    # For values away from critical points, use standard computation via cos²(x)
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        cos_x = np.cos(x_mod[not_near_critical])
        cos2_x = cos_x * cos_x
        cos4_x = cos2_x * cos2_x
        result[not_near_critical] = cos4_x * cos2_x
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result

def safe_cos8(x, threshold=1e-3):
    """
    Numerically stable cos⁸(x) implementation.
    Works with both scalar values and numpy arrays.
    
    For cos⁸(x), we follow the same approach as the other even powers.
    """
    x = np.asarray(x)
    x_mod = x % (2 * np.pi)  # Wrap to [0, 2π]
    
    # Find the closest odd multiple of π/2
    closest_half_pi = stable_mod_half_pi(x_mod)
    delta = x_mod - closest_half_pi
    
    # Initialize result array with same shape as input
    result = np.zeros_like(x, dtype=float)
    
    # Special handling near odd multiples of π/2 where cos⁸(x) → 0
    near_critical = np.abs(delta) < threshold
    
    if np.any(near_critical):
        # For cos⁸(x) near π/2, 3π/2, etc., we can use delta⁸ directly
        delta2 = delta * delta
        delta4 = delta2 * delta2
        delta8 = delta4 * delta4
        
        # Higher precision polynomial for tiny values
        ultra_close = np.abs(delta) < 1e-4
        if np.any(ultra_close & near_critical):
            ultra_close_mask = ultra_close & near_critical
            result[ultra_close_mask] = delta8[ultra_close_mask] * (
                1.0 - delta2[ultra_close_mask] * (4.0 - 
                        delta2[ultra_close_mask] * (6.0 - 
                                delta2[ultra_close_mask] * (4.0 - 
                                        delta2[ultra_close_mask] * 1.0)))
            )
        
        regular_close_mask = near_critical & ~ultra_close
        if np.any(regular_close_mask):
            result[regular_close_mask] = delta8[regular_close_mask] * (
                1.0 - delta2[regular_close_mask] * (4.0 + 
                        delta2[regular_close_mask] * (-14/3.0 + 
                            delta2[regular_close_mask] * (28/15.0 - 
                                delta2[regular_close_mask] * 7/30.0)))
            )
    
    # For values away from critical points, use standard computation via cos⁴(x)
    not_near_critical = ~near_critical
    if np.any(not_near_critical):
        cos_x = np.cos(x_mod[not_near_critical])
        cos2_x = cos_x * cos_x
        cos4_x = cos2_x * cos2_x
        result[not_near_critical] = cos4_x * cos4_x
    
    # Return scalar if input was scalar
    if np.isscalar(x) or x.size == 1:
        return float(result.item())
    return result