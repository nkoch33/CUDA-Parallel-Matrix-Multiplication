#!/usr/bin/env python3
"""
NumPy Matrix Multiplication Benchmark
Comparison script for CUDA implementation
"""

import numpy as np
import time
import sys
import argparse
from typing import Tuple

def matrix_multiply_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """NumPy matrix multiplication using optimized BLAS."""
    return np.dot(A, B)

def matrix_multiply_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Naive Python implementation for comparison."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def benchmark_numpy(matrix_size: int, num_runs: int = 5) -> Tuple[float, float]:
    """Benchmark NumPy matrix multiplication."""
    print(f"Benchmarking NumPy with {matrix_size}x{matrix_size} matrices...")
    
    # Generate random matrices
    np.random.seed(42)  # For reproducible results
    A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    B = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    
    # Warmup
    _ = matrix_multiply_numpy(A, B)
    
    # Benchmark NumPy (optimized)
    times_numpy = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        C_numpy = matrix_multiply_numpy(A, B)
        end_time = time.perf_counter()
        times_numpy.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Benchmark naive implementation (smaller matrices only)
    times_naive = []
    if matrix_size <= 512:  # Only for smaller matrices due to performance
        for _ in range(num_runs):
            start_time = time.perf_counter()
            C_naive = matrix_multiply_naive(A, B)
            end_time = time.perf_counter()
            times_naive.append((end_time - start_time) * 1000)
    
    avg_time_numpy = np.mean(times_numpy)
    avg_time_naive = np.mean(times_naive) if times_naive else None
    
    return avg_time_numpy, avg_time_naive

def calculate_gflops(matrix_size: int, time_ms: float) -> float:
    """Calculate GFLOPS for matrix multiplication."""
    # Matrix multiplication: 2 * N^3 operations
    operations = 2 * matrix_size ** 3
    time_seconds = time_ms / 1000.0
    gflops = operations / (time_seconds * 1e9)
    return gflops

def main():
    parser = argparse.ArgumentParser(description='NumPy Matrix Multiplication Benchmark')
    parser.add_argument('matrix_size', type=int, nargs='?', default=1024,
                       help='Matrix size (default: 1024)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of benchmark runs (default: 5)')
    parser.add_argument('--compare-cuda', action='store_true',
                       help='Compare with CUDA results if available')
    
    args = parser.parse_args()
    
    print("NumPy Matrix Multiplication Benchmark")
    print("=" * 40)
    print(f"Matrix size: {args.matrix_size}x{args.matrix_size}")
    print(f"Number of runs: {args.runs}")
    print()
    
    # Run benchmark
    numpy_time, naive_time = benchmark_numpy(args.matrix_size, args.runs)
    
    # Calculate performance metrics
    numpy_gflops = calculate_gflops(args.matrix_size, numpy_time)
    
    print("Results:")
    print(f"NumPy (BLAS) time: {numpy_time:.2f} ms")
    print(f"NumPy GFLOPS: {numpy_gflops:.2f}")
    
    if naive_time:
        naive_gflops = calculate_gflops(args.matrix_size, naive_time)
        print(f"Naive Python time: {naive_time:.2f} ms")
        print(f"Naive Python GFLOPS: {naive_gflops:.2f}")
        print(f"NumPy speedup vs naive: {naive_time/numpy_time:.2f}x")
    
    print()
    
    # Memory usage estimation
    memory_mb = (3 * args.matrix_size * args.matrix_size * 4) / (1024 * 1024)  # 3 matrices, float32
    print(f"Estimated memory usage: {memory_mb:.1f} MB")
    
    # Theoretical peak performance comparison
    print("\nPerformance Analysis:")
    print(f"Operations: {2 * args.matrix_size**3:,}")
    print(f"Memory accesses: {3 * args.matrix_size**2 * 4:,} bytes")
    
    if args.compare_cuda:
        print("\nTo compare with CUDA results:")
        print(f"Run: ./cuda_matrix_mult {args.matrix_size}")
        print("Expected CUDA speedup: 10-50x depending on GPU")

if __name__ == "__main__":
    main()
