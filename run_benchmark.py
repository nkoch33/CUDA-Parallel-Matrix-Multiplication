#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner
Runs both NumPy and CUDA benchmarks for comparison
"""

import subprocess
import sys
import time
import os
import argparse
from typing import Dict, List, Tuple

def run_command(command: str, timeout: int = 300) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, str(e)

def run_numpy_benchmark(matrix_size: int) -> Dict:
    """Run NumPy benchmark and parse results."""
    print(f"Running NumPy benchmark for {matrix_size}x{matrix_size} matrices...")
    
    success, output = run_command(f"python numpy_comparison.py {matrix_size}")
    
    if not success:
        return {"error": f"NumPy benchmark failed: {output}"}
    
    # Parse output for timing information
    lines = output.split('\n')
    numpy_time = None
    
    for line in lines:
        if "NumPy (BLAS) time:" in line:
            try:
                numpy_time = float(line.split(':')[1].strip().split()[0])
                break
            except (IndexError, ValueError):
                continue
    
    return {
        "numpy_time_ms": numpy_time,
        "output": output
    }

def run_cuda_benchmark(matrix_size: int) -> Dict:
    """Run CUDA benchmark and parse results."""
    print(f"Running CUDA benchmark for {matrix_size}x{matrix_size} matrices...")
    
    # Check if CUDA executable exists
    cuda_exe = "cuda_matrix_mult.exe" if os.name == 'nt' else "./cuda_matrix_mult"
    if not os.path.exists(cuda_exe):
        return {"error": f"CUDA executable not found: {cuda_exe}"}
    
    success, output = run_command(f"{cuda_exe} {matrix_size}")
    
    if not success:
        return {"error": f"CUDA benchmark failed: {output}"}
    
    # Parse output for timing information
    lines = output.split('\n')
    results = {}
    
    for line in lines:
        if "CPU time:" in line:
            try:
                results["cpu_time_ms"] = float(line.split(':')[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "GPU simple time:" in line:
            try:
                results["gpu_simple_time_ms"] = float(line.split(':')[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "GPU tiled time:" in line:
            try:
                results["gpu_tiled_time_ms"] = float(line.split(':')[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "Speedup vs CPU:" in line and "tiled" in line.lower():
            try:
                results["cuda_speedup"] = float(line.split(':')[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
    
    results["output"] = output
    return results

def print_comparison_table(results: List[Dict], matrix_sizes: List[int]):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*80)
    print(f"{'Matrix Size':<12} {'NumPy (ms)':<12} {'CUDA (ms)':<12} {'Speedup':<10} {'Status':<10}")
    print("-"*80)
    
    for i, size in enumerate(matrix_sizes):
        result = results[i]
        
        if "error" in result:
            print(f"{size}x{size:<8} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'FAILED':<10}")
            continue
        
        numpy_time = result.get("numpy_time_ms", "N/A")
        cuda_time = result.get("gpu_tiled_time_ms", "N/A")
        speedup = result.get("cuda_speedup", "N/A")
        
        if isinstance(numpy_time, float) and isinstance(cuda_time, float):
            calculated_speedup = numpy_time / cuda_time
            status = "SUCCESS"
        else:
            calculated_speedup = "N/A"
            status = "PARTIAL"
        
        print(f"{size}x{size:<8} {numpy_time:<12} {cuda_time:<12} {speedup:<10} {status:<10}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive CUDA vs NumPy Benchmark')
    parser.add_argument('--sizes', nargs='+', type=int, default=[512, 1024, 2048],
                       help='Matrix sizes to test (default: 512 1024 2048)')
    parser.add_argument('--numpy-only', action='store_true',
                       help='Run only NumPy benchmarks')
    parser.add_argument('--cuda-only', action='store_true',
                       help='Run only CUDA benchmarks')
    parser.add_argument('--output', type=str,
                       help='Save results to file')
    
    args = parser.parse_args()
    
    print("CUDA vs NumPy Matrix Multiplication Benchmark")
    print("=" * 50)
    print(f"Matrix sizes: {args.sizes}")
    print(f"NumPy only: {args.numpy_only}")
    print(f"CUDA only: {args.cuda_only}")
    print()
    
    results = []
    
    for size in args.sizes:
        print(f"\n{'='*20} Testing {size}x{size} {'='*20}")
        
        result = {"matrix_size": size}
        
        # Run NumPy benchmark
        if not args.cuda_only:
            numpy_result = run_numpy_benchmark(size)
            result.update(numpy_result)
        
        # Run CUDA benchmark
        if not args.numpy_only:
            cuda_result = run_cuda_benchmark(size)
            result.update(cuda_result)
        
        results.append(result)
        
        # Print individual results
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            if "numpy_time_ms" in result:
                print(f"NumPy time: {result['numpy_time_ms']:.2f} ms")
            if "gpu_tiled_time_ms" in result:
                print(f"CUDA time: {result['gpu_tiled_time_ms']:.2f} ms")
            if "cuda_speedup" in result:
                print(f"CUDA speedup: {result['cuda_speedup']:.2f}x")
    
    # Print comparison table
    if not args.numpy_only and not args.cuda_only:
        print_comparison_table(results, args.sizes)
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
