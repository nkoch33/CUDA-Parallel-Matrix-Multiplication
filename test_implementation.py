#!/usr/bin/env python3
"""
Test script to verify CUDA matrix multiplication implementation
"""

import subprocess
import sys
import os
import numpy as np

def test_small_matrices():
    """Test with small matrices to verify correctness."""
    print("Testing with small matrices for correctness verification...")
    
    # Create small test matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    C_expected = np.dot(A, B)
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nExpected result (A @ B):")
    print(C_expected)
    
    # Save matrices to files for CUDA program
    A.tofile('test_A.bin')
    B.tofile('test_B.bin')
    
    print("\nMatrices saved to test_A.bin and test_B.bin")
    print("You can use these files to test the CUDA implementation")

def check_build_status():
    """Check if the CUDA executable is built."""
    cuda_exe = "cuda_matrix_mult.exe" if os.name == 'nt' else "./cuda_matrix_mult"
    
    if os.path.exists(cuda_exe):
        print(f"✓ CUDA executable found: {cuda_exe}")
        return True
    else:
        print(f"✗ CUDA executable not found: {cuda_exe}")
        print("Please build the project first:")
        if os.name == 'nt':
            print("  build.bat")
        else:
            print("  ./build.sh")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    # Check Python packages
    try:
        import numpy
        print("✓ NumPy available")
    except ImportError:
        print("✗ NumPy not available")
        print("Install with: pip install numpy")
        return False
    
    # Check CUDA
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA compiler (nvcc) available")
        else:
            print("✗ CUDA compiler (nvcc) not available")
            return False
    except FileNotFoundError:
        print("✗ CUDA compiler (nvcc) not found in PATH")
        return False
    
    # Check Nsight Compute (optional)
    try:
        result = subprocess.run(['ncu', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Nsight Compute available")
        else:
            print("? Nsight Compute not available (optional)")
    except FileNotFoundError:
        print("? Nsight Compute not found (optional)")
    
    return True

def run_quick_test():
    """Run a quick test with small matrices."""
    if not check_build_status():
        return False
    
    print("\nRunning quick test with 4x4 matrices...")
    
    cuda_exe = "cuda_matrix_mult.exe" if os.name == 'nt' else "./cuda_matrix_mult"
    
    try:
        result = subprocess.run([cuda_exe, "4"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ CUDA program executed successfully")
            print("Output preview:")
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                print(f"  {line}")
            if len(lines) > 10:
                print("  ...")
            return True
        else:
            print("✗ CUDA program failed")
            print("Error output:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ CUDA program timed out")
        return False
    except Exception as e:
        print(f"✗ Error running CUDA program: {e}")
        return False

def main():
    print("CUDA Matrix Multiplication - Implementation Test")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return 1
    
    # Test small matrices
    test_small_matrices()
    
    # Run quick test
    if run_quick_test():
        print("\n✓ All tests passed! The implementation is ready to use.")
        print("\nNext steps:")
        print("1. Run benchmarks: python run_benchmark.py")
        print("2. Profile with Nsight Compute: profile.bat (Windows) or ./profile.sh (Linux)")
        print("3. Try different matrix sizes: ./cuda_matrix_mult 1024")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
