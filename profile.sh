#!/bin/bash

echo "CUDA Matrix Multiplication Profiling with Nsight Compute"
echo "========================================================"

# Check if executable exists
if [ ! -f "./cuda_matrix_mult" ]; then
    echo "Error: cuda_matrix_mult not found. Please build the project first."
    echo "Run: ./build.sh"
    exit 1
fi

# Check if Nsight Compute is available
if ! command -v ncu &> /dev/null; then
    echo "Error: Nsight Compute (ncu) not found in PATH."
    echo "Please install NVIDIA Nsight Compute and add it to your PATH."
    exit 1
fi

echo "Starting profiling session..."
echo

# Profile with different matrix sizes
echo "Profiling 512x512 matrices..."
ncu --config-file nsight_compute_config.ncu --output-file profile_512x512 ./cuda_matrix_mult 512

echo
echo "Profiling 1024x1024 matrices..."
ncu --config-file nsight_compute_config.ncu --output-file profile_1024x1024 ./cuda_matrix_mult 1024

echo
echo "Profiling 2048x2048 matrices..."
ncu --config-file nsight_compute_config.ncu --output-file profile_2048x2048 ./cuda_matrix_mult 2048

echo
echo "Profiling completed!"
echo "Results saved as:"
echo "  - profile_512x512.ncu-rep"
echo "  - profile_1024x1024.ncu-rep"
echo "  - profile_2048x2048.ncu-rep"
echo
echo "Open these files in Nsight Compute GUI for detailed analysis."
