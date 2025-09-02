#!/bin/bash

echo "Building CUDA Matrix Multiplication Project..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
make -j$(nproc)

# Copy executable to parent directory for easy access
cp cuda_matrix_mult ../cuda_matrix_mult

echo "Build completed successfully!"
echo ""
echo "Usage:"
echo "  ./cuda_matrix_mult [matrix_size]"
echo ""
echo "Examples:"
echo "  ./cuda_matrix_mult 1024"
echo "  ./cuda_matrix_mult 2048"
