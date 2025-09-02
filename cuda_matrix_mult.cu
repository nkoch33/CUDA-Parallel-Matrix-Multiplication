#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

// TILE_SIZE defines the size of the shared memory tile
// Common values: 16, 32 (32x32 = 1024 threads per block, good for most GPUs)
#define TILE_SIZE 32

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Simple matrix multiplication kernel (baseline)
__global__ void matrix_mult_simple(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized matrix multiplication kernel with shared memory tiling
__global__ void matrix_mult_tiled(float* A, float* B, float* C, int N) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Global indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load data into shared memory
        int A_col = tile * TILE_SIZE + tx;
        int B_row = tile * TILE_SIZE + ty;
        
        // Boundary checks for loading
        if (row < N && A_col < N) {
            As[ty][tx] = A[row * N + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (B_row < N && col < N) {
            Bs[ty][tx] = B[B_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU reference implementation
void matrix_mult_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Initialize matrix with random values
void init_matrix(float* matrix, int N, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < N * N; i++) {
        matrix[i] = dis(gen);
    }
}

// Verify results between CPU and GPU
bool verify_results(float* cpu_result, float* gpu_result, int N, float tolerance = 1e-5f) {
    for (int i = 0; i < N * N; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

// Performance measurement utility
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

int main(int argc, char* argv[]) {
    // Default matrix size
    int N = 1024;
    
    // Parse command line arguments
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            printf("Invalid matrix size. Using default size 1024.\n");
            N = 1024;
        }
    }
    
    printf("CUDA Matrix Multiplication Benchmark\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("=====================================\n\n");
    
    // Allocate host memory
    size_t matrix_size = N * N * sizeof(float);
    float* h_A = (float*)malloc(matrix_size);
    float* h_B = (float*)malloc(matrix_size);
    float* h_C_cpu = (float*)malloc(matrix_size);
    float* h_C_gpu_simple = (float*)malloc(matrix_size);
    float* h_C_gpu_tiled = (float*)malloc(matrix_size);
    
    // Allocate device memory
    float* d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_C, matrix_size));
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    init_matrix(h_A, N);
    init_matrix(h_B, N);
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice));
    
    // Set up CUDA grid and block dimensions
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    Timer timer;
    double cpu_time, gpu_simple_time, gpu_tiled_time;
    
    // CPU computation
    printf("Running CPU reference implementation...\n");
    timer.start();
    matrix_mult_cpu(h_A, h_B, h_C_cpu, N);
    cpu_time = timer.stop();
    printf("CPU time: %.2f ms\n\n", cpu_time);
    
    // GPU simple kernel
    printf("Running GPU simple kernel...\n");
    CUDA_CHECK(cudaMemset(d_C, 0, matrix_size));
    timer.start();
    matrix_mult_simple<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_simple_time = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_C_gpu_simple, d_C, matrix_size, cudaMemcpyDeviceToHost));
    printf("GPU simple time: %.2f ms\n", gpu_simple_time);
    printf("Speedup vs CPU: %.2fx\n\n", cpu_time / gpu_simple_time);
    
    // GPU tiled kernel
    printf("Running GPU tiled kernel...\n");
    CUDA_CHECK(cudaMemset(d_C, 0, matrix_size));
    timer.start();
    matrix_mult_tiled<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_tiled_time = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled, d_C, matrix_size, cudaMemcpyDeviceToHost));
    printf("GPU tiled time: %.2f ms\n", gpu_tiled_time);
    printf("Speedup vs CPU: %.2fx\n", cpu_time / gpu_tiled_time);
    printf("Speedup vs simple GPU: %.2fx\n\n", gpu_simple_time / gpu_tiled_time);
    
    // Verify results
    printf("Verifying results...\n");
    bool simple_correct = verify_results(h_C_cpu, h_C_gpu_simple, N);
    bool tiled_correct = verify_results(h_C_cpu, h_C_gpu_tiled, N);
    
    printf("Simple kernel correct: %s\n", simple_correct ? "YES" : "NO");
    printf("Tiled kernel correct: %s\n", tiled_correct ? "YES" : "NO");
    
    // Performance metrics
    double gflops = (2.0 * N * N * N) / (gpu_tiled_time * 1e6); // Convert ms to seconds
    printf("\nPerformance Metrics:\n");
    printf("Peak GFLOPS (tiled): %.2f\n", gflops);
    printf("Memory bandwidth: %.2f GB/s\n", 
           (3.0 * matrix_size) / (gpu_tiled_time * 1e6)); // 3 matrices: A, B, C
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu_simple);
    free(h_C_gpu_tiled);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    printf("\nBenchmark completed successfully!\n");
    return 0;
}
