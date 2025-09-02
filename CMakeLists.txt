cmake_minimum_required(VERSION 3.18)
project(CUDAMatrixMultiplication LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture (adjust based on your GPU)
# Common architectures:
# - sm_50: Maxwell (GTX 900 series)
# - sm_60: Pascal (GTX 1000 series)
# - sm_70: Volta (V100, Titan V)
# - sm_75: Turing (RTX 2000 series, GTX 1600 series)
# - sm_80: Ampere (RTX 3000 series, A100)
# - sm_86: Ampere (RTX 3000 series)
# - sm_89: Ada Lovelace (RTX 4000 series)

# Auto-detect CUDA architecture or set manually
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Default to Turing architecture
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -Wall,-Wextra")

# Add executable
add_executable(cuda_matrix_mult cuda_matrix_mult.cu)

# Set CUDA properties
set_target_properties(cuda_matrix_mult PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries
target_link_libraries(cuda_matrix_mult ${CUDA_LIBRARIES})

# Optional: Add OpenMP for CPU parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cuda_matrix_mult OpenMP::OpenMP_CXX)
    target_compile_definitions(cuda_matrix_mult PRIVATE USE_OPENMP)
endif()

# Installation
install(TARGETS cuda_matrix_mult
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Add custom targets for different matrix sizes
add_custom_target(benchmark_512
    COMMAND cuda_matrix_mult 512
    DEPENDS cuda_matrix_mult
    COMMENT "Running benchmark with 512x512 matrices"
)

add_custom_target(benchmark_1024
    COMMAND cuda_matrix_mult 1024
    DEPENDS cuda_matrix_mult
    COMMENT "Running benchmark with 1024x1024 matrices"
)

add_custom_target(benchmark_2048
    COMMAND cuda_matrix_mult 2048
    DEPENDS cuda_matrix_mult
    COMMENT "Running benchmark with 2048x2048 matrices"
)

add_custom_target(benchmark_4096
    COMMAND cuda_matrix_mult 4096
    DEPENDS cuda_matrix_mult
    COMMENT "Running benchmark with 4096x4096 matrices"
)
