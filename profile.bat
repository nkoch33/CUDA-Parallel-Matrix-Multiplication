@echo off
echo CUDA Matrix Multiplication Profiling with Nsight Compute
echo ========================================================

REM Check if executable exists
if not exist cuda_matrix_mult.exe (
    echo Error: cuda_matrix_mult.exe not found. Please build the project first.
    echo Run: build.bat
    pause
    exit /b 1
)

REM Check if Nsight Compute is available
where ncu >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Nsight Compute (ncu) not found in PATH.
    echo Please install NVIDIA Nsight Compute and add it to your PATH.
    pause
    exit /b 1
)

echo Starting profiling session...
echo.

REM Profile with different matrix sizes
echo Profiling 512x512 matrices...
ncu --config-file nsight_compute_config.ncu --output-file profile_512x512 cuda_matrix_mult.exe 512

echo.
echo Profiling 1024x1024 matrices...
ncu --config-file nsight_compute_config.ncu --output-file profile_1024x1024 cuda_matrix_mult.exe 1024

echo.
echo Profiling 2048x2048 matrices...
ncu --config-file nsight_compute_config.ncu --output-file profile_2048x2048 cuda_matrix_mult.exe 2048

echo.
echo Profiling completed!
echo Results saved as:
echo   - profile_512x512.ncu-rep
echo   - profile_1024x1024.ncu-rep
echo   - profile_2048x2048.ncu-rep
echo.
echo Open these files in Nsight Compute GUI for detailed analysis.
echo.
pause
