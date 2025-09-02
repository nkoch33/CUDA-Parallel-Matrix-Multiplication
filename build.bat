@echo off
echo Building CUDA Matrix Multiplication Project...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64

REM Build the project
echo Building project...
cmake --build . --config Release

REM Copy executable to parent directory for easy access
copy Release\cuda_matrix_mult.exe ..\cuda_matrix_mult.exe

echo Build completed successfully!
echo.
echo Usage:
echo   cuda_matrix_mult.exe [matrix_size]
echo.
echo Examples:
echo   cuda_matrix_mult.exe 1024
echo   cuda_matrix_mult.exe 2048
echo.
pause
