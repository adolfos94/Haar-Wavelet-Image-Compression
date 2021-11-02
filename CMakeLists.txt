# CMakeList.txt : CMake project for Haar Wavelet Image Compression, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project ("Haar Wavelet Image Compression" LANGUAGES CXX CUDA)

FIND_PACKAGE( OpenCV REQUIRED )                              
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS})

# Add source to this project's executable.
add_executable (CMakeTarget "Haar Wavelet Image Compression.cpp" "src/HaarWavelet.cu")
TARGET_LINK_LIBRARIES (CMakeTarget ${OpenCV_LIBS})    
TARGET_LINK_LIBRARIES (CMakeTarget ${CUDA_LIBS})    