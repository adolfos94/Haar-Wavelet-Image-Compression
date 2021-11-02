#pragma once

#include "../pch.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DELTA 30
#define DIM_MASCARA 8
#define LONG_MASCARA 64

namespace HaarWaveletWrapper
{
	void MultIzquierdaGPU(dim3 gridSize, dim3 blockSize, float* const imagenOriginal, float* const imagenSalida,
		float* const mascara, int filas, int columnas, bool setQuantization);

	void MultDerechaGPU(dim3 gridSize, dim3 blockSize, float* const imagenOriginal, float* const imagenSalida,
		float* const mascara, int filas, int columnas, bool setQuantization);

	void MultDerechaCPU(float* const imagenOriginal, float* const imagenSalida,
		float* const mascara, int filas, int columnas, bool setQuantization);

	void MultIzquierdaCPU(float* const imagenOriginal, float* const imagenSalida,
		float* const mascara, int filas, int columnas, bool setQuantization);
};