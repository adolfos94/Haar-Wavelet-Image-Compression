#include "HaarWavelet.h"

__global__ void multDerechaGPU(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {
	int fila = (blockIdx.x * blockDim.x) + threadIdx.x;
	int columna = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (columna >= columnas || fila >= filas) {
		return;
	}

	int indice = fila * columnas + columna;

	int filaIni = (blockIdx.x * blockDim.x);
	int columIni = (blockIdx.y * blockDim.y);

	int r = threadIdx.x;  /// renglon inicial del bloque 8 x 8
	int c = threadIdx.y;  /// columna inicial del bloque 8 x 8

	float total = 0;
	for (int k = 0; k < DIM_MASCARA; k++) {
		int indMascara = (r * DIM_MASCARA) + k;
		int indImagen = ((filaIni + k) * columnas) + (c + columIni);
		if (setQuantization)
			total = total + mascara[indMascara] * imagenOriginal[indImagen];
		else
			total = total + mascara[indMascara] * (imagenOriginal[indImagen] * DELTA);
	}
	imagenSalida[indice] = total;
}

__global__ void multIzquierdaGPU(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {
	int fila = (blockIdx.x * blockDim.x) + threadIdx.x;
	int columna = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (columna >= columnas || fila >= filas) {
		return;
	}

	int indice = fila * columnas + columna;

	int filaIni = (blockIdx.x * blockDim.x);
	int columIni = (blockIdx.y * blockDim.y);

	int r = threadIdx.x;  /// renglon inicial del bloque 8 x 8
	int c = threadIdx.y;  /// columna inicial del bloque 8 x 8

	float total = 0;
	for (int k = 0; k < DIM_MASCARA; k++) {
		int indMascara = (k * DIM_MASCARA) + c;
		int indImagen = ((filaIni + r) * filas) + (k + columIni);
		total = total + mascara[indMascara] * imagenOriginal[indImagen];
	}
	if (setQuantization)
		imagenSalida[indice] = std::round(total / DELTA);
	else
		imagenSalida[indice] = total;
}

void HaarWaveletWrapper::MultDerechaCPU(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {
	for (int fila = 0; fila < filas; fila += DIM_MASCARA) {
		for (int columna = 0; columna < columnas; columna += DIM_MASCARA) {
			for (int r = 0; r < DIM_MASCARA; r++) {
				for (int c = 0; c < DIM_MASCARA; c++) {
					float total = 0;
					for (int k = 0; k < DIM_MASCARA; k++) {
						int indMascara = (r * DIM_MASCARA) + k;
						int indImagen = ((fila + k) * columnas) + (c + columna);
						if (setQuantization)
							total = total + mascara[indMascara] * imagenOriginal[indImagen];
						else
							total = total + mascara[indMascara] * (imagenOriginal[indImagen] * DELTA);
					}
					int indice = (fila + r) * columnas + (columna + c);
					imagenSalida[indice] = total;
				}
			}
		}
	}
}

void HaarWaveletWrapper::MultIzquierdaCPU(float* const imagenOriginal, float* const imagenSalida,
	float* const mascara, int filas, int columnas, bool setQuantization) {
	for (int fila = 0; fila < filas; fila += DIM_MASCARA) {
		for (int columna = 0; columna < columnas; columna += DIM_MASCARA) {
			for (int r = 0; r < DIM_MASCARA; r++) {
				for (int c = 0; c < DIM_MASCARA; c++) {
					float total = 0;
					for (int k = 0; k < DIM_MASCARA; k++) {
						int indMascara = (k * DIM_MASCARA) + c;
						int indImagen = ((fila + r) * filas) + (k + columna);
						total = total + mascara[indMascara] * imagenOriginal[indImagen];
					}
					int indice = (fila + r) * columnas + (columna + c);
					if (setQuantization)
						imagenSalida[indice] = round(total / DELTA);
					else
						imagenSalida[indice] = total;
				}
			}
		}
	}
}

void HaarWaveletWrapper::MultIzquierdaGPU(dim3 gridSize, dim3 blockSize, float* const imagenOriginal, float* const imagenSalida, float* const mascara, int filas, int columnas, bool setQuantization)
{
	multIzquierdaGPU << <gridSize, blockSize >> >
		(imagenOriginal, imagenSalida, mascara,
			filas, columnas, setQuantization);
}

void HaarWaveletWrapper::MultDerechaGPU(dim3 gridSize, dim3 blockSize, float* const imagenOriginal, float* const imagenSalida, float* const mascara, int filas, int columnas, bool setQuantization)
{
	multDerechaGPU << <gridSize, blockSize >> >
		(imagenOriginal, imagenSalida, mascara,
			filas, columnas, setQuantization);
}