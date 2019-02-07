# Haar Parallel DWT

This is an implementation of the Haar Wavelet in CUDA. The purpose of this project, is to compress images using the DWT. It is a hybrid scheme, that uses RLE as entropy coder.

based on: [Haar Wavelet Image Compression](http://aix1.uottawa.ca/~jkhoury/haar.htm)

## Haar DWT in CUDA

```
    __global__ void multDerechaGPU(float* const imagenOriginal, float* const imagenSalida, 
	float* const mascara, int filas, int columnas, bool setQuantization)
```
```
    __global__ void multIzquierdaGPU(float* const imagenOriginal, float* const imagenSalida, 
	float* const mascara, int filas, int columnas,bool setQuantization)
```

## Quality Parameters
 - Compress Ratio
 - MSE
 - PSNR
 - SSIM