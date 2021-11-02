// Haar Wavelet Image Compression.cpp : Defines the entry point for the application.
// @author: Adolfo Solis-Rosas.

#include "Haar Wavelet Image Compression.h"

using namespace cv;
using namespace std;

// quality-metric
namespace qm {
#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)

	// sigma on block_size
	double sigma(Mat m, int i, int j, int block_size)
	{
		double sd = 0;

		Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
		Mat m_squared(block_size, block_size, CV_64F);

		multiply(m_tmp, m_tmp, m_squared);

		// E(x)
		double avg = mean(m_tmp)[0];
		// E(x²)
		double avg_2 = mean(m_squared)[0];

		sd = sqrt(avg_2 - avg * avg);

		return sd;
	}

	// Covariance
	double cov(Mat m1, Mat m2, int i, int j, int block_size)
	{
		Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
		Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
		Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));

		multiply(m1_tmp, m2_tmp, m3);

		double avg_ro = mean(m3)[0]; // E(XY)
		double avg_r = mean(m1_tmp)[0]; // E(X)
		double avg_o = mean(m2_tmp)[0]; // E(Y)

		double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)

		return sd_ro;
	}

	// Mean squared error
	double eqm(Mat img1, Mat img2)
	{
		int i, j;
		double eqm = 0;
		int height = img1.rows;
		int width = img1.cols;

		for (i = 0; i < height; i++)
			for (j = 0; j < width; j++)
				eqm += (img1.at<double>(i, j) - img2.at<double>(i, j)) * (img1.at<double>(i, j) - img2.at<double>(i, j));

		eqm /= height * width;

		return eqm;
	}

	/**
	 *    Compute the PSNR between 2 images
	 */
	double psnr(Mat img_src, Mat img_compressed, int block_size)
	{
		int D = 255;
		return (10 * log10((D * D) / eqm(img_src, img_compressed)));
	}

	/**
	 * Compute the SSIM between 2 images
	 */
	double ssim(Mat img_src, Mat img_compressed, int block_size, bool show_progress = false)
	{
		double ssim = 0;

		int nbBlockPerHeight = img_src.rows / block_size;
		int nbBlockPerWidth = img_src.cols / block_size;

		for (int k = 0; k < nbBlockPerHeight; k++)
		{
			for (int l = 0; l < nbBlockPerWidth; l++)
			{
				int m = k * block_size;
				int n = l * block_size;

				double avg_o = mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
				double avg_r = mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
				double sigma_o = sigma(img_src, m, n, block_size);
				double sigma_r = sigma(img_compressed, m, n, block_size);
				double sigma_ro = cov(img_src, img_compressed, m, n, block_size);

				ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
			}
			// Progress
			if (show_progress)
				cout << "\r>>SSIM [" << (int)((((double)k) / nbBlockPerHeight) * 100) << "%]";
		}
		ssim /= nbBlockPerHeight * nbBlockPerWidth;

		if (show_progress)
		{
			cout << "\r>>SSIM [100%]" << endl;
			cout << "SSIM : " << ssim << endl;
		}

		return ssim;
	}
}

Mat toZigZag(Mat imageMatrix) {
	Mat imageArray = Mat::zeros(1, imageMatrix.cols * imageMatrix.rows, CV_32F);

	int indexImageArray = 0;

	int index_X = 0;
	int index_Y = 0;

	while (indexImageArray < (imageMatrix.cols * imageMatrix.rows)) {
		if (index_X < imageMatrix.cols - 1) {
			imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
			indexImageArray++;
			index_X++;

			while (index_X > 0) {
				imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
				indexImageArray++;
				index_X--;
				index_Y++;
			}
		}
		else if (index_X == imageMatrix.cols - 1) {
			imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
			indexImageArray++;
			index_Y++;

			while (index_Y < imageMatrix.rows - 1) {
				imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
				indexImageArray++;
				index_X--;
				index_Y++;
			}
		}
		if (index_Y < imageMatrix.rows - 1) {
			imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
			indexImageArray++;
			index_Y++;

			while (index_Y > 0) {
				imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
				indexImageArray++;
				index_Y--;
				index_X++;
			}
		}
		else if (index_Y == imageMatrix.rows - 1) {
			imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
			indexImageArray++;
			index_X++;

			while (index_X < imageMatrix.cols - 1) {
				imageArray.at<float>(0, indexImageArray) = imageMatrix.at<float>(index_X, index_Y);
				indexImageArray++;
				index_Y--;
				index_X++;
			}
		}
	}
	return imageArray;
}

Mat invZigZag(Mat imageArray) {
	Mat imageMatrix = Mat::zeros(sqrt(imageArray.cols), sqrt(imageArray.cols), CV_32F);

	int indexImageArray = 0;

	int index_X = 0;
	int index_Y = 0;

	while (indexImageArray < (imageMatrix.cols * imageMatrix.rows)) {
		if (index_X < imageMatrix.cols - 1) {
			imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
			indexImageArray++;
			index_X++;

			while (index_X > 0) {
				imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
				indexImageArray++;
				index_X--;
				index_Y++;
			}
		}
		else if (index_X == imageMatrix.cols - 1) {
			imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
			indexImageArray++;
			index_Y++;

			while (index_Y < imageMatrix.rows - 1) {
				imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
				indexImageArray++;
				index_X--;
				index_Y++;
			}
		}
		if (index_Y < imageMatrix.rows - 1) {
			imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
			indexImageArray++;
			index_Y++;

			while (index_Y > 0) {
				imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
				indexImageArray++;
				index_Y--;
				index_X++;
			}
		}
		else if (index_Y == imageMatrix.rows - 1) {
			imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
			indexImageArray++;
			index_X++;

			while (index_X < imageMatrix.cols - 1) {
				imageMatrix.at<float>(index_X, index_Y) = imageArray.at<float>(0, indexImageArray);
				indexImageArray++;
				index_Y--;
				index_X++;
			}
		}
	}
	return imageMatrix;
}

Mat encodeRLE(Mat imageArray) {
	Mat imageArrayCoded;

	float _total_rep = 1;
	float _total_nums_non_con = 0;

	for (int i = 0; i < imageArray.cols; i++) {
		float actual;
		float next;

		if (i < imageArray.cols - 2) {
			actual = imageArray.at<float>(0, i);
			next = imageArray.at<float>(0, i + 1);
		}
		else {
			actual = imageArray.at<float>(0, i);
			next = INT_MAX;
		}

		if (actual == next) {
			_total_rep++;
		}
		else {
			_total_nums_non_con++;
			_total_rep = 1;
		}
	}

	imageArrayCoded = Mat(2, _total_nums_non_con, CV_32F, float(0));
	_total_nums_non_con = 0;

	for (int i = 0; i < imageArray.cols; i++) {
		float actual;
		float next;

		if (i < imageArray.cols - 2) {
			actual = imageArray.at<float>(0, i);
			next = imageArray.at<float>(0, i + 1);
		}
		else {
			actual = imageArray.at<float>(0, i);
			next = INT_MAX;
		}

		if (actual == next) {
			_total_rep++;
		}
		else {
			imageArrayCoded.at<float>(0, _total_nums_non_con) = actual;
			imageArrayCoded.at<float>(1, _total_nums_non_con) = _total_rep;
			_total_nums_non_con++;
			_total_rep = 1;
		}
	}

	return imageArrayCoded;
}

Mat decodeRLE(Mat imageArrayCoded) {
	Mat imageArray;

	float _total_rep = 0;

	for (int i = 0; i < imageArrayCoded.cols; i++)
		_total_rep = _total_rep + imageArrayCoded.at<float>(1, i);

	imageArray = Mat(1, _total_rep, CV_32F, float(0));

	int indexImageArray = 0;

	for (int i = 0; i < imageArrayCoded.cols; i++) {
		float val = imageArrayCoded.at<float>(0, i);
		float num_rep = imageArrayCoded.at<float>(1, i);

		for (int j = 0; j < num_rep; j++) {
			imageArray.at<float>(0, indexImageArray) = val;
			indexImageArray++;
		}
	}

	return imageArray;
}

int main()
{
	cudaFree(0);

	Mat imagen;

	bool setQuantization;

	float* imagenOriginal, * dImagenOriginal, * dMascara;
	float* dimagenTransf1, * dimagenTransf2, * dimagenTransf3, * dimagenTransf4;
	float* imagenTransf1GPU;

	// Path of the image to upload to Device.
	const string nombreImagen = "../../../img/IM-0001-0007.tif";

	imagen = imread(nombreImagen.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	imagen.convertTo(imagen, CV_32F);

	if (imagen.empty()) {
		cout << "Empty Image...";
		return 0;
	}

	int filas = imagen.rows;
	int columnas = imagen.cols;
	const size_t numPixeles = filas * columnas;

	float mascara[LONG_MASCARA] =
	{
		0.3536, 0.3536, 0.5000, 0, 0.7071, 0, 0, 0,
		0.3536, 0.3536, 0.5000, 0, -0.7071, 0, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, 0.7071, 0, 0,
		0.3536, 0.3536, -0.5000, 0, 0, -0.7071, 0, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, 0.7071, 0,
		0.3536, -0.3536, 0, 0.5000, 0, 0, -0.7071, 0,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, 0.7071,
		0.3536, -0.3536, 0, -0.5000, 0, 0, 0, -0.7071
	};

	float mascara_inv[LONG_MASCARA] =
	{
		0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
		0.3536, 0.3536, 0.3536, 0.3536, -0.3536, -0.3536, -0.3536, -0.3536,
		0.5000, 0.5000, -0.5000, -0.5000, 0, 0, 0, 0,
		0, 0, 0, 0, 0.5000, 0.5000, -0.5000, -0.5000,
		0.7071, -0.7071, 0, 0, 0, 0, 0, 0,
		0, 0, 0.7071, -0.7071, 0, 0, 0, 0,
		0, 0, 0, 0, 0.7071, -0.7071, 0, 0,
		0, 0, 0, 0, 0, 0, 0.7071, -0.7071
	};

	imagenOriginal = (float*)imagen.ptr<float>(0);
	imagenTransf1GPU = (float*)malloc(sizeof(float) * numPixeles);

	int N = DIM_MASCARA, M = DIM_MASCARA;

	const dim3 gridSize(filas / M, columnas / N, 1);
	const dim3 blockSize(M, N, 1);

	cudaMalloc(&dImagenOriginal, sizeof(float) * numPixeles);
	cudaMalloc(&dMascara, sizeof(float) * LONG_MASCARA);
	cudaMalloc(&dimagenTransf1, sizeof(float) * numPixeles);
	cudaMalloc(&dimagenTransf2, sizeof(float) * numPixeles);
	cudaMalloc(&dimagenTransf3, sizeof(float) * numPixeles);
	cudaMalloc(&dimagenTransf4, sizeof(float) * numPixeles);

	cudaMemcpy(dImagenOriginal, imagenOriginal, sizeof(float) * numPixeles, cudaMemcpyHostToDevice);
	cudaMemcpy(dMascara, mascara_inv, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);

	clock_t timer1 = clock();

	setQuantization = true;

	HaarWaveletWrapper::MultDerechaGPU(gridSize, blockSize, dImagenOriginal, dimagenTransf1, dMascara,
		filas, columnas, setQuantization);

	cudaMemcpy(dMascara, mascara, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);

	HaarWaveletWrapper::MultIzquierdaGPU(gridSize, blockSize, dimagenTransf1, dimagenTransf2, dMascara,
		filas, columnas, setQuantization);

	cudaDeviceSynchronize();
	cudaMemcpy(imagenTransf1GPU, dimagenTransf2, sizeof(float) * numPixeles, cudaMemcpyDeviceToHost);

	Mat imagenWaveletTransformadaGPU(filas, columnas, CV_32F, imagenTransf1GPU);

	Mat ImageArray = toZigZag(imagenWaveletTransformadaGPU);

	Mat ImageArrayCompressed = encodeRLE(ImageArray);

	int InputBitcost = imagen.rows * imagen.cols * 8;
	cout << "InputBitcost: " << InputBitcost << endl;

	float OutputBitcost = ImageArrayCompressed.rows * ImageArrayCompressed.cols * 8;
	cout << "OutputBitcost: " << OutputBitcost << endl;

	// ------ END -------

	ImageArray = decodeRLE(ImageArrayCompressed);

	imagenWaveletTransformadaGPU = invZigZag(ImageArray);

	cudaMemcpy(dImagenOriginal, (float*)imagenWaveletTransformadaGPU.ptr<float>(0), sizeof(float) * numPixeles, cudaMemcpyHostToDevice);
	cudaMemcpy(dMascara, mascara, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);

	setQuantization = false;

	HaarWaveletWrapper::MultDerechaGPU(gridSize, blockSize, dImagenOriginal, dimagenTransf1, dMascara,
		filas, columnas, setQuantization);

	cudaMemcpy(dMascara, mascara_inv, sizeof(float) * LONG_MASCARA, cudaMemcpyHostToDevice);

	HaarWaveletWrapper::MultIzquierdaGPU(gridSize, blockSize, dimagenTransf1, dimagenTransf2, dMascara,
		filas, columnas, setQuantization);

	cudaDeviceSynchronize();
	cudaMemcpy(imagenTransf1GPU, dimagenTransf2, sizeof(float) * numPixeles, cudaMemcpyDeviceToHost);

	timer1 = clock() - timer1;

	printf("Image Size: [%d, %d]\n", filas, columnas);
	printf("GPU Execution Time is %10.3f ms.\n", ((timer1) / double(CLOCKS_PER_SEC) * 1000));

	Mat imagenRecuperada(filas, columnas, CV_32F, imagenTransf1GPU);

	imagenRecuperada.convertTo(imagenRecuperada, CV_64F);
	imagen.convertTo(imagen, CV_64F);

	// ------ IMAGE METRICS -----
	cout << "------IMAGE METRICS-----" << endl;
	// CR
	cout << "The CR value is: " << InputBitcost / OutputBitcost << endl;
	// MSE
	cout << "The MSE value is: " << qm::eqm(imagen, imagenRecuperada) << endl;
	// PSNR
	cout << "The PSNR value is: " << qm::psnr(imagen, imagenRecuperada, 1) << endl;
	// SSIM
	cout << "The SSIM value is: " << qm::ssim(imagen, imagenRecuperada, 1) << endl;

	imagenRecuperada.convertTo(imagenRecuperada, CV_8UC1);
	imagen.convertTo(imagen, CV_8UC1);

	imshow("Original", imagen);
	imshow("Compressed in GPU", imagenRecuperada);
	waitKey(0);

	return 0;
};