//============================================================================
// Name    : Dip3.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip3.h"

#include <stdexcept>


namespace dip3 {

const char * const filterModeNames[NUM_FILTER_MODES] = {
    "FM_SPATIAL_CONVOLUTION",
    "FM_FREQUENCY_CONVOLUTION",
    "FM_SEPERABLE_FILTER",
    "FM_INTEGRAL_IMAGE",
};



/**
 * @brief Generates 1D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel1D(int kSize){

	float sigma, mean;
	/*normalizing the values*/
	float sum = 0;
	/*1-D array of size = kSize*/
	cv::Mat_<float> result = cv::Mat_<float>::zeros(1, kSize);
	/*calculate median*/
	if ((kSize%5) == 0)
	{
		sigma = kSize / 5.0f;
	}
	else
	{
		sigma = (kSize + 1)/ 5.0f;
	}

	//std::cout << "sigma:" << sigma << "\n";
	/*calculate mean*/
	if ((kSize%2) == 0)
	{
		mean = kSize / 2.0f;
	}
	else
	{
		mean = (kSize + 1) / 2.0f;
	}
	/*calculate gaussian blur*/
	for (int i = 0; i < kSize; i++)
	{
		float denominator, power;
		denominator = sqrt(2 * CV_PI)*sigma;
		power = pow(((i+1) - mean), 2) / (2 * pow(sigma,2));
		result.at<float>(i) = exp(-power) / denominator;
		sum = sum + result.at<float>(i);
	}
	/*normalize the distribution*/
	result = result / sum;
	//std::cout << result << "\n";
	//std::cout << "Sum:" << sum << "\n";
    return result;
}

/**
 * @brief Generates 2D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel2D(int kSize){

	float sigmaX, sigmaY, meanX, meanY;
	/*normalizing the values*/
	float sum = 0;
	/*1-D array of size = kSize*/
	cv::Mat result = cv::Mat(kSize, kSize, CV_32FC1, 0.);
	/*calculate median*/
	if ((kSize % 5) == 0)
	{
		sigmaX = sigmaY = kSize / 5.0f;
	}
	else
	{
		sigmaX = sigmaY = (kSize + 1) / 5.0f;
	}
	/*calculate mean*/
	if ((kSize % 2) == 0)
	{
		meanX = meanY = kSize / 2.0f;
	}
	else
	{
		meanX = meanY = (kSize + 1) / 2.0f;
	}
	/*calculate gaussian blur*/
	for (int i = 0; i < kSize; i++)
	{
		for (int j = 0; j < kSize; j++)
		{
			float denominator, power, xDist, yDist;
			denominator = 2 * CV_PI * sigmaX * sigmaY;
			xDist = pow(((i + 1) - meanX), 2);
			yDist = pow(((j + 1) - meanY), 2);
			power = 0.5*((xDist / pow(sigmaX, 2)) + (yDist / pow(sigmaY, 2)));
			result.at<float>(i,j) = exp(-power) / denominator;
			sum = sum + result.at<float>(i,j);
		}
	}
	/*normalize the distribution*/
	result = result / sum;
	//std::cout << result << "\n";
	//std::cout << "Sum:" << sum << "\n";
	return result;
   
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @return Circular shifted matrix
*/
int modulo(int x, int y)
{
	return (x < 0 ? (((x % y) + y) % y) : (x % y));
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @returns Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float> &in, int dx, int dy)
{
	int mat_size[2];
	mat_size[0] = in.rows;
	mat_size[1] = in.cols;
	cv::Mat_<float> result = cv::Mat_<float>::zeros(in.rows, in.cols);

	for (int i = 0; i < in.rows; i++)
	{
		for (int j = 0; j < in.cols; j++)
		{
			int row = modulo(i + dx, in.rows);
			int col = modulo(j + dy, in.cols);
			result.at<float>(row, col) = in(i, j);
		}
	}
	return result.clone();
}


/**
 * @brief Performes convolution by multiplication in frequency domain
 * @param in Input image
 * @param kernel Filter kernel
 * @returns Output image
 */
cv::Mat_<float> frequencyConvolution(const cv::Mat_<float>& in, const cv::Mat_<float>& kernel){

	int rows, cols;
	cv::Mat_<float> result = cv::Mat::zeros(in.size(), in.type());
	cv::Mat_<float> tempKernel = cv::Mat::zeros(in.size(), in.type());
	cv::Mat_<float> spcInput = cv::Mat::zeros(in.size(), in.type());
	cv::Mat_<float> spcKernel = cv::Mat::zeros(in.size(), in.type());
	cv::Mat_<float> tempKernelShift = cv::Mat::zeros(in.size(), in.type());
	cv::Mat_<float> resultSpc = cv::Mat::zeros(in.size(), in.type());;

	rows = 0;
	cols = 0;

	/*Input spectrum : DFT*/
	cv::dft(in, spcInput, 0);

	/*copy kernel into another matrix same size as the input*/
	for (int i = 0; i < kernel.rows; i++)
	{
		for (int j = 0; j < kernel.cols; j++)
		{
			tempKernel.at<float>(i, j) = kernel.at<float>(i, j);
		}
	}

	/*shift kernel*/
	tempKernelShift = circShift(tempKernel, -kernel.rows / 2, -kernel.rows / 2);

	/*Spectrum of kernel*/
	cv::dft(tempKernelShift, spcKernel, 0);

	/*multiplying input and kernel*/
	cv::mulSpectrums(spcInput, spcKernel, resultSpc, 0);

	/*reversing from frequency to time domain : IDFT*/
	cv::dft(resultSpc, result, cv::DFT_INVERSE + cv::DFT_SCALE);
	return result;
}


/**
 * @brief  Performs UnSharp Masking to enhance fine image structures
 * @param in The input image
 * @param filterMode How convolution for smoothing operation is done
 * @param size Size of used smoothing kernel
 * @param thresh Minimal intensity difference to perform operation
 * @param scale Scaling of edge enhancement
 * @returns Enhanced image
 */
cv::Mat_<float> usm(const cv::Mat_<float>& in, FilterMode filterMode, int size, float thresh, float scale)
{

	cv::Mat_<float> smoothImg = cv::Mat::zeros(in.size(), in.type());
	cv::Mat_<float> noiseImg = cv::Mat::zeros(in.size(), in.type());
	cv::Mat_<float> scaleImg = cv::Mat::zeros(in.size(), in.type());
	cv::Mat_<float> finalImg = cv::Mat::zeros(in.size(), in.type());

	/*applying smoothing filters to the input image*/
	smoothImg = smoothImage(in, size, filterMode);

	/*extracting noise by subtracting smoothed image from the original image*/
	for (int i = 0; i < in.rows; i++)
	{
		for (int j = 0; j < in.cols; j++)
		{
			noiseImg.at<float>(i, j) = in.at<float>(i, j) - smoothImg.at<float>(i, j);
		}
	}


	/*scaling the noisy image*/
	for (int i = 0; i < in.rows; i++)
	{
		for (int j = 0; j < in.cols; j++)
		{
			if (noiseImg.at<float>(i,j) > 0){
				scaleImg.at<float>(i, j) = in.at<float>(i, j) + (scale*(noiseImg.at<float>(i, j)-thresh));
			}
			else if (noiseImg.at<float>(i, j) < 0){
				scaleImg.at<float>(i, j) = in.at<float>(i, j) + (scale*(noiseImg.at<float>(i, j) + thresh));
			}
			else if (std::abs(noiseImg.at<float>(i, j)) < thresh){
				scaleImg.at<float>(i, j) = in.at<float>(i, j);
			}
			else {

			}
		}

	}
   return scaleImg;
}

cv::Mat_<float> getBorderedImage(const cv::Mat_<float> &src, int rowMiddle, int colMiddle)
{
	// Create bordered matrix.
	cv::Mat_<float> borderedSrc = cv::Mat::ones(src.rows + (rowMiddle * 2),
		src.cols + (colMiddle * 2),
		CV_32FC1);
	// Copy the original matrix.
	for (unsigned i = 0; i < src.rows; i++) {
		for (unsigned j = 0; j < src.cols; j++) {
			borderedSrc.at<float>(i + rowMiddle, j + colMiddle) = src(i, j);
		}
	}
	return borderedSrc.clone();
}

cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
	cv::Mat_<float> result = cv::Mat::zeros(src.size(), src.type());
	cv::Mat_<float> tempKernel = cv::Mat::zeros(kernel.size(), kernel.type());
	int colMiddle, rowMiddle;
	int row, col, i, j;
	double sum = 0;
	colMiddle = ((kernel.cols - 1) / 2);
	rowMiddle = ((kernel.rows - 1) / 2);

	/*flip columns*/
	for (row = 0; row < kernel.rows; row++) {
		for (col = 0; col < kernel.cols; col++) {
			if ((col != colMiddle) && (col < colMiddle))
			{
				tempKernel[row][col] = kernel[row][kernel.cols - 1 - col];
				tempKernel[row][kernel.cols - 1 - col] = kernel[row][col];
			}
			else if (col == colMiddle)
			{
				tempKernel[row][col] = kernel[row][col];
			}
			else
			{

			}
		}
	}

	/*flip rows*/
	for (col = 0; col < kernel.cols; col++) {
		for (row = 0; row < kernel.rows; row++) {
			if ((row != rowMiddle) && (row < rowMiddle))
			{
				tempKernel[row][col] = tempKernel[kernel.rows - 1 - row][col];
				tempKernel[kernel.rows - 1 - row][col] = tempKernel[row][col];
			}
			else if (row == rowMiddle)
			{
				tempKernel[row][col] = tempKernel[row][col];
			}
			else
			{

			}
		}
	}

	// 1. Border handling.
	int border_size = kernel.rows / 2;
	cv::Mat_<float> bordered_src = getBorderedImage(src, border_size, border_size);

	//cv::Mat_<float> result = cv::Mat(src);
//	float sum;
	// Go through the image
	for (unsigned i = border_size; i < src.rows + border_size; i++)
	{
		for (unsigned j = border_size; j < src.cols + border_size; j++)
		{
			// Convolve
			sum = 0.0f;
			for (unsigned r = i - border_size; r <= i + border_size; r++)
				for (unsigned c = j - border_size; c <= j + border_size; c++)
					sum += bordered_src(r, c) * tempKernel(r - i + border_size, c - j + border_size);
			result.at<float>(i - border_size, j - border_size) = sum;
		}
	}

	return result;
}



/**
 * @brief Convolution in spatial domain for 1D kernel
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution1D(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
	cv::Mat_<float> result = cv::Mat::zeros(src.size(), src.type());
	cv::Mat_<float> tempKernel = cv::Mat::zeros(kernel.size(), kernel.type());
	int colMiddle, rowMiddle;
	int row, col, i, j;
	double sum = 0;
	colMiddle = ((kernel.cols - 1) / 2);
	rowMiddle = ((kernel.rows - 1) / 2);

	/*flip columns*/
	for (row = 0; row < kernel.rows; row++) {
		for (col = 0; col < kernel.cols; col++) {
			if ((col != colMiddle) && (col < colMiddle))
			{
				tempKernel[row][col] = kernel[row][kernel.cols - 1 - col];
				tempKernel[row][kernel.cols - 1 - col] = kernel[row][col];
			}
			else if (col == colMiddle)
			{
				tempKernel[row][col] = kernel[row][col];
			}
			else
			{

			}
		}
	}

	/*flip rows*/
	for (col = 0; col < kernel.cols; col++) {
		for (row = 0; row < kernel.rows; row++) {
			if ((row != rowMiddle) && (row < rowMiddle))
			{
				tempKernel[row][col] = tempKernel[kernel.rows - 1 - row][col];
				tempKernel[kernel.rows - 1 - row][col] = tempKernel[row][col];
			}
			else if (row == rowMiddle)
			{
				tempKernel[row][col] = tempKernel[row][col];
			}
			else
			{

			}
		}
	}

	cv::Mat_<float> bordered_src = getBorderedImage(src, rowMiddle, colMiddle);

	// Go through the image
	for (unsigned i = rowMiddle; i < src.rows + rowMiddle; i++) {
		for (unsigned j = colMiddle; j < src.cols + colMiddle; j++) {
			// Convolve
			sum = 0.0f;
			for (unsigned r = i - rowMiddle; r <= i + rowMiddle; r++)
				for (unsigned c = j - colMiddle; c <= j + colMiddle; c++)
					sum += bordered_src(r, c) * tempKernel(r - i + rowMiddle, c - j + colMiddle);
			result.at<float>(i - rowMiddle, j - colMiddle) = sum;
		}
	}

	return result;
}

cv::Mat_<float> separableFilter(const cv::Mat_<float> &src, const cv::Mat_<float> &kernel)
{
	// TO DO !!

	/*performing first 1D convolution*/
	cv::Mat_<float> firstConv = spatialConvolution1D(src, kernel);

	/*transposing the horizontally convolved image*/
	cv::Mat_<float> temp = cv::Mat_<float>::zeros(src.rows, src.cols);
	cv::transpose(firstConv, temp);

	/*performing second 1D convolution*/
	cv::Mat_<float> secondConv = spatialConvolution1D(temp, kernel);
	// write transposed to output image
	cv::Mat_<float> result = cv::Mat_<float>::zeros(src.rows, src.cols);
	cv::transpose(secondConv, result);

	return result;
}


/**
 * @brief Convolution in spatial domain by integral images
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> satFilter(const cv::Mat_<float>& src, int size){

   // optional

   return src;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * @brief Performs a smoothing operation but allows the algorithm to be chosen
 * @param in Input image
 * @param size Size of filter kernel
 * @param type How is smoothing performed?
 * @returns Smoothed image
 */
cv::Mat_<float> smoothImage(const cv::Mat_<float>& in, int size, FilterMode filterMode)
{
    switch(filterMode) {
        case FM_SPATIAL_CONVOLUTION: return spatialConvolution(in, createGaussianKernel2D(size));	// 2D spatial convolution
        case FM_FREQUENCY_CONVOLUTION: return frequencyConvolution(in, createGaussianKernel2D(size));	// 2D convolution via multiplication in frequency domain
        case FM_SEPERABLE_FILTER: return separableFilter(in, createGaussianKernel1D(size));	// seperable filter
        case FM_INTEGRAL_IMAGE: return satFilter(in, size);		// integral image
        default: 
            throw std::runtime_error("Unhandled filter type!");
    }
}



}

