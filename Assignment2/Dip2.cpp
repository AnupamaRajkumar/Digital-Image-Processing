 //============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip2.h"

using namespace std;


namespace dip2 {

float distance(int currentX, int currentY, int neighborX, int neighborY)
{
	return float(sqrt(pow(currentX - neighborX, 2) + pow(currentY - neighborY, 2)));	
}

float CalculateKernelBilateralFilter(float distance, float sigma)
{
	return exp(-(pow(distance, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}
/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> getBorderedImage(const cv::Mat_<float> &src, int borderSize)
{
	// Create bordered matrix.
	cv::Mat_<float> borderedSrc = cv::Mat::zeros(src.rows + (borderSize * 2),
		src.cols + (borderSize * 2),
		CV_32FC1);
	// Copy the original matrix.
	for (unsigned i = 0; i < src.rows; i++)
	{
		for (unsigned j = 0; j < src.cols; j++)
		{
			borderedSrc.at<float>(i + borderSize, j + borderSize) = src(i, j);
		}
	}
	return borderedSrc;
}

/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */

cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
	cv::Mat_<float> result = cv::Mat::zeros(src.size(), src.type());
	cv::Mat_<float> tempKernel = cv::Mat::zeros(kernel.size(), kernel.type());
	int colMiddle, rowMiddle;
	int row, col, i, j;
	double sum = 0;
	colMiddle = ((kernel.cols-1) / 2);
	rowMiddle = ((kernel.rows-1) / 2);

	/*flip columns*/
	for( row = 0; row < kernel.rows; row++){
		for ( col = 0; col < kernel.cols; col++) {
			if ((col != colMiddle) && (col < colMiddle))
			{
				tempKernel[row][col] = kernel[row][kernel.cols - 1 - col];
				tempKernel[row][kernel.cols - 1 - col] = kernel[row][col];
			}
			else if(col == colMiddle)
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
	cv::Mat_<float> bordered_src = getBorderedImage(src, border_size);

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
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float>& src, int kSize)
{
	cv::Mat_<float> result = cv::Mat::zeros(src.size(), src.type());
	cv::Mat kernel = cv::Mat(kSize, kSize, CV_32FC1, 1.0/(kSize*kSize));
	result = spatialConvolution(src, kernel);
	return result;
}

/**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */

cv::Mat_<float> medianFilter(const cv::Mat_<float> &src, int kSize) {
	cv::Mat original = src.clone();

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			vector<float> tmp;
			for (int x = -(kSize / 2); x <= kSize / 2; x++) {
				for (int y = -(kSize / 2); y <= kSize / 2; y++) {
					if (not((i - x) < 0 || (i - x) >= src.rows || (j - y) < 0 || (j - y) >= src.cols)) {
						tmp.push_back(src.at<float>(i - x, j - y));
					}
					else {
						tmp.push_back(0);
					}
				}
			}
			sort(tmp.begin(), tmp.end());
			float value;
			if (tmp.size() % 2 == 0) {
				value = (tmp[tmp.size() / 2 - 1] + tmp[tmp.size() / 2]) / 2;
			}
			else {
				value = tmp[tmp.size() / 2];
			}
			original.at<float>(i, j) = value;
		}
	}
	return original.clone();
}

/**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */

cv::Mat_<float> bilateralFilter(const cv::Mat_<float>& src, int kSize, float sigma_spatial, float sigma_radiometric)
{
	cv::Mat_<float> result = cv::Mat::ones(src.size(), src.type());
	cv::Mat_<float> spatWts, radioWts;
	float imgFilt = 0, accWt = 0;
	float hSpat = 1, hRad = 1, wtCombined = 1;
	int neighbourX = 0, neighbourY = 0;
	int half = kSize / 2;

	for (int i = half; i < src.rows - half; i++) {
		for (int j = half; j < src.cols - half; j++) {
			for (int row = 0; row < kSize; row++)
			{
				for (int col = 0; col < kSize; col++)
				{
					neighbourX = i - half + row;
					neighbourY = j - half + col;
					if (not(neighbourX < 0 or neighbourX > src.rows or neighbourY < 0 or neighbourY > src.cols)) {
						hSpat = CalculateKernelBilateralFilter(distance(i, j, neighbourX, neighbourY), sigma_spatial);
						hRad = CalculateKernelBilateralFilter(src.at<float>(neighbourX, neighbourY) - src.at<float>(i, j), sigma_radiometric);
						wtCombined = hSpat * hRad;
						imgFilt = imgFilt + (src[neighbourX][neighbourY] * wtCombined);
						accWt = accWt + wtCombined;
					}

				}
			}
			imgFilt = imgFilt / accWt;
			result[i][j] = imgFilt;
			accWt = 0;
			imgFilt = 0;
		}
	}
	return result;
}



/**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
cv::Mat_<float> nlmFilter(const cv::Mat_<float>& src, int searchSize, double sigma)
{
    return src.clone();
}

/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
{
	switch (noiseType) {
	case NOISE_TYPE_1:
		cout << "\nFor Shot noise, median filter has better filtering";
		return NR_MEDIAN_FILTER;
		break;
	case NOISE_TYPE_2:
		cout << "\nFor Gaussian noise, moving average filter has better filtering";
		return NR_MOVING_AVERAGE_FILTER;
		break;
	default:
		throw std::runtime_error("Unhandled noise type!");
	}
}



cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm)
{

    // for each combination find reasonable filter parameters
	switch (noiseReductionAlgorithm) {
	case dip2::NR_MOVING_AVERAGE_FILTER:
		switch (noiseType) {
		case NOISE_TYPE_1:
			return dip2::averageFilter(src, 5);
		case NOISE_TYPE_2:
			return dip2::averageFilter(src, 3);
		default:
			throw std::runtime_error("Unhandled noise type!");
		}
	case dip2::NR_MEDIAN_FILTER:
		switch (noiseType) {
		case NOISE_TYPE_1:
			return dip2::medianFilter(src, 5);
		case NOISE_TYPE_2:
			return dip2::medianFilter(src, 5);
		default:
			throw std::runtime_error("Unhandled noise type!");
		}
	case dip2::NR_BILATERAL_FILTER:
		switch (noiseType) {
		case NOISE_TYPE_1:
			return dip2::bilateralFilter(src, 3, 500.0f, 700.0f);
		case NOISE_TYPE_2:
			return dip2::bilateralFilter(src, 3, 500.0f, 700.0f);
		default:
			throw std::runtime_error("Unhandled noise type!");
		}
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}


// Helpers, don't mind these

const char *noiseTypeNames[NUM_NOISE_TYPES] = {
    "NOISE_TYPE_1",
    "NOISE_TYPE_2",
};

const char *noiseReductionAlgorithmNames[NUM_FILTERS] = {
    "NR_MOVING_AVERAGE_FILTER",
    "NR_MEDIAN_FILTER",
    "NR_BILATERAL_FILTER",
};


}
