//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip5.h"

using namespace std;
namespace dip5 {


	/**
	 * @brief Creates a zero-borded image.
	 */
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

	/**
	 * @brief Convolution in spatial domain.
	 * @details Performs spatial convolution of image and filter kernel.
	 * @params src Input image
	 * @params kernel Filter kernel
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

/**
* @brief Generates gaussian filter kernel of given size
* @param kSize Kernel size (used to calculate standard deviation)
* @returns The generated filter kernel
*/
cv::Mat_<float> createGaussianKernel1D(float sigma)
{
	float mean, sum;
	unsigned kSize = getOddKernelSizeForSigma(sigma);

	/*normalizing the values*/
	sum = 0;

	/*1-D array of size = kSize*/
	cv::Mat_<float> result = cv::Mat_<float>::zeros(1, kSize);

	/*calculate mean*/
	if ((kSize % 2) == 0)
	{
		mean = kSize / 2.;
	}
	else
	{
		mean = (kSize + 1) / 2.;
	}
	/*calculate gaussian blur*/
	for (int i = 0; i < kSize; i++)
	{
		float denominator, power;
		denominator = sqrt(2 * CV_PI)*sigma;
		power = pow(((i + 1) - mean), 2) / (2 * pow(sigma, 2));
		result.at<float>(i) = exp(-power) / denominator;
		sum = sum + result.at<float>(i);
	}
	/*normalize the distribution*/
	result = result / sum;
	return result;
}
        

/**
* @brief Convolution in spatial domain by seperable filters
* @param src Input image
* @param size Size of filter kernel
* @returns Convolution result
*/
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernelX, const cv::Mat_<float>& kernelY)
{
    // But do mind that this one gets two different kernels for horizontal and vertical convolutions.
	int rows, cols;
	cv::Mat_<float> temp, temp1;
	cv::Mat_<float> result, resultF;
	temp1 = cv::Mat::zeros(src.size(), src.type());
	temp = cv::Mat::zeros(src.cols, src.rows, src.type());
	result = cv::Mat::zeros(temp.size(), temp.type());
	resultF = cv::Mat::zeros(src.size(), src.type());
	rows = 0;
	cols = 0;
	/*performing first 1D convolution*/
	//std::cout << "KernelX:" << kernelX << "\n";
	temp1 = spatialConvolution1D(src, kernelX);

	//cout << "temp1:" << temp1<< "\n";
	/*transposing temp1 */
	for (int i = 0; i < temp1.rows; i++)
	{
		for (int j = 0; j < temp1.cols; j++)
		{
			temp.at<float>(j, i) = temp1.at<float>(i, j);
		}
	}
	//cout << "Transposed image:" << temp << "\n";
	/*performing second 1D convolution*/
	result = spatialConvolution1D(temp, kernelY);

	/*transposing temp */
	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			resultF.at<float>(j, i) = result.at<float>(i, j);
		}
	}
	//cout << "ResultF:" << resultF << "\n";
    return resultF;
}

    
/**
 * @brief Creates kernel representing fst derivative of a Gaussian kernel (1-dimensional)
 * @param sigma standard deviation of the Gaussian kernel
 * @returns the calculated kernel
 */
cv::Mat_<float> createFstDevKernel1D(float sigma) 
{
    int kSize = getOddKernelSizeForSigma(sigma);
	float factor1 = 0., factor2 = 0.;
	cv::Mat_<float> result = cv::Mat_<float>::zeros(1, kSize);

	for (int i = -kSize/2; i <= kSize/2; i++) {
		factor1 = - (i) / (2. * CV_PI *pow(sigma, 4));
		float temp = pow((i), 2) / (2. * pow(sigma, 2));
		factor2 = exp(-temp);
		result.at<float>(0, i+kSize/2) = factor1*factor2;
	}
    return result;
}


/**
 * @brief Calculates the directional gradients through convolution
 * @param img The input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param gradX Matrix through which to return the x component of the directional gradients
 * @param gradY Matrix through which to return the y component of the directional gradients
 */
void calculateDirectionalGradients(const cv::Mat_<float>& img, float sigmaGrad,
                            cv::Mat_<float>& gradX, cv::Mat_<float>& gradY)
{
	gradX.create(img.rows, img.cols);
	gradY.create(img.rows, img.cols);
	cv::Mat_<float> kernelGauss		= createGaussianKernel1D(sigmaGrad);
	cv::Mat_<float> kernelGaussDev	= createFstDevKernel1D(sigmaGrad);
	gradX = separableFilter(img, kernelGaussDev, kernelGauss);
	gradY = separableFilter(img, kernelGauss, kernelGaussDev);
}

/**
 * @brief Calculates the structure tensors (per pixel)
 * @param gradX The x component of the directional gradients
 * @param gradY The y component of the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation".
 * @param A00 Matrix through which to return the A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 Matrix through which to return the A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 Matrix through which to return the A_{1,1} elements of the structure tensor of each pixel.
 */
void calculateStructureTensor(const cv::Mat_<float>& gradX, const cv::Mat_<float>& gradY, float sigmaNeighborhood,
                            cv::Mat_<float>& A00, cv::Mat_<float>& A01, cv::Mat_<float>& A11)
{
	A00.create(gradX.rows, gradX.cols);
	A01.create(gradX.rows, gradX.cols);
	A11.create(gradX.rows, gradX.cols);
	cv::Mat_<float> kernelGauss = createGaussianKernel1D(sigmaNeighborhood);
	cv::Mat_<float> gxgx, gxgy, gygy;

	gxgx	=	gradX.mul(gradX);
	A00		=	separableFilter(gxgx, kernelGauss, kernelGauss);
	gygy	=	gradY.mul(gradY);
	A11		=	separableFilter(gygy, kernelGauss, kernelGauss);
	gxgy	=	gradX.mul(gradY);
	A01		=	separableFilter(gxgy, kernelGauss, kernelGauss);
}

/**
 * @brief Calculates the feature point weight and isotropy from the structure tensors.
 * @param A00 The A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 The A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 The A_{1,1} elements of the structure tensor of each pixel.
 * @param weight Matrix through which to return the weights of each pixel.
 * @param isotropy Matrix through which to return the isotropy of each pixel.
 */
void calculateFoerstnerWeightIsotropy(const cv::Mat_<float>& A00, const cv::Mat_<float>& A01, const cv::Mat_<float>& A11,
                                    cv::Mat_<float>& weight, cv::Mat_<float>& isotropy)
{
    weight.create(A00.rows, A00.cols);
    isotropy.create(A00.rows, A00.cols);
	cv::Mat_<float> trace		=	A00 + A11;
	cv::Mat_<float> determinant =   A00.mul(A11) - A01.mul(A01);

	for (int i = 0; i < A00.rows; i++)
	{
		for (int j = 0; j < A00.cols; j++)
		{
			weight.at<float>(i, j)		=	determinant.at<float>(i, j) / max(trace.at<float>(i, j), 1e-8f);
			isotropy.at<float>(i, j)	=	4*determinant.at<float>(i, j) / max(pow(trace.at<float>(i, j),2), 1e-8f);
		}
	}
	
}


/**
 * @brief Finds Foerstner interest points in an image and returns their location.
 * @param img The greyscale input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation" of the structure tensor.
 * @param minWeight Threshold on the weight as a fraction of the mean of all locally maximal weights.
 * @param minIsotropy Threshold on the isotropy of interest points.
 * @returns List of interest point locations.
 */
std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float>& img, float sigmaGrad, float sigmaNeighborhood, float minWeight, float minIsotropy)
{
	
	cv::Mat_<float> gradX, gradY, A00, A11, A01, weight, isotropy;
	vector<cv::Vec2i> keyPoints;
	float meanWeight;

	/*compute directional gradient*/
	calculateDirectionalGradients(img, sigmaGrad, gradX, gradY);
	/*compute structure tensor*/
	calculateStructureTensor(gradX, gradY, sigmaNeighborhood, A00, A01, A11);
	/*calculate weight and isotropies*/
	calculateFoerstnerWeightIsotropy(A00, A01, A11, weight, isotropy);

	cv::Scalar myMatMean	= mean(weight);
	meanWeight				= myMatMean.val[0];

	for (int c = 0; c < weight.cols; c++)
	{
		for (int r = 0; r < weight.rows; r++)
		{
			if ((weight.at<float>(r, c) > (minWeight*meanWeight)) &&
				(isotropy.at<float>(r, c) > minIsotropy) &&
				isLocalMaximum(weight, c, r)){
				
				cv::Vec2i kp = cv::Vec2i(c, r);
				keyPoints.emplace_back(kp);
			}
		}
	}
    return keyPoints;
}


/* *****************************
  GIVEN FUNCTIONS
***************************** */


// Use this to compute kernel sizes so that the unit tests can simply hard checks for correctness.
unsigned getOddKernelSizeForSigma(float sigma)
{
    unsigned kSize = (unsigned) std::ceil(5.0f * sigma) | 1;
    if (kSize < 3) kSize = 3;
    return kSize;
}

bool isLocalMaximum(const cv::Mat_<float>& weight, int x, int y)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            int x_ = std::min(std::max(x+j, 0), weight.cols-1);
            int y_ = std::min(std::max(y+i, 0), weight.rows-1);
            if (weight(y_, x_) > weight(y, x))
                return false;
        }
    return true;
}

}
