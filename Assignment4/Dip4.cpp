//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip4.h"

namespace dip4 {

using namespace std::complex_literals;

/*

===== std::complex cheat sheet ===== 

Initialization:

std::complex<float> a(1.0f, 2.0f);
std::complex<float> a = 1.0f + 2.0if;

Common Operations:

std::complex<float> a, b, c;

a = b + c;
a = b - c;
a = b * c;
a = b / c;

std::sin, std::cos, std::tan, std::sqrt, std::pow, std::exp, .... all work as expected

Access & Specific Operations:

std::complex<float> a = ...;

float real = a.real();
float imag = a.imag();
float phase = std::arg(a);
float magnitude = std::abs(a);
float squared_magnitude = std::norm(a);

std::complex<float> complex_conjugate_a = std::conj(a);

*/
    

/**
 * @brief Computes the complex valued forward DFT of a real valued input
 * @param input real valued input
 * @return Complex valued output, each pixel storing real and imaginary parts
 */
cv::Mat_<std::complex<float>> DFTReal2Complex(const cv::Mat_<float>& input)
{
	cv::Mat_<float> padded = input.clone();
	cv::Mat_<float> planes[] = { padded, cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat_<std::complex<float>> complexInput;
	merge(planes, 2, complexInput);
	dft(complexInput, complexInput);
	return complexInput.clone();
}

    
/**
 * @brief Computes the real valued inverse DFT of a complex valued input
 * @param input Complex valued input, each pixel storing real and imaginary parts
 * @return Real valued output
 */
cv::Mat_<float> IDFTComplex2Real(const cv::Mat_<std::complex<float>>& input)
{
	cv::Mat_<float> result = cv::Mat::zeros(input.size(), input.type());
	cv::idft(input, result, cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);		 	//
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
 * @brief Computes the thresholded inverse filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return The inverse filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeInverseFilter(const cv::Mat_<std::complex<float>>& input, const float eps)
{
	float maxMagnitude, magnitude;
	std::complex < float> num;
	num = 1;
	maxMagnitude = std::abs(input.at<std::complex<float>>(0, 0));
	cv::Mat_<std::complex<float>> result = cv::Mat::zeros(input.size(), input.type());
	/*finding value of clipping factor*/
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			magnitude = std::abs(input.at<std::complex<float>>(i, j));
			if (maxMagnitude < magnitude)
			{
				maxMagnitude = magnitude;
			}
		}
	}
	maxMagnitude = eps * maxMagnitude;
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			if (std::abs(input.at<std::complex<float>>(i,j)) >= maxMagnitude)
			{
				result.at<std::complex<float>>(i, j) = num / input.at<std::complex<float>>(i, j);
			}
			else if (std::abs(input.at<std::complex<float>>(i, j)) < maxMagnitude)
			{
				result.at<std::complex<float>>(i, j) = 1 / maxMagnitude;
			}
			else
			{
				continue;
			}
		}

	}
    return result;
}


/**
 * @brief Applies a filter (in frequency domain)
 * @param input Image in frequency domain (complex valued)
 * @param filter Filter in frequency domain (complex valued), same size as input
 * @return The filtered image, complex valued, in frequency domain
 */
cv::Mat_<std::complex<float>> applyFilter(const cv::Mat_<std::complex<float>>& input, const cv::Mat_<std::complex<float>>& filter)
{
	//std::cout << "In applyFilter";
	cv::Mat_<std::complex<float>> result = cv::Mat::zeros(input.size(), input.type());
	cv::mulSpectrums(input, filter, result, false, false);
    return result;
}


/**
 * @brief Function applies the inverse filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return Restorated output image
 */
cv::Mat_<float> inverseFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, const float eps)
{
	cv::Mat_<float> result = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<float> tempFilter = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<float> tempFilterShift = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<std::complex<float>> BlurFilter = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<std::complex<float>> BlurImage = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<std::complex<float>> RestoredImage = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<std::complex<float>> FilterSpectrum = cv::Mat::zeros(degraded.size(), degraded.type());
	/*zero-padding the filter to make it same size as the degraded input image*/
	for (int i = 0; i < filter.rows; i++)
	{
		for (int j = 0; j < filter.cols; j++)
		{
			tempFilter.at<float>(i, j) = filter.at<float>(i, j);
		}
	}
	/*circular shifting the filter kernel*/
	tempFilterShift = circShift(tempFilter, -filter.rows/2, -filter.rows/2);
	/*Calculating complex spectra of degraded signal*/
	BlurImage = DFTReal2Complex(degraded);
	/*Calculating complex spectra of filter signal*/
	FilterSpectrum = DFTReal2Complex(tempFilterShift);
	/*Calculating inverse filter*/
	BlurFilter = computeInverseFilter(FilterSpectrum, eps);
	/*Apply filter - multiply BlurImage with degraded image to restore the original image*/
	RestoredImage = applyFilter(BlurImage, BlurFilter);
	/*Restored image in time domain*/
	result = IDFTComplex2Real(RestoredImage);
    return result;
}


/**
 * @brief Computes the Wiener filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param snr Signal to noise ratio
 * @return The wiener filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeWienerFilter(const cv::Mat_<std::complex<float>>& input, const float snr)
{
	cv::Mat_<std::complex<float>> result = cv::Mat::zeros(input.size(), input.type());
	std::complex<float> conjugate;
	float magnitude, inverseSNR;
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			conjugate = std::conj(input.at< std::complex<float>>(i, j));
			magnitude = std::pow(std::abs(input.at< std::complex<float>>(i, j)), 2);
			inverseSNR = 1 / std::pow(snr, 2);
			result.at< std::complex<float>>(i, j) = conjugate / (magnitude + inverseSNR);
		}
	}
    return result;
}

/**
 * @brief Function applies the wiener filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param snr Signal to noise ratio of the input image
 * @return Restorated output image
 */
cv::Mat_<float> wienerFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, float snr)
{
	cv::Mat_<float> result = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<float> tempFilter = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<float> tempFilterShift = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<std::complex<float>> BlurImage = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<std::complex<float>> BlurFilter = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<std::complex<float>> FilterSpectrum = cv::Mat::zeros(degraded.size(), degraded.type());
	cv::Mat_<std::complex<float>> RestoredImage = cv::Mat::zeros(degraded.size(), degraded.type());
	/*zero-padding the filter to make it same size as the degraded input image*/
	for (int i = 0; i < filter.rows; i++)
	{
		for (int j = 0; j < filter.cols; j++)
		{
			tempFilter.at<float>(i, j) = filter.at<float>(i, j);
		}
	}
	/*circular shifting the filter kernel*/
	tempFilterShift = circShift(tempFilter, -filter.rows / 2, -filter.rows / 2);
	/*Calculating complex spectra of degraded signal*/
	BlurImage = DFTReal2Complex(degraded);
	/*Calculating complex spectra of filter signal*/
	FilterSpectrum = DFTReal2Complex(tempFilterShift);
	/*calculating weiner filter*/
	BlurFilter = computeWienerFilter(FilterSpectrum, snr);
	/*Apply filter - multiply BlurImage with degraded image to restore the original image*/
	RestoredImage = applyFilter(BlurImage, BlurFilter);
	/*Restored image in time domain*/
	result = IDFTComplex2Real(RestoredImage);
	//std::cout << "result:" << result << "\n";
    return result;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * function degrades the given image with gaussian blur and additive gaussian noise
 * @param img Input image
 * @param degradedImg Degraded output image
 * @param filterDev Standard deviation of kernel for gaussian blur
 * @param snr Signal to noise ratio for additive gaussian noise
 * @return The used gaussian kernel
 */
cv::Mat_<float> degradeImage(const cv::Mat_<float>& img, cv::Mat_<float>& degradedImg, float filterDev, float snr)
{

    int kSize = round(filterDev*3)*2 - 1;
   
    cv::Mat gaussKernel = cv::getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    cv::Mat imgs = img.clone();
    cv::dft( imgs, imgs, CV_DXT_FORWARD, img.rows);
    cv::Mat kernels = cv::Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++) 
        for(int j=0; j<kSize; j++) 
            kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	cv::dft( kernels, kernels, CV_DXT_FORWARD );
	cv::mulSpectrums( imgs, kernels, imgs, 0 );
	cv::dft( imgs, degradedImg, CV_DXT_INV_SCALE, img.rows );
	
    cv::Mat mean, stddev;
    cv::meanStdDev(img, mean, stddev);

    cv::Mat noise = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    cv::randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    cv::threshold(degradedImg, degradedImg, 255, 255, CV_THRESH_TRUNC);
    cv::threshold(degradedImg, degradedImg, 0, 0, CV_THRESH_TOZERO);

    return gaussKernel;
}


}
