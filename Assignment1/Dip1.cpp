//============================================================================
// Name        : Dip1.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================


#include "Dip1.h"

#include <stdexcept>


namespace dip1 {


/**
 * @brief function that performs some kind of (simple) image processing
 * @param img input image
 * @returns output image
 */
cv::Mat doSomethingThatMyTutorIsGonnaLike(const cv::Mat& img) {
    /*chnaging contrast and brightenss of the image*/
	/*outputimg(i,j) = alpha*inputimg(i,j)+beta*/
	double alpha;
	double beta;

	alpha = 5;
	beta = 100;
	cv::Mat output_img = cv::Mat::zeros(img.size(), img.type());
	/*For each RGB channel of the image*/
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			for (int c = 0; c < img.channels(); c++) {
				output_img.at<cv::Vec3b>(y, x)[c] =
					cv::saturate_cast<uchar>(alpha*img.at<cv::Vec3b>(y, x)[c] + beta);
			}
		}
	}
    return output_img;
}





/******************************
      GIVEN FUNCTIONS
 ******************************/

/**
 * @brief function loads input image, calls processing function, and saves result
 * @param fname path to input image
 */
void run(const std::string &filename) {

    // window names
    std::string win1 = "Original image";
    std::string win2 = "Result";

    // some images
    cv::Mat inputImage, outputImage;

    // load image
    std::cout << "loading image" << std::endl;
    inputImage = cv::imread(filename);
    std::cout << "done" << std::endl;
    
    // check if image can be loaded
    if (!inputImage.data)
        throw std::runtime_error(std::string("ERROR: Cannot read file ") + filename);

    // show input image
    cv::namedWindow(win1.c_str());
    cv::imshow(win1.c_str(), inputImage);
    
    // do something (reasonable!)
    outputImage = doSomethingThatMyTutorIsGonnaLike(inputImage);
    
    // show result
    cv::namedWindow(win2.c_str());
    cv::imshow(win2.c_str(), outputImage);
    
    // save result
    cv::imwrite("result.jpg", outputImage);
    
    // wait a bit
    cv::waitKey(0);
}


}
