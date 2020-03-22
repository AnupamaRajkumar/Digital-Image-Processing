//============================================================================
// Name        : unit_test.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Dip3.h"

#include <opencv2/opencv.hpp>

#include <iostream>

#if 0

extern const std::uint64_t data_usmImage[];
extern const std::size_t data_usmImage_size;

extern const std::uint64_t data_inputImage[];
extern const std::size_t data_inputImage_size;


using namespace std;
using namespace cv;
using namespace dip3;

union FloatInt {
    uint32_t i;
    float f;
};

inline bool fastmathIsFinite(float f)
{
    FloatInt f2i;
    f2i.f = f;
    return ((f2i.i >> 23) & 0xFF) != 0xFF;
}


bool matrixIsFinite(const Mat_<float> &mat) {
    
    for (unsigned r = 0; r < mat.rows; r++)
        for (unsigned c = 0; c < mat.cols; c++)
            if (!fastmathIsFinite(mat(r, c)))
                return false;
    
    return true;
}

bool test_createGaussianKernel1D(void)
{
   Mat k = createGaussianKernel1D(11);

   if (k.rows != 1){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Wrong size!" << endl;
      return false;
   }
   if (k.cols != 11){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Wrong size!" << endl;
      return false;
   }

    if (!matrixIsFinite(k)){
        cout << "ERROR: Dip3::createGaussianKernel1D(): Inf/nan values in result!" << endl;
        return false;
    }
   
   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Sum of all kernel elements is not one!" << endl;
      return false;
   }
   if (sum(k >= k.at<float>(0,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Seems like kernel is not centered!" << endl;
      return false;
   }
   cout << "Message: Dip3::createGaussianKernel1D() seems to be correct" << endl;
    return true;
}

bool test_createGaussianKernel2D(void)
{
   Mat k = createGaussianKernel2D(11);
   
   if (k.rows != 11){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Wrong size!" << endl;
      return false;
   }
   if (k.cols != 11){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Wrong size!" << endl;
      return false;
   }

    if (!matrixIsFinite(k)){
        cout << "ERROR: Dip3::createGaussianKernel2D(): Inf/nan values in result!" << endl;
        return false;
    }

   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::test_createGaussianKernel2D(): Sum of all kernel elements is not one!" << endl;
      return false;
   }
   if (sum(k >= k.at<float>(5,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::test_createGaussianKernel2D(): Seems like kernel is not centered!" << endl;
      return false;
   }
   cout << "Message: Dip3::test_createGaussianKernel2D() seems to be correct" << endl;
    return true;
}

bool test_circShift(void)
{   
    {
        Mat_<float> in(3,3);
        in.setTo(0.0f);
        in.at<float>(0,0) = 1;
        in.at<float>(0,1) = 2;
        in.at<float>(1,0) = 3;
        in.at<float>(1,1) = 4;
        Mat_<float> ref(3,3);
        ref.setTo(0.0f);
        ref.at<float>(0,0) = 4;
        ref.at<float>(0,2) = 3;
        ref.at<float>(2,0) = 2;
        ref.at<float>(2,2) = 1;
        
        Mat_<float> res = circShift(in, -1, -1);
        if (!matrixIsFinite(res)){
            cout << "ERROR: Dip3::circShift(): Inf/nan values in result!" << endl;
            return false;
        }

        if (sum((res == ref)).val[0]/255 != 9){
            cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
            return false;
        }
    }
    {
        cv::Mat_<float> in(30, 40);
        cv::randn(in, cv::Scalar(0.0f), cv::Scalar(1.0f));
        
        cv::Mat_<float> tmp;
        tmp = circShift(in, -5, -10);
        tmp = circShift(tmp, 10, -10);
        tmp = circShift(tmp, -5, 20);

        if (!matrixIsFinite(tmp)){
            cout << "ERROR: Dip3::circShift(): Inf/nan values in result!" << endl;
            return false;
        }

        if (sum(tmp != in).val[0] != 0){
            cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
            return false;
        }
    }
	cout << "Message: Dip3::circShift() seems to be correct" << endl;
    return true;
}

bool test_frequencyConvolution(void)
{   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat_<float> output = frequencyConvolution(input, kernel);
   
    if (!matrixIsFinite(output)){
        cout << "ERROR: Dip3::frequencyConvolution(): Inf/nan values in result!" << endl;
        return false;
    }


   if ( (sum(output < 0).val[0] > 0) or (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains too large/small values!" << endl;
      return false;
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains wrong values!" << endl;
            return false;
         }
      }
   }
   cout << "Message: Dip3::frequencyConvolution() seems to be correct" << endl;
    return true;
}

bool test_separableConvolution(void)
{   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(1,3, CV_32FC1, 1./3.);

   Mat_<float> output = separableFilter(input, kernel);
   
    if (!matrixIsFinite(output)){
        cout << "ERROR: Dip3::separableConvolution(): Inf/nan values in result!" << endl;
        return false;
    }


   if ( (sum(output < 0).val[0] > 0) or (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::separableFilter(): Convolution result contains too large/small values!" << endl;
      return false;
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};

   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip3::separableFilter(): Convolution result contains wrong values!" << endl;
            return false;
         }
      }
   }
   cout << "Message: Dip3::separableFilter() seems to be correct" << endl;
    return true;
}

bool test_usm(void)
{   
    
    for (unsigned filterType = 0; filterType < 3; filterType++) {
        cv::Mat img = cv::imdecode(cv::_InputArray((const char *)data_inputImage, data_inputImage_size), 0);
        img.convertTo(img, CV_32FC1);
        cv::Mat expectedOutput = cv::imdecode(cv::_InputArray((const char *)data_usmImage, data_usmImage_size), 0);
        expectedOutput.convertTo(expectedOutput, CV_32FC1);

        // distort image with gaussian blur
        int size = 9;
        GaussianBlur(img, img, Size(floor(size/2)*2+1,floor(size/2)*2+1), size/5., size/5.);

        cv::Mat_<float> usm_img = dip3::usm(img, (dip3::FilterMode) filterType, 9, 1.0f, 1.5f);
	    cv::threshold(usm_img, usm_img, 0, 0, cv::THRESH_TOZERO);
	    cv::threshold(usm_img, usm_img, 255, 255, cv::THRESH_TRUNC);

        if ((usm_img.rows != expectedOutput.rows) || (usm_img.cols != expectedOutput.cols)) {
            cout << "ERROR: Dip3::usm() with filter mode " << filterModeNames[filterType] << ": Output has wrong size!" << endl;
            //return false;
        }

        if (!matrixIsFinite(usm_img)){
            cout << "ERROR: Dip3::usm() with filter mode " << filterModeNames[filterType] << ": Inf/nan values in result!" << endl;
            //return false;
        }

        float mse = 0.0f;
        for (unsigned y = 10; y < expectedOutput.rows-10; y++) {
            float subsum = 0.0f;
            for (unsigned x = 10; x < expectedOutput.cols-10; x++) {
                float d = usm_img(y, x) - expectedOutput.at<float>(y, x);
                subsum += d*d;
            }
            mse += subsum;
        }
        mse /= (expectedOutput.rows-20)*(expectedOutput.cols-20);

        if (mse > 1.5f) {
            cout << "ERROR: Dip3::usm() with filter mode " << filterModeNames[filterType] << ": Difference to expected output too high!" << endl;
            cout << "  expected mse < 1.5, but got " << mse << endl;
            //return false;
        }
    }

    cout << "Message: Dip3::usm() seems to be correct" << endl;
    return true;
}

int main(int argc, char** argv) {

    bool ok = true;

    ok &= test_createGaussianKernel1D();	//done
    ok &= test_createGaussianKernel2D();	//done
    ok &= test_circShift();					//done
    ok &= test_frequencyConvolution();		//done
    ok &= test_separableConvolution();		//done
    ok &= test_usm();

    if (!ok)
        return -1;
    else
    	return 0;
} 



const std::uint64_t data_inputImage[] = {
   0xa1a0a0d474e5089ul, 0x524448490d000000ul, 0x78000000a0000000ul, 0xa6c9500000000008ul, 0x43436923010000b3ul, 0x6f72702043434950ul, 0x91280000656c6966ul, 0x861450c34ab1909dul, 0x838a20ea52a954bful, 0xa5c998b82d706608ul, 0xa7560ac62141042aul, 0x14a490c498b14934ul, 0x820e987d137c0dful, 0xa37fece0a06f80beul, 0xe3ff870bc598383ul, 0x12765a17bdfffce7ul, 
   0xb85559a42ee2e5a6ul, 0xde97b2bc39787f7eul, 0x5841ba6d2acdb0e8ul, 0xbe7cf1a13bcf7de6ul, 0xb9e6af19e97d1962ul, 0x573a50cb8a3b4f3ful, 0x9d8bed60545e6165ul, 0xf03b7eb1561b9559ul, 0x48b34a3b620fc50ful, 0xd9b0c8d289de24fcul, 0xdb9a78fe1a64d3f5ul, 0x5b55f4dce2ece374ul, 0x2988cd878a731cb8ul, 0x239d4cd27a2a1213ul, 0x94f701052ea4f61cul, 0x2a6699bd5884d284ul, 
   0x3440e5c9ca5446eul, 0x79e759b790d36e91ul, 0x70932f2263c9194aul, 0xefdff987934f2a47ul, 0xe798dad37ab38fb5ul, 0x3c6b5505add41141ul, 0x33dac2195847f786ul, 0xadbf7f96b21bae74ul, 0x2fc6f9fe67a9c661ul, 0x6eaefd1d4f5034bdul, 0x7359487009000000ul, 0xd4350000d4350000ul, 0x20000008e5655e01ul, 0x6dda785441444900ul, 0x39df75e72593d77cul, 0x7771c9df39bee75ful, 
   0x112258bbb16136ul, 0xb2b2ca28a4880601ul, 0x5f965c96cb83f254ul, 0x55d1f952e03ff65dul, 0x3255483f2595707eul, 0x48942a44ba2adb2dul, 0xd8b177632200948aul, 0x1cdcee6793b3b3bcul, 0xcf5f0edc3f1f773aul, 0xa7dbdc277bd83062ul, 0xcbe0df39df9dfc4ful, 0xe9b3e10088008000ul, 0xfe1004004040817ul, 0xc08800008fa7110ul, 0xe980021dfc621910ul, 0x87fa291e4a0fc26bul, 
   0xa2f0c1dfa77fa28ul, 0xa11397e2880065f1ul, 0x7f4d0210021dbdc8ul, 0xdf8c2dfc52bb909ful, 0x404cc918f859010dul, 0x2122b2a74bf148ul, 0x1062867d144a2f4dul, 0xebd3748912e89977ul, 0x77d3789104309c89ul, 0x457bf8c40080b58dul, 0x4fe1848a950c440ul, 0x537d4499aa700909ul, 0x43315394890f4595ul, 0x41f4df2f24bc3121ul, 0x424d431aee8bc50cul, 0x483722c47558a578ul, 
   0x91a6ca67e1088654ul, 0x12cfc5f2f3d7927dul, 0xb7f8906a163744e3ul, 0x226e8c2477792112ul, 0x145380099a9c40f4ul, 0x4e07dc07148c0ac6ul, 0x899ce40301008d6ful, 0x7a9c13725130b7ul, 0xa8620878de040210ul, 0x7123356339d8c032ul, 0x2054381a17091231ul, 0x705c7c20e8702861ul, 0x17b167a20c22247aul, 0x3024a1375341af4eul, 0x88c6a18810224d63ul, 0xc24f452fa9a46762ul, 
   0x522caad082c48da0ul, 0xc648c55745362c62ul, 0x12301d88c0f855e9ul, 0x4004076e72273225ul, 0x325b1c4d0d80d384ul, 0x44eb8a30fa288f0aul, 0xbcde43bac5d7a15dul, 0x1057d14ff54dcbb1ul, 0x20037b894c61f120ul, 0x7753ff5e2ff53061ul, 0x524c3cf628295314ul, 0x84cf30c039a8610bul, 0xb123e67fd652c1c1ul, 0x85b44e1653069c6ful, 0x422b7727e049ffd1ul, 0xc12988457786113aul, 
   0x310c14302894aca3ul, 0x184e071a6f9828ceul, 0xbbc074728e0f5365ul, 0xe4616312015711ful, 0x38dccf0b0bcd2a76ul, 0xccd3880289c0e1f2ul, 0x7ae50a0cf5321532ul, 0xc472a67434139067ul, 0x866fe24555209489ul, 0x558a45646b33e62aul, 0x58be171880acf489ul, 0xd8987fec4b338a2eul, 0x7721656c941001dul, 0x4be3f1d556fa1043ul, 0x2e8525f524e36713ul, 0xe58be26af63002faul, 
   0xa795545b53906a17ul, 0x1b1222a23d5c594ful, 0x3b27e8feef77fcfbul, 0x79ef3f32cf40e695ul, 0xf88ff92de570dbe1ul, 0xf440e04c5da7131aul, 0xa329d202538c5618ul, 0x65ba2b912c06cef2ul, 0x3adeffffbf280040ul, 0xbda68e2a164a5bful, 0x7b915caf2a9bae48ul, 0xf888f58512243c06ul, 0x8821530c0000dacul, 0xf7c5a6b8a5c66180ul, 0xc6e100a4c3e71044ul, 0xc807993c14bf8777ul, 
   0x62921dc1dce426a3ul, 0x8082888f1d4d11dul, 0xa994ef1207388288ul, 0x3a78354f2c5c8952ul, 0xc4889151af146049ul, 0xedf16ee4987958c2ul, 0x8e73cb26379bfb0ful, 0xd13f99f5810b5423ul, 0x6b627372e27f8f88ul, 0xd7175c932de89e4eul, 0x77c4d93a59928806ul, 0x7c6125e72010ba95ul, 0x67617cff0ff66010ul, 0x68144175bfd03ce8ul, 0x21c18d9e514263e7ul, 0x34e22014154d8625ul, 
   0xc854c28fc8b867c1ul, 0xc2f611b24f8428d2ul, 0xd9442108a954b644ul, 0xfaef02b1aa2a9673ul, 0x59155f7d07dbfd85ul, 0x2c42a718d7c3cb35ul, 0x9aa7aef887af2531ul, 0x10d9ec78d4df2b8cul, 0xa632d24c5c7ecb0eul, 0x8caf4beaee784995ul, 0x48c29bdc14d9c915ul, 0xf022e1e9c32d15acul, 0x12a277b4325c4bbaul, 0x45e8b4574a186716ul, 0x8d7f8d156181e52cul, 0xc28250cd0a76b22ul, 
   0x16137df7a303f41ul, 0xd671a5d03f16ee7eul, 0x2518a646d6f096cul, 0x7e117429f15e404ul, 0xf3a71d58b6a017e1ul, 0x7803e5c6026dc620ul, 0xc7fb15fe337e131cul, 0x404abd4bb25f6627ul, 0xc8dd5337b8cd78f2ul, 0x257c8c1530ee9465ul, 0xea1c88061c7d124dul, 0x494899a7486414c0ul, 0x45774a6a62763bc8ul, 0xc66efe3fbbd40000ul, 0x94da500c9958fb13ul, 0xe04733e0bad55baaul, 
   0xc49411825e2616aaul, 0x6ced3c25a6514944ul, 0x1594507054e8c000ul, 0xeda8c61a310f4ae2ul, 0xd4a0bafa2757220aul, 0x3b6dc656d0e4ba2aul, 0x23fb5b93f90cf77bul, 0x182e0345a436d55ul, 0x4002cb4e0dfd0bf0ul, 0x900f6d3dbfa52228ul, 0x65a01142812b2eb8ul, 0x3950b4b0154c1084ul, 0xbbd95ef6120883c9ul, 0x7bd3fcb5de7a10dbul, 0xe9cb793744963ee7ul, 0x2d00b3e7c71060bbul, 
   0xc25bc12a0ed3c742ul, 0x2a291ad94c636f22ul, 0x404241c048f7181bul, 0x6528d4f4b85e7a00ul, 0x47e5ea7e51f7c0bcul, 0x3457153c16dae5c2ul, 0x21bf4a9c1136e533ul, 0x374d381e33f73d26ul, 0x8e5ce420c31334a9ul, 0x1158418bf476ed52ul, 0x6d67be3bed9da58aul, 0x583e58ecfada9259ul, 0xa8a0865b590be958ul, 0xa8cf56374f1c8700ul, 0xb4e0c0a1712db784ul, 0x42e72e62e117d007ul, 
   0x7d6e8fb9e10378d2ul, 0x4c04cf283d0346ful, 0x65376c1a0760317cul, 0xd53ff7c30c8405a6ul, 0xe44c9778289f116eul, 0x21d0e0458513238ful, 0x314c46667070cc50ul, 0xd9fd1f010856b11ful, 0xa84b53d2fd2039bful, 0x3d4375d6cf424b2bul, 0x77222db96cc27131ul, 0x1aa99ac6338053c5ul, 0xb304b33018cd249aul, 0x5c5c60a9cd263544ul, 0x40a5f8fc293c0e92ul, 0xaf881d92b0fb3492ul, 
   0x32db8670446616e5ul, 0xc888c7b4e8cf44b3ul, 0x9ee357e93011e930ul, 0x5a914c4a9081c398ul, 0xfba8c362639a04c4ul, 0x4261034d4622cb8cul, 0x6b517c5f152109bul, 0x99ab7970f1464736ul, 0x91c5bb04b77ded53ul, 0xa24261c93ec8d2e4ul, 0x440f4842d30cba44ul, 0xb1d74a74fe813f15ul, 0x18d6cb88fbea3189ul, 0x1099611471185ac7ul, 0xfeaf03f61716ad6dul, 0x4796b49a5ed76499ul, 
   0xba360491ac67a042ul, 0xb3a4e43aa5cf1f60ul, 0xde29c140453fdfe1ul, 0x84c01098c584110aul, 0x9292df4f5a18c12cul, 0x51be22023afb1084ul, 0xa811f7ad78e1ffbful, 0xe2bde1d2b92ca48eul, 0xf3c2f647d78bf79eul, 0x390fc053f7988acaul, 0x33d424c5bb3a41d3ul, 0x5945db0e02584ec9ul, 0x90362a618f11208aul, 0x7df5ff8b8de090e3ul, 0x1bfb9aecb607b532ul, 0xd5ff365d27e9fc18ul, 
   0x3d4233c8ab516b79ul, 0x8242fb0e10398bdful, 0x19f098f0951309caul, 0x631ac8722cd1a713ul, 0x56559a510931ca80ul, 0xb8237607c15df2feul, 0x7cb71ffb1ded4708ul, 0x2b6696b20a5e2061ul, 0xa46d426ec6f39096ul, 0xb22a605d24841388ul, 0x9945fa4f11820c93ul, 0x3479109953e36f9aul, 0xfc8eabcb15731671ul, 0x9247fadbd37c9dc5ul, 0x2ac86db9c7938400ul, 0x6013e711a54a89a4ul, 
   0x7714d8f47190c808ul, 0x8c45080daa7f2e26ul, 0xc5a789849ced2483ul, 0x52e1787f4e3c93deul, 0x9f2a385d91e687edul, 0x366b0eeb92270613ul, 0x3e2b1655b97b7cceul, 0xfe28a43fc56e040ul, 0x22cbc26252393ec2ul, 0xb1be311ba1213807ul, 0x91934c061a48f7a4ul, 0xf6ab7d7a9e4e1714ul, 0x283cd68ee1252700ul, 0x650d87781caefdeaul, 0x87c3ed2b71368ceeul, 0x71925a9f92801041ul, 
   0x54fa9da625a12b00ul, 0x390dc64795d53a68ul, 0x9875c5c49ab14c6aul, 0x57d647a23836f8f5ul, 0x959c7ed55183f27ful, 0xb4466307e5a0fcacul, 0x2be5ad4ff9ffdf37ul, 0x9daa0f96b7e11a3ful, 0xf17d3a4314e0d527ul, 0x639417eaf34d2f14ul, 0xd617e462d7c0388eul, 0xe878d10844c90d8ul, 0xff55959e62f942c3ul, 0xead8d9d79dfeffe0ul, 0x9fd83c753414aeul, 0xcb2555f1dca09300ul, 
   0x452e8014edaf5a72ul, 0x153f499fb710675dul, 0x6145a8f83366957ful, 0xab8a01f5344a1d29ul, 0xe5f83ca9d9875c7aul, 0x5cbefd60386dff5bul, 0x80c1d72f750e2cc9ul, 0x2f1a21333d5cece1ul, 0x92836ae7aebf13d2ul, 0x1c2ae380090ac690ul, 0x808880caa9dec446ul, 0x2be7da8a4efc73aful, 0xac0406281c603028ul, 0x5ebe41c8009e3f74ul, 0xa9b53ed2dd06f25dul, 0xf23c74ee6fba0b3eul, 
   0x9135f45e6852631aul, 0x34e024104446b062ul, 0x140b0b141c934569ul, 0x8408b765c4b0a4c6ul, 0x14e98ee14527edc6ul, 0x290408d6f9d98089ul, 0x57f6232168e19573ul, 0xa286ad258fcc6d49ul, 0x3d5f42440a8110d7ul, 0x8a243713ff38bccbul, 0x33179d3c9f723571ul, 0x9452110e5f32c255ul, 0xb2a3dd2489729e3bul, 0x115cd26b6781fe70ul, 0x8717060979a1d0abul, 0xb342f41e632cbd66ul, 
   0xb160aa19269aa824ul, 0x3a438801154ab014ul, 0x2c2f8145db04ef0eul, 0x4b746fd00bec68c5ul, 0x6f7d7eb86aa771bcul, 0x56ec5c4d475559d3ul, 0x56cfc16a955b0fd7ul, 0x2ae5a40987230a27ul, 0x86a5b6a80adafe76ul, 0x453fdc90ace29070ul, 0x620110c03bf12ab5ul, 0xa491e3c7f2cc6428ul, 0x264eb294d3cdc476ul, 0x70ed7edccfa0081aul, 0x9ed7abdd9b9a91ebul, 0xaee69daa3190be6ul, 
   0xf773816540702e66ul, 0x13b20749cbd00dc4ul, 0x700bc3e36e9061f5ul, 0x18bab42dc4d4a0baul, 0xc2425fc34e90a04ful, 0x826921488a9fe598ul, 0xece775dcebf83f97ul, 0xb196244fb0dc5d5cul, 0x17ee2b3c64cc9d0ful, 0x787cf777bb103f24ul, 0x824e8795295fc5c7ul, 0x904496f09d0023d7ul, 0xc556a99595c01baaul, 0x53861aa4a24607b0ul, 0x448a69b9ad394ul, 0x17e4af5793f8c378ul, 
   0x528985a48b7db8ful, 0x9dd92a17df92ab7ful, 0x80017b1824062db2ul, 0xde1dbe72fe1f392dul, 0x8a1898a2403d387dul, 0xc911a11c48331b55ul, 0x4bbb929a9571122dul, 0xb0f6858e0e030a8aul, 0x4acbefeffd0ff9deul, 0x646bb73881fd6afeul, 0xbd02376a055980d7ul, 0xced9218a6156fa2ful, 0xc476100b2d5ee953ul, 0x6a54bdd384295518ul, 0x623d7c424de090beul, 0x54cfac250ba9e241ul, 
   0x37e7f22c5c062ccful, 0xf5a2498c61bd3f97ul, 0x9e667f95e0bb9bd7ul, 0x5e8ea2f98eeda96dul, 0x707f6ac64e29889ful, 0x98a02d72e8ed7982ul, 0x8949524429a20f44ul, 0xcd1087a2e113466ful, 0x55084ebe3f310c70ul, 0xdffb3efe2c45768eul, 0x9afabe52ba974bbbul, 0xce3136a6f26aadbbul, 0xd14c913528ef3798ul, 0x6bab8dead4dca153ul, 0x53274262c753a0adul, 0x12c462092ba56bc2ul, 
   0x747907d4d524720ul, 0xe217e8de8e3728fcul, 0xb12bb9ffabe67eb2ul, 0x98dfaf8e0f95fe51ul, 0xe4342fd77b43bdd1ul, 0x64819f045664f50dul, 0x10dff9adf47cc2a0ul, 0x379d4fe4ffd9af77ul, 0x939e01f01b885121ul, 0x62b7305f8a91856ful, 0xcc9f266a74471f51ul, 0x73be110f6a388855ul, 0xf071e56d4cfd16dful, 0x9a3eeda1f373b65aul, 0x8c28ffd5fd4c8d20ul, 0xb6073677a3fd920dul, 
   0xeb935c2d93b8a24ful, 0xa67f8fc01cd8b22aul, 0xa63659b86737b0c2ul, 0x130b022314005278ul, 0x4e1fdf21ac878c0ful, 0x72bcf9517980a2c6ul, 0x350fb06ca3f5c824ul, 0x8af4114e6d2d9ddul, 0xef4a991bd951c2dbul, 0x8838101abacee367ul, 0xc5303108d94aca1eul, 0xa8fe4e884a24f923ul, 0x8486b13d1f1f7a14ul, 0x51c3f0690a7a64c5ul, 0xb0ffafe6c2215cfcul, 0xbe425bec4e54cde7ul, 
   0x250caf576bdcae27ul, 0x5fdb94778fce49e6ul, 0x54cfc9f59f3b0712ul, 0xab2aa64a66bed6fdul, 0xa1f1cce06a51c6ul, 0xdf23938c05c71350ul, 0x1134522033074c9ful, 0x11dacdc8e5e64a5ul, 0xfeefc408fee8c020ul, 0xcc86327331d23e0ul, 0xcc2ad8c78fabcff6ul, 0xb87fce66fbaf9ddeul, 0x6ff005dea99cdd2aul, 0xe088a6bcb0cec6f9ul, 0x9bf67897dee79baul, 0xf9f94c98ab72bf20ul, 
   0x24939a25b8862f01ul, 0xa1e9ca986558a1aful, 0x95e6bbb4ffabfcebul, 0xf1336c417d927f4ul, 0x401c0467cf6b8d9bul, 0xe8cf3fd62863bd03ul, 0xac1647877809c333ul, 0xfde59b87fcd1819cul, 0x7b2e713126221553ul, 0xce26a8241640482ful, 0x6e3f3e6d702c201aul, 0x2edf88c6e50e8e45ul, 0x6af15b5dce1b9f8cul, 0x2d92c7e3d5f879bul, 0xfbf377a7be0ae713ul, 0xa659b5daf8fa453dul, 
   0xc9a5dae0b658c7ebul, 0xeb519b1dc9feb77bul, 0xf335c40243b03c6ul, 0x4405a422741b880ul, 0x8919a39279d21120ul, 0xe16b7ecf62c255ebul, 0x95839373eb5b0be3ul, 0x76b93ec77c68fceeul, 0xf86e9140cb3ecbaful, 0x2a15c97b5d1e4579ul, 0xc2caf64560f73639ul, 0x122e6ab41e807ef1ul, 0xb1a12310012335cful, 0x1021714e86058767ul, 0x2826e63069185653ul, 0x7d911100ffc88cc6ul, 
   0xfb7cc9dc3fdbd7f3ul, 0x74a00e4e963f77cul, 0x1e83175d0bd51ec3ul, 0x757954756f32efedul, 0x508ab31b24a09f51ul, 0xb8767abc283dc081ul, 0x21ae99309f6a9938ul, 0xa78b5f5cfb694640ul, 0x2d48450f96c20d56ul, 0xe734e872e231211aul, 0x9fa191fd1cda3e44ul, 0x730f9bcae77ebbd8ul, 0xabbd4a730bcd07b7ul, 0x9d518cddcb97e3a0ul, 0x9a97df927b55b45cul, 0xfcd5d2c406c641deul, 
   0xa690908c12ab42caul, 0x8ee72c4a0b85265cul, 0xb7f5c4c6c2938039ul, 0xc1cad754e8106bc4ul, 0xe0ae7effb5584072ul, 0xa8f27de5828a6656ul, 0x8347d52bf41caef4ul, 0x3518d159f886fa99ul, 0x86f7facb5046b7aaul, 0x84ba7b6f66b7878dul, 0x18339fb47146466aul, 0x4e2723ca506d99a4ul, 0xbaa2bf639a08c9cbul, 0x9072b40e5093f225ul, 0xa3cca59a96efea26ul, 0x2270f8182818aeebul, 
   0x7f52f5c68bc3e5a9ul, 0xee72fae1f08ef0beul, 0x333a8147b19bbcf3ul, 0xc50055b5d363ef74ul, 0x9090733240d93af0ul, 0xd52088e9150848a3ul, 0x529a222b6a9e1060ul, 0xe13064925fcae458ul, 0xaad982faf371e292ul, 0xf9b9977378c68caeul, 0x7cfedb59df4eb176ul, 0xe3c6d42fbfbab8b0ul, 0xfbbb3968667927d9ul, 0xfe7e3fdd7c1ddfdeul, 0xaaa39ad3e4c28f28ul, 0xb5083251c0b3f7cbul, 
   0xf838adcfcfb47f7ful, 0x53c4c098b7e8a780ul, 0xaae111071553b463ul, 0x206ee48fba57f50bul, 0x7fb19cdcb9499d79ul, 0x7d54aea9b7832bdcul, 0xcc3958b856bfdc79ul, 0xe0f571e3389e1c3ful, 0xe6e7bbe3e023b352ul, 0x3896b88676e6c39dul, 0x1550e0376b73bd9ful, 0x45b5c949ab348030ul, 0x78c202df8ce7667cul, 0x97381faff5690d29ul, 0xab2bbffb4ab3a5c6ul, 0x9b7c57172e19dd26ul, 
   0xd4cf3559be95e6d4ul, 0x9ba662689fd6db83ul, 0x50e2ce5c2f0ca96ful, 0x21a72661045b5f54ul, 0x8339fdd86c59b8a3ul, 0x5525309d4505fc45ul, 0xf0e41c280150662cul, 0x8f85789a669c0cf3ul, 0x39db00ffcfcc3342ul, 0x2b5c2ccf70f47b76ul, 0x4b33f78d2c6b42fdul, 0xf596a4aa57ac2db3ul, 0x8fcedc496f59ffeful, 0x2825f9f2b34b2766ul, 0xb070691416574ab4ul, 0xdc6183aa0976e6d8ul, 
   0xf4e85b9c706d25a1ul, 0xea78f86944b09aceul, 0x54bb13f488b9c231ul, 0x5d79e60baead9daful, 0x740cc567c96a3c28ul, 0xdf2c7be0b3759b8ful, 0x15eb71f5ec8c29dful, 0x50783c586c6b2d4bul, 0x589d1dcf2cfff684ul, 0x498d67832842381ul, 0x973093aab263e41eul, 0xa387c972204d0b09ul, 0xfb9c1fd3234b53daul, 0x68f5131a4f320fcaul, 0x264a22b19d467522ul, 0x2ca977237aa65a19ul, 
   0x45717fbfb3e30394ul, 0x906619cb9e36bcdful, 0xbc7c6fb41feb52cdul, 0x8df7b5ecef6c99ful, 0x8644f18ac93d94e1ul, 0x59aa4ef0a20de0c2ul, 0x1e9409ec62474cb2ul, 0xb9ba0211c74f10a2ul, 0xe77113c68ba5fbb7ul, 0x516acacdcb573961ul, 0xcb833693d77df5ecul, 0x5e2989ccd3fedf98ul, 0x9d966dddca66cd9cul, 0xd398d57a7e1f6e81ul, 0x12c2d83884e5becdul, 0x2db80aec3c771d7ul, 
   0xb7642ff071eb2e98ul, 0x533d2b24104cc530ul, 0xc9783bce6f864637ul, 0x1ed87d61f77969faul, 0xe6672791fdb9ef1aul, 0xe1a1a3034a163667ul, 0x9c16776a5abe56b6ul, 0x9babdf461625a149ul, 0x497bb85d4ced7f65ul, 0x22be0413be28ecdaul, 0xd4a45843d3a466c5ul, 0xa711961261644bb6ul, 0x1938b26ea9a98511ul, 0x1b27f74ff6f010b2ul, 0xeab91d6f563aff92ul, 0xbbe7e6fee9adb6a4ul, 
   0x8fce4e42e1cde2adul, 0x87ff5da007cb6773ul, 0x6333f143abfb8d87ul, 0x1c9f644d736aa2cdul, 0xb9809378e07a3c6cul, 0x70f0613a41f8a6baul, 0xbd312f3384bd4d2eul, 0xd22840c99fb11007ul, 0x9dddb41eefbb22ecul, 0x319ba95e4dbebb3bul, 0x9ea0a99d89d5e7b3ul, 0x5b273db95bcb1e19ul, 0xe0ffb247e5ee6792ul, 0x4366717a6ba99eccul, 0x9460db32179346b6ul, 0xfc4fa921ab034a16ul, 
   0x1d9e80b05f1be2d4ul, 0xe2c6eff179ebf368ul, 0x77becc0022053c45ul, 0xeb27e82e86c3fb28ul, 0xb0dabfc67b3ba627ul, 0xf3ec98d6d2ca85c5ul, 0x67b7c8f6dcf27b5eul, 0xef67271196bca5b6ul, 0x9ed475cb50529436ul, 0x307a5cbcbb5b24c6ul, 0x8b84e9b9eecc599eul, 0xa694c57a6235afc0ul, 0x4d4bdc21ca678669ul, 0xf89fbdf824a6f80dul, 0xde4e502d5566064ul, 0xed105eb7e6e66179ul, 
   0xe489d65918050163ul, 0x5e1b6e18af9f5043ul, 0x6b50bb32820b92dful, 0xbe8ee858bf9596ceul, 0x3079053bff307936ul, 0x4507a42e47cd142ful, 0x914458b9a72711a9ul, 0xdafc4605029d0adbul, 0x8cd9b214030a4877ul, 0x58dd7e764659273eul, 0x565eace437cf1a16ul, 0x1ce0962a60b8491dul, 0x7692c497f5f7305ful, 0xfc2d2de1b95a8e4ul, 0xcbd4db4e72bb58eul, 0xf9a3d3fef9e9ae4ful, 
   0x93c060b266041f73ul, 0x315705b93177a7e7ul, 0xc87d9311d4c9db49ul, 0x8cefb7f50404cc29ul, 0x22b953eec91d64a1ul, 0x7db1aa5a92ef7b68ul, 0x6ff3114f64705aa9ul, 0x25525b9dd3909ff9ul, 0xcda8412d09ecd831ul, 0x5c158bbd94ace6bcul, 0xe2cdde3272de1d6aul, 0xccf42ae8a6fa8adul, 0xa6b89100686531d3ul, 0x6840fa7094e41d2ful, 0x397a8f2aafdf97faul, 0xcf66e2727052ecb0ul, 
   0x9a8aca86fcb72bd6ul, 0xcf2c78fd1eaa5c2ful, 0x43682dcc9c6fb5aeul, 0x6b72b172b8686ad3ul, 0xa785db0568d958f9ul, 0xe80c6efbb59aca4cul, 0x96026ac6d4f5ed92ul, 0x2f81ddc2fc45a123ul, 0xc6baf7e5fddc5f18ul, 0x46667b2f1b0b3b7dul, 0xc715b7a577eaafceul, 0x6455ae696633ec0ul, 0x7aba8efb6ab990d2ul, 0xb59192b52bbbced1ul, 0x81a0f95e03b37868ul, 0xd18b112523b37a92ul, 
   0xd60fc8a14b7139d6ul, 0x40923a9c022337a7ul, 0x8980000a80900731ul, 0x315aaad8cf476e7dul, 0xab23e097b9b2f5eeul, 0x8339e6b525e7a9b0ul, 0x3a535ce5a8bf2163ul, 0x2591eca393f80223ul, 0xd72743c3f2a4162eul, 0xf1e7646735d9b778ul, 0xb0984dc7716157b8ul, 0x244e59e0329718a6ul, 0xe20e2055f8961000ul, 0x7f2faf201127fb27ul, 0x64e418b57947f434ul, 0x533b20c72c76b1c3ul, 
   0x8e374130cc4d1512ul, 0x5730a9b6c7ed5b8ful, 0x9383f3d0f05417aeul, 0xeef75c7d61ed0511ul, 0x4ff928f476a03b76ul, 0x1e01157c7758bfc9ul, 0x53a96e988cb090bbul, 0x37f3dd82cfb704f4ul, 0xb0bb2a2a97d03baul, 0xf32a513ce832eb8aul, 0x9e7a4e90ffdbc152ul, 0xdba3eaab3601bb6cul, 0x3fc8cf43596b27f7ul, 0x6286aeb1ad91d531ul, 0x5296cc889a2091e6ul, 0x530976e2a28e62f3ul, 
   0x47e315d225613113ul, 0xe939d0e918837c25ul, 0x78bc544dbab2f53cul, 0x95c6b95a1d69cb7bul, 0xc66e8749bef57f5bul, 0x6dee3e9a492fba7dul, 0x2aa1be0a678b123dul, 0xa3a6792878c5058ful, 0x96883e8a684bb681ul, 0x749920fd64f4c3c4ul, 0x3c032c20886126a6ul, 0x918a395e4c9b38bbul, 0x71bc194d33b6a99ul, 0xec9617b375df67e7ul, 0x40d3f0b335b18f9ful, 0xfa0f0fb2798a4d6eul, 
   0x3f0c3312f2db1a5ul, 0xca0c0ad0645b9bd6ul, 0x20404036f9ee373eul, 0xa7c0314a66f03219ul, 0x9bf0ac28088831a7ul, 0xdf82ace9a159d168ul, 0x63e782f262ce9b41ul, 0xaf8a7be7962c10a9ul, 0x37ae5fa97cedace4ul, 0x3bbf1307e3facd8aul, 0x4d5ce5c5e2d2fab4ul, 0xdaec34bccdbd3353ul, 0x8043b95e3cbc5faeul, 0x70a13192c0c007ccul, 0x8c0b0c4769be1531ul, 0xa6c6cb0921909ab0ul, 
   0x2b51445adef1716bul, 0xcb55010aa8b952b9ul, 0x55a48a37576aaba2ul, 0xafbc85b9a3bba901ul, 0x93932817e292b366ul, 0xd260d688a651fd27ul, 0x627d0f0b8dc0ce1bul, 0x1d9301458e200440ul, 0xd13dcc3a316e438ul, 0x18a896a6238ac2eeul, 0x6a749f47885c94d1ul, 0x19e6f0c66cb7c8dbul, 0x5f7c923c4f77637dul, 0x98ca2e94e5e79810ul, 0xf20ae626b28798f7ul, 0x5ad91b933daf7d8ul, 
   0x408dbe6ea13d1e24ul, 0xf91f14b8ec8002c0ul, 0x752c1570b0304f09ul, 0xf8370ac27b822788ul, 0xfe53136514813d4ful, 0xe4bf758009de5f0bul, 0xdb6cf6d32df1dc2cul, 0xbaef6ad7b477badcul, 0x6e081a29a4b2392eul, 0xe36263b86f627477ul, 0x4c1487b09005d8d1ul, 0x18dc97a63ad86d8ful, 0xc2e09b250cdd3b22ul, 0xf889a5e9826cbf2aul, 0x83cdf1b4d4b70668ul, 0x10b6e59965718413ul, 
   0xc5cbce417ef82218ul, 0x136ba0b144d4569cul, 0x7964a2d1fd8dedc7ul, 0x3a59d0004f91eedeul, 0x97c52ca66fea60c0ul, 0x7241880886020221ul, 0x13f1f22eaebd2becul, 0x15bd2133e38858cdul, 0x1aeff080094ddd04ul, 0x482f1629bde1dafaul, 0x298c492dcd65a824ul, 0x83dc56143ea72d93ul, 0x2e5e9ed864ee991ul, 0x60d1eb8f2730a05ful, 0x445749c28185ea30ul, 0xcea7262febbf21f7ul, 
   0x5da9d0a8f8eee592ul, 0x4008f24fa7800416ul, 0x47fbd7833fd5fc1ul, 0xaa0abedbe330b884ul, 0xdca658b5dc384190ul, 0xdc08337641ef87cul, 0xc77a7b65f5c91082ul, 0xdecd13e63a31646ul, 0x1432d3e931448311ul, 0x6e266670dc1b989aul, 0xefbd9cce99992fc5ul, 0x1f3dccf0d4258223ul, 0xccdcb2bb858dfc51ul, 0x3901e279bbbbbd80ul, 0x6e7965a16c8e770ul, 0xeb8629cd10434cf1ul, 
   0xcf84758630185317ul, 0xc58abf1911532374ul, 0x8ddb8b5c745330eful, 0xde00102af1722a9dul, 0xaacc400037eadb1bul, 0x3075bd3f2099eee7ul, 0x659679e8611100f0ul, 0xabd1834c14040102ul, 0xb9dc04a0c0122eb9ul, 0x3d29f7c0822aa985ul, 0x12c2b7188624c6aaul, 0x447449c8ab53478eul, 0x8c90108d7d13fa93ul, 0xb8ed6008b8d9366dul, 0xac0882ae4a382f8bul, 0xd92f301328e203a1ul, 
   0x90022249e8022713ul, 0xa0554415773cfa17ul, 0xc9623983c51e9dd3ul, 0x83353a29a6e12c74ul, 0x17b56d041084d9fcul, 0x2bef416078004015ul, 0xe51be7bdf53e0a79ul, 0xd2bf2bf520302e2ul, 0xf47e0236c11040a0ul, 0x7a70a0c103e4aa82ul, 0x384497adf6a09509ul, 0xc06231969b8b0c75ul, 0x7cf223a9568a051ful, 0xb77803d738f9b7aaul, 0x6c14fc60008197b2ul, 0x98bb02dba78005faul, 
   0x22ad2e5f4239cad3ul, 0x6cb5a29d04164e63ul, 0xcec07465ecf80000ul, 0x42a572e5964c59c4ul, 0xa9c41837af57ca80ul, 0x5843953aab3e2103ul, 0xa9d2284e4c25488ful, 0x2f2b537daf4b4593ul, 0x6f505f0e57f4e699ul, 0xd8371c4d64e00021ul, 0x835802c5919e1c1eul, 0x4967663f5572d036ul, 0x7fd90222404f32edul, 0xa59a4fa2657b3b31ul, 0xa354d6ac5172bc8cul, 0xd2f2ead599c228acul, 
   0xbc7da5bfad9b9b8aul, 0x912044ee4845a94dul, 0x453a2811a7d2ceb0ul, 0xf2d5f55ec38335c3ul, 0x7988100d1c2fd7daul, 0x695b82020687a5eaul, 0x675bdbb1c3e593f8ul, 0xb17585603dadb04cul, 0x18f7cf41b5b5d61ul, 0xae773f822ea2e941ul, 0xcc8585851a146996ul, 0x95e599e6f58bf9faul, 0xcdad949ccaa5cbc2ul, 0x356e5d4ff8e8acaful, 0xb8670a455377a939ul, 0x8ff71c7c9c4a573ul, 
   0xb0cbe1777cd15f23ul, 0x4444dc5d7fb80805ul, 0x672d95bdb54d6b40ul, 0x7b45df7b319096c8ul, 0x2c37dc75e5d4e9b6ul, 0xed50ce4717a05901ul, 0xc14cb301b355ab8ful, 0x5556a542cae5e48cul, 0x372545b255de4172ul, 0x8b9824c9076c9939ul, 0xe44e5f1f14c4c8a1ul, 0x625bb72f1622ebe3ul, 0x8b13f2fce93cf15eul, 0xb34440263e97e121ul, 0x891d31f6790c905eul, 0xeeba964bb77244f3ul, 
   0xb645b5d7bb50a288ul, 0x614a8656cdd0abaul, 0x2514517c955d6601ul, 0xa28988339729444ful, 0xa3e58a8cced11791ul, 0xf145db8fa707efb7ul, 0x78496ffb7201ac2ful, 0x8477bd5e5fae7bf0ul, 0x985eaf0b343fa0b9ul, 0x880887592f0786f1ul, 0x82dc56b0de292f68ul, 0x77a121f554f59a7dul, 0x5c8ecac6d05d8d96ul, 0xbce6e454d64c8da9ul, 0x594d546ca13c5352ul, 0x13d299f04480e574ul, 
   0xbe6eaa48819cf00cul, 0xf0843144a39a61bcul, 0x3d611254c1a785e2ul, 0xed9c9832f3e2fc55ul, 0x572c77bf25676f43ul, 0xeaf6797dc7bbc9ceul, 0xc302209f4be399f5ul, 0x10e90d991996b949ul, 0x59809d6b2f15cb0eul, 0xaee975960b37f19eul, 0xb99115edf53361beul, 0xcb181689d4554e5eul, 0x524f230810b28a06ul, 0x7a0b2aab2832d5dul, 0x1d4e9e50dc547145ul, 0x33718efcde11009ful, 
   0x2152299e07c420d1ul, 0xcb8bf1ee8863033ful, 0x41fd89cf341ac79bul, 0x1463cce42b90c7ful, 0x4cd25b73575ac11ul, 0xc07518585496342dul, 0x37efc84d0d8fe2bcul, 0xcdd931c756bf0db3ul, 0x1aa1e5944b3c6402ul, 0xc98b4651301ba2a8ul, 0x55720414a62be0ul, 0xb5fa17495bdc9726ul, 0xc2004442747e4eeful, 0x12c7c64ab71ac2dbul, 0xa5dcfa7acc0720b7ul, 0x27d57e7b6127b87ul, 
   0x68807a71585a4432ul, 0xd05cfc93f14a1a54ul, 0x79ae9b4b0115ebb6ul, 0xd4699f64744ce0b6ul, 0xcb119389661929aful, 0xc14344e0cfcb758cul, 0x47be8527f2047cf1ul, 0x5296f81aa11a58e6ul, 0xd24e9bb86ee7fbbdul, 0x21dff7dc37eaful, 0x8984d584b7ebfda8ul, 0x4bf69df6f16a8f8eul, 0xef194bebebb9e3eeul, 0xa92042eb2cdcab51ul, 0xc95a300ba1058585ul, 0x6cd6fd96660fe464ul, 
   0xb9ce902b61528991ul, 0x5e66a96c9ec349e0ul, 0x6040a2bb7ccc343ul, 0xf5cf13f8020824c9ul, 0x1d743ec8fe98983cul, 0x17ffbd64068eef3bul, 0xd1deebe77deddc56ul, 0x972ff9bd800872fful, 0xf2b4f9bae41aa7f6ul, 0x7f57c285e5711010ul, 0x9c6ad7b8a361dce7ul, 0xad4572c0c5087499ul, 0x62f3aa1edf2e78b8ul, 0x364bb2f4df39bad9ul, 0xb48da73935b305c9ul, 0xd1d7ed58fec65680ul, 
   0xe81604278cd45619ul, 0xc68db619a4b260ccul, 0x98f7ec60c13a7558ul, 0x5b1ecb3fc7603077ul, 0xe4bfcb80004134acul, 0x7feffaa401dfdffbul, 0x97747d35616d7583ul, 0xf6fc108860a59d4eul, 0xae41f4fcefbb2ad7ul, 0x961a1c3e8159fc76ul, 0xac5eaf96820512a8ul, 0xd9d3e6daa517c93eul, 0x35b8b97c8ca9a4faul, 0xb08ae10adbea459ul, 0xaf75b38e010519aul, 0x5d65e7fcc448c02ul, 
   0xe3b1ccf6cf03b021ul, 0xe6a30563b1b0284ful, 0xf7ebebeb95e7650bul, 0x5bffdd1abbf2a001ul, 0x1686f42ddfee7a02ul, 0x816564a79da9ab09ul, 0xfedaf2bf57ba4345ul, 0xe689171bdaf2830ful, 0x1a33bb0f0bb31ecful, 0xaad2d41029144053ul, 0x4662c1c6e37a6a27ul, 0x7b927e5b200e545bul, 0x6c3403e33129d76eul, 0x4b0fd7d5a758c8b7ul, 0x22b2559fbb98f1dul, 0x378647111d26553dul, 
   0x3a349e58ad92fc0eul, 0x501c0f49b5354b66ul, 0x7825d97f63d6bb7eul, 0xa28593bc732f7e3ul, 0x412662e9d979fcabul, 0xfad67def27ee9ceaul, 0xe62cf54a00eec6fcul, 0xbecd667db8436cb3ul, 0x5dd749f0596c9597ul, 0x822c6f72b7ea295ful, 0xd27b9e1eacb1b3d7ul, 0x73b027f3eb3d156cul, 0x64ed6c993e3ec186ul, 0xb864255ac83d93b1ul, 0x37516c4f2bcf602bul, 0x5afab5bdf573cb17ul, 
   0x990fab7dd99ae0a9ul, 0xb8497783f648bc5ful, 0xa4932f4f93faebfcul, 0x1737222b8e4e2671ul, 0xeb69efc10c0f91c3ul, 0x57eac8cba076bdfdul, 0x30ceb98cf7c22a81ul, 0xcae376182c81bb27ul, 0x41e86e4183d1991bul, 0xd2b212347335bdd3ul, 0xf0ffa2739d8a0fdul, 0xef3f0ecf18c0edacul, 0xf8162dc3228b191eul, 0x98bef9b5d5f5b54bul, 0xdafeb40a1bfb7797ul, 0x9f17e7f217bbb40cul, 
   0x9aff7fe467be0147ul, 0x58ffb6026fddf372ul, 0xc48e3f1de306d64cul, 0x6760af5cd71e28e1ul, 0xff28e70472410517ul, 0x94cad9d97fde8fcdul, 0xff0c27369815c0c5ul, 0x7b3fc8f64924bb07ul, 0xbe5fb33307a366d0ul, 0x8ccec5c7c017995dul, 0x26f3ce7e3fc12545ul, 0xb922babfbf41b02aul, 0x9c4f3035a60b8ceful, 0xf1e3059ac39df9c7ul, 0x4144498b04000070ul, 0x2b9a1f7f39dd9a54ul, 
   0xfc356fd2ce83e220ul, 0xffdc3f700bfdee1ful, 0x57f7a4237a2dcb7ul, 0x498c162e6fd90000ul, 0xa4b75e4788ec78f7ul, 0x7021523ba33c919dul, 0xa07282fd27c796e6ul, 0xc2f8799915ccd4faul, 0xd061c5db55f9bddul, 0xbb6cf54eec30d3a7ul, 0x9bb59ce4e2e140d5ul, 0x4f82f39385525d5ul, 0x65b839b2cbc32810ul, 0xa5c6b23d5f757a69ul, 0x542b90c64f6f48caul, 0xeaefe5feff302954ul, 
   0x7d1fe63dfb567f7bul, 0xf7152d446fc6cd6dul, 0xb0a0201e71e25467ul, 0xf605e808cf9cae76ul, 0x5cb0e614c625b2c4ul, 0x8c937b306567240ul, 0xdc8ff6b04e7b58daul, 0x84ce9278195b44c1ul, 0x7edc65ecf1cec7a7ul, 0x3a69baad27065af2ul, 0xea2c22f55038c65aul, 0xdfeef2c9737ec7e2ul, 0x341a40cdba65ce1dul, 0x964f759b8188befdul, 0x83d7f56ff103179eul, 0x62e5c810bcbd1e0ful, 
   0x721fd4e91103ba7ul, 0x8cdcd9a7b4febceeul, 0x4b4ad3032cf0bd7aul, 0xae0acb4d4c3065ebul, 0xc0f935071c554b01ul, 0xebcb219166a1c592ul, 0x85142f4fda0e4f33ul, 0xcdd0291f9f98d747ul, 0x4056717e61fde5e7ul, 0x5def1fb63f333a3cul, 0xa3283e8a84d6027dul, 0xf8b0b1fbd95e78ecul, 0xbfd05ac2e77f033cul, 0xc74ffd4a941186feul, 0x752dbc4bea71898cul, 0xca5eb2bc40110622ul, 
   0x7c75c0e35b67a6c8ul, 0x3ecd19f9dd5a0ecful, 0x427b524ba92a3208ul, 0xde89fd3a4c6facb2ul, 0x9faa2f159913a968ul, 0x20aeba572beb65fdul, 0x67b7ded6b4b6ccddul, 0x6e01b3b9dedef184ul, 0xfaeb05f707763f3dul, 0xb776e6cc73ad7f61ul, 0xfabf1e37f3e60f56ul, 0xbaf93e73d1b2226bul, 0x9ab6861bfdeaaa3ful, 0xe5f95b824a5afe4dul, 0xa39a97cbf56cd41aul, 0x23c741a3fa760feul, 
   0xa07db6828aac8260ul, 0x31dcd5279fac4a55ul, 0xeb555c82071cf203ul, 0x95f207e326771918ul, 0x3762ab9a651d5ff5ul, 0xfe1378fc7fc067bul, 0x74300580f37f6ad7ul, 0x1d7703ab71fe5f75ul, 0xe9e0a3a7d64a28cbul, 0x2537fd7ceeec6c6ful, 0xb051400ec0b9c4f4ul, 0xf8fbecacc2c7d0f7ul, 0x35e33e6c293a290bul, 0x3f9f27d58577e525ul, 0x4273fc2f8277101aul, 0x85b676d66f2bf908ul, 
   0x1d4572c0f3017a1cul, 0x73730aea940f327dul, 0x3d43881e1a3ef901ul, 0xb62abcef7cb7efebul, 0xa4f14f345785f8feul, 0x55caceef4bcbff4ful, 0x37bb35697c5a33a7ul, 0x22405cfd5dd6ab3ul, 0x22240adf2d1748f6ul, 0xe4a4df8d813ae0a1ul, 0xcbd9dcc3ab84b84ful, 0xe559e3dcfe6cbf2ul, 0xcfd52a56404443e9ul, 0x78efaa6a36dd2aacul, 0xe4d295353ee898cul, 0x15a4e0a15965cbcbul, 
   0x7622aa2b1c352900ul, 0xc2ebb73be48f8584ul, 0xef7b5db9dd5ab14cul, 0xd7367e5af16e41ceul, 0x143c159ba0552a5ful, 0x5d747434533c104ul, 0xb7e1189049018e19ul, 0xadb90746639487d2ul, 0xfcf55dfa6cfb7e37ul, 0x149e93dd1928bd7aul, 0x33e4800898914c08ul, 0xa7cc049122c922b1ul, 0xd82575d5f9efa58eul, 0xa7ff0e7fd40ff13cul, 0x4bf368670a93d85bul, 0x22b0b37f83bc82f9ul, 
   0x2f4985f432b2e62ul, 0x2db8c9501033ce55ul, 0x8b2d99d4705a8a87ul, 0x7e3679f4c4c8a28ul, 0xe9d0788831dfdc23ul, 0xcecc8d62e97a519cul, 0x31d32b7998cf6b66ul, 0xed4000739ca10ffbul, 0xef2ebc993a827be5ul, 0xa5f87fef0b04184ful, 0xa8547a3f9b13dd9ful, 0xbc9874f764e29afful, 0xc2f236b4138fb33aul, 0x115faddcabae992ful, 0x8df275dc3ed15efdul, 0x4a0e87471b065646ul, 
   0x7eeeb2b0b1f14685ul, 0x80a21e9c004458f2ul, 0x6c90a93fb8045164ul, 0x86c73ede179f4d1ful, 0xadceaf35133b5b5bul, 0x1dd119eb08e88606ul, 0xbb6642dab5d6ba69ul, 0x24bbe6cd9fb907fbul, 0xb3b6c6d8ef83c9fful, 0x2db6ca459e592540ul, 0x7c6f7cf77bcccdd5ul, 0x82e5c5e1e0827443ul, 0xdbe39cdacb07e07bul, 0x3fcdcb94ab75b0b6ul, 0x9f57b66e3070f33ful, 0xfff85ebb5d3fedebul, 
   0xd39cf77347101f71ul, 0x444e454900000000ul, 0x6c90a93f826042aeul, 

};
const std::size_t data_inputImage_size = 9748;


const std::uint64_t data_usmImage[] = {
   0xa1a0a0d474e5089ul, 0x524448490d000000ul, 0x78000000a0000000ul, 0xa6c9500000000008ul, 0x41444900200000b3ul, 0x6593e7c184017854ul, 0xc7999bdfe7de81c9ul, 0x160cf4cdabae535dul, 0x83758a1159f525d8ul, 0xffea288649714801ul, 0xdd5dd7183b122afful, 0xb7b9e5f3263dae65ul, 0x5d1e7a14058307baul, 0x4842042420ab2d5eul, 0x9f401714a5db0242ul, 0x9052c71c69e4bb20ul, 
   0xa5cccc020a405d84ul, 0xc531084c54f29e4ul, 0x54631329a6504841ul, 0x2e028219ca71c699ul, 0x475b5dfe9d3fff85ul, 0x7199892108100849ul, 0x4a530849090831b1ul, 0x33c4248c04e8c5ceul, 0x408282920cccacful, 0x5ce725c20a0b9294ul, 0x33418a5c5d8106cul, 0x4eaa119f024858dbul, 0x804cc40928a7bfful, 0xe242499e63061021ul, 0xf8908108292036c4ul, 0x8b99e2100b0cf8c4ul, 
   0x6301836c5c5d89cdul, 0x18c4e20333108463ul, 0x9c4021043fd24283ul, 0x9084cf333c408408ul, 0x8421241493310210ul, 0x64bf31b1b062710ul, 0xcc060c609f18c0cful, 0xcd06030667e21089ul, 0x4204ce2662049feul, 0x81021212113e6662ul, 0x1b06271021219e84ul, 0x8c6c60cccc066663ul, 0x8041999831b1b1b1ul, 0x9333333019988040ul, 0xc42044e2671019feul, 0x9e8421084242119ful, 
   0x9831b18ccf108421ul, 0xd8d8d8c6c1830199ul, 0x89f102133f306318ul, 0x7133f167fa4cf301ul, 0x424242044fe20426ul, 0x338810842433a108ul, 0xb18c1830180c609ful, 0x133e60c636363631ul, 0xd7d33f3019988102ul, 0x113e210267889fcul, 0x8433a12108484842ul, 0x6318c52ec2010810ul, 0x1b1b1b18d83060c0ul, 0x3e2108133cc0631bul, 0x96ff35f4ce62731ul, 0x48484210099c4204ul, 
   0x84908424209d0908ul, 0xc60c06318b929598ul, 0xc18db1b1b1b18c60ul, 0xc062710842133980ul, 0x20420417f35f4cfcul, 0x4242424842040266ul, 0x51201049d0842433ul, 0x318c60318d8b94caul, 0x18318d8d8d8d8c6ul, 0x7e60c02108421333ul, 0x4020420497f35f41ul, 0x6684848668484020ul, 0x10adb01884292108ul, 0xc6301b05ca79c94aul, 0x306c6c6c6c636318ul, 0xf333c48424083018ul, 
   0x204092fe6be89cccul, 0x1212424842040810ul, 0x3ca8a6216684219aul, 0x2a69c71e70631099ul, 0x18d8d8c636318c18ul, 0x1092133cc1831b1bul, 0xafe6be99f980c4e2ul, 0x2742100810810810ul, 0x9484284212424248ul, 0xa75555594c32862ul, 0xdb1831b1630c3fa5ul, 0xc18c6c631b1b1b18ul, 0x80ccc42424844e60ul, 0x810215fcd7d13f9ul, 0x2424827424081021ul, 0xc69948aa90a48121ul, 
   0xf6e3e63adba9a9c9ul, 0x1b18678c631b6638ul, 0x80c18318c63631bul, 0x9999cc0601024849ul, 0x810842010afe6be9ul, 0x210921219d090840ul, 0x6ab8b90d3c9d4214ul, 0x54c3b6ec34a9aeaaul, 0x8c6c6c6d8678c33cul, 0x668493398318c631ul, 0x35f44e6273333020ul, 0x840810842100857ful, 0x6214908424867424ul, 0xab6cbb4d49c7ec62ul, 0xc62e72238eec3e44ul, 0x636318c6c6d827c6ul, 
   0x398402413a679830ul, 0x42bf9afa6662731ul, 0x9d09210810842108ul, 0xfea624214812121ul, 0xf1e0dbaf58bea9c7ul, 0x18b1a7a2aa0c3878ul, 0x8c63631b1b04f8dbul, 0x73008104e84cf301ul, 0x10afe6be80c4e62ul, 0x6742424084210842ul, 0xc70f98aa10884248ul, 0xa6b5179757b4a1ddul, 0xdf73a8b3747f71e3ul, 0xc6c636c6d867c68ful, 0x388413a113e60318ul, 0xd7f35f4066662731ul, 
   0x2424210842108408ul, 0xe38eb10884248274ul, 0x517d72f6963dbf76ul, 0x8bd62eac6efdf533ul, 0xb1b619f0643fb369ul, 0x2402273018c631b1ul, 0xa0333333333c4334ul, 0x4242042042bf9adful, 0x40909219a1212108ul, 0x66ef5d9bf7ea5421ul, 0xddc1c30ca1d65abdul, 0xd1fbf7f78c5cbd5ul, 0x7e60c6c63636c33eul, 0x627333c42424841ul, 0x81024bf98b7e80cul, 0x9d0909090842421ul, 
   0xf43bbf7e96282812ul, 0xa0e34f4753558bd4ul, 0x1bb7777f45abedf5ul, 0x18c631b1b609f18cul, 0xcccf8909210267ccul, 0x82fe60b7e80c189ul, 0x8484842109210810ul, 0x7f71e50428424866ul, 0xcb82a9a2ed8ef43cul, 0xdde3db2f17aba538ul, 0xb1b1b609f068dd87ul, 0xcf88668467f30631ul, 0x67e60b7e80c189ccul, 0x848484020402666ul, 0x2148484909090909ul, 0xb55590f1fdf63e56ul, 
   0x45cde5cdc9a79275ul, 0x1b0cf8c987361feaul, 0x42044fe60c6318dbul, 0xdfa030627333e212ul, 0x402040267313e602ul, 0x4848484248484242ul, 0xe387f61a77588442ul, 0x2d3a34c0eba87471ul, 0x4a54fd0fe94d1756ul, 0xfcc631b1b63609f1ul, 0x30666667e212425bul, 0x204cf313e602dfa0ul, 0x8424248484848040ul, 0xdfb1976a42248484ul, 0x8698554acc61faeeul, 0xe2c71865aad5da1cul, 
   0x18d8678db18b8a52ul, 0xccfc4248437f983ul, 0xf999e602dfa03018ul, 0x2121212109988044ul, 0x2cd5422124212109ul, 0x49499ca7187371c2ul, 0xc74d5d6ac5a53f63ul, 0xc33c13e2973904a1ul, 0x12113e26798306c6ul, 0xc6fa0ccccccc4f88ul, 0x48100804cfccce60ul, 0x909084242484848ul, 0x714287f61f1aaaa9ul, 0x1562dd0c61e313a8ul, 0x19f1b6294227a1c7ul, 0x99f1139830631b6ul, 
   0xbe8c4e627333c409ul, 0x13310082fcc060c1ul, 0x908421212424242ul, 0xc1debbf7315750a1ul, 0x5d4e5dc7f42e9aaaul, 0x19f80c043838cb1dul, 0x113e20301830619eul, 0xbe99cccf30089c42ul, 0x10080406673060c1ul, 0x4210908490848420ul, 0xf63fbca4d3521048ul, 0x4f8f0d22f2fa8705ul, 0x8ac82c71e4ea92fbul, 0x418301831867f851ul, 0x1be80c2bf884227cul, 0x4020113999cc60cul, 
   0x9084842109090842ul, 0x1edbb8d5d62cd880ul, 0xdc334dc5f2d252c7ul, 0xb29a695d43ea7bbful, 0x31827c5db114a56dul, 0x75fc408497e60c18ul, 0x9899e627318306faul, 0x8420484242108409ul, 0x7fa9179777501084ul, 0xb757eb57390feedaul, 0xba2e3efb87e1f0ddul, 0xe2e104214a5cc654ul, 0x4082fcc18306304ful, 0xc060c6037d2afe20ul, 0x848421084266267cul, 0xabcbb42021090810ul, 
   0x6ae43f43db4fb785ul, 0x58f8ffde1db6f2f5ul, 0x3e4b8a994d975796ul, 0x60c18c13e2e2e29ul, 0xa0105fc408417e63ul, 0x420417e60c18c06ful, 0xc421084210842108ul, 0xfb6c3ddad6ac5eacul, 0xfeeb2fac5a4a1861ul, 0x4bd78bf2fd5dc7c3ul, 0x90c38f1e3c94b1c6ul, 0xf983060c18c13f8bul, 0x37d1389cfe204099ul, 0x4089c40603063180ul, 0x8421021084210842ul, 0xfdfdfed51796ea4cul, 
   0x9b0d29d42829726eul, 0x22f2f9bfabf972a7ul, 0xfd4f4ff71971638cul, 0xc18306318d827c50ul, 0x44e20bf102133e60ul, 0x2273060c63006fa0ul, 0x42108481084844eul, 0x7a72f5d5da1c0842ul, 0x30f205293dbbc3fful, 0xddfd5f97ab1c74c1ul, 0xf0f0f9028798cbdbul, 0xc18c63619f17d771ul, 0x89f100844f98c060ul, 0xc18c60cefd331338ul, 0x1210810908044f98ul, 0xb9bcb55610810842ul, 
   0xbb18eefb7edf37a9ul, 0x5712e287ecdc7a36ul, 0x58ca9572f97edeaful, 0x1b643371f0ff5242ul, 0xcc60c180c631b0cful, 0x4cf133889c402133ul, 0x8133e60c6318c3bful, 0x8810810921084210ul, 0x7ffefc2f6e2f6c59ul, 0xf9545abd637777eul, 0x6f6e2f6a4cc74f0ful, 0xf2aeaa992a6dd7aeul, 0x8d8db0cf153f71f6ul, 0x204ce6318301831ul, 0xc3bf4cc4cc4ce201ul, 0x81084267e6318c18ul, 
   0xb528100810849090ul, 0xda7e1fc7eed597d7ul, 0xe1f1f771eae8bbb8ul, 0x1553acb17d62f500ul, 0x63054fd8c353a871ul, 0x8360c180c630cf1bul, 0x7100998804081139ul, 0xfe6318c6303bf422ul, 0x310209d084210844ul, 0xbbfbfaacbb750e13ul, 0xdfa9755d5aa99b0ful, 0x568b976a4a1c3fdeul, 0x59275d550a0c5653ul, 0x6666318678db1911ul, 0x100810201018318cul, 0xd831846fa4cc4020ul, 
   0x8421084080cccc18ul, 0xfb1d554b3c40904eul, 0xd5cbcb5707f7f7a7ul, 0x6eaba3a787bbf632ul, 0x84218acc777185d9ul, 0xcf1b02555575d554ul, 0x160318c603018db0ul, 0x2040201008102133ul, 0x9cc06318c630237dul, 0xc408490cd0842109ul, 0xfbe3f4f8d3aaa1ccul, 0xdc69d572f2faed1dul, 0x37186daeab8377eful, 0x3adb6ea90a12c1bbul, 0x8c06031867c6c64aul, 0x2010204201139831ul, 
   0x8c630609be820840ul, 0x3a12121089f30631ul, 0xdc3a4db526040841ul, 0xcb57d717d7d7777dul, 0x7554e8e3f43ca365ul, 0xc56c5c9fbefdc735ul, 0xb609f1b1945269baul, 0x4083018c60cccc1ul, 0x6fa0881020408102ul, 0x4060c6318c60c602ul, 0x9021092124242420ul, 0xf1ffdeef517166adul, 0xa6d972cdd5cd0c3ul, 0x3c62baaa12e53c94ul, 0x554843b31c61fa1eul, 0x39830cf8d8678293ul, 
   0x408402044e631831ul, 0xc6009be829081020ul, 0x49084ce6318c6318ul, 0x5c5c4c0842424848ul, 0xcd0fddde9ffbb75cul, 0x92e48210a4d97ac5ul, 0xdbb8fe871e54621dul, 0xcf14a44d3755564ful, 0xe630667318db0cf0ul, 0x4842102040810099ul, 0x8d8c63183004df41ul, 0x242424248484cf31ul, 0xe1cdb7d756aa2884ul, 0xca36ab16ac6edde3ul, 0xf1e72482e72510aul, 0xcb176da9c1e3cd9bul, 
   0x318678db1b1739caul, 0x40998810183199e6ul, 0x9be82142108420ul, 0x84ccc631b18c6318ul, 0x6ac0842424242484ul, 0x7da7c7c7575c5f2eul, 0xb9c208672d458b49ul, 0xb8fbbc7fb0cb8410ul, 0x6ce51bad5f5db12dul, 0x333980c6d8db1b1bul, 0x2108420402013398ul, 0x8c631808dbf410a1ul, 0x8490cd089cc1b631ul, 0x6bb7d62e26121090ul, 0x95acb97a9ebb361dul, 0xf39452b173b62e73ul, 
   0x16913771ff7c7db0ul, 0x36c6c6c4e475d757ul, 0x113889f30199e636ul, 0x8ebe821484842108ul, 0x4ce63636318c60c2ul, 0xea80424212124248ul, 0x10e3a7bbcbb2fad5ul, 0x8a9871e42e8bab16ul, 0xd869893554e434cbul, 0x96eacc61c7371eeful, 0x36c6c6361043155dul, 0x1026798040627cc6ul, 0x31cff510a1210842ul, 0x4cf31b1b18c63018ul, 0xdbaa592109082742ul, 0x710ddc70f186d576ul, 
   0x8b1c78f98abd797dul, 0xc34d1aecb887ec3dul, 0x8943fb8e9e538c30ul, 0xb1b1b1b020a55575ul, 0x81138cf09267e60dul, 0x4a7fa8c448484210ul, 0x4cf318db18c60c02ul, 0xd7550821210904e8ul, 0xe5c43f53ce53d1b6ul, 0x6440e3c74d2af5e5ul, 0xe434d5b2c59eddcul, 0x6398eee3c3a69d4dul, 0xe636c6c6c6d8b80aul, 0x102036c9e48c4217ul, 0x4c7fa88524210842ul, 0x33cc636363183001ul, 
   0xc5aa810848490921ul, 0x689952bbd8fea962ul, 0x555390fd2a45ab97ul, 0xc78f8db754edbb4dul, 0xbe9bb71c755f5dd2ul, 0xb18d8d9f7d8f9084ul, 0x718aaa10b3e60d8dul, 0x21210840840836c8ul, 0x636333818c7fa485ul, 0x4848668426798363ul, 0xc79b0ed9796ec4c8ul, 0x4a9a6ea932545863ul, 0x8af6dbbc9a6da465ul, 0x8babd5f2cb6fdfa1ul, 0xe3e1f1d2b103f638ul, 0x4cfcc6d8d8c637a1ul, 
   0x42662db21c62550aul, 0xa43fd46290908108ul, 0x3e230636c63189c0ul, 0xf5abd519090cd091ul, 0x4751718e1f1fed3aul, 0xdaac4a9269aac62bul, 0x78ffb4f163b6ed38ul, 0xd8ecacbebf5fad3ful, 0x163a7bbfde1d3a2eul, 0x892113983636c630ul, 0x8408422731729810ul, 0x27229f7f498a4210ul, 0x9092125f98d8d8c6ul, 0xb6ec755ab7528c84ul, 0xaeab10840aba9edbul, 0x8fdb61f29542a36bul, 
   0x7dbbb1fdeef8ff76ul, 0xa65cb8b8bd62a8b5ul, 0x3630190e9e9fc7e3ul, 0xacf80a48580c636ul, 0x498a121081084267ul, 0x636c631998829f7ful, 0x76a438484908497eul, 0xb261c78f1e62e8b9ul, 0x5615d72eaed562b0ul, 0x77c786a3bdb66e39ul, 0xe5eba8b175693e3ful, 0xeee3f6f3aab9797aul, 0x2901831b1b18c198ul, 0x844e636297250c4ul, 0x4fbfa4c509081021ul, 0x667e636318ccc41ul, 
   0xbb2e5da908334204ul, 0xa61c61861c65c944ul, 0x69ed75edd5f5db12ul, 0x3975b531a7dbb7eaul, 0xba69a6af6ee1ffbeul, 0x7e874d26bae9a2c5ul, 0x2920318d8c630628ul, 0x210227318a54f246ul, 0x29f7f498a1084204ul, 0x460bf318d8c19988ul, 0x2355eb75d48419a0ul, 0x46efdc7776e3e408ul, 0x1c3ce9bebc5eaf2dul, 0x1eaac5dddecdbb77ul, 0x6daaaf63f1fa7feeul, 0xc9a658bb2cd0e52bul, 
   0x208520318c6318c6ul, 0x4081084cf98a52e7ul, 0x627229f7f490a108ul, 0x8484180ccf98c630ul, 0x59c52ecb96eda908ul, 0xc8e6d3edc79b0e98ul, 0xbb61a976f6f6eae2ul, 0xa2edc5f2eaa61edbul, 0x754e6d87c3f8fe3cul, 0x4a1d65cbd73ee3e5ul, 0x64844e6318c18db1ul, 0x4204081139831717ul, 0xc4e4621fea310420ul, 0x841830199e63180ul, 0xd554a75b6d74d4a1ul, 0x6cddb0f4f0f90c75ul, 
   0x1c65abdbebeb348ful, 0xbc5f379618dddb76ul, 0x8fdbbc3ff7a7955eul, 0x9368bd736c3c714aul, 0x9f33f318318db14aul, 0x2204081008113e61ul, 0x1830189c0a43fd24ul, 0xb3628108306333ccul, 0x2a8abaaa14189a68ul, 0xbb5513f43bbb66fdul, 0x398eec3a61b75abeul, 0xdcda62d5eafcbc5dul, 0x1fd8e2aac9b774fdul, 0x60c0cf173904dd72ul, 0x9988306304fe1266ul, 0x87fa8c5988100998ul, 
   0xc062730199999814ul, 0x22e5cbd438420c60ul, 0xd8f1e3d802a84a16ul, 0xf53136eb17a84317ul, 0xd36dedf6fad250fdul, 0x52ac4b1c3dddde3ful, 0x62e731942ac727eeul, 0xe29739221480c18cul, 0xcc4cc4ce22730619ul, 0x9999818c7fa48404ul, 0x3204631833399999ul, 0xe4a066352f5f5bb1ul, 0x7550a4a3938cb8b1ul, 0x977567394d3cad5dul, 0x3771eefefee3aeaful, 0x50aa1c98658ab6a5ul, 
   0xa8520c18362a6986ul, 0x33cc18db17218793ul, 0x18ff5108227133f1ul, 0xc6333333cc062703ul, 0xb192edd5e5a4c99cul, 0xa90a109878e1e714ul, 0xb8a994d65daa84a2ul, 0x1f0f7695eb74394cul, 0x8fea613558b55336ul, 0xc18c6638c3d18aa5ul, 0x362e7df4a9552920ul, 0xea21113e21bfcc06ul, 0x18333e603138a29ful, 0x5a6eaead5454ce63ul, 0x45218ddfb0e9884ful, 0xce08bd6eea849885ul, 
   0x3c315b421871d3c5ul, 0x872ba2e6fae31c3cul, 0xd98fd8f263aa7efdul, 0x58c32863149018c6ul, 0xf88333883060318aul, 0x83018021cff51085ul, 0xe5aab33cc63199f9ul, 0x6fdc72c531329d7aul, 0xbcbd62945202ac73ul, 0xed3e3fdc7922e8b8ul, 0xfae6e5dfb4d26c72ul, 0x14a61d3cdc668bc5ul, 0x92318db271fb1943ul, 0xff3018318a5c9440ul, 0x60ccc41d7d0429ful, 0x54250460c63199f3ul, 
   0xbf72ba9a2613558bul, 0xd8c4213643a84beful, 0x1f769c5d9737d62cul, 0x798daea726ddddeul, 0xd369fee32e2f9777ul, 0x60a5c87ec69898e7ul, 0xf3018362b3c0846cul, 0x66723afa0a4bf989ul, 0xd9060c63180ccf30ul, 0xabadb14f6b2e5ea1ul, 0x141c50aaae0ddc7cul, 0x19a2f6fafaed4c42ul, 0xd5da83fbc3f8ffeeul, 0x4f4cd1716a850ebaul, 0x61fa1a7a2aa8c34ful, 0x31b636c0933c052aul, 
   0x6fd042fe627cc060ul, 0xc630667983199883ul, 0x3b5baf5db1333318ul, 0x14ea103776e39aa4ul, 0xf2f9716948218a62ul, 0xddef6fd7fbbc3166ul, 0xaaa872548b972facul, 0x8f2934d8e6da7bb2ul, 0x6c9062996c1fa1c3ul, 0x20ccf980cccc636cul, 0x99c09be820666666ul, 0x9998c63183339831ul, 0xc3d10aa59babcb49ul, 0xd36d34d491e3c3d3ul, 0x1ebb7d7d6aa022d6ul, 0xb97969b6efd7fa7ful, 
   0xe6fde71532a458baul, 0x78f1c356eb96afa1ul, 0x636c6ccf10560ed8ul, 0x2066665bfcc189ccul, 0xcccc06333c09be8ul, 0xb9babd56666318c6ul, 0xb0f87e3f6e30c85dul, 0xcd168b97562ed577ul, 0x7fbe1f2eb3797b42ul, 0xe44ba2c5cb17ac77ul, 0xaf5d5e769eee3c69ul, 0xa48d3e3c3bbb69dbul, 0xc619e0b8a5144314ul, 0x830189fcc230199cul, 0x99983066605dfa09ul, 0x2cd0e666318c6301ul, 
   0x778fdb8c5b6b2e6ful, 0xd62f9756eea5f61ful, 0x36b5eb56a8821cb5ul, 0x75d554ea6a430ff7ul, 0x5d5d58fdbb8fee70ul, 0x937ef1f6c3fdc45ful, 0x14a9a65494a75d53ul, 0xccf999f3339819ful, 0x80c189c0bbf48306ul, 0xeb127318d8c18301ul, 0x7edfb285227add76ul, 0xabb7d71717a9ec78ul, 0xc34dbacd594564aaul, 0xca92aa42062930e1ul, 0x3caeeb8bba471fd8ul, 0x2a9b6a77ef1f6fddul, 
   0xc0cf8a9861e35ac5ul, 0xc180c4f989f333cul, 0xc06030189c5efd2ul, 0x155554220c1b18c6ul, 0x4a1edfb8c49a10f2ul, 0x59f56bd73717966cul, 0x1b39ce8baeaa5c81ul, 0xe87469e70426c717ul, 0x4d9b766d3e34abd6ul, 0xb6c3962aa429b55ful, 0x33398ccf304f8b1ful, 0xce2f7e901830199ful, 0x731b18c60cccccccul, 0x51b6c530f218c522ul, 0xdf5e5bb6ea12c61eul, 0xb69a1ca6911c7d2cul, 
   0x2ec924392e314b92ul, 0xe3870e1eb55db1c5ul, 0x76fd12141d6390f1ul, 0x8c199e6c33c54fd3ul, 0x2f7e9183060305f9ul, 0x631b18c60333999eul, 0x4a4558e739042840ul, 0x3b2f2f2ed355d621ul, 0x2149a684a54123f7ul, 0xa9678c8c54c0a2e7ul, 0x5dec78d30c32a65bul, 0xe13f43a7fb4d2c05ul, 0x60313f983199cc33ul, 0x33305f8bdfa41830ul, 0x14c4222731b18c63ul, 0x534d3550e4f252e5ul, 
   0x20856530dd7add35ul, 0x794aad1c7cadaac5ul, 0x8c2ac59a80a536ccul, 0x7285c0707ea794fdul, 0x3330619fa38feedcul, 0x80c6030099980c63ul, 0x6313e61138bdfa41ul, 0x5dd5310160c63630ul, 0xdae8bb57a1c3fb95ul, 0xc42656b2d5bbaaaaul, 0x873ebb8d2edb10a4ul, 0x7aed4a0c1938d34aul, 0xc0747ec69fbee3e9ul, 0x8678cc71fdfb0f25ul, 0xc02273063183018dul, 0x5f8817bf40603180ul, 
   0x50e4060c6c631830ul, 0xd58f1f1e6f36ebb7ul, 0xbaac5aa989aac5c5ul, 0xc4d562453aab102aul, 0x52d90e3121c87b7eul, 0xfb4f379b45c5c598ul, 0xc7f7ee7043b31c7eul, 0x8c18318678c050c3ul, 0x3060c189c44e631ul, 0xc630627c44e20ffaul, 0xcbb2f6ea99cc6318ul, 0x65f5c5ee871f770ful, 0x459aac526cbb6283ul, 0x4987f6185213055bul, 0x69492aa426438ca1ul, 0x9861fdfbc787acb1ul, 
   0x82fbee3878ca141cul, 0x18318c60c631b630ul, 0x83fe80c18c066710ul, 0x8c6318cccc020100ul, 0xfee98bd66a566731ul, 0x3d6eb8be5eaf3eeeul, 0xd5dd6aed55295539ul, 0x19764fdf72a21721ul, 0xaaa1c1345da9c9c7ul, 0xd1458c3db8f4ff63ul, 0xd8c20e4e3fb0c3ceul, 0x9f1180c6318c6318ul, 0xc40223fe83060c18ul, 0x1673063630630667ul, 0xbf586e3e1e99aeabul, 0x56283f63a562f17cul, 
   0x10e2956b6ab962edul, 0xcdd1869e4b9cc79cul, 0xdb371810a58baac5ul, 0xfbec69a65d159f6dul, 0x18d8d8318c025b31ul, 0x3ffa3018c19cfe63ul, 0xd8c60c60cce20101ul, 0xfefee34a5299e60ul, 0x37eebaf6e6e2f561ul, 0xabaea2ddb75419bul, 0xa7ec6529725c0520ul, 0x3d8b1e422f5abcb0ul, 0xfa18654951fbb61cul, 0x63060c0236729a71ul, 0x6660c673f98c631bul, 0xc666610089c4ffaul, 
   0x3940a489cc1b18c6ul, 0xaedcdf5c5dd886ecul, 0x52aebaa4bed3e3cbul, 0xa2b3c970248292aaul, 0xcbb31c78dd62e2e6ul, 0xf7ed34aa9c1cd870ul, 0xd8c06100ccb8b1c7ul, 0xc18319fcc18d8d8ul, 0x3060ccf102013fe8ul, 0xa19449020c18d8d8ul, 0x68bc5eaeac5eb12ful, 0x5290a41881bb0f8eul, 0xabb4262976288294ul, 0x6fd0cc7ec61765f5ul, 0xfdbb369aaeaa7bb7ul, 0xb1b1830211999764ul, 
   0xfd060319cfe631b1ul, 0x63060c060402270bul, 0x75485c8084ce631bul, 0xf1fec3af2f97775dul, 0x92105082146d830ul, 0xe3e55d6aeae5eb14ul, 0xaae776e3e3912a7eul, 0x99976d18776ecbadul, 0x98d8c636c6333125ul, 0x4cc2ff4060c633ful, 0xccc6d8d8c060cce2ul, 0xdd41575d62cc4084ul, 0x73efb8f9b2cbabf5ul, 0x294aa94c4214b901ul, 0xdc61172e2e2f54c4ul, 0xcdb371e1ca90a7edul, 
   0xa5d8a9c7f7ea3529ul, 0xc60d8d8c06668c94ul, 0xd018306271333033ul, 0xc1830601026620bful, 0x7550a10813398678ul, 0x716af97cb81c6acbul, 0x653c882c74ff70fcul, 0xd5318ab6db69aac7ul, 0xa731fd86ad572edul, 0x314aa18dd9b8ff79ul, 0x41464a52ec58c3f7ul, 0x67304f8306c63180ul, 0x11399ff460c18302ul, 0x4ccd8678c1833388ul, 0xb8e6ababd6221020ul, 0xef4f08bcdf2fd43bul, 
   0x4d8ece4087066c3ful, 0x75d484293aeae8bbul, 0xd9bf71a0ef63c68dul, 0xb8a9c6191d6227eful, 0x631b19989042e4b8ul, 0xfa0318313f9867c0ul, 0xc6c630667c4080cful, 0x9bab36390204cf36ul, 0x5dbd5f2e50f4f0e5ul, 0x75363976eeef1f4ful, 0x64aa3aa4d7750ec8ul, 0x78f054c34d6aea62ul, 0xd34c315426430fe8ul, 0x8c60c80a4ccb8a54ul, 0x30667e6362ec180dul, 0x199c4089c2ffa306ul, 
   0x80409998db1b18cul, 0x50fee1f85cdd59abul, 0x7ddeef1edd79bedful, 0x510aa14d558babecul, 0x3c9546dd4ca79294ul, 0x494a36630c3ce715ul, 0xc24838b9c5d94f25ul, 0x80c1867c4e6d8c60ul, 0x10200bfeb18c1830ul, 0x9f118c6c6c6c199ful, 0x3e9ea8bcdd5aab08ul, 0xe6d36e6f17f5e43cul, 0x98f6b373749bf7a7ul, 0xa48d34f31a7ea683ul, 0x4a5c94b8b94ca2a6ul, 0x4272e63ce0a621c1ul, 
   0xfc20817636361208ul, 0xbfeb6318c4f9830cul, 0x1b1b199981331332ul, 0x375696731813331bul, 0x8bd5f7eff7fb57aful, 0x5f47ddeef95ede2ful, 0xddb769e4c389babful, 0x3172522a998ca80eul, 0xca61a625508636c6ul, 0xb1760c18482b14a5ul, 0x7cc6c63153ce8a62ul, 0x810157ff45363062ul, 0x41831b1b6306678ul, 0x717edc5c4678b841ul, 0xde6fd650f4fddf9dul, 0xbe5870fc7fedd17eul, 
   0xe3ddfde3e4b9d57cul, 0x6297044c3fa1b4d8ul, 0xe538c3ca4c42cf8bul, 0x6d89928090610962ul, 0x7cc6c65a38f1f317ul, 0xc4015bfe8bb1b062ul, 0xc5018318db1b0667ul, 0xbdd329e729974554ul, 0x58eee9fc7d2af378ul, 0xeeff742f57e2f8bful, 0x9fbbc1775edf5c3eul, 0xecdc7aebd790fc7eul, 0x859d08410b94cbb3ul, 0xa0db08d94e30ca18ul, 0x6acbd66994f04218ul, 0x99e63600963c79bful, 
   0x8058bbffa294d8c1ul, 0x8666630cf18c1998ul, 0xe1e3865d176db550ul, 0x9ffb966fd7cbc638ul, 0x7976f97cdfacb6eeul, 0x59bdbcbcdf61f77cul, 0x6fae9bbedff7fd86ul, 0x9fb1e34fdb66e3d7ul, 0x24a314c428214108ul, 0xac562e739236534ful, 0xb865f5baa34f24aaul, 0x3398c082577ef77ful, 0x80c5effa29763183ul, 0x40cccc609f183138ul, 0xfb8e588b57975668ul, 0xf6babc5e6a66c3cdul, 
   0xbf8be2fb630f0fc7ul, 0x57ab4c3fef8f0db9ul, 0xbef87f0fe3bb6e6ful, 0xd87edbf71b737abful, 0x6218a5318e8e6e3dul, 0xe22136724a295755ul, 0x652842aea90eca69ul, 0x87f6fc7f1ab17354ul, 0x9998c4e6c04214edul, 0x89c406c9ffe8a567ul, 0xd6e849ccccd867c1ul, 0xfc7dda6acbd7b737ul, 0xbb375e5f2f29fef0ul, 0xb7abe6f170878ffbul, 0xf37eb863fdeee1abul, 0x7cb376f87f77caf6ul, 
   0xff7fbfbf43dad5f3ul, 0x550c55d753639b7eul, 0xa78298a0669574d3ul, 0xd49c65495554a329ul, 0xc7fbf5ff1ffdaa2cul, 0x7636123630612081ul, 0x83331018bdff45c9ul, 0xcbab752cf300cf8dul, 0xf77c7c6aeb3717dbul, 0xe9f1da6e2e9dfbd3ul, 0x4dbabf2fdbbb9b6eul, 0xd79be5fafbf7bb9bul, 0xbfd7c5dedf8ff6ddul, 0x77ddff7bb4d4bd7aul, 0x6d8a8ea6baea63f7ul, 0x45310a410c536cb3ul, 
   0x959c9704318a364aul, 0x7efdc3f77c3fb66aul, 0x739029098a18462cul, 0x811399bfeb252e2eul, 0xdd5533cc1827c6c1ul, 0x8d5973717abeb972ul, 0x78dd6a861eeeeff7ul, 0xf97c5f974fbefefbul, 0xebdd0c3eefef9d76ul, 0xeafefbeef87c3757ul, 0xfb7fde1cb6dd5fabul, 0xc75b458b57df7c7ul, 0x428310a75162ed4aul, 0x4214a5c0cd18518cul, 0xa7fb87c3feff70e0ul, 0xca1b4d898b8468c3ul, 
   0x7c0d7fd6c1721c70ul, 0xaa167333304f8c66ul, 0xcbcbcdf58b35d96eul, 0xde4eb83db8fc786aul, 0x2fabf6ee96dbbfbcul, 0xfd79f70fbb87d65ful, 0xf4fb87e9ffaae5faul, 0xfdf0f0e22f6f57e6ul, 0x64db6cb9690eeff7ul, 0x428310853a8bb754ul, 0x14b171763334010cul, 0xc3eee1fbb87bbf72ul, 0x1d2af58b4e2e032eul, 0x5f7e829889fb371ful, 0xcf98c33c60ccc425ul, 0xf5716aeb68bb528cul, 
   0x8731ddbb8e3ae5e2ul, 0xfd6a3fbbbc3e9e58ul, 0x7ef8fa7e97b7abe6ul, 0x1fbbd3f5b7d7aba3ul, 0x7f65375edeaf8bfbul, 0x32f5ba2c3efdeef7ul, 0x2678507577588195ul, 0x951884c5c5d80490ul, 0x830f8fc3fdbfef77ul, 0xbfdc17179bcb8ec1ul, 0x3ffa4cb16a95dbbbul, 0xcc1b1b18cce44d16ul, 0x2f54da2c59b13667ul, 0x48b1869a775c5e6ful, 0x5e21a787c3e1f765ul, 0xf0fc7fdb55f2fdbdul, 
   0xef77cabebe5e6fb1ul, 0xdc685cbc5f97863eul, 0xd5d72eedb9dfb87cul, 0xc4315926dd429c74ul, 0x13629763248d8410ul, 0xdfff3ff778e95553ul, 0x5f6f2e380bec7e3ful, 0xa7d77f7ddf776ebful, 0x96aedffa1737178bul, 0x6c17e63631830215ul, 0x5e5ea9b65ab566c5ul, 0x7db0f258a54917adul, 0xaf97cdd50cd87c78ul, 0xcdf2e5b0ffbdbd2ful, 0xc5f961e3ff7db36dul, 0xae6da7c7872c45edul, 
   0x4dd5471d3575cbbbul, 0x11b39c8290c62174ul, 0xc74aaa98a36725d8ul, 0xa58e1eefd7fd7fbbul, 0xfc6aeafedf6e2e38ul, 0x17f5f2e136efdbf1ul, 0xe2345d5e2fa97abful, 0xf58864cfcc6318c4ul, 0xa6ba9562eebd62f2ul, 0x6dd874f0428ca2a9ul, 0x1f53797db9a99b36ul, 0x3d76eaf97937ee1eul, 0x86abdb9baf77787cul, 0xd75eb3421874f761ul, 0x8c5192875d431fb9ul, 0x9738bb6ca7195255ul, 
   0x7f1ff48c42141c5cul, 0x6f6e8fd8e6c3fefaul, 0xfc7feef137ebf57eul, 0x37fafede6f975ef1ul, 0xc1842b75e5f9756ful, 0x5cb3624f89998318ul, 0xa52a8509baf5f5dul, 0x6ee3c650a558a041ul, 0x7c7a76cb9b8b2fbful, 0xee1f85ededddca78ul, 0x7ee42e8bdb8bf7deul, 0x4c32ebaddd4bebb7ul, 0x4dd52145c942aea1ul, 0x34d3cec54e3f7d2cul, 0x1f1eec300a0c6215ul, 0xbf97f59ed3edbbb6ul, 
   0x5f787f0fe6d59bf9ul, 0x7eafabeaf9bedeaeul, 0x633381745e6ff6a5ul, 0xe5bba8702271180cul, 0x89330e0ea2e5f5d5ul, 0x3fb1ec953555425cul, 0xf31f1e6caeab7564ul, 0x177eeef6cd66e6e2ul, 0xc71ed874d45d562eul, 0x923aa44fd29aaed8ul, 0x61fd87da8bb5210bul, 0x110621c9a69e7628ul, 0xfef8ff61fbefa522ul, 0xf7d317ebfd7e2f49ul, 0x378bf2f8b861f87ful, 0x2fd7eaa57ebc5f2ful, 
   0x59c44f98c189c0bbul, 0xa6b3d5d97ae5bb6cul, 0x56226eba994f0289ul, 0x9b37ec35d49534f0ul, 0xb369fecaecbcb8bcul, 0x3fb37ee0a55b4d29ul, 0x2ac73e8729b2e586ul, 0x66dd82c5c5ea6c46ul, 0x26490998e3c83bdbul, 0x4c787ddfd94d8884ul, 0xddf8ff1f49bedeaful, 0xead5ab9bc5f37daeul, 0x27022ededfe2adf6ul, 0xb352b008113980c6ul, 0xb971797aa62a9b68ul, 0x1ed75885ce4b836eul, 
   0x874d2a90e0295727ul, 0xf7d87f5309a2f5deul, 0x56839bf71a08a293ul, 0x9b6aa7a6fda76ebul, 0xf1f6d6b57d737279ul, 0x15938d32c4d0c71ul, 0x3c7c7fa921da2aa3ul, 0xf8fe9fbb16afd7b4ul, 0x7b75755edf5faf10ul, 0x9c08b97dbf856bfdul, 0x6d542ce210180c18ul, 0x9bdbdb8baa172757ul, 0x1f31563b39294555ul, 0x728a421515629cc6ul, 0x2963ca798c34d051ul, 0xdd6ad47b61e71b25ul, 
   0xa7176d497dc7bb8dul, 0xc3d3c2af6e2f74beul, 0x91630e1f26daee63ul, 0x7d2a55094a2a145cul, 0xc7e3f774e5abda1ful, 0x7d63a8b9717d7169ul, 0x27022e5eefe96bfdul, 0xdae9b1271089f306ul, 0xc5f6e2f9692a6936ul, 0xe3e4290e2e7156aaul, 0xa9aac4210c429590ul, 0x52ec9061cdfb4d2bul, 0xab1683f7384c30f2ul, 0x8d1a6c53f69fecaeul, 0xf73613797db8bc87ul, 0xc1db4f8eaab5614ful, 
   0x4310c514a83145c6ul, 0x4870f87cd82d58b5ul, 0x7fae6a55b6db658bul, 0xc0667022d5eefe23ul, 0x2d58b352610204ccul, 0x25d1717966a0e09bul, 0x523319429487176cul, 0xf0e8d96eaa98850ul, 0x870f3822ac638f0ful, 0xc70e3a2ed5b054e3ul, 0xf2fb53769914e8c3ul, 0x92e5aaee3eda6de5ul, 0xa06214a96f5d87cdul, 0xc3555ebab97a84d8ul, 0xb6eaaa7554bed3e3ul, 0xb57bbf88df8bed4aul, 
   0x42ce204cf3018088ul, 0x758a0a4d17ac5cb3ul, 0x534d5626d8b9475dul, 0xd458bd5310a36105ul, 0x124ab18d3e3fdfdeul, 0x30000b8c3bf6b9eul, 0xfdfb8e5441444958ul, 0xfa1e30db69b18848ul, 0x7ebcdf5f5de4427eul, 0xe1fea45863361ed5ul, 0x1a624109b25a3f61ul, 0x3b7e8558bedf5edcul, 0xeddd4a75d528224cul, 0xc1999117afb7f16bul, 0xcb7750ac0804089cul, 0xece4b841d4d16ac5ul, 
   0x945535d76dd6213ul, 0xb371f3134dd558c5ul, 0xdc78f042924a4ddbul, 0xf184dd755558a61eul, 0xc77eca90a61fdc78ul, 0xe8439f79d66f17e5ul, 0xa420923143f69e9ful, 0x6dd5f2f969fb0c3aul, 0xe218c519249861a7ul, 0xe0659bcdf8b5d5f2ul, 0xd3550a2710227cc4ul, 0xe3ce0aa6ba2dd35ul, 0xb16eaa13629538d3ul, 0x55290863aeb69aecul, 0x52164d3f63865031ul, 0x58ea93df61c31548ul, 
   0x36a8e6cdfb0e9ad5ul, 0x309bdbcdeefa9f1eul, 0x52111c7bbc7c72e5ul, 0xd5e7d869176a528cul, 0x14b920b9c6cbdbebul, 0x9bcdf8b58bead503ul, 0x998804cccccc465ul, 0x8529d4d162cd542ul, 0x1362972186987194ul, 0xd172d58b962ed362ul, 0x8b9ca79297266156ul, 0x4cfa1c3357558a0dul, 0x43c7db371f29298cul, 0xbe5f3751e1e9a458ul, 0x987c7f1fa7fa5d68ul, 0x8bd52aebaa631d52ul, 
   0x2c459babeb3d2c65ul, 0xc5d50854f25288c5ul, 0xccc0cb37dbf8a5dbul, 0x75cb56290266227cul, 0x98e34ca5c5d92897ul, 0xb96eda6ab18a0db3ul, 0x2971b600eb2d5abeul, 0xf245cb352014bb6ul, 0xd3dd8645210b90fdul, 0xafd77f78fadaafbeul, 0x7ddf8f8fa32cdcdeul, 0xcb1769b6baac71fful, 0x8bbb12e5269a6eb6ul, 0x4d94cbb05d92a545ul, 0x66f77f8a58bf2eacul, 0x1088133c589cc011ul, 
   0x94a9e434d1b2e5abul, 0xa9a6eaac52c332e2ul, 0x3c1755cb56e9baebul, 0xdd4856c529805b65ul, 0x527e878e1a6b596aul, 0x3ae58639bf71c755ul, 0xcbd7b7b5dbb87e6dul, 0xc63c7d3f8f9b4c3aul, 0x8aba8b562e5cb96eul, 0x71e4a47575d4ec81ul, 0x4dfadd543b25c6caul, 0x3333002f5feff8adul, 0x53c934d4c521099ful, 0x5bb6ab167301729eul, 0x22d5baaa4a3abb2dul, 0x58a5c92084cc6195ul, 
   0xfbf53122f5cbd621ul, 0x96ddb0d26ea7b6edul, 0xa1ddf6f87d1bcbd5ul, 0x87fb2c55d659beb9ul, 0x755ab96ae59bbd87ul, 0xc5534d57261c6549ul, 0xaa1c828bec78e9e0ul, 0xab6e6fd5d1af5ba6ul, 0xa12058c4e603068ul, 0x899cdb14a50a531ul, 0x9e4cb57b73797169ul, 0x71e394dd7af5d1faul, 0x755487173928c624ul, 0x3c3dd878cb1775ebul, 0x5f1b71fec4dd4becul, 0xf43f1ff87fb5e6fdul, 
   0x7da8b355162f97cbul, 0x65ab972dd4e5da78ul, 0x5753754ef7d87282ul, 0xa42954298737eca2ul, 0x3aeb37fbfd01ada6ul, 0x2112406c603060d8ul, 0x4090831725ca2ac4ul, 0x9d62fd79bcdf5eb1ul, 0xf9d9717373ec7f4aul, 0x5099ce4a29881fb0ul, 0x10067635ffcbcb35ul, 0xecac7f97ddeef4f8ul, 0x115693667fdfdfccul, 0x91e2fd9e7757866ul, 0x8606060e0e366e51ul, 0x32337173317e5f9ful, 
   0x601898987f0360ul, 0xbad0d1270cf7b000ul, 0x4e45490000000014ul, 0xa42954826042ae44ul, 


};
const std::size_t data_usmImage_size = 9117;

#endif


