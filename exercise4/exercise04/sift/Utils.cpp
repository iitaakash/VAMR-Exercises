#include "Utils.hpp"

// checking boundary condition
bool IsValidPt(const cv::Mat& in,const int& x, const int& y){
    if(x - 7 < 0 || x + 9 >= in.rows || y - 7 < 0 || y + 9 >= in.cols){
           return false;
    }
    return true;
}

// x - row
// y - col
// get patches
bool GetPatch(const cv::Mat& in,const int& x, const int& y, int size, cv::Mat& out){
    if(!IsValidPt(in, x, y)){
        return false;
    }
    out = in(cv::Range(x - size + 1,  x + size + 1), cv::Range(y - size + 1,  y + size + 1));
    return true;
}

// get image gradients
void GetImageGradients(const cv::Mat& image, cv::Mat& mag, cv::Mat& ang){
    cv::Mat dx, dy;
    cv::Sobel(image, dx, CV_32F, 1,0);
    cv::Sobel(image, dy, CV_32F, 0,1);
    // dx.convertTo(dx, CV_32F);
    // dy.convertTo(dy, CV_32F);
    cv::cartToPolar(dx, dy, mag, ang);
}

float NormalFn(float mu, float sig, float x){
    const float pi = 3.14159265358;
    return (1.0 / std::sqrt(2 * pi * sig * sig)) * std::exp(-1 * (std::pow(x - mu ,2) / (2 * sig * sig)) );
}

void MultiplyGaussian(cv::Mat& image, float mu, float sig){
    float center = (float)image.rows / 2.0;
    for (size_t i = 0; i < image.rows; i++)
    {
        for (size_t j = 0; j < image.cols; j++)
        {   
            float distance = std::sqrt(std::pow((float)i - center, 2) +  std::pow((float)j - center, 2));
            image.at<float>(i,j) = image.at<float>(i,j) * NormalFn(mu, sig, distance);
        }
    }
}

void GetAngle(cv::Mat& image){
    for (size_t i = 0; i < image.rows; i++)
    {
        for (size_t j = 0; j < image.cols; j++)
        {
            image.at<float>(i,j) = std::floor((image.at<float>(i,j) / 6.28318) * 8.0);
            if (int(image.at<float>(i,j)) == 8){
                image.at<float>(i,j) = 7;
            }
        }
    }
    image.convertTo(image, CV_8U);
}

std::array<int, 8> GetCode(cv::Mat mag, cv::Mat ang){
    std::array<float, 8> out;
    for (size_t i = 0; i < 8; i++)
    {
        out[i] = 0.0;
    }

    for (size_t i = 0; i < mag.rows; i++)
    {
        for (size_t j = 0; j < mag.cols; j++)
        {
            int angle = ang.at<uchar>(i,j);
            float magni = mag.at<float>(i,j);
            out[angle] = out[angle] + magni;
        }
    }

    std::array<int, 8> out1;
    for (size_t i = 0; i < 8; i++)
    {
        if(out[i] > 255.0){
            out[i] = 255.0;
        }
        out1[i] = out[i];
    }

    return out1;
}

std::array<int, 128> GenerateDescriptor(cv::Mat img, int x, int y){

    cv::Mat img1;
    img = img * 255.0;
    img.convertTo(img1, CV_8U);
    std::array<int, 128> out;

    // check boundary conditions
    if(!IsValidPt(img, x, y)){
        out.fill(0);
        return out;
    }

    // get gradients
    cv::Mat mag_image, ang_image, mag_patch, ang_patch;
    GetImageGradients(img, mag_image, ang_image);

    // get patches
    int patch_size = 8;
    GetPatch(mag_image, x, y, patch_size, mag_patch);
    GetPatch(ang_image, x, y, patch_size, ang_patch);

    // // for debug
    // cv::Mat ppatch;
    // GetPatch(img, x, y, patch_size, ppatch);
    // ppatch.convertTo(ppatch, CV_8U);
    // cv::circle(img1, cv::Point(y,x),16, cv::Scalar(255,255,255));
    // cv::imshow("ii1", img1);
    // cv::imshow("iimage", ppatch);
    // cv::waitKey(0);

    // std::cout << mag_patch << std::endl;
    // std::cout << ang_patch << std::endl;

    // MultiplyGaussian(mag_patch, 0, 24);
    GetAngle(ang_patch);

    // std::cout << ang_patch << std::endl;


    // std::cout << mag_patch << std::endl;

    size_t index = 0;
    for (size_t i = 0; i < 16; i = i + 4)
    {
        for (size_t j = 0; j < 16; j = j + 4)
        {
            cv::Mat ang_bin, mag_bin;
            ang_bin = ang_patch(cv::Range(i,  i + 4), cv::Range(j , j + 4));
            mag_bin = mag_patch(cv::Range(i,  i + 4), cv::Range(j , j + 4));
            std::array<int, 8> m = GetCode(mag_bin, ang_bin);
            for (size_t k = 0; k < 8; k++)
            {
                out[index] = m[k];
                index++;
            }
        }
    }
    for (size_t k = 0; k < 128; k++)
    {
        std::cout << out[k] << " , ";
    }
    std::cout << std::endl;

    return out;
}