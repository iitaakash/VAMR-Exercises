#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>


template <typename T>
cv::Mat ToCVDesMat(const std::vector<std::array<T, 128>>& val){
    int n = val.size();

    cv::Mat out = cv::Mat::zeros(cv::Size(128, n), CV_8U);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; i < 128; i++)
        {
            out.at<uchar>(i,j) =  val[i][j];
        }
        
    }
    return out;
}

// checking boundary condition
bool IsValidPt(const cv::Mat& in,const int& x, const int& y);

// x - row
// y - col
// get patches
bool GetPatch(const cv::Mat& in,const int& x, const int& y, cv::Mat& out);

// get image gradients
void GetImageGradients(const cv::Mat& image, cv::Mat& mag, cv::Mat& ang);

float NormalFn(float mu, float sig, float x);

std::array<int, 128> GenerateDescriptor(const cv::Mat img, int x, int y);


#endif // !UTILS_HPP