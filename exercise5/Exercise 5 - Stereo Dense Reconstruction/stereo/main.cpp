#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include "Timer.hpp"
#include "Stereo.hpp"


int main(int argc, const char* argv[])
{
    float scale = 0.5;
    cv::Mat left_img = cv::imread("../data/left/000000.png", 0);
    cv::resize(left_img, left_img, cv::Size(), scale, scale);
    
    cv::Mat right_img = cv::imread("../data/right/000000.png", 0);
    cv::resize(right_img, right_img, cv::Size(), scale, scale);

    float kmat[] = {7.188560000000e+02, 0, 6.071928000000e+02, 
                    0, 7.188560000000e+02, 1.852157000000e+02,
                    0, 0, 1};

    cv::Mat K(3,3, CV_32F, kmat);

    float baseline = 0.54;
    float patch_radius = 5;
    float min_disp = 5;
    float max_disp = 50;

    std::shared_ptr<Stereo> stereo = std::make_shared<Stereo>(K, patch_radius, baseline, min_disp, max_disp);

    Timer t;
    cv::Mat img = stereo->GetDisparity(left_img, right_img);
    std::cout << t.TimeElapsed() << std::endl;
    cv::imshow("disp", img);
    cv::waitKey(0);

    return 0;
}