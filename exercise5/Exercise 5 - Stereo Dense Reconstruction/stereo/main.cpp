#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <string>
#include "Timer.hpp"
#include "Stereo.hpp"


int main(int argc, const char* argv[])
{
    float scale = 0.5;
    float kmat[] = {7.188560000000e+02, 0, 6.071928000000e+02, 
                    0, 7.188560000000e+02, 1.852157000000e+02,
                    0, 0, 1};

    cv::Mat K(3,3, CV_32F, kmat);

    float baseline = 0.54;
    float patch_radius = 5;
    float min_disp = 5;
    float max_disp = 50;

    std::shared_ptr<Stereo> stereo = std::make_shared<Stereo>(K, patch_radius, baseline, min_disp, max_disp);

    cv::namedWindow("disparity",cv::WINDOW_KEEPRATIO);


    for (int i = 0 ; i <= 99 ; i++){
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << i << ".png";
        std::string left_path = "../data/left/" + ss.str();
        std::string right_path = "../data/right/" + ss.str();
        cv::Mat left_img = cv::imread(left_path, 0);
        cv::resize(left_img, left_img, cv::Size(), scale, scale);
        
        cv::Mat right_img = cv::imread(right_path, 0);
        cv::resize(right_img, right_img, cv::Size(), scale, scale);

        Timer t;
        cv::Mat img = stereo->GetDisparity(left_img, right_img);
        std::cout << t.TimeElapsed() << std::endl;

        cv::imshow("disparity", img);
        int k = cv::waitKey(1);
        if (k == 32){
            break;
        }
    }

    return 0;
}