#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include "Timer.hpp"
#include "Stereo.hpp"
#include "Points.hpp"


int main(int argc, const char* argv[])
{
    float scale = 0.5;
    // float kmat[] = {7.188560000000e+02, 0, 6.071928000000e+02, 
    //                 0, 7.188560000000e+02, 1.852157000000e+02,
    //                 0, 0, 1};

    Eigen::Matrix3f K;

    K << 7.188560000000e+02, 0, 6.071928000000e+02, 
        0, 7.188560000000e+02, 1.852157000000e+02,
        0, 0, 1;

    float baseline = 0.54;
    float patch_radius = 5;
    float min_disp = 5;
    float max_disp = 50;

    std::shared_ptr<Stereo> stereo = std::make_shared<Stereo>(K, patch_radius, baseline, min_disp, max_disp);

    cv::namedWindow("disparity",cv::WINDOW_KEEPRATIO);


    for (int i = 1 ; i <= 99 ; i++){
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << i << ".png";
        std::string left_path = "../data/left/" + ss.str();
        std::string right_path = "../data/right/" + ss.str();
        std::cout << "Frame : " << i << std::endl;
        cv::Mat left_img = cv::imread(left_path, 0);
        cv::resize(left_img, left_img, cv::Size(), scale, scale);
        
        cv::Mat right_img = cv::imread(right_path, 0);
        cv::resize(right_img, right_img, cv::Size(), scale, scale);

        Timer t;
        cv::Mat disp_img = stereo->GetDisparity(left_img, right_img);
        std::cout << t.TimeElapsed() << std::endl;

        // cv::imshow("left_image", left_img);
        cv::imshow("disparity", 5*disp_img);
        int k = cv::waitKey(0);
        if (k == 32){
            break;
        }

        Points* p1 = stereo->GetPointCloud(left_img, disp_img);
        p1->Save("my_points.ply");
        break;
    }

    return 0;
}