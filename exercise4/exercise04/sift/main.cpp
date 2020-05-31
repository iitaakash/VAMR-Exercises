#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/xfeatures2d.hpp> //Thanks to Alessandro
#include <memory>

#include "Sift.hpp"

int main(int argc, const char* argv[])
{

    cv::Mat data = cv::imread("../img_1.jpg", 0); //Load as grayscale

    cv::Mat input;
    cv::resize(data, input, cv::Size(), 0.2, 0.2);

    std::cout << input.cols << " " << input.rows << std::endl;

    std::shared_ptr<Sift> sift = std::make_shared<Sift>();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat des;
    sift->Detect(input, keypoints);
    // sift->DetectAndCompute(input, keypoints, des);

    // cv::Ptr<cv::xfeatures2d::SIFT> siftPtr = cv::xfeatures2d::SIFT::create();
    // siftPtr->detect(input, keypoints);
    // siftPtr->detectAndCompute(input,cv::noArray(),keypoints,des);

    std::cout << keypoints.size() << std::endl;
    // std::cout << des.cols << " " << des.rows << std::endl;

    // Add results to image and save.
    cv::Mat output;
    // cv::drawKeypoints(input, keypoints, output, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(input, keypoints, output);
    cv::imshow("sift", output);

    cv::waitKey(0);

    return 0;
}