#ifndef SIFT_HPP
#define SIFT_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "Timer.hpp"


struct SSD{
  SSD(float sc, int disp): score(sc), disparity(disp){}
  float score;
  int disparity;
};

class Stereo {
public:
  Stereo(cv::Mat K, float patch_radius, float baseline, float min_disp, float max_disp);
  ~Stereo();

  cv::Mat GetDisparity(cv::Mat left_image, cv::Mat right_image);

private:

    std::vector<SSD> GetDispArray( cv::Mat left_patch,  cv::Mat right_patch, int index);

    float Ssd(const cv::Mat &im1, const cv::Mat &im2);

    cv::Mat k_;
    float patch_radius_;
    float baseline_;
    float min_disp_;
    float max_disp_;
};

#endif // !SIFT_HPP