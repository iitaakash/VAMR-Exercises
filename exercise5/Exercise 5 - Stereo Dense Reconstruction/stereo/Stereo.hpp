#ifndef SIFT_HPP
#define SIFT_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "Timer.hpp"


struct SSD{
  SSD(const float& sc, const int& disp): score(sc), disparity(disp){}
  float score;
  int disparity;
};

class Stereo {
public:
  Stereo(const cv::Mat& K, const float& patch_radius,const float& baseline,const float& min_disp,const float& max_disp);
  ~Stereo();

  cv::Mat GetDisparity(const cv::Mat& left_image, const cv::Mat& right_image);

private:

    inline std::vector<SSD> GetDispArray(const cv::Mat& left_patch, const cv::Mat& right_patch,const int& index);

    inline float Ssd(const cv::Mat &im1, const cv::Mat &im2);

    cv::Mat k_;
    float patch_radius_;
    float baseline_;
    float min_disp_;
    float max_disp_;
    int patch_size_;
    int patch_count_;
};

#endif // !SIFT_HPP