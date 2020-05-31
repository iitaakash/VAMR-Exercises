#ifndef SIFT_HPP
#define SIFT_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
#include "Timer.hpp"
#include "Points.hpp"


struct SSD{
  SSD(const float& sc, const int& disp): score(sc), disparity(disp){}
  float score;
  int disparity;
};

class Stereo {
public:
  Stereo(const Eigen::Matrix3f& K, const float& patch_radius,const float& baseline,const float& min_disp,const float& max_disp);
  ~Stereo();

  cv::Mat GetDisparity(const cv::Mat& left_image, const cv::Mat& right_image);

  Points* GetPointCloud(const cv::Mat& left_image, const cv::Mat& disparity);

private:

    inline std::vector<SSD> GetDispArray(const cv::Mat& left_patch, const cv::Mat& right_patch,const int& index);

    inline float Ssd(const cv::Mat &im1, const cv::Mat &im2);

    Eigen::Matrix3f k_;
    Eigen::Matrix3f kinv_;
    Points* points_;
    float patch_radius_;
    float baseline_;
    float min_disp_;
    float max_disp_;
    int patch_size_;
    int patch_count_;
};

#endif // !SIFT_HPP