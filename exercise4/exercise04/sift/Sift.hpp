#ifndef SIFT_HPP
#define SIFT_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>


class Sift {
    public:
        Sift(int octave = 5, int scale = 3, float sigma = 1.6);
        ~Sift();

        void Detect(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints);
        void DetectAndCompute(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints, cv::Mat& descriptors);

    private:
        void DetectKeyPoints(std::vector<cv::KeyPoint> &keypoints);
        void GenerateOctaveData(cv::Mat image, std::vector<cv::Mat>& blurred_images, std::vector<cv::Mat>& dog_voxel);
        int octave_;
        float scale_;
        float sigma_;

        std::vector< std::vector<cv::Mat> > voxels_;
};

#endif // !SIFT_HPP