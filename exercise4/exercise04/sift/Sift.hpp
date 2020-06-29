#ifndef SIFT_HPP
#define SIFT_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>


struct KPoint{
    KPoint(){
        x = 0;
        y = 0;
        oct = 0;
        blur_no = 0;
    }
    KPoint(float _x, float _y, int _oct, int _blur_no){
        x = _x;
        y = _y;
        oct = _oct;
        blur_no = _blur_no;
    }
    float x;
    float y;
    int oct;
    int blur_no;
};

class Sift {
    public:
        Sift(int octave = 5, int scale = 3, float sigma = 1.0);
        ~Sift();

        void Detect(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints);
        void DetectAndCompute(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints, cv::Mat& descriptors);

    private:
        void DetectKeyPoints(std::vector<cv::KeyPoint> &keypoints);

        void ComputeImagePyramids(const cv::Mat& image);
        void ComputeBlurredImages();
        void ComputeDOGVoxels();
        cv::Mat ComputeDescriptor(const cv::Mat& image, const KPoint& kp); 
        cv::Mat GenerateDescriptor(const cv::Mat& mag, const cv::Mat& ang);

        int octave_;
        int scale_;
        float sigma_;
        float thresh_;
        std::vector< std::vector<cv::Mat> > voxels_;
        std::vector<std::vector<cv::Mat>> blurred_images_;
        std::vector<cv::Mat> pyramid_;
        std::vector<KPoint> kpts_;
};

#endif // !SIFT_HPP