#include "Sift.hpp"

Sift::Sift(int octave, int scale, float sigma)
    : octave_(octave), scale_(scale), sigma_(sigma) {}

Sift::~Sift() {}


void Sift::Detect(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints){
    voxels_.clear();

    // cv::Mat output = input.clone();
    cv::Mat output;
    input.convertTo(output, CV_32F);


    // generate octaves
    for (size_t i = 0; i < octave_; i++)
    {
        // resize the image
        float n = std::pow(2,i);
        cv::Mat downsampled_image;
        cv::resize(output, downsampled_image, cv::Size(), 1/n, 1/n);

        std::vector<cv::Mat> blurred;
        std::vector<cv::Mat> dog_voxel;
    
        GenerateOctaveData(downsampled_image, blurred, dog_voxel);
        voxels_.emplace_back(dog_voxel);
    }

    // supress the points
    for(auto& oct : voxels_){
        for(auto& im : oct){
            im.setTo(0.0, im < 0.04);
        }
    }

    DetectKeyPoints(keypoints);

    // keypoints.emplace_back(cv::KeyPoint(100,100,50));
}


void DetectAndCompute(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints, cv::Mat& descriptors){
    // detect keypoints
    Detect(input, keypoints);

    // for all keypoints
    
}


void Sift::DetectKeyPoints(std::vector<cv::KeyPoint> &keypoints){
    for(size_t m = 0; m < octave_ ; m++)
    {
        auto vox = voxels_[m];
        // std::cout << __LINE__ << std::endl;
        for (size_t i = 1; i < scale_ + 1; i++)
        {

            // std::cout << vox[i] << std::endl;
            for (size_t j = 1; j < vox[i].rows - 1; j++)
            {
                for (size_t k = 1; k < vox[i].cols - 1; k++)
                {
                    float value = vox[i].at<float>(j,k);
                    value = value - 0.04;
                    if(value > vox[i].at<float>(j+1,k) && value > vox[i].at<float>(j,k+1) && value > vox[i].at<float>(j-1,k) &&
                    value > vox[i].at<float>(j,k-1) && value > vox[i].at<float>(j+1,k+1) && value > vox[i].at<float>(j-1,k-1) &&
                    value > vox[i].at<float>(j-1,k+1) && value > vox[i].at<float>(j+1,k-1) &&

                    value > vox[i-1].at<float>(j+1,k) && value > vox[i-1].at<float>(j,k+1) && value > vox[i-1].at<float>(j-1,k) &&
                    value > vox[i-1].at<float>(j,k-1) && value > vox[i-1].at<float>(j+1,k+1) && value > vox[i-1].at<float>(j-1,k-1) &&
                    value > vox[i-1].at<float>(j-1,k+1) && value > vox[i-1].at<float>(j+1,k-1) && value > vox[i-1].at<float>(j,k) && 

                    value > vox[i+1].at<float>(j+1,k) && value > vox[i+1].at<float>(j,k+1) && value > vox[i+1].at<float>(j-1,k) &&
                    value > vox[i+1].at<float>(j,k-1) && value > vox[i+1].at<float>(j+1,k+1) && value > vox[i+1].at<float>(j-1,k-1) &&
                    value > vox[i+1].at<float>(j-1,k+1) && value > vox[i+1].at<float>(j+1,k-1) && value > vox[i+1].at<float>(j,k)){
                        // true
                        int x = k*std::pow(2,m);
                        int y = j*std::pow(2,m);
                        float scale = (m*(scale_) + i);
                        keypoints.emplace_back(cv::KeyPoint(x, y, scale, -1, 0, m));
                    }
                }
                
            }
        }
    }
}


void Sift::GenerateOctaveData(cv::Mat image, std::vector<cv::Mat>& blurred_images, std::vector<cv::Mat>& dog_voxel){
    blurred_images.clear();
    dog_voxel.clear();
    for (int i = -1; i <= scale_ + 1; i++)
    {
        float sigma = std::pow(2,float(i)/scale_) * sigma_;
        cv::Mat blurred_image;
        cv::GaussianBlur(image, blurred_image, cv::Size(0,0), sigma, sigma);
        if (i > -1){
            // cv::Mat dog = blurred_image -  blurred_images.back();
            cv::Mat dog;
            cv::absdiff( blurred_image , blurred_images.back(), dog);
            dog_voxel.emplace_back(dog);
        }
        blurred_images.emplace_back(blurred_image);
        // cv::imshow("image", blurred_image);
        // cv::waitKey(0);
    }
    
}
