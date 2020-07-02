#include "Sift.hpp"

Sift::Sift(int octave, int scale, float sigma)
    : octave_(octave), scale_(scale), sigma_(sigma) {
        thresh_ = 0.04;
    }

Sift::~Sift() {}


void Sift::Detect(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints){

    cv::Mat output;
    input.convertTo(output, CV_32F);

    image_ = input.clone();

    output = output / 255.0;

    // compute image pyramids
    ComputeImagePyramids(output);

    // compute blurred images
    ComputeBlurredImages();

    // compute DOG voxels
    ComputeDOGVoxels();

    DetectKeyPoints(keypoints);
}

void Sift::ComputeImagePyramids(const cv::Mat& image){
    pyramid_.clear();
    for (int i = 0; i < octave_; i++)
    {
        // resize the image
        cv::Mat downsampled_image;
        if (i == 0){
            downsampled_image = image;
        }else{
            cv::resize(pyramid_.back(), downsampled_image, cv::Size(), 0.5, 0.5);
        }

        pyramid_.emplace_back(downsampled_image);
    }
}


void Sift::ComputeBlurredImages(){
    blurred_images_.clear();

    for (const cv::Mat& image : pyramid_ ){
        std::vector<cv::Mat> blurred_images;
        for (int i = -1; i <= scale_ + 1; i++)
        {
            float sigma = std::pow(2,float(i)/scale_) * sigma_;
            int k = 2*std::ceil(2*sigma) + 1;
            cv::Mat blurred_image;
            cv::GaussianBlur(image, blurred_image, cv::Size(k,k), sigma, sigma);
            blurred_images.emplace_back(blurred_image);
        }
        blurred_images_.emplace_back(blurred_images);
    }
}


void Sift::ComputeDOGVoxels(){
    voxels_.clear();
    for(const auto& octave : blurred_images_){
        std::vector<cv::Mat> dog_layer;
        for (int i = 1; i < octave.size(); i++)
        {
            cv::Mat img_cur, img_prev, dog;
            octave[i].convertTo(img_cur, CV_32F);
            octave[i-1].convertTo(img_prev, CV_32F);
            cv::absdiff(img_cur, img_prev, dog);
            dog_layer.emplace_back(dog);
        }
        voxels_.emplace_back(dog_layer);
    }
}



void Sift::DetectAndCompute(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints, cv::Mat& descriptors){
    // detect keypoints
    Detect(input, keypoints);

    // compute descriptor
    int nkp = kpts_.size();
    std::vector<std::array<int, 128>> out;

    // for all keypoints
    for (int i = 0; i < nkp; i++)
    {
       KPoint kp = kpts_[i];
       cv::Mat blur_img = blurred_images_[kp.oct][kp.blur_no].clone();
       std::array<int, 128> des = GenerateDescriptor(blur_img, kp.y, kp.x);
       out.emplace_back(des);
    }
    descriptors = ToCVDesMat<int>(out);
}


void Sift::DetectKeyPoints(std::vector<cv::KeyPoint> &keypoints){
    keypoints.clear();
    kpts_.clear();
    for(int m = 0; m < voxels_.size() ; m++)
    {
        std::vector<cv::Mat> vox = voxels_[m];
        for (int s = 1; s < vox.size() - 1; s++)
        {
            for (int j = 1; j < vox[s].rows - 1; j++)
            {
                for (int k = 1; k < vox[s].cols - 1; k++)
                {
                    float value = vox[s].at<float>(j,k);

                    if(value >= thresh_ && value > vox[s].at<float>(j+1,k) && value > vox[s].at<float>(j,k+1) && value > vox[s].at<float>(j-1,k) &&
                    value > vox[s].at<float>(j,k-1) && value > vox[s].at<float>(j+1,k+1) && value > vox[s].at<float>(j-1,k-1) &&
                    value > vox[s].at<float>(j-1,k+1) && value > vox[s].at<float>(j+1,k-1) &&

                    value > vox[s-1].at<float>(j+1,k) && value > vox[s-1].at<float>(j,k+1) && value > vox[s-1].at<float>(j-1,k) &&
                    value > vox[s-1].at<float>(j,k-1) && value > vox[s-1].at<float>(j+1,k+1) && value > vox[s-1].at<float>(j-1,k-1) &&
                    value > vox[s-1].at<float>(j-1,k+1) && value > vox[s-1].at<float>(j+1,k-1) && value > vox[s-1].at<float>(j,k) && 

                    value > vox[s+1].at<float>(j+1,k) && value > vox[s+1].at<float>(j,k+1) && value > vox[s+1].at<float>(j-1,k) &&
                    value > vox[s+1].at<float>(j,k-1) && value > vox[s+1].at<float>(j+1,k+1) && value > vox[s+1].at<float>(j-1,k-1) &&
                    value > vox[s+1].at<float>(j-1,k+1) && value > vox[s+1].at<float>(j+1,k-1) && value > vox[s+1].at<float>(j,k)){

                        int x = k*std::pow(2,m);
                        int y = j*std::pow(2,m);
                        float scale = (m*(scale_) + s);
                        keypoints.emplace_back(cv::KeyPoint(x, y, scale, -1, 0, m));
                        kpts_.emplace_back(KPoint(k,j,m,s));
                    }
                }
                
            }
        }
    }
}