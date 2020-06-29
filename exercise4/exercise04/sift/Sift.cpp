#include "Sift.hpp"

Sift::Sift(int octave, int scale, float sigma)
    : octave_(octave), scale_(scale), sigma_(sigma) {
        thresh_ = 0.04;
    }

Sift::~Sift() {}


void Sift::Detect(const cv::Mat &input, std::vector<cv::KeyPoint> &keypoints){

    cv::Mat output;
    input.convertTo(output, CV_32F);

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
    descriptors = cv::Mat::zeros(cv::Size(128, nkp), CV_8U);

    // for all keypoints
    for (int i = 0; i < nkp; i++)
    {
       KPoint kp = kpts_[i];
       cv::Mat blur_img = blurred_images_[kp.oct][kp.blur_no];

       if(kp.x - 7 < 0 || kp.x + 9 >= blur_img.cols || kp.y - 7 < 0 || kp.y + 9 >= blur_img.rows){
           continue;
       }

       cv::Mat des = ComputeDescriptor(blur_img, kp);
       descriptors.row(i) = des.clone(); 
    }
    
}

cv::Mat Sift::ComputeDescriptor(const cv::Mat& image, const KPoint& kp){

    cv::Mat dx, dy;
    cv::Sobel(image, dx, CV_32F, 1,0);
    cv::Sobel(image, dy, CV_32F, 0,1);

    cv::Mat ang, mag;
    cv::cartToPolar(dx, dy, mag, ang);

    std::cout << mag(cv::Range(kp.y - 7,  kp.y + 9), cv::Range(kp.x - 7,  kp.x + 9)).size()<< std::endl;

    cv::Mat patch_mag = mag(cv::Range(kp.y - 7,  kp.y + 9), cv::Range(kp.x - 7,  kp.x + 9));
    cv::Mat patch_ang = ang(cv::Range(kp.y - 7,  kp.y + 9), cv::Range(kp.x - 7,  kp.x + 9));
    
    return GenerateDescriptor(patch_mag, patch_ang);

    return cv::Mat::ones(cv::Size(128,1), CV_8U);
}

float normal(float mu, float sig, float x){
    const float pi = 3.14159265358;
    return (1.0 / std::sqrt(2 * pi * sig)) * std::exp(-1 * (std::pow(x - mu ,2) / (2 * std::pow(sig,2)) ));
}

cv::Mat Sift::GenerateDescriptor(const cv::Mat& mag, const cv::Mat& ang){
    cv::Mat mag_scale = mag.clone();
    for (size_t i = 0; i < mag.cols; i++)
    {
        for (size_t j = 0; i < mag.rows; i++)
        {
            float distance = std::sqrt((j - 8.0) * (j - 8.0) + (i - 8.0) * (i - 8.0));
            float gauss = normal(0, 24.0, distance);
            mag_scale.at<float>(j,i) = mag_scale.at<float>(j,i) * gauss;
        }
        
    }
    
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