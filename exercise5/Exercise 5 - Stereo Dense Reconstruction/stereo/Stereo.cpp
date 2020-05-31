#include "Stereo.hpp"

Stereo::Stereo(cv::Mat K, float patch_radius, float baseline, float min_disp,
               float max_disp)
    : k_(K), patch_radius_(patch_radius), baseline_(baseline),
      min_disp_(min_disp), max_disp_(max_disp) {}

Stereo::~Stereo() {}

cv::Mat Stereo::GetDisparity(cv::Mat left_image,
                            cv::Mat right_image) {

  left_image.convertTo(left_image, CV_32F);
  right_image.convertTo(right_image, CV_32F);

  // 620 188
  cv::Mat out = cv::Mat::zeros(left_image.rows, left_image.cols, CV_8UC1);  

   
                           

  for (int j = patch_radius_; j < left_image.rows - patch_radius_ - 1; j++) {
        cv::Mat right_patch = right_image(cv::Range(j - patch_radius_, j + patch_radius_ + 1),cv::Range::all());
    for (int i = patch_radius_ + max_disp_ ; i < left_image.cols - patch_radius_ -1; i++) {

      cv::Mat left_patch =
          left_image(cv::Range(j - patch_radius_, j + patch_radius_ + 1),
           cv::Range(i - patch_radius_, i + patch_radius_ + 1));

      
      std::vector<SSD> disp_array = GetDispArray(left_patch, right_patch, i);

      // find min score for disparray
      float sc = std::numeric_limits<float>::max();
      int disp_ind = 0;
      for (int i = 0; i < disp_array.size(); i++)
      {
          if (disp_array[i].score < sc){
              sc = disp_array[i].score;
              disp_ind = disp_array[i].disparity;
          }
          /* code */
      }
      // assign index to image
      out.at<uchar>(j,i) = disp_ind * 5;

    }
  }
  return out;
}

std::vector<SSD> Stereo::GetDispArray(cv::Mat left_patch,
                                        cv::Mat right_patch, int index) {
  std::vector<SSD> out;

  // left pass
  int start_index = std::max(patch_radius_, index - max_disp_);
  int end_index = std::max(patch_radius_, index - min_disp_);

  for (int i = start_index ; i <= end_index; i++)
  {
    cv::Mat right_crop_patch = right_patch(cv::Range::all(), cv::Range(i - patch_radius_, i + patch_radius_ + 1));
    Timer t;
    t.Start(); 
    float score = Ssd(left_patch, right_crop_patch);
    std::cout << t.TimeElapsed() << std::endl;
    out.emplace_back(SSD(score, std::abs(float(i - index)))); 
  }

  // // right pass
  // start_index = std::min(right_patch.cols - patch_radius_ - 1, index + min_disp_);
  // end_index = std::min(right_patch.cols - patch_radius_ - 1, index + max_disp_);

  // for (int i = start_index ; i <= end_index; i++)
  // {
  //   cv::Mat right_crop_patch = right_patch(cv::Range::all(), cv::Range(i - patch_radius_, i + patch_radius_ + 1));
  //   float score = Ssd(left_patch, right_crop_patch);
  //   out.emplace_back(SSD(score, std::abs(float(i - index)))); 
  // }
  
  // for (size_t i = patch_radius_ + min_disp_ ; i < right_patch.cols - patch_radius_; i++) {
  //   cv::Mat right = right_patch(cv::Range::all(), cv::Range(i - patch_radius_, i + patch_radius_ + 1));
  //   float score = Ssd(left_patch, right);
  //   // std::cout << left_image.cols << "  " << left_image.rows << std::endl;
  //   // std::cout << right.cols << "  " << right.rows << std::endl;
  //   out.emplace_back(score);
  // }
  return out;
}

float Stereo::Ssd(const cv::Mat &im1, const cv::Mat &im2){
    cv::Mat out, outsq;
    cv::absdiff(im1,im2,out);
    cv::multiply(out,out,outsq);
    cv::Scalar out1 = cv::sum(outsq);
    return out1[0];
}