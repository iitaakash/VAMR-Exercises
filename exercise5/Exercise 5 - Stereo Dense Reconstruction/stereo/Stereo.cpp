#include "Stereo.hpp"

Stereo::Stereo(const Eigen::Matrix3f& K,const float& patch_radius,const float& baseline,const float& min_disp,
              const float& max_disp)
    : k_(K), patch_radius_(patch_radius), baseline_(baseline),
      min_disp_(min_disp), max_disp_(max_disp) {
        patch_size_ = 2*patch_radius_ + 1;
        patch_count_ = patch_size_ * patch_size_;
        kinv_ = k_.inverse();
        points_ = new Points();
      }

Stereo::~Stereo() {
  delete points_;
}

cv::Mat Stereo::GetDisparity(const cv::Mat& left_img,
                            const cv::Mat& right_img) {
  
  Timer t;
  float tot_time = 0.0;
  float disp_time = 0.0;
  cv::Mat left_image, right_image;
  left_img.convertTo(left_image, CV_32F);
  right_img.convertTo(right_image, CV_32F);

  // 620 188
  cv::Mat out = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32FC1);

  // // to be deleted!
  // cv::Mat right_patch = cv::Mat::zeros(patch_size_, left_image.cols, CV_32F); 
  // cv::Mat left_patch = cv::Mat::zeros(patch_size_, patch_size_, CV_32F);                   

  for (int j = patch_radius_; j < left_image.rows - patch_radius_ - 1; j++) {

    cv::Mat right_patch = right_image(cv::Range(j - patch_radius_, j + patch_radius_ + 1),cv::Range::all());

    for (int i = patch_radius_ + max_disp_ ; i < left_image.cols - patch_radius_ - 1; i++) {

      cv::Mat left_patch = left_image( cv::Range(j - patch_radius_, j + patch_radius_ + 1), cv::Range(i - patch_radius_, i + patch_radius_ + 1));

      // std::vector<SSD> disp_array;
      t.Start();
      std::vector<SSD> disp_array = GetDispArray(left_patch, right_patch, i);
      tot_time += t.TimeElapsed();

      // find min score for disparray
      float score1 = std::numeric_limits<float>::max();
      int disp_min = 0;
      int index_ssd = 0;;
      float score2 = score1;
      float score3 = score1;
      for (int i = 0; i < disp_array.size(); i++)
      {
          if (disp_array[i].score < score1){
            score3 = score2;
            score2 = score1;
            score1 = disp_array[i].score;
            disp_min = disp_array[i].disparity;
            index_ssd = i;
          }else if(disp_array[i].score < score2){
            score3 = score2;
            score2 = disp_array[i].score;
          }else if(disp_array[i].score < score3){
            score3 = disp_array[i].score;
          }
      }

      // outlier rejection
      float thresh = 1.2247449;
      if (score2 <= thresh*score1 && score3 <= thresh*score1){
        continue;
      }
      if(disp_min <= min_disp_ || disp_min >= max_disp_){
        continue;
      }
      if (index_ssd == 0 || index_ssd == disp_array.size() - 1){
        continue;
      }

      std::vector<float> X = {float(disp_array[index_ssd-1].disparity), float(disp_array[index_ssd].disparity), float(disp_array[index_ssd+1].disparity)};
      std::vector<float> Y = {disp_array[index_ssd-1].score, disp_array[index_ssd].score, disp_array[index_ssd+1].score};


      float d = SubDisparity(X,Y);

      // std::cout << d  << " and " << disp_min << std::endl;

      out.at<float>(j,i) = d;

    }
  }
  std::cout << "time for inner: " << tot_time << std::endl;
  std::cout << "time for disp: " << disp_time << std::endl;
  return out;
}

inline std::vector<SSD> Stereo::GetDispArray(const cv::Mat& left_patch,
                                        const cv::Mat& right_patch, const int& index) {
  std::vector<SSD> out;

  // left pass
  int start_index = std::max(patch_radius_, index - max_disp_);
  int end_index = std::max(patch_radius_, index - min_disp_);

  // to be deleted!
  // cv::Mat right_crop_patch = cv::Mat::zeros(patch_size_, patch_size_, CV_32F);

  for (int i = start_index ; i <= end_index; i++)
  {
    cv::Mat right_crop_patch = right_patch(cv::Range::all(), cv::Range(i - patch_radius_, i + patch_radius_ + 1));
    float score = Ssd(left_patch, right_crop_patch);
    out.emplace_back(SSD(score, std::abs(float(i - index)))); 
  }

  return out;
}

inline float Stereo::Ssd(const cv::Mat &im1, const cv::Mat &im2){
    // return 0.0;
    float score = 0.0;
    for (int i = 0; i < im1.rows; i++)
    {
      const float* im1_data = im1.ptr<float>(i);
      const float* im2_data = im2.ptr<float>(i);
      for (int j = 0; j < im1.cols; j++)
      {
        score += std::abs(im1_data[j] - im2_data[j]);
      }
    }
    // for (int i = 0; i < im1.cols; i++)
    // {
    //   for (int j = 0; j < im1.rows; j++)
    //   {
    //     score += std::abs(im1.at<float>(j,i) - im2.at<float>(j,i));
    //   }
    // }
    return score;
}


Points* Stereo::GetPointCloud(const cv::Mat& left_image, const cv::Mat& disparity){
    
    for (int i = 0; i < left_image.rows; i++)
    {
      for (int j = 0; j < left_image.cols; j++)
      {
        float disp = disparity.at<float>(i,j);

        if (disp >= min_disp_){
            float depth = (baseline_ * k_(0,0))/disp;
            Eigen::Vector3f pt3d = depth * kinv_ * Eigen::Vector3f(j,i,1.0);
            points_->AddPoint(Point(pt3d(0),pt3d(1),pt3d(2),left_image.at<uchar>(i,j)));
        }
      }
      
    }
    return points_;
    
}

float Stereo::SubDisparity(const std::vector<float>& x, const std::vector<float>& y){
    Eigen::Matrix3f a;
    a << x[0]*x[0], x[0], 1,
         x[1]*x[1], x[1], 1,
         x[2]*x[2], x[2], 1; 
    Eigen::Vector3f coeff = a.inverse() * Eigen::Vector3f(y[0], y[1], y[2]);

    return -1 * (coeff[1]/(2 * coeff[0]));
}


