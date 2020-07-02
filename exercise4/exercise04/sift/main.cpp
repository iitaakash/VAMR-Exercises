#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
// #include <opencv2/xfeatures2d.hpp> //Thanks to Alessandro
#include <opencv2/features2d.hpp>
#include <memory>

#include "Sift.hpp"

int main(int argc, const char* argv[])
{   
    float rescale_factor = 0.3;

    cv::Mat image_left = cv::imread("../img_1.jpg", 0);
    cv::Mat image_right = cv::imread("../img_2.jpg", 0);

    cv::resize(image_left, image_left, cv::Size(), rescale_factor, rescale_factor);
    cv::resize(image_right, image_right, cv::Size(), rescale_factor, rescale_factor);

    std::shared_ptr<Sift> sift = std::make_shared<Sift>();

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;

    sift->DetectAndCompute(image_left, kp1, des1);
    std::cout << "Keypoint Size : " << kp1.size() << std::endl;
    cv::Mat output;
    cv::drawKeypoints(image_left, kp1, output);
    cv::imshow("left", output);

    sift->DetectAndCompute(image_right, kp2, des2);
    std::cout << "Keypoint Size : " << kp2.size() << std::endl;
    cv::drawKeypoints(image_right, kp2, output);
    cv::imshow("right", output);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(4);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( des1, des2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    cv::Mat img_matches;
    cv::drawMatches( image_left, kp1, image_right, kp2, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    cv::imshow("Good Matches", img_matches );




    // cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    // std::vector<cv::KeyPoint> keypoints1, keypoints2;
    // cv::Mat descriptors1, descriptors2;
    // detector->detectAndCompute( image_left, cv::noArray(), keypoints1, descriptors1 );
    // detector->detectAndCompute( image_right, cv::noArray(), keypoints2, descriptors2 );
    // std::cout << keypoints1.size() << std::endl;
    // std::cout << descriptors1.type() << std::endl;
    // //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // // Since SURF is a floating-point descriptor NORM_L2 is used
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(4);
    // std::vector< std::vector<cv::DMatch> > knn_matches;
    // matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    // //-- Filter matches using the Lowe's ratio test
    // const float ratio_thresh = 0.7f;
    // std::vector<cv::DMatch> good_matches;
    // for (size_t i = 0; i < knn_matches.size(); i++)
    // {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //     {
    //         good_matches.push_back(knn_matches[i][0]);
    //     }
    // }
    // //-- Draw matches
    // cv::Mat img_matches;
    // cv::drawMatches( image_left, keypoints1, image_right, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
    //              cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // //-- Show detected matches
    // cv::imshow("Good Matches", img_matches );

    cv::waitKey(0);

    return 0;
}