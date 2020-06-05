#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <stdlib.h>
#include<time.h> 

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "Timer.hpp"
#include "Triangulate.hpp"

int main(int argc, const char* argv[])
{
    srand(time(0)); 

    Eigen::Matrix<float, 3, 4> m1;
    m1 << 500, 0, 320, 0,
            0, 500, 240, 0,
            0, 0, 1, 0;

    Eigen::Matrix<float, 3, 4> m2;
    m2 << 500, 0, 320, -100,
            0, 500, 240, 0,
            0, 0, 1, 0;

    Eigen::Vector4f avg(0.0,0.0,0.0,0.0);

    Timer t;

    float num_points = 1000;
    float sigma = 0.1;

    std::vector<Eigen::Vector3f> X1;
    std::vector<Eigen::Vector3f> X2;

    for (size_t i = 0; i < num_points; i++)
    {
        Eigen::Vector4f point;
        point << rand()%10, rand()%10, (rand()%10)*5 + 10, 1;

        Eigen::Vector3f x1 = m1 * point;
        Eigen::Vector3f x2 = m2 * point;

        Homogenise(x1);
        Homogenise(x2);

        X1.emplace_back(x1);
        X2.emplace_back(x2);

        Eigen::Vector3f noisy_x1 = x1 + (sigma + float(rand()%10)) * Eigen::Vector3f::Ones();
        Eigen::Vector3f noisy_x2 = x2 + (sigma + float(rand()%10)) * Eigen::Vector3f::Ones();
    }

    Eigen::Matrix3f F = EstimateFMatrix(X1,X2);


    // std::cout << avg / num_points << std::endl;

    // std::cout << "Time Elapsed: " << t.TimeElapsed()/num_points << std::endl;

    return 0;
}