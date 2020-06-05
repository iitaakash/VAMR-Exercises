#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "Timer.hpp"
#include "Triangulate.hpp"




int main(int argc, const char* argv[])
{
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

    for (size_t i = 0; i < num_points; i++)
    {
        Eigen::Vector4f point;
        point << rand()%10, rand()%10, (rand()%10)*5 + 10, 1;

        auto p1 = m1 * point;
        auto p2 = m2 * point;

        Eigen::Vector4f point_est = LinearTriangulation(p1, p2, m1, m2);
        avg += (point_est - point).cwiseAbs();
    }

    std::cout << avg / num_points << std::endl;

    std::cout << "Time Elapsed: " << t.TimeElapsed()/num_points << std::endl;

    return 0;
}