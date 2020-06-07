#ifndef TRIANGULATE_HPP
#define TRIANGULATE_HPP

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

Eigen::Matrix3f Cross2Matrix(const Eigen::Vector3f& x);

void Homogenise(Eigen::Vector3f& x);

void Homogenise(Eigen::Vector4f& x);

std::vector<Eigen::Vector4f>
LinearTriangulation(const std::vector<Eigen::Vector3f> &p1,
                    const std::vector<Eigen::Vector3f> &p2,
                    const Eigen::Matrix<float, 3, 4> &m1,
                    const Eigen::Matrix<float, 3, 4> &m2);

Eigen::Vector4f
LinearTriangulation(const Eigen::Vector3f &p1,
                    const Eigen::Vector3f &p2,
                    const Eigen::Matrix<float, 3, 4> &m1,
                    const Eigen::Matrix<float, 3, 4> &m2);

Eigen::Matrix3f EfromF(const Eigen::Matrix3f& F, const Eigen::Matrix3f& K1, const Eigen::Matrix3f& K2);

void DecomposeRT(const Eigen::Matrix3f& E, Eigen::Matrix3f& R, Eigen::Vector3f& T);

Eigen::Matrix3f EstimateFMatrix(const std::vector<Eigen::Vector3f>& p1, const std::vector<Eigen::Vector3f>& p2);

float DistPoint2EpipolarLine(const Eigen::Matrix3f& F, const std::vector<Eigen::Vector3f>& p1, const std::vector<Eigen::Vector3f>& p2);


#endif // !TRIANGULATE_HPP