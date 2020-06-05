#include "Triangulate.hpp"

Eigen::Matrix3f Cross2Matrix(const Eigen::Vector3f &x)
{
    Eigen::Matrix3f out;
    out << 0, -x(2), x(1),
        x[2], 0, -x[0],
        -x[1], x[0], 0;
    return out;
}

void Homogenise(Eigen::Vector3f &x)
{
    if (x(2) == 0.0)
    {
        return;
    }
    else
    {
        x = x / x(2);
    }
}

void Homogenise(Eigen::Vector4f &x)
{
    if (x(3) == 0.0)
    {
        return;
    }
    else
    {
        x = x / x(3);
    }
}

std::vector<Eigen::Vector4f> LinearTriangulation(const std::vector<Eigen::Vector3f> &p1,
                                                 const std::vector<Eigen::Vector3f> &p2,
                                                 const Eigen::Matrix<float, 3, 4> &m1,
                                                 const Eigen::Matrix<float, 3, 4> &m2)
{

    std::vector<Eigen::Vector4f> out;
    for (size_t i = 0; i < p1.size(); i++)
    {

        Eigen::Vector4f out_pt = LinearTriangulation(p1[i], p2[i], m1, m2);
        out.emplace_back(out_pt);
    }
    return out;
}

Eigen::Vector4f LinearTriangulation(const Eigen::Vector3f &p1,
                                    const Eigen::Vector3f &p2,
                                    const Eigen::Matrix<float, 3, 4> &m1,
                                    const Eigen::Matrix<float, 3, 4> &m2)
{

    Eigen::Matrix<float, 3, 4> p1cross = Cross2Matrix(p1) * m1;
    Eigen::Matrix<float, 3, 4> p2cross = Cross2Matrix(p2) * m2;

    Eigen::Matrix<float, 6, 4> A;
    A << p1cross, p2cross;

    // SVD of A
    Eigen::JacobiSVD<Eigen::Matrix<float, 6, 4>> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU);

    // column corresponding to smallest eigen value
    Eigen::Vector4f point = svd.matrixV().col(3);

    Homogenise(point);

    return point;
}

Eigen::Matrix3f EfromF(const Eigen::Matrix3f &F, const Eigen::Matrix3f &K1, const Eigen::Matrix3f &K2)
{
    return K2.transpose() * F * K1;
}

void DecomposeRT(const Eigen::Matrix3f &E, Eigen::Matrix3f &R, Eigen::Vector3f &T)
{
    // todo: implement this function
    return;
}

Eigen::Matrix3f EstimateFMatrix(const std::vector<Eigen::Vector3f> &p1, const std::vector<Eigen::Vector3f> &p2)
{
    const int num_points = p1.size();
    Eigen::Matrix<float, Eigen::Dynamic, 9> Q(num_points,9);
    for (int i = 0; i < num_points; i++)
    {
        Q.row(i) << p2[i](0) * p1[i](0), p2[i](0) * p1[i](1), 
                    p2[i](0), p2[i](1) * p1[i](0), p2[i](1) * p1[i](1),
                    p2[i](1), p1[i](0), p1[i](1), 1;
        
    }
    // SVD of Q
    Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, 9>> svd(Q, Eigen::ComputeFullV | Eigen::ComputeFullU);

    // column corresponding to smallest eigen value
    Eigen::Matrix<float, 9, 1> Farray = svd.matrixV().col(8);
    // std::cout << Farray << std::endl;

    Eigen::Map<Eigen::Matrix3f> F1(Farray.data());
    // std::cout << F << std::endl;

    Eigen::JacobiSVD<Eigen::Matrix3f> svd1(F1, Eigen::ComputeFullV | Eigen::ComputeFullU);

    Eigen::Matrix3f singular_matrix = svd1.singularValues().asDiagonal().toDenseMatrix();
    singular_matrix(2,2) = 0.0;

    Eigen::Matrix3f F = svd1.matrixU() * singular_matrix * svd1.matrixV().transpose();
    std::cout << F << std::endl;

    return F;
}