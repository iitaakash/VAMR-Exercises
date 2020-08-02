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
    Eigen::MatrixXf Q(num_points,9);
    for (int i = 0; i < num_points; i++)
    {
        float x1 = p1[i](0);
        float y1 = p1[i](1);
        float z1 = p1[i](2);
        float x2 = p2[i](0);
        float y2 = p2[i](1);
        float z2 = p2[i](2);
        
        Q.row(i) << x2 * x1, x2 * y1, x2 * z1,
                    y2 * x1, y2 * y1, y2 * z1, 
                    z2 * x1, z2 * y1, z2 * z1;
        
    }
    // SVD of Q
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(Q, Eigen::ComputeThinV | Eigen::ComputeThinU);

    // column corresponding to smallest eigen value
    Eigen::Matrix<float, 9, 1> Farray = svd.matrixV().col(8);
    // std::cout << Farray << std::endl;

    Eigen::Map<Eigen::Matrix3f> F1(Farray.data());

    Eigen::JacobiSVD<Eigen::Matrix3f> svd1(F1, Eigen::ComputeFullV | Eigen::ComputeFullU);

    Eigen::Matrix3f singular_matrix = svd1.singularValues().asDiagonal().toDenseMatrix();
    singular_matrix(2,2) = 0.0;

    Eigen::Matrix3f F = svd1.matrixU() * singular_matrix * svd1.matrixV().transpose();
    std::cout << F << std::endl;

    return F;
}


float DistPoint2EpipolarLine(const Eigen::Matrix3f& F, const std::vector<Eigen::Vector3f>& p1, const std::vector<Eigen::Vector3f>& p2){
    float score = 0.0;
    float num_pts = p1.size();
    for (size_t i = 0; i < num_pts; i++)
    {
        Eigen::Vector3f epi1 = F.transpose() * p2[i];
        Eigen::Vector3f epi2 = F * p1[i];

        float deno1 = epi1(0) * epi1(0) + epi1(1) * epi1(1);
        float deno2 = epi2(0) * epi2(0) + epi2(1) * epi2(1);

        score += (((epi1.transpose() * p1[i])(0) * (epi1.transpose() * p1[i])(0)) / deno1) + (((epi2.transpose() * p2[i])(0) * (epi2.transpose() * p2[i])(0)) / deno2);
    }

    score = score / num_pts;
    return std::sqrt(score);
}