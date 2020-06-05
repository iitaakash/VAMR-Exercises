#include "Triangulate.hpp"



Eigen::Matrix3f Cross2Matrix(const Eigen::Vector3f& x){
    Eigen::Matrix3f out;
    out <<    0,  -x(2),  x(1),
           x[2],      0, -x[0],
          -x[1],   x[0],     0;
    return out;
}

void Homogenise(Eigen::Vector3f& x){
    if(x(2) == 0.0){
        return;
    }else{
        x = x / x(2);
    }
}

void Homogenise(Eigen::Vector4f& x){
    if(x(3) == 0.0){
        return;
    }else{
        x = x / x(3);
    }
}

std::vector<Eigen::Vector4f> LinearTriangulation(const std::vector<Eigen::Vector3f> &p1,
                    const std::vector<Eigen::Vector3f> &p2,
                    const Eigen::Matrix<float, 3, 4> &m1,
                    const Eigen::Matrix<float, 3, 4> &m2){
    
    std::vector<Eigen::Vector4f> out;
    for (size_t i = 0; i < p1.size(); i++)
    {
        
        Eigen::Vector4f out_pt =  LinearTriangulation(p1[i], p2[i], m1, m2);
        out.emplace_back(out_pt);
    }
    return out;
}


Eigen::Vector4f LinearTriangulation(const Eigen::Vector3f &p1,
                    const Eigen::Vector3f &p2,
                    const Eigen::Matrix<float, 3, 4> &m1,
                    const Eigen::Matrix<float, 3, 4> &m2){

    Eigen::Matrix<float, 3, 4> p1cross = Cross2Matrix(p1) * m1;
    Eigen::Matrix<float, 3, 4> p2cross = Cross2Matrix(p2) * m2;

    Eigen::Matrix<float, 6, 4> A;
    A << p1cross, p2cross;

    // sdv of A
    Eigen::JacobiSVD<Eigen::Matrix<float, 6, 4>> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU );

    // column corresponding to smallest eigen value
    Eigen::Vector4f point = svd.matrixV().col(3);

    Homogenise(point);

    return point;

}