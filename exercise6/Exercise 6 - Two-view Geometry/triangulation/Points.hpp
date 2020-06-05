#ifndef POINTS_HPP
#define POINTS_HPP

#include <fstream>
#include <string>
#include <vector>

struct Point {
  Point() : x(0.0), y(0.0), z(0.0), i(0) {}
  Point(const float &pt_x, const float &pt_y, const float &pt_z,
        const float &intensity)
      : x(pt_x), y(pt_y), z(pt_z), i(intensity) {}
  float x, y, z;
  int i;
};

class Points {
private:
  std::vector<Point> points_;

public:
  Points() {}

  ~Points() { points_.clear(); }

  void AddPoint(const Point &pt) { points_.emplace_back(pt); }

  void AddPoints(const std::vector<Point> &points) {
    for (const auto &pt : points) {
      AddPoint(pt);
    }
  }

  void Clear() { points_.clear(); }

  bool Save(const std::string &file_name) {
    std::ofstream myfile;
    myfile.open(file_name);
    myfile << "ply \nformat ascii 1.0 \n"
                "element vertex ";

    myfile << points_.size() << "\n";

    myfile << "property float x \n"
              "property float y \n"
              "property float z \n" 
              "property uchar red \n"
              "property uchar green \n"
              "property uchar blue \n"
              "end_header \n";

    for (const Point &pt : points_) {
        myfile << pt.x << " " << pt.y << " " << pt.z << " " << pt.i << " " << pt.i << " "<< pt.i << "\n";
    }
    myfile.close();
    return true;
  }
};

#endif // !POINTS_HPP