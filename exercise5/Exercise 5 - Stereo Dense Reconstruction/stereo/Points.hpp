#ifndef POINTS_HPP
#define POINTS_HPP

#include <vector>
#include <string>



struct Point{
    Point():x(0.0),y(0.0),z(0.0){}
    Point(const float& pt_x, const float& pt_y, const float& pt_z): x(pt_x),y(pt_x),z(pt_x){}
    float x,y,z;
};

class Points
{
private:
    std::vector<Point> points_;
public:
    Points(){}

    ~Points(){
        points_.clear();
    }

    void AddPoint(const Point& pt){
        points_.emplace_back(pt);
    }

    void AddPoints(const std::vector<Point>& points){
        for (const auto& pt : points){
            AddPoint(pt);
        }
    }

    void Clear(){
        points_.clear();
    }

    bool Save(const std::string& file_name){
        return true;
    }

};

#endif // !POINTS_HPP