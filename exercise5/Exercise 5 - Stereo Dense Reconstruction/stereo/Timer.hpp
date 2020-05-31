#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>


class Timer
{
private:
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;
public:
    Timer(){
        start_time_ = std::chrono::steady_clock::now();
    }
    ~Timer(){}
    void Start(){
        start_time_ = std::chrono::steady_clock::now();
    }
    float TimeElapsed(){
        end_time_ = std::chrono::steady_clock::now();
        float time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
        time = time / 1000.0;
        return time;
    }
};

#endif // !TIMER_HPP