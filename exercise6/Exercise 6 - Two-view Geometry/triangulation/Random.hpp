#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <stdlib.h>
#include <time.h> 

#include <random>

class Random
{
private:
    std::default_random_engine generator_{static_cast<long unsigned int>(time(0))};
    std::normal_distribution<float> distribution_;
public:
    Random(){ srand(time(0)); }
    Random(const uint32_t& i){ srand(i); }
    ~Random(){}

    inline float Rand(){
        return distribution_(generator_);
    }

     inline float Rand3(){
        return ((((float) rand() / (RAND_MAX)) - 0.5)*6);
    }
};

#endif // !RANDOM_HPP