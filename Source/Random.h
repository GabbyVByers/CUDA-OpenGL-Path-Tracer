#pragma once

#include "vec3.h"

inline float randomFloat(float min, float max)
{
    return ((rand() / (float)RAND_MAX) * (max - min)) + min;
}

inline vec3 randomVec3(float min, float max)
{
    return vec3
    {
        randomFloat(min, max),
        randomFloat(min, max),
        randomFloat(min, max)
    };
}

