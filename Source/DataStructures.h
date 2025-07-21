#pragma once

#include "vec3.h"

struct Sphere
{
    vec3 position;
    float radius;
    vec3 color;
};

struct hitInfo
{
    bool didHit;
    vec3 hitLocation;
    vec3 hitColor;
    vec3 hitNormal;
};

struct Ray
{
    vec3 origin;
    vec3 direction;
};

struct Camera
{
    vec3 position;
    vec3 direction;
    float depth;
    vec3 up;
};