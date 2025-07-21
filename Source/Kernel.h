#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vec3.h"
#include "Quaternions.h"
#include "DataStructures.h"
#include <cfloat>

using uint8 = unsigned char;

__device__ inline hitInfo raySpheresIntersection(const Ray& ray, const Sphere* spheres, const int& numSpheres)
{
    hitInfo info = { false };
    float closest_t = FLT_MAX;
    
    for (int i = 0; i < numSpheres; i++)
    {
        vec3 V = ray.origin - spheres[i].position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (spheres[i].radius * spheres[i].radius);

        float discriminant = (b * b) - (4.0f * a * c);
        if (discriminant <= 0.0f)
            continue;

        float t1 = ((-b) + sqrt(discriminant)) / (2.0f * a);
        float t2 = ((-b) - sqrt(discriminant)) / (2.0f * a);
        float t = fmin(t1, t2);

        if (t <= 0.0f)
            continue;

        info.didHit = true;
        
        if (t < closest_t)
        {
            closest_t = t;
            info.hitColor = spheres[i].color;
            info.hitLocation = ray.origin + (ray.direction * t);
            info.hitNormal = info.hitLocation - spheres[i].position;
            normalize(info.hitNormal);
        }
    }

    return info;
}

__global__ inline void renderKernel(uchar4* pixels,
                                    int width,
                                    int height,
                                    Sphere* devSpheres,
                                    int numSpheres,
                                    Camera camera)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float u = ((x / (float)width) * 2.0f - 1.0f) * (width / (float)height);
    float v = (y / (float)height) * 2.0f - 1.0f;

    Ray ray =
    {
        camera.position,
        (camera.direction * camera.depth) + (camera.up * v) + ((camera.direction * camera.up) * u)
    };
    normalize(ray.direction);

    hitInfo info = raySpheresIntersection(ray, devSpheres, numSpheres);

    if (info.didHit)
    {
        uint8 r = info.hitColor.x * 255.0f;
        uint8 g = info.hitColor.y * 255.0f;
        uint8 b = info.hitColor.z * 255.0f;
        pixels[idx] = make_uchar4(r, g, b, 255);
        return;
    }
    else
    {
        pixels[idx] = make_uchar4(0, 0, 0, 255);
        return;
    }
}

