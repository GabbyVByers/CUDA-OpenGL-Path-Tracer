#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "quaternions.h"

using uint8 = unsigned char;

__global__ inline void renderKernel(uchar4* pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float u = ((x / (float)width) * 2.0f - 1.0f) * (width / (float)height);
    float v = (y / (float)height) * 2.0f - 1.0f;

    uint8 r = 0;
    uint8 g = 0;
    uint8 b = 0;

    r = fmin(fmax(0.0f, u), 1.0f) * 255.0f;
    g = fmin(fmax(0.0f, v), 1.0f) * 255.0f;

    pixels[idx] = make_uchar4(r, g, b, 255);
}

// Raytracing Kernel Schitzo-Post (aka Pseudo Code)

// Ray Origin = Camera Position
// Ray Direction = Normalize : (Camera.Direction * Camera.Depth) + (U * Camera.Up) + (V * Camera.Right)

// For each Sphere; Perform Ray-Sphere intersection test

