#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ inline void renderKernel(uchar4* pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float fx = x / (float)width;
    float fy = y / (float)height;

    unsigned char r = (unsigned char)(255.0f * fx);
    unsigned char g = (unsigned char)(255.0f * fy);
    unsigned char b = 255;
    pixels[idx] = make_uchar4(r, g, b, 255);
}

