#include "InteropRenderer.h"
#include <iostream>
#include "Random.h"
#include "DataStructures.h"

int main()
{
    InteropRenderer renderer(1920, 1080, "CUDA OpenGL Path Tracer", false);

    // Spheres
    int numSpheres = 50;
    Sphere* hostSpheres = nullptr;
    hostSpheres = new Sphere[numSpheres];
    for (int i = 0; i < numSpheres; i++)
    {
        Sphere sphere;
        sphere.position = randomVec3(-20.0f, 20.0f);
        sphere.color = randomVec3(0.0f, 1.0f);
        sphere.radius = 1.0f;
        hostSpheres[i] = sphere;
    }
    Sphere* devSpheres = nullptr;
    cudaMalloc((void**)&devSpheres, sizeof(Sphere) * numSpheres);
    cudaMemcpy(devSpheres, hostSpheres, sizeof(Sphere) * numSpheres, cudaMemcpyHostToDevice);

    // Camera
    Camera camera;
    camera.position = { -30.0f, 0.0f, 0.0f };
    camera.direction = { 1.0f, 0.0f, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.depth = 1.0f;

    while (!glfwWindowShouldClose(renderer.window))
    {
        renderer.launchCudaKernel(devSpheres, numSpheres, camera);
        renderer.processKeyboardInput(camera);
        renderer.processMouseInput(camera);
        renderer.renderTexturedQuad();
    }

    return 0;
}

