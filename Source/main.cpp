#include "InteropRenderer.h"
#include <iostream>
#include "random.h"

struct Sphere
{
    vec3 position;
    float radius;
    vec3 color;
};

int main()
{
    InteropRenderer renderer(1920, 1080, "CUDA OpenGL Path Tracer", false);

    int numSpheres = 30;
    Sphere* hostSpheres = nullptr;
    hostSpheres = new Sphere[numSpheres];

    for (int i = 0; i < numSpheres; i++)
    {
        Sphere sphere;
        sphere.position = randomVec3(0.0f, 1.0f);
        sphere.color    = randomVec3(0.0f, 1.0f);
        sphere.radius   = randomFloat(1.0f, 2.0f);
    }

    Sphere* deviceSpheres = nullptr;

    while (!glfwWindowShouldClose(renderer.window))
    {
        renderer.launchCudaKernel();
        renderer.renderTexturedQuad();
    }

    return 0;
}

