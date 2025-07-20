#include "InteropRenderer.h"

int main()
{
    InteropRenderer renderer(1920, 1080, "CUDA OpenGL Path Tracer", false);

    while (!glfwWindowShouldClose(renderer.window))
    {
        renderer.launchCudaKernel();
        renderer.renderTexturedQuad();
    }

    return 0;
}

