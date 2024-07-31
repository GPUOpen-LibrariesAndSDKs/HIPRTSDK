# [18_shadow_ray](../tutorials/18_shadow_ray)

<br />

This demo shows how to implement a Shadow Ray in HIPRT. A shadow ray is a ray that is cast from a point on a surface (the point of intersection where the primary ray hits) towards a light source. The purpose of the shadow ray is to determine whether the point is in shadow or illuminated by the light source.

<div align="center">
    <img src="../tutorials/imgs/18_shadow_ray.png" alt="img" width="300"/>
</div>

<br />
<br />

## Overview

This demo utilizes a more complex framework than the previous ones and provides a broader overview of the HIPRT features. It is integrated with an OBJ importer, enabling the rendering of sophisticated scenes.<br />
From a high level point of view, the code of this demo is very simple: Basically we just need to specify the path to the OBJ, and the HIP kernel to execute on this scene:

```cpp
  void run()
  {
    Camera camera = createCamera();

    setupScene(
      camera,
      "../common/meshes/cornellpot/cornellpot.obj",
      "../common/meshes/cornellpot/" );
    render(
      "18_ShadowRay.png",
      "../common/ShadowRayKernel.h",
      "ShadowRayKernel"
      );
    deleteScene( m_scene );
    return;
  }
```

<br />


## Create Scene



The `SceneDemo::createScene` function leverages the HIPRT API to construct a 3D scene from an OBJ file. It begins by initializing the HIPRT context and loading the OBJ file using the TinyObj library. For instance, the context is created with:
```cpp
  hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, scene.m_ctx );
```

<br />
<br />

The function processes the loaded geometry and materials, translating them into the HIPRT format. It creates vertices, normals, and indices buffers, and organizes them into appropriate structures. For example, the geometry data is extracted and stored as follows:

```cpp
  for (size_t i = 0; i < shapes.size(); ++i) {
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<uint32_t> indices;
    // process each shape's vertices, normals, and indices
  }
```

<br />
<br />

The scene is subdivided for parallel processing, taking advantage of multiple threads to build geometries and instances:
```cpp
  std::vector<std::thread> threads(threadCount);
  for (size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
    threads[threadIndex] = std::thread([&](uint32_t threadIndex) {
      // build geometries in parallel
      }, threadIndex);
  }
```

<br />
<br />

For each shape in the OBJ file, it constructs the necessary HIPRT geometry inputs and builds BVH structures using the HIPRT API. Once geometries are created, they are added to the scene along with their corresponding materials, lights, and masks. Here’s an example of building the geometries:
```cpp
  hiprtGeometryBuildInput geomInput;
  geomInput.type = hiprtPrimitiveTypeTriangleMesh;
  geomInput.primitive.triangleMesh = mesh;
  hiprtCreateGeometries(scene.m_ctx, 1, &geomInput, options, &m_scene.m_geometries[i]);
```

<br />
<br />

Finally, it assembles the HIPRT scene by combining all geometries and instances, applying transformations if provided, and builds the scene's BVH:
```cpp
  hiprtSceneBuildInput sceneInput;
  sceneInput.instanceCount = static_cast<uint32_t>(shapes.size());
  hiprtBuildScene(scene.m_ctx, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, sceneLocal);
```

<br />
<br />

The constructed scene is stored in the provided SceneData object, ready for rendering. The function also handles memory allocation and cleanup through utility functions, ensuring efficient management of GPU resources.

<br />


## Render Scene



The `SceneDemo::render` function handles rendering the scene created by createScene using the HIPRT API. The function starts by allocating memory for the destination image buffer, ensuring it's initialized to zero:
```cpp
  uint8_t* dst;
  OrochiUtils::malloc(dst, m_res.x * m_res.y * 4);
  OrochiUtils::memset(dst, 0, m_res.x * m_res.y * 4);
  m_scene.m_garbageCollector.push_back(dst);
```

<br />
<br />


Next, it sets up the stack configuration for ray tracing, defining stack sizes and compile-time options for the kernel:
```cpp
  uint32_t stackSize = 64u;
  const uint32_t sharedStackSize = 16u;
  std::string blockSizeDef = "-DBLOCK_SIZE=" + std::to_string(64);
  std::string sharedStackSizeDef = "-DSHARED_STACK_SIZE=" + std::to_string(sharedStackSize);
```

<br />
<br />

The function then creates a global stack buffer used by HIPRT for managing ray tracing stacks:
```cpp
  hiprtGlobalStackBufferInput stackBufferInput{
    hiprtStackTypeGlobal,
    hiprtStackEntryTypeInteger,
    stackSize,
    static_cast<uint32_t>(m_res.x * m_res.y)
  };
  hiprtGlobalStackBuffer stackBuffer;
  hiprtCreateGlobalStackBuffer(m_scene.m_ctx, stackBufferInput, stackBuffer);
```

<br />
<br />

The rendering kernel is compiled from bitcode using the provided kernel path and function name:
```cpp
  oroFunction func = nullptr;
  buildTraceKernelFromBitcode(m_scene.m_ctx, kernelPath.u8string().c_str(), funcName.c_str(), func);
```

<br />
<br />

Arguments for the kernel launch are prepared, including the scene data, camera data, and various buffers:
```cpp
  int2 res = make_int2(m_res.x, m_res.y);
  void* args[] = {
    &m_scene.m_scene,
    &dst,
    &res,
    &stackBuffer,
    &m_camera,
    &m_scene.m_bufMaterialIndices,
    &m_scene.m_bufMaterials,
    &m_scene.m_bufMatIdsPerInstance,
    &m_scene.m_indices,
    &m_scene.m_indexOffsets,
    &m_scene.m_normals,
    &m_scene.m_normalOffsets,
    &m_scene.m_numOfLights,
    &m_scene.m_lights,
    &aoRadius,
    &funcTable
  };
```

<br />
<br />

The ray tracing kernel is then launched with the specified block dimensions:

```cpp
  std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
  launchKernel(func, m_res.x, m_res.y, 8, 8, args);
  OrochiUtils::waitForCompletion();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
```

<br />
<br />

After the kernel execution, the global stack buffer is destroyed, and the rendered image is written to a file:

```cpp
  hiprtDestroyGlobalStackBuffer(m_scene.m_ctx, stackBuffer);
  writeImage(imgPath.value().u8string().c_str(), m_res.x, m_res.y, dst);
```
<br />

The `render` function efficiently manages GPU resources, compiles the necessary kernel, and executes the ray tracing operations to produce the final rendered image.

<br />

## Shadow Ray Kernel

This kernel performs shadow ray tracing to compute lighting and shading for each pixel in the rendered image. The main steps are as follows:

<br />

### Light Sampling

The sampleLightVertex function samples a light source and calculates the probability density function (PDF) for the light sample. It determines the light vertex and normal, and evaluates the light surface integral:

```cpp
  __device__ float3 sampleLightVertex(const Light& light, float3 x, float3& lVtxOut, float3& lNormalOut, float& pdf, float2 xi) {
    // Calculate light vertex and normal
    // Evaluate light surface integral
  }
```

<br />

### Kernel Entry Point

The ShadowRayKernel function is the main entry point for the kernel. It initializes various parameters, including resolution and camera data, and sets up the global and shared stack buffers for ray tracing:

```cpp
  extern "C" __global__ void __launch_bounds__(64) ShadowRayKernel(
    hiprtScene scene,
    uint8_t* image,
    int2 resolution,
    hiprtGlobalStackBuffer globalStackBuffer,
    Camera camera,
    uint32_t* matIndices,
    Material* materials,
    uint32_t* matOffsetPerInstance,
    uint32_t* indices,
    uint32_t* indxOffsets,
    float3* normals,
    uint32_t* normOffset,
    uint32_t* numOfLights,
    Light* lights,
    float aoRadius
  ) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;
```

<br />

### Ray Generation and Traversal

The kernel generates a ray for each pixel, performs scene traversal to find intersections, and calculates the lighting based on the materials and lights in the scene:

```cpp
  hiprtRay ray = generateRay(x, y, resolution, camera, seed, false);
  hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr(scene, ray, stack, instanceStack);
  hiprtHit hit = tr.getNextHit();

  if (hit.hasHit()) {
    // Process hit information
    // Calculate shading and lighting
  }
```

<br />

### Shadow Ray Calculation

For each hit, the kernel calculates shadow rays to determine visibility of light sources, taking into account occlusion.<br />
Notice the double loop: one iterates over each light, and the other iterates over each Sampler Per Pixel, to improve the quality of the Shadow Ray estimation.

```cpp
  if (matIndex == hiprtInvalidValue || !materials[matIndex].light()) {
    constexpr uint32_t Spp = 256;
    float3 est{};
    for (uint32_t l = 0; l < numOfLights[0]; l++) {
      for (uint32_t p = 0; p < Spp; p++) {
        float3 lightVtx;
        float3 lNormal;
        Light light = lights[l];
        float pdf = 0.0f;
        float3 le = sampleLightVertex(light, surfacePt, lightVtx, lNormal, pdf, make_float2(randf(seed), randf(seed)));

        // Calculate light visibility
        hiprtRay shadowRay;
        shadowRay.origin = surfacePt + 1.0e-3f * Ng;
        shadowRay.direction = normalize(lightDir);
        shadowRay.maxT = 0.99f * sqrtf(dot(lightVec, lightVec));

        hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> tr(scene, shadowRay, stack, instanceStack);
        hiprtHit hitShadow = tr.getNextHit();
        int lightVisibility = hitShadow.hasHit() ? 0 : 1;

        if (pdf != 0.0f)
          est += lightVisibility * le * max(0.0f, dot(Ng, normalize(lightDir))) / pdf;
      }
    }

    finalColor = 1.0f / Spp * est * diffuseColor / hiprt::Pi;
  }
```


<br />

### Writing the Result

Finally, the kernel writes the computed color for each pixel to the output image buffer:

```cpp
  color.x = finalColor.x * 255;
  color.y = finalColor.y * 255;
  color.z = finalColor.z * 255;

  image[index * 4 + 0] = clamp(color.x, 0, 255);
  image[index * 4 + 1] = clamp(color.y, 0, 255);
  image[index * 4 + 2] = clamp(color.z, 0, 255);
  image[index * 4 + 3] = 255;
```

This kernel efficiently calculates lighting and shading for each pixel, taking into account shadows and occlusions, to produce a high-quality rendered image.
