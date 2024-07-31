# [19_primary_ray](../tutorials/19_primary_ray)

<br />

This demo shows examples of primary ray renderings. It can be useful for obtaining various information about the rendered scene and is quite straightforward to implement.

<div align="center">
    <img src="../tutorials/imgs/19_primary_ray.png" alt="img" width="300"/>
</div>

## Overview

This demo uses the same framework presented in [the demo of shadow ray](18_shadow_ray.md), so we won't describe it in detail again. Here is the high-level source code for creating the scene. <br />
It creates the scene and renders it with different primary ray options using the `PrimaryRayKernel`.


```cpp
  std::vector<OPTION_DEF> optionsList = {
    OPTION_DEF(VisualizeNormal,"normal"), 
    OPTION_DEF(VisualizeUv,"uv"), 
    OPTION_DEF(VisualizeId,"primId"), 
    OPTION_DEF(VisualizeHitDist,"depth"), 
    };

  Camera camera = createCamera();
  setupScene(
    camera,
    "../common/meshes/cornellpot/cornellpot.obj",
    "../common/meshes/cornellpot/",
    false,
    std::nullopt,
    hiprtBuildFlagBitPreferFastBuild
    );

  for(const auto& o : optionsList)
  {
    const std::string  kernelName = "PrimaryRayKernel_" + std::to_string( o.i );
    render(
      "19_primaryRay_" + o.name + ".png",
      "../common/PrimaryRayKernel.h",
      kernelName 
      );
  }

  deleteScene( m_scene );
```

<br />

## Primary Ray Kernel

This kernel is designed to demonstrate primary ray tracing, capturing various attributes such as UV coordinates, depth, and more from primary ray information.

<br />


### Kernel Entry Points

The kernel has multiple entry points, each configured to visualize a different attribute. These entry points call a templated PrimaryRayKernel with specific visualization options:

```cpp
  extern "C" __global__ void PrimaryRayKernel_0(...){ PrimaryRayKernel<VisualizeColor>(...); }
  extern "C" __global__ void PrimaryRayKernel_1(...){ PrimaryRayKernel<VisualizeUv>(...); }
  extern "C" __global__ void PrimaryRayKernel_2(...){ PrimaryRayKernel<VisualizeId>(...); }
  extern "C" __global__ void PrimaryRayKernel_3(...){ PrimaryRayKernel<VisualizeHitDist>(...); }
  extern "C" __global__ void PrimaryRayKernel_4(...){ PrimaryRayKernel<VisualizeNormal>(...); }
  extern "C" __global__ void PrimaryRayKernel_5(...){ PrimaryRayKernel<VisualizeAo>(...); }
```
<br />


### Ray Generation and Traversal

Each thread generates a primary ray for a pixel, performs scene traversal to find intersections, and determines the hit information:

```cpp
  hiprtRay ray = generateRay(x, y, resolution, camera, seed, false);
  hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr(scene, ray, stack, instanceStack);
  hiprtHit hit = tr.getNextHit();
```
<br />


### Color Computation

Based on the template parameter, different attributes are visualized:

for example, the Diffuse Color:

```cpp
template <>
__device__ int3 getColor<VisualizeColor>(hiprtScene scene, const hiprtHit& hit, uint32_t* matIndices,
                                         Material* materials, uint32_t* matOffsetPerInstance) {
    const uint32_t matOffset = matOffsetPerInstance[hit.instanceID] + hit.primID;
    const uint32_t matIndex = matIndices[matOffset];
    float3 diffuseColor = materials[matIndex].m_diffuse;
    int3 color = {diffuseColor.x * 255, diffuseColor.y * 255, diffuseColor.z * 255};
    return color;
}
```

Another example, the UV Coordinates:

```cpp
template <>
__device__ int3 getColor<VisualizeUv>(hiprtScene scene, const hiprtHit& hit, uint32_t* matIndices,
                                      Material* materials, uint32_t* matOffsetPerInstance) {
    int3 color = {clamp(static_cast<uint32_t>(hit.uv.x * 255), 0, 255), clamp(static_cast<uint32_t>(hit.uv.y * 255), 0, 255), 0};
    return color;
}

```





