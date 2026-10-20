# HIP RT SDK

**HIP RT tutorials. Premake automatically builds HIPRT from its [public source repository](https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT) when `hiprt/hiprt.h` is missing. Existing installations are preserved.**

HIP RT is a ray tracing library in HIP. The APIs are designed to be minimal and low level, making it easy to write a ray tracing application in HIP. We designed the library and APIs to be simple to use and integrate into any existing HIP applications. Although there are a few other ray tracing APIs which, we designed HIP RT to be simpler and easier to use, so you do not need to learn many new kernel types. 

## Features

- Ray-triangle intersection
- Ray-custom primitive intersection
- Ray mask to filter geometries
- Several bounding volume hierarchy (BVH) options
- Offline BVH load and store
- BVH import
- Motion blur

## Requirement

HIP RT runs on AMD and NVIDIA® GPUs. HIP and CUDA® APIs are loaded dynamically at runtime. Building HIP RT from source with the scripts requires the GPU SDK and build tools listed under [Building the tutorials](#building-the-tutorials). Hardware-accelerated ray tracing only works on RDNA™ 2 GPUs (Radeon™ RX 6000 series or newer).

The supported AMD GPU architecture families are:

- Navi4x (Radeon™ RX 9000 series)
- Navi3x (Radeon™ RX 7000 series)
- Navi2x (Radeon™ RX 6000 series)
- Navi1x (Radeon™ RX 5000 series)

You will also need AMD Software: Adrenalin Edition 23.30 or later for Windows and ROCm 5.7 for Linux.

----


## Directories

- [hiprt](hiprt)
  - Headers, library and GPU kernels installed by the source build scripts. See [details](hiprt/README.md).
- [tutorials](tutorials)
  - Tutorial source code
- [contrib](contrib)
  - External dependencies


## Building the tutorials

Install Git, Python 3.8+, CMake 3.19+ and HIP SDK / ROCm with `hipcc` and HIPRTC. Windows requires Visual Studio 2022 with the C++ tools; Linux requires a C++17 compiler and Make. Set `HIP_PATH` to your HIP installation (default on Linux: `/opt/rocm`).

Clone the repository, then run the commands below from its root. Premake initializes missing submodules and builds HIPRT if needed.

### Windows

```bat
cd tutorials
..\tools\premake5\win\premake5.exe vs2022
```

Open `tutorials/hiprtSdkTutorial.sln` in Visual Studio 2022 and build **Release / x64**. Then, from `tutorials`, run a demo:

```bat
cd dist
bin\Release\19_primary_ray64.exe
```

### Linux

```sh
chmod +x tools/premake5/linux64/premake5
cd tutorials
../tools/premake5/linux64/premake5 gmake2
make config=release_x64 -j
cd dist
export LD_LIBRARY_PATH="../../hiprt/linux64:${HIP_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bin/Release/19_primary_ray64
```

## Introduction to the HIP RT APIs

The minimum example can be found at [tutorials/00_context_creation/main.cpp](tutorials/00_context_creation/main.cpp). On AMD platforms, you need to create a HIP context and device to initialize `hiprt` using `hiprtCreateContext` by passing `hiprtContextCreationInput` object where you need to set the native API context, device, and device type (HIP or CUDA). 

After that, use `hiprtCreateGeometry` and `hiprtBuildGeometry` to create `hiprtGeometry`. Once you have finished setting up objects on the host side, you can pass the context object to your kernel. 

An example of a minimum kernel can be found at [here](tutorials/01_geom_intersection/TestKernel.h). This is a simple HIP kernel that we are passing `hiprtGeometry ` to. To perform an intersection test, simply fill in `hiprtRay`, then create a `hiprtGeomTraversalClosest` object, then call `getNextHit()`. It is that simple.

## References 
- Introducing HIP RT – a ray tracing library in HIP, [GPUOpen Blog](https://gpuopen.com/learn/introducing-hiprt/)
- HIP Ray Tracing 2.0 introduces improved custom function handling, Bitcode linking, and more, [GPUOpen Blog](https://gpuopen.com/learn/hiprt-v2-release-optimizations-and-updates/)
- Introducing HIP RT v2.1 - batch construction for small geometries, transformation query functions, and more, [GPUOpen Blog](https://gpuopen.com/learn/hiprt_2_1_batch_construction_transformation_functions/)
- Introducing HIP RT v2.2, [GPUOpen Blog](https://gpuopen.com/learn/introducing-hip-rt-v2-2/)
