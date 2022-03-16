# HIPRT SDK

HIPRT is a ray tracing library in HIP. The APIs are designed to be minimum and lower level. 


## Directories

- hiprt
  - The header and the library
- tutorials
  - Tutorial source code
- contrib
  - External dependencies


## Requirement

HIPRT runs on AMD and NVIDIA GPUs. HIP and CUDA apis are dynamically loaded so you do not need to have these SDKs if your have these dlls installed with the driver package. 

For AMD GPUs, you need a driver newer than 21.40. 

The supported AMD GPUs are 

- Navi2x
- Vega2x
- Vega1x

## Building the Tutorial


First you need to clone the repository, then init and update the submodule

````
git submodule init
git submodule update
````

After that, run premake like this on Windows (visual studio 2019)

````
cd tutorials
../tools/premake5/win/premake5.exe vs2019
````

Then open the solution, compile & run. These tutorials are made to run on both AMD and NVIDIA by specifying the device index. 


## Introduction to the HIPRT APIs

The minimum example can be found at [tutorials/00_context_creation/main.cpp](tutorials/00_context_creation/main.cpp). On AMD platform, you need to create hip context and device to initialize hiprt using `hiprtCreateContext` by passing `hiprtContextCreationInput` object where you need to set the native API context, device, and device type (HIP or CUDA). 

After that, use `hiprtCreateGeometry` and `hiprtBuildGeometry` to create `hiprtGeometry`. Once you finished setting up objects on the host side, you can pass the object to your kernel. 

An example of minimum kernel can be found at [here](tutorials/01_geom_intersection/TestKernel.h). This is just a HIP kernel but we are passing `hiprtGeometry ` to it. To perform intersection test, simply fill `hiprtRay`, then create `hiprtGeomTraversalClosest` object, call, `getNextHit()`. It is that simple. 

## References
