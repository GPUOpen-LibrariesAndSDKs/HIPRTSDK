# [07_custom_bvh_import](../tutorials/07_custom_bvh_import)

<br />

<div align="center">
    <img src="../tutorials/imgs/07_custom_bvh_import.png" alt="img" width="300"/>
</div>


One interesting feature that HIPRT supports is importing a custom BVH. The
build input structure has an optional list of nodes defining a topology and bounding boxes of the
custom BVH. If the list is specified, then HIPRT skips the build and just converts the nodes from
the API format to the internal format. This feature might be useful for research, allowing users to
benchmark their BVH builders with HIPRT hardware-accelerated kernels.

<br />
This demo illustrates that.

<br />
From a high level point of view of what you need to do:

you don't need the usual `geomTemp`, that you build with `hiprtGetGeometryBuildTemporaryBufferSize`.

Also, in the `buildFlags` of `hiprtBuildOptions`, you need to specify `hiprtBuildFlagBitCustomBvhImport`.

That's pretty much it, exept of course the build of the custom BVH itself.

It's done inside `buildBvh` in the demo.

the main task of `buildBvh` is to build the `nodeList` of `hiprtGeometryBuildInput`. ( this node list is optional, and only used for custom BVH.



...TODO - WIP ....
