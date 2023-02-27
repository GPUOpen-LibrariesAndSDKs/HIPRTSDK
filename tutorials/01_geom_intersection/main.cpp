//
// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	void run() 
	{
		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt ) );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	  = 2;
		mesh.triangleStride	  = sizeof( hiprtInt3 );
		int triangleIndices[] = { 0, 1, 2, 3, 4, 5 };
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&mesh.triangleIndices, mesh.triangleCount * sizeof( hiprtInt3 ) ) );
		CHECK_ORO(
			oroMemcpyHtoD( (oroDeviceptr)mesh.triangleIndices, triangleIndices, mesh.triangleCount * sizeof( hiprtInt3 ) ) );

		mesh.vertexCount  = 6;
		mesh.vertexStride = sizeof( hiprtFloat3 );
		hiprtFloat3 vertices[] = {
			{ 0.0f, 0.0f, 0.0f },
			{ 1.0f, 0.0f, 0.0f },
			{ 0.5f, 1.0f, 0.0f },
			{ 0.0f, 0.0f, 1.0f },
			{ 1.0f, 0.0f, 1.0f },
			{ 0.5f, 1.0f, 1.0f } };
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&mesh.vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
		CHECK_ORO( oroMemcpyHtoD( (oroDeviceptr)mesh.vertices, vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.triangleMesh.primitive = &mesh;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize ) );
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&geomTemp, geomTempSize ) );

		hiprtGeometry geom;
		CHECK_HIPRT( hiprtCreateGeometry( ctxt, &geomInput, &options, &geom ) );
		CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, 0, geom ) );

		oroFunction func;
		buildTraceKernelFromBitcode( ctxt, "../common/TutorialKernels.h", "GeomIntersectionKernel", func );

		u8* pixels;
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&pixels, m_res.x * m_res.y * 4 ) );

		void* args[] = { &geom, &pixels, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "01_geom_intersection.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( oroFree( (oroDeviceptr)mesh.triangleIndices ) );
		CHECK_ORO( oroFree( (oroDeviceptr)mesh.vertices ) );
		CHECK_ORO( oroFree( (oroDeviceptr)geomTemp ) );
		CHECK_ORO( oroFree( (oroDeviceptr)pixels ) );

		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom ) );
		CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
	}
};

int main( int argc, char** argv )
{
	Tutorial tutorial;
	tutorial.init( 0 );
	tutorial.run();

	return 0;
}
