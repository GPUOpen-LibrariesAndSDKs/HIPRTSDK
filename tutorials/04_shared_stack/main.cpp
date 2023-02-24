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
#include <tutorials/common/CornellBox.h>
#include <numeric>

class Tutorial : public TutorialBase
{
  public:
	void run() 
	{
		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt ) );

		int			stackSize			  = 64;
		const int	sharepixelsackSize	  = 16;
		const int	blockWidth			  = 8;
		const int	blockHeight			  = 8;
		const int	blockSize			  = blockWidth * blockHeight;
		std::string blockSizeDef		  = "-D BLOCK_SIZE=" + std::to_string( blockSize );
		std::string sharepixelsackSizeDef = "-D SHARED_STACK_SIZE=" + std::to_string( sharepixelsackSize );

		std::vector<const char*> opts;
		opts.push_back( blockSizeDef.c_str() );
		opts.push_back( sharepixelsackSizeDef.c_str() );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= CornellBoxTriangleCount;
		mesh.triangleStride = sizeof( hiprtInt3 );
		std::array<int, 3 * CornellBoxTriangleCount> triangleIndices;
		std::iota( triangleIndices.begin(), triangleIndices.end(), 0 );
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&mesh.triangleIndices, mesh.triangleCount * sizeof( hiprtInt3 ) ) );
		CHECK_ORO(
			oroMemcpyHtoD( (oroDeviceptr)mesh.triangleIndices, triangleIndices.data(), mesh.triangleCount * sizeof( hiprtInt3 ) ) );

		mesh.vertexCount  = 3 * mesh.triangleCount;
		mesh.vertexStride = sizeof( hiprtFloat3 );
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&mesh.vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			(oroDeviceptr)mesh.vertices, (hiprtFloat3*)cornellBoxVertices.data(), mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

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
		buildTraceKernelFromBitcode( ctxt, "../common/TutorialKernels.h", "SharedStackKernel", func, &opts );

		u8* pixels;
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&pixels, m_res.x * m_res.y * 4 ) );

		int* stackBuffer;
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&stackBuffer, m_res.x * m_res.y * stackSize * sizeof( int ) ) );

		void* args[] = { &geom, &pixels, &m_res, &stackBuffer, &stackSize };
		launchKernel( func, m_res.x, m_res.y, blockWidth, blockHeight, args );
		writeImage( "04_shared_stack.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( oroFree( (oroDeviceptr)stackBuffer ) );
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


