//
// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <tutorials/common/TestBase.h>
#include <cstdlib>
#include <numeric>
#include <array>

const int CORNELL_BOX_TRIANGLE_COUNT = 32;
const int CORNELL_BOX_MAT_COUNT		 = 4;

const static std::array<hiprtFloat3, CORNELL_BOX_TRIANGLE_COUNT* 3> cornellBoxVertices = { { // Floor  -- white lambert
	{ 0.0f, 0.0f, 0.0f },
	{ 0.0f, 0.0f, 559.2f },
	{ 556.0f, 0.0f, 559.2f },
	{ 0.0f, 0.0f, 0.0f },
	{ 556.0f, 0.0f, 559.2f },
	{ 556.0f, 0.0f, 0.0f },

	// Ceiling -- white lambert
	{ 0.0f, 548.8f, 0.0f },
	{ 556.0f, 548.8f, 0.0f },
	{ 556.0f, 548.8f, 559.2f },

	{ 0.0f, 548.8f, 0.0f },
	{ 556.0f, 548.8f, 559.2f },
	{ 0.0f, 548.8f, 559.2f },

	// Back wall -- white lambert
	{ 0.0f, 0.0f, 559.2f },
	{ 0.0f, 548.8f, 559.2f },
	{ 556.0f, 548.8f, 559.2f },

	{ 0.0f, 0.0f, 559.2f },
	{ 556.0f, 548.8f, 559.2f },
	{ 556.0f, 0.0f, 559.2f },

	// Right wall -- green lambert
	{ 0.0f, 0.0f, 0.0f },
	{ 0.0f, 548.8f, 0.0f },
	{ 0.0f, 548.8f, 559.2f },

	{ 0.0f, 0.0f, 0.0f },
	{ 0.0f, 548.8f, 559.2f },
	{ 0.0f, 0.0f, 559.2f },

	// Left wall -- red lambert
	{ 556.0f, 0.0f, 0.0f },
	{ 556.0f, 0.0f, 559.2f },
	{ 556.0f, 548.8f, 559.2f },

	{ 556.0f, 0.0f, 0.0f },
	{ 556.0f, 548.8f, 559.2f },
	{ 556.0f, 548.8f, 0.0f },

	// Short block -- white lambert
	{ 130.0f, 165.0f, 65.0f },
	{ 82.0f, 165.0f, 225.0f },
	{ 242.0f, 165.0f, 274.0f },

	{ 130.0f, 165.0f, 65.0f },
	{ 242.0f, 165.0f, 274.0f },
	{ 290.0f, 165.0f, 114.0f },

	{ 290.0f, 0.0f, 114.0f },
	{ 290.0f, 165.0f, 114.0f },
	{ 240.0f, 165.0f, 272.0f },

	{ 290.0f, 0.0f, 114.0f },
	{ 240.0f, 165.0f, 272.0f },
	{ 240.0f, 0.0f, 272.0f },

	{ 130.0f, 0.0f, 65.0f },
	{ 130.0f, 165.0f, 65.0f },
	{ 290.0f, 165.0f, 114.0f },

	{ 130.0f, 0.0f, 65.0f },
	{ 290.0f, 165.0f, 114.0f },
	{ 290.0f, 0.0f, 114.0f },

	{ 82.0f, 0.0f, 225.0f },
	{ 82.0f, 165.0f, 225.0f },
	{ 130.0f, 165.0f, 65.0f },

	{ 82.0f, 0.0f, 225.0f },
	{ 130.0f, 165.0f, 65.0f },
	{ 130.0f, 0.0f, 65.0f },

	{ 240.0f, 0.0f, 272.0f },
	{ 240.0f, 165.0f, 272.0f },
	{ 82.0f, 165.0f, 225.0f },

	{ 240.0f, 0.0f, 272.0f },
	{ 82.0f, 165.0f, 225.0f },
	{ 82.0f, 0.0f, 225.0f },

	// Tall block -- white lambert
	{ 423.0f, 330.0f, 247.0f },
	{ 265.0f, 330.0f, 296.0f },
	{ 314.0f, 330.0f, 455.0f },

	{ 423.0f, 330.0f, 247.0f },
	{ 314.0f, 330.0f, 455.0f },
	{ 472.0f, 330.0f, 406.0f },

	{ 423.0f, 0.0f, 247.0f },
	{ 423.0f, 330.0f, 247.0f },
	{ 472.0f, 330.0f, 406.0f },

	{ 423.0f, 0.0f, 247.0f },
	{ 472.0f, 330.0f, 406.0f },
	{ 472.0f, 0.0f, 406.0f },

	{ 472.0f, 0.0f, 406.0f },
	{ 472.0f, 330.0f, 406.0f },
	{ 314.0f, 330.0f, 456.0f },

	{ 472.0f, 0.0f, 406.0f },
	{ 314.0f, 330.0f, 456.0f },
	{ 314.0f, 0.0f, 456.0f },

	{ 314.0f, 0.0f, 456.0f },
	{ 314.0f, 330.0f, 456.0f },
	{ 265.0f, 330.0f, 296.0f },

	{ 314.0f, 0.0f, 456.0f },
	{ 265.0f, 330.0f, 296.0f },
	{ 265.0f, 0.0f, 296.0f },

	{ 265.0f, 0.0f, 296.0f },
	{ 265.0f, 330.0f, 296.0f },
	{ 423.0f, 330.0f, 247.0f },

	{ 265.0f, 0.0f, 296.0f },
	{ 423.0f, 330.0f, 247.0f },
	{ 423.0f, 0.0f, 247.0f },

	// Ceiling light -- emmissive
	{ 343.0f, 548.6f, 227.0f },
	{ 213.0f, 548.6f, 227.0f },
	{ 213.0f, 548.6f, 332.0f },

	{ 343.0f, 548.6f, 227.0f },
	{ 213.0f, 548.6f, 332.0f },
	{ 343.0f, 548.6f, 332.0f } } };

static std::array<int, CORNELL_BOX_TRIANGLE_COUNT> cornellBoxMatIndices = { {
	0, 0,						  // Floor         -- white lambert
	0, 0,						  // Ceiling       -- white lambert
	0, 0,						  // Back wall     -- white lambert
	1, 1,						  // Right wall    -- green lambert
	2, 2,						  // Left wall     -- red lambert
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Short block   -- white lambert
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // Tall block    -- white lambert
	3, 3						  // Ceiling light -- emmissive
} };

const std::array<hiprtFloat3, CORNELL_BOX_MAT_COUNT> cornellBoxEmissionColors = {
	{ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 15.0f, 15.0f, 5.0f }

	} };

const std::array<hiprtFloat3, CORNELL_BOX_MAT_COUNT> cornellBoxDiffuseColors = {
	{ { 0.80f, 0.80f, 0.80f }, { 0.05f, 0.80f, 0.05f }, { 0.80f, 0.05f, 0.05f }, { 0.50f, 0.00f, 0.00f } } };


class Test : public TestBase
{
  public:
	void run() 
	{
		hiprtContext ctxt;
		hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );

		int			stackSize		   = 64;
		const int	sharedStackSize	   = 24;// make it 4KB to get a good occupancy
		const int	blockWidth		   = 8;
		const int	blockHeight		   = 8;
		const int	blockSize		   = blockWidth * blockHeight;
		std::string blockSizeDef	   = "-D BLOCK_SIZE=" + std::to_string( blockSize );
		std::string sharedStackSizeDef = "-D SHARED_STACK_SIZE=" + std::to_string( sharedStackSize );

		std::vector<const char*> opts;
		opts.push_back( blockSizeDef.c_str() );
		opts.push_back( sharedStackSizeDef.c_str() );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= CORNELL_BOX_TRIANGLE_COUNT;
		mesh.triangleStride = sizeof( int ) * 3;
		dMalloc( (char*&)mesh.triangleIndices, 3 * mesh.triangleCount * sizeof( int ) );
		std::array<int, 3 * CORNELL_BOX_TRIANGLE_COUNT> idx;
		std::iota( idx.begin(), idx.end(), 0 );
		dCopyHtoD( (int*)mesh.triangleIndices, idx.data(), 3 * mesh.triangleCount );

		mesh.vertexCount  = 3 * mesh.triangleCount;
		mesh.vertexStride = sizeof( hiprtFloat3 );
		dMalloc( (char*&)mesh.vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) );
		dCopyHtoD( (hiprtFloat3*)mesh.vertices, (hiprtFloat3*)cornellBoxVertices.data(), mesh.vertexCount );
		waitForCompletion();

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.triangleMesh.primitive = &mesh;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize );
		dMalloc( (u8*&)geomTemp, geomTempSize );

		hiprtGeometry geom;
		hiprtCreateGeometry( ctxt, &geomInput, &options, &geom );
		hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, 0, geom );

		oroFunction func;
		buildTraceKernel( ctxt, "../04_global_stack/TestKernel.h", "CornellBoxKernel", func, nullptr, &opts );

		u8* dst;
		dMalloc( dst, m_res.x * m_res.y * 4 );
		hiprtInt2 res = make_hiprtInt2( m_res.x, m_res.y );

		int* stackBuffer;
		dMalloc( stackBuffer, m_res.x * m_res.y * stackSize );

		void* args[] = { &geom, &dst, &res, &stackBuffer, &stackSize };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImageFromDevice( "04_global_stack.png", m_res.x, m_res.y, dst );

		dFree( mesh.triangleIndices );
		dFree( mesh.vertices );
		dFree( geomTemp );
		dFree( dst );
		hiprtDestroyGeometry( ctxt, geom );
		hiprtDestroyContext( ctxt );
	}
};

int main( int argc, char** argv )
{
	Test test;
	test.init( 0 );
	test.run();

	return 0;
}


