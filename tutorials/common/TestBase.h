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

#pragma once
#include <hiprt/hiprt.h>
#include <hiprt/hiprt_vec.h>
#include <Orochi/Orochi.h>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include "../common/shared.h"

#ifndef ASSERT
#if defined( _MSC_VER )
#define ASSERT( cond )  \
	if ( !( cond ) )    \
	{                   \
		__debugbreak(); \
	}
#elif defined( __GNUC__ )
#include <signal.h>
#define ASSERT( cond )    \
	if ( !( cond ) )      \
	{                     \
		raise( SIGTRAP ); \
	}
#else
#define ASSERT( cond )
#endif
#endif
typedef unsigned char u8;
#define M_PI 3.1415f

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

class TestBase
{
  public:

	struct SceneData
	{
		int* m_bufMaterialIndices;
		int* m_bufMatIdsPerInstance; // count of material ids per instance use to calculate offset in material Idx buffer for
									 // instance
		Material_t*		   m_bufMaterials;
		hiprtFloat3*			   m_vertices;
		int*			   m_vertexOffsets;
		hiprtFloat3*			   m_normals;
		int*			   m_normalOffsets;
		uint32_t*		   m_indices;
		int*			   m_indexOffsets;
		Light_t*		   m_lights;
		int*			   m_numOfLights;
		hiprtScene		   m_scene;
		std::vector<void*> m_garbageCollector;
		hiprtContext	   m_ctx;
	};

	void init( int deviceIndex = 0 );

	virtual ~TestBase() {}

	virtual void run() = 0;

	void setUp( Camera& c, const char* filepath, const char* dirpath, bool enableRayMask = false, hiprtFrame* frame = nullptr, hiprtBuildFlagBits bvhBuildFlag = hiprtBuildFlagBitPreferFastBuild, bool time = false);
	hiprtError createScene( SceneData& scene, std::string fileName, std::string mtlBaseDir, bool enableRayMask = false, hiprtFrame* frame = nullptr, hiprtBuildFlagBits bvhBuildFlag = hiprtBuildFlagBitPreferFastBuild, bool time = false);
	void render( const char* imgPath, const char* kernelPath, const char* funcName, int width = 512, int height = 512, float aoRadius = 1.0f);

	bool readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes = 0 );
	hiprtError buildTraceProgram( hiprtContext ctxt, const char* path, const char* functionName, orortcProgram& progOut, std::vector<const char*>* opts);
	hiprtError buildTraceGetBinary( orortcProgram& prog, size_t& size, char* binary );

	hiprtError buildTraceKernel(
		hiprtContext	   ctxt,
		const char*		   path,
		const char*		   functionName,
		oroFunction&	   function,
		std::vector<char>* binaryOut = nullptr,
		std::vector<const char*>* opts		= nullptr );

	void launchKernel( oroFunction func, int nx, int ny, void** args, size_t threadPerBlockX = 8, size_t threadPerBlockY = 8, size_t threadPerBlockZ = 1 );

	void writeImage( const char* path, int w, int h, u8* data );

	void writeImageFromDevice( const char* path, int w, int h, u8* data )
	{
		u8* tmp = new u8[w * h * 4];
		u8* tmp1 = new u8[w * h * 4];

		dCopyDtoH( tmp, data, w * h * 4 );
		waitForCompletion();
		for(int j=0; j<h; j++)
		for(int i=0; i<w; i++)
		{
			int idx = i+j*w;
			int dIdx = i+(h-1-j)*w;
			for(int k=0; k<4; k++)
				tmp1[dIdx*4+k] = tmp[idx*4+k];
		}
		writeImage( path, w, h, tmp1 );
		delete [] tmp;
		delete [] tmp1;
	};

	template <typename T>
	void dMalloc( T*& ptr, int n )
	{
		oroError e = oroMalloc( (oroDeviceptr*)&ptr, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	void dFree( void* ptr )
	{
		oroError e = oroFree( (oroDeviceptr)ptr );
		ASSERT( e == oroSuccess );
	}

	void dMemset( void* ptr, int val, int n )
	{
		oroError e = oroMemset( (oroDeviceptr)ptr, val, n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void dCopyHtoD( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyHtoD( (oroDeviceptr)dst, src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void dCopyDtoH( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyDtoH( dst, (oroDeviceptr)src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void dCopyDtoD( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyDtoD( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	void waitForCompletion()
	{
		auto e = oroStreamSynchronize( 0 );
		ASSERT( e == oroSuccess );
	}

  protected:
	hiprtContextCreationInput m_ctxtInput;
	oroCtx					  m_oroCtx;
	oroDevice				  m_oroDevice;
	hiprtInt2				  m_res;
	SceneData				  m_scene;
	Camera*					  m_camera;
};