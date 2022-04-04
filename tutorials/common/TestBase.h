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

#define ASSERT( cond )  \
	if ( !( cond ) )    \
	{                   \
		__debugbreak(); \
	}

typedef unsigned char u8;
#define M_PI 3.1415f

class TestBase
{
  public:
	void init( int deviceIndex = 0 );

	virtual ~TestBase() {}

	virtual void run() = 0;

	void readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes = 0 );
	hiprtError buildTraceProgram( hiprtContext ctxt, const char* path, const char* functionName, orortcProgram& progOut, std::vector<const char*>* opts);
	hiprtError buildTraceGetBinary( orortcProgram& prog, size_t& size, char* binary );

	hiprtError buildTraceKernel(
		hiprtContext	   ctxt,
		const char*		   path,
		const char*		   functionName,
		oroFunction&	   function,
		hiprtArray<char>* binaryOut = nullptr,
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
};