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
#if !defined( __KERNELCC__ )

#if !defined( HIPRT_EXPORTS )
#define HIPRT_HOST_DEVICE
#define HIPRT_INLINE inline
#else
#include <hiprt/impl/Common.h>
#endif

struct hiprtInt2
{
	int x, y;
};

struct hiprtFloat2
{
	float x, y;
};

struct hiprtInt3
{
	int x, y, z;
};

struct hiprtFloat3
{
	float x, y, z;
};

struct hiprtInt4
{
	int x, y, z, w;
};

struct hiprtFloat4
{
	float x, y, z, w;
};

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtInt2 make_hiprtInt2( int x, int y ) { return { x, y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtFloat2 make_hiprtFloat2( float x, float y ) { return { x, y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtInt3 make_hiprtInt3( int x, int y, int z ) { return { x, y, z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtFloat3 make_hiprtFloat3( float x, float y, float z ) { return { x, y, z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtInt4 make_hiprtInt4( int x, int y, int z, int w ) { return { x, y, z, w }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtFloat4 make_hiprtFloat4( float x, float y, float z, float w ) { return { x, y, z, w }; }

template<typename T>
struct hiprtArray
{
  public:
	hiprtArray() { m_data = 0; }
	~hiprtArray()
	{
		clear();
	}
	void	 setSize( int size ) 
	{
		clear();
		m_data = new T[size]; 
	}
	void clear() 
	{
		if ( m_data ) delete[] m_data;
	}
	const T* getPtr() const { return m_data; }

  private:
	T* m_data;
};

#if defined( HIPRT_EXPORTS )
#define int2 hiprtInt2
#define int3 hiprtInt3
#define int4 hiprtInt4

#define float2 hiprtFloat2
#define float3 hiprtFloat3
#define float4 hiprtFloat4

#define make_int2 make_hiprtInt2
#define make_int3 make_hiprtInt3
#define make_int4 make_hiprtInt4

#define make_float2 make_hiprtFloat2
#define make_float3 make_hiprtFloat3
#define make_float4 make_hiprtFloat4
#endif
#endif
