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

#pragma once

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define __KERNELCC__
#endif

#include <hiprt/hiprt_vec.h>
#if !defined( __KERNELCC__ )
#include <cmath>
#endif

#if !defined( __KERNELCC__ )
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

struct float4x4
{
	union
	{
		float4 r[4];
		float  e[4][4];
	};
};

#define RT_MIN( a, b ) ( ( ( b ) < ( a ) ) ? ( b ) : ( a ) )
#define RT_MAX( a, b ) ( ( ( b ) > ( a ) ) ? ( b ) : ( a ) )

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const float2 a ) { return make_int2( (int)a.x, (int)a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int3& a ) { return make_int2( a.x, a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int4& a ) { return make_int2( a.x, a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int c ) { return make_int2( c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int2& a, const int2& b ) { return make_int2( a.x + b.x, a.y + b.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a, const int2& b ) { return make_int2( a.x - b.x, a.y - b.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int2& a, const int2& b ) { return make_int2( a.x * b.x, a.y * b.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int2& a, const int2& b ) { return make_int2( a.x / b.x, a.y / b.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator+=( int2& a, const int2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator-=( int2& a, const int2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator*=( int2& a, const int2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator/=( int2& a, const int2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator+=( int2& a, const int c )
{
	a.x += c;
	a.y += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator-=( int2& a, const int c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator*=( int2& a, const int c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator/=( int2& a, const int c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a ) { return make_int2( -a.x, -a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int2& a, const int c ) { return make_int2( a.x + c, a.y + c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int c, const int2& a ) { return make_int2( c + a.x, c + a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a, const int c ) { return make_int2( a.x - c, a.y - c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int c, const int2& a ) { return make_int2( c - a.x, c - a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int2& a, const int c ) { return make_int2( c * a.x, c * a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int c, const int2& a ) { return make_int2( c * a.x, c * a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int2& a, const int c ) { return make_int2( a.x / c, a.y / c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int c, const int2& a ) { return make_int2( c / a.x, c / a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const float3& a ) { return make_int3( (int)a.x, (int)a.y, (int)a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int4& a ) { return make_int3( a.x, a.y, a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int2& a, const int c ) { return make_int3( a.x, a.y, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int c ) { return make_int3( c, c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int3& a, const int3& b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a, const int3& b )
{
	return make_int3( a.x - b.x, a.y - b.y, a.z - b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int3& a, const int3& b )
{
	return make_int3( a.x * b.x, a.y * b.y, a.z * b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int3& a, const int3& b )
{
	return make_int3( a.x / b.x, a.y / b.y, a.z / b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator+=( int3& a, const int3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator-=( int3& a, const int3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator*=( int3& a, const int3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator/=( int3& a, const int3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator+=( int3& a, const int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator-=( int3& a, const int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator*=( int3& a, const int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator/=( int3& a, const int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a ) { return make_int3( -a.x, -a.y, -a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int3& a, const int c ) { return make_int3( c + a.x, c + a.y, c + a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int c, const int3& a ) { return make_int3( c + a.x, c + a.y, c + a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a, const int c ) { return make_int3( a.x - c, a.y - c, a.z - c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int c, const int3& a ) { return make_int3( c - a.x, c - a.y, c - a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int3& a, const int c ) { return make_int3( c * a.x, c * a.y, c * a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int c, const int3& a ) { return make_int3( c * a.x, c * a.y, c * a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int3& a, const int c ) { return make_int3( a.x / c, a.y / c, a.z / c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int c, const int3& a ) { return make_int3( c / a.x, c / a.y, c / a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const float4& a ) { return make_int4( (int)a.x, (int)a.y, (int)a.z, (int)a.w ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int2& a, const int c0, const int c1 )
{
	return make_int4( a.x, a.y, c0, c1 );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int3& a, const int c ) { return make_int4( a.x, a.y, a.z, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int c ) { return make_int4( c, c, c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int4& a, const int4& b )
{
	return make_int4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a, const int4& b )
{
	return make_int4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int4& a, const int4& b )
{
	return make_int4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int4& a, const int4& b )
{
	return make_int4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator+=( int4& a, const int4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator-=( int4& a, const int4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator*=( int4& a, const int4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator/=( int4& a, const int4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator+=( int4& a, const int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator-=( int4& a, const int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator*=( int4& a, const int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator/=( int4& a, const int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a ) { return make_int4( -a.x, -a.y, -a.z, -a.w ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int4& a, const int c )
{
	return make_int4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int c, const int4& a )
{
	return make_int4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a, const int c )
{
	return make_int4( a.x - c, a.y - c, a.z - c, a.w - c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int c, const int4& a )
{
	return make_int4( c - a.x, c - a.y, c - a.z, c - a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int4& a, const int c )
{
	return make_int4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int c, const int4& a )
{
	return make_int4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int4& a, const int c )
{
	return make_int4( a.x / c, a.y / c, a.z / c, a.w / c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int c, const int4& a )
{
	return make_int4( c / a.x, c / a.y, c / a.z, c / a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int2& a, const int2& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int2& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int c, const int2& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int2& a, const int2& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int2& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int c, const int2& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int3& a, const int3& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	int z = RT_MAX( a.z, b.z );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int3& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int c, const int3& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int3& a, const int3& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	int z = RT_MIN( a.z, b.z );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int3& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int c, const int3& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int4& a, const int4& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	int z = RT_MAX( a.z, b.z );
	int w = RT_MAX( a.w, b.w );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int4& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	int w = RT_MAX( a.w, c );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int c, const int4& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	int w = RT_MAX( a.w, c );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int4& a, const int4& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	int z = RT_MIN( a.z, b.z );
	int w = RT_MIN( a.w, b.w );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int4& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	int w = RT_MIN( a.w, c );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int c, const int4& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	int w = RT_MIN( a.w, c );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const int2& a ) { return make_float2( (float)a.x, (float)a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float3& a ) { return make_float2( a.x, a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float4& a ) { return make_float2( a.x, a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float c ) { return make_float2( c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float2& a, const float2& b )
{
	return make_float2( a.x + b.x, a.y + b.y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a, const float2& b )
{
	return make_float2( a.x - b.x, a.y - b.y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float2& a, const float2& b )
{
	return make_float2( a.x * b.x, a.y * b.y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float2& a, const float2& b )
{
	return make_float2( a.x / b.x, a.y / b.y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator+=( float2& a, const float2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator-=( float2& a, const float2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator*=( float2& a, const float2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator/=( float2& a, const float2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator+=( float2& a, const float c )
{
	a.x += c;
	a.y += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator-=( float2& a, const float c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator*=( float2& a, const float c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator/=( float2& a, const float c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a ) { return make_float2( -a.x, -a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float2& a, const float c ) { return make_float2( a.x + c, a.y + c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float c, const float2& a ) { return make_float2( c + a.x, c + a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a, const float c ) { return make_float2( a.x - c, a.y - c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float c, const float2& a ) { return make_float2( c - a.x, c - a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float2& a, const float c ) { return make_float2( c * a.x, c * a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float c, const float2& a ) { return make_float2( c * a.x, c * a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float2& a, const float c ) { return make_float2( a.x / c, a.y / c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float c, const float2& a ) { return make_float2( c / a.x, c / a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const int3& a ) { return make_float3( (float)a.x, (float)a.y, (float)a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float4& a ) { return make_float3( a.x, a.y, a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float2& a, const float c ) { return make_float3( a.x, a.y, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float c ) { return make_float3( c, c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float3& a, const float3& b )
{
	return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a, const float3& b )
{
	return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float3& a, const float3& b )
{
	return make_float3( a.x * b.x, a.y * b.y, a.z * b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float3& a, const float3& b )
{
	return make_float3( a.x / b.x, a.y / b.y, a.z / b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator+=( float3& a, const float3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator-=( float3& a, const float3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator*=( float3& a, const float3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator/=( float3& a, const float3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator+=( float3& a, const float c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator-=( float3& a, const float c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator*=( float3& a, const float c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator/=( float3& a, const float c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a ) { return make_float3( -a.x, -a.y, -a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float3& a, const float c )
{
	return make_float3( c + a.x, c + a.y, c + a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float c, const float3& a )
{
	return make_float3( c + a.x, c + a.y, c + a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a, const float c )
{
	return make_float3( a.x - c, a.y - c, a.z - c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float c, const float3& a )
{
	return make_float3( c - a.x, c - a.y, c - a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float3& a, const float c )
{
	return make_float3( c * a.x, c * a.y, c * a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float c, const float3& a )
{
	return make_float3( c * a.x, c * a.y, c * a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float3& a, const float c )
{
	return make_float3( a.x / c, a.y / c, a.z / c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float c, const float3& a )
{
	return make_float3( c / a.x, c / a.y, c / a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const int4& a )
{
	return make_float4( (float)a.x, (float)a.y, (float)a.z, (float)a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float2& a, const float c0, const float c1 )
{
	return make_float4( a.x, a.y, c0, c1 );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float3& a, const float c ) { return make_float4( a.x, a.y, a.z, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float c ) { return make_float4( c, c, c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float4& a, const float4& b )
{
	return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a, const float4& b )
{
	return make_float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float4& a, const float4& b )
{
	return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float4& a, const float4& b )
{
	return make_float4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator+=( float4& a, const float4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator-=( float4& a, const float4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator*=( float4& a, const float4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator/=( float4& a, const float4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator+=( float4& a, const float c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator-=( float4& a, const float c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator*=( float4& a, const float c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator/=( float4& a, const float c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a ) { return make_float4( -a.x, -a.y, -a.z, -a.w ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float4& a, const float c )
{
	return make_float4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float c, const float4& a )
{
	return make_float4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a, const float c )
{
	return make_float4( a.x - c, a.y - c, a.z - c, a.w - c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float c, const float4& a )
{
	return make_float4( c - a.x, c - a.y, c - a.z, c - a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float4& a, const float c )
{
	return make_float4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float c, const float4& a )
{
	return make_float4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float4& a, const float c )
{
	return make_float4( a.x / c, a.y / c, a.z / c, a.w / c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float c, const float4& a )
{
	return make_float4( c / a.x, c / a.y, c / a.z, c / a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 cross( const float3& a, const float3& b )
{
	return make_float3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

HIPRT_HOST_DEVICE HIPRT_INLINE float dot( const float4& a, const float4& b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 normalize( const float3& a ) { return a / sqrtf( dot( a, a ) ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float4x4& m, const float4& v )
{
	return make_float4( dot( m.r[0], v ), dot( m.r[1], v ), dot( m.r[2], v ), dot( m.r[3], v ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4x4 operator*( const float4x4& a, const float4x4& b )
{
	float4x4 m;
	for ( int r = 0; r < 4; ++r )
	{
		for ( int c = 0; c < 4; ++c )
		{
			m.e[r][c] = 0.0f;
			for ( int k = 0; k < 4; ++k )
				m.e[r][c] += a.e[r][k] * b.e[k][c];
		}
	}

	return m;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4x4 Perspective( float y_fov, float aspect, float n, float f )
{
	float a = 1.0f / tanf( y_fov / 2.0f );

	float4x4 m;
	m.r[0] = make_float4( a / aspect, 0.0f, 0.0f, 0.0f );
	m.r[1] = make_float4( 0.0f, a, 0.0f, 0.0f );
	m.r[2] = make_float4( 0.0f, 0.0f, f / ( f - n ), n * f / ( n - f ) );
	m.r[3] = make_float4( 0.0f, 0.0f, 1.0f, 0.0f );

	return m;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4x4 LookAt( const float3& eye, const float3& at, const float3& up )
{
	float3 f = normalize( at - eye );
	float3 s = normalize( cross( up, f ) );
	float3 t = cross( f, s );

	float4x4 m;
	m.r[0] = make_float4( s, -dot( s, eye ) );
	m.r[1] = make_float4( t, -dot( t, eye ) );
	m.r[2] = make_float4( f, -dot( f, eye ) );
	m.r[3] = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );

	return m;
}
