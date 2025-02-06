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
#include <hiprt/hiprt_math.h>

#if !defined( __KERNELCC__ )
#include <cmath>
#endif

#include <hiprt/hiprt_common.h>
#if defined( __KERNELCC__ )
#include <hiprt/hiprt_device.h>
#endif

static constexpr bool UseDynamicStack = false;

#if defined( __KERNELCC__ )
typedef typename hiprt::conditional<UseDynamicStack, hiprtDynamicStack, hiprtGlobalStack>::type Stack;
typedef hiprtEmptyInstanceStack																	InstanceStack;
#endif

struct float4x4
{
	union
	{
		float4 r[4];
		float  e[4][4];
	};
};

enum
{
	VisualizeColor,
	VisualizeUv,
	VisualizeId,
	VisualizeHitDist,
	VisualizeNormal,
	VisualizeAo
};

struct Material
{
	float3 m_diffuse;
	float3 m_emission;

	HIPRT_HOST_DEVICE HIPRT_INLINE bool light() { return m_emission.x + m_emission.y + m_emission.z > 0.0f; }
};

struct Light
{
	float3 m_le;
	float3 m_lv0;
	float3 m_lv1;
	float3 m_lv2;
	float3 pad;
};

struct Camera
{
	float4 m_rotation;
	float3 m_translation;
	float  m_fov;
};

HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t lcg( uint32_t& seed )
{
	constexpr uint32_t LcgA = 1103515245u;
	constexpr uint32_t LcgC = 12345u;
	constexpr uint32_t LcgM = 0x00FFFFFFu;
	seed					= ( LcgA * seed + LcgC );
	return seed & LcgM;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float randf( uint32_t& seed )
{
	return ( static_cast<float>( lcg( seed ) ) / static_cast<float>( 0x01000000 ) );
}

template <uint32_t N>
HIPRT_HOST_DEVICE HIPRT_INLINE uint2 tea( uint32_t val0, uint32_t val1 )
{
	uint32_t v0 = val0;
	uint32_t v1 = val1;
	uint32_t s0 = 0;

	for ( uint32_t n = 0; n < N; n++ )
	{
		s0 += 0x9e3779b9;
		v0 += ( ( v1 << 4 ) + 0xa341316c ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + 0xc8013ea4 );
		v1 += ( ( v0 << 4 ) + 0xad90777d ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + 0x7e95761e );
	}

	return make_uint2( v0, v1 );
}


HIPRT_HOST_DEVICE HIPRT_INLINE float dot4( const float4& a, const float4& b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float4x4& m, const float4& v )
{
	return { dot4( m.r[0], v ), dot4( m.r[1], v ), dot4( m.r[2], v ), dot4( m.r[3], v ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4x4 Perspective( float y_fov, float aspect, float n, float f )
{
	float a = 1.0f / tanf( y_fov / 2.0f );

	float4x4 m;
	m.r[0] = { a / aspect, 0.0f, 0.0f, 0.0f };
	m.r[1] = { 0.0f, a, 0.0f, 0.0f };
	m.r[2] = { 0.0f, 0.0f, f / ( f - n ), n * f / ( n - f ) };
	m.r[3] = { 0.0f, 0.0f, 1.0f, 0.0f };

	return m;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4x4 LookAt( const float3& eye, const float3& at, const float3& up )
{
	float3 f = hiprt::normalize( at - eye );
	float3 s = hiprt::normalize( hiprt::cross( up, f ) );
	float3 t = hiprt::cross( f, s );

	float4x4 m;
	m.r[0] = hiprt::make_float4( s, -hiprt::dot( s, eye ) );
	m.r[1] = hiprt::make_float4( t, -hiprt::dot( t, eye ) );
	m.r[2] = hiprt::make_float4( f, -hiprt::dot( f, eye ) );
	m.r[3] = { 0.0f, 0.0f, 0.0f, 1.0f };

	return m;
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

HIPRT_HOST_DEVICE HIPRT_INLINE float3 rotate( const float4& rotation, const float3& p )
{
	float3 a = sinf( rotation.w / 2.0f ) * hiprt::normalize( hiprt::make_float3( rotation ) );
	float  c = cosf( rotation.w / 2.0f );
	return 2.0f * hiprt::dot( a, p ) * a + ( c * c - hiprt::dot( a, a ) ) * p + 2.0f * c * hiprt::cross( a, p );
}

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtRay
generateRay( float x, float y, int2 res, const Camera& camera, uint32_t& seed, bool isMultiSamples )
{
	const float	 offset		= ( isMultiSamples ) ? randf( seed ) : 0.5f;
	const float2 sensorSize = { 0.024f * ( res.x / static_cast<float>( res.y ) ), 0.024f };
	const float2 xy			= float2{ ( x + offset ) / res.x, ( y + offset ) / res.y } - float2{ 0.5f, 0.5f };
	const float3 dir = { xy.x * sensorSize.x, xy.y * sensorSize.y, sensorSize.y / ( 2.0f * tan( camera.m_fov / 2.0f ) ) };

	const float3 holDir	 = rotate( camera.m_rotation, { 1.0f, 0.0f, 0.0f } );
	const float3 upDir	 = rotate( camera.m_rotation, { 0.0f, -1.0f, 0.0f } );
	const float3 viewDir = rotate( camera.m_rotation, { 0.0f, 0.0f, -1.0f } );

	hiprtRay ray;
	ray.origin	  = camera.m_translation;
	ray.direction = hiprt::normalize( dir.x * holDir + dir.y * upDir + dir.z * viewDir );
	return ray;
}
