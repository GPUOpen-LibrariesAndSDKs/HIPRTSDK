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

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

#define M_PI 3.1415926535898f
#define TWO_PI 6.28318530718f

typedef float4 Quaternion;

struct Camera
{
	float4		m_translation; // eye/rayorigin
	Quaternion	m_quat;
	float		m_fov;
	float		m_near;
	float		m_far;
	float		padd;
};

__device__ float dot3F4( const float4& a, const float4& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ const float4 cross3( const float3 aa, const float3 bb )
{
	return make_float4( aa.y * bb.z - aa.z * bb.y, aa.z * bb.x - aa.x * bb.z, aa.x * bb.y - aa.y * bb.x, 0 );
}

__device__ float3 cross( const float3& a, const float3& b )
{
	return make_float3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}

__device__ float	  dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ float3 normalize( const float3& a ) { return a / sqrtf( dot( a, a ) ); }

__device__ unsigned int lcg( unsigned int& seed )
{
	const unsigned int LCG_A = 1103515245u;
	const unsigned int LCG_C = 12345u;
	const unsigned int LCG_M = 0x00FFFFFFu;
	seed					 = ( LCG_A * seed + LCG_C );
	return seed & LCG_M;
}

__device__ float randf( unsigned int& seed ) { return ( (float)lcg( seed ) / (float)0x01000000 ); }

template <unsigned int N>
__device__ uint2 tea( unsigned int val0, unsigned int val1 )
{
	unsigned int v0 = val0;
	unsigned int v1 = val1;
	unsigned int s0 = 0;

	for ( unsigned int n = 0; n < N; n++ )
	{
		s0 += 0x9e3779b9;
		v0 += ( ( v1 << 4 ) + 0xa341316c ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + 0xc8013ea4 );
		v1 += ( ( v0 << 4 ) + 0xad90777d ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + 0x7e95761e );
	}

	return make_uint2( v0, v1 );
}

__device__ float3 sampleHemisphereCosine( float3 n, unsigned int& seed )
{
	float phi		  = TWO_PI * randf( seed );
	float sinThetaSqr = randf( seed );
	float sinTheta	  = sqrt( sinThetaSqr );

	float3 axis = fabs( n.x ) > 0.001f ? make_float3( 0.0f, 1.0f, 0.0f ) : make_float3( 1.0f, 0.0f, 0.0f );
	float3 t	= cross( axis, n );
	t			= normalize( t );
	float3 s	= cross( n, t );

	return normalize( s * cos( phi ) * sinTheta + t * sin( phi ) * sinTheta + n * sqrt( 1.0f - sinThetaSqr ) );
}

inline __device__ float4 qtMul( const float4& a, const float4& b )
{
	float4 ans;
	ans = make_float4( cross( make_float3( a ), make_float3( b ) ), 0.0f );
	// ans += a.w * b + b.w * a;
	ans = ans + make_float4( a.w * b.x, a.w * b.y, a.w * b.z, a.w * b.w ) +
		  make_float4( b.w * a.x, b.w * a.y, b.w * a.z, b.w * a.w );
	ans.w = a.w * b.w - dot( make_float3( a ), make_float3( b ) );
	return ans;
}

inline __device__ float4 qtInvert( const float4& q )
{
	float4 ans;
	ans	  = -q;
	ans.w = q.w;
	return ans;
}

inline __device__ float3 qtRotate( const float4& q, const float3& p )
{
	float4 qp	= make_float4( p, 0.0f );
	float4 qInv = qtInvert( q );
	float4 out	= qtMul( qtMul( q, qp ), qInv );
	return make_float3( out );
}

__device__ void
generateRay( float x, float y, int2 res, Camera* cam, float4* to, float4* from, unsigned int& seed, bool isMultiSamples )
{
	float  fov = cam->m_fov;
	float2 m_sensorSize;

	m_sensorSize.x		= 0.024f * ( res.x / (float)res.y );
	m_sensorSize.y		= 0.024f;
	float		 offset = ( isMultiSamples ) ? randf( seed ) : 0.5f;
	const float2 xy		= make_float2( ( x + offset ) / res.x, ( y + offset ) / res.y ) - make_float2( 0.5f, 0.5f );
	float3 dir = make_float3( xy.x * m_sensorSize.x, xy.y * m_sensorSize.y, m_sensorSize.y / ( 2.f * tan( fov / 2.f ) ) );

	const float3 holDir	 = qtRotate( cam->m_quat, make_float3( 1.f, 0.f, 0.f ) );
	const float3 upDir	 = qtRotate( cam->m_quat, make_float3( 0.f, 1.f, 0.f ) );
	const float3 viewDir = qtRotate( cam->m_quat, make_float3( 0.f, 0.f, -1.f) );
	dir					 = dir.x * holDir + dir.y * upDir + dir.z * viewDir;

	from[0]		= cam->m_translation;
	to[0]		= cam->m_translation + make_float4( dir.x * cam->m_far, dir.y * cam->m_far, dir.z * cam->m_far, 0.0f );
}

__global__ void AORayKernel(
	hiprtScene	   scene,
	unsigned char* gDst,
	int2		   cRes,
	int*		   globalStackBuffer,
	int			   stackSize,
	Camera*		   cam,
	float		   aoRadius )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	int	   nSpp			= 512;
	int3   color		= { 0, 0, 0 };
	float4 diffuseColor = make_float4( 1.0f, 1.0f, 1.0f, 1.0f );
	float  ao			= 0.0f;
	int	   nAOSamples	= 32;

	typedef hiprtCustomSharedStack<SHARED_STACK_SIZE> Stack;
	__shared__ int									  sharedStackBuffer[SHARED_STACK_SIZE * BLOCK_SIZE];

	int*  threadSharedStackBuffer = sharedStackBuffer + SHARED_STACK_SIZE * ( threadIdx.x + threadIdx.y * blockDim.x );
	int*  threadGlobalStackBuffer = globalStackBuffer + stackSize * ( gIdx + gIdy * cRes.x );
	Stack stack( threadGlobalStackBuffer, threadSharedStackBuffer );

	for ( int p = 0; p < nSpp; p++ )
	{
		unsigned int seed = tea<16>( gIdx + gIdy * cRes.x, p ).x;
		float4		 to;
		float4		 from;
		generateRay( gIdx, gIdy, cRes, cam, &to, &from, seed, true );

		hiprtRay ray;
		float4	 dir  = ( to - from );
		ray.origin	  = make_float3( from.x, from.y, from.z );
		ray.direction = normalize( make_float3( dir.x, dir.y, dir.z ) );
		ray.maxT	  = 100000.0f;

		hiprtSceneTraversalClosestCustomStack<Stack> tr( scene, ray, 0xffffffff, stack );
		{
			hiprtHit hit = tr.getNextHit();

			if ( hit.hasHit() )
			{
				float3 surfacePt = ray.origin + hit.t * ( 1.0f - 1.0e-2f ) * ray.direction;

				float3 Ng = hit.normal;
				if ( dot( ray.direction, Ng ) > 0.f ) Ng = -Ng;
				Ng = normalize( Ng );

				hiprtRay aoRay;
				aoRay.origin = surfacePt;
				aoRay.maxT	 = aoRadius;
				hiprtHit aoHit;

				for ( int i = 0; i < nAOSamples; i++ )
				{
					aoRay.direction = sampleHemisphereCosine( Ng, seed );
					hiprtSceneTraversalClosestCustomStack<Stack> tr( scene, aoRay, 0xffffffff, stack );
					aoHit = tr.getNextHit();
					ao += !aoHit.hasHit() ? 1.0f : 0.0f;
				}
				
			}
		}
	}

	ao = ao / ( nSpp * nAOSamples );

	color.x = ( ao * diffuseColor.x ) * 255;
	color.y = ( ao * diffuseColor.y ) * 255;
	color.z = ( ao * diffuseColor.z ) * 255;

	int dstIdx			 = gIdx + gIdy * cRes.x;
	gDst[dstIdx * 4 + 0] = color.x;
	gDst[dstIdx * 4 + 1] = color.y;
	gDst[dstIdx * 4 + 2] = color.z;
	gDst[dstIdx * 4 + 3] = 255;
}