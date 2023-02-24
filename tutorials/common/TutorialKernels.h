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

#include <common/Common.h>
#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef SHARED_STACK_SIZE
#define SHARED_STACK_SIZE 16
#endif

constexpr float Pi	  = 3.14159265358979323846f;
constexpr float TwoPi = 2.0f * Pi;

__device__ float3 gammaCorrect( float3 a )
{
	float g = 1.0f / 2.2f;
	return { pow( a.x, g ), pow( a.y, g ), pow( a.z, g ) };
}

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
	float phi		  = TwoPi * randf( seed );
	float sinThetaSqr = randf( seed );
	float sinTheta	  = sqrt( sinThetaSqr );

	float3 axis = fabs( n.x ) > 0.001f ? make_float3( 0.0f, 1.0f, 0.0f ) : make_float3( 1.0f, 0.0f, 0.0f );
	float3 t	= cross( axis, n );
	t			= normalize( t );
	float3 s	= cross( n, t );

	return normalize( s * cos( phi ) * sinTheta + t * sin( phi ) * sinTheta + n * sqrt( 1.0f - sinThetaSqr ) );
}

__device__ bool cutoutFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	const float	  scale = 16.0f;
	const float2& uv	= hit.uv;
	float2		  texCoord[2];
	texCoord[0] = ( 1.0f - uv.x - uv.y ) * make_float2( 0.0f, 0.0f ) + uv.x * make_float2( 0.0f, 1.0f ) +
				  uv.y * make_float2( 1.0f, 1.0f );
	texCoord[1] = ( 1.0f - uv.x - uv.y ) * make_float2( 0.0f, 0.0f ) + uv.x * make_float2( 1.0f, 1.0f ) +
				  uv.y * make_float2( 1.0f, 0.0f );
	if ( ( int( scale * texCoord[hit.primID].x ) + int( scale * texCoord[hit.primID].y ) ) & 1 ) return true;
	return false;
}

// check if there is a hit before ray.maxT. if there is, set it to tOut. hiprt will overwrite ray.maxT after this function
__device__ bool intersectCircle( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float4* o = (const float4*)data;
	float2		  c = make_float2( o[hit.primID].x, o[hit.primID].y );
	const float	  r = o[hit.primID].w;

	c.x			 = c.x - ray.origin.x;
	c.y			 = c.y - ray.origin.y;
	float d		 = sqrtf( c.x * c.x + c.y * c.y );
	bool  hasHit = d < r;
	if ( !hasHit ) return false;

	hit.normal = normalize( make_float3( d, d, d ) );

	return true;
}

// check if there is a hit before ray.maxT. if there is, set it to tOut. hiprt will overwrite ray.maxT after this function
__device__ bool intersectSphere( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	float3 from	  = ray.origin;
	float3 to	  = from + ray.direction * ray.maxT;
	float4 sphere = ( (const float4*)data )[hit.primID];
	float3 center = make_float3( sphere );
	float  r	  = sphere.w;

	float3 m  = from - center;
	float3 d  = to - from;
	float  a  = dot( d, d );
	float  b  = 2.0f * dot( m, d );
	float  c  = dot( m, m ) - r * r;
	float  dd = b * b - 4.0f * a * c;
	if ( dd < 0.0f ) return false;

	float t = ( -b - sqrtf( dd ) ) / ( 2.0f * a );
	if ( t > 1.0f ) return false;

	hit.t	   = t * ray.maxT;
	hit.normal = normalize( from + ray.direction * hit.t - center );

	return true;
}

extern "C" __global__ void GeomIntersectionKernel( hiprtGeometry geom, u8* pixels, int2 res )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float3 o = { x / (float)res.x, y / (float)res.y, -1.0f };
	float3 d = { 0.0f, 0.0f, 1.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = d;

	hiprtGeomTraversalClosest tr( geom, ray );
	hiprtHit				  hit = tr.getNextHit();

	int pixelIndex			   = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = hit.hasHit() ? ( (float)x / res.x ) * 255 : 0;
	pixels[pixelIndex * 4 + 1] = hit.hasHit() ? ( (float)y / res.y ) * 255 : 0;
	pixels[pixelIndex * 4 + 2] = 0;
	pixels[pixelIndex * 4 + 3] = 255;
}

extern "C" __global__ void SceneIntersectionKernel( hiprtScene scene, u8* pixels, int2 res )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float3 o = { x / (float)res.x - 0.5f, y / (float)res.y - 0.5f, -1.0f };
	float3 d = { 0.0f, 0.0f, 1.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = d;

	hiprtSceneTraversalClosest tr( scene, ray, 0xffffffff );
	hiprtHit				   hit = tr.getNextHit();

	int pixelIndex			   = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = hit.hasHit() ? ( (float)x / res.x ) * 255 : 0;
	pixels[pixelIndex * 4 + 1] = hit.hasHit() ? ( (float)y / res.y ) * 255 : 0;
	pixels[pixelIndex * 4 + 2] = 0;
	pixels[pixelIndex * 4 + 3] = 255;
}

extern "C" __global__ void CustomIntersectionKernel( hiprtGeometry geom, u8* pixels, hiprtFuncTable table, int2 res )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	ray.origin	  = { x / (float)res.x - 0.5f, y / (float)res.y - 0.5f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 100000.0f;

	hiprtGeomCustomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit						hit = tr.getNextHit();

	int pixelIndex			   = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = hit.hasHit() ? ( hit.normal.x + 1.0f ) / 2.0f * 255 : 0;
	pixels[pixelIndex * 4 + 1] = hit.hasHit() ? ( hit.normal.y + 1.0f ) / 2.0f * 255 : 0;
	pixels[pixelIndex * 4 + 2] = hit.hasHit() ? ( hit.normal.z + 1.0f ) / 2.0f * 255 : 0;
	pixels[pixelIndex * 4 + 3] = 255;
}

extern "C" __global__ void SharedStackKernel( hiprtGeometry geom, u8* pixels, int2 res, int* globalStackBuffer, int stackSize )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float3 o   = { 278.0f, 273.0f, -900.0f };
	float2 d   = { 2.0f * x / (float)res.x - 1.0f, 2.0f * y / (float)res.y - 1.0f };
	float3 uvw = { -387.817566f, -387.817566f, 1230.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction =
		ray.direction /
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

	__shared__ int	 sharedStackBuffer[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtGlobalStack stack( globalStackBuffer, stackSize, sharedStackBuffer, SHARED_STACK_SIZE );
	hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> tr( geom, ray, stack );

	hiprtHit hit = tr.getNextHit();

	int3 color = { 0, 0, 0 };
	if ( hit.hasHit() )
	{
		float3 n = normalize( hit.normal );
		color.x	 = ( ( n.x + 1.0f ) * 0.5f ) * 255;
		color.y	 = ( ( n.y + 1.0f ) * 0.5f ) * 255;
		color.z	 = ( ( n.z + 1.0f ) * 0.5f ) * 255;
	}

	int pixelIndex			   = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = min( 255, color.x );
	pixels[pixelIndex * 4 + 1] = min( 255, color.y );
	pixels[pixelIndex * 4 + 2] = min( 255, color.z );
	pixels[pixelIndex * 4 + 3] = 255;
}

extern "C" __global__ void
CustomBvhImportKernel( hiprtGeometry geom, u8* pixels, int2 res, int* matIndices, float3* diffusColors )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float3 o   = { 278.0f, 273.0f, -900.0f };
	float2 d   = { 2.0f * x / (float)res.x - 1.0f, 2.0f * y / (float)res.y - 1.0f };
	float3 uvw = { -387.817566f, -387.817566f, 1230.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction =
		ray.direction /
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

	hiprtGeomTraversalAnyHit tr( geom, ray, hiprtTraversalHintDefault );
	while ( tr.getCurrentState() != hiprtTraversalStateFinished )
	{
		hiprtHit hit = tr.getNextHit();

		int3 color = { 0, 0, 0 };
		if ( hit.hasHit() )
		{
			int			matIndex	 = matIndices[hit.primID];
			const float alpha		 = 1.0f / 3.0f;
			float3		diffuseColor = alpha * diffusColors[matIndex];
			color.x					 = diffuseColor.x * 255;
			color.y					 = diffuseColor.y * 255;
			color.z					 = diffuseColor.z * 255;
		}

		int pixelIndex			   = x + y * res.x;
		pixels[pixelIndex * 4 + 0] = min( 255, int( pixels[pixelIndex * 4 + 0] ) + color.x );
		pixels[pixelIndex * 4 + 1] = min( 255, int( pixels[pixelIndex * 4 + 1] ) + color.y );
		pixels[pixelIndex * 4 + 2] = min( 255, int( pixels[pixelIndex * 4 + 2] ) + color.z );
		pixels[pixelIndex * 4 + 3] = 255;
	}
}

extern "C" __global__ void AmbientOcclusionKernel( hiprtGeometry geom, u8* pixels, int2 res, float aoRadius )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int	   spp			= 512;
	int	   aoSamples	= 32;
	int3   color		= { 0, 0, 0 };
	float4 diffuseColor = make_float4( 1.0f, 1.0f, 1.0f, 1.0f );
	float  ao			= 0.0f;

	for ( int p = 0; p < spp; p++ )
	{
		unsigned int seed = tea<16>( x + y * res.x, p ).x;

		float3 o   = { 278.0f, 273.0f, -900.0f };
		float2 d   = { 2.0f * ( x + randf( seed ) ) / (float)res.x - 1.0f, 2.0f * ( y + randf( seed ) ) / (float)res.y - 1.0f };
		float3 uvw = { -387.817566f, -387.817566f, 1230.0f };

		hiprtRay ray;
		ray.origin	  = o;
		ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
		ray.direction =
			ray.direction /
			sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

		hiprtGeomTraversalClosest tr( geom, ray );
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

				for ( int i = 0; i < aoSamples; i++ )
				{
					aoRay.direction = sampleHemisphereCosine( Ng, seed );
					hiprtGeomTraversalAnyHit tr( geom, aoRay );
					aoHit = tr.getNextHit();
					ao += !aoHit.hasHit() ? 1.0f : 0.0f;
				}
			}
		}
	}

	ao = ao / ( spp * aoSamples );

	color.x = ( ao * diffuseColor.x ) * 255;
	color.y = ( ao * diffuseColor.y ) * 255;
	color.z = ( ao * diffuseColor.z ) * 255;

	int pixelIndex			   = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = color.x;
	pixels[pixelIndex * 4 + 1] = color.y;
	pixels[pixelIndex * 4 + 2] = color.z;
	pixels[pixelIndex * 4 + 3] = 255;
}

extern "C" __global__ void MotionBlurKernel( hiprtScene scene, u8* pixels, int2 res )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const unsigned int samples = 32;

	hiprtRay ray;
	float3	 o	  = { x / (float)res.x - 0.5f, y / (float)res.y - 0.5f, -1.f };
	float3	 d	  = { 0.f, 0.f, 1.f };
	ray.origin	  = o;
	ray.direction = d;

	float3 colors[2] = { { 1.0f, 0.0f, 0.5f }, { 0.0f, 0.5f, 1.0f } };

	float3 color = { 0.0f, 0.0f, 0.0f };
	for ( int i = 0; i < samples; ++i )
	{
		float					   time = i / (float)samples;
		hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, time );
		hiprtHit				   hit = tr.getNextHit();
		if ( hit.hasHit() )
		{
			float3 diffuseColor = colors[hit.instanceID];
			color.x += diffuseColor.x;
			color.y += diffuseColor.y;
			color.z += diffuseColor.z;
		}
	}

	int pixelIndex			   = x + y * res.x;
	color					   = gammaCorrect( color / samples );
	pixels[pixelIndex * 4 + 0] = color.x * 255;
	pixels[pixelIndex * 4 + 1] = color.y * 255;
	pixels[pixelIndex * 4 + 2] = color.z * 255;
	pixels[pixelIndex * 4 + 3] = 255;
}

extern "C" __global__ void MultiCustomIntersectionKernel( hiprtScene scene, u8* pixels, hiprtFuncTable table, int2 res )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	ray.origin	  = { x / (float)res.x - 0.5f, y / (float)res.y - 0.5f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 100000.0f;

	hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit				   hit = tr.getNextHit();

	int pixelIndex			   = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = hit.hasHit() ? ( hit.normal.x + 1.f ) / 2.0f * 255 : 0;
	pixels[pixelIndex * 4 + 1] = hit.hasHit() ? ( hit.normal.y + 1.f ) / 2.0f * 255 : 0;
	pixels[pixelIndex * 4 + 2] = hit.hasHit() ? ( hit.normal.z + 1.f ) / 2.0f * 255 : 0;
	pixels[pixelIndex * 4 + 3] = 255;
}

extern "C" __global__ void CutoutKernel( hiprtGeometry geom, u8* pixels, hiprtFuncTable table, int2 res )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	float3	 o	  = { x / (float)res.x, y / (float)res.y, -1.0f };
	float3	 d	  = { 0.0f, 0.0f, 1.0f };
	ray.origin	  = o;
	ray.direction = d;

	hiprtGeomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit				  hit = tr.getNextHit();

	int pixelIndex			   = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = hit.hasHit() ? 255 : 0;
	pixels[pixelIndex * 4 + 1] = hit.hasHit() ? 255 : 0;
	pixels[pixelIndex * 4 + 2] = hit.hasHit() ? 255 : 0;
	pixels[pixelIndex * 4 + 3] = 255;
}

extern "C" __global__ void ConcurrentSceneBuildKernel( hiprtScene scene, u8* pixels, hiprtFuncTable table, int2 res )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	float3	 o	  = { x / (float)res.x, y / (float)res.y, -1.f };
	float3	 d	  = { 0.f, 0.f, 1.f };
	ray.origin	  = o;
	ray.direction = d;
	ray.maxT	  = 1000.f;

	float3 colors[2][3] = {
		{ { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0 } },
		{ { 0.0f, 1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 0.0 } },
	};

	hiprtSceneTraversalAnyHit tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
	while ( 1 )
	{
		hiprtHit hit = tr.getNextHit();

		int3 color = { 0, 0, 0 };
		if ( hit.hasHit() )
		{
			float3 diffuseColor = colors[hit.instanceID][hit.primID];
			color.x				= diffuseColor.x * 255;
			color.y				= diffuseColor.y * 255;
			color.z				= diffuseColor.z * 255;
		}

		int pixelIndex = x + y * res.x;
		pixels[pixelIndex * 4 + 0] += color.x;
		pixels[pixelIndex * 4 + 1] += color.y;
		pixels[pixelIndex * 4 + 2] += color.z;
		pixels[pixelIndex * 4 + 3] = 255;

		if ( tr.getCurrentState() == hiprtTraversalStateFinished ) break;
	}
}
