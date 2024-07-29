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

#include <common/Aabb.h>
#include <common/Common.h>
#include <common/FluidSimulation.h>

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef SHARED_STACK_SIZE
#define SHARED_STACK_SIZE 16
#endif

__device__ float3 gammaCorrect( float3 a )
{
	float g = 1.0f / 2.2f;
	return { pow( a.x, g ), pow( a.y, g ), pow( a.z, g ) };
}

__device__ bool cutoutFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	constexpr float Scale = 16.0f;
	const float2&	uv	  = hit.uv;
	float2			texCoord[2];
	texCoord[0] = ( 1.0f - uv.x - uv.y ) * make_float2( 0.0f, 0.0f ) + uv.x * make_float2( 0.0f, 1.0f ) +
				  uv.y * make_float2( 1.0f, 1.0f );
	texCoord[1] = ( 1.0f - uv.x - uv.y ) * make_float2( 0.0f, 0.0f ) + uv.x * make_float2( 1.0f, 1.0f ) +
				  uv.y * make_float2( 1.0f, 0.0f );
	if ( ( int( Scale * texCoord[hit.primID].x ) + int( Scale * texCoord[hit.primID].y ) ) & 1 ) return true;
	return false;
}

// check if there is a hit before ray.maxT. if there is, set it to tOut. hiprt will overwrite ray.maxT after this function
__device__ bool intersectCircle( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float4* o = reinterpret_cast<const float4*>( data );
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
	float4 sphere = reinterpret_cast<const float4*>( data )[hit.primID];
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

__device__ bool intersectParticleImpactSphere( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	float3		from	 = ray.origin;
	Particle	particle = reinterpret_cast<const Particle*>( data )[hit.primID];
	Simulation* sim		 = reinterpret_cast<Simulation*>( payload );
	float3		center	 = particle.Pos;
	float		r		 = sim->m_smoothRadius;

	float3 d  = center - from;
	float  r2 = dot( d, d );
	if ( r2 >= r * r ) return false;

	hit.t	   = r2;
	hit.normal = d;

	return true;
}

extern "C" __global__ void GeomIntersectionKernel( hiprtGeometry geom, uint8_t* pixels, int2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	float3 o = { x / static_cast<float>( res.x ), y / static_cast<float>( res.y ), -1.0f };
	float3 d = { 0.0f, 0.0f, 1.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = d;

	hiprtGeomTraversalClosest tr( geom, ray );
	hiprtHit				  hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? ( static_cast<float>( x ) / res.x ) * 255 : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? ( static_cast<float>( y ) / res.y ) * 255 : 0;
	pixels[index * 4 + 2] = 0;
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void SceneIntersectionKernel( hiprtScene scene, uint8_t* pixels, int2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	float3 o = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
	float3 d = { 0.0f, 0.0f, 1.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = d;

	hiprtSceneTraversalClosest tr( scene, ray, 0xffffffff );
	hiprtHit				   hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? ( static_cast<float>( x ) / res.x ) * 255 : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? ( static_cast<float>( y ) / res.y ) * 255 : 0;
	pixels[index * 4 + 2] = 0;
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void CustomIntersectionKernel( hiprtGeometry geom, uint8_t* pixels, hiprtFuncTable table, int2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin	  = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 100000.0f;

	hiprtGeomCustomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit						hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? ( hit.normal.x + 1.0f ) / 2.0f * 255 : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? ( hit.normal.y + 1.0f ) / 2.0f * 255 : 0;
	pixels[index * 4 + 2] = hit.hasHit() ? ( hit.normal.z + 1.0f ) / 2.0f * 255 : 0;
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void CornellBoxKernel( hiprtGeometry geom, uint8_t* pixels, int2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	float3 o   = { 278.0f, 273.0f, -900.0f };
	float2 d   = { 2.0f * x / static_cast<float>( res.x ) - 1.0f, 2.0f * y / static_cast<float>( res.y ) - 1.0f };
	float3 uvw = { -387.817566f, -387.817566f, 1230.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction =
		ray.direction /
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

	hiprtGeomTraversalClosest tr( geom, ray );
	hiprtHit				  hit = tr.getNextHit();

	int3 color = { 0, 0, 0 };
	if ( hit.hasHit() )
	{
		float3 n = normalize( hit.normal );
		color.x	 = ( ( n.x + 1.0f ) * 0.5f ) * 255;
		color.y	 = ( ( n.y + 1.0f ) * 0.5f ) * 255;
		color.z	 = ( ( n.z + 1.0f ) * 0.5f ) * 255;
	}

	pixels[index * 4 + 0] = min( 255, color.x );
	pixels[index * 4 + 1] = min( 255, color.y );
	pixels[index * 4 + 2] = min( 255, color.z );
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void
SharedStackKernel( hiprtGeometry geom, uint8_t* pixels, int2 res, hiprtGlobalStackBuffer globalStackBuffer )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	float3 o   = { 278.0f, 273.0f, -900.0f };
	float2 d   = { 2.0f * x / static_cast<float>( res.x ) - 1.0f, 2.0f * y / static_cast<float>( res.y ) - 1.0f };
	float3 uvw = { -387.817566f, -387.817566f, 1230.0f };

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction =
		ray.direction /
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

	__shared__ int		   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	hiprtGlobalStack									   stack( globalStackBuffer, sharedStackBuffer );
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

	pixels[index * 4 + 0] = min( 255, color.x );
	pixels[index * 4 + 1] = min( 255, color.y );
	pixels[index * 4 + 2] = min( 255, color.z );
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void
CustomBvhImportKernel( hiprtGeometry geom, uint8_t* pixels, int2 res, int* matIndices, float3* diffusColors )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	float3 o   = { 278.0f, 273.0f, -900.0f };
	float2 d   = { 2.0f * x / static_cast<float>( res.x ) - 1.0f, 2.0f * y / static_cast<float>( res.y ) - 1.0f };
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

		pixels[index * 4 + 0] = min( 255, static_cast<int>( pixels[index * 4 + 0] ) + color.x );
		pixels[index * 4 + 1] = min( 255, static_cast<int>( pixels[index * 4 + 1] ) + color.y );
		pixels[index * 4 + 2] = min( 255, static_cast<int>( pixels[index * 4 + 2] ) + color.z );
		pixels[index * 4 + 3] = 255;
	}
}

__device__ float3 sampleHemisphereCosine( float3 n, uint32_t& seed )
{
	float phi		  = 2.0f * hiprt::Pi * randf( seed );
	float sinThetaSqr = randf( seed );
	float sinTheta	  = sqrt( sinThetaSqr );

	float3 axis = fabs( n.x ) > 0.001f ? make_float3( 0.0f, 1.0f, 0.0f ) : make_float3( 1.0f, 0.0f, 0.0f );
	float3 t	= cross( axis, n );
	t			= normalize( t );
	float3 s	= cross( n, t );

	return normalize( s * cos( phi ) * sinTheta + t * sin( phi ) * sinTheta + n * sqrt( 1.0f - sinThetaSqr ) );
}

extern "C" __global__ void AmbientOcclusionKernel( hiprtGeometry geom, uint8_t* pixels, int2 res, float aoRadius )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	constexpr uint32_t Spp			= 32u;
	constexpr uint32_t AoSamples	= 32u;
	int3			   color		= { 0, 0, 0 };
	float4			   diffuseColor = make_float4( 1.0f, 1.0f, 1.0f, 1.0f );
	float			   ao			= 0.0f;

	for ( int p = 0; p < Spp; p++ )
	{
		uint32_t seed = tea<16>( x + y * res.x, p ).x;

		float3 o = { 278.0f, 273.0f, -900.0f };
		float2 d = {
			2.0f * ( x + randf( seed ) ) / static_cast<float>( res.x ) - 1.0f,
			2.0f * ( y + randf( seed ) ) / static_cast<float>( res.y ) - 1.0f };
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
				if ( dot( ray.direction, Ng ) > 0.0f ) Ng = -Ng;
				Ng = normalize( Ng );

				hiprtRay aoRay;
				aoRay.origin = surfacePt;
				aoRay.maxT	 = aoRadius;
				hiprtHit aoHit;

				for ( int i = 0; i < AoSamples; i++ )
				{
					aoRay.direction = sampleHemisphereCosine( Ng, seed );
					hiprtGeomTraversalAnyHit tr( geom, aoRay );
					aoHit = tr.getNextHit();
					ao += !aoHit.hasHit() ? 1.0f : 0.0f;
				}
			}
		}
	}

	ao = ao / ( Spp * AoSamples );

	color.x = ( ao * diffuseColor.x ) * 255;
	color.y = ( ao * diffuseColor.y ) * 255;
	color.z = ( ao * diffuseColor.z ) * 255;

	pixels[index * 4 + 0] = color.x;
	pixels[index * 4 + 1] = color.y;
	pixels[index * 4 + 2] = color.z;
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void MotionBlurKernel( hiprtScene scene, uint8_t* pixels, int2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	constexpr uint32_t Samples = 32;

	hiprtRay ray;
	float3	 o	  = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
	float3	 d	  = { 0.0f, 0.0f, 1.0f };
	ray.origin	  = o;
	ray.direction = d;

	const float3 colors[2] = { { 1.0f, 0.0f, 0.5f }, { 0.0f, 0.5f, 1.0f } };

	float3 color = { 0.0f, 0.0f, 0.0f };
	for ( int i = 0; i < Samples; ++i )
	{
		float					   time = i / static_cast<float>( Samples );
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

	color				  = gammaCorrect( color / Samples );
	pixels[index * 4 + 0] = color.x * 255;
	pixels[index * 4 + 1] = color.y * 255;
	pixels[index * 4 + 2] = color.z * 255;
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void MultiCustomIntersectionKernel( hiprtScene scene, uint8_t* pixels, hiprtFuncTable table, int2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin	  = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 100000.0f;

	hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit				   hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? ( hit.normal.x + 1.0f ) / 2.0f * 255 : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? ( hit.normal.y + 1.0f ) / 2.0f * 255 : 0;
	pixels[index * 4 + 2] = hit.hasHit() ? ( hit.normal.z + 1.0f ) / 2.0f * 255 : 0;
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void CutoutKernel( hiprtGeometry geom, uint8_t* pixels, hiprtFuncTable table, int2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	float3	 o	  = { x / static_cast<float>( res.x ), y / static_cast<float>( res.y ), -1.0f };
	float3	 d	  = { 0.0f, 0.0f, 1.0f };
	ray.origin	  = o;
	ray.direction = d;

	hiprtGeomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit				  hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? 255 : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? 255 : 0;
	pixels[index * 4 + 2] = hit.hasHit() ? 255 : 0;
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void SceneBuildKernel( hiprtScene scene, uint8_t* pixels, hiprtFuncTable table, int2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	float3	 o	  = { x / static_cast<float>( res.x ), y / static_cast<float>( res.y ), -1.0f };
	float3	 d	  = { 0.0f, 0.0f, 1.0f };
	ray.origin	  = o;
	ray.direction = d;
	ray.maxT	  = 1000.0f;

	const float3 colors[2][3] = {
		{ { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0 } },
		{ { 0.0f, 1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 0.0 } },
	};

	hiprtSceneTraversalAnyHit tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
	while ( true )
	{
		hiprtHit hit = tr.getNextHit();

		int3 color = { 0, 0, 0 };
		if ( hit.hasHit() )
		{
			const uint32_t instanceID	= hit.instanceIDs[1] != hiprtInvalidValue ? hit.instanceIDs[1] : hit.instanceIDs[0];
			const float3   diffuseColor = colors[instanceID][hit.primID];
			color.x						= diffuseColor.x * 255;
			color.y						= diffuseColor.y * 255;
			color.z						= diffuseColor.z * 255;
		}

		pixels[index * 4 + 0] += color.x;
		pixels[index * 4 + 1] += color.y;
		pixels[index * 4 + 2] += color.z;
		pixels[index * 4 + 3] = 255;

		if ( tr.getCurrentState() == hiprtTraversalStateFinished ) break;
	}
}

__device__ float calculateDensity( float r2, float h, float densityCoef )
{
	// Implements this equation:
	// W_poly6(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
	// densityCoef = particleMass * 315.0f / (64.0f * pi * h^9)
	const float d2 = h * h - r2;

	return densityCoef * d2 * d2 * d2;
}

__device__ float calculatePressure( float rho, float rho0, float pressureStiffness )
{
	// Implements this equation:
	// Pressure = B * ((rho / rho_0)^3 - 1)
	const float rhoRatio = rho / rho0;

	return pressureStiffness * max( rhoRatio * rhoRatio * rhoRatio - 1.0f, 0.0f );
}

__device__ float3
calculateGradPressure( float r, float d, float pressure, float pressure_j, float rho_j, float3 disp, float pressureGradCoef )
{
	float avgPressure = 0.5 * ( pressure + pressure_j );
	// Implements this equation:
	// W_spkiey(r, h) = 15 / (pi * h^6) * (h - r)^3
	// GRAD(W_spikey(r, h)) = -45 / (pi * h^6) * (h - r)^2
	// pressureGradCoef = particleMass * -45.0f / (pi * h^6)

	return pressureGradCoef * avgPressure * d * d * disp / ( rho_j * r );
}

__device__ float3
calculateVelocityLaplace( float d, float3 velocity, float3 velocity_j, float rho_j, float viscosityLaplaceCoef )
{
	float3 velDisp = ( velocity_j - velocity );
	// Implements this equation:
	// W_viscosity(r, h) = 15 / (2 * pi * h^3) * (-r^3 / (2 * h^3) + r^2 / h^2 + h / (2 * r) - 1)
	// LAPLACIAN(W_viscosity(r, h)) = 45 / (pi * h^6) * (h - r)
	// viscosityLaplaceCoef = particleMass * viscosity * 45.0f / (pi * h^6)

	return viscosityLaplaceCoef * d * velDisp / rho_j;
}

extern "C" __global__ void
DensityKernel( hiprtGeometry geom, float* densities, const Particle* particles, Simulation* sim, hiprtFuncTable table )
{
	const uint32_t index	= blockIdx.x * blockDim.x + threadIdx.x;
	Particle	   particle = particles[index];

	hiprtRay ray;
	ray.origin	  = particle.Pos;
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.minT	  = 0.0f;
	ray.maxT	  = 0.0f;

	hiprtGeomCustomTraversalAnyHit tr( geom, ray, hiprtTraversalHintDefault, sim, table );

	float rho = 0.0f;
	while ( tr.getCurrentState() != hiprtTraversalStateFinished )
	{
		hiprtHit hit = tr.getNextHit();
		if ( !hit.hasHit() ) continue;

		rho += calculateDensity( hit.t, sim->m_smoothRadius, sim->m_densityCoef );
	}

	densities[index] = rho;
}

extern "C" __global__ void ForceKernel(
	hiprtGeometry	geom,
	float3*			accelerations,
	const Particle* particles,
	const float*	densities,
	Simulation*		sim,
	hiprtFuncTable	table )
{
	const uint32_t index	= blockIdx.x * blockDim.x + threadIdx.x;
	Particle	   particle = particles[index];
	float		   rho		= densities[index];

	hiprtRay ray;
	ray.origin	  = particle.Pos;
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.minT	  = 0.0f;
	ray.maxT	  = 0.0f;

	float pressure = calculatePressure( rho, sim->m_restDensity, sim->m_pressureStiffness );

	hiprtGeomCustomTraversalAnyHit tr( geom, ray, hiprtTraversalHintDefault, sim, table );

	float3 force = make_float3( 0.0f );
	while ( tr.getCurrentState() != hiprtTraversalStateFinished )
	{
		hiprtHit hit = tr.getNextHit();
		if ( !hit.hasHit() ) continue;
		if ( hit.primID == index ) continue;

		Particle hitParticle = particles[hit.primID];
		float	 hitRho		 = densities[hit.primID];

		float3 disp		   = hit.normal;
		float  r		   = sqrtf( hit.t );
		float  d		   = sim->m_smoothRadius - r;
		float  hitPressure = calculatePressure( hitRho, sim->m_restDensity, sim->m_pressureStiffness );

		force += calculateGradPressure( r, d, pressure, hitPressure, hitRho, disp, sim->m_pressureGradCoef );
		force += calculateVelocityLaplace( d, particle.Velocity, hitParticle.Velocity, hitRho, sim->m_viscosityLaplaceCoef );
	}

	accelerations[index] = rho > 0.0f ? force / rho : make_float3( 0.0f );
}

extern "C" __global__ void IntegrationKernel(
	Particle* particles, Aabb* particleAabbs, const float3* accelerations, const Simulation* sim, const PerFrame* perFrame )
{
	const uint32_t index		= blockIdx.x * blockDim.x + threadIdx.x;
	Particle	   particle		= particles[index];
	float3		   acceleration = accelerations[index];

	// Apply the forces from the map walls
	for ( uint32_t i = 0; i < 6; ++i )
	{
		float d = dot( make_float4( particle.Pos, 1.0f ), sim->m_planes[i] );
		acceleration += min( d, 0.0f ) * -sim->m_wallStiffness * make_float3( sim->m_planes[i] );
	}

	// Apply gravity
	acceleration += perFrame->m_gravity;

	// Integrate
	particle.Velocity += perFrame->m_timeStep * acceleration;
	particle.Pos += perFrame->m_timeStep * particle.Velocity;

	Aabb aabb;
	aabb.m_min = particle.Pos - sim->m_smoothRadius;
	aabb.m_max = particle.Pos + sim->m_smoothRadius;

	// Update
	particles[index]	 = particle;
	particleAabbs[index] = aabb;
}

extern "C" __global__ void
VisualizationKernel( const Particle* particles, const float* densities, uint8_t* pixels, int2 res, const float4x4* viewProj )
{
	const uint32_t index	= blockIdx.x * blockDim.x + threadIdx.x;
	Particle	   particle = particles[index];
	float		   rho		= densities[index];

	// To clip space
	float4 pos = ( *viewProj ) * make_float4( particle.Pos, 1.0f );

	// Normalize to NDC
	pos.x = pos.x / pos.w;
	pos.y = pos.y / pos.w;
	pos.z = pos.z / pos.w;

	// To viewport
	int x = ( pos.x * 0.5f + 0.5f ) * res.x;
	int y = ( 0.5f - pos.y * 0.5f ) * res.y;

	float visRho = rho / 4000.0f;

	int pixelIndex			   = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = visRho * 255.0f;
	pixels[pixelIndex * 4 + 1] = 0;
	pixels[pixelIndex * 4 + 2] = ( 1.0f - visRho ) * 255.0f;
	pixels[pixelIndex * 4 + 3] = 255;
}
