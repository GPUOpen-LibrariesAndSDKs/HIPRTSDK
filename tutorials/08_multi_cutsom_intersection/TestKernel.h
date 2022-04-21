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

__device__ float dot3F4( const float4& a, const float4& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float  dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 normalize( const float3& a ) { return a / sqrtf( dot( a, a ) ); }

// check if there is a hit before ray.maxT. if there is, set it to tOut. hiprt will overwrite ray.maxT after this function
__device__ bool intersectCircle(
	const hiprtRay& ray,
	unsigned int	primIdx,
	const void*		userPtr,
	const void*		payload,
	float2&			uvOut,
	float3&			normalOut,
	float&			tOut )
{
	const float4* o		 = (const float4*)userPtr;
    float2  c = make_float2( o[primIdx].x, o[primIdx].y);
	const float	  r		 = o[primIdx].w;

	c.x			   = c.x - ray.origin.x;
	c.y			   = c.y - ray.origin.y;
	float d		   = sqrtf( c.x * c.x + c.y * c.y );
	bool hit = d < r;
	
	normalOut = normalize( make_float3( hit * d, hit * d, hit * d ) );

	return hit;
}

//check if there is a hit before ray.maxT. if there is, set it to tOut. hiprt will overwrite ray.maxT after this function
__device__ bool intersectSphere(
	const hiprtRay& ray,
	unsigned int	primIdx,
	const void*		userPtr,
	const void*		payload,
	float2&			uvOut,
	float3&			normalOut,
	float&			tOut )
{
	const float4 from = make_float4( ray.origin.x, ray.origin.y, ray.origin.z, 0.f );
	const float4  to	 = from + make_float4( ray.direction.x, ray.direction.y, ray.direction.z, 0.f ) * ray.maxT;
	const float4* o		 = (const float4*)userPtr;
	const float4  sphere = make_float4( o[primIdx].x, o[primIdx].y, o[primIdx].z, 0.f );
	const float	  r		 = o[primIdx].w;

	float4 m  = from - sphere;
	float4 d  = to - from;
	float  a  = dot3F4( d, d );
	float  b  = 2 * dot3F4( m, d );
	float  c  = dot3F4( m, m ) - r * r;
	float  dd = b * b - 4.f * a * c;
	if ( dd < 0.f ) return false;

	float t = ( -b - sqrtf( dd ) ) / ( 2.f * a );
	if( t > 1.f )
		return false;

	tOut	  = t * ray.maxT;
	float4 n  = from + make_float4( ray.direction.x, ray.direction.y, ray.direction.z, 0.f ) * tOut - sphere;
	normalOut = normalize( make_float3( n.x, n.y, n.z ) );
	return true;
}

__device__ hiprtIntersectFunc sphereIntersect = intersectSphere;
__device__ hiprtIntersectFunc circleIntersect = intersectCircle;

extern "C" __global__ void CustomIntersectionKernel( hiprtScene scene, unsigned char* gDst, hiprtCustomFuncTable table, int2 cRes )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	ray.origin	  = { gIdx / (float)cRes.x - 0.5f, gIdy / (float)cRes.y - 0.5f, -1.f };
	ray.direction = { 0.f, 0.f, 1.f };
	ray.maxT	  = 1000.f;

	hiprtSceneTraversalClosest tr( scene, ray, 0xffffffff, table );
	hiprtHit						hit = tr.getNextHit();

	int	 dstIdx			 = gIdx + gIdy * cRes.x;
	gDst[dstIdx * 4 + 0] = hit.hasHit() ? (hit.normal.x+1.f)/2.f * 255 : 0;
	gDst[dstIdx * 4 + 1] = hit.hasHit() ? (hit.normal.y+1.f)/2.f * 255 : 0;
	gDst[dstIdx * 4 + 2] = hit.hasHit() ? (hit.normal.z+1.f)/2.f * 255 : 0;
	gDst[dstIdx * 4 + 3] = 255;
}

