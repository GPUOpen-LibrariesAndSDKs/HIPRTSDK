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

__device__ bool cutoutFilter(
	const hiprtRay& ray,
	unsigned int	instanceIdx,
	unsigned int	primIdx,
	const void*		data,
	void*			payload,
	float2			uv,
	float3			normal,
	float			t )
{
	const float SCALE = 16.0f;
	float2		texCoord[2];
	texCoord[0] = ( 1.0f - uv.x - uv.y ) * make_float2( 0.0f, 0.0f ) + uv.x * make_float2( 0.0f, 1.0f ) +
				  uv.y * make_float2( 1.0f, 1.0f );
	texCoord[1] = ( 1.0f - uv.x - uv.y ) * make_float2( 0.0f, 0.0f ) + uv.x * make_float2( 1.0f, 1.0f ) +
				  uv.y * make_float2( 1.0f, 0.0f );
	if ( ( int( SCALE * texCoord[primIdx].x ) + int( SCALE * texCoord[primIdx].y ) ) & 1 ) return true;
	return false;
}

__device__ hiprtFilterFunc	  filterFunc1 = cutoutFilter;

__global__ void CutoutKernel( hiprtGeometry geom, unsigned char* gDst, hiprtCustomFuncTable table, int2 cRes )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	float3	 o	  = { gIdx / (float)cRes.x, gIdy / (float)cRes.y, -1.0f };
	float3	 d	  = { 0.0f, 0.0f, 1.0f };
	ray.origin	  = o;
	ray.direction = d;
	ray.maxT	  = 1000.f;

	hiprtGeomTraversalClosest tr( geom, ray, *(hiprtCustomFuncSet*)table );
	hiprtHit				  hit = tr.getNextHit();

	int dstIdx			 = gIdx + gIdy * cRes.x;
	gDst[dstIdx * 4 + 0] = hit.hasHit() ? 255 : 0;
	gDst[dstIdx * 4 + 1] = hit.hasHit() ? 255 : 0;
	gDst[dstIdx * 4 + 2] = hit.hasHit() ? 255 : 0;
	gDst[dstIdx * 4 + 3] = 255;
}
