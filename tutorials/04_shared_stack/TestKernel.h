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

__global__ void CornellBoxKernel(
	hiprtGeometry  geom,
	unsigned char* gDst,
	int2		   cRes,
	int*		   globalStackBuffer,
	int			   stackSize )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	float3	 o	  = { 278.0f, 273.0f, -900.0f };
	float2	 d	  = { 2.0f * gIdx / (float)cRes.x - 1.0f, 2.0f * ( 1.0f - gIdy / (float)cRes.y ) - 1.0f };
	float3	 uvw  = { -387.817566f, -387.817566f, 1230.0f };
	ray.origin	  = o;
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction =
		ray.direction /
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );
	ray.maxT = 100000.0f;

	typedef hiprtCustomSharedStack<SHARED_STACK_SIZE> Stack;
	__shared__ int sharedStackBuffer[SHARED_STACK_SIZE * BLOCK_SIZE];

	int*  threadSharedStackBuffer = sharedStackBuffer + SHARED_STACK_SIZE * ( threadIdx.x + threadIdx.y * blockDim.x );
	int*  threadGlobalStackBuffer = globalStackBuffer + stackSize * ( gIdx + gIdy * cRes.x );
	Stack stack( threadGlobalStackBuffer, threadSharedStackBuffer );

	hiprtGeomTraversalClosestCustomStack<Stack> tr( geom, ray, stack );
	{
		hiprtHit hit = tr.getNextHit();

		int3 color = { 0, 0, 0 };
		if ( hit.hasHit() )
		{
			float3 n = hiprt::normalize( hit.normal );
			color.x	 = ( ( n.x + 1.0f ) * 0.5f ) * 255;
			color.y	 = ( ( n.y + 1.0f ) * 0.5f ) * 255;
			color.z	 = ( ( n.z + 1.0f ) * 0.5f ) * 255;
		}

		int dstIdx			 = gIdx + gIdy * cRes.x;
		gDst[dstIdx * 4 + 0] = min( 255, color.x );
		gDst[dstIdx * 4 + 1] = min( 255, color.y );
		gDst[dstIdx * 4 + 2] = min( 255, color.z );
		gDst[dstIdx * 4 + 3] = 255;
	}
}