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

extern "C" __global__ void SceneIntersection( hiprtScene scene, unsigned char* gDst, int2 cRes )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	float3	 o	  = { gIdx / (float)cRes.x - 0.5f, gIdy / (float)cRes.y - 0.5f, -1.f };
	float3	 d	  = { 0.0f, 0.0f, 1.0f };
	ray.origin	  = o;
	ray.direction = d;
	ray.maxT	  = 1000.f;

	hiprtSceneTraversalClosest tr( scene, ray, 0xffffffff );
	hiprtHit				   hit = tr.getNextHit();

	int dstIdx			 = gIdx + gIdy * cRes.x;
	gDst[dstIdx * 4 + 0] = hit.hasHit() ? ( (float)gIdx / cRes.x ) * 255 : 0;
	gDst[dstIdx * 4 + 1] = hit.hasHit() ? ( (float)gIdy / cRes.y ) * 255 : 0;
	gDst[dstIdx * 4 + 2] = 0;
	gDst[dstIdx * 4 + 3] = 255;
}
