__device__ bool intersectCircle(
	const hiprtRay& ray,
	unsigned int	primIdx,
	const void*		userPtr,
	const void*		payload,
	float2&			uvOut,
	float3&			normalOut,
	float&			tOut )
{
	const float* o = (const float*)userPtr;
	float		 r = 0.1f;
	float2		 c = { o[primIdx], 0.5f };
	c.x			   = c.x - ray.origin.x;
	c.y			   = c.y - ray.origin.y;
	float d		   = sqrtf( c.x * c.x + c.y * c.y );

	bool hit = d < r;
	return hit;
}

__device__ hiprtIntersectFunc circleFunc = intersectCircle;

extern "C" __global__ void CustomIntersectionKernel( hiprtGeometry geom, unsigned char* gDst, hiprtCustomFuncTable table, int2 cRes )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	hiprtRay ray;
	ray.origin	  = { gIdx / (float)cRes.x, gIdy / (float)cRes.y, -1.f };
	ray.direction = { 0.f, 0.f, 1.f };
	ray.maxT	  = 1000.f;

	int								hitIdx = hiprtInvalidValue;
	hiprtGeomCustomTraversalClosest tr( geom, ray, *(hiprtCustomFuncSet*)table );
	hiprtHit						hit = tr.getNextHit();
	if ( hit.hasHit() ) hitIdx = hit.primID;

	int3 colors[]		 = { { 255, 0, 128 }, { 0, 128, 255 }, { 128, 255, 0 } };
	int	 dstIdx			 = gIdx + gIdy * cRes.x;
	gDst[dstIdx * 4 + 0] = hitIdx != hiprtInvalidValue ? colors[hitIdx].x : 0;
	gDst[dstIdx * 4 + 1] = hitIdx != hiprtInvalidValue ? colors[hitIdx].y : 0;
	gDst[dstIdx * 4 + 2] = hitIdx != hiprtInvalidValue ? colors[hitIdx].z : 0;
	gDst[dstIdx * 4 + 3] = 255;
}

