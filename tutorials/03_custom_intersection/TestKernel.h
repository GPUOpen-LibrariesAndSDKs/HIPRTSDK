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
	float3	 o	  = { gIdx / (float)cRes.x, gIdy / (float)cRes.y, -1.f };
	float3	 d	  = { 0.f, 0.f, 1.f };
	ray.origin	  = o;
	ray.direction = d;
	ray.maxT	  = 1000.f;

	int								hitIdx = hiprtInvalidValue;
	hiprtGeomCustomTraversalClosest tr( geom, ray, *(hiprtCustomFuncSet*)table );
	hiprtHit						hit = tr.getNextHit();
	if ( hit.hasHit() ) hitIdx = hit.primID;

	int2 colors[]		 = { { 255, 0 }, { 0, 255 }, { 255, 255 } };
	int	 dstIdx			 = gIdx + gIdy * cRes.x;
	gDst[dstIdx * 4 + 0] = hitIdx != hiprtInvalidValue ? colors[hitIdx].x : 0;
	gDst[dstIdx * 4 + 1] = hitIdx != hiprtInvalidValue ? colors[hitIdx].y : 0;
	gDst[dstIdx * 4 + 2] = 0;
	gDst[dstIdx * 4 + 3] = 255;
}

