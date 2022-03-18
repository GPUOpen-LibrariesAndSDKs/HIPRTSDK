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
