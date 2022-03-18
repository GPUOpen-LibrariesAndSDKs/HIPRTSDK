#include <hiprt/hiprt_device.h>


__device__ float3 gammaCorrect( float3 a )
{
	float g = 1.f / 2.2f;
	return { pow( a.x, g ), pow( a.y, g ), pow( a.z, g ) };
}
extern "C"
__global__ void MotionBlurKernel( hiprtScene scene, unsigned char* gDst, int2 cRes )
{
	const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int gIdy = blockIdx.y * blockDim.y + threadIdx.y;

	const unsigned int samples = 32;

	hiprtRay ray;
	ray.origin	  = { gIdx / (float)cRes.x - 0.5f, gIdy / (float)cRes.y - 0.5f, -1.f };
	ray.direction = { 0.f, 0.f, 1.f };
	ray.time	  = 0.0f;
	ray.maxT	  = 1000.f;

	float3 colors[2] = { { 1.0f, 0.0f, 0.5f }, { 0.0f, 0.5f, 1.0f } };

	float3 color = { 0.0f, 0.0f, 0.0f };
	for ( int i = 0; i < samples; ++i )
	{
		ray.time = i / (float)samples;
		hiprtSceneTraversalClosest tr( scene, ray, 0xffffffff );
		hiprtHit				   hit = tr.getNextHit();
		if ( hit.hasHit() )
		{
			float3 diffuseColor = colors[hit.instanceID];
			color.x += diffuseColor.x;
			color.y += diffuseColor.y;
			color.z += diffuseColor.z;
		}
	}

	int dstIdx			 = gIdx + gIdy * cRes.x;
	color				 = gammaCorrect( color / samples );
	gDst[dstIdx * 4 + 0] = color.x * 255;
	gDst[dstIdx * 4 + 1] = color.y * 255;
	gDst[dstIdx * 4 + 2] = color.z * 255;
	gDst[dstIdx * 4 + 3] = 255;
}

