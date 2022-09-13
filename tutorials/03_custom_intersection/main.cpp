#include <tutorials/common/TestBase.h>
#include <cstdlib>
#include <numeric>

class Test : public TestBase
{
  public:
	void run() 
	{
		hiprtContext ctxt;
		hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );

		const int nSpheres = 8*2;
#if 0
		const hiprtFloat4 spheres[] = { 
			{ 0.25f, 0.5f, 0.f, 0.05f },
			{ 0.50f, 0.5f, 0.f, 0.1f },
			{ 0.75f, 0.5f, 0.f, 0.2f },
		};
#else

		hiprtFloat4 spheres[nSpheres];
		for(int i=0; i<nSpheres / 2; i++)
		{
			float r = 0.1f;
			float t = i/(float)(nSpheres/2) * 2.f * 3.1415f;
			spheres[i] = { sin(t) * 0.4f, cos(t) * 0.4f, 0.f, r };
		}
		for ( int i = 0; i < nSpheres / 2; i++ )
		{
			float r	   = 0.1f;
			float t	   = i / (float)(nSpheres/2) * 2.f * 3.1415f + 0.2f;
			spheres[i+nSpheres/2] = { sin( t ) * 0.35f, cos( t ) * 0.35f, 0.4f, r };
		}

#endif
		hiprtAABBListPrimitive list;
		list.aabbCount	= nSpheres;
		list.aabbStride = 8 * sizeof( float );
		dMalloc( (hiprtFloat4*&)list.aabbs, nSpheres*2 );

		hiprtFloat4 b[nSpheres*2];

		for ( int i = 0; i < nSpheres ; i++)
		{
			const hiprtFloat4& c = spheres[i];
			b[i * 2 + 0]		 = { c.x - c.w, c.y - c.w, c.z - c.w, 0.f };
			b[i * 2 + 1]		 = { c.x + c.w, c.y + c.w, c.z + c.w, 0.f };
		}

		dCopyHtoD( (hiprtFloat4*)list.aabbs, b, nSpheres*2 );

		hiprtGeometryBuildInput geomInput;
		geomInput.type				  = hiprtPrimitiveTypeAABBList;
		geomInput.aabbList.primitive  = &list;
		geomInput.aabbList.customType = 0;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize );
		dMalloc( (u8*&)geomTemp, geomTempSize );

		hiprtGeometry geom;
		hiprtCreateGeometry( ctxt, &geomInput, &options, &geom );
		hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, 0, geom );

		oroModule		 module;
		oroFunction		 func;
		std::vector<char> binary;
		buildTraceKernel( ctxt, "../03_custom_intersection/TestKernel.h", "CustomIntersectionKernel", func, &binary );
		oroError ee = oroModuleLoadData( &module, binary.data() );
		ASSERT( ee == oroSuccess );

		hiprtCustomFuncSet hCustomFuncSet;
		oroDeviceptr	   dFuncPtr;
		size_t			   numBytes = 0;

		ee = oroModuleGetGlobal( &dFuncPtr, &numBytes, module, "sphereIntersect" );
		ASSERT( ee == oroSuccess );
		oroMemcpyDtoH( &hCustomFuncSet.intersectFunc, dFuncPtr, numBytes );

		hiprtFloat4* centers;
		dMalloc( centers, nSpheres );
		dCopyHtoD( centers, (hiprtFloat4*)spheres, nSpheres );
		waitForCompletion();
		hCustomFuncSet.intersectFuncData = centers;

		hiprtCustomFuncTable dFuncSet;
		hiprtCreateCustomFuncTable( ctxt, &dFuncSet );
		hiprtSetCustomFuncTable( ctxt, dFuncSet, 0, hCustomFuncSet );

		u8* dst;
		dMalloc( dst, m_res.x * m_res.y * 4 );
		hiprtInt2 res = make_hiprtInt2( m_res.x, m_res.y );

		void* args[] = { &geom, &dst, &dFuncSet, &res };
		launchKernel( func, m_res.x, m_res.y, args );

		writeImageFromDevice( "03_custom_intersection.png", m_res.x, m_res.y, dst );

		dFree( list.aabbs );
		dFree( geomTemp );
		dFree( centers );
		dFree( dst );
		hiprtDestroyGeometry( ctxt, geom );
		hiprtDestroyContext( ctxt );
	}
};

int main( int argc, char** argv )
{
	Test test;
	test.init( 0 );
	test.run();

	return 0;
}


