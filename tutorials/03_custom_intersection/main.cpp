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

		hiprtAABBListPrimitive list;
		list.aabbCount	= 3;
		list.aabbStride = 8 * sizeof( float );
		dMalloc( (hiprtFloat4*&)list.aabbs, 6 );

		hiprtFloat4 b[] = {
			{ 0.15f, 0.40f, 0.0f, 0.0f },
			{ 0.35f, 0.60f, 0.0f, 0.0f },
			{ 0.40f, 0.40f, 0.0f, 0.0f },
			{ 0.60f, 0.60f, 0.0f, 0.0f },
			{ 0.65f, 0.40f, 0.0f, 0.0f },
			{ 0.85f, 0.60f, 0.0f, 0.0f } };
		dCopyHtoD( (hiprtFloat4*)list.aabbs, b, 6 );

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
		hiprtArray<char> binary;
		buildTraceKernel( ctxt, "../03_custom_intersection/TestKernel.h", "CustomIntersectionKernel", func, &binary );
		oroError ee = oroModuleLoadData( &module, binary.getPtr() );
		ASSERT( ee == oroSuccess );

		hiprtCustomFuncSet hCustomFuncSet;
		oroDeviceptr	   dFuncPtr;
		size_t			   numBytes = 0;

		ee = oroModuleGetGlobal( &dFuncPtr, &numBytes, module, "circleFunc" );
		ASSERT( ee == oroSuccess );
		oroMemcpyDtoH( &hCustomFuncSet.intersectFunc, dFuncPtr, numBytes );

		float* centers;
		dMalloc( centers, 3 );
		float h[] = { 0.25f, 0.5f, 0.75f };
		dCopyHtoD( centers, h, 3 );
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

		writeImageFromDevice( "CustomIntersection.png", m_res.x, m_res.y, dst );

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


