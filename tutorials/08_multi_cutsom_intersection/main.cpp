#include <tutorials/common/TestBase.h>
#include <cstdlib>
#include <numeric>

class Test : public TestBase
{
  public:
	void run() 
	{
		constexpr int sphereCustomTypeIndex = 0;
		constexpr int circleCustomTypeIndex = 1;

		hiprtContext ctxt;
		hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );
	
		//define shapes 
		const int nSpheres = 1;
		const hiprtFloat4 spheres[] = { 
			{ -0.3f, 0.0f, 0.f, 0.2f },
		};

		const int nCircles = 1;
		const hiprtFloat4 circles[] = { { 0.3f, 0.0f, 0.f, 0.2f } };

		// create sphere geometry object
		hiprtGeometry sphereGeom;
		hiprtAABBListPrimitive sphereAabbList;
		{
			sphereAabbList.aabbCount = nSpheres;
			sphereAabbList.aabbStride = 8 * sizeof( float );
			dMalloc( (hiprtFloat4*&)sphereAabbList.aabbs, nSpheres * 2 );

			hiprtFloat4 b[nSpheres * 2];
			for ( int i = 0; i < nSpheres; i++ )
			{
				const hiprtFloat4& c = spheres[i];
				b[i * 2 + 0]		 = { c.x - c.w, c.y - c.w, c.z - c.w, 0.f };
				b[i * 2 + 1]		 = { c.x + c.w, c.y + c.w, c.z + c.w, 0.f };
			}

			dCopyHtoD( (hiprtFloat4*)sphereAabbList.aabbs, b, nSpheres * 2 );

			hiprtGeometryBuildInput geomInput;
			geomInput.type				  = hiprtPrimitiveTypeAABBList;
			geomInput.aabbList.primitive  = &sphereAabbList;
			geomInput.aabbList.customType = sphereCustomTypeIndex;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize );
			dMalloc( (u8*&)geomTemp, geomTempSize );

			hiprtCreateGeometry( ctxt, &geomInput, &options, &sphereGeom);
			hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, 0, sphereGeom );
			dFree( geomTemp );
		}

		//create Circle geometry object
		hiprtGeometry		   circleGeom;
		hiprtAABBListPrimitive circleAabbList;
		{
			circleAabbList.aabbCount  = nCircles;
			circleAabbList.aabbStride = 8 * sizeof( float );
			dMalloc( (hiprtFloat4*&)circleAabbList.aabbs, nCircles * 2 );

			hiprtFloat4 b[nCircles * 2];
			for ( int i = 0; i < nCircles; i++ )
			{
				const hiprtFloat4& c = circles[i];
				b[i * 2 + 0]		 = { c.x - c.w, c.y - c.w, c.z - c.w, 0.f };
				b[i * 2 + 1]		 = { c.x + c.w, c.y + c.w, c.z + c.w, 0.f };
			}

			dCopyHtoD( (hiprtFloat4*)circleAabbList.aabbs, b, nCircles * 2 );

			hiprtGeometryBuildInput geomInput;
			geomInput.type				  = hiprtPrimitiveTypeAABBList;
			geomInput.aabbList.primitive  = &circleAabbList;
			geomInput.aabbList.customType = circleCustomTypeIndex;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize );
			dMalloc( (u8*&)geomTemp, geomTempSize );

			hiprtCreateGeometry( ctxt, &geomInput, &options, &circleGeom );
			hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, 0, circleGeom );
			dFree( geomTemp );
		}

		// create scene 
		hiprtScene			 scene;
		hiprtDevicePtr		 sceneTemp;
		hiprtSceneBuildInput sceneInput;

		sceneInput.instanceCount			= 2;
		sceneInput.instanceMasks			= nullptr;
		sceneInput.instanceTransformHeaders = nullptr;
		dMalloc( (char*&)sceneInput.instanceGeometries, sceneInput.instanceCount * sizeof( void* ) );
		hiprtDevicePtr geoms[] = { sphereGeom, circleGeom };
		dCopyHtoD( (hiprtDevicePtr*)sceneInput.instanceGeometries, geoms, sceneInput.instanceCount );

		std::vector<hiprtFrame> frames;
		hiprtFrame frame;
		frame.translation	  = make_hiprtFloat3( 0.0f, 0.0f, 0.0f );
		frame.scale			  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
		frame.rotation		  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
		sceneInput.frameCount = sceneInput.instanceCount;
		
		for ( int i = 0; i < sceneInput.instanceCount; i++ )
		{
			frames.push_back( frame);
		}
		dMalloc( (hiprtFrame*&)sceneInput.instanceFrames, frames.size() );
		dCopyHtoD( (hiprtFrame*&)sceneInput.instanceFrames, frames.data(), frames.size() );

		size_t sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		hiprtGetSceneBuildTemporaryBufferSize( ctxt, &sceneInput, &options, &sceneTempSize );
		dMalloc( (u8*&)sceneTemp, sceneTempSize );

		hiprtCreateScene( ctxt, &sceneInput, &options, &scene );
		hiprtBuildScene( ctxt, hiprtBuildOperationBuild, &sceneInput, &options, sceneTemp, 0, scene );

		oroModule		 module;
		oroFunction		 func;
		std::vector<char> binary;
		buildTraceKernel( ctxt, "../08_multi_cutsom_intersection/TestKernel.h", "CustomIntersectionKernel", func, &binary );
		oroError ee = oroModuleLoadData( &module, binary.data() );
		ASSERT( ee == oroSuccess );

		//get function pointer to sphere intersector
		hiprtCustomFuncSet hCustomFuncSphere;
		oroDeviceptr	   dSphereFuncPtr;
		size_t			   numBytes = 0;

		ee = oroModuleGetGlobal( &dSphereFuncPtr, &numBytes, module, "sphereIntersect" );
		ASSERT( ee == oroSuccess );
		oroMemcpyDtoH( &hCustomFuncSphere.intersectFunc, dSphereFuncPtr, numBytes );

		hiprtFloat4* sphereCentres;
		dMalloc( sphereCentres, nSpheres );
		dCopyHtoD( sphereCentres, (hiprtFloat4*)spheres, nSpheres );
		waitForCompletion();
		hCustomFuncSphere.intersectFuncData = sphereCentres;

		//get function pointer to circle intersector
		hiprtCustomFuncSet hCustomFuncCircle;
		oroDeviceptr	   dcircleFuncPtr;
		numBytes = 0;

		ee = oroModuleGetGlobal( &dcircleFuncPtr, &numBytes, module, "circleIntersect" );
		ASSERT( ee == oroSuccess );
		oroMemcpyDtoH( &hCustomFuncCircle.intersectFunc, dcircleFuncPtr, numBytes );

		hiprtFloat4* circleCentres;
		dMalloc( circleCentres, nCircles );
		dCopyHtoD( circleCentres, (hiprtFloat4*)circles, nCircles );
		waitForCompletion();
		hCustomFuncCircle.intersectFuncData = circleCentres;

		// prepare function table
		hiprtCustomFuncTable dFuncSet;
		hiprtCreateCustomFuncTable( ctxt, &dFuncSet );
		//please note : this customTypeIndices must match with one filled in geomInput.aabbList.customType
		hiprtSetCustomFuncTable( ctxt, dFuncSet, sphereCustomTypeIndex, hCustomFuncSphere );
		hiprtSetCustomFuncTable( ctxt, dFuncSet, circleCustomTypeIndex, hCustomFuncCircle );

		u8* dst;
		dMalloc( dst, m_res.x * m_res.y * 4 );
		hiprtInt2 res = make_hiprtInt2( m_res.x, m_res.y );

		void* args[] = { &scene, &dst, &dFuncSet, &res };
		launchKernel( func, m_res.x, m_res.y, args );

		writeImageFromDevice( "08_multi_cutsom_intersection.png", m_res.x, m_res.y, dst );

		dFree( sphereAabbList.aabbs );
		dFree( circleAabbList.aabbs );
		dFree( sceneInput.instanceFrames );
		dFree( sceneInput.instanceGeometries );
		dFree( sceneTemp );
		dFree( sphereCentres );
		dFree( circleCentres );
		dFree( dst );
		hiprtDestroyGeometry( ctxt, sphereGeom );
		hiprtDestroyGeometry( ctxt, circleGeom );
		hiprtDestroyScene( ctxt, scene );
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


