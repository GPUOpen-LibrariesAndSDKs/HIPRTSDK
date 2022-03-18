#include <tutorials/common/TestBase.h>


class Test : public TestBase
{
  public:
	void run() 
	{
		hiprtContext ctxt;
		hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= 2;
		mesh.triangleStride = sizeof( int ) * 3;
		dMalloc( (char*&)mesh.triangleIndices, 3 * mesh.triangleCount * sizeof( int ) );
		int idx[] = { 0, 1, 2, 3, 4, 5 };
		dCopyHtoD( (int*)mesh.triangleIndices, idx, 3 * mesh.triangleCount );

		mesh.vertexCount  = 6;
		mesh.vertexStride = sizeof( hiprtFloat3 );
		dMalloc( (char*&)mesh.vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) );
		hiprtFloat3 v[] = {
			{ 0.0f, 0.0f, 0.0f },
			{ 1.0f, 0.0f, 0.0f },
			{ 0.5f, 1.0f, 0.0f },
			{ 0.0f, 0.0f, 1.0f },
			{ 1.0f, 0.0f, 1.0f },
			{ 0.5f, 1.0f, 1.0f } };
		dCopyHtoD( (hiprtFloat3*)mesh.vertices, v, mesh.vertexCount );
		waitForCompletion();

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.triangleMesh.primitive = &mesh;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize );
		dMalloc( (u8*&)geomTemp, geomTempSize );

		hiprtGeometry geom;
		hiprtCreateGeometry( ctxt, &geomInput, &options, &geom );
		hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, 0, geom );

		hiprtScene			 scene;
		hiprtDevicePtr		 sceneTemp;
		hiprtSceneBuildInput sceneInput;

		sceneInput.instanceCount			= 1;
		sceneInput.instanceMasks			= nullptr;
		sceneInput.instanceTransformHeaders = nullptr;
		dMalloc( (char*&)sceneInput.instanceGeometries, sceneInput.instanceCount * sizeof( void* ) );
		hiprtDevicePtr geoms[] = { geom };
		dCopyHtoD( (hiprtDevicePtr*)sceneInput.instanceGeometries, geoms, sceneInput.instanceCount );

		hiprtFrame frame;
		frame.translation	  = make_hiprtFloat3( 0.0f, 0.0f, 0.0f );
		frame.scale			  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
		frame.rotation		  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
		sceneInput.frameCount = 1;
		dMalloc( (hiprtFrame*&)sceneInput.instanceFrames, 1 );
		dCopyHtoD( (hiprtFrame*&)sceneInput.instanceFrames, &frame, 1 );

		size_t sceneTempSize;
		hiprtGetSceneBuildTemporaryBufferSize( ctxt, &sceneInput, &options, &sceneTempSize );
		dMalloc( (u8*&)sceneTemp, sceneTempSize );

		hiprtCreateScene( ctxt, &sceneInput, &options, &scene );
		hiprtBuildScene( ctxt, hiprtBuildOperationBuild, &sceneInput, &options, sceneTemp, 0, scene );

		oroFunction func;
		buildTraceKernel( ctxt, "../02_scene_intersection/TestKernel.h", "SceneIntersection", func );

		u8* dst;
		dMalloc( dst, m_res.x * m_res.y *  4 );
		hiprtInt2 res = make_hiprtInt2( m_res.x, m_res.y );

		void* args[] = { &scene, &dst, &res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImageFromDevice( "SceneIntersection.png", m_res.x, m_res.y, dst );

		dFree( sceneInput.instanceGeometries );
		dFree( sceneInput.instanceFrames );
		dFree( mesh.triangleIndices );
		dFree( mesh.vertices );
		dFree( geomTemp );
		dFree( dst );
		hiprtDestroyGeometry( ctxt, geom );
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


