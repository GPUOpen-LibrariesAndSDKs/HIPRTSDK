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

		oroFunction func;
		buildTraceKernel( ctxt, "../01_geom_intersection/TestKernel.h", "MeshIntersectionKernel", func );

		u8* dst;
		dMalloc( dst, m_res.x * m_res.y * 4 );
		hiprtInt2 res = make_hiprtInt2( m_res.x, m_res.y );

		void* args[] = { &geom, &dst, &res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImageFromDevice(
			"MeshIntersection.png", m_res.x, m_res.y, dst );

		dFree( mesh.triangleIndices );
		dFree( mesh.vertices );
		dFree( geomTemp );
		dFree( dst );
		hiprtDestroyGeometry( ctxt, geom );
		hiprtDestroyContext( ctxt );
	}
};

int main( int argc, char** argv )
{
	Test test;
	test.init( 1 );
	test.run();

	return 0;
}


