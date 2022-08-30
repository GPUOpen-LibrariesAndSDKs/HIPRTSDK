#include <tutorials/common/TestBase.h>
#include <cstdlib>
#include <numeric>

class Test : public TestBase
{
  public:
	void run(){
		hiprtContext ctxt;
		hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= 2;
		mesh.triangleStride = sizeof( int ) * 3;
		dMalloc( (char*&)mesh.triangleIndices, 3 * mesh.triangleCount * sizeof( int ) );
		int idx[] = { 0, 1, 2, 0, 2, 3 };
		dCopyHtoD( (int*)mesh.triangleIndices, idx, 3 * mesh.triangleCount );

		mesh.vertexCount  = 4;
		mesh.vertexStride = sizeof( float3 );
		dMalloc( (char*&)mesh.vertices, mesh.vertexCount * sizeof( float3 ) );
		float3 v[] = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } };
		dCopyHtoD( (float3*)mesh.vertices, v, mesh.vertexCount );
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

		oroModule		  module;
		oroFunction		  func;
		std::vector<char> binary;
		buildTraceKernel( ctxt, "../09_alpha_geom_cut/TestKernel.h", "CutoutKernel", func, &binary );

		oroError ee = oroModuleLoadData( &module, binary.data() );
		ASSERT( ee == oroSuccess );

		hiprtCustomFuncSet hCustomFuncSet;
		oroDeviceptr	   dFuncPtr;
		size_t			   numBytes = 0;

		ee = oroModuleGetGlobal( &dFuncPtr, &numBytes, module, "filterFunc1" );
		ASSERT( ee == oroSuccess );
		oroMemcpyDtoH( &hCustomFuncSet.filterFunc, dFuncPtr, numBytes );

		hiprtCustomFuncTable dFuncSet;
		hiprtCreateCustomFuncTable( ctxt, &dFuncSet );
		hiprtSetCustomFuncTable( ctxt, dFuncSet, 0, hCustomFuncSet );

		u8* dst;
		dMalloc( dst, m_res.x * m_res.y * 4 );
		hiprtInt2 res = make_hiprtInt2( m_res.x, m_res.y );

		void* args[] = { &geom, &dst, &res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImageFromDevice( "09_alpha_geom_cut.png", m_res.x, m_res.y, dst );

		dFree( mesh.triangleIndices );
		dFree( mesh.vertices );
		dFree( geomTemp );
		dFree( dst );
		hiprtDestroyCustomFuncTable( ctxt, dFuncSet );
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


