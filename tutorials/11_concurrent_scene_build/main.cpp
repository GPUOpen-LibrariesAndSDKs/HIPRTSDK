//
// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <numeric>
#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	void run()
	{
		const int	circleCount = 3;
		hiprtFloat4 circles[]	= { { 0.25f, 0.5f, 0.0f, 0.1f }, { 0.5f, 0.5f, 0.0f, 0.1f }, { 0.75f, 0.5f, 0.0f, 0.1f } };

		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt ) );

		hiprtGeometry		   geomCircles;
		hiprtDevicePtr		   geomTempCircles;
		hiprtApiStream		   streamCircles;
		hiprtAABBListPrimitive list;
		{
			list.aabbCount	= circleCount;
			list.aabbStride = 2 * sizeof( hiprtFloat4 );
			hiprtFloat4 aabbs[2 * circleCount];
			for ( int i = 0; i < circleCount; i++ )
			{
				const hiprtFloat4& c = circles[i];
				aabbs[i * 2 + 0]	 = { c.x - c.w, c.y - c.w, c.z, 0.0f };
				aabbs[i * 2 + 1]	 = { c.x + c.w, c.y + c.w, c.z, 0.0f };
			}
			CHECK_ORO( oroMalloc( (oroDeviceptr*)&list.aabbs, 2 * list.aabbCount * sizeof( hiprtFloat4 ) ) );
			CHECK_ORO( oroMemcpyHtoD( (oroDeviceptr)list.aabbs, aabbs, 2 * list.aabbCount * sizeof( hiprtFloat4 ) ) );

			hiprtGeometryBuildInput geomInput;
			geomInput.type				 = hiprtPrimitiveTypeAABBList;
			geomInput.aabbList.primitive = &list;
			geomInput.geomType			 = 0;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize ) );
			CHECK_ORO( oroMalloc( (oroDeviceptr*)&geomTemp, geomTempSize ) );

			CHECK_ORO( oroStreamCreate( (oroStream*)&streamCircles ) );
			CHECK_HIPRT( hiprtCreateGeometry( ctxt, &geomInput, &options, &geomCircles ) );
			CHECK_HIPRT( hiprtBuildGeometry(
				ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, streamCircles, geomCircles ) );
			CHECK_ORO( oroFree( (oroDeviceptr)geomTemp ) );
		}

		hiprtGeometry			   geomTris;
		hiprtDevicePtr			   geomTempTris;
		hiprtApiStream			   streamTris;
		hiprtTriangleMeshPrimitive mesh;
		{
			mesh.triangleCount	= 3;
			mesh.triangleStride = sizeof( hiprtInt3 );
			std::vector<int> triangleIndices( 3 * mesh.triangleCount );
			std::iota( triangleIndices.begin(), triangleIndices.end(), 0 );
			CHECK_ORO( oroMalloc( (oroDeviceptr*)&mesh.triangleIndices, mesh.triangleCount * sizeof( hiprtInt3 ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				(oroDeviceptr)mesh.triangleIndices, triangleIndices.data(), mesh.triangleCount * sizeof( hiprtInt3 ) ) );

			mesh.vertexCount	   = 9;
			mesh.vertexStride	   = sizeof( hiprtFloat3 );
			hiprtFloat3 vertices[] = {
				{ 0.15f, 0.40f, 0.0f },
				{ 0.35f, 0.40f, 0.0f },
				{ 0.25f, 0.60f, 0.0f },
				{ 0.40f, 0.40f, 0.0f },
				{ 0.60f, 0.40f, 0.0f },
				{ 0.50f, 0.60f, 0.0f },
				{ 0.65f, 0.40f, 0.0f },
				{ 0.85f, 0.40f, 0.0f },
				{ 0.75f, 0.60f, 0.0f } };
			CHECK_ORO( oroMalloc( (oroDeviceptr*)&mesh.vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
			CHECK_ORO( oroMemcpyHtoD( (oroDeviceptr)mesh.vertices, vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

			hiprtGeometryBuildInput geomInput;
			geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInput.triangleMesh.primitive = &mesh;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize ) );
			CHECK_ORO( oroMalloc( (oroDeviceptr*)&geomTemp, geomTempSize ) );

			CHECK_ORO( oroStreamCreate( (oroStream*)&streamTris ) );
			CHECK_HIPRT( hiprtCreateGeometry( ctxt, &geomInput, &options, &geomTris ) );
			CHECK_HIPRT(
				hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, streamTris, geomTris ) );
			CHECK_ORO( oroFree( (oroDeviceptr)geomTemp ) );
		}

		CHECK_ORO( oroDeviceSynchronize() );

		hiprtScene			 scene;
		hiprtSceneBuildInput sceneInput;
		{
			sceneInput.instanceCount			= 2;
			sceneInput.instanceMasks			= nullptr;
			sceneInput.instanceTransformHeaders = nullptr;
			hiprtDevicePtr geoms[]				= { geomCircles, geomTris };
			CHECK_ORO( oroMalloc(
				(oroDeviceptr*)&sceneInput.instanceGeometries, sceneInput.instanceCount * sizeof( hiprtDevicePtr ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				(oroDeviceptr)sceneInput.instanceGeometries, geoms, sceneInput.instanceCount * sizeof( hiprtDevicePtr ) ) );

			const float	  o = 0.05f;
			hiprtFrameSRT frames[2];
			frames[0].translation = make_hiprtFloat3( 0.0f, o, 0.0f );
			frames[0].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[0].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
			frames[1].translation = make_hiprtFloat3( 0.0f, -o, 0.0f );
			frames[1].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[1].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );

			sceneInput.frameCount = 2;
			CHECK_ORO(
				oroMalloc( (oroDeviceptr*)&sceneInput.instanceFrames, sceneInput.frameCount * sizeof( hiprtFrameSRT ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				(oroDeviceptr)sceneInput.instanceFrames, frames, sceneInput.frameCount * sizeof( hiprtFrameSRT ) ) );

			size_t			  sceneTempSize;
			hiprtDevicePtr	  sceneTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( ctxt, &sceneInput, &options, &sceneTempSize ) );
			CHECK_ORO( oroMalloc( (oroDeviceptr*)&sceneTemp, sceneTempSize ) );

			CHECK_HIPRT( hiprtCreateScene( ctxt, &sceneInput, &options, &scene ) );
			CHECK_HIPRT( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, &sceneInput, &options, sceneTemp, 0, scene ) );
			CHECK_ORO( oroFree( (oroDeviceptr)sceneTemp ) );
		}

		hiprtFuncNameSet funcNameSet;
		funcNameSet.intersectFuncName			   = "intersectCircle";
		std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

		oroFunction func;
		buildTraceKernelFromBitcode(
			ctxt, "../common/TutorialKernels.h", "ConcurrentSceneBuildKernel", func, nullptr, &funcNameSets, 1, 1 );

		hiprtFuncDataSet funcDataSet;
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&funcDataSet.intersectFuncData, circleCount * sizeof( hiprtFloat4 ) ) );
		CHECK_ORO( oroMemcpyHtoD( (oroDeviceptr)funcDataSet.intersectFuncData, circles, circleCount * sizeof( hiprtFloat4 ) ) );

		hiprtFuncTable funcTable;
		CHECK_HIPRT( hiprtCreateFuncTable( ctxt, 1, 1, &funcTable ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

		u8* pixels;
		CHECK_ORO( oroMalloc( (oroDeviceptr*)&pixels, m_res.x * m_res.y * 4 ) );
		CHECK_ORO( oroMemset( (oroDeviceptr)pixels, 0, m_res.x * m_res.y * 4 ) );

		void* args[] = { &scene, &pixels, &funcTable, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "11_concurrent_scene_build.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( oroFree( (oroDeviceptr)funcDataSet.intersectFuncData ) );
		CHECK_ORO( oroFree( (oroDeviceptr)sceneInput.instanceGeometries ) );
		CHECK_ORO( oroFree( (oroDeviceptr)sceneInput.instanceFrames ) );
		CHECK_ORO( oroFree( (oroDeviceptr)mesh.triangleIndices ) );
		CHECK_ORO( oroFree( (oroDeviceptr)mesh.vertices ) );
		CHECK_ORO( oroFree( (oroDeviceptr)list.aabbs ) );
		CHECK_ORO( oroFree( (oroDeviceptr)pixels ) );

		CHECK_ORO( oroStreamDestroy( (oroStream)streamCircles ) );
		CHECK_ORO( oroStreamDestroy( (oroStream)streamTris ) );

		CHECK_HIPRT( hiprtDestroyFuncTable( ctxt, funcTable ) );
		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geomTris ) );
		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geomCircles ) );
		CHECK_HIPRT( hiprtDestroyScene( ctxt, scene ) );
		CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
	}
};

int main( int argc, char** argv )
{
	Tutorial tutorial;
	tutorial.init( 0 );
	tutorial.run();

	return 0;
}
