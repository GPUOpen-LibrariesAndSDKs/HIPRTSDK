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
		constexpr uint32_t CircleCount = 3u;
		hiprtFloat4		   circles[] = { { 0.25f, 0.5f, 0.0f, 0.1f }, { 0.5f, 0.5f, 0.0f, 0.1f }, { 0.75f, 0.5f, 0.0f, 0.1f } };

		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtGeometryBuildInput geomBuildInputs[2];
		hiprtGeometry			geoms[2];

		hiprtAABBListPrimitive list;
		{
			list.aabbCount	= CircleCount;
			list.aabbStride = 2 * sizeof( hiprtFloat4 );
			hiprtFloat4 aabbs[2 * CircleCount];
			for ( int i = 0; i < CircleCount; i++ )
			{
				const hiprtFloat4& c = circles[i];
				aabbs[i * 2 + 0]	 = { c.x - c.w, c.y - c.w, c.z, 0.0f };
				aabbs[i * 2 + 1]	 = { c.x + c.w, c.y + c.w, c.z, 0.0f };
			}
			CHECK_ORO(
				oroMalloc( reinterpret_cast<oroDeviceptr*>( &list.aabbs ), 2 * list.aabbCount * sizeof( hiprtFloat4 ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( list.aabbs ), aabbs, 2 * list.aabbCount * sizeof( hiprtFloat4 ) ) );

			geomBuildInputs[0].type				  = hiprtPrimitiveTypeAABBList;
			geomBuildInputs[0].primitive.aabbList = list;
			geomBuildInputs[0].geomType			  = 0;
		}

		hiprtTriangleMeshPrimitive mesh;
		{
			mesh.triangleCount	= 3;
			mesh.triangleStride = sizeof( hiprtInt3 );
			std::vector<int> triangleIndices( 3 * mesh.triangleCount );
			std::iota( triangleIndices.begin(), triangleIndices.end(), 0 );
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &mesh.triangleIndices ), mesh.triangleCount * sizeof( hiprtInt3 ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( mesh.triangleIndices ),
				triangleIndices.data(),
				mesh.triangleCount * sizeof( hiprtInt3 ) ) );

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
			CHECK_ORO(
				oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh.vertices ), mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( mesh.vertices ), vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

			geomBuildInputs[1].type					  = hiprtPrimitiveTypeTriangleMesh;
			geomBuildInputs[1].primitive.triangleMesh = mesh;
		}

		{
			hiprtBuildOptions options;
			options.buildFlags			   = hiprtBuildFlagBitPreferFastBuild;
			options.batchBuildMaxPrimCount = 16u; // this enables batch construction

			size_t		   geomTempSize;
			hiprtDevicePtr geomTemp;
			CHECK_HIPRT( hiprtGetGeometriesBuildTemporaryBufferSize( ctxt, 2, geomBuildInputs, options, geomTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

			hiprtGeometry* geomAddrs[] = { &geoms[0], &geoms[1] };
			CHECK_HIPRT( hiprtCreateGeometries( ctxt, 2, geomBuildInputs, options, geomAddrs ) );
			CHECK_HIPRT(
				hiprtBuildGeometries( ctxt, hiprtBuildOperationBuild, 2, geomBuildInputs, options, geomTemp, 0, geoms ) );
			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );
		}

		hiprtScene			 sceneMid;
		hiprtSceneBuildInput sceneInputMid;
		{
			hiprtInstance instance0;
			instance0.type	   = hiprtInstanceTypeGeometry;
			instance0.geometry = geoms[0];

			hiprtInstance instance1;
			instance1.type	   = hiprtInstanceTypeGeometry;
			instance1.geometry = geoms[1];

			hiprtInstance instances[] = { instance0, instance1 };

			sceneInputMid.instanceCount			   = 2;
			sceneInputMid.instanceMasks			   = nullptr;
			sceneInputMid.instanceTransformHeaders = nullptr;
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInputMid.instances ),
				sceneInputMid.instanceCount * sizeof( hiprtInstance ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInputMid.instances ),
				instances,
				sceneInputMid.instanceCount * sizeof( hiprtInstance ) ) );

			const float	  o = 0.05f;
			hiprtFrameSRT frames[2];
			frames[0].translation = make_hiprtFloat3( 0.0f, o, 0.0f );
			frames[0].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[0].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
			frames[1].translation = make_hiprtFloat3( 0.0f, -o, 0.0f );
			frames[1].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[1].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );

			sceneInputMid.frameCount = 2;
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInputMid.instanceFrames ),
				sceneInputMid.frameCount * sizeof( hiprtFrameSRT ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInputMid.instanceFrames ),
				frames,
				sceneInputMid.frameCount * sizeof( hiprtFrameSRT ) ) );

			size_t			  sceneTempSize;
			hiprtDevicePtr	  sceneTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInputMid, options, sceneTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneTemp ), sceneTempSize ) );

			CHECK_HIPRT( hiprtCreateScene( ctxt, sceneInputMid, options, sceneMid ) );
			CHECK_HIPRT( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInputMid, options, sceneTemp, 0, sceneMid ) );
			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneTemp ) ) );
		}

		hiprtScene			 sceneTop;
		hiprtSceneBuildInput sceneInputTop;
		{
			hiprtInstance instance;
			instance.type  = hiprtInstanceTypeScene;
			instance.scene = sceneMid;

			hiprtInstance instances[] = { instance, instance };

			sceneInputTop.instanceCount			   = 2;
			sceneInputTop.instanceMasks			   = nullptr;
			sceneInputTop.instanceTransformHeaders = nullptr;
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInputTop.instances ),
				sceneInputTop.instanceCount * sizeof( hiprtInstance ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInputTop.instances ),
				instances,
				sceneInputTop.instanceCount * sizeof( hiprtInstance ) ) );

			constexpr float Offset = 0.12f;
			hiprtFrameSRT	frames[2];
			frames[0].translation = make_float3( 0.0f, Offset, 0.0f );
			frames[0].scale		  = make_float3( 1.0f, 1.0f, 1.0f );
			frames[0].rotation	  = make_float4( 0.0f, 0.0f, 1.0f, 0.0f );
			frames[1].translation = make_float3( 0.0f, -Offset, 0.0f );
			frames[1].scale		  = make_float3( 1.0f, 1.0f, 1.0f );
			frames[1].rotation	  = make_float4( 0.0f, 0.0f, 1.0f, 0.0f );

			sceneInputTop.frameCount = 2;
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInputTop.instanceFrames ),
				sceneInputTop.frameCount * sizeof( hiprtFrameSRT ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInputTop.instanceFrames ),
				frames,
				sceneInputTop.frameCount * sizeof( hiprtFrameSRT ) ) );

			size_t			  sceneTempSize;
			hiprtDevicePtr	  sceneTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInputTop, options, sceneTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneTemp ), sceneTempSize ) );

			CHECK_HIPRT( hiprtCreateScene( ctxt, sceneInputTop, options, sceneTop ) );
			CHECK_HIPRT( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInputTop, options, sceneTemp, 0, sceneTop ) );
			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneTemp ) ) );
		}

		hiprtFuncNameSet funcNameSet;
		funcNameSet.intersectFuncName			   = "intersectCircle";
		std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

		oroFunction func;
		buildTraceKernelFromBitcode(
			ctxt, "../common/TutorialKernels.h", "SceneBuildKernel", func, nullptr, &funcNameSets, 1, 1 );

		hiprtFuncDataSet funcDataSet;
		CHECK_ORO(
			oroMalloc( const_cast<oroDeviceptr*>( &funcDataSet.intersectFuncData ), CircleCount * sizeof( hiprtFloat4 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			const_cast<oroDeviceptr>( funcDataSet.intersectFuncData ), circles, CircleCount * sizeof( hiprtFloat4 ) ) );

		hiprtFuncTable funcTable;
		CHECK_HIPRT( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

		uint8_t* pixels;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pixels ), m_res.x * m_res.y * 4 ) );
		CHECK_ORO( oroMemset( reinterpret_cast<oroDeviceptr>( pixels ), 0, m_res.x * m_res.y * 4 ) );

		void* args[] = { &sceneTop, &pixels, &funcTable, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "15_multi_level_instancing.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( oroFree( const_cast<oroDeviceptr>( funcDataSet.intersectFuncData ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInputTop.instances ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInputTop.instanceFrames ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInputMid.instances ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInputMid.instanceFrames ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh.triangleIndices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh.vertices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( list.aabbs ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pixels ) ) );

		CHECK_HIPRT( hiprtDestroyFuncTable( ctxt, funcTable ) );
		CHECK_HIPRT( hiprtDestroyGeometries( ctxt, 2, geoms ) );
		CHECK_HIPRT( hiprtDestroyScene( ctxt, sceneTop ) );
		CHECK_HIPRT( hiprtDestroyScene( ctxt, sceneMid ) );
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
