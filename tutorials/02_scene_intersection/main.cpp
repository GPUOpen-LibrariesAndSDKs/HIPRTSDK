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

#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	void run()
	{
		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount		   = 2;
		mesh.triangleStride		   = sizeof( hiprtInt3 );
		uint32_t triangleIndices[] = { 0, 1, 2, 3, 4, 5 };
		CHECK_ORO(
			oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh.triangleIndices ), mesh.triangleCount * sizeof( hiprtInt3 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			reinterpret_cast<oroDeviceptr>( mesh.triangleIndices ),
			triangleIndices,
			mesh.triangleCount * sizeof( hiprtInt3 ) ) );

		mesh.vertexCount		   = 6;
		mesh.vertexStride		   = sizeof( hiprtFloat3 );
		constexpr float S		   = 0.5f;
		constexpr float T		   = 0.8f;
		hiprtFloat3		vertices[] = {
			{ S, S, 0.0f },
			{ S + T * S, -S * S, 0.0f },
			{ S - T * S, -S * S, 0.0f },
			{ -S, S, 0.0f },
			{ -S + T * S, -S * S, 0.0f },
			{ -S - T * S, -S * S, 0.0f } };
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh.vertices ), mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			reinterpret_cast<oroDeviceptr>( mesh.vertices ), vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

		hiprtGeometry geom;
		CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
		CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

		hiprtInstance instance;
		instance.type	  = hiprtInstanceTypeGeometry;
		instance.geometry = geom;

		hiprtSceneBuildInput sceneInput;
		sceneInput.instanceCount			= 1;
		sceneInput.instanceMasks			= nullptr;
		sceneInput.instanceTransformHeaders = nullptr;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneInput.instances ), sizeof( hiprtInstance ) ) );
		CHECK_ORO(
			oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( sceneInput.instances ), &instance, sizeof( hiprtInstance ) ) );

		hiprtFrameSRT frame;
		frame.translation	  = make_hiprtFloat3( 0.0f, 0.0f, 0.0f );
		frame.scale			  = make_hiprtFloat3( 0.5f, 0.5f, 0.5f );
		frame.rotation		  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
		sceneInput.frameCount = 1;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceFrames ), sizeof( hiprtFrameSRT ) ) );
		CHECK_ORO(
			oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( sceneInput.instanceFrames ), &frame, sizeof( hiprtFrameSRT ) ) );

		size_t		   sceneTempSize;
		hiprtDevicePtr sceneTemp = nullptr;
		CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneTemp ), sceneTempSize ) );

		hiprtScene scene = nullptr;
		CHECK_HIPRT( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
		CHECK_HIPRT( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

		// we can free the Temporary Buffer just after hiprtBuildScene, as this buffer is only used during the build.
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneTemp ) ) );
		sceneTemp = nullptr;

		oroFunction func = nullptr;
		buildTraceKernelFromBitcode( ctxt, "../common/TutorialKernels.h", "SceneIntersectionKernel", func );

		uint8_t* pixels = nullptr;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pixels ), m_res.x * m_res.y * 4 ) );

		void* args[] = { &scene, &pixels, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "02_scene_intersection.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInput.instances ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInput.instanceFrames ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh.triangleIndices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh.vertices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pixels ) ) );

		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom ) );
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
