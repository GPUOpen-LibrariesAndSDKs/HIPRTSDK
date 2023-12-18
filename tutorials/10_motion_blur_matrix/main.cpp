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

		hiprtGeometry			   geom0;
		hiprtTriangleMeshPrimitive mesh0;
		{
			mesh0.triangleCount		   = 1;
			mesh0.triangleStride	   = sizeof( hiprtInt3 );
			uint32_t triangleIndices[] = { 0, 1, 2 };
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &mesh0.triangleIndices ), mesh0.triangleCount * sizeof( hiprtInt3 ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( mesh0.triangleIndices ),
				triangleIndices,
				mesh0.triangleCount * sizeof( hiprtInt3 ) ) );

			mesh0.vertexCount		   = 3;
			mesh0.vertexStride		   = sizeof( hiprtFloat3 );
			constexpr float Scale	   = 0.15f;
			hiprtFloat3		vertices[] = {
				{ Scale * sinf( 0.0f ), Scale * cosf( 0.0f ), 0.0f },
				{ Scale * sinf( hiprt::Pi * 2.0f / 3.0f ), Scale * cosf( hiprt::Pi * 2.0f / 3.0f ), 0.0f },
				{ Scale * sinf( hiprt::Pi * 4.0f / 3.0f ), Scale * cosf( hiprt::Pi * 4.0f / 3.0f ), 0.0f } };
			CHECK_ORO(
				oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh0.vertices ), mesh0.vertexCount * sizeof( hiprtFloat3 ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( mesh0.vertices ), vertices, mesh0.vertexCount * sizeof( hiprtFloat3 ) ) );

			hiprtGeometryBuildInput geomInput;
			geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInput.primitive.triangleMesh = mesh0;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom0 ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom0 ) );
			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );
		}

		hiprtGeometry			   geom1;
		hiprtTriangleMeshPrimitive mesh1;
		{
			mesh1.triangleCount		   = 1;
			mesh1.triangleStride	   = sizeof( hiprtInt3 );
			uint32_t triangleIndices[] = { 0, 1, 2 };
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &mesh1.triangleIndices ), mesh1.triangleCount * sizeof( hiprtInt3 ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( mesh1.triangleIndices ),
				triangleIndices,
				mesh1.triangleCount * sizeof( hiprtInt3 ) ) );

			mesh1.vertexCount		   = 3;
			mesh1.vertexStride		   = sizeof( hiprtFloat3 );
			constexpr float Scale	   = 0.15f;
			hiprtFloat3		vertices[] = {
				{ Scale * sinf( 0.0f ), Scale * cosf( 0.0f ), 0.0f },
				{ Scale * sinf( hiprt::Pi * 2.0f / 3.0f ), Scale * cosf( hiprt::Pi * 2.0f / 3.0f ), 0.0f },
				{ Scale * sinf( hiprt::Pi * 4.0f / 3.0f ), Scale * cosf( hiprt::Pi * 4.0f / 3.0f ), 0.0f } };
			CHECK_ORO(
				oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh1.vertices ), mesh1.vertexCount * sizeof( hiprtFloat3 ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( mesh1.vertices ), vertices, mesh1.vertexCount * sizeof( hiprtFloat3 ) ) );

			hiprtGeometryBuildInput geomInput;
			geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInput.primitive.triangleMesh = mesh1;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom1 ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom1 ) );
			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );
		}

		hiprtScene			 scene;
		hiprtSceneBuildInput sceneInput;
		{
			hiprtInstance instance0;
			instance0.type	   = hiprtInstanceTypeGeometry;
			instance0.geometry = geom0;

			hiprtInstance instance1;
			instance1.type	   = hiprtInstanceTypeGeometry;
			instance1.geometry = geom1;

			hiprtInstance instances[] = { instance0, instance1 };

			sceneInput.instanceCount = 2;
			sceneInput.instanceMasks = nullptr;
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInput.instances ),
				sceneInput.instanceCount * sizeof( hiprtInstance ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInput.instances ),
				instances,
				sceneInput.instanceCount * sizeof( hiprtInstance ) ) );

			constexpr float	 Offset = 0.3f;
			hiprtFrameMatrix matrices[5];

			matrices[0]				 = hiprtFrameMatrix{};
			matrices[0].matrix[0][0] = 1.0f;
			matrices[0].matrix[1][1] = 1.0f;
			matrices[0].matrix[2][2] = 1.0f;
			matrices[0].matrix[0][3] = -0.25f;
			matrices[0].matrix[1][3] = -Offset;
			matrices[0].matrix[2][3] = 0.0f;
			matrices[0].time		 = 0.0f;

			matrices[1]				 = hiprtFrameMatrix{};
			matrices[1].matrix[0][0] = 1.0f;
			matrices[1].matrix[1][1] = 1.0f;
			matrices[1].matrix[2][2] = 1.0f;
			matrices[1].matrix[0][3] = 0.0f;
			matrices[1].matrix[1][3] = -Offset;
			matrices[1].matrix[2][3] = 0.0f;
			matrices[1].time		 = 0.35f;

			matrices[2]				 = hiprtFrameMatrix{};
			matrices[2].matrix[0][0] = cosf( hiprt::Pi * 0.25f );
			matrices[2].matrix[0][1] = -sinf( hiprt::Pi * 0.25f );
			matrices[2].matrix[1][0] = sinf( hiprt::Pi * 0.25f );
			matrices[2].matrix[1][1] = cosf( hiprt::Pi * 0.25f );
			matrices[2].matrix[2][2] = 1.0f;
			matrices[2].matrix[0][3] = 0.25f;
			matrices[2].matrix[1][3] = -Offset;
			matrices[2].matrix[2][3] = 0.0f;
			matrices[2].time		 = 1.0f;

			matrices[3]				 = hiprtFrameMatrix{};
			matrices[3].matrix[0][0] = 1.0f;
			matrices[3].matrix[1][1] = 1.0f;
			matrices[3].matrix[2][2] = 1.0f;
			matrices[3].matrix[0][3] = 0.0f;
			matrices[3].matrix[1][3] = Offset;
			matrices[3].matrix[2][3] = 0.0f;
			matrices[3].time		 = 0.0f;

			matrices[4]				 = hiprtFrameMatrix{};
			matrices[4].matrix[0][0] = 0.0f;
			matrices[4].matrix[0][1] = -1.0f;
			matrices[4].matrix[1][0] = 1.0f;
			matrices[4].matrix[1][1] = 0.0f;
			matrices[4].matrix[2][2] = 1.0f;
			matrices[4].matrix[0][3] = 0.0f;
			matrices[4].matrix[1][3] = Offset;
			matrices[4].matrix[2][3] = 0.0f;
			matrices[4].time		 = 1.0f;

			sceneInput.frameCount = 5;
			sceneInput.frameType  = hiprtFrameTypeMatrix;
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceFrames ),
				sceneInput.frameCount * sizeof( hiprtFrameMatrix ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInput.instanceFrames ),
				matrices,
				sceneInput.frameCount * sizeof( hiprtFrameMatrix ) ) );

			hiprtTransformHeader headers[2];
			headers[0].frameIndex = 0;
			headers[0].frameCount = 3;
			headers[1].frameIndex = 3;
			headers[1].frameCount = 2;
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceTransformHeaders ),
				sceneInput.instanceCount * sizeof( hiprtTransformHeader ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInput.instanceTransformHeaders ),
				headers,
				sceneInput.instanceCount * sizeof( hiprtTransformHeader ) ) );

			size_t			  sceneTempSize;
			hiprtDevicePtr	  sceneTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneTemp ), sceneTempSize ) );

			CHECK_HIPRT( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
			CHECK_HIPRT( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );
			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneTemp ) ) );
		}

		oroFunction func;
		buildTraceKernelFromBitcode( ctxt, "../common/TutorialKernels.h", "MotionBlurKernel", func );

		uint8_t* pixels;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pixels ), m_res.x * m_res.y * 4 ) );

		void* args[] = { &scene, &pixels, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "10_motion_blur_matrix.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInput.instances ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInput.instanceFrames ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInput.instanceTransformHeaders ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh0.vertices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh0.triangleIndices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh1.vertices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh1.triangleIndices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pixels ) ) );

		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom0 ) );
		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom1 ) );
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
