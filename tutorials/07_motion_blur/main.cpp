//
// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <tutorials/common/TestBase.h>
#include <numeric>

class Test : public TestBase
{
  public:
	void run() 
	{
		hiprtContext ctxt;
		hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );

		hiprtGeometry			   geomTris0;
		hiprtDevicePtr			   geomTempTris0;
		hiprtTriangleMeshPrimitive mesh0;
		{
			mesh0.triangleCount	 = 1;
			mesh0.triangleStride = sizeof( int ) * 3;
			dMalloc( (char*&)mesh0.triangleIndices, 3 * mesh0.triangleCount * sizeof( int ) );
			std::vector<int> idx( 3 * mesh0.triangleCount );
			std::iota( idx.begin(), idx.end(), 0 );
			dCopyHtoD( (int*)mesh0.triangleIndices, idx.data(), 3 * mesh0.triangleCount );

			mesh0.vertexCount  = 3;
			mesh0.vertexStride = sizeof( hiprtFloat3 );
			dMalloc( (char*&)mesh0.vertices, mesh0.vertexCount * sizeof( hiprtFloat3 ) );
			float  s   = 0.15f;
			hiprtFloat3 v[] = {
				{ s * sin( 0.f ), s * cos( 0.f ), 0.0f },
				{ s * sin( M_PI * 2.f / 3.f ), s * cos( M_PI * 2.f / 3.f ), 0.0f },
				{ s * sin( M_PI * 4.f / 3.f ), s * cos( M_PI * 4.f / 3.f ), 0.0f } };
			dCopyHtoD( (hiprtFloat3*)mesh0.vertices, v, mesh0.vertexCount );

			hiprtGeometryBuildInput geomInput;
			geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInput.triangleMesh.primitive = &mesh0;

			size_t			  geomTempSize;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize );
			dMalloc( (u8*&)geomTempTris0, geomTempSize );

			hiprtCreateGeometry( ctxt, &geomInput, &options, &geomTris0 );
			hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTempTris0, 0, geomTris0 );
		}

		hiprtGeometry			   geomTris1;
		hiprtDevicePtr			   geomTempTris1;
		hiprtTriangleMeshPrimitive mesh1;
		{
			mesh1.triangleCount	 = 1;
			mesh1.triangleStride = sizeof( int ) * 3;
			dMalloc( (char*&)mesh1.triangleIndices, 3 * mesh1.triangleCount * sizeof( int ) );
			std::vector<int> idx( 3 * mesh1.triangleCount );
			std::iota( idx.begin(), idx.end(), 0 );
			dCopyHtoD( (int*)mesh1.triangleIndices, idx.data(), 3 * mesh1.triangleCount );

			mesh1.vertexCount  = 3;
			mesh1.vertexStride = sizeof( hiprtFloat3 );
			dMalloc( (char*&)mesh1.vertices, mesh1.vertexCount * sizeof( hiprtFloat3 ) );
			float  s   = 0.15f;
			hiprtFloat3 v[] = {
				{ s * sin( 0.f ), s * cos( 0.f ), 0.0f },
				{ s * sin( M_PI * 2.f / 3.f ), s * cos( M_PI * 2.f / 3.f ), 0.0f },
				{ s * sin( M_PI * 4.f / 3.f ), s * cos( M_PI * 4.f / 3.f ), 0.0f } };
			dCopyHtoD( (hiprtFloat3*)mesh1.vertices, v, mesh1.vertexCount );

			hiprtGeometryBuildInput geomInput;
			geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInput.triangleMesh.primitive = &mesh1;

			size_t			  geomTempSize;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			hiprtGetGeometryBuildTemporaryBufferSize( ctxt, &geomInput, &options, &geomTempSize );
			dMalloc( (u8*&)geomTempTris1, geomTempSize );

			hiprtCreateGeometry( ctxt, &geomInput, &options, &geomTris1 );
			hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTempTris1, 0, geomTris1 );
		}

		hiprtScene			 scene;
		hiprtDevicePtr		 sceneTemp;
		hiprtSceneBuildInput sceneInput;
		{
			sceneInput.instanceCount = 2;
			sceneInput.instanceMasks = nullptr;
			dMalloc( (char*&)sceneInput.instanceGeometries, sceneInput.instanceCount * sizeof( void* ) );
			hiprtDevicePtr geoms[] = { geomTris0, geomTris1 };
			dCopyHtoD( (hiprtDevicePtr*)sceneInput.instanceGeometries, geoms, sceneInput.instanceCount );

			const float offset = 0.3f;
			hiprtFrame	frames[5];
			frames[0].translation = make_hiprtFloat3( -0.25f, -offset, 0.0f );
			frames[0].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[0].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
			frames[0].time		  = 0.0f;
			frames[1].translation = make_hiprtFloat3( 0.0f, -offset, 0.0f );
			frames[1].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[1].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
			frames[1].time		  = 0.35f;
			frames[2].translation = make_hiprtFloat3( 0.25f, -offset, 0.0f );
			frames[2].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[2].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, M_PI * 0.25f );
			frames[2].time		  = 1.0f;
			frames[3].translation = make_hiprtFloat3( 0.0f, offset, 0.0f );
			frames[3].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[3].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.f );
			frames[3].time		  = 0.0f;
			frames[4].translation = make_hiprtFloat3( 0.0f, offset, 0.0f );
			frames[4].scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frames[4].rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, M_PI * 0.5f );
			frames[4].time		  = 1.0f;

			sceneInput.frameCount = 5;
			dMalloc( (hiprtFrame*&)sceneInput.instanceFrames, sceneInput.frameCount );
			dCopyHtoD( (hiprtFrame*&)sceneInput.instanceFrames, frames, sceneInput.frameCount );

			hiprtTransformHeader headers[2];
			headers[0].frameIndex = 0;
			headers[0].frameCount = 3;
			headers[1].frameIndex = 3;
			headers[1].frameCount = 2;
			dMalloc( (hiprtTransformHeader*&)sceneInput.instanceTransformHeaders, sceneInput.instanceCount );
			dCopyHtoD( (hiprtTransformHeader*&)sceneInput.instanceTransformHeaders, headers, sceneInput.instanceCount );

			size_t			  sceneTempSize;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			hiprtGetSceneBuildTemporaryBufferSize( ctxt, &sceneInput, &options, &sceneTempSize );
			dMalloc( (u8*&)sceneTemp, sceneTempSize );

			hiprtCreateScene( ctxt, &sceneInput, &options, &scene );
			hiprtBuildScene( ctxt, hiprtBuildOperationBuild, &sceneInput, &options, sceneTemp, 0, scene );
		}

		oroModule	module;
		oroFunction func;
		std::vector<char> binary;
		buildTraceKernel( ctxt, "../07_motion_blur/TestKernel.h", "MotionBlurKernel", func, &binary);

		u8* dst;
		dMalloc( dst, m_res.x * m_res.y * 4 );
		dMemset( dst, 0, m_res.x * m_res.y * 4 );
		hiprtInt2 res = make_hiprtInt2( m_res.x, m_res.y );

		void* args[] = { &scene, &dst, &res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImageFromDevice( "07_motion_blur.png", m_res.x, m_res.y, dst );

		dFree( sceneInput.instanceGeometries );
		dFree( sceneInput.instanceFrames );
		dFree( sceneInput.instanceTransformHeaders );
		dFree( mesh0.vertices );
		dFree( mesh0.triangleIndices );
		dFree( mesh1.vertices );
		dFree( mesh1.triangleIndices );
		dFree( geomTempTris0 );
		dFree( geomTempTris1 );
		dFree( sceneTemp );
		dFree( dst );
		hiprtDestroyGeometry( ctxt, geomTris0 );
		hiprtDestroyGeometry( ctxt, geomTris1 );
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


