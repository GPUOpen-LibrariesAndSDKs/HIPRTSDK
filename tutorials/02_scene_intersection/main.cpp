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

		float		s	= 0.5f;
		hiprtFloat3 v[] = {
			{ 0.5f + s * sin( 0.f ), s * cos( 0.f ), 0.0f },
			{ 0.5f + s * sin( M_PI * 2.f / 3.f ), s * cos( M_PI * 2.f / 3.f ), 0.0f },
			{ 0.5f + s * sin( M_PI * 4.f / 3.f ), s * cos( M_PI * 4.f / 3.f ), 0.0f },
			{ -0.5f + s * sin( 0.f ), s * cos( 0.f ), 0.0f },
			{ -0.5f + s * sin( M_PI * 2.f / 3.f ), s * cos( M_PI * 2.f / 3.f ), 0.0f },
			{ -0.5f + s * sin( M_PI * 4.f / 3.f ), s * cos( M_PI * 4.f / 3.f ), 0.0f } };
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
		frame.scale			  = make_hiprtFloat3( 0.5f, 0.5f, 0.5f );
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


