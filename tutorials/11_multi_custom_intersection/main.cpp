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
		constexpr uint32_t SphereTypeIndex = 0;
		constexpr uint32_t CircleTypeIndex = 1;
		constexpr uint32_t GeomTypesCount  = 2;

		hiprtFloat4 sphere = make_hiprtFloat4( -0.3f, 0.0f, 0.0f, 0.2f );
		hiprtFloat4 circle = make_hiprtFloat4( 0.3f, 0.0f, 0.0f, 0.2f );

		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtGeometry		   geomSpheres;
		hiprtAABBListPrimitive listSpheres;
		{
			listSpheres.aabbCount  = 1;
			listSpheres.aabbStride = 2 * sizeof( hiprtFloat4 );
			hiprtFloat4 aabb[]	   = {
				{ sphere.x - sphere.w, sphere.y - sphere.w, sphere.z - sphere.w, 0.0f },
				{ sphere.x + sphere.w, sphere.y + sphere.w, sphere.z + sphere.w, 0.0f } };
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &listSpheres.aabbs ), 2 * sizeof( hiprtFloat4 ) ) );
			CHECK_ORO( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( listSpheres.aabbs ), aabb, 2 * sizeof( hiprtFloat4 ) ) );

			hiprtGeometryBuildInput geomInput;
			geomInput.type				 = hiprtPrimitiveTypeAABBList;
			geomInput.primitive.aabbList = listSpheres;
			geomInput.geomType			 = SphereTypeIndex;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geomSpheres ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geomSpheres ) );
			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );
		}

		hiprtGeometry		   geomCircles;
		hiprtAABBListPrimitive listCircles;
		{
			listCircles.aabbCount  = 1;
			listCircles.aabbStride = 2 * sizeof( hiprtFloat4 );
			hiprtFloat4 aabb[]	   = {
				{ circle.x - circle.w, circle.y - circle.w, circle.z - circle.w, 0.0f },
				{ circle.x + circle.w, circle.y + circle.w, circle.z + circle.w, 0.0f } };
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &listCircles.aabbs ), 2 * sizeof( hiprtFloat4 ) ) );
			CHECK_ORO( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( listCircles.aabbs ), aabb, 2 * sizeof( hiprtFloat4 ) ) );

			hiprtGeometryBuildInput geomInput;
			geomInput.type				 = hiprtPrimitiveTypeAABBList;
			geomInput.primitive.aabbList = listCircles;
			geomInput.geomType			 = CircleTypeIndex;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geomCircles ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geomCircles ) );
			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );
		}

		hiprtScene			 scene;
		hiprtSceneBuildInput sceneInput;
		{
			hiprtInstance instSpheres;
			instSpheres.type	 = hiprtInstanceTypeGeometry;
			instSpheres.geometry = geomSpheres;

			hiprtInstance instCircles;
			instCircles.type	 = hiprtInstanceTypeGeometry;
			instCircles.geometry = geomCircles;

			hiprtInstance instances[] = { instSpheres, instCircles };

			sceneInput.instanceCount			= GeomTypesCount;
			sceneInput.instanceMasks			= nullptr;
			sceneInput.instanceTransformHeaders = nullptr;
			sceneInput.frameCount				= sceneInput.instanceCount;
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInput.instances ),
				sceneInput.instanceCount * sizeof( hiprtInstance ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInput.instances ),
				instances,
				sceneInput.instanceCount * sizeof( hiprtInstance ) ) );

			std::vector<hiprtFrameSRT> frames;
			hiprtFrameSRT			   frame;
			frame.translation = make_hiprtFloat3( 0.0f, 0.0f, 0.0f );
			frame.scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			frame.rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
			for ( int i = 0; i < sceneInput.instanceCount; i++ )
				frames.push_back( frame );

			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceFrames ), frames.size() * sizeof( hiprtFrameSRT ) ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( sceneInput.instanceFrames ),
				frames.data(),
				frames.size() * sizeof( hiprtFrameSRT ) ) );

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

		std::vector<hiprtFuncNameSet> funcNameSets( GeomTypesCount );
		funcNameSets[SphereTypeIndex].intersectFuncName = "intersectSphere";
		funcNameSets[CircleTypeIndex].intersectFuncName = "intersectCircle";

		oroFunction func;
		buildTraceKernelFromBitcode(
			ctxt,
			"../common/TutorialKernels.h",
			"MultiCustomIntersectionKernel",
			func,
			nullptr,
			&funcNameSets,
			GeomTypesCount,
			1 );

		std::vector<hiprtFuncDataSet> funcDataSets( GeomTypesCount );
		CHECK_ORO(
			oroMalloc( const_cast<oroDeviceptr*>( &funcDataSets[SphereTypeIndex].intersectFuncData ), sizeof( hiprtFloat4 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			const_cast<oroDeviceptr>( funcDataSets[SphereTypeIndex].intersectFuncData ), &sphere, sizeof( hiprtFloat4 ) ) );
		CHECK_ORO(
			oroMalloc( const_cast<oroDeviceptr*>( &funcDataSets[CircleTypeIndex].intersectFuncData ), sizeof( hiprtFloat4 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			const_cast<oroDeviceptr>( funcDataSets[CircleTypeIndex].intersectFuncData ), &circle, sizeof( hiprtFloat4 ) ) );

		hiprtFuncTable funcTable;
		CHECK_HIPRT( hiprtCreateFuncTable( ctxt, GeomTypesCount, 1, funcTable ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, SphereTypeIndex, 0, funcDataSets[SphereTypeIndex] ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, CircleTypeIndex, 0, funcDataSets[CircleTypeIndex] ) );

		uint8_t* pixels;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pixels ), m_res.x * m_res.y * 4 ) );

		void* args[] = { &scene, &pixels, &funcTable, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "11_multi_custom_intersection.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( listSpheres.aabbs ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( listCircles.aabbs ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInput.instanceFrames ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneInput.instances ) ) );
		CHECK_ORO( oroFree( const_cast<oroDeviceptr>( funcDataSets[SphereTypeIndex].intersectFuncData ) ) );
		CHECK_ORO( oroFree( const_cast<oroDeviceptr>( funcDataSets[CircleTypeIndex].intersectFuncData ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pixels ) ) );

		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geomSpheres ) );
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
