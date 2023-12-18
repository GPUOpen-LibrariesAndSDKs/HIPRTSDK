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
#include <tutorials/common/BvhBuilder.h>
#include <tutorials/common/CornellBox.h>
#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	void buildBvh( hiprtGeometryBuildInput& buildInput );
	void run()
	{
		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= CornellBoxTriangleCount;
		mesh.triangleStride = sizeof( hiprtInt3 );
		std::array<uint32_t, 3 * CornellBoxTriangleCount> triangleIndices;
		std::iota( triangleIndices.begin(), triangleIndices.end(), 0 );
		CHECK_ORO(
			oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh.triangleIndices ), mesh.triangleCount * sizeof( hiprtInt3 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			reinterpret_cast<oroDeviceptr>( mesh.triangleIndices ),
			triangleIndices.data(),
			mesh.triangleCount * sizeof( hiprtInt3 ) ) );

		mesh.vertexCount  = 3 * mesh.triangleCount;
		mesh.vertexStride = sizeof( hiprtFloat3 );
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh.vertices ), mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			reinterpret_cast<oroDeviceptr>( mesh.vertices ),
			const_cast<hiprtFloat3*>( cornellBoxVertices.data() ),
			mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh;
		buildBvh( geomInput );

		hiprtDevicePtr	  geomTemp = nullptr;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitCustomBvhImport;

		hiprtGeometry geom;
		CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
		CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

		oroFunction func;
		buildTraceKernelFromBitcode( ctxt, "../common/TutorialKernels.h", "CustomBvhImportKernel", func );

		uint8_t* pixels;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pixels ), m_res.x * m_res.y * 4 ) );

		uint32_t* matIndices;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &matIndices ), mesh.triangleCount * sizeof( uint32_t ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			reinterpret_cast<oroDeviceptr>( matIndices ),
			cornellBoxMatIndices.data(),
			mesh.triangleCount * sizeof( uint32_t ) ) );

		hiprtFloat3* diffusColors;
		CHECK_ORO(
			oroMalloc( reinterpret_cast<oroDeviceptr*>( &diffusColors ), CornellBoxMaterialCount * sizeof( hiprtFloat3 ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			reinterpret_cast<oroDeviceptr>( diffusColors ),
			const_cast<hiprtFloat3*>( cornellBoxDiffuseColors.data() ),
			CornellBoxMaterialCount * sizeof( hiprtFloat3 ) ) );

		void* args[] = { &geom, &pixels, &m_res, &matIndices, &diffusColors };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "07_custom_bvh_import.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( matIndices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( diffusColors ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh.triangleIndices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( mesh.vertices ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pixels ) ) );

		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom ) );
		CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
	}
};

void Tutorial::buildBvh( hiprtGeometryBuildInput& buildInput )
{
	std::vector<hiprtBvhNode> nodes;
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh )
	{
		std::vector<Aabb>	 primBoxes( buildInput.primitive.triangleMesh.triangleCount );
		std::vector<uint8_t> verticesRaw(
			buildInput.primitive.triangleMesh.vertexCount * buildInput.primitive.triangleMesh.vertexStride );
		std::vector<uint8_t> trianglesRaw(
			buildInput.primitive.triangleMesh.triangleCount * buildInput.primitive.triangleMesh.triangleStride );
		CHECK_ORO( oroMemcpyDtoH(
			verticesRaw.data(),
			reinterpret_cast<oroDeviceptr>( buildInput.primitive.triangleMesh.vertices ),
			buildInput.primitive.triangleMesh.vertexCount * buildInput.primitive.triangleMesh.vertexStride ) );
		CHECK_ORO( oroMemcpyDtoH(
			trianglesRaw.data(),
			reinterpret_cast<oroDeviceptr>( buildInput.primitive.triangleMesh.triangleIndices ),
			buildInput.primitive.triangleMesh.triangleCount * buildInput.primitive.triangleMesh.triangleStride ) );
		for ( uint32_t i = 0; i < buildInput.primitive.triangleMesh.triangleCount; ++i )
		{
			hiprtInt3 triangle =
				*reinterpret_cast<hiprtInt3*>( trianglesRaw.data() + i * buildInput.primitive.triangleMesh.triangleStride );
			hiprtFloat3 v0 = *reinterpret_cast<const hiprtFloat3*>(
				verticesRaw.data() + triangle.x * buildInput.primitive.triangleMesh.vertexStride );
			hiprtFloat3 v1 = *reinterpret_cast<const hiprtFloat3*>(
				verticesRaw.data() + triangle.y * buildInput.primitive.triangleMesh.vertexStride );
			hiprtFloat3 v2 = *reinterpret_cast<const hiprtFloat3*>(
				verticesRaw.data() + triangle.z * buildInput.primitive.triangleMesh.vertexStride );
			primBoxes[i].reset();
			primBoxes[i].grow( v0 );
			primBoxes[i].grow( v1 );
			primBoxes[i].grow( v2 );
		}
		BvhBuilder::build( buildInput.primitive.triangleMesh.triangleCount, primBoxes, nodes );
	}
	else if ( buildInput.type == hiprtPrimitiveTypeAABBList )
	{
		std::vector<Aabb>	 primBoxes( buildInput.primitive.aabbList.aabbCount );
		std::vector<uint8_t> primBoxesRaw( buildInput.primitive.aabbList.aabbCount * buildInput.primitive.aabbList.aabbStride );
		CHECK_ORO( oroMemcpyDtoH(
			primBoxesRaw.data(),
			reinterpret_cast<oroDeviceptr>( buildInput.primitive.aabbList.aabbs ),
			buildInput.primitive.aabbList.aabbCount * buildInput.primitive.aabbList.aabbStride ) );
		for ( uint32_t i = 0; i < buildInput.primitive.aabbList.aabbCount; ++i )
		{
			hiprtFloat4* ptr =
				reinterpret_cast<hiprtFloat4*>( primBoxesRaw.data() + i * buildInput.primitive.aabbList.aabbStride );
			primBoxes[i].m_min = make_hiprtFloat3( ptr[0] );
			primBoxes[i].m_max = make_hiprtFloat3( ptr[1] );
		}
		BvhBuilder::build( buildInput.primitive.aabbList.aabbCount, primBoxes, nodes );
	}
	CHECK_ORO(
		oroMalloc( reinterpret_cast<oroDeviceptr*>( &buildInput.nodeList.nodes ), nodes.size() * sizeof( hiprtBvhNode ) ) );
	CHECK_ORO( oroMemcpyHtoD(
		reinterpret_cast<oroDeviceptr>( buildInput.nodeList.nodes ), nodes.data(), nodes.size() * sizeof( hiprtBvhNode ) ) );
	buildInput.nodeList.nodeCount = static_cast<uint32_t>( nodes.size() );
}

int main( int argc, char** argv )
{
	Tutorial tutorial;
	tutorial.init( 0 );
	tutorial.run();

	return 0;
}
