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
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <array>

#pragma once
#include <algorithm>
#include <cfloat>
#include <hiprt/hiprt.h>
#include <queue>

struct Aabb
{
	Aabb() { reset(); }

	Aabb( const hiprtFloat3& p ) : m_min( p ), m_max( p ) {}

	Aabb( const hiprtFloat3& mi, const hiprtFloat3& ma ) : m_min( mi ), m_max( ma ) {}

	Aabb( const Aabb& rhs, const Aabb& lhs )
	{
		m_min.x = fminf( lhs.m_min.x, rhs.m_min.x );
		m_min.y = fminf( lhs.m_min.y, rhs.m_min.y );
		m_min.z = fminf( lhs.m_min.z, rhs.m_min.z );
		m_max.x = fmaxf( lhs.m_max.x, rhs.m_max.x );
		m_max.y = fmaxf( lhs.m_max.y, rhs.m_max.y );
		m_max.z = fmaxf( lhs.m_max.z, rhs.m_max.z );
	}

	void reset( void )
	{
		m_min = make_hiprtFloat3( FLT_MAX, FLT_MAX, FLT_MAX );
		m_max = make_hiprtFloat3( -FLT_MAX, -FLT_MAX, -FLT_MAX );
	}

	Aabb& grow( const hiprtFloat3& p )
	{
		m_min.x = fminf( m_min.x, p.x );
		m_min.y = fminf( m_min.y, p.y );
		m_min.z = fminf( m_min.z, p.z );
		m_max.x = fmaxf( m_max.x, p.x );
		m_max.y = fmaxf( m_max.y, p.y );
		m_max.z = fmaxf( m_max.z, p.z );
		return *this;
	}

	Aabb& grow( const Aabb& rhs )
	{
		m_min.x = fminf( m_min.x, rhs.m_min.x );
		m_min.y = fminf( m_min.y, rhs.m_min.y );
		m_min.z = fminf( m_min.z, rhs.m_min.z );
		m_max.x = fmaxf( m_max.x, rhs.m_max.x );
		m_max.y = fmaxf( m_max.y, rhs.m_max.y );
		m_max.z = fmaxf( m_max.z, rhs.m_max.z );
		return *this;
	}

	hiprtFloat3 center() const
	{
		hiprtFloat3 c;
		c.x = ( m_max.x + m_min.x ) * 0.5f;
		c.y = ( m_max.y + m_min.y ) * 0.5f;
		c.z = ( m_max.z + m_min.z ) * 0.5f;
		return c;
	}

	hiprtFloat3 extent() const
	{
		hiprtFloat3 e;
		e.x = m_max.x - m_min.x;
		e.y = m_max.y - m_min.y;
		e.z = m_max.z - m_min.z;
		return e;
	}

	float area() const
	{
		hiprtFloat3 ext = extent();
		return 2 * ( ext.x * ext.y + ext.x * ext.z + ext.y * ext.z );
	}

	hiprtFloat3 m_min;
	hiprtFloat3 m_max;
};

struct QueueEntry
{
	int	 m_nodeIndex;
	int	 m_begin;
	int	 m_end;
	Aabb m_box;
	QueueEntry( int nodeIndex, int begin, int end, const Aabb& box )
		: m_nodeIndex( nodeIndex ), m_begin( begin ), m_end( end ), m_box( box )
	{
	}
};

class BvhBuilder
{
  public:
	BvhBuilder( void )	= delete;
	BvhBuilder& operator=( const BvhBuilder& ) = delete;

	static void build( uint32_t nPrims, const std::vector<Aabb>& primBoxes, std::vector<hiprtBvhNode>& nodes );
};

void BvhBuilder::build( uint32_t nPrims, const std::vector<Aabb>& primBoxes, std::vector<hiprtBvhNode>& nodes )
{
	ASSERT( nPrims >= 2 );
	std::vector<Aabb> rightBoxes( nPrims );
	std::vector<int>  tmpIndices( nPrims );
	std::vector<int>  leftIndices( nPrims );

	std::vector<int> indices[3];
	for ( int k = 0; k < 3; ++k )
	{
		indices[k].resize( nPrims );
		for ( int i = 0; i < nPrims; ++i )
			indices[k][i] = i;
		std::sort( indices[k].begin(), indices[k].end(), [&]( int a, int b ) {
			hiprtFloat3 ca = primBoxes[a].center();
			hiprtFloat3 cb = primBoxes[b].center();
			return reinterpret_cast<float*>( &ca )[k] > reinterpret_cast<float*>( &cb )[k];
		} );
	}

	Aabb box;
	for ( int i = 0; i < nPrims; ++i )
		box.grow( primBoxes[i] );

	std::queue<QueueEntry> queue;
	queue.push( QueueEntry( 0, 0, nPrims, box ) );
	nodes.push_back( hiprtBvhNode() );
	while ( !queue.empty() )
	{
		int	 nodeIndex = queue.front().m_nodeIndex;
		int	 begin	   = queue.front().m_begin;
		int	 end	   = queue.front().m_end;
		Aabb box	   = queue.front().m_box;
		queue.pop();

		float minCost  = FLT_MAX;
		int	  minAxis  = 0;
		int	  minIndex = 0;
		Aabb  minLeftBox, minRightBox;
		for ( int k = 0; k < 3; ++k )
		{

			rightBoxes[end - 1] = primBoxes[indices[k][end - 1]];
			for ( int i = end - 2; i >= begin; --i )
				rightBoxes[i] = Aabb( primBoxes[indices[k][i]], rightBoxes[i + 1] );

			Aabb leftBox, rightBox;
			for ( int i = begin; i < end - 1; ++i )
			{
				int leftCount  = ( i + 1 ) - begin;
				int rightCount = end - ( i + 1 );
				leftBox.grow( primBoxes[indices[k][i]] );
				rightBox   = rightBoxes[i + 1];
				float cost = leftBox.area() * leftCount + rightBox.area() * rightCount;
				if ( cost < minCost )
				{
					minCost		= cost;
					minIndex	= i + 1;
					minAxis		= k;
					minLeftBox	= leftBox;
					minRightBox = rightBox;
				}
				ASSERT( leftBox.area() <= box.area() );
				ASSERT( rightBox.area() <= box.area() );
			}
		}

		ASSERT( minIndex > begin );
		ASSERT( end > minIndex );

		memset( leftIndices.data(), 0, nPrims * sizeof( int ) );
		for ( int i = begin; i < minIndex; ++i )
		{
			int index		   = indices[minAxis][i];
			leftIndices[index] = 1;
		}

		for ( int j = 0; j < 3; ++j )
		{
			if ( j != minAxis )
			{
				int k = begin;
				int l = minIndex;
				for ( int i = begin; i < end; ++i )
				{
					int index = indices[j][i];
					if ( leftIndices[indices[j][i]] )
						tmpIndices[k++] = index;
					else
						tmpIndices[l++] = index;
				}
				ASSERT( k == minIndex );
				ASSERT( l == end );
				memcpy( &indices[j][begin], &tmpIndices[begin], ( end - begin ) * sizeof( int ) );
			}
		}

		nodes[nodeIndex].boundingBoxMin	 = box.m_min;
		nodes[nodeIndex].boundingBoxMax	 = box.m_max;
		nodes[nodeIndex].childIndices[2] = hiprtInvalidValue;
		nodes[nodeIndex].childIndices[3] = hiprtInvalidValue;

		if ( minIndex - begin == 1 )
		{
			nodes[nodeIndex].childIndices[0]   = indices[minAxis][begin];
			nodes[nodeIndex].childNodeTypes[0] = hiprtBvhNodeTypeLeaf;
		}
		else
		{
			nodes[nodeIndex].childIndices[0]   = static_cast<int>( nodes.size() );
			nodes[nodeIndex].childNodeTypes[0] = hiprtBvhNodeTypeInternal;
			queue.push( QueueEntry( nodes[nodeIndex].childIndices[0], begin, minIndex, minLeftBox ) );
			nodes.push_back( hiprtBvhNode() );
		}

		if ( end - minIndex == 1 )
		{
			nodes[nodeIndex].childIndices[1]   = indices[minAxis][minIndex];
			nodes[nodeIndex].childNodeTypes[1] = hiprtBvhNodeTypeLeaf;
		}
		else
		{
			nodes[nodeIndex].childIndices[1]   = static_cast<int>( nodes.size() );
			nodes[nodeIndex].childNodeTypes[1] = hiprtBvhNodeTypeInternal;
			queue.push( QueueEntry( nodes[nodeIndex].childIndices[1], minIndex, end, minRightBox ) );
			nodes.push_back( hiprtBvhNode() );
		}
	}
}

class Test : public TestBase
{
  public:
	void buildBvh( hiprtGeometryBuildInput& buildInput );
	void run() 
	{
		using namespace std;

		hiprtContext ctxt;
		hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= CORNELL_BOX_TRIANGLE_COUNT;
		mesh.triangleStride = sizeof( int ) * 3;
		dMalloc( (char*&)mesh.triangleIndices, 3 * mesh.triangleCount * sizeof( int ) );
		std::array<int, 3 * CORNELL_BOX_TRIANGLE_COUNT> idx;
		std::iota( idx.begin(), idx.end(), 0 );
		dCopyHtoD( (int*)mesh.triangleIndices, idx.data(), 3 * mesh.triangleCount );

		mesh.vertexCount  = 3 * mesh.triangleCount;
		mesh.vertexStride = sizeof( hiprtFloat3 );
		dMalloc( (char*&)mesh.vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) );
		dCopyHtoD( (hiprtFloat3*)mesh.vertices, (hiprtFloat3*)cornellBoxVertices.data(), mesh.vertexCount );
		waitForCompletion();

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.triangleMesh.primitive = &mesh;

		hiprtBvhNodeList nodes;
		geomInput.nodes = &nodes;
		buildBvh( geomInput );

		hiprtDevicePtr	  geomTemp = nullptr;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitCustomBvhImport;

		hiprtGeometry geom;
		hiprtError e = hiprtCreateGeometry( ctxt, &geomInput, &options, &geom );
		hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, &geomInput, &options, geomTemp, 0, geom );

		oroFunction func;
		buildTraceKernel( ctxt, "../05_custom_bvh_import/TestKernel.h", "CornellBoxKernel", func );

		u8* dst;
		dMalloc( dst, m_res.x * m_res.y * 4 );
		hiprtInt2 res = make_hiprtInt2( m_res.x, m_res.y );

		hiprtFloat3* diffusColors;
		dMalloc( diffusColors, CORNELL_BOX_MAT_COUNT );
		dCopyHtoD( diffusColors, (hiprtFloat3*)cornellBoxDiffuseColors.data(), CORNELL_BOX_MAT_COUNT );

		void* args[] = { &geom, &dst, &res};
		launchKernel( func, m_res.x, m_res.y, args );
		writeImageFromDevice( "05_custom_bvh_import.png", m_res.x, m_res.y, dst );

		dFree( mesh.triangleIndices );
		dFree( mesh.vertices );
		dFree( dst );
		hiprtDestroyGeometry( ctxt, geom );
		hiprtDestroyContext( ctxt );
	}
};

void Test::buildBvh( hiprtGeometryBuildInput& buildInput )
{
	std::vector<hiprtBvhNode> nodes;
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh )
	{
		std::vector<Aabb> primBoxes( buildInput.triangleMesh.primitive->triangleCount );
		std::vector<u8>	  verticesRaw(
			  buildInput.triangleMesh.primitive->vertexCount * buildInput.triangleMesh.primitive->vertexStride );
		std::vector<u8> trianglesRaw(
			buildInput.triangleMesh.primitive->triangleCount * buildInput.triangleMesh.primitive->triangleStride );
		dCopyDtoH(
			verticesRaw.data(),
			(u8*)buildInput.triangleMesh.primitive->vertices,
			buildInput.triangleMesh.primitive->vertexCount * buildInput.triangleMesh.primitive->vertexStride );
		dCopyDtoH(
			trianglesRaw.data(),
			(u8*)buildInput.triangleMesh.primitive->triangleIndices,
			buildInput.triangleMesh.primitive->triangleCount * buildInput.triangleMesh.primitive->triangleStride );
		for ( int i = 0; i < buildInput.triangleMesh.primitive->triangleCount; ++i )
		{
			hiprtInt3 triangle = *(hiprtInt3*)( trianglesRaw.data() + i * buildInput.triangleMesh.primitive->triangleStride );
			hiprtFloat3 v0 = *( (
				  const hiprtFloat3*)( (u8*)verticesRaw.data() + triangle.x * buildInput.triangleMesh.primitive->vertexStride ) );
			hiprtFloat3 v1 = *( (
				  const hiprtFloat3*)( (u8*)verticesRaw.data() + triangle.y * buildInput.triangleMesh.primitive->vertexStride ) );
			hiprtFloat3 v2 =
				*( (const hiprtFloat3*)( (u8*)verticesRaw.data() + triangle.z * buildInput.triangleMesh.primitive->vertexStride ) );
			primBoxes[i].reset();
			primBoxes[i].grow( v0 );
			primBoxes[i].grow( v1 );
			primBoxes[i].grow( v2 );
		}
		BvhBuilder::build( buildInput.triangleMesh.primitive->triangleCount, primBoxes, nodes );
	}
	else if ( buildInput.type == hiprtPrimitiveTypeAABBList )
	{
		std::vector<Aabb> primBoxes( buildInput.aabbList.primitive->aabbCount );
		std::vector<u8>	  primBoxesRaw( buildInput.aabbList.primitive->aabbCount * buildInput.aabbList.primitive->aabbStride );
		dCopyDtoH(
			primBoxesRaw.data(),
			(u8*)buildInput.aabbList.primitive->aabbs,
			buildInput.aabbList.primitive->aabbCount * buildInput.aabbList.primitive->aabbStride );
		for ( int i = 0; i < buildInput.aabbList.primitive->aabbCount; ++i )
		{
			hiprtFloat4* ptr   = (hiprtFloat4*)( primBoxesRaw.data() + i * buildInput.aabbList.primitive->aabbStride );
			primBoxes[i].m_min = *(hiprtFloat3*)( ptr + 0 );
			primBoxes[i].m_max = *(hiprtFloat3*)( ptr + 1 );
		}
		BvhBuilder::build( buildInput.aabbList.primitive->aabbCount, primBoxes, nodes );
	}
	dMalloc( (hiprtBvhNode*&)buildInput.nodes->nodes, nodes.size() );
	dCopyHtoD( (hiprtBvhNode*)buildInput.nodes->nodes, nodes.data(), nodes.size() );
	buildInput.nodes->nodeCount = nodes.size();
}

int main( int argc, char** argv )
{
	Test test;
	test.init( 0 );
	test.run();

	return 0;
}


