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

#pragma once
#include <cassert>
#include <cstdint>
#include <queue>
#include <tutorials/common/Aabb.h>

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
	BvhBuilder( void )						   = delete;
	BvhBuilder& operator=( const BvhBuilder& ) = delete;

	static void build( uint32_t nPrims, const std::vector<Aabb>& primBoxes, std::vector<hiprtBvhNode>& nodes )
	{
		assert( nPrims >= 2 );
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
					assert( leftBox.area() <= box.area() );
					assert( rightBox.area() <= box.area() );
				}
			}

			assert( minIndex > begin );
			assert( end > minIndex );

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
					assert( k == minIndex );
					assert( l == end );
					memcpy( &indices[j][begin], &tmpIndices[begin], ( end - begin ) * sizeof( int ) );
				}
			}

			nodes[nodeIndex].childAabbsMin[0] = minLeftBox.m_min;
			nodes[nodeIndex].childAabbsMax[0] = minLeftBox.m_max;
			nodes[nodeIndex].childAabbsMin[1] = minRightBox.m_min;
			nodes[nodeIndex].childAabbsMax[1] = minRightBox.m_max;
			nodes[nodeIndex].childIndices[2]  = hiprtInvalidValue;
			nodes[nodeIndex].childIndices[3]  = hiprtInvalidValue;

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
};
