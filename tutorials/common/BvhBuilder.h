#pragma once
#include <hiprt/hiprt_types.h>
#include <cassert>
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
};
