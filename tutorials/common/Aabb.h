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
#include <hiprt/hiprt_types.h>
#include <tutorials/common/Common.h>

struct Aabb
{
	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb() { reset(); }

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb( const hiprtFloat3& p ) : m_min( p ), m_max( p ) {}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb( const hiprtFloat3& mi, const hiprtFloat3& ma ) : m_min( mi ), m_max( ma ) {}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb( const Aabb& rhs, const Aabb& lhs )
	{
		m_min.x = fminf( lhs.m_min.x, rhs.m_min.x );
		m_min.y = fminf( lhs.m_min.y, rhs.m_min.y );
		m_min.z = fminf( lhs.m_min.z, rhs.m_min.z );
		m_max.x = fmaxf( lhs.m_max.x, rhs.m_max.x );
		m_max.y = fmaxf( lhs.m_max.y, rhs.m_max.y );
		m_max.z = fmaxf( lhs.m_max.z, rhs.m_max.z );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void reset( void )
	{
		m_min = make_hiprtFloat3( hiprt::FltMax, hiprt::FltMax, hiprt::FltMax );
		m_max = make_hiprtFloat3( -hiprt::FltMax, -hiprt::FltMax, -hiprt::FltMax );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb& grow( const hiprtFloat3& p )
	{
		m_min.x = fminf( m_min.x, p.x );
		m_min.y = fminf( m_min.y, p.y );
		m_min.z = fminf( m_min.z, p.z );
		m_max.x = fmaxf( m_max.x, p.x );
		m_max.y = fmaxf( m_max.y, p.y );
		m_max.z = fmaxf( m_max.z, p.z );
		return *this;
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb& grow( const Aabb& rhs )
	{
		m_min.x = fminf( m_min.x, rhs.m_min.x );
		m_min.y = fminf( m_min.y, rhs.m_min.y );
		m_min.z = fminf( m_min.z, rhs.m_min.z );
		m_max.x = fmaxf( m_max.x, rhs.m_max.x );
		m_max.y = fmaxf( m_max.y, rhs.m_max.y );
		m_max.z = fmaxf( m_max.z, rhs.m_max.z );
		return *this;
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE hiprtFloat3 center() const
	{
		hiprtFloat3 c;
		c.x = ( m_max.x + m_min.x ) * 0.5f;
		c.y = ( m_max.y + m_min.y ) * 0.5f;
		c.z = ( m_max.z + m_min.z ) * 0.5f;
		return c;
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE hiprtFloat3 extent() const
	{
		hiprtFloat3 e;
		e.x = m_max.x - m_min.x;
		e.y = m_max.y - m_min.y;
		e.z = m_max.z - m_min.z;
		return e;
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE float area() const
	{
		hiprtFloat3 ext = extent();
		return 2 * ( ext.x * ext.y + ext.x * ext.z + ext.y * ext.z );
	}

	hiprtFloat3 m_min;
	hiprtFloat3 m_max;
};
