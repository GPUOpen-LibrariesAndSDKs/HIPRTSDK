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
#include <numeric>
#include <array>

#pragma once
#include <algorithm>
#include <cfloat>
#include <hiprt/hiprt.h>
#include <queue>

#ifndef ASSERT
#if defined( _MSC_VER )
#define ASSERT( cond )  \
	if ( !( cond ) )    \
	{                   \
		__debugbreak(); \
	}
#elif defined( __GNUC__ )
#include <signal.h>
#define ASSERT( cond )    \
	if ( !( cond ) )      \
	{                     \
		raise( SIGTRAP ); \
	}
#else
#define ASSERT( cond )
#endif
#endif

class Test : public TestBase
{
  public:

	inline float dot3F4( const hiprtFloat4& a, const hiprtFloat4& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

	inline hiprtFloat3 cross3( const hiprtFloat3& aa, const hiprtFloat3& bb )
	{
		return make_hiprtFloat3( aa.y * bb.z - aa.z * bb.y, aa.z * bb.x - aa.x * bb.z, aa.x * bb.y - aa.y * bb.x );
	}

	inline hiprtFloat4 normalize3( const hiprtFloat4& a )
	{
		float length  = sqrtf( dot3F4( a, a ) );
		length		  = ( length == 0.0f ) ? 1.0f : length;
		const float f = ( 1.0f / length );
		return make_hiprtFloat4( a.x * f, a.y * f, a.z * f, a.w * f );
	}

	inline Quaternion qtSet( const hiprtFloat4& axis, float angle )
	{
		hiprtFloat4 nAxis = normalize3( axis );

		Quaternion q;
		q.x = nAxis.x * sin( angle / 2 );
		q.y = nAxis.y * sin( angle / 2 );
		q.z = nAxis.z * sin( angle / 2 );
		q.w = cos( angle / 2 );
		return q;
	}

	void run() 
	{
		Camera camera;
		camera.m_translation = make_hiprtFloat4( 0.0, 2.5, 5.8, 0.0f );
		camera.m_quat		 = qtSet( make_hiprtFloat4( 0.0f, 0.0f, 0.0f, 0.0f ), 1.0f );
		camera.m_fov		 = 45.0f * M_PI / 180.f;
		camera.m_near		 = 0.0f;
		camera.m_far		 = 100000.0f;
		float aoRadius		 = 1.4f;

		setUp( camera, "../common/meshes/cornellbox/cornellBox.obj", "../common/meshes/cornellbox/" );
		render( "06_obj_AO.png", "../06_obj_AO/TestKernel.h", "AORayKernel", m_res.x, m_res.y, aoRadius);
	}
};

int main( int argc, char** argv )
{
	Test test;
	test.init( 0 );
	test.run();

	return 0;
}


