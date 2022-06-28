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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>
#include <stdio.h>
#include <string.h>

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

int main( int argc, char** argv )
{
	const int deviceIndex = 0;

	hiprtContextCreationInput m_ctxtInput;
	oroCtx					  m_oroCtx;
	oroDevice				  m_oroDevice;
	{
		oroInitialize( ( oroApi )( ORO_API_HIP | ORO_API_CUDA ), 0 );

		oroError e = oroInit( 0 );
		ASSERT( e == oroSuccess );
		e = oroDeviceGet( &m_oroDevice, deviceIndex );
		ASSERT( e == oroSuccess );
		e = oroCtxCreate( &m_oroCtx, 0, m_oroDevice );
		ASSERT( e == oroSuccess );

		oroDeviceProp props;
		e = oroGetDeviceProperties( &props, m_oroDevice );
		ASSERT( e == oroSuccess );
		printf( "executing on %s\n", props.name );

		if ( strstr( props.name, "Radeon" ) != 0 )
			m_ctxtInput.deviceType = hiprtDeviceAMD;
		else
			m_ctxtInput.deviceType = hiprtDeviceNVIDIA;
		m_ctxtInput.ctxt   = oroGetRawCtx( m_oroCtx );
		m_ctxtInput.device = oroGetRawDevice( m_oroDevice );
	}

	hiprtContext ctxt;
	hiprtError e = hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );
	ASSERT( e == hiprtSuccess );

	e = hiprtDestroyContext( ctxt );
	ASSERT( e == hiprtSuccess );

	return 0;
}


