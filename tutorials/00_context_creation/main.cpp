#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>
#include <stdio.h>
#include <string.h>

#define ASSERT( cond )  \
	if ( !( cond ) )    \
	{                   \
		__debugbreak(); \
	}

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


