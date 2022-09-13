#pragma once
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>
#include <hiprt/hiprt_vec.h>

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


class HipTestBase
{
  public:
	void init(int deviceIdx = 0)
	{
		hipError_t e = hipInit( 0 );
		ASSERT( e == hipSuccess );

		e = hipGetDevice( &m_hipDevice );
		ASSERT( e == hipSuccess );

		e = hipCtxCreate( &m_hipCtx, 0, m_hipDevice );
		ASSERT( e == hipSuccess );

		hipDeviceProp_t props;
		hipGetDeviceProperties( &props, m_hipDevice );
		ASSERT( e == hipSuccess );
		printf( "executing on %s\n", props.name );

		if ( strstr( props.name, "Radeon" ) != 0 )
			m_ctxtInput.deviceType = hiprtDeviceAMD;
		else
			m_ctxtInput.deviceType = hiprtDeviceNVIDIA;

		m_ctxtInput.ctxt = m_hipCtx;
		m_ctxtInput.device = m_hipDevice;
	}

	void waitForCompletion()
	{
		auto e = hipStreamSynchronize( 0 );
		ASSERT( e == hipSuccess );
	}

	hipCtx_t	m_hipCtx;
	hipDevice_t m_hipDevice;
	hiprtContextCreationInput m_ctxtInput;

	virtual ~HipTestBase() { hipCtxDestroy( m_hipCtx ); }

	virtual void run() = 0;
};
