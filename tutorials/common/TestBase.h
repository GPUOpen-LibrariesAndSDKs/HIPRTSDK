#pragma once
#include <hiprt/hiprt.h>
#include <hiprt/hiprt_vec.h>
#include <Orochi/Orochi.h>
#include <string>
#include <fstream>
#include <vector>

#define ASSERT( cond )  \
	if ( !( cond ) )    \
	{                   \
		__debugbreak(); \
	}

typedef unsigned char u8;
#define M_PI 3.1415f

class TestBase
{
  public:
	void init( int deviceIndex = 0 );

	virtual ~TestBase() {}

	virtual void run() = 0;

	void readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes = 0 );

	void buildTraceKernel(
		hiprtContext	   ctxt,
		const char*		   path,
		const char*		   functionName,
		oroFunction&	   function,
		hiprtArray<char>* binaryOut = 0 );

	void launchKernel( oroFunction func, int nx, int ny, void** args );

	void writeImage( const char* path, int w, int h, u8* data );

	void writeImageFromDevice( const char* path, int w, int h, u8* data )
	{
		u8* tmp = new u8[w * h * 4];
		u8* tmp1 = new u8[w * h * 4];

		dCopyDtoH( tmp, data, w * h * 4 );
		waitForCompletion();
		for(int j=0; j<h; j++)
		for(int i=0; i<w; i++)
		{
			int idx = i+j*w;
			int dIdx = i+(h-1-j)*w;
			for(int k=0; k<4; k++)
				tmp1[dIdx*4+k] = tmp[idx*4+k];
		}
		writeImage( path, w, h, tmp1 );
		delete [] tmp;
		delete [] tmp1;
	};

	template <typename T>
	void dMalloc( T*& ptr, int n )
	{
		oroError e = oroMalloc( (oroDeviceptr*)&ptr, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	void dFree( void* ptr )
	{
		oroError e = oroFree( (oroDeviceptr)ptr );
		ASSERT( e == oroSuccess );
	}

	void dMemset( void* ptr, int val, int n )
	{
		oroError e = oroMemset( (oroDeviceptr)ptr, val, n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void dCopyHtoD( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyHtoD( (oroDeviceptr)dst, src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void dCopyDtoH( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyDtoH( dst, (oroDeviceptr)src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void dCopyDtoD( T* dst, T* src, int n )
	{
		oroError e = oroMemcpyDtoD( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	void waitForCompletion()
	{
		auto e = oroStreamSynchronize( 0 );
		ASSERT( e == oroSuccess );
	}

  protected:
	hiprtContextCreationInput m_ctxtInput;
	oroCtx					  m_oroCtx;
	oroDevice				  m_oroDevice;
	hiprtInt2				  m_res;
};