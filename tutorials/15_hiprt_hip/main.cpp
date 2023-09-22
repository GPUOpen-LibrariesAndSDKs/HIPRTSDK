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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#if defined( __USE_HIP__ )
#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>
#include <hiprt/hiprt_vec.h>

#define CHECK_HIP( error ) ( checkHip( error, __FILE__, __LINE__ ) )
void checkHip( hipError_t res, const char* file, int line )
{
	if ( res != hipSuccess )
	{
		std::cerr << "HIP error: '" << hipGetErrorString( res ) << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

#define CHECK_HIPRT( error ) ( checkHiprt( error, __FILE__, __LINE__ ) )
void checkHiprt( hiprtError res, const char* file, int line )
{
	if ( res != hiprtSuccess )
	{
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}
#endif

int main( int argc, char** argv )
{
#if defined( __USE_HIP__ )
	CHECK_HIP( hipInit( 0 ) );

	hipDevice_t hipDevice;
	CHECK_HIP( hipGetDevice( &hipDevice ) );

	hipCtx_t hipCtx;
	CHECK_HIP( hipCtxCreate( &hipCtx, 0, hipDevice ) );

	hipDeviceProp_t props;
	CHECK_HIP( hipGetDeviceProperties( &props, hipDevice ) );
	std::cout << "Executing on '" << props.name << "'" << std::endl;

	hiprtContextCreationInput ctxtInput;
	if ( std::string( props.name ).find( "NVIDIA" ) != std::string::npos )
		ctxtInput.deviceType = hiprtDeviceNVIDIA;
	else
		ctxtInput.deviceType = hiprtDeviceAMD;
	ctxtInput.ctxt	 = hipCtx;
	ctxtInput.device = hipDevice;

	hiprtContext ctxt;
	CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, ctxtInput, ctxt ) );
	CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
#endif
	return 0;
}
