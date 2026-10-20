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

#include <tutorials/common/TutorialBase.h>
#include <hiprt/hiprt_libpath.h>

int main( int argc, char** argv )
{
	const int deviceIndex = 0;

	CHECK_ORO( static_cast<oroError>(
		oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0, hiprtGetHipPaths(), hiprtGetHiprtcPaths() ) ) );

	CHECK_ORO( oroInit( 0 ) );

	oroDevice oroDevice;
	CHECK_ORO( oroDeviceGet( &oroDevice, deviceIndex ) );

	oroCtx oroCtx;
	CHECK_ORO( oroCtxCreate( &oroCtx, 0, oroDevice ) );

	oroDeviceProp props;
	CHECK_ORO( oroGetDeviceProperties( &props, oroDevice ) );
	std::cout << "Executing on '" << props.name << "'" << std::endl;

	hiprtContextCreationInput ctxtInput;
	if ( std::string( props.name ).find( "NVIDIA" ) != std::string::npos )
		ctxtInput.deviceType = hiprtDeviceNVIDIA;
	else
		ctxtInput.deviceType = hiprtDeviceAMD;
	ctxtInput.ctxt	 = oroGetRawCtx( oroCtx );
	ctxtInput.device = oroGetRawDevice( oroDevice );

	hiprtContext ctxt;
	CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, ctxtInput, ctxt ) );
	CHECK_HIPRT( hiprtDestroyContext( ctxt ) );

	return 0;
}
