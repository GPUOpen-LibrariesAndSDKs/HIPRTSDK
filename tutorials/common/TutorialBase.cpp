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

#include <tutorials/common/TutorialBase.h>

#include <hiprt/hiprt_libpath.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <contrib/stbi/stbi_image_write.h>

void checkOro( oroError res, const char* file, uint32_t line )
{
	if ( res != oroSuccess )
	{
		const char* msg;
		oroGetErrorString( res, &msg );
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkOrortc( orortcResult res, const char* file, uint32_t line )
{
	if ( res != ORORTC_SUCCESS )
	{
		std::cerr << "ORORTC error: '" << orortcGetErrorString( res ) << "' [ " << res << " ] on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkHiprt( hiprtError res, const char* file, uint32_t line )
{
	if ( res != hiprtSuccess )
	{
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void TutorialBase::init( uint32_t deviceIndex )
{
	m_res = make_hiprtInt2( 512, 512 );

	CHECK_ORO(
		static_cast<oroError>( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0, g_hip_paths, g_hiprtc_paths ) ) );

	CHECK_ORO( oroInit( 0 ) );
	CHECK_ORO( oroDeviceGet( &m_oroDevice, deviceIndex ) );
	CHECK_ORO( oroCtxCreate( &m_oroCtx, 0, m_oroDevice ) );

	oroDeviceProp props;
	CHECK_ORO( oroGetDeviceProperties( &props, m_oroDevice ) );

	std::cout << "hiprt ver." << HIPRT_VERSION_STR << std::endl;
	std::cout << "Executing on '" << props.name << "'" << std::endl;
	if ( std::string( props.name ).find( "NVIDIA" ) != std::string::npos )
		m_ctxtInput.deviceType = hiprtDeviceNVIDIA;
	else
		m_ctxtInput.deviceType = hiprtDeviceAMD;

	m_ctxtInput.ctxt   = oroGetRawCtx( m_oroCtx );
	m_ctxtInput.device = oroGetRawDevice( m_oroDevice );
	hiprtSetLogLevel( hiprtLogLevelError );
}

bool TutorialBase::readSourceCode(
	const std::filesystem::path& path, std::string& sourceCode, std::optional<std::vector<std::filesystem::path>> includes )
{
	std::fstream f( path );
	if ( f.is_open() )
	{
		size_t sizeFile;
		f.seekg( 0, std::fstream::end );
		size_t size = sizeFile = static_cast<size_t>( f.tellg() );
		f.seekg( 0, std::fstream::beg );
		if ( includes )
		{
			sourceCode.clear();
			std::string line;
			while ( std::getline( f, line ) )
			{
				if ( line.find( "#include" ) != std::string::npos )
				{
					size_t		pa	= line.find( "<" );
					size_t		pb	= line.find( ">" );
					std::string buf = line.substr( pa + 1, pb - pa - 1 );
					includes.value().push_back( buf );
					sourceCode += line + '\n';
				}
				sourceCode += line + '\n';
			}
		}
		else
		{
			sourceCode.resize( size, ' ' );
			f.read( &sourceCode[0], size );
		}
		f.close();
	}
	else
		return false;
	return true;
}

void TutorialBase::buildTraceKernelFromBitcode(
	hiprtContext				   ctxt,
	const char*					   path,
	const char*					   functionName,
	oroFunction&				   functionOut,
	std::vector<const char*>*	   opts,
	std::vector<hiprtFuncNameSet>* funcNameSets,
	uint32_t					   numGeomTypes,
	uint32_t					   numRayTypes )
{
	std::vector<const char*>		   options;
	std::vector<std::filesystem::path> includeNamesData;
	std::string						   sourceCode;

	if ( !readSourceCode( path, sourceCode, includeNamesData ) )
	{
		std::cerr << "Unable to find file '" << path << "'" << std::endl;
		;
		exit( EXIT_FAILURE );
	}

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	for ( int i = 0; i < includeNamesData.size(); i++ )
	{
		if ( !readSourceCode( std::string( "../../" ) / includeNamesData[i], headersData[i] ) )
		{
			if ( !readSourceCode( std::string( "../" ) / includeNamesData[i], headersData[i] ) )
			{
				std::cerr << "Failed to find header file '" << includeNamesData[i] << "' in path ../ or ../../!" << std::endl;
				exit( EXIT_FAILURE );
			}
		}
		includeNames.push_back( includeNamesData[i].string().c_str() );
		headers.push_back( headersData[i].c_str() );
	}

	if ( opts )
	{
		for ( const auto o : *opts )
			options.push_back( o );
	}

	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
	if ( isAmd )
	{
		options.push_back( "-fgpu-rdc" );
		options.push_back( "-Xclang" );
		options.push_back( "-disable-llvm-passes" );
		options.push_back( "-Xclang" );
		options.push_back( "-mno-constructor-aliases" );
	}
	else
	{
		options.push_back( "--device-c" );
		options.push_back( "-arch=compute_60" );
	}
	options.push_back( "-std=c++17" );
	options.push_back( "-I../" );
	options.push_back( "-I../../" );

	orortcProgram prog;
	CHECK_ORORTC( orortcCreateProgram(
		&prog, sourceCode.data(), path, static_cast<int>( headers.size() ), headers.data(), includeNames.data() ) );
	CHECK_ORORTC( orortcAddNameExpression( prog, functionName ) );

	orortcResult e = orortcCompileProgram( prog, static_cast<int>( options.size() ), options.data() );
	if ( e != ORORTC_SUCCESS )
	{
		size_t logSize;
		CHECK_ORORTC( orortcGetProgramLogSize( prog, &logSize ) );

		if ( logSize )
		{
			std::string log( logSize, '\0' );
			orortcGetProgramLog( prog, &log[0] );
			std::cerr << log << std::endl;
		}
		exit( EXIT_FAILURE );
	}

	std::string bitCodeBinary;
	size_t		size = 0;
	if ( isAmd )
		CHECK_ORORTC( orortcGetBitcodeSize( prog, &size ) );
	else
		CHECK_ORORTC( orortcGetCodeSize( prog, &size ) );
	assert( size != 0 );

	bitCodeBinary.resize( size );
	if ( isAmd )
		CHECK_ORORTC( orortcGetBitcode( prog, (char*)bitCodeBinary.data() ) );
	else
		CHECK_ORORTC( orortcGetCode( prog, (char*)bitCodeBinary.data() ) );

	hiprtApiFunction function;
	CHECK_HIPRT( hiprtBuildTraceKernelsFromBitcode(
		ctxt,
		1,
		&functionName,
		path,
		bitCodeBinary.data(),
		size,
		numGeomTypes,
		numRayTypes,
		funcNameSets != nullptr ? funcNameSets->data() : nullptr,
		&function,
		false ) );

	functionOut = *reinterpret_cast<oroFunction*>( &function );
}

void TutorialBase::launchKernel( oroFunction func, uint32_t nx, uint32_t ny, void** args )
{
	launchKernel( func, nx, ny, 8, 8, args );
}

void TutorialBase::launchKernel( oroFunction func, uint32_t nx, uint32_t ny, uint32_t bx, uint32_t by, void** args )
{
	hiprtInt3 nb;
	nb.x = ( nx + bx - 1 ) / bx;
	nb.y = ( ny + by - 1 ) / by;
	CHECK_ORO( oroModuleLaunchKernel( func, nb.x, nb.y, 1, bx, by, 1, 0, 0, args, 0 ) );
}

void TutorialBase::writeImage( const std::string& path, uint32_t width, uint32_t height, uint8_t* pixels )
{
	std::vector<uint8_t> image( width * height * 4 );
	CHECK_ORO( oroMemcpyDtoH( image.data(), reinterpret_cast<oroDeviceptr>( pixels ), width * height * 4 ) );
	stbi_write_png( path.c_str(), width, height, 4, image.data(), width * 4 );
	std::cout << "image written at " << path.c_str() << std::endl;
}
