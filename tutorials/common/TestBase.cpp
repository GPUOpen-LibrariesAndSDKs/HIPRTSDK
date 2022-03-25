#include <tutorials/common/TestBase.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <contrib/stbi/stbi_image_write.h>

inline constexpr hiprtError operator|=( hiprtError x, hiprtError y )
{
	return static_cast<hiprtError>( static_cast<int>( x ) | static_cast<int>( y ) );
}

void TestBase::init( int deviceIndex )
{
	m_res = make_hiprtInt2( 512, 512 );

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

void TestBase::readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes )
{
	std::fstream f( path );
	if ( f.is_open() )
	{
		size_t sizeFile;
		f.seekg( 0, std::fstream::end );
		size_t size = sizeFile = (size_t)f.tellg();
		f.seekg( 0, std::fstream::beg );
		if ( includes )
		{
			sourceCode.clear();
			std::string line;
			char		buf[512];
			while ( std::getline( f, line ) )
			{
				if ( strstr( line.c_str(), "#include" ) != 0 )
				{
					const char* a = strstr( line.c_str(), "<" );
					const char* b = strstr( line.c_str(), ">" );
					int			n = b - a - 1;
					memcpy( buf, a + 1, n );
					buf[n] = '\0';
					includes->push_back( buf );
					sourceCode += line + '\n';
				}
				else
				{
					sourceCode += line + '\n';
				}
			}
		}
		else
		{
			sourceCode.resize( size, ' ' );
			f.read( &sourceCode[0], size );
		}
		f.close();
	}
}

hiprtError TestBase::buildTraceProgram( hiprtContext ctxt, const char* path, const char* functionName, orortcProgram& progOut )
{
	std::vector<std::string> includeNamesData;
	std::string				 sourceCode;
	readSourceCode( path, sourceCode, &includeNamesData );

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	for ( int i = 0; i < includeNamesData.size(); i++ )
	{
		readSourceCode( std::string( "../" ) + includeNamesData[i], headersData[i] );
		includeNames.push_back( includeNamesData[i].c_str() );
		headers.push_back( headersData[i].c_str() );
	}
	std::vector<const char*> opts;
	return hiprtBuildTraceProgram(
		ctxt,
		functionName,
		sourceCode.c_str(),
		path,
		headers.size(),
		headers.data(),
		includeNames.data(),
		opts.data(),
		opts.size(),
		&progOut );
}

hiprtError TestBase::buildTraceGetBinary( orortcProgram& prog, size_t& size, char* binary )
{
	return hiprtBuildTraceGetBinary( &prog, &size, binary );
}

hiprtError TestBase::buildTraceKernel(
	hiprtContext ctxt, const char* path, const char* functionName, oroFunction& function, hiprtArray<char>* binaryOut )
{
	oroModule	module;
	oroFunction func;
	hiprtError	error = hiprtSuccess;

	orortcProgram prog;
	size_t		  binarySize = 0;
	error |= buildTraceProgram( ctxt, path, functionName, prog );
	error |= buildTraceGetBinary( prog, binarySize, nullptr );

	std::vector<char> binary( binarySize );
	error |= buildTraceGetBinary( prog, binarySize, binary.data() );

	if ( binaryOut )
	{
		binaryOut->setSize( binary.size() );
		memcpy( (void*)binaryOut->getPtr(), binary.data(), sizeof( char ) * binary.size() );
	}

	const char* loweredName;
	orortcGetLoweredName( prog, functionName, &loweredName );
	oroModuleLoadData( &module, binary.data() );
	oroModuleGetFunction( &function, module, loweredName );
	orortcDestroyProgram( &prog );

	return error;
}

void TestBase::launchKernel( oroFunction func, int nx, int ny, void** args )
{
	hiprtInt3 tpb = { 16, 16, 1 };
	hiprtInt3 nb;
	nb.x	   = ( nx + tpb.x - 1 ) / tpb.x;
	nb.y	   = ( ny + tpb.y - 1 ) / tpb.y;
	oroError e = oroModuleLaunchKernel( func, nb.x, nb.y, 1, tpb.x, tpb.y, 1, 0, 0, args, 0 );
	ASSERT( e == oroSuccess );
}

void TestBase::writeImage( const char* path, int w, int h, u8* data ) { stbi_write_png( path, w, h, 4, data, w * 4 ); }


