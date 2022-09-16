

#include <Orochi/Orochi.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

inline
oroApi getApiType( int argc, char** argv )
{
	oroApi api = ORO_API_HIP;
	if( argc >= 2 )
	{
		if( strcmp( argv[1], "hip" ) == 0 )
			api = ORO_API_HIP;
		if( strcmp( argv[1], "cuda" ) == 0 )
			api = ORO_API_CUDA;
	}
	return api;
}

inline void checkError( oroError e )
{
	const char* pStr;
	oroGetErrorString( e, &pStr );
	if( e != oroSuccess ) 
		printf( "ERROR==================\n%s\n", pStr);
}

#define ERROR_CHECK( e ) checkError( e )

int main(int argc, char** argv )
{
	oroApi api = getApiType( argc, argv );

	int a = oroInitialize( api, 0 );
	if( a != 0 )
	{
		printf("initialization failed\n");
		return 0;
	}
	printf( ">> executing on %s\n", ( api == ORO_API_HIP )? "hip":"cuda" );

	printf(">> testing initialization\n");
	oroError e;
	e = oroInit( 0 );
	oroDevice device;
	e = oroDeviceGet( &device, 0 );
	oroCtx ctx;
	e = oroCtxCreate( &ctx, 0, device );

	printf(">> testing device props\n");
	{
		oroDeviceProp props;
		oroGetDeviceProperties( &props, device );
		printf("executing on %s (%s)\n", props.name, props.gcnArchName );
	}
	printf(">> testing kernel execution\n");
	{
		oroFunction function;
		{
			const char* code = "extern \"C\" __global__ "
							   "void testKernel()"
							   "{ int a = threadIdx.x; }";
			const char* funcName = "testKernel";
			orortcProgram prog;
			orortcResult e;
			e = orortcCreateProgram( &prog, code, funcName, 0, 0, 0 );
			std::vector<const char*> opts; 
			opts.push_back( "-I ../" );

			e = orortcCompileProgram( prog, opts.size(), opts.data() );
			if( e != ORORTC_SUCCESS )
			{
				size_t logSize;
				orortcGetProgramLogSize(prog, &logSize);
				if (logSize) 
				{
					std::string log(logSize, '\0');
					orortcGetProgramLog(prog, &log[0]);
					std::cout << log << '\n';
				};
			}
			size_t codeSize;
			e = orortcGetCodeSize(prog, &codeSize);

			std::vector<char> codec(codeSize);
			e = orortcGetCode(prog, codec.data());
			e = orortcDestroyProgram(&prog);
			oroModule module;
			oroError ee = oroModuleLoadData(&module, codec.data());
			ee = oroModuleGetFunction(&function, module, funcName);		
		}

		oroStream stream;
		oroStreamCreate( &stream );
		
		oroEvent start, stop;
		oroEventCreateWithFlags( &start, 0 );
		oroEventCreateWithFlags( &stop, 0 );
		oroEventRecord( start, stream );

		void** args = {};
		oroError e = oroModuleLaunchKernel( function, 1,1,1, 32,1,1, 0, stream, args, 0 );

		oroEventRecord( stop, stream );

		oroDeviceSynchronize();

		oroStreamDestroy( stream );

		float milliseconds = 0.0f;
		oroEventElapsedTime( &milliseconds, start, stop );
		printf( ">> kernel - %.5f ms\n", milliseconds );
		oroEventDestroy( start );
		oroEventDestroy( stop );
	}
	printf(">> done\n");
	return 0;
}