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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <contrib/stbi/stbi_image_write.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "../common/tiny_obj_loader.h"
#include <iostream>
#include <map>


namespace std
{
inline bool operator<( const tinyobj::index_t& a, const tinyobj::index_t& b )
{
	if ( a.vertex_index < b.vertex_index ) return true;
	if ( a.vertex_index > b.vertex_index ) return false;

	if ( a.normal_index < b.normal_index ) return true;
	if ( a.normal_index > b.normal_index ) return false;

	if ( a.texcoord_index < b.texcoord_index ) return true;
	if ( a.texcoord_index > b.texcoord_index ) return false;

	return false;
}
} // namespace std

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

void TestBase::setUp(
	Camera&			   c,
	const char*		   filepath,
	const char*		   dirpath,
	bool			   enableRayMask,
	hiprtFrame*		   frame,
	hiprtBuildFlagBits bvhBuildFlag,
	bool			   time )
{
	hiprtError e = createScene( m_scene, filepath, dirpath, enableRayMask, frame, bvhBuildFlag, time );
	assert( e == hiprtSuccess );

	dMalloc( m_camera, 1 );
	dCopyHtoD( m_camera, &c, 1 );
	m_scene.m_garbageCollector.push_back( (void*)m_camera );
}

hiprtError TestBase::createScene(
	SceneData&		   scene,
	std::string		   fileName,
	std::string		   mtlBaseDir,
	bool			   enableRayMask,
	hiprtFrame*		   frame,
	hiprtBuildFlagBits bvhBuildFlag,
	bool			   time )
{
	hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &scene.m_ctx );

	tinyobj::attrib_t				 attrib;
	std::vector<tinyobj::shape_t>	 shapes;
	std::vector<tinyobj::material_t> materials;
	std::string						 err;
	std::string						 warning;

	bool ret = tinyobj::LoadObj( &attrib, &shapes, &materials, &warning, &err, fileName.c_str(), mtlBaseDir.c_str() );

	if ( !warning.empty() )
	{
		std::cout << "OBJ Loader WARN : " << warning << '\n';
	}

	if ( !err.empty() )
	{
		std::cout << "OBJ Loader ERROR : " << err << '\n';
	}

	if ( !ret )
	{
		std::cout << "Failed to load obj file\n";
		std::exit( EXIT_FAILURE );
	}

	std::vector<hiprtDevicePtr> geomtries;
	std::vector<Material_t>		shapeMaterials; // materials for all instances
	std::vector<Light_t>		lights;
	std::vector<int>			materialIndices; // material ids for all instances
	std::vector<unsigned int>	instanceMask;
	std::vector<hiprtFloat3>			allVertices;
	std::vector<hiprtFloat3>	allNormals;
	std::vector<uint32_t>		allIndices;
	int							numOfLights = 0;
	hiprtError					error		= hiprtSuccess;

	// Prefix sum to calculate the offsets in to global vert,index and material buffer
	int						 vertexPrefixSum = 0;
	int						 normalPrefixSum = 0;
	int						 indexPrefixSum	 = 0;
	int						 matIdxPrefixSum = 0;
	std::vector<int>		 indicesOffsets;
	std::vector<int>		 verticesOffsets;
	std::vector<int>		 normalsOffsets;
	std::vector<int>		 matIdxOffset;

	indicesOffsets.resize( shapes.size() );
	verticesOffsets.resize( shapes.size() );
	normalsOffsets.resize( shapes.size() );
	matIdxOffset.resize( shapes.size() );

	auto convert = []( const tinyobj::real_t c[3] ) -> hiprtFloat4 { return hiprtFloat4{ c[0], c[1], c[2], 0.0f }; };
	auto isLight = []( const tinyobj::real_t c[3] ) { return ( c[0] + c[1] + c[2] ) != 0.0f; };

	for ( const auto& mat : materials )
	{
		Material_t m;
		m.m_diffuse	 = convert( mat.diffuse );
		m.m_emission = convert( mat.emission );
		m.m_params.x = isLight( mat.emission );
		shapeMaterials.push_back( m );
	}

	size_t maxTriSize	  = 0;
	size_t maxVertSize	  = 0;
	size_t maxTempbufSize = 0;
	// allocate temp buffers
	for ( size_t i = 0; i < shapes.size(); ++i )
	{
		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= shapes[i].mesh.num_face_vertices.size();
		mesh.triangleStride = sizeof( uint32_t ) * 3;
		mesh.vertexCount	= shapes[i].mesh.indices.size();
		mesh.vertexStride	= sizeof( hiprtFloat3 );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.triangleMesh.primitive = &mesh;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = bvhBuildFlag;
		hiprtGetGeometryBuildTemporaryBufferSize( scene.m_ctx, &geomInput, &options, &geomTempSize );

		if ( maxTempbufSize < geomTempSize ) maxTempbufSize = geomTempSize;
		if ( maxTriSize < mesh.triangleCount ) maxTriSize = mesh.triangleCount;
		if ( maxVertSize < mesh.vertexCount ) maxVertSize = mesh.vertexCount;
	}

	hiprtDevicePtr tempGeomBuffer;
	hiprtDevicePtr tempVertBuffer;
	hiprtDevicePtr tempTriBuffer;

	dMalloc( (char*&)tempGeomBuffer, maxTempbufSize );
	dMalloc( (char*&)tempTriBuffer, 3 * maxTriSize * sizeof( uint32_t ) );
	dMalloc( (char*&)tempVertBuffer, maxVertSize * sizeof( hiprtFloat3 ) );

	for ( size_t i = 0; i < shapes.size(); ++i )
	{
		std::vector<hiprtFloat3>		vertices;
		std::vector<hiprtFloat3>		normals;
		std::vector<uint32_t>			indices;
		hiprtFloat3*					v = (hiprtFloat3*)attrib.vertices.data();
		std::map<tinyobj::index_t, int> knownIndex;

		for ( size_t face = 0; face < shapes[i].mesh.num_face_vertices.size(); face++ )
		{
			tinyobj::index_t idx0 = shapes[i].mesh.indices[3 * face + 0];
			tinyobj::index_t idx1 = shapes[i].mesh.indices[3 * face + 1];
			tinyobj::index_t idx2 = shapes[i].mesh.indices[3 * face + 2];

			if ( knownIndex.find( idx0 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx0] );
			}
			else
			{
				knownIndex[idx0] = vertices.size();
				indices.push_back( knownIndex[idx0] );
				vertices.push_back( v[idx0.vertex_index] );
				normals.push_back( v[idx0.normal_index] );
			}

			if ( knownIndex.find( idx1 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx1] );
			}
			else
			{
				knownIndex[idx1] = vertices.size();
				indices.push_back( knownIndex[idx1] );
				vertices.push_back( v[idx1.vertex_index] );
				normals.push_back( v[idx1.normal_index] );
			}

			if ( knownIndex.find( idx2 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx2] );
			}
			else
			{
				knownIndex[idx2] = vertices.size();
				indices.push_back( knownIndex[idx2] );
				vertices.push_back( v[idx2.vertex_index] );
				normals.push_back( v[idx2.normal_index] );
			}

			if ( shapeMaterials[shapes[i].mesh.material_ids[face]].m_params.x )
			{
				Light_t l;
				l.m_le = make_hiprtFloat4(
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.x + 40.f,
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.y + 40.f,
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.z + 40.f,
					0.0f );

				size_t idx = indices.size() - 1;
				l.m_lv0	   = vertices[indices[idx - 2]];
				l.m_lv1	   = vertices[indices[idx - 1]];
				l.m_lv2	   = vertices[indices[idx - 0]];

				lights.push_back( l );
				numOfLights++;
			}

			materialIndices.push_back( shapes[i].mesh.material_ids[face] );
		}

		verticesOffsets[i] = vertexPrefixSum;
		vertexPrefixSum += vertices.size();
		indicesOffsets[i] = indexPrefixSum;
		indexPrefixSum += indices.size();
		matIdxOffset[i] = matIdxPrefixSum;
		matIdxPrefixSum += shapes[i].mesh.material_ids.size();
		normalsOffsets[i] = normalPrefixSum;
		normalPrefixSum += normals.size();

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	 = shapes[i].mesh.num_face_vertices.size();
		mesh.triangleStride	 = sizeof( uint32_t ) * 3;
		mesh.triangleIndices = tempTriBuffer;
		dCopyHtoD( (uint32_t*)mesh.triangleIndices, indices.data(), 3 * mesh.triangleCount );

		mesh.vertexCount  = vertices.size();
		mesh.vertexStride = sizeof( hiprtFloat3 );
		mesh.vertices	  = tempVertBuffer;
		dCopyHtoD( (hiprtFloat3*)mesh.vertices, (hiprtFloat3*)vertices.data(), vertices.size() );
		waitForCompletion();

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.triangleMesh.primitive = &mesh;

		hiprtBuildOptions options;
		options.buildFlags = bvhBuildFlag;

		hiprtGeometry geom;
		error |= hiprtCreateGeometry( scene.m_ctx, &geomInput, &options, &geom );
		error |= hiprtBuildGeometry( scene.m_ctx, hiprtBuildOperationBuild, &geomInput, &options, tempGeomBuffer, 0, geom );
		geomtries.push_back( geom );

		unsigned int mask = ~0u;
		if ( enableRayMask && ( i % 2 == 0 ) ) mask = 0u;

		instanceMask.push_back( mask );

		allNormals.insert( allNormals.end(), normals.begin(), normals.end() );
		allIndices.insert( allIndices.end(), indices.begin(), indices.end() );
	}

	// copy vertex offset
	dMalloc( scene.m_vertexOffsets, verticesOffsets.size() );
	dCopyHtoD( scene.m_vertexOffsets, verticesOffsets.data(), verticesOffsets.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_vertexOffsets );

	// copy normals
	dMalloc( scene.m_normals, allNormals.size() );
	dCopyHtoD( scene.m_normals, allNormals.data(), allNormals.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_normals );

	// copy normal offsets
	dMalloc( scene.m_normalOffsets, normalsOffsets.size() );
	dCopyHtoD( scene.m_normalOffsets, normalsOffsets.data(), normalsOffsets.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_normalOffsets );

	// copy indices
	dMalloc( scene.m_indices, allIndices.size() );
	dCopyHtoD( scene.m_indices, allIndices.data(), allIndices.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_indices );

	// copy index offsets
	dMalloc( scene.m_indexOffsets, indicesOffsets.size() );
	dCopyHtoD( scene.m_indexOffsets, indicesOffsets.data(), indicesOffsets.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_indexOffsets );

	// copy material indices
	dMalloc( scene.m_bufMaterialIndices, materialIndices.size() );
	dCopyHtoD( scene.m_bufMaterialIndices, materialIndices.data(), materialIndices.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_bufMaterialIndices );

	// copy material offset
	dMalloc( scene.m_bufMatIdsPerInstance, matIdxOffset.size() );
	dCopyHtoD( scene.m_bufMatIdsPerInstance, matIdxOffset.data(), matIdxOffset.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_bufMatIdsPerInstance );

	// copy materials
	dMalloc( scene.m_bufMaterials, shapeMaterials.size() );
	dCopyHtoD( scene.m_bufMaterials, (Material_t*)shapeMaterials.data(), shapeMaterials.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_bufMaterials );

	// copy light
	dMalloc( scene.m_lights, lights.size() );
	dCopyHtoD( scene.m_lights, (Light_t*)lights.data(), lights.size() );
	scene.m_garbageCollector.push_back( (void*)scene.m_lights );

	// copy materials
	dMalloc( scene.m_numOfLights, 1 );
	dCopyHtoD( scene.m_numOfLights, &numOfLights, 1 );
	scene.m_garbageCollector.push_back( (void*)scene.m_numOfLights );

	waitForCompletion();

	// prepare scene
	hiprtScene			 sceneLocal;
	hiprtDevicePtr		 sceneTemp;
	hiprtSceneBuildInput sceneInput;
	{
		sceneInput.instanceCount = shapes.size();
		dMalloc( (char*&)sceneInput.instanceMasks, sceneInput.instanceCount * sizeof( unsigned int ) );
		dCopyHtoD( (unsigned int*)sceneInput.instanceMasks, instanceMask.data(), sceneInput.instanceCount );
		scene.m_garbageCollector.push_back( (void*)sceneInput.instanceMasks );

		dMalloc( (char*&)sceneInput.instanceGeometries, sceneInput.instanceCount * sizeof( void* ) );
		dCopyHtoD( (hiprtDevicePtr*)sceneInput.instanceGeometries, geomtries.data(), sceneInput.instanceCount );
		scene.m_garbageCollector.push_back( (void*)sceneInput.instanceGeometries );

		std::vector<hiprtFrame> frames;
		hiprtFrame				transform;
		if ( frame == nullptr )
		{
			transform.translation = make_hiprtFloat3( 0.0f, 0.0f, 0.0f );
			transform.scale		  = make_hiprtFloat3( 1.0f, 1.0f, 1.0f );
			transform.rotation	  = make_hiprtFloat4( 0.0f, 0.0f, 1.0f, 0.0f );
		}

		sceneInput.frameCount				= sceneInput.instanceCount;
		sceneInput.instanceTransformHeaders = nullptr;

		for ( int i = 0; i < sceneInput.instanceCount; i++ )
		{
			frames.push_back( ( frame == nullptr ) ? transform : *frame );
		}

		dMalloc( (hiprtFrame*&)sceneInput.instanceFrames, frames.size() );
		dCopyHtoD( (hiprtFrame*&)sceneInput.instanceFrames, frames.data(), frames.size() );
		scene.m_garbageCollector.push_back( (void*)sceneInput.instanceFrames );

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = bvhBuildFlag;
		error |= hiprtGetSceneBuildTemporaryBufferSize( scene.m_ctx, &sceneInput, &options, &sceneTempSize );
		dMalloc( (u8*&)sceneTemp, sceneTempSize );
		scene.m_garbageCollector.push_back( (void*)sceneTemp );

		error |= hiprtCreateScene( scene.m_ctx, &sceneInput, &options, &sceneLocal );
		error |= hiprtBuildScene( scene.m_ctx, hiprtBuildOperationBuild, &sceneInput, &options, sceneTemp, 0, sceneLocal );
		scene.m_scene = sceneLocal;
	}

	dFree( tempGeomBuffer );
	dFree( tempVertBuffer );
	dFree( tempTriBuffer );
	return error;
}

void TestBase::render(
	const char* imgPath, const char* kernelPath, const char* funcName, int width, int height, float aoRadius )
{
	u8* dst;
	dMalloc( dst, width * height * 4 );
	m_scene.m_garbageCollector.push_back( (void*)dst );

	int			stackSize		   = 64;
	const int	sharedStackSize	   = 24;
	const int	blockWidth		   = 8;
	const int	blockHeight		   = 8;
	const int	blockSize		   = blockWidth * blockHeight;
	std::string blockSizeDef	   = "-D BLOCK_SIZE=" + std::to_string( blockSize );
	std::string sharedStackSizeDef = "-D SHARED_STACK_SIZE=" + std::to_string( sharedStackSize );

	std::vector<const char*> opts;
	opts.push_back( blockSizeDef.c_str() );
	opts.push_back( sharedStackSizeDef.c_str() );

	int* stackBuffer;
	dMalloc( stackBuffer, width * height * stackSize );
	m_scene.m_garbageCollector.push_back( (void*)stackBuffer );

	oroFunction func;
	buildTraceKernel( m_scene.m_ctx, kernelPath, funcName, func, nullptr, &opts );

	hiprtInt2  res	 = make_hiprtInt2( width, height );
	void* args[] = {
		&m_scene.m_scene,
		&dst,
		&res,
		&stackBuffer,
		&stackSize,
		&m_camera,
		&aoRadius };
	
	launchKernel( func, width, height, args, blockWidth, blockHeight);
	waitForCompletion();

	writeImageFromDevice( imgPath, width, height, dst );
}

bool TestBase::readSourceCode( const std::string& path, std::string& sourceCode, std::vector<std::string>* includes )
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

					int n = b - a - 1;
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
	else
		return false;
	return true;
}

hiprtError TestBase::buildTraceProgram( hiprtContext ctxt, const char* path, const char* functionName, orortcProgram& progOut, std::vector<const char*>* opts)
{
	std::vector<std::string> includeNamesData;
	std::string				 sourceCode;
	readSourceCode( path, sourceCode, &includeNamesData );

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	for ( int i = 0; i < includeNamesData.size(); i++ )
	{
		readSourceCode( std::string( "../../" ) + includeNamesData[i], headersData[i] );

		includeNames.push_back( includeNamesData[i].c_str() );
		headers.push_back( headersData[i].c_str() );
	}
	
	return hiprtBuildTraceProgram(
		ctxt,
		functionName,
		sourceCode.c_str(),
		path,
		headers.size(),
		headers.data(),
		includeNames.data(),
		opts != nullptr ? opts->data() : nullptr,
		opts != nullptr ? opts->size() : 0,
		&progOut );
}

hiprtError TestBase::buildTraceGetBinary( orortcProgram& prog, size_t& size, char* binary )
{
	return hiprtBuildTraceGetBinary( &prog, &size, binary );
}

hiprtError TestBase::buildTraceKernel(
	hiprtContext			  ctxt,
	const char*				  path,
	const char*				  functionName,
	oroFunction&			  function,
	std::vector<char>*		  binaryOut,
	std::vector<const char*>* opts )
{
	oroModule	module;
	oroFunction func;
	hiprtError	error = hiprtSuccess;

	orortcProgram prog;
	size_t		  binarySize = 0;
	error |= buildTraceProgram( ctxt, path, functionName, prog, opts);

	error |= buildTraceGetBinary( prog, binarySize, nullptr );

	std::vector<char> binary( binarySize );
	error |= buildTraceGetBinary( prog, binarySize, binary.data() );

	if ( binaryOut )
	{
		binaryOut->resize( binary.size() );
		memcpy( (void*)binaryOut->data(), binary.data(), sizeof( char ) * binary.size() );
	}

	const char* loweredName;
	orortcGetLoweredName( prog, functionName, &loweredName );
	oroModuleLoadData( &module, binary.data() );
	oroModuleGetFunction( &function, module, loweredName );
	orortcDestroyProgram( &prog );

	return error;
}

void TestBase::launchKernel( oroFunction func, int nx, int ny, void** args, size_t threadPerBlockX, size_t threadPerBlockY, size_t threadPerBlockZ)
{
	hiprtInt3 nb;
	nb.x	   = ( nx + threadPerBlockX - 1 ) / threadPerBlockX;
	nb.y	   = ( ny + threadPerBlockY - 1 ) / threadPerBlockY;
	oroError e = oroModuleLaunchKernel( func, nb.x, nb.y, 1, threadPerBlockX, threadPerBlockY, threadPerBlockZ, 0, 0, args, 0 );
	ASSERT( e == oroSuccess );
}

void TestBase::writeImage( const char* path, int w, int h, u8* data ) { stbi_write_png( path, w, h, 4, data, w * 4 ); }


