//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "SceneDemo.h"

#include <thread>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "BvhBuilder.h"

#include "hiprt/hiprt_common.h"

#include "Orochi/OrochiUtils.h"

void SceneDemo::setupScene(
	Camera&						 camera,
	const std::string&			 filePath,
	const std::string&			 dirPath,
	bool						 enableRayMask,
	std::optional<hiprtFrameSRT> frame,
	hiprtBuildFlags				 bvhBuildFlag )
{
	m_camera = camera;
	createScene( m_scene, filePath, dirPath, enableRayMask, frame, bvhBuildFlag );
}

void SceneDemo::createScene(
	SceneData&					 scene,
	const std::string&			 filename,
	const std::string&			 mtlBaseDir,
	bool						 enableRayMask,
	std::optional<hiprtFrameSRT> frame,
	hiprtBuildFlags				 bvhBuildFlag )
{
	hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, scene.m_ctx );

	tinyobj::attrib_t				 attrib;
	std::vector<tinyobj::shape_t>	 shapes;
	std::vector<tinyobj::material_t> materials;
	std::string						 err;
	std::string						 warning;

	bool ret = tinyobj::LoadObj( &attrib, &shapes, &materials, &warning, &err, filename.c_str(), mtlBaseDir.c_str() );

	if ( !warning.empty() )
	{
		std::cerr << "OBJ Loader WARN : " << warning << std::endl;
	}

	if ( !err.empty() )
	{
		std::cerr << "OBJ Loader ERROR : " << err << std::endl;
		std::exit( EXIT_FAILURE );
	}

	if ( !ret )
	{
		std::cerr << "Failed to load obj file" << std::endl;
		std::exit( EXIT_FAILURE );
	}

	if ( shapes.empty() )
	{
		std::cerr << "No shapes in obj file (run 'git lfs fetch' and 'git lfs pull' in 'test/common/meshes/lfs')" << std::endl;
		std::exit( EXIT_FAILURE );
	}

	std::vector<Material> shapeMaterials; // materials for all instances
	std::vector<Light>	  lights;
	std::vector<uint32_t> materialIndices; // material ids for all instances
	std::vector<uint32_t> instanceMask;
	std::vector<float3>	  allVertices;
	std::vector<float3>	  allNormals;
	std::vector<uint32_t> allIndices;
	std::vector<Aabb>	  geomBoxes;

	uint32_t numOfLights = 0;

	// Prefix sum to calculate the offsets in to global vert,index and material buffer
	uint32_t				 vertexPrefixSum = 0u;
	uint32_t				 normalPrefixSum = 0u;
	uint32_t				 indexPrefixSum	 = 0u;
	uint32_t				 matIdxPrefixSum = 0u;
	std::vector<uint32_t>	 indicesOffsets;
	std::vector<uint32_t>	 verticesOffsets;
	std::vector<uint32_t>	 normalsOffsets;
	std::vector<uint32_t>	 matIdxOffset;
	std::chrono::nanoseconds bvhBuildTime{};

	indicesOffsets.resize( shapes.size() );
	verticesOffsets.resize( shapes.size() );
	normalsOffsets.resize( shapes.size() );
	matIdxOffset.resize( shapes.size() );

	auto convert = []( const tinyobj::real_t c[3] ) -> float3 { return float3{ c[0], c[1], c[2] }; };

	for ( const auto& mat : materials )
	{
		Material m;
		m.m_diffuse	 = convert( mat.diffuse );
		m.m_emission = convert( mat.emission );
		shapeMaterials.push_back( m );
	}

#ifdef ENABLE_EMBREE
	RTCDevice embreeDevice;
	if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport )
	{
		embreeDevice = rtcNewDevice( "" );
		rtcSetDeviceErrorFunction(
			embreeDevice,
			[]( void* userPtr, enum RTCError code, const char* str ) { std::cerr << str << std::endl; },
			nullptr );
	}
#endif

	auto compare = []( const tinyobj::index_t& a, const tinyobj::index_t& b ) {
		if ( a.vertex_index < b.vertex_index ) return true;
		if ( a.vertex_index > b.vertex_index ) return false;

		if ( a.normal_index < b.normal_index ) return true;
		if ( a.normal_index > b.normal_index ) return false;

		if ( a.texcoord_index < b.texcoord_index ) return true;
		if ( a.texcoord_index > b.texcoord_index ) return false;

		return false;
	};

	for ( size_t i = 0; i < shapes.size(); ++i )
	{
		std::vector<float3>										  vertices;
		std::vector<float3>										  normals;
		std::vector<uint32_t>									  indices;
		const int64_t											  vfloat3Count = attrib.vertices.size() / 3;
		const int64_t											  nfloat3Count = attrib.normals.size() / 3;
		const float3*											  v = reinterpret_cast<float3*>( attrib.vertices.data() );
		const float3*											  n = reinterpret_cast<float3*>( attrib.normals.data() );
		std::map<tinyobj::index_t, uint32_t, decltype( compare )> knownIndex( compare );
		Aabb													  geomBox;

		for ( size_t face = 0; face < shapes[i].mesh.num_face_vertices.size(); face++ )
		{
			tinyobj::index_t idx0 = shapes[i].mesh.indices[3 * face + 0];
			tinyobj::index_t idx1 = shapes[i].mesh.indices[3 * face + 1];
			tinyobj::index_t idx2 = shapes[i].mesh.indices[3 * face + 2];

#ifdef _DEBUG
			// just a sanity check of the OBJ parsing.
			if ( idx0.vertex_index >= vfloat3Count || idx0.normal_index >= nfloat3Count || idx1.vertex_index >= vfloat3Count ||
				 idx1.normal_index >= nfloat3Count || idx2.vertex_index >= vfloat3Count || idx2.normal_index >= nfloat3Count )
			{
				assert( false );
			}
#endif

			if ( knownIndex.find( idx0 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx0] );
			}
			else
			{
				knownIndex[idx0] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx0] );
				vertices.push_back( v[idx0.vertex_index] );
				normals.push_back( n[idx0.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( knownIndex.find( idx1 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx1] );
			}
			else
			{
				knownIndex[idx1] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx1] );
				vertices.push_back( v[idx1.vertex_index] );
				normals.push_back( n[idx1.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( knownIndex.find( idx2 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx2] );
			}
			else
			{
				knownIndex[idx2] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx2] );
				vertices.push_back( v[idx2.vertex_index] );
				normals.push_back( n[idx2.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( !shapeMaterials.empty() && shapeMaterials[shapes[i].mesh.material_ids[face]].light() )
			{
				Light l;
				l.m_le = make_float3(
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.x + 40.f,
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.y + 40.f,
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.z + 40.f );

				size_t idx = indices.size() - 1;
				l.m_lv0	   = vertices[indices[idx - 2]];
				l.m_lv1	   = vertices[indices[idx - 1]];
				l.m_lv2	   = vertices[indices[idx - 0]];

				lights.push_back( l );
				numOfLights++;
			}

			materialIndices.push_back(
				shapes[i].mesh.material_ids[face] >= 0 ? shapes[i].mesh.material_ids[face] : hiprtInvalidValue );
		}

		verticesOffsets[i] = vertexPrefixSum;
		vertexPrefixSum += static_cast<uint32_t>( vertices.size() );
		indicesOffsets[i] = indexPrefixSum;
		indexPrefixSum += static_cast<uint32_t>( indices.size() );
		matIdxOffset[i] = matIdxPrefixSum;
		matIdxPrefixSum += static_cast<uint32_t>( shapes[i].mesh.material_ids.size() );
		normalsOffsets[i] = normalPrefixSum;
		normalPrefixSum += static_cast<uint32_t>( normals.size() );

		uint32_t mask = ~0u;
		if ( enableRayMask && ( i % 2 == 0 ) ) mask = 0u;

		instanceMask.push_back( mask );
		geomBoxes.push_back( geomBox );

		allVertices.insert( allVertices.end(), vertices.begin(), vertices.end() );
		allNormals.insert( allNormals.end(), normals.begin(), normals.end() );
		allIndices.insert( allIndices.end(), indices.begin(), indices.end() );
	}

	uint32_t threadCount = std::min( std::thread::hardware_concurrency(), 16u );

	if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport ) threadCount = 1;

	std::vector<std::thread>			  threads( threadCount );
	std::vector<std::chrono::nanoseconds> bvhBuildTimes( threadCount );
	std::vector<oroStream>				  streams( threadCount );
	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		CHECK_ORO( oroStreamCreate( &streams[threadIndex] ) );
	}

	oroCtx ctx;
	CHECK_ORO( oroCtxGetCurrent( &ctx ) );

	m_scene.m_geometries.resize( shapes.size() );
	m_scene.m_instances.resize( shapes.size() );
	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		threads[threadIndex] = std::thread(
			[&]( uint32_t threadIndex ) {
				CHECK_ORO( oroCtxSetCurrent( ctx ) );

				std::vector<hiprtGeometry*>			 geomAddrs;
				std::vector<hiprtGeometryBuildInput> geomInputs;
				for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
				{
					hiprtTriangleMeshPrimitive mesh;

					uint32_t* indices	= &allIndices[indicesOffsets[i]];
					mesh.triangleCount	= static_cast<uint32_t>( shapes[i].mesh.num_face_vertices.size() );
					mesh.triangleStride = sizeof( uint32_t ) * 3;
					OrochiUtils::malloc(
						reinterpret_cast<uint8_t*&>( mesh.triangleIndices ), 3 * mesh.triangleCount * sizeof( uint32_t ) );
					OrochiUtils::copyHtoDAsync(
						reinterpret_cast<uint32_t*>( mesh.triangleIndices ),
						indices,
						3 * mesh.triangleCount,
						streams[threadIndex] );

					float3* vertices  = &allVertices[verticesOffsets[i]];
					mesh.vertexCount  = ( i + 1 == shapes.size() ) ? vertexPrefixSum - verticesOffsets[i]
																   : verticesOffsets[i + 1] - verticesOffsets[i];
					mesh.vertexStride = sizeof( float3 );
					OrochiUtils::malloc( reinterpret_cast<uint8_t*&>( mesh.vertices ), mesh.vertexCount * sizeof( float3 ) );
					OrochiUtils::copyHtoDAsync(
						reinterpret_cast<float3*>( mesh.vertices ), vertices, mesh.vertexCount, streams[threadIndex] );

					hiprtGeometryBuildInput geomInput;
					geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
					geomInput.primitive.triangleMesh = mesh;
					geomInput.geomType				 = 0;

#ifdef ENABLE_EMBREE
					if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport )
						buildEmbreeGeometryBvh( embreeDevice, vertices, indices, geomInput );
#endif

					geomInputs.push_back( geomInput );
					geomAddrs.push_back( &m_scene.m_geometries[i] );
				}

				if ( !geomInputs.empty() )
				{
					hiprtBuildOptions options;
					options.buildFlags = bvhBuildFlag;

					size_t geomTempSize;
					CHECK_HIPRT( hiprtGetGeometriesBuildTemporaryBufferSize(
						scene.m_ctx, static_cast<uint32_t>( geomInputs.size() ), geomInputs.data(), options, geomTempSize ) );

					hiprtDevicePtr tempGeomBuffer = nullptr;
					if ( geomTempSize > 0 ) OrochiUtils::malloc( reinterpret_cast<uint8_t*&>( tempGeomBuffer ), geomTempSize );

					CHECK_HIPRT( hiprtCreateGeometries(
						scene.m_ctx,
						static_cast<uint32_t>( geomInputs.size() ),
						geomInputs.data(),
						options,
						geomAddrs.data() ) );

					std::vector<hiprtGeometry> geoms;
					for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
						geoms.push_back( m_scene.m_geometries[i] );

					std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
					CHECK_HIPRT( hiprtBuildGeometries(
						scene.m_ctx,
						hiprtBuildOperationBuild,
						static_cast<uint32_t>( geomInputs.size() ),
						geomInputs.data(),
						options,
						tempGeomBuffer,
						streams[threadIndex],
						geoms.data() ) );
					std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
					bvhBuildTimes[threadIndex] += end - begin;

					size_t j = 0;
					for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
					{
						m_scene.m_geometries[i]			= geoms[j++];
						m_scene.m_instances[i].type		= hiprtInstanceTypeGeometry;
						m_scene.m_instances[i].geometry = m_scene.m_geometries[i];
					}

					for ( auto& geomInput : geomInputs )
					{
						OrochiUtils::free( geomInput.primitive.triangleMesh.triangleIndices );
						OrochiUtils::free( geomInput.primitive.triangleMesh.vertices );
						if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport )
						{
							OrochiUtils::free( geomInput.nodeList.nodes );
							OrochiUtils::free( geomInput.primitive.triangleMesh.trianglePairIndices );
						}
					}

					if ( geomTempSize > 0 ) OrochiUtils::free( tempGeomBuffer );

					OrochiUtils::waitForCompletion( streams[threadIndex] );
				}
			},
			threadIndex );
	}

	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		threads[threadIndex].join();
		CHECK_ORO( oroStreamDestroy( streams[threadIndex] ) );
		bvhBuildTime = std::max( bvhBuildTime, bvhBuildTimes[threadIndex] );
	}

	// copy vertex offset
	OrochiUtils::malloc( scene.m_vertexOffsets, verticesOffsets.size() );
	OrochiUtils::copyHtoD( scene.m_vertexOffsets, verticesOffsets.data(), verticesOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_vertexOffsets );

	// copy normals
	OrochiUtils::malloc( scene.m_normals, allNormals.size() );
	OrochiUtils::copyHtoD( scene.m_normals, allNormals.data(), allNormals.size() );
	scene.m_garbageCollector.push_back( scene.m_normals );

	// copy normal offsets
	OrochiUtils::malloc( scene.m_normalOffsets, normalsOffsets.size() );
	OrochiUtils::copyHtoD( scene.m_normalOffsets, normalsOffsets.data(), normalsOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_normalOffsets );

	// copy indices
	OrochiUtils::malloc( scene.m_indices, allIndices.size() );
	OrochiUtils::copyHtoD( scene.m_indices, allIndices.data(), allIndices.size() );
	scene.m_garbageCollector.push_back( scene.m_indices );

	// copy index offsets
	OrochiUtils::malloc( scene.m_indexOffsets, indicesOffsets.size() );
	OrochiUtils::copyHtoD( scene.m_indexOffsets, indicesOffsets.data(), indicesOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_indexOffsets );

	// copy material indices
	OrochiUtils::malloc( scene.m_bufMaterialIndices, materialIndices.size() );
	OrochiUtils::copyHtoD( scene.m_bufMaterialIndices, materialIndices.data(), materialIndices.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMaterialIndices );

	// copy material offset
	OrochiUtils::malloc( scene.m_bufMatIdsPerInstance, matIdxOffset.size() );
	OrochiUtils::copyHtoD( scene.m_bufMatIdsPerInstance, matIdxOffset.data(), matIdxOffset.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMatIdsPerInstance );

	// copy materials
	if ( shapeMaterials.empty() )
	{ // default material to prevent crash
		Material mat;
		mat.m_diffuse  = make_float3( 1.0f );
		mat.m_emission = make_float3( 0.0f );
		shapeMaterials.push_back( mat );
	}
	OrochiUtils::malloc( scene.m_bufMaterials, shapeMaterials.size() );
	OrochiUtils::copyHtoD( scene.m_bufMaterials, shapeMaterials.data(), shapeMaterials.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMaterials );

	// copy light
	if ( !lights.empty() )
	{
		OrochiUtils::malloc( scene.m_lights, lights.size() );
		OrochiUtils::copyHtoD( scene.m_lights, lights.data(), lights.size() );
		scene.m_garbageCollector.push_back( scene.m_lights );
	}

	// copy light num
	OrochiUtils::malloc( scene.m_numOfLights, 1 );
	OrochiUtils::copyHtoD( scene.m_numOfLights, &numOfLights, 1 );
	scene.m_garbageCollector.push_back( scene.m_numOfLights );

	// prepare scene
	hiprtScene			 sceneLocal;
	hiprtDevicePtr		 sceneTemp = nullptr;
	hiprtSceneBuildInput sceneInput;
	{
		sceneInput.instanceCount = static_cast<uint32_t>( shapes.size() );
		OrochiUtils::malloc( reinterpret_cast<uint32_t*&>( sceneInput.instanceMasks ), sceneInput.instanceCount );
		OrochiUtils::copyHtoD(
			reinterpret_cast<uint32_t*>( sceneInput.instanceMasks ), instanceMask.data(), sceneInput.instanceCount );
		scene.m_garbageCollector.push_back( sceneInput.instanceMasks );

		OrochiUtils::malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
		OrochiUtils::copyHtoD(
			reinterpret_cast<hiprtInstance*>( sceneInput.instances ), m_scene.m_instances.data(), sceneInput.instanceCount );
		scene.m_garbageCollector.push_back( sceneInput.instances );

		std::vector<hiprtFrameSRT> frames;
		hiprtFrameSRT			   transform;
		if ( !frame )
		{
			transform.translation = make_float3( 0.0f, 0.0f, 0.0f );
			transform.scale		  = make_float3( 1.0f, 1.0f, 1.0f );
			transform.rotation	  = make_float4( 0.0f, 0.0f, 1.0f, 0.0f );
		}

		sceneInput.frameCount				= sceneInput.instanceCount;
		sceneInput.instanceTransformHeaders = nullptr;

		for ( uint32_t i = 0; i < sceneInput.instanceCount; i++ )
		{
			frames.push_back( frame ? frame.value() : transform );
		}

		OrochiUtils::malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), frames.size() );
		OrochiUtils::copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), frames.data(), frames.size() );
		scene.m_garbageCollector.push_back( sceneInput.instanceFrames );

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = bvhBuildFlag;
		CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( scene.m_ctx, sceneInput, options, sceneTempSize ) );
		if ( sceneTempSize > 0 )
		{
			OrochiUtils::malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );
			scene.m_garbageCollector.push_back( sceneTemp );
		}

#ifdef ENABLE_EMBREE
		if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport )
		{
			buildEmbreeSceneBvh( embreeDevice, geomBoxes, frames, sceneInput );
			scene.m_garbageCollector.push_back( sceneInput.nodeList.nodes );
		}
#endif

		CHECK_HIPRT( hiprtCreateScene( scene.m_ctx, sceneInput, options, sceneLocal ) );

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		CHECK_HIPRT( hiprtBuildScene( scene.m_ctx, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, sceneLocal ) );
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		bvhBuildTime += ( end - begin );

		std::cout << "Bvh build time : " << std::chrono::duration_cast<std::chrono::milliseconds>( bvhBuildTime ).count()
				  << " ms" << std::endl;
		scene.m_scene = sceneLocal;
	}

#ifdef ENABLE_EMBREE
	if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport ) rtcReleaseDevice( embreeDevice );
#endif
}

void SceneDemo::render(
	std::optional<std::filesystem::path> imgPath,
	const std::filesystem::path&		 kernelPath,
	const std::string&					 funcName,
	float								 aoRadius )
{
	uint8_t* dst;
	OrochiUtils::malloc( dst, m_res.x * m_res.y * 4 );
	OrochiUtils::memset( dst, 0, m_res.x * m_res.y * 4 );
	m_scene.m_garbageCollector.push_back( dst );

	uint32_t	   stackSize		  = 64u;
	const uint32_t sharedStackSize	  = 16u;
	const uint32_t blockWidth		  = 8u;
	const uint32_t blockHeight		  = 8u;
	const uint32_t blockSize		  = blockWidth * blockHeight;
	std::string	   blockSizeDef		  = "-DBLOCK_SIZE=" + std::to_string( blockSize );
	std::string	   sharedStackSizeDef = "-DSHARED_STACK_SIZE=" + std::to_string( sharedStackSize );

	std::vector<const char*> opts;
	opts.push_back( blockSizeDef.c_str() );
	opts.push_back( sharedStackSizeDef.c_str() );
	// opts.push_back( "-G" );

	hiprtGlobalStackBufferInput stackBufferInput{
		hiprtStackTypeGlobal, hiprtStackEntryTypeInteger, stackSize, static_cast<uint32_t>( m_res.x * m_res.y ) };
	if constexpr ( UseDynamicStack ) stackBufferInput.type = hiprtStackTypeDynamic;
	hiprtGlobalStackBuffer stackBuffer;
	CHECK_HIPRT( hiprtCreateGlobalStackBuffer( m_scene.m_ctx, stackBufferInput, stackBuffer ) );

	oroFunction	   func		 = nullptr;
	hiprtFuncTable funcTable = nullptr;

	buildTraceKernelFromBitcode( m_scene.m_ctx, kernelPath.u8string().c_str(), funcName.c_str(), func );

	int2  res	 = make_int2( m_res.x, m_res.y );
	void* args[] = {
		&m_scene.m_scene,
		&dst,
		&res,
		&stackBuffer,
		&m_camera,
		&m_scene.m_bufMaterialIndices,
		&m_scene.m_bufMaterials,
		&m_scene.m_bufMatIdsPerInstance,
		&m_scene.m_indices,
		&m_scene.m_indexOffsets,
		&m_scene.m_normals,
		&m_scene.m_normalOffsets,
		&m_scene.m_numOfLights,
		&m_scene.m_lights,
		&aoRadius,
		&funcTable };

	int numRegs;
	CHECK_ORO( oroFuncGetAttribute( &numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, func ) );

	int numSmem;
	CHECK_ORO( oroFuncGetAttribute( &numSmem, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func ) );

	std::cout << "Trace kernel: registers " << numRegs << ", shared memory " << numSmem << std::endl;
	OrochiUtils::waitForCompletion();
	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
	launchKernel( func, m_res.x, m_res.y, blockWidth, blockHeight, args );

	OrochiUtils::waitForCompletion();
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	CHECK_HIPRT( hiprtDestroyGlobalStackBuffer( m_scene.m_ctx, stackBuffer ) );

	std::cout << "Ray cast time: " << std::chrono::duration_cast<std::chrono::milliseconds>( end - begin ).count() << " ms"
			  << std::endl;

	writeImage( imgPath.value().u8string().c_str(), m_res.x, m_res.y, dst );

	return;
}

void SceneDemo::deleteScene( SceneData& scene )
{
	CHECK_HIPRT( hiprtDestroyScene( scene.m_ctx, scene.m_scene ) );
	CHECK_HIPRT(
		hiprtDestroyGeometries( scene.m_ctx, static_cast<uint32_t>( scene.m_geometries.size() ), scene.m_geometries.data() ) );
	CHECK_HIPRT( hiprtDestroyContext( scene.m_ctx ) );
}
