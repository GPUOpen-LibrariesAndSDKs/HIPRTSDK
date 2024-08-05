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

#pragma once

#include <tutorials/common/TutorialBase.h>

class SceneDemo : public TutorialBase
{
  public:
	SceneDemo()
	{
		m_oroCtx = nullptr;

		m_ctxtInput.ctxt	   = nullptr;
		m_ctxtInput.device	   = 0;
		m_ctxtInput.deviceType = (hiprtDeviceType)-1;

		m_scene.Clear();
	}

	void buildBvh( hiprtGeometryBuildInput& buildInput );

#ifdef ENABLE_EMBREE
	void buildEmbreeBvh(
		RTCDevice embreeDevice, std::vector<RTCBuildPrimitive>& embreePrims, std::vector<hiprtBvhNode>& nodes, void* geomData );

	void buildEmbreeGeometryBvh(
		RTCDevice embreeDevice, const float3* vertices, const uint32_t* indices, hiprtGeometryBuildInput& buildInput );

	void buildEmbreeSceneBvh(
		RTCDevice						  embreeDevice,
		const std::vector<Aabb>&		  geomBoxes,
		const std::vector<hiprtFrameSRT>& frames,
		hiprtSceneBuildInput&			  buildInput );
#endif

	struct SceneData
	{
		void Clear()
		{
			m_bufMaterialIndices   = nullptr;
			m_bufMatIdsPerInstance = nullptr;
			m_bufMaterials		   = nullptr;
			m_vertices			   = nullptr;
			m_vertexOffsets		   = nullptr;
			m_normals			   = nullptr;
			m_normalOffsets		   = nullptr;
			m_indices			   = nullptr;
			m_indexOffsets		   = nullptr;
			m_lights			   = nullptr;
			m_numOfLights		   = nullptr;
			m_scene				   = nullptr;
			m_ctx				   = nullptr;
		}

		uint32_t* m_bufMaterialIndices;
		uint32_t* m_bufMatIdsPerInstance; // count of material ids per instance use to calculate offset in material Idx buffer
										  // for instance
		Material*				   m_bufMaterials;
		float3*					   m_vertices;
		uint32_t*				   m_vertexOffsets;
		float3*					   m_normals;
		uint32_t*				   m_normalOffsets;
		uint32_t*				   m_indices;
		uint32_t*				   m_indexOffsets;
		Light*					   m_lights;
		uint32_t*				   m_numOfLights;
		hiprtScene				   m_scene;
		std::vector<hiprtGeometry> m_geometries;
		std::vector<hiprtInstance> m_instances;
		std::vector<void*>		   m_garbageCollector;
		hiprtContext			   m_ctx;
	};

	Camera createCamera()
	{
		Camera camera;
		camera.m_translation = make_float3( 0.0f, 2.5f, 5.8f );
		camera.m_rotation	 = make_float4( 0.0f, 0.0f, 1.0f, 0.0f );
		camera.m_fov		 = 45.0f * hiprt::Pi / 180.f;
		return camera;
	}

	void createScene(
		SceneData&					 scene,
		const std::string&			 filename,
		const std::string&			 mtlBaseDir,
		bool						 enableRayMask = false,
		std::optional<hiprtFrameSRT> frame		   = std::nullopt,
		hiprtBuildFlags				 bvhBuildFlag  = hiprtBuildFlagBitPreferFastBuild );

	void setupScene(
		Camera&						 camera,
		const std::string&			 filePath,
		const std::string&			 dirPath,
		bool						 enableRayMask = false,
		std::optional<hiprtFrameSRT> frame		   = std::nullopt,
		hiprtBuildFlags				 bvhBuildFlag  = hiprtBuildFlagBitPreferFastBuild );

	void deleteScene( SceneData& scene );

	void TearDown()
	{
		for ( auto p : m_scene.m_garbageCollector )
			free( p );

		CHECK_ORO( oroCtxDestroy( m_oroCtx ) );
	}

	void render(
		std::optional<std::filesystem::path> imgPath,
		const std::filesystem::path&		 kernelPath,
		const std::string&					 funcName = "PrimaryRayKernel",
		float								 aoRadius = 0.0f );

	SceneData m_scene;
	Camera	  m_camera;
};
