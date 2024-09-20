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

#include <filesystem>
#include <tutorials/common/Aabb.h>
#include <tutorials/common/FluidSimulation.h>
#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	static constexpr std::string_view OutputDir		  = "fluid_simulation_output";
	static constexpr bool			  ExportAllFrames = false;
	static constexpr uint32_t		  FrameCount	  = 200u;
	static constexpr uint32_t		  FrameToExport	  = 165u;

	static constexpr float PoolVolumeDim			  = 1.0f;
	static constexpr float PoolSpaceDivision		  = 50.0f;
	static constexpr float InitParticleVolumeDim	  = 0.6f;
	static constexpr float InitParticleVolumeCenter[] = {
		-0.45f * ( PoolVolumeDim - InitParticleVolumeDim ),
		PoolVolumeDim - InitParticleVolumeDim * 0.5f,
		0.45f * ( PoolVolumeDim - InitParticleVolumeDim ) };
	static constexpr float ParticleRestDensity	= 1000.0f;
	static constexpr float ParticleSmoothRadius = PoolVolumeDim / PoolSpaceDivision;

	void run()
	{
		// Init constant data
		Simulation sim;
		{
			sim.m_smoothRadius		= ParticleSmoothRadius;
			sim.m_pressureStiffness = 200.0f;
			sim.m_restDensity		= ParticleRestDensity;
			sim.m_wallStiffness		= 3000.0f;
			sim.m_particleCount		= 131072;
			sim.m_planes[0]			= { 0.0f, 1.0f, 0.0f, 0.0f };
			sim.m_planes[1]			= { 0.0f, -1.0f, 0.0f, PoolVolumeDim };
			sim.m_planes[2]			= { 1.0f, 0.0f, 0.0f, 0.5f * PoolVolumeDim };
			sim.m_planes[3]			= { -1.0f, 0.0f, 0.0f, 0.5f * PoolVolumeDim };
			sim.m_planes[4]			= { 0.0f, 0.0f, 1.0f, 0.5f * PoolVolumeDim };
			sim.m_planes[5]			= { 0.0f, 0.0f, -1.0f, 0.5f * PoolVolumeDim };

			const float initVolume	   = InitParticleVolumeDim * InitParticleVolumeDim * InitParticleVolumeDim;
			const float mass		   = sim.m_restDensity * initVolume / sim.m_particleCount;
			const float viscosity	   = 0.4f;
			sim.m_densityCoef		   = mass * 315.0f / ( 64.0f * hiprt::Pi * pow( sim.m_smoothRadius, 9.0f ) );
			sim.m_pressureGradCoef	   = mass * -45.0f / ( hiprt::Pi * pow( sim.m_smoothRadius, 6.0f ) );
			sim.m_viscosityLaplaceCoef = mass * viscosity * 45.0f / ( hiprt::Pi * pow( sim.m_smoothRadius, 6.0f ) );
		}

		// Init data
		std::vector<Particle> particles( sim.m_particleCount );

		const auto smoothRadius = ParticleSmoothRadius;
		const auto dimSize		= static_cast<uint32_t>( ceil( std::cbrt( sim.m_particleCount ) ) );
		const auto slcSize		= dimSize * dimSize;
		for ( uint32_t i = 0; i < sim.m_particleCount; ++i )
		{
			const auto n = i % slcSize;
			auto	   x = ( n % dimSize ) / static_cast<float>( dimSize );
			auto	   y = ( n / dimSize ) / static_cast<float>( dimSize );
			auto	   z = ( i / slcSize ) / static_cast<float>( dimSize );
			x			 = InitParticleVolumeDim * ( x - 0.5f ) + InitParticleVolumeCenter[0];
			y			 = InitParticleVolumeDim * ( y - 0.5f ) + InitParticleVolumeCenter[1];
			z			 = InitParticleVolumeDim * ( z - 0.5f ) + InitParticleVolumeCenter[2];

			particles[i].Pos	  = { x, y, z };
			particles[i].Velocity = { 0.0f, 0.0f, 0.0f };
		}

		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtAABBListPrimitive list;
		list.aabbCount	= sim.m_particleCount;
		list.aabbStride = sizeof( Aabb );
		std::vector<Aabb> aabbs( sim.m_particleCount );
		for ( uint32_t i = 0; i < sim.m_particleCount; ++i )
		{
			const hiprtFloat3& c = particles[i].Pos;
			aabbs[i].m_min.x	 = c.x - smoothRadius;
			aabbs[i].m_max.x	 = c.x + smoothRadius;
			aabbs[i].m_min.y	 = c.y - smoothRadius;
			aabbs[i].m_max.y	 = c.y + smoothRadius;
			aabbs[i].m_min.z	 = c.z - smoothRadius;
			aabbs[i].m_max.z	 = c.z + smoothRadius;
		}
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &list.aabbs ), sim.m_particleCount * sizeof( Aabb ) ) );
		CHECK_ORO(
			oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( list.aabbs ), aabbs.data(), sim.m_particleCount * sizeof( Aabb ) ) );

		hiprtGeometryBuildInput geomInput;
		geomInput.type				 = hiprtPrimitiveTypeAABBList;
		geomInput.primitive.aabbList = list;
		geomInput.geomType			 = 0;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

		hiprtGeometry geom;
		CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );

		hiprtFuncNameSet funcNameSet;
		funcNameSet.intersectFuncName			   = "intersectParticleImpactSphere";
		std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

		hiprtFuncDataSet funcDataSet;
		CHECK_ORO( oroMalloc(
			const_cast<oroDeviceptr*>( &funcDataSet.intersectFuncData ), sim.m_particleCount * sizeof( Particle ) ) );
		CHECK_ORO( oroMemcpyHtoD(
			const_cast<oroDeviceptr>( funcDataSet.intersectFuncData ),
			particles.data(),
			sim.m_particleCount * sizeof( Particle ) ) );

		hiprtFuncTable funcTable;
		CHECK_HIPRT( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

		Simulation* pSim;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pSim ), sizeof( Simulation ) ) );
		CHECK_ORO( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( pSim ), &sim, sizeof( Simulation ) ) );

		// Density
		float*		densities;
		oroFunction densityFunc;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &densities ), sim.m_particleCount * sizeof( float ) ) );
		buildTraceKernelFromBitcode(
			ctxt, "../common/TutorialKernels.h", "DensityKernel", densityFunc, nullptr, &funcNameSets, 1, 1 );

		// Force
		hiprtFloat3* accelerations;
		oroFunction	 forceFunc;
		CHECK_ORO(
			oroMalloc( reinterpret_cast<oroDeviceptr*>( &accelerations ), sim.m_particleCount * sizeof( hiprtFloat3 ) ) );
		buildTraceKernelFromBitcode(
			ctxt, "../common/TutorialKernels.h", "ForceKernel", forceFunc, nullptr, &funcNameSets, 1, 1 );

		// Integration
		PerFrame*	pPerFrame;
		oroFunction intFunc;
		{
			PerFrame perFrame;
			perFrame.m_timeStep = 1.0f / 320.0f;
			perFrame.m_gravity	= { 0.0f, -9.8f, 0.0f };
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pPerFrame ), sizeof( PerFrame ) ) );
			CHECK_ORO( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( pPerFrame ), &perFrame, sizeof( PerFrame ) ) );
			buildTraceKernelFromBitcode( ctxt, "../common/TutorialKernels.h", "IntegrationKernel", intFunc );
		}

		// Visualization
		float4x4*	pViewProj;
		uint8_t*	pixels;
		oroFunction visFunc;
		{
			// Projection matrix
			float	 aspect = m_res.x / static_cast<float>( m_res.y );
			float4x4 proj	= Perspective( hiprt::Pi / 4.0f, aspect, 1.0f, 40.0f );

			// View matrix
			hiprtFloat3 focusPt = { 0.0f, 0.5f, 0.0f };
			hiprtFloat3 eyePt	= focusPt - make_hiprtFloat3( -0.5f, -0.5f, 2.0f );
			float4x4	view	= LookAt( eyePt, focusPt, make_hiprtFloat3( 0.0f, 1.0f, 0.0f ) );

			float4x4 viewProj = proj * view;

			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pViewProj ), sizeof( float4x4 ) ) );
			CHECK_ORO( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( pViewProj ), &viewProj, sizeof( float4x4 ) ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pixels ), m_res.x * m_res.y * 4 ) );
			buildTraceKernelFromBitcode( ctxt, "../common/TutorialKernels.h", "VisualizationKernel", visFunc );
		}

		// Launch kernels
		const uint32_t b  = 64;
		uint32_t	   nb = ( sim.m_particleCount + b - 1 ) / b;

		if constexpr ( ExportAllFrames ) std::filesystem::create_directory( OutputDir );

		// Simulate
		for ( uint32_t i = 0; i < FrameCount; ++i )
		{
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

			void* dArgs[] = { &geom, &densities, &funcDataSet.intersectFuncData, &pSim, &funcTable };
			CHECK_ORO( oroModuleLaunchKernel( densityFunc, nb, 1, 1, b, 1, 1, 0, 0, dArgs, 0 ) );

			void* fArgs[] = { &geom, &accelerations, &funcDataSet.intersectFuncData, &densities, &pSim, &funcTable };
			CHECK_ORO( oroModuleLaunchKernel( forceFunc, nb, 1, 1, b, 1, 1, 0, 0, fArgs, 0 ) );

			void* iArgs[] = { &funcDataSet.intersectFuncData, &list.aabbs, &accelerations, &pSim, &pPerFrame };
			CHECK_ORO( oroModuleLaunchKernel( intFunc, nb, 1, 1, b, 1, 1, 0, 0, iArgs, 0 ) );

			// Visualize
			CHECK_ORO( oroMemset( reinterpret_cast<oroDeviceptr>( pixels ), 0, m_res.x * m_res.y * 4 ) );
			void* vArgs[] = { &funcDataSet.intersectFuncData, &densities, &pixels, &m_res, &pViewProj };
			CHECK_ORO( oroModuleLaunchKernel( visFunc, nb, 1, 1, b, 1, 1, 0, 0, vArgs, 0 ) );

			if constexpr ( ExportAllFrames )
			{
				std::string imageName = std::string( OutputDir ) + "/" + std::to_string( i ) + ".png";
				writeImage( imageName.c_str(), m_res.x, m_res.y, pixels );
			}
			else
			{
				std::cout << "Fluid simulation: frame " << i << " done." << std::endl;
				if ( i == FrameToExport ) writeImage( "16_fluid_simulation.png", m_res.x, m_res.y, pixels );
			}
		}

		CHECK_ORO( oroFree( const_cast<oroDeviceptr>( funcDataSet.intersectFuncData ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( list.aabbs ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pSim ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( densities ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( accelerations ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pPerFrame ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pViewProj ) ) );
		CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pixels ) ) );

		CHECK_HIPRT( hiprtDestroyFuncTable( ctxt, funcTable ) );
		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom ) );
		CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
	}
};

int main( int argc, char** argv )
{
	Tutorial tutorial;
	tutorial.init( 0 );
	tutorial.run();

	return 0;
}
