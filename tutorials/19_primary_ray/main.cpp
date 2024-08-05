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

#include <numeric>
#include <tutorials/common/CornellBox.h>
#include <tutorials/common/SceneDemo.h>
#include <tutorials/common/TutorialBase.h>

class Tutorial : public SceneDemo
{
  public:
	void run()
	{

		struct OPTION_DEF
		{
			OPTION_DEF( uint32_t i_, const std::string& name_ )
			{
				i	 = i_;
				name = name_;
			}
			uint32_t	i;
			std::string name;
		};
		std::vector<OPTION_DEF> optionsList = {
			OPTION_DEF( VisualizeNormal, "normal" ),
			OPTION_DEF( VisualizeUv, "uv" ),
			OPTION_DEF( VisualizeId, "primId" ),
			OPTION_DEF( VisualizeHitDist, "depth" ),
		};

		Camera camera = createCamera();
		setupScene(
			camera,
			"../common/meshes/cornellpot/cornellpot.obj",
			"../common/meshes/cornellpot/",
			false,
			std::nullopt,
			hiprtBuildFlagBitPreferFastBuild );

		for ( const auto& o : optionsList )
		{
			const std::string kernelName = "PrimaryRayKernel_" + std::to_string( o.i );

			render( "19_primary_ray_" + o.name + ".png", "../common/PrimaryRayKernel.h", kernelName );
		}

		deleteScene( m_scene );
	}
};

int main( int argc, char** argv )
{
	Tutorial tutorial;
	tutorial.init( 0 );
	tutorial.run();
	return 0;
}
