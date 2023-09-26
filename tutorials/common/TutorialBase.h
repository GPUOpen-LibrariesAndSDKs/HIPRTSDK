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

#pragma once
#include <Orochi/Orochi.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <hiprt/hiprt.h>
#include <hiprt/hiprt_vec.h>
#include <optional>
#include <string>
#include <tutorials/common/Common.h>
#include <vector>

#define CHECK_ORO( error ) ( checkOro( error, __FILE__, __LINE__ ) )
void checkOro( oroError res, const char* file, int line );

#define CHECK_HIPRT( error ) ( checkHiprt( error, __FILE__, __LINE__ ) )
void checkHiprt( hiprtError res, const char* file, int line );

#define CHECK_ORORTC( error ) ( checkOrortc( error, __FILE__, __LINE__ ) )
void checkOrortc( orortcResult res, const char* file, int line );

typedef unsigned char u8;

constexpr float Pi	  = 3.14159265358979323846f;
constexpr float TwoPi = 2.0f * Pi;

class TutorialBase
{
  public:
	virtual ~TutorialBase() { printf( "success\n" ); }
	void init( int deviceIndex = 0 );

	virtual void run() = 0;

	void buildTraceKernelFromBitcode(
		hiprtContext				   ctxt,
		const char*					   path,
		const char*					   functionName,
		oroFunction&				   functionOut,
		std::vector<const char*>*	   opts			= nullptr,
		std::vector<hiprtFuncNameSet>* funcNameSets = nullptr,
		int							   numGeomTypes = 0,
		int							   numRayTypes	= 1 );

	void launchKernel( oroFunction func, int nx, int ny, void** args );
	void launchKernel( oroFunction func, int nx, int ny, int bx, int by, void** args );

	static void writeImage( const std::string& path, int w, int h, u8* pixels );

	static bool readSourceCode(
		const std::filesystem::path&					  path,
		std::string&									  sourceCode,
		std::optional<std::vector<std::filesystem::path>> includes = std::nullopt );

  protected:
	hiprtContextCreationInput m_ctxtInput;
	oroCtx					  m_oroCtx;
	oroDevice				  m_oroDevice;
	hiprtInt2				  m_res;
};
