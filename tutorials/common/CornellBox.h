#pragma once
#include <array>
#include <hiprt/hiprt_types.h>

constexpr uint32_t CornellBoxTriangleCount = 32;
constexpr uint32_t CornellBoxMaterialCount = 4;

const static std::array<hiprtFloat3, CornellBoxTriangleCount* 3> cornellBoxVertices = { { // Floor  -- white lambert
																						  { 0.0f, 0.0f, 0.0f },
																						  { 0.0f, 0.0f, 559.2f },
																						  { 556.0f, 0.0f, 559.2f },
																						  { 0.0f, 0.0f, 0.0f },
																						  { 556.0f, 0.0f, 559.2f },
																						  { 556.0f, 0.0f, 0.0f },

																						  // Ceiling -- white lambert
																						  { 0.0f, 548.8f, 0.0f },
																						  { 556.0f, 548.8f, 0.0f },
																						  { 556.0f, 548.8f, 559.2f },

																						  { 0.0f, 548.8f, 0.0f },
																						  { 556.0f, 548.8f, 559.2f },
																						  { 0.0f, 548.8f, 559.2f },

																						  // Back wall -- white lambert
																						  { 0.0f, 0.0f, 559.2f },
																						  { 0.0f, 548.8f, 559.2f },
																						  { 556.0f, 548.8f, 559.2f },

																						  { 0.0f, 0.0f, 559.2f },
																						  { 556.0f, 548.8f, 559.2f },
																						  { 556.0f, 0.0f, 559.2f },

																						  // Right wall -- green lambert
																						  { 0.0f, 0.0f, 0.0f },
																						  { 0.0f, 548.8f, 0.0f },
																						  { 0.0f, 548.8f, 559.2f },

																						  { 0.0f, 0.0f, 0.0f },
																						  { 0.0f, 548.8f, 559.2f },
																						  { 0.0f, 0.0f, 559.2f },

																						  // Left wall -- red lambert
																						  { 556.0f, 0.0f, 0.0f },
																						  { 556.0f, 0.0f, 559.2f },
																						  { 556.0f, 548.8f, 559.2f },

																						  { 556.0f, 0.0f, 0.0f },
																						  { 556.0f, 548.8f, 559.2f },
																						  { 556.0f, 548.8f, 0.0f },

																						  // Short block -- white lambert
																						  { 130.0f, 165.0f, 65.0f },
																						  { 82.0f, 165.0f, 225.0f },
																						  { 242.0f, 165.0f, 274.0f },

																						  { 130.0f, 165.0f, 65.0f },
																						  { 242.0f, 165.0f, 274.0f },
																						  { 290.0f, 165.0f, 114.0f },

																						  { 290.0f, 0.0f, 114.0f },
																						  { 290.0f, 165.0f, 114.0f },
																						  { 240.0f, 165.0f, 272.0f },

																						  { 290.0f, 0.0f, 114.0f },
																						  { 240.0f, 165.0f, 272.0f },
																						  { 240.0f, 0.0f, 272.0f },

																						  { 130.0f, 0.0f, 65.0f },
																						  { 130.0f, 165.0f, 65.0f },
																						  { 290.0f, 165.0f, 114.0f },

																						  { 130.0f, 0.0f, 65.0f },
																						  { 290.0f, 165.0f, 114.0f },
																						  { 290.0f, 0.0f, 114.0f },

																						  { 82.0f, 0.0f, 225.0f },
																						  { 82.0f, 165.0f, 225.0f },
																						  { 130.0f, 165.0f, 65.0f },

																						  { 82.0f, 0.0f, 225.0f },
																						  { 130.0f, 165.0f, 65.0f },
																						  { 130.0f, 0.0f, 65.0f },

																						  { 240.0f, 0.0f, 272.0f },
																						  { 240.0f, 165.0f, 272.0f },
																						  { 82.0f, 165.0f, 225.0f },

																						  { 240.0f, 0.0f, 272.0f },
																						  { 82.0f, 165.0f, 225.0f },
																						  { 82.0f, 0.0f, 225.0f },

																						  // Tall block -- white lambert
																						  { 423.0f, 330.0f, 247.0f },
																						  { 265.0f, 330.0f, 296.0f },
																						  { 314.0f, 330.0f, 455.0f },

																						  { 423.0f, 330.0f, 247.0f },
																						  { 314.0f, 330.0f, 455.0f },
																						  { 472.0f, 330.0f, 406.0f },

																						  { 423.0f, 0.0f, 247.0f },
																						  { 423.0f, 330.0f, 247.0f },
																						  { 472.0f, 330.0f, 406.0f },

																						  { 423.0f, 0.0f, 247.0f },
																						  { 472.0f, 330.0f, 406.0f },
																						  { 472.0f, 0.0f, 406.0f },

																						  { 472.0f, 0.0f, 406.0f },
																						  { 472.0f, 330.0f, 406.0f },
																						  { 314.0f, 330.0f, 456.0f },

																						  { 472.0f, 0.0f, 406.0f },
																						  { 314.0f, 330.0f, 456.0f },
																						  { 314.0f, 0.0f, 456.0f },

																						  { 314.0f, 0.0f, 456.0f },
																						  { 314.0f, 330.0f, 456.0f },
																						  { 265.0f, 330.0f, 296.0f },

																						  { 314.0f, 0.0f, 456.0f },
																						  { 265.0f, 330.0f, 296.0f },
																						  { 265.0f, 0.0f, 296.0f },

																						  { 265.0f, 0.0f, 296.0f },
																						  { 265.0f, 330.0f, 296.0f },
																						  { 423.0f, 330.0f, 247.0f },

																						  { 265.0f, 0.0f, 296.0f },
																						  { 423.0f, 330.0f, 247.0f },
																						  { 423.0f, 0.0f, 247.0f },

																						  // Ceiling light -- emmissive
																						  { 343.0f, 548.6f, 227.0f },
																						  { 213.0f, 548.6f, 227.0f },
																						  { 213.0f, 548.6f, 332.0f },

																						  { 343.0f, 548.6f, 227.0f },
																						  { 213.0f, 548.6f, 332.0f },
																						  { 343.0f, 548.6f, 332.0f } } };

static std::array<uint32_t, CornellBoxTriangleCount> cornellBoxMatIndices = { {
	0, 0,						  // Floor         -- white lambert
	0, 0,						  // Ceiling       -- white lambert
	0, 0,						  // Back wall     -- white lambert
	1, 1,						  // Right wall    -- green lambert
	2, 2,						  // Left wall     -- red lambert
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Short block   -- white lambert
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // Tall block    -- white lambert
	3, 3						  // Ceiling light -- emmissive
} };

const std::array<hiprtFloat3, CornellBoxMaterialCount> cornellBoxEmissionColors = {
	{ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 15.0f, 15.0f, 5.0f }

	} };

const std::array<hiprtFloat3, CornellBoxMaterialCount> cornellBoxDiffuseColors = {
	{ { 0.80f, 0.80f, 0.80f }, { 0.05f, 0.80f, 0.05f }, { 0.80f, 0.05f, 0.05f }, { 0.50f, 0.00f, 0.00f } } };
