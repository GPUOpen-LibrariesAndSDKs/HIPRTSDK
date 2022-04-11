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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#if !defined( HIPRT_DEVICE_H )
#define HIPRT_DEVICE_H

/** \brief Ray traversal type.
 *
 */
enum hiprtTraversalType
{
	/*!< 0 or 1 element iterator with any hit along the ray */
	hiprtTraversalTerminateAtAnyHit = 1,
	/*!< 0 or 1 element iterator with a closest hit along the ray */
	hiprtTraversalTerminateAtClosestHit = 2,
};

/** \brief Traversal state.
 *
 * On-device traversal can be in either hit state (and can be continued using
 * hiprtNextHit) or finished state.
 */
typedef enum
{
	hiprtTraversalStateInit,
	hiprtTraversalStateFinished,
	hiprtTraversalStateHit
} hiprtTraversalState;

struct _hiprtContext;
struct _hiprtCustomFuncTable;

typedef void*				   hiprtDevicePtr;
typedef hiprtDevicePtr		   hiprtGeometry;
typedef hiprtDevicePtr		   hiprtScene;
typedef uint32_t			   hiprtBuildFlags;
typedef uint32_t			   hiprtRayMask;
typedef _hiprtContext*		   hiprtContext;
typedef _hiprtCustomFuncTable* hiprtCustomFuncTable;

/** \brief Insersection function for custom primitives.
 *
 * \param ray Ray.
 * \param primID Primtive ID.
 * \param userPtr User data.
 * \param payload Payload for additional outputs.
 * \param uv Output texture coordinates.
 * \param normal Output normal.
 * \param t Output distance.
 * \return A flag indicating hit.
 */
typedef bool ( *hiprtIntersectFunc )(
	const hiprtRay& ray,
	uint32_t primID,
	const void* userPtr,
	const void* payload,
	hiprtFloat2& uv,
	hiprtFloat3& normal,
	float& t );

/** \brief Set of functions for custom primitives.
 *
 */
typedef struct
{
	hiprtIntersectFunc intersectFunc;
	const void*		   intersectFuncData;
} hiprtCustomFuncSet;

/** \brief Various constants.
 *
 */
enum : uint32_t
{
	hiprtInvalidValue = ~0u,
};

/** \brief Ray data structure.
 *
 */
struct hiprtRay
{
	/*!< Ray origin */
	hiprtFloat3 origin;
	/*!< Ray time for motion blur */
	float time;
	/*!< Ray direction */
	hiprtFloat3 direction;
	/*!< Ray maximum distance */
	float maxT;
};

/** \brief Ray hit data structure.
 *
 */
struct hiprtHit
{
	/*!< Instance ID */
	uint32_t instanceID;
	/*!< Primitive ID */
	uint32_t primID;
	/*!< Texture coordinates */
	hiprtFloat2 uv;
	/*!< Geeometric normal (not normalized) */
	hiprtFloat3 normal;
	/*!< Distance */
	float t;
};

/** \brief A stack using (slow) local memory internally.
 *
 */
template <uint32_t PrivateStackSize>
class hiprtCustomPrivateStack
{
  public:
	HIPRT_DEVICE hiprtCustomPrivateStack();
	HIPRT_DEVICE int  pop();
	HIPRT_DEVICE void push( int val );
	HIPRT_DEVICE bool empty();
	HIPRT_DEVICE void reset();
};

/** \brief A stack using both (fast) shared memory and (slow) global memory.
 *
 * The stack uses shared memory if there is enough space.
 * Otherwise, it uses global memory as a backup.
 */
template <uint32_t SharedStackSize>
class hiprtCustomSharedStack
{
  public:
	HIPRT_DEVICE hiprtCustomSharedStack( int* globalStackBuffer, int* sharedStackBuffer = nullptr );
	HIPRT_DEVICE int  pop();
	HIPRT_DEVICE void push( int val );
	HIPRT_DEVICE bool empty();
	HIPRT_DEVICE void reset();
};

typedef hiprtCustomPrivateStack<48> hiprtPrivateStack48;
typedef hiprtCustomPrivateStack<64> hiprtPrivateStack64;
typedef hiprtCustomPrivateStack<128> hiprtPrivateStack128;

typedef hiprtCustomSharedStack<0> hiprtGlobalStack;
typedef hiprtCustomSharedStack<16> hiprtSharedStack16;
typedef hiprtCustomSharedStack<24> hiprtSharedStack24;

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing triangles.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalClosest( hiprtGeometry geom, const hiprtRay& ray );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing triangles.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalAnyHit( hiprtGeometry geom, const hiprtRay& ray );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing custom primitives.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomCustomTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalClosest( hiprtGeometry geom, const hiprtRay& ray, hiprtCustomFuncSet funcSet );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing custom primitives.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomCustomTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalAnyHit( hiprtGeometry geom, const hiprtRay& ray, hiprtCustomFuncSet funcSet );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtScene.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtSceneTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalClosest( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask );
	HIPRT_DEVICE hiprtSceneTraversalClosest( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask, hiprtCustomFuncSet funcSet );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtScene.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtSceneTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalAnyHit( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask );
	HIPRT_DEVICE hiprtSceneTraversalAnyHit( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask, hiprtCustomFuncSet funcSet );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing triangles.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalClosestCustomStack( hiprtGeometry geom, const hiprtRay& ray, hiprtStack& stack );
	HIPRT_DEVICE hiprtHit getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing triangles.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalAnyHitCustomStack( hiprtGeometry geom, const hiprtRay& ray, hiprtStack& stack );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing custom primitives.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomCustomTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalClosestCustomStack( hiprtGeometry geom, const hiprtRay& ray, hiprtCustomFuncSet funcSet, hiprtStack& stack );
	HIPRT_DEVICE hiprtHit getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing custom primitives.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomCustomTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalAnyHitCustomStack( hiprtGeometry geom, const hiprtRay& ray, hiprtCustomFuncSet funcSet, hiprtStack& stack );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtScene.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtSceneTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalClosestCustomStack( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask, hiprtStack& stack );
	HIPRT_DEVICE hiprtSceneTraversalClosestCustomStack( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask, hiprtCustomFuncSet funcSet, hiprtStack& stack );
	HIPRT_DEVICE hiprtHit getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtScene.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtSceneTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalAnyHitCustomStack( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask, hiprtStack& stack );
	HIPRT_DEVICE hiprtSceneTraversalAnyHitCustomStack( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask, hiprtCustomFuncSet funcSet, hiprtStack& stack );
	HIPRT_DEVICE hiprtHit getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};
#endif
