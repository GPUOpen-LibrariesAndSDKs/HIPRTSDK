#if !defined( HIPRT_DEVICE_H )
#define HIPRT_DEVICE_H

enum hiprtTraversalType
{
	hiprtTraversalTerminateAtAnytHit = 1,	 /*!< 0 or 1 element iterator with
											  any hit along the ray*/
	hiprtTraversalTerminateAtClosestHit = 2, /*!< 0 or 1 element iterator with a
											closest hit along the ray*/
};

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
typedef unsigned int		   hiprtBuildFlags;
typedef unsigned int		   hiprtRayMask;
typedef _hiprtContext*		   hiprtContext;
typedef _hiprtCustomFuncTable* hiprtCustomFuncTable;

typedef bool ( *hiprtIntersectFunc )(
	const hiprtRay& ray, unsigned int primIdx, const void* userPtr, const void* payload, float2& uv, float3& normal, float& t );

typedef struct
{
	hiprtIntersectFunc intersectFunc;
	const void*		   intersectFuncData;
} hiprtCustomFuncSet;

enum
{
	hiprtInvalidValue = ~0u,
	hiprtSentinel	  = 0x76543210
};

struct hiprtRay
{
	float3 origin;
	float  time;
	float3 direction;
	float  maxT;
};

struct hiprtHit
{
	unsigned int instanceID;
	unsigned int primID;
	float2		 uv;
	float3		 normal;
	float		 t;

	HIPRT_DEVICE bool hasHit() const { return primID != hiprtInvalidValue; }
};

namespace hiprt
{
template <hiprtTraversalType TraversalType>
class GeomTraversal
{
  public:
	HIPRT_DEVICE		  GeomTraversal( hiprtGeometry geom, const hiprtRay& ray, hiprtRayMask mask );
	HIPRT_DEVICE hiprtHit getNextHit();
};

template class GeomTraversal<hiprtTraversalTerminateAtAnytHit>;
template class GeomTraversal<hiprtTraversalTerminateAtClosestHit>;

template <hiprtTraversalType TraversalType>
class GeomCustomTraversal
{
  public:
	HIPRT_DEVICE
	GeomCustomTraversal( hiprtGeometryCustom geom, const hiprtRay& ray, hiprtRayMask mask, hiprtCustomFuncTable funcTable );
	HIPRT_DEVICE hiprtHit getNextHit();
};

template class GeomCustomTraversal<hiprtTraversalTerminateAtAnytHit>;
template class GeomCustomTraversal<hiprtTraversalTerminateAtClosestHit>;

template <hiprtTraversalType TraversalType>
class SceneTraversal
{
  public:
	HIPRT_DEVICE SceneTraversal( hiprtScene geom, const hiprtRay& ray, hiprtRayMask mask, hiprtCustomFuncTable funcTable );
	HIPRT_DEVICE hiprtHit getNextHit();
};

template class GeomCustomTraversal<hiprtTraversalTerminateAtAnytHit>;
template class GeomCustomTraversal<hiprtTraversalTerminateAtClosestHit>;
}; // namespace hiprt

typedef hiprt::GeomTraversal<hiprtTraversalTerminateAtAnytHit>	  hiprtGeomTraversalAnyHit;
typedef hiprt::GeomTraversal<hiprtTraversalTerminateAtClosestHit> hiprtGeomTraversalClosest;

typedef hiprt::GeomCustomTraversal<hiprtTraversalTerminateAtAnytHit>	hiprtGeomCustomTraversalAnyHit;
typedef hiprt::GeomCustomTraversal<hiprtTraversalTerminateAtClosestHit> hiprtGeomCustomTraversalClosest;

typedef hiprt::SceneTraversal<hiprtTraversalTerminateAtAnytHit>	   hiprtScenemTraversalAnyHit;
typedef hiprt::SceneTraversal<hiprtTraversalTerminateAtClosestHit> hiprtSceneTraversalClosest;

#endif
