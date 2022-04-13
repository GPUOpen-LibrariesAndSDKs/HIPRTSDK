#pragma once

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define __KERNELCC__
#endif

#if !defined( __KERNELCC__ )
#define HOST
#define DEVICE
#define HOST_DEVICE

#define int2 hiprtInt2
#define int3 hiprtInt3
#define int4 hiprtInt4

#define float2 hiprtFloat2
#define float3 hiprtFloat3
#define float4 hiprtFloat4

#define make_int2 make_hiprtInt2
#define make_int3 make_hiprtInt3
#define make_int4 make_hiprtInt4

#define make_float2 make_hiprtFloat2
#define make_float3 make_hiprtFloat3
#define make_float4 make_hiprtFloat4

#else
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#endif

#ifdef __CUDACC__
#define INLINE __forceinline__
#else
#define INLINE inline
#endif

typedef float4 Quaternion;

struct Material_t
{
	float4		m_diffuse;
	float4		m_emission;
	float4		m_params; // m_params.x - is light
} ;

struct Light_t
{
	float4		m_le;
	float3		m_lv0;
	float3		m_lv1;
	float3		m_lv2;
	float3		pad;
};

struct Camera
{
	float4	   m_translation; // eye/rayorigin
	Quaternion m_quat;
	float	   m_fov;
	float	   m_near;
	float	   m_far;
	float	   padd;
};

#define RT_MIN( a, b ) ( ( ( b ) < ( a ) ) ? ( b ) : ( a ) )
#define RT_MAX( a, b ) ( ( ( b ) > ( a ) ) ? ( b ) : ( a ) )

#if !defined( __KERNELCC__ )
HOST_DEVICE INLINE int2 make_int2( const float2 a ) { return make_int2( (int)a.x, (int)a.y ); }

HOST_DEVICE INLINE int2 make_int2( const int3& a ) { return make_int2( a.x, a.y ); }

HOST_DEVICE INLINE int2 make_int2( const int4& a ) { return make_int2( a.x, a.y ); }

HOST_DEVICE INLINE int2 make_int2( const int c ) { return make_int2( c, c ); }


HOST_DEVICE INLINE int2 operator+( const int2& a, const int2& b ) { return make_int2( a.x + b.x, a.y + b.y ); }

HOST_DEVICE INLINE int2 operator-( const int2& a, const int2& b ) { return make_int2( a.x - b.x, a.y - b.y ); }

HOST_DEVICE INLINE int2 operator*( const int2& a, const int2& b ) { return make_int2( a.x * b.x, a.y * b.y ); }

HOST_DEVICE INLINE int2 operator/( const int2& a, const int2& b ) { return make_int2( a.x / b.x, a.y / b.y ); }

HOST_DEVICE INLINE int2& operator+=( int2& a, const int2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HOST_DEVICE INLINE int2& operator-=( int2& a, const int2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HOST_DEVICE INLINE int2& operator*=( int2& a, const int2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HOST_DEVICE INLINE int2& operator/=( int2& a, const int2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HOST_DEVICE INLINE int2& operator+=( int2& a, const int c )
{
	a.x += c;
	a.y += c;
	return a;
}

HOST_DEVICE INLINE int2& operator-=( int2& a, const int c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HOST_DEVICE INLINE int2& operator*=( int2& a, const int c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HOST_DEVICE INLINE int2& operator/=( int2& a, const int c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HOST_DEVICE INLINE int2 operator-( const int2& a ) { return make_int2( -a.x, -a.y ); }

HOST_DEVICE INLINE int2 operator+( const int2& a, const int c ) { return make_int2( a.x + c, a.y + c ); }

HOST_DEVICE INLINE int2 operator+( const int c, const int2& a ) { return make_int2( c + a.x, c + a.y ); }

HOST_DEVICE INLINE int2 operator-( const int2& a, const int c ) { return make_int2( a.x - c, a.y - c ); }

HOST_DEVICE INLINE int2 operator-( const int c, const int2& a ) { return make_int2( c - a.x, c - a.y ); }

HOST_DEVICE INLINE int2 operator*( const int2& a, const int c ) { return make_int2( c * a.x, c * a.y ); }

HOST_DEVICE INLINE int2 operator*( const int c, const int2& a ) { return make_int2( c * a.x, c * a.y ); }

HOST_DEVICE INLINE int2 operator/( const int2& a, const int c ) { return make_int2( a.x / c, a.y / c ); }

HOST_DEVICE INLINE int2 operator/( const int c, const int2& a ) { return make_int2( c / a.x, c / a.y ); }

HOST_DEVICE INLINE int3 make_int3( const float3& a ) { return make_int3( (int)a.x, (int)a.y, (int)a.z ); }

HOST_DEVICE INLINE int3 make_int3( const int4& a ) { return make_int3( a.x, a.y, a.z ); }

HOST_DEVICE INLINE int3 make_int3( const int2& a, const int c ) { return make_int3( a.x, a.y, c ); }

HOST_DEVICE INLINE int3 make_int3( const int c ) { return make_int3( c, c, c ); }

HOST_DEVICE INLINE int3 operator+( const int3& a, const int3& b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

HOST_DEVICE INLINE int3 operator-( const int3& a, const int3& b )
{
	return make_int3( a.x - b.x, a.y - b.y, a.z - b.z );
}

HOST_DEVICE INLINE int3 operator*( const int3& a, const int3& b )
{
	return make_int3( a.x * b.x, a.y * b.y, a.z * b.z );
}

HOST_DEVICE INLINE int3 operator/( const int3& a, const int3& b )
{
	return make_int3( a.x / b.x, a.y / b.y, a.z / b.z );
}

HOST_DEVICE INLINE int3& operator+=( int3& a, const int3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HOST_DEVICE INLINE int3& operator-=( int3& a, const int3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HOST_DEVICE INLINE int3& operator*=( int3& a, const int3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HOST_DEVICE INLINE int3& operator/=( int3& a, const int3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HOST_DEVICE INLINE int3& operator+=( int3& a, const int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HOST_DEVICE INLINE int3& operator-=( int3& a, const int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HOST_DEVICE INLINE int3& operator*=( int3& a, const int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HOST_DEVICE INLINE int3& operator/=( int3& a, const int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HOST_DEVICE INLINE int3 operator-( const int3& a ) { return make_int3( -a.x, -a.y, -a.z ); }

HOST_DEVICE INLINE int3 operator+( const int3& a, const int c ) { return make_int3( c + a.x, c + a.y, c + a.z ); }

HOST_DEVICE INLINE int3 operator+( const int c, const int3& a ) { return make_int3( c + a.x, c + a.y, c + a.z ); }

HOST_DEVICE INLINE int3 operator-( const int3& a, const int c ) { return make_int3( a.x - c, a.y - c, a.z - c ); }

HOST_DEVICE INLINE int3 operator-( const int c, const int3& a ) { return make_int3( c - a.x, c - a.y, c - a.z ); }

HOST_DEVICE INLINE int3 operator*( const int3& a, const int c ) { return make_int3( c * a.x, c * a.y, c * a.z ); }

HOST_DEVICE INLINE int3 operator*( const int c, const int3& a ) { return make_int3( c * a.x, c * a.y, c * a.z ); }

HOST_DEVICE INLINE int3 operator/( const int3& a, const int c ) { return make_int3( a.x / c, a.y / c, a.z / c ); }

HOST_DEVICE INLINE int3 operator/( const int c, const int3& a ) { return make_int3( c / a.x, c / a.y, c / a.z ); }

HOST_DEVICE INLINE int4 make_int4( const float4& a ) { return make_int4( (int)a.x, (int)a.y, (int)a.z, (int)a.w ); }

HOST_DEVICE INLINE int4 make_int4( const int2& a, const int c0, const int c1 )
{
	return make_int4( a.x, a.y, c0, c1 );
}

HOST_DEVICE INLINE int4 make_int4( const int3& a, const int c ) { return make_int4( a.x, a.y, a.z, c ); }

HOST_DEVICE INLINE int4 make_int4( const int c ) { return make_int4( c, c, c, c ); }

HOST_DEVICE INLINE int4 operator+( const int4& a, const int4& b )
{
	return make_int4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HOST_DEVICE INLINE int4 operator-( const int4& a, const int4& b )
{
	return make_int4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

HOST_DEVICE INLINE int4 operator*( const int4& a, const int4& b )
{
	return make_int4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

HOST_DEVICE INLINE int4 operator/( const int4& a, const int4& b )
{
	return make_int4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

HOST_DEVICE INLINE int4& operator+=( int4& a, const int4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HOST_DEVICE INLINE int4& operator-=( int4& a, const int4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HOST_DEVICE INLINE int4& operator*=( int4& a, const int4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HOST_DEVICE INLINE int4& operator/=( int4& a, const int4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HOST_DEVICE INLINE int4& operator+=( int4& a, const int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HOST_DEVICE INLINE int4& operator-=( int4& a, const int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HOST_DEVICE INLINE int4& operator*=( int4& a, const int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HOST_DEVICE INLINE int4& operator/=( int4& a, const int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HOST_DEVICE INLINE int4 operator-( const int4& a ) { return make_int4( -a.x, -a.y, -a.z, -a.w ); }

HOST_DEVICE INLINE int4 operator+( const int4& a, const int c )
{
	return make_int4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HOST_DEVICE INLINE int4 operator+( const int c, const int4& a )
{
	return make_int4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HOST_DEVICE INLINE int4 operator-( const int4& a, const int c )
{
	return make_int4( a.x - c, a.y - c, a.z - c, a.w - c );
}

HOST_DEVICE INLINE int4 operator-( const int c, const int4& a )
{
	return make_int4( c - a.x, c - a.y, c - a.z, c - a.w );
}

HOST_DEVICE INLINE int4 operator*( const int4& a, const int c )
{
	return make_int4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HOST_DEVICE INLINE int4 operator*( const int c, const int4& a )
{
	return make_int4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HOST_DEVICE INLINE int4 operator/( const int4& a, const int c )
{
	return make_int4( a.x / c, a.y / c, a.z / c, a.w / c );
}

HOST_DEVICE INLINE int4 operator/( const int c, const int4& a )
{
	return make_int4( c / a.x, c / a.y, c / a.z, c / a.w );
}


HOST_DEVICE INLINE int2 max( const int2& a, const int2& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 max( const int2& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 max( const int c, const int2& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 min( const int2& a, const int2& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 min( const int2& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 min( const int c, const int2& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int3 max( const int3& a, const int3& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	int z = RT_MAX( a.z, b.z );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 max( const int3& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 max( const int c, const int3& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 min( const int3& a, const int3& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	int z = RT_MIN( a.z, b.z );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 min( const int3& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 min( const int c, const int3& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int4 max( const int4& a, const int4& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	int z = RT_MAX( a.z, b.z );
	int w = RT_MAX( a.w, b.w );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 max( const int4& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	int w = RT_MAX( a.w, c );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 max( const int c, const int4& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	int w = RT_MAX( a.w, c );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 min( const int4& a, const int4& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	int z = RT_MIN( a.z, b.z );
	int w = RT_MIN( a.w, b.w );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 min( const int4& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	int w = RT_MIN( a.w, c );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 min( const int c, const int4& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	int w = RT_MIN( a.w, c );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE float2 make_float2( const int2& a ) { return make_float2( (float)a.x, (float)a.y ); }

HOST_DEVICE INLINE float2 make_float2( const float3& a ) { return make_float2( a.x, a.y ); }

HOST_DEVICE INLINE float2 make_float2( const float4& a ) { return make_float2( a.x, a.y ); }

HOST_DEVICE INLINE float2 make_float2( const float c ) { return make_float2( c, c ); }

HOST_DEVICE INLINE float2 operator+( const float2& a, const float2& b )
{
	return make_float2( a.x + b.x, a.y + b.y );
}

HOST_DEVICE INLINE float2 operator-( const float2& a, const float2& b )
{
	return make_float2( a.x - b.x, a.y - b.y );
}

HOST_DEVICE INLINE float2 operator*( const float2& a, const float2& b )
{
	return make_float2( a.x * b.x, a.y * b.y );
}

HOST_DEVICE INLINE float2 operator/( const float2& a, const float2& b )
{
	return make_float2( a.x / b.x, a.y / b.y );
}

HOST_DEVICE INLINE float2& operator+=( float2& a, const float2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HOST_DEVICE INLINE float2& operator-=( float2& a, const float2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HOST_DEVICE INLINE float2& operator*=( float2& a, const float2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HOST_DEVICE INLINE float2& operator/=( float2& a, const float2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HOST_DEVICE INLINE float2& operator+=( float2& a, const float c )
{
	a.x += c;
	a.y += c;
	return a;
}

HOST_DEVICE INLINE float2& operator-=( float2& a, const float c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HOST_DEVICE INLINE float2& operator*=( float2& a, const float c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HOST_DEVICE INLINE float2& operator/=( float2& a, const float c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HOST_DEVICE INLINE float2 operator-( const float2& a ) { return make_float2( -a.x, -a.y ); }

HOST_DEVICE INLINE float2 operator+( const float2& a, const float c ) { return make_float2( a.x + c, a.y + c ); }

HOST_DEVICE INLINE float2 operator+( const float c, const float2& a ) { return make_float2( c + a.x, c + a.y ); }

HOST_DEVICE INLINE float2 operator-( const float2& a, const float c ) { return make_float2( a.x - c, a.y - c ); }

HOST_DEVICE INLINE float2 operator-( const float c, const float2& a ) { return make_float2( c - a.x, c - a.y ); }

HOST_DEVICE INLINE float2 operator*( const float2& a, const float c ) { return make_float2( c * a.x, c * a.y ); }

HOST_DEVICE INLINE float2 operator*( const float c, const float2& a ) { return make_float2( c * a.x, c * a.y ); }

HOST_DEVICE INLINE float2 operator/( const float2& a, const float c ) { return make_float2( a.x / c, a.y / c ); }

HOST_DEVICE INLINE float2 operator/( const float c, const float2& a ) { return make_float2( c / a.x, c / a.y ); }

HOST_DEVICE INLINE float3 make_float3( const int3& a ) { return make_float3( (float)a.x, (float)a.y, (float)a.z ); }

HOST_DEVICE INLINE float3 make_float3( const float4& a ) { return make_float3( a.x, a.y, a.z ); }

HOST_DEVICE INLINE float3 make_float3( const float2& a, const float c ) { return make_float3( a.x, a.y, c ); }

HOST_DEVICE INLINE float3 make_float3( const float c ) { return make_float3( c, c, c ); }

HOST_DEVICE INLINE float3 operator+( const float3& a, const float3& b )
{
	return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

HOST_DEVICE INLINE float3 operator-( const float3& a, const float3& b )
{
	return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

HOST_DEVICE INLINE float3 operator*( const float3& a, const float3& b )
{
	return make_float3( a.x * b.x, a.y * b.y, a.z * b.z );
}

HOST_DEVICE INLINE float3 operator/( const float3& a, const float3& b )
{
	return make_float3( a.x / b.x, a.y / b.y, a.z / b.z );
}

HOST_DEVICE INLINE float3& operator+=( float3& a, const float3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HOST_DEVICE INLINE float3& operator-=( float3& a, const float3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HOST_DEVICE INLINE float3& operator*=( float3& a, const float3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HOST_DEVICE INLINE float3& operator/=( float3& a, const float3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HOST_DEVICE INLINE float3& operator+=( float3& a, const float c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HOST_DEVICE INLINE float3& operator-=( float3& a, const float c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HOST_DEVICE INLINE float3& operator*=( float3& a, const float c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HOST_DEVICE INLINE float3& operator/=( float3& a, const float c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HOST_DEVICE INLINE float3 operator-( const float3& a ) { return make_float3( -a.x, -a.y, -a.z ); }

HOST_DEVICE INLINE float3 operator+( const float3& a, const float c )
{
	return make_float3( c + a.x, c + a.y, c + a.z );
}

HOST_DEVICE INLINE float3 operator+( const float c, const float3& a )
{
	return make_float3( c + a.x, c + a.y, c + a.z );
}

HOST_DEVICE INLINE float3 operator-( const float3& a, const float c )
{
	return make_float3( a.x - c, a.y - c, a.z - c );
}

HOST_DEVICE INLINE float3 operator-( const float c, const float3& a )
{
	return make_float3( c - a.x, c - a.y, c - a.z );
}

HOST_DEVICE INLINE float3 operator*( const float3& a, const float c )
{
	return make_float3( c * a.x, c * a.y, c * a.z );
}

HOST_DEVICE INLINE float3 operator*( const float c, const float3& a )
{
	return make_float3( c * a.x, c * a.y, c * a.z );
}

HOST_DEVICE INLINE float3 operator/( const float3& a, const float c )
{
	return make_float3( a.x / c, a.y / c, a.z / c );
}

HOST_DEVICE INLINE float3 operator/( const float c, const float3& a )
{
	return make_float3( c / a.x, c / a.y, c / a.z );
}

HOST_DEVICE INLINE float4 make_float4( const int4& a )
{
	return make_float4( (float)a.x, (float)a.y, (float)a.z, (float)a.w );
}

HOST_DEVICE INLINE float4 make_float4( const float2& a, const float c0, const float c1 )
{
	return make_float4( a.x, a.y, c0, c1 );
}

HOST_DEVICE INLINE float4 make_float4( const float3& a, const float c ) { return make_float4( a.x, a.y, a.z, c ); }

HOST_DEVICE INLINE float4 make_float4( const float c ) { return make_float4( c, c, c, c ); }

HOST_DEVICE INLINE float4 operator+( const float4& a, const float4& b )
{
	return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HOST_DEVICE INLINE float4 operator-( const float4& a, const float4& b )
{
	return make_float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

HOST_DEVICE INLINE float4 operator*( const float4& a, const float4& b )
{
	return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

HOST_DEVICE INLINE float4 operator/( const float4& a, const float4& b )
{
	return make_float4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

HOST_DEVICE INLINE float4& operator+=( float4& a, const float4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HOST_DEVICE INLINE float4& operator-=( float4& a, const float4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HOST_DEVICE INLINE float4& operator*=( float4& a, const float4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HOST_DEVICE INLINE float4& operator/=( float4& a, const float4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HOST_DEVICE INLINE float4& operator+=( float4& a, const float c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HOST_DEVICE INLINE float4& operator-=( float4& a, const float c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HOST_DEVICE INLINE float4& operator*=( float4& a, const float c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HOST_DEVICE INLINE float4& operator/=( float4& a, const float c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HOST_DEVICE INLINE float4 operator-( const float4& a ) { return make_float4( -a.x, -a.y, -a.z, -a.w ); }

HOST_DEVICE INLINE float4 operator+( const float4& a, const float c )
{
	return make_float4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HOST_DEVICE INLINE float4 operator+( const float c, const float4& a )
{
	return make_float4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HOST_DEVICE INLINE float4 operator-( const float4& a, const float c )
{
	return make_float4( a.x - c, a.y - c, a.z - c, a.w - c );
}

HOST_DEVICE INLINE float4 operator-( const float c, const float4& a )
{
	return make_float4( c - a.x, c - a.y, c - a.z, c - a.w );
}

HOST_DEVICE INLINE float4 operator*( const float4& a, const float c )
{
	return make_float4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HOST_DEVICE INLINE float4 operator*( const float c, const float4& a )
{
	return make_float4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HOST_DEVICE INLINE float4 operator/( const float4& a, const float c )
{
	return make_float4( a.x / c, a.y / c, a.z / c, a.w / c );
}

HOST_DEVICE INLINE float4 operator/( const float c, const float4& a )
{
	return make_float4( c / a.x, c / a.y, c / a.z, c / a.w );
}
#endif

HOST_DEVICE INLINE float3 cross( const float3& a, const float3& b )
{
	return make_float3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}

HOST_DEVICE INLINE float dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

HOST_DEVICE INLINE float3 normalize( const float3& a ) { return a / sqrtf( dot( a, a ) ); }

HOST_DEVICE INLINE float dot3F4( const float4& a, const float4& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

HOST_DEVICE INLINE const float4 cross3( const float3 aa, const float3 bb )
{
	return make_float4( aa.y * bb.z - aa.z * bb.y, aa.z * bb.x - aa.x * bb.z, aa.x * bb.y - aa.y * bb.x, 0 );
}

HOST_DEVICE INLINE float4 qtMul( const float4& a, const float4& b )
{
	float4 ans;
	ans = make_float4( cross( make_float3( a ), make_float3( b ) ), 0.0f );
	// ans += a.w * b + b.w * a;
	ans = ans + make_float4( a.w * b.x, a.w * b.y, a.w * b.z, a.w * b.w ) +
		  make_float4( b.w * a.x, b.w * a.y, b.w * a.z, b.w * a.w );
	ans.w = a.w * b.w - dot( make_float3( a ), make_float3( b ) );
	return ans;
}

HOST_DEVICE INLINE float4 qtInvert( const float4& q )
{
	float4 ans;
	ans	  = -q;
	ans.w = q.w;
	return ans;
}

HOST_DEVICE INLINE float3 qtRotate( const float4& q, const float3& p )
{
	float4 qp	= make_float4( p, 0.0f );
	float4 qInv = qtInvert( q );
	float4 out	= qtMul( qtMul( q, qp ), qInv );
	return make_float3( out );
}