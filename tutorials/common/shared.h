typedef hiprtFloat4 Quaternion;

typedef struct Material
{
	hiprtFloat4 m_diffuse;
	hiprtFloat4 m_emission;
	hiprtFloat4 m_params; // m_params.x - is light
} Material_t;

typedef struct Light
{
	hiprtFloat4 m_le;
	hiprtFloat3 m_lv0;
	hiprtFloat3 m_lv1;
	hiprtFloat3 m_lv2;
	hiprtFloat3 pad;
}Light_t;

typedef struct Camera
{
	hiprtFloat4 m_translation; // eye/rayorigin
	Quaternion m_quat;
	float	   m_fov;
	float	   m_near;
	float	   m_far;
	float	   padd;
}Camera;