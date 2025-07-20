#pragma once

struct vec3
{
	float x;
	float y;
	float z;

	vec3 operator + (const vec3& other) const
	{
		return { x + other.x, y + other.y, z + other.z };
	}

	vec3 operator - (const vec3& other) const
	{
		return { x - other.x, y - other.y, z - other.z };
	}

	vec3 operator * (const vec3& other) const
	{
		return { ((y * other.z) - (z * other.y)), ((z * other.x) - (x * other.z)), ((x * other.y) - (y * other.x)) };
	}

	vec3 operator * (const float& val) const
	{
		return { x * val, y * val, z * val };
	}

	vec3& operator += (const vec3& other)
	{
		x += other.x; y += other.y; z += other.z;
		return *this;
	}

	vec3& operator -= (const vec3& other)
	{
		x -= other.x; y -= other.y; z -= other.z;
		return *this;
	}

	vec3& operator *= (const float& val)
	{
		x = x * val; y = y * val; z = z * val;
		return *this;
	}

	vec3& operator = (const float& val)
	{
		x = val; y = val; z = val;
		return *this;
	}
};

inline float dot(const vec3& a, const vec3& b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

inline float fastInvSqrt(float number)
{
	int i;
	float y;

	y = number;
	i = *(int*)&y;
	i = 0x5f375a86 - (i >> 1);
	y = *(float*)&i;
	y = y * (1.5f - (number * 0.5f * y * y));

	return y;
}

inline void normalize(vec3& vec)
{
	float lengthSq = (vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z);
	float invSqrt = fastInvSqrt(lengthSq);
	vec *= invSqrt;
}

