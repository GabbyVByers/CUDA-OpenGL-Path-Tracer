#pragma once

#include <math.h>

struct vec3
{
	float x;
	float y;
	float z;

	vec3 operator + (const vec3& vec) const
	{
		return { x + vec.x, y + vec.y, z + vec.z };
	}

	vec3 operator - (const vec3& vec) const
	{
		return { x - vec.x, y - vec.y, z - vec.z };
	}

	vec3 operator * (const vec3& vec) const
	{
		return { ((y * vec.z) - (z * vec.y)), ((z * vec.x) - (x * vec.z)), ((x * vec.y) - (y * vec.x)) };
	}

	vec3 operator * (const float& value) const
	{
		return { x * value, y * value, z * value };
	}

	vec3 operator / (const float& value) const
	{
		return { x / value, y / value, z / value };
	}

	vec3& operator += (const vec3& vec)
	{
		x += vec.x; y += vec.y; z += vec.z;
		return *this;
	}

	vec3& operator -= (const vec3& vec)
	{
		x -= vec.x; y -= vec.y; z -= vec.z;
		return *this;
	}

	vec3& operator *= (const float& value)
	{
		x = x * value; y = y * value; z = z * value;
		return *this;
	}

	vec3& operator /= (const float& value)
	{
		x = x / value; y = y / value; z = z / value;
		return *this;
	}

	vec3& operator = (const float& value)
	{
		x = value; y = value; z = value;
		return *this;
	}
};

inline float dot(const vec3& a, const vec3& b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

inline void normalize(vec3& vec)
{
	float lengthSq = (vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z);
	vec /= sqrt(lengthSq);
}

