#ifndef CUDA_UTILS
#define CUDA_UTILS

__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

__device__ float3 operator*(const float3 &a, const float i){

	return make_float3(a.x * i, a.y * i, a.z * i);

}
#endif