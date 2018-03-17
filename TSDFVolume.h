#ifndef TSDFVolume_h
#define TSDFVolume_h

#include <iostream>
#include <string>


class TSDFVolume{
public:
	TSDFVolume(int x, int y, int z, float ox, float oy, float oz, float size);

	dim3 get_size(){
		return m_size;
	}

	~TSDFVolume();

	void deallocate();

	void Integrate(float* depth_map,float* cam_K, float* cam2base);

	float* get_grid(){
		return m_distances;
	}

	float3 get_origin(){
		return origin;
	}
	
	float get_voxelsize(){
		return voxel_size;
	}

	float3* get_deform(){
		return m_deform;
	}

private:
	float3 origin;

	dim3 m_size;

	float voxel_size;
	float trunc_margin;
	// Per grid point data
    float *m_distances;
    //  Confidence weight for distance and colour
    float *m_weights;
	// translation vector for each node
	float3 *m_deform;
	float3 *grid_coord;
};
#endif