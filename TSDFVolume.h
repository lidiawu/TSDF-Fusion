#ifndef TSDFVolume
#define TSDFVolum

#include <iostream>
#include <string>

class TSDFVolume{
public:
	TSDFVolume(int x, int y, int z, float ox, float oy, float oz, float size);

	int get_dimx(){
		return dim_x;
	}

	int get_dimy(){
		return dim_y;
	}



private:
	float origin_x;
	float origin_y;
	float origin_z;
	int dim_x;
	int dim_y;
	int dim_z;
	float voxel_size;

	// Per grid point data
    float *m_distances;
    // Colour data, RGB as 3xuchar
    uchar3 *m_colours;
    //  Confidence weight for distance and colour
    float *m_weights;
}