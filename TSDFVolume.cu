#include "TSDFVolume.h"
#include <iostream>

using namespace std;

__host__
void TSDFVolume::TSDFVolume(){int x, int y, int z, float ox, float oy, float oz, float size}{
		dim_x = x;
		dim_y = y;
		dim_z = z;

		origin_x = ox;
		origin_y = oy;
		origin_z = oz;

		voxel_size = size;

		cudaError_t err;
		size_t data_size = dim_x * dim_y * dim_z * sizeof( float );

        err = cudaMalloc( &m_distances, data_size );
		if(err != cudaSuccess)
			cout <<  "Couldn't allocate space for distance data for TSDF" << endl;

        err = cudaMalloc( &m_weights, data_size );
		if (err != cudaSuccess)
			cout << "Couldn't allocate space for weight data for TSDF" << endl;

        err = cudaMalloc( &m_colours,  dim_x * dim_y * dim_z * sizeof( uchar3 ) );
		if(err != cudaSuccess)
			cout << "Couldn't allocate space for colour data for TSDF" << endl;
       
	}