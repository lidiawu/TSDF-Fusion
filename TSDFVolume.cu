#include "TSDFVolume.h"
#include <iostream>
#include <string>

using namespace std;

__host__
TSDFVolume::TSDFVolume(int x, int y, int z, float ox, float oy, float oz, float size){
		m_size.x = x;
		m_size.y = y;
		m_size.z = z;

		origin.x = ox;
		origin.y = oy;
		origin.z = oz;

		voxel_size = size;
		trunc_margin = voxel_size * 5;

		cudaError_t err;
		size_t data_size = x * y * z * sizeof( float );

        err = cudaMalloc( &m_distances, data_size );
		if(err != cudaSuccess)
			cout <<  "Couldn't allocate space for distance data for TSDF" << endl;
		float * voxel_grid_TSDF = new float[x * y * z];
		for(int i = 0; i< x*y*z;i++)
			voxel_grid_TSDF[i] = 1.0f;
		cudaMemcpy(m_distances, voxel_grid_TSDF, data_size, cudaMemcpyHostToDevice);

        err = cudaMalloc( &m_weights, data_size );
		if (err != cudaSuccess)
			cout << "Couldn't allocate space for weight data for TSDF" << endl;
		cudaMemset(m_weights,0,data_size);
       
		err = cudaMalloc(&m_deform, x * y * z * sizeof( float3 ));
		if(err != cudaSuccess)
			cout << "Couldn't allocate space for deformation data for TSDF" << endl;
		cudaMemset(m_deform, 0,x * y * z * sizeof( float3 ));
	}

TSDFVolume::~TSDFVolume() {
    std::cout << "Destroying TSDFVolume" << std::endl;
    deallocate( );
}


/**
 * Deallocate storage for this TSDF
 */
void TSDFVolume::deallocate( ) {
    // Remove existing data
    if ( m_distances ) {
        cudaFree( m_distances ) ;
        m_distances = 0;
    }
    if ( m_weights ) {
        cudaFree( m_weights );
        m_weights = 0;
    }
    if ( m_deform ) {
        cudaFree( m_deform );
        m_deform = 0;
    }
}

__global__
void Integrate_kernal(float * cam_K, float * cam2base, float * depth_im,
               dim3 size, float3 origin, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight) {

  int pt_grid_z = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  for (int pt_grid_x = 0; pt_grid_x < size.x; ++pt_grid_x) {

    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = origin.x + pt_grid_x * voxel_size;
    float pt_base_y = origin.y + pt_grid_y * voxel_size;
    float pt_base_z = origin.z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
    float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
    int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= 640 || pt_pix_y < 0 || pt_pix_y >= 480)
      continue;

    float depth_val = depth_im[pt_pix_y * 640 + pt_pix_x];

    if (depth_val <= 0 || depth_val > 6)
      continue;

    float diff = depth_val - pt_cam_z;

    if (diff <= -trunc_margin)
      continue;

    // Integrate
    int volume_idx = pt_grid_z * size.y * size.x + pt_grid_y * size.x + pt_grid_x;
    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + 1.0f;
    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
  }
}

__host__
void TSDFVolume::Integrate(float* depth_map,float* cam_K, float* cam2base){
	 float * gpu_cam_K;
     float * gpu_cam2base;
	 float * gpu_depth_im;

	 cudaMalloc(&gpu_depth_im, 480 * 640 * sizeof(float));
	 cudaMemcpy(gpu_depth_im, depth_map, 480 * 640 * sizeof(float), cudaMemcpyHostToDevice);

	 cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
	 cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

	 cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
	 cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

	 Integrate_kernal<<< m_size.z, m_size.y >>>(gpu_cam_K, gpu_cam2base, gpu_depth_im, m_size, origin, voxel_size, trunc_margin,m_distances, m_weights);
}