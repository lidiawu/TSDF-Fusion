
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "utils.hpp"
#include <Eigen/Dense>

#include "MC_edge_table.cu"
#include "MC_triangle_table.cu"

#define CUDART_NAN_F            __int_as_float(0x7fffffff)

using namespace cv;
using namespace std;

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(float * cam_K, float * cam2base, float * depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight) {

  int pt_grid_z = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

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
    if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
      continue;

    float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

    if (depth_val <= 0 || depth_val > 6)
      continue;

    float diff = depth_val - pt_cam_z;

    if (diff <= -trunc_margin)
      continue;

    // Integrate
    int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + 1.0f;
    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
  }
}

__device__
float3 compute_intersection_for_edge( int edge_index,
                                      const float voxel_values[8],
                                      const float3 cube_vertices[8] ) {
	// The expectation here is that 
	// : Cube vertices is populated with real world coordinates of the cube being marched
	// : voxel values are the corresponding weights of the vertices
	// : edge index is an edge to be intersected
	// : The vertex at one end of that edge has a negative weigt and at the otehr, a positive weight
	// : The intersection should be at the zero crossing

	// Check assumptions
	int v1_index = EDGE_VERTICES[edge_index][0];
	int v2_index = EDGE_VERTICES[edge_index][1];

	float3 start_vertex = cube_vertices[v1_index];
	float start_weight = voxel_values[v1_index];

	float3 end_vertex   = cube_vertices[v2_index];
	float end_weight = voxel_values[v2_index];

	if (  (start_weight > 0 ) &&  (end_weight <  0 ) ) {
		// Swap start and end
		float3 temp3 = start_vertex;
		start_vertex = end_vertex;
		end_vertex = temp3;

		float temp = start_weight;
		start_weight = end_weight;
		end_weight = temp;
	} else if ( ( start_weight * end_weight ) > 0 ) {
		printf( "Intersected edge expected to have differenlty signed weights at each end\n");
		asm("trap;");
	}

	float ratio = ( 0 - start_weight ) / ( end_weight - start_weight);


	// Work out where this lies
	float3 edge = make_float3(end_vertex.x - start_vertex.x, end_vertex.y -start_vertex.y, end_vertex.z - start_vertex.z);
	float3 delta = make_float3(ratio * edge.x, ratio * edge.y, ratio * edge.z);
	float3 intersection = make_float3(start_vertex.x + delta.x, start_vertex.y + delta.y, start_vertex.z + delta.z); 

	return intersection;
}

/**
 * @param cube_index the value descrbining which cube this is
 * @param voxel_values The distance values from the TSDF corresponding to the 8 voxels forming this cube
 * @param cube_vertices The vertex coordinates in real space of the cube being considered
 * @param intersects Populated by this function, the point on each of the 12 edges where an intersection occurs
 * There are a maximum of 12. Non-intersected edges are skipped that is, if only edge 12 is intersected then intersects[11]
 * will have a value the other values will be NaN
 * @return The number of intersects found
 */
__device__
int compute_edge_intersects( uint8_t cube_index,
                             const float voxel_values[8],
                             const float3 cube_vertices[8],
                             float3 intersects[12]) {
	// Get the edges impacted
	int num_edges_impacted = 0;
	if ( ( cube_index != 0x00) && ( cube_index != 0xFF ) ) {
		uint16_t intersected_edge_flags = EDGE_TABLE[cube_index];
		uint16_t mask = 0x01;
		for ( int i = 0; i < 12; i++ ) {
			if ( ( intersected_edge_flags & mask ) > 0 ) {

				intersects[i] = compute_intersection_for_edge( i, voxel_values, cube_vertices);

				num_edges_impacted++;
			} else {
				intersects[i].x = CUDART_NAN_F;
				intersects[i].y = CUDART_NAN_F;
				intersects[i].z = CUDART_NAN_F;
			}
			mask = mask << 1;
		}
	}
	return num_edges_impacted;
}

__device__
void compute_cube_vertices(int x, int y, int z, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float3 cube_vertices[8]){
	  float pt_base_x = voxel_grid_origin_x + (float) x * voxel_size;
      float pt_base_y = voxel_grid_origin_y + (float) y * voxel_size;
      float pt_base_z = voxel_grid_origin_z + (float) z * voxel_size;
	  cube_vertices[0] = make_float3(pt_base_x,pt_base_y,pt_base_z);
	  cube_vertices[1] = make_float3(pt_base_x,pt_base_y,pt_base_z + voxel_size);
	  cube_vertices[2] = make_float3(pt_base_x,pt_base_y+ voxel_size,pt_base_z + voxel_size);
	  cube_vertices[3] = make_float3(pt_base_x,pt_base_y+ voxel_size,pt_base_z);
	  cube_vertices[4] = make_float3(pt_base_x + voxel_size,pt_base_y,pt_base_z);
	  cube_vertices[5] = make_float3(pt_base_x + voxel_size,pt_base_y,pt_base_z + voxel_size);
      cube_vertices[6] = make_float3(pt_base_x + voxel_size,pt_base_y + voxel_size,pt_base_z + voxel_size);
	  cube_vertices[7] = make_float3(pt_base_x + voxel_size,pt_base_y + voxel_size,pt_base_z);
	  
}
/**
 * @param values An array of eight values from the TSDF indexed per the edge_table include file
 * return an 8 bit value representing the cube type (where a bit is set if the value in tha location
 * is negative i.e. behind the surface otherwise it's clear
 */
__device__
uint8_t cube_type_for_values( const float values[8] ) {
	uint8_t mask = 0x01;
	uint8_t cube_type = 0x00;
	for ( int i = 0; i < 8; i++ ) {
		if (values[i] < 0) {
			cube_type = cube_type | mask;
		}
		mask = mask << 1;
	}
	return cube_type;
}

/**
 * Compute the triangles for two planes of data
 * @param tsdf_values_layer_1 The first layer of values
 * @param tsdf_values_layer_2 The second layer of values
 * @param dim_x,y,z dimemsion of TSDF
 * @param origin_x,y,z origin coordinate of TSDF
 * @param voxel_size size of voxel
 * @param vz The index of the plane being considered
 * @param vertices An array of 12 vertices per cube
 * @param triangels An array of 5 triangles per cube
 */
__global__
void mc_kernel( const float * tsdf_values_layer_1,
                const float * tsdf_values_layer_2,
                int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
				float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                float voxel_size,
                int   vz,

                // Output variables
				float3 *vertices,
                int3 *triangles ) {


	// Extract the voxel X and Y coordinates which describe the position in the layers
	// We use layer1 = z0, layer2 = z1
	int vx = threadIdx.x + (blockIdx.x * blockDim.x);
	int vy = threadIdx.y + (blockIdx.y * blockDim.y);

	// If this thread is in range (we only go up to penultimate X and Y values)
	if ( ( vx < voxel_grid_dim_x - 1 ) && ( vy < voxel_grid_dim_y - 1 ) ) {

		// Compute index of the voxel to address (used to index voxel data)
		int voxel_index =  (voxel_grid_dim_x * vy) + vx;

		// Compute cube index (ised to index output tris and verts)
		int cube_index =      ((voxel_grid_dim_x - 1) * vy) + vx;
		int vertex_index   =  cube_index * 12;
		int triangle_index =  cube_index *  5;

		// Load voxel values for the cube
		float voxel_values[8] = {
			tsdf_values_layer_1[voxel_index],							//	vx,   vy,   vz
			tsdf_values_layer_1[voxel_index + 1],						//	vx,   vy,   vz+1
			tsdf_values_layer_2[voxel_index + 1],		//	vx,   vy+1, vz+1
			tsdf_values_layer_2[voxel_index],		//	vx,   vy+1, vz
			tsdf_values_layer_1[voxel_index + voxel_grid_dim_x],						//	vx+1, vy,	vz
			tsdf_values_layer_1[voxel_index + voxel_grid_dim_x+ 1],						//	vx+1, vy, 	vz+1
			tsdf_values_layer_2[voxel_index + voxel_grid_dim_x+ 1],	//	vx+1, vy+1, vz+1
			tsdf_values_layer_2[voxel_index +voxel_grid_dim_x],	//	vx+1, vy+1, vz
		};

		// Compute the cube type
		uint8_t cube_type = cube_type_for_values( voxel_values );

		// If it's a non-trivial cube_type, process it
		if ( ( cube_type != 0 ) && ( cube_type != 0xFF ) ) {

			// Compuyte the coordinates of the vertices of the cube
			float3 cube_vertices[8];
			compute_cube_vertices(vx, vy, vz, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, cube_vertices);

			// Compute intersects (up to 12 per cube)
			float3 	intersects[12];
			compute_edge_intersects( cube_type, voxel_values, cube_vertices, intersects);

			// Copy these back into the return vaue array at the appropriate point for this thread
			for ( int i = 0; i < 12; i++ ) {
				vertices[vertex_index + i] = intersects[i];
			}

			// These intersects form triangles in line with the MC triangle table
			// We compute all five triangles because we're writing to a fixed size array
			// and we need to ensure that every thread knows where to write.
			int i = 0;
			for ( int t = 0; t < 5; t++ ) {
				triangles[triangle_index + t].x = TRIANGLE_TABLE[cube_type][i++];
				triangles[triangle_index + t].y = TRIANGLE_TABLE[cube_type][i++];
				triangles[triangle_index + t].z = TRIANGLE_TABLE[cube_type][i++];
			}
		} else {
			// Set all triangle to have -ve indices
			for ( int t = 0; t < 5; t++ ) {
				triangles[triangle_index + t].x = -1;
				triangles[triangle_index + t].y = -1;
				triangles[triangle_index + t].z = -1;
			}
		}
	}
}

__host__
void process_kernel_output( int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
							const float3          * h_vertices,
                            const int3            * h_triangles,
                            vector<float3>&    vertices,
                            vector<int3>&      triangles) {
	using namespace Eigen;

	// For all but last row of voxels
	int cube_index = 0;
	for ( int y = 0; y < voxel_grid_dim_y - 1; y++ ) {

		// For all but last column of voxels
		for ( int x = 0; x < voxel_grid_dim_x - 1; x++ ) {

			// get pointers to vertices and triangles for this voxel
			const float3* verts = h_vertices  + ( cube_index * 12 );
			const int3* tris  = h_triangles + ( cube_index * 5  );

			// Iterate until we have 5 triangles or there are none left
			int tri_index = 0;
			while ( ( tri_index < 5) && ( tris[tri_index].x != -1 ) ) {

				// Get the raw vertex IDs
				int tri_vertices[3];
				tri_vertices[0] = tris[tri_index].x;
				tri_vertices[1] = tris[tri_index].y;
				tri_vertices[2] = tris[tri_index].z;


				// Remap triangle vertex indices to global indices
				int remap[] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
				for ( int tv = 0; tv < 3; tv++ ) {

					int vid = tri_vertices[tv];
					int vertexid = remap[ vid ];
					if ( vertexid == -1 ) {
						// This vertex hasnt been remapped (ie stored) yet
						vertices.push_back( verts[ vid ] );

						// Get the new ID
						vertexid = vertices.size() - 1;

						// And enter in remap table
						remap[ vid ] = vertexid;
					}
					tri_vertices[tv] = vertexid;
				}

				// Store the triangle
				int3 triangle = make_int3(tri_vertices[0], tri_vertices[1], tri_vertices[2]);
				triangles.push_back( triangle );

				tri_index++;
			}
			cube_index++;
		}
	}
	
}

__host__
void extract_surface(int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,  float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,float voxel_size, float* voxel_grid_TSDF, float* voxel_grid_weight, vector<float3>& vertices, vector<int3>& triangles){
	using namespace Eigen;

	// Allocate storage on device and locally
	// Fail if not possible
	size_t num_cubes_per_layer = (voxel_grid_dim_x - 1) * (voxel_grid_dim_y - 1);

	// Device vertices
	float3* d_vertices;
	size_t num_vertices =  num_cubes_per_layer * 12;
	cudaError_t err = cudaMalloc( &d_vertices, num_vertices * sizeof( float3 ) );
	if ( err != cudaSuccess ) {
		cout << "Couldn't allocate device memory for vertices" << endl;
		throw std::bad_alloc( );
	}

	// Device triangles
	int3* d_triangles;
	size_t num_triangles  = num_cubes_per_layer * 5;
	err = cudaMalloc( &d_triangles, num_triangles * sizeof(int3) );
	if ( err != cudaSuccess ) {
		cudaFree( d_vertices );
		cout << "Couldn't allocate device memory for triangles" << endl;
		throw std::bad_alloc( );
	}

	// Host vertices
	float3* h_vertices = new float3[ num_vertices ];
	if ( !h_vertices ) {
		cudaFree( d_vertices);
		cudaFree( d_triangles);
		cout << "Couldn't allocate host memory for vertices" << endl;
		throw std::bad_alloc( );
	}

	// Host triangles
	int3 *h_triangles = new int3 [  num_triangles ];
	if ( !h_triangles ) {
		delete [] h_vertices;
		cudaFree( d_vertices);
		cudaFree( d_triangles);
		cout << "Couldn't allocate host memory for triangles" << endl;
		throw std::bad_alloc( );
	}
	


	// Now iterate over each slice
	size_t layer_size =  voxel_grid_dim_x * voxel_grid_dim_y;
	for ( int vz = 0; vz < voxel_grid_dim_z - 1; vz++ ) {

		// Set up for layer
		const float * layer1_data = &(voxel_grid_TSDF[vz * layer_size] );
		const float * layer2_data = &(voxel_grid_TSDF[(vz + 1) * layer_size] );


		// invoke the kernel
		dim3 block( 16, 16, 1 );
		dim3 grid ((voxel_grid_dim_x + block.x - 1) /block.x, (voxel_grid_dim_y + block.y - 1)/block.y, 1 );
		mc_kernel <<< grid, block >>>( layer1_data, layer2_data,
		                               voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
									   voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
									   voxel_size,
		                               vz, d_vertices, d_triangles );

		err = cudaDeviceSynchronize( );
		//check_cuda_error( "device synchronize failed " , err);

		// Copy the device vertices and triangles back to host
		err = cudaMemcpy( h_vertices, d_vertices, num_vertices * sizeof( float3 ), cudaMemcpyDeviceToHost);
		//check_cuda_error( "Copy of vertex data fom device failed " , err);

		err = cudaMemcpy( h_triangles, d_triangles, num_triangles * sizeof( int3 ), cudaMemcpyDeviceToHost);
		//check_cuda_error( "Copy of triangle data from device failed " , err);

		// All through all the triangles and vertices and add them to master lists
		process_kernel_output(voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z, h_vertices, h_triangles, vertices, triangles);

	}

	// Free memory and done
	err = cudaFree( d_vertices);
	//check_cuda_error( "extract_vertices: Free device vertex memory failed " , err);
	err = cudaFree( d_triangles);
	//check_cuda_error( "extract_vertices: Free device triangle memory failed " , err);

	delete [] h_vertices;
	delete [] h_triangles;
}

// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
int main(int argc, char * argv[]) {

  // Location of camera intrinsic file
  std::string cam_K_file = "E:\\GrUVi\\wuqiw\\tsdf-fusion-master\\tsdf-fusion-master\\data\\camera-intrinsics.txt";

  // Location of folder containing RGB-D frames and camera pose files
  std::string data_path = "E:\\GrUVi\\wuqiw\\tsdf-fusion-master\\tsdf-fusion-master\\data\\rgbd-frames";
  int base_frame_idx = 150;
  int first_frame_idx = 150;
  float num_frames = 50;

  float cam_K[3 * 3];
  float base2world[4 * 4];
  float cam2base[4 * 4];
  float cam2world[4 * 4];
  int im_width = 640;
  int im_height = 480;
  float* depth_im = new float[480 * 640];

  // Voxel grid parameters (change these to change voxel grid resolution, etc.)
  float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
  float voxel_grid_origin_y = -1.5f;
  float voxel_grid_origin_z = 0.5f;
  float voxel_size = 0.006f;
  float trunc_margin = voxel_size * 5;
  int voxel_grid_dim_x = 500;
  int voxel_grid_dim_y = 500;
  int voxel_grid_dim_z = 500;

  // Manual parameters
  if (argc > 1) {
    cam_K_file = argv[1];
    data_path = argv[2];
    base_frame_idx = atoi(argv[3]);
    first_frame_idx = atoi(argv[4]);
    num_frames = atof(argv[5]);
    voxel_grid_origin_x = atof(argv[6]);
    voxel_grid_origin_y = atof(argv[7]);
    voxel_grid_origin_z = atof(argv[8]);
    voxel_size = atof(argv[9]);
    trunc_margin = atof(argv[10]);
  }

  // Read camera intrinsics
  std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
  std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);

  // Read base frame camera pose
  std::ostringstream base_frame_prefix;
  base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
  std::string base2world_file = data_path + "\\frame-" + base_frame_prefix.str() + ".pose.txt";
  std::vector<float> base2world_vec = LoadMatrixFromFile(base2world_file, 4, 4);
  std::copy(base2world_vec.begin(), base2world_vec.end(), base2world);

  // Invert base frame camera pose to get world-to-base frame transform 
  float base2world_inv[16] = {0};
  invert_matrix(base2world, base2world_inv);

  // Initialize voxel grid
  float * voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  float * voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    voxel_grid_TSDF[i] = 1.0f;
  memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  // Load variables to GPU memory
  float * gpu_voxel_grid_TSDF;
  float * gpu_voxel_grid_weight;
  cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  checkCUDA(__LINE__, cudaGetLastError());
  cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  checkCUDA(__LINE__, cudaGetLastError());
  float * gpu_cam_K;
  float * gpu_cam2base;
  float * gpu_depth_im;
  cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
  cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
  cudaMalloc(&gpu_depth_im, im_height * im_width * sizeof(float));
  checkCUDA(__LINE__, cudaGetLastError());

  // Loop through each depth frame and integrate TSDF voxel grid
  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; ++frame_idx) {

    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;

    // // Read current frame depth
    std::string depth_im_file = data_path + "\\frame-" + curr_frame_prefix.str() + ".depth.png";
    ReadDepth(depth_im_file, im_height, im_width, depth_im);

    // Read base frame camera pose
    std::string cam2world_file = data_path + "\\frame-" + curr_frame_prefix.str() + ".pose.txt";
    std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
    std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

    // Compute relative camera pose (camera-to-base frame)
    multiply_matrix(base2world_inv, cam2world, cam2base);

    cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_depth_im, depth_im, im_height * im_width * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    std::cout << "Fusing: " << depth_im_file << std::endl;

    Integrate<<< voxel_grid_dim_z, voxel_grid_dim_y >>>(gpu_cam_K, gpu_cam2base, gpu_depth_im, im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                                         voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
                                                         gpu_voxel_grid_TSDF, gpu_voxel_grid_weight);
  }

  // Load TSDF voxel grid from GPU to CPU memory
  cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  checkCUDA(__LINE__, cudaGetLastError());

  // Compute surface points from TSDF voxel grid and save to point cloud .ply file
  std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
  SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z, 
                                  voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                  voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

  // Save TSDF voxel grid and its parameters to disk as binary file (float array)
  std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
  std::string voxel_grid_saveto_path = "E:\\GrUVi\\wuqiw\\tsdf-fusion-master\\tsdf-fusion-master\\data\\tsdf.bin";
  std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
  float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
  float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
  float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
  outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
  outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
  outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
  outFile.write((char*)&voxel_size, sizeof(float));
  outFile.write((char*)&trunc_margin, sizeof(float));
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
  outFile.close();
  
  delete depth_im;
  return 0;
}


