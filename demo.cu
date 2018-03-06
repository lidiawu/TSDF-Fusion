
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "TSDFVolume.h"
#include "MarchingCubes.h"



using namespace cv;
using namespace std;


// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
int main(int argc, char * argv[]) {

  // Location of camera intrinsic file
  std::string cam_K_file = "E:\\GrUVi\\wuqiw\\tsdf-mc\\data\\camera-intrinsics.txt";

  // Location of folder containing RGB-D frames and camera pose files
  std::string data_path = "E:\\GrUVi\\wuqiw\\tsdf-mc\\data\\rgbd-frames";
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
  TSDFVolume volume(500,500,500,-1.5f,-1.5f, 0.5f, 0.006f);
  

  
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

  

   std::cout << "Fusing: " << depth_im_file << std::endl;

	volume.Integrate(depth_im,cam_K,cam2base);

  }

  // Compute surface points from TSDF voxel grid and save to point cloud .ply file
  std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
  vector<float3> vertices ;
  vector<int3> triangles;
  extract_surface(volume, vertices, triangles);
  write_to_ply("tsdf_test.ply",vertices,triangles);

  
  delete depth_im;
  return 0;
}


