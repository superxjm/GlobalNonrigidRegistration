#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "Helpers/xUtils.h"
#include "Helpers/UtilsMath.h"
//#include "Helpers/xUtilsCuda.cuh"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

void CreateLocalVertexAndNoramlFromDepthFromHostData(
	float* _h_depth_map,
	float4* _h_local_vertex_map,
	float4* _h_local_normal_map,
	int* _h_index_map,
	float4* _h_camera_fxfycxcy,
	int _width,
	int _height,
	int _camera_num);
void CreateLocalVertexAndNoramlFromDepthFromDeviceData(
	float* _d_depth_map,
	float4* _d_local_vertex_map,
	float4* _d_local_normal_map,
	int* _d_index_map,
	float4* _d_camera_fxfycxcx,
	int _width,
	int _height,
	int _camera_num);



void CreateGlobalVertexAndNormalFromLocalVertexAndNormalFromHostData(
	float4* _h_local_vertex_map,
	float4* _h_local_normal_map,
	float4* _h_global_vertex_map,
	float4* _h_global_normal_map,
	float4* _h_camera_pose_t,
	int _width,
	int _height,
	int _camera_num);
void CreateGlobalVertexAndNormalFromLocalVertexAndNormalFromDeviceData(
	float4* _d_local_vertex_map,
	float4* _d_local_normal_map,
	float4* _d_global_vertex_map,
	float4* _d_global_normal_map,
	float4* _d_camera_pose_t,
	int _width,
	int _height,
	int _camera_num);



void CudaProjectVertexToDepthImage(
	int _vertex_num,
	float4 *_d_vertex,
	float* _d_depth,
	float4 *_d_camera_pose_inv_t,
	float4 _d_camera_fxfycxcy,
	int _width, int _height);

void CudaProjectVertexToDepthImage(
	int _vertex_num,
	float4 *_d_vertex,
	float* _d_depth,
	int*   _d_index,
	float4 *_d_camera_pose_inv_t,
	float4 _d_camera_fxfycxcy,
	int _width, int _height);

void FindMatchingPointsPerspectiveFromMultiCamera(
	float4* _d_deform_vertex,
	float4* _d_target_vertex,
	float4* _d_camera_pose_inv_t,
	float4* _d_camera_fxfycxcy,
	int* _d_index_map,
	int* _d_matching_points,
	float* _d_min_matching_dist,
	int _matching_point_num,
	int _width,
	int _height,
	int _camera_num);



template <class Type>
__global__ void InitArrayKernel(Type *data, Type initdata, int _N)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;

	if (u >= _N) return;

	data[u] = initdata;
}