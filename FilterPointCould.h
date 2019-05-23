#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "wUtils.h"

class TBFilterPointCould
{
public:
	TBFilterPointCould();
	~TBFilterPointCould();

	void SetTopVertex(std::vector<cv::Vec4f> *_vertex, cv::Mat_<float> *_camera_pose, cv::Mat_<float> *_camera_fxfycxcy, int _width, int _height);
	void SetBottomVertex(std::vector<cv::Vec4f> *_vertex, cv::Mat_<float> *_camera_pose, cv::Mat_<float> *_camera_fxfycxcy, int _width, int _height);
	void Fileter(int _edge_range, int _dis_thresh);
	
	std::vector<float4>& get_top_filter_vertex()
	{
		h_top_filter_vertex_.resize(d_top_filter_vertex_.size());
		checkCudaErrors(cudaMemcpy(h_top_filter_vertex_.data(), RAW_PTR(d_top_filter_vertex_), h_top_filter_vertex_.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	
		return h_top_filter_vertex_;
	}

	std::vector<float4>& get_bottom_filter_vertex()
	{
		h_bottom_filter_vertex_.resize(d_bottom_filter_vertex_.size());
		checkCudaErrors(cudaMemcpy(h_bottom_filter_vertex_.data(), RAW_PTR(d_bottom_filter_vertex_), h_bottom_filter_vertex_.size() * sizeof(float4), cudaMemcpyDeviceToHost));

		return h_bottom_filter_vertex_;
	}
	
	

private:
    
	int                            trows_, tcols_;
	thrust::device_vector<float4>  d_top_vertex_;
	thrust::device_vector<float4>  d_top_camera_pose_inv_t_;
	float4                         h_top_camera_fxfycxcy_;
	thrust::device_vector<float>   d_top_depth_map_;
	thrust::device_vector<int>     d_top_index_map_;
	thrust::device_vector<float4>  d_top_filter_vertex_;
	thrust::device_vector<int>     d_top_filter_vertex_num_;
	thrust::device_vector<float>   d_top_nearest_dis_;
	thrust::device_vector<int>     d_top_nearest_idx_;
	std::vector<float4>            h_top_filter_vertex_;

	int                            brows_, bcols_;
	thrust::device_vector<float4>  d_bottom_vertex_;
	thrust::device_vector<float4>  d_bottom_camera_pose_inv_t_;
	float4                         h_bottom_camera_fxfycxcy_;
	thrust::device_vector<float>   d_bottom_depth_map_;
	thrust::device_vector<int>     d_bottom_index_map_;
	thrust::device_vector<float4>  d_bottom_filter_vertex_;
	thrust::device_vector<int>     d_bottom_filter_vertex_num_;
	std::vector<float4>            h_bottom_filter_vertex_;

};

