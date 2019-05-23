#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <fstream>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include "Helpers/xUtils.h"

class wObject
{
public:
	wObject();
	~wObject();

	void ReadMeshObj(std::string _fileName);
	void ReadMeshPly(std::string _fileName);
	void WriteMeshObj(std::string _fileName);
	void WriteMeshPly(std::string _fileName);

	int get_vertex_size() { return vertices_.size(); }
	cv::Vec4f& get_vertex(int _i) { return vertices_[_i]; }
	cv::Vec4b& get_color(int _i) { return colors_[_i]; }
	cv::Vec3f& get_normal(int _i) { return normals_[_i]; }

	std::vector<cv::Vec4f>& get_vertex() { return vertices_; }
	std::vector<cv::Vec4b>& get_color() { return colors_; }
	std::vector<cv::Vec3f>& get_normal() { return normals_; }

	//void Transform(std::vector<float> *_m);
	//void Transform(thrust::device_vector<float> &_d_m);

private:
	std::vector<cv::Vec4f> vertices_;
	std::vector<cv::Vec4b> colors_;
	std::vector<cv::Vec3f> normals_;
	std::vector<cv::Vec3i> faceIndices_;

	//thrust::device_vector<float4> d_vertices_;
	//thrust::device_vector<float4>  d_colors_;
	//thrust::device_vector<float3>  d_normals_;

};
