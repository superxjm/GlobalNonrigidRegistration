#pragma once
#include "Display/DrawObject.h"
#include "Helpers/xUtils.h"
#include <cuda_gl_interop.h>
#include <xDeformation.h>
#include <InputData.h>

/*namespace ConstantColor
{
	cv::Vec4b constantColor[10] = { cv::Vec4b(255, 128,128,255), cv::Vec4b(255, 255,128,255),  cv::Vec4b(128, 255,128,255), cv::Vec4b(0, 128,255,255),cv::Vec4b(128, 128,192,255),
		cv::Vec4b(128, 64,64,255), cv::Vec4b(0, 128,128,255), cv::Vec4b(128, 255,64,255), cv::Vec4b(128, 128,128,255), cv::Vec4b(192, 192,192,255) };
};*/

class Register
{
public:
	Register();
	~Register();

	void CreateDeformation(int _vbo_id, std::vector<int> _sum_vertex_num);

	//void RegisterRun(int _frag_id, std::vector<int> _sum_vertex_num);
	void FindCorrespondencePoints(std::vector<cv::Mat_<float>> &_camera_pose,
		std::vector<cv::Mat_<float>> &_camera_fxfycxcy,
		int _method, float _dist_thresh1, float _dist_thresh2, float _angle_thresh,
		int _width, int _height);
	void RegisterRun(int _iter_num, std::vector<int> _sum_vertex_num, 
		std::vector<cv::Mat_<float>> &_camera_pose, 
		std::vector<cv::Mat_<float>> &_camera_fxfycxcy,
		int _method, float _dist_thresh1, float _dist_thresh2, float _angle_thresh,
		int _width, int _height);
	void CalcEachFragColor();

	void CopyData();

	std::vector<cv::Vec4f>& get_nodes();
	std::vector<cv::Vec4f>& get_deformed();
	std::vector<cv::Vec4b>& get_deformed_color();
	std::vector<cv::Vec4f>& get_deformed_normal();
	std::vector<cv::Vec4f>& get_source();
	std::vector<cv::Vec4b>& get_source_color();

	std::vector<cv::Vec4f>& get_source2deformed_vertex();
	std::vector<int>&       get_source2deformed_index();
	std::vector<cv::Vec4b>& get_source2deformed_color();

	std::vector<cv::Vec4f>& get_corr_vertex();
	std::vector<cv::Vec4b>& get_corr_color();
	std::vector<int>&       get_corr_index();

	std::vector<cv::Vec4f>& get_deformed_sample_vertex();
	std::vector<cv::Vec4b>& get_deformed_sample_color();
	std::vector<cv::Vec4f>& get_source_sample_vertex();
	std::vector<cv::Vec4b>& get_source_sample_color();

private:
	xDeformation *deform_;


	int vbo_id_;
	int frag_id_;
	VBOType *vbo_;
	struct cudaGraphicsResource* vbo_CUDA_;
	int frag_num_;
	std::vector<int> sum_vertex_num_;

	float imgScale, dispScal;
	int depthWidth, depthHeight, colorWidth, colorHeight;
	float fx, fy, cx, cy;
	cv::Mat          color_;
	cv::Mat          full_color_;
	cv::Mat_<uchar>  gray_;
	cv::cuda::GpuMat d_gray_;

	std::vector<cv::Vec4f> h_nodes_;
	std::vector<cv::Vec4f> h_deformed_;
	std::vector<cv::Vec4f> h_deformed_normal_;
	std::vector<cv::Vec4f> h_source_;
	std::vector<cv::Vec4b> h_color_;

	std::vector<cv::Vec4f> h_deformed_sample_vertex_;
	std::vector<cv::Vec4b> h_deformed_sample_color_;
	std::vector<cv::Vec4f> h_source_sample_vertex_;
	std::vector<cv::Vec4b> h_source_sample_color_;

	std::vector<cv::Vec4f> h_source2deformed_vertex_;
	std::vector<int>   h_source2deformed_index_;
	std::vector<cv::Vec4b> h_source2deformed_color_;

	std::vector<cv::Vec4f> h_corr_vertex_;
	std::vector<cv::Vec4b> h_corr_color_;
	std::vector<int> h_corr_index_;
};

