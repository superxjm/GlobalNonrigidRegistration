#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "wObject.h"
#include "FilterPointCould.h"
#include "MeshPointsIO.h"

class Config
{
public:
	Config();
	~Config();
	
	void ReadConfig(std::string _filename);

	std::vector<VBOType>&  get_vbo() { return vbo_; }
	std::vector<int>&      get_sum_vertex_num() { return sum_vertex_num_; }
	std::vector<cv::Mat_<float>>& get_camera_pose() { return camera_pose_; }
	std::vector<cv::Mat_<float>>& get_camera_fxfycxcy() { return camera_fxfycxcy_; }
	int                    get_camera_width() { return camera_width_; }
	int                    get_camera_height() { return camera_height_; }

private:
	void ReadDepth(std::string _filename, cv::Mat_<float> &_depth_map);

private:
	int                           object_num_;
	std::vector<std::string>      object_filename_;
	int                           camera_width_;
	int                           camera_height_;
	std::vector<cv::Mat_<float>>  camera_pose_;
	std::vector<cv::Mat_<float>>  camera_fxfycxcy_;
	int                           depth_width_;
	int                           depth_height_;
	std::vector<cv::Mat_<float>>  depth_map_;

	std::vector<int>              sum_vertex_num_;
	std::vector<wObject>          objects_;
	TBFilterPointCould            tb_fpc_;
	std::vector<VBOType>          vbo_;
};

