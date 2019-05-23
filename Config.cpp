#include "Config.h"



Config::Config()
{
}


Config::~Config()
{
}

void Config::ReadConfig(std::string _filename)
{
	object_filename_.clear();
	camera_pose_.clear();
	camera_fxfycxcy_.clear();
	objects_.clear();
	vbo_.clear();
	sum_vertex_num_.clear();

	cv::FileStorage fs(_filename, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		std::cout << "Open Config.xml Failed\n";
		return;
	}
	cv::FileNode fn;
	cv::FileNodeIterator fn_it;

	fs["ObjectNum"] >> object_num_;
	fs["CameraWidth"] >> camera_width_;
	fs["CameraHeight"] >> camera_height_;

	fn = fs["ObjectFileName"];
	for (fn_it = fn.begin(); fn_it != fn.end(); fn_it++)
	{
		object_filename_.push_back((std::string)*fn_it);
	}

	fn = fs["CameraPose"];
	for (fn_it = fn.begin(); fn_it != fn.end(); fn_it++)
	{
		cv::Mat m;
		cv::read(*fn_it, m);
		camera_pose_.push_back(m);
	}

	fn = fs["CameraFxFyCxCy"];
	for (fn_it = fn.begin(); fn_it != fn.end(); fn_it++)
	{
		cv::Mat m;
		cv::read(*fn_it, m);
		camera_fxfycxcy_.push_back(m);
	}

	/*fs["CameraDepthWidth"] >> depth_width_;
	fs["CameraDepthHeight"] >> depth_height_;
	fn = fs["CameraDepthFilename"];
	cv::Mat_<float> depth_map;
	depth_map_.clear();
	for (fn_it = fn.begin(); fn_it != fn.end(); fn_it++)
	{
		//ReadDepth((std::string)*fn_it, depth_map);
		//depth_map_.push_back(depth_map);
	}*/

	fs.release();

	//read mesh
	sum_vertex_num_.resize(object_num_ + 1);
	sum_vertex_num_[0] = 0;
	for (int i = 0; i < object_num_; i++)
	{
		wObject wo;
		wo.ReadMeshPly(object_filename_[i]);
		sum_vertex_num_[i+1] = sum_vertex_num_[i] + wo.get_vertex_size();
		objects_.push_back(wo);
	}

	/*tb_fpc_.SetTopVertex(&objects_[0].get_vertex(), &camera_pose_[0], &camera_fxfycxcy_[0], camera_width_, camera_height_);
	tb_fpc_.SetBottomVertex(&objects_[3].get_vertex(), &camera_pose_[3], &camera_fxfycxcy_[3], camera_width_, camera_height_);
	tb_fpc_.Fileter(5, 0.10);
	std::vector<float4> top_vertex = tb_fpc_.get_top_filter_vertex();
	std::vector<float4> bottom_vertex = tb_fpc_.get_bottom_filter_vertex();
	WritePointsToPly<float, float, char>("filter000000.ply", top_vertex.size(),
		reinterpret_cast<float*>(top_vertex.data()), 4,
		nullptr, 3,
		nullptr, 4);*/

	vbo_.resize(sum_vertex_num_.back());
	int p = 0;
	for (int i = 0; i < object_num_; i++)
	{
		for (int j = 0; j < objects_[i].get_vertex_size(); j++)
		{
			vbo_[p].posConf.x = objects_[i].get_vertex(j)[0];
			vbo_[p].posConf.y = objects_[i].get_vertex(j)[1];
			vbo_[p].posConf.z = objects_[i].get_vertex(j)[2];
			memcpy(&vbo_[p].colorTime.x, &objects_[i].get_color(j), sizeof(cv::Vec4b));
			vbo_[p].colorTime.y = i;
			//memcpy(&rgba, &vbo[p].colorTime.x, sizeof(cv::Vec4b));
			vbo_[p].normalRad.x = objects_[i].get_normal(j)[0];
			vbo_[p].normalRad.y = objects_[i].get_normal(j)[1];
			vbo_[p].normalRad.z = objects_[i].get_normal(j)[2];
			p++;
		}
	}
}

void Config::ReadDepth(std::string _filename, cv::Mat_<float> &_depth_map)
{
	_depth_map.create(depth_height_, depth_width_);
	std::ifstream in_file;
	in_file.open(_filename, std::ios::in | std::ios::binary);
	//cv::Mat_<int> depth(depth_height_, depth_width_);
	std::vector<unsigned int> data(depth_width_*depth_height_);
	char *szBuf = new char[depth_width_*depth_height_ * sizeof(unsigned int)];
	in_file.read(szBuf, depth_width_*depth_height_ * sizeof(unsigned int));
	memcpy(data.data(), szBuf, depth_width_*depth_height_ * sizeof(unsigned int));
	for (int i = 0; i < depth_width_*depth_height_; i++)
	{
		_depth_map(i / depth_width_, i%depth_width_) = data[i];
	}
	
	//depth.convertTo(_depth_map, CV_32FC1);
	delete szBuf;
}