#pragma once
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

extern void wReadMeshObj(std::string _fileName, std::vector<cv::Vec4f> _vertex, std::vector<cv::Vec4b> _color, std::vector<cv::Vec3f> _normal);
extern void wReadMeshPly(std::string _filename, std::vector<cv::Vec4f> _vertex, std::vector<cv::Vec4b> _color, std::vector<cv::Vec3f> _normal);

template<class VertexType, class NormalType, class ColorType>
void WritePointsToPly(std::string _file_name, int _vertex_num,
	VertexType *_vertex, int _vertex_component_num,
	NormalType *_normal, int _normal_component_num,
	ColorType *_color, int _color_component_num)
{
	if (_vertex == nullptr) return;

	std::ofstream out_file(_file_name);
	if (!out_file.is_open()) return;
	out_file << "ply\n";
	out_file << "format ascii 1.0\ncomment VCGLIB generated\n";
	out_file << "element vertex " << _vertex_num << "\n";
	out_file << "property float x\nproperty float y\nproperty float z\n";
	if (_normal != nullptr)
	{
		out_file << "property float nx\nproperty float ny\nproperty float nz\n";
	}
	if (_color != nullptr)
	{
		out_file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
	}
	out_file << "element face " << 0 << "\n";
	out_file << "property list uchar int vertex_indices\n";
	out_file << "end_header\n";

	for (int i = 0; i < _vertex_num; i++)
	{
		out_file << _vertex[i*_vertex_component_num] << " " << _vertex[i*_vertex_component_num + 1] << " " << _vertex[i*_vertex_component_num + 2] << " ";
		if (_normal != nullptr)
			out_file << _normal[i*_normal_component_num] << " " << _normal[i*_normal_component_num + 1] << " " << _normal[i*_normal_component_num + 2] << " ";
		if (_color != nullptr)
			out_file << (int)_color[i*_color_component_num] << " " << (int)_color[i*_color_component_num + 1] << " " << (int)_color[i*_color_component_num + 2];
		out_file << "\n";
	}

	out_file.clear();
	out_file.close();
}

template<class VertexType, class NormalType>
void WritePointsToPly(std::string _file_name, int _vertex_num,
	VertexType *_vertex, int _vertex_component_num,
	NormalType *_normal, int _normal_component_num,
	float *_color, int _color_component_num)
{
	if (_vertex == nullptr) return;

	std::ofstream out_file(_file_name);
	if (!out_file.is_open()) return;
	out_file << "ply\n";
	out_file << "format ascii 1.0\ncomment VCGLIB generated\n";
	out_file << "element vertex " << _vertex_num << "\n";
	out_file << "property float x\nproperty float y\nproperty float z\n";
	if (_normal != nullptr)
	{
		out_file << "property float nx\nproperty float ny\nproperty float nz\n";
	}
	if (_color != nullptr)
	{
		out_file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
	}
	out_file << "element face " << 0 << "\n";
	out_file << "property list uchar int vertex_indices\n";
	out_file << "end_header\n";

	for (int i = 0; i < _vertex_num; i++)
	{
		out_file << _vertex[i*_vertex_component_num] << " " << _vertex[i*_vertex_component_num + 1] << " " << _vertex[i*_vertex_component_num + 2] << " ";
		if (_normal != nullptr)
			out_file << _normal[i*_normal_component_num] << " " << _normal[i*_normal_component_num + 1] << " " << _normal[i*_normal_component_num + 2] << " ";
		if (_color != nullptr)
			out_file << (int)(_color[i*_color_component_num] * 255) << " " << (int)(_color[i*_color_component_num + 1] * 255) << " " << (int)(_color[i*_color_component_num + 2] * 255);
		out_file << "\n";
	}

	out_file.clear();
	out_file.close();
}