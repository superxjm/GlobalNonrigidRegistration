#include "MeshPointsIO.h"

void wReadMeshObj(std::string _fileName, std::vector<cv::Vec4f> _vertex, std::vector<cv::Vec4b> _color, std::vector<cv::Vec3f> _normal)
{
	_vertex.clear();
	_color.clear();
	_normal.clear();

	std::ifstream in_file(_fileName);
	if (!in_file.is_open()) return;
	char line[1024];
	float x, y, z;
	int r, g, b;
	int ix, iy, iz;
	std::string st;

	while (in_file.getline(line, sizeof(line)))
	{
		std::stringstream lineStream(line);
		lineStream >> st;
		if (st == "vn")
		{
			lineStream >> x >> y >> z;
			_normal.push_back(cv::Vec3f(x, y, z));
		}
		if (st == "v")
		{
			lineStream >> x >> y >> z >> r >> g >> b;
			_vertex.push_back(cv::Vec4f(x, y, z, 1));
			//std::cout << x << " " << y << " " << z << "\n";
			_color.push_back(cv::Vec4b(r, g, b, 1));
		}
	}
	in_file.clear();
	in_file.close();
}

void wReadMeshPly(std::string _filename, std::vector<cv::Vec4f> _vertex, std::vector<cv::Vec4b> _color, std::vector<cv::Vec3f> _normal)
{
	std::ifstream in_file(_filename);
	if (!in_file.is_open()) return;
	char line[1024];
	std::string st;
	float x, y, z;
	float nx, ny, nz;
	int r, g, b;

	while (true)
	{
		in_file.getline(line, sizeof(line));
		std::stringstream lineStream(line);
		lineStream >> st;
		if (st == "end_header") break;
	}

	while (in_file.getline(line, sizeof(line)))
	{
		std::stringstream lineStream(line);
		lineStream >> x >> y >> z >> nx >> ny >> nz >> r >> g >> b;
		_vertex.push_back(cv::Vec4f(x, y, z, 1));
		_normal.push_back(cv::Vec3f(nx, ny, nz));
		_color.push_back(cv::Vec4b(r, g, b, 1));
	}
}