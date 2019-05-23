#include "wObject.h"

wObject::wObject()
{
}


wObject::~wObject()
{
}

void wObject::ReadMeshObj(std::string _fileName)
{
	vertices_.clear();
	colors_.clear();
	normals_.clear();
	faceIndices_.clear();

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
			normals_.push_back(cv::Vec3f(x, y, z));
		}
		if (st == "v")
		{
			lineStream >> x >> y >> z >> r >> g >> b;
			vertices_.push_back(cv::Vec4f(x, y, z, 1));
			//std::cout << x << " " << y << " " << z << "\n";
			colors_.push_back(cv::Vec4b(r, g, b, 1));
		}
		if (st == "f")
		{
			lineStream >> ix >> iy >> iz;
			faceIndices_.push_back(cv::Vec3i(ix - 1, iy - 1, iz - 1));
		}
	}
	in_file.clear();
	in_file.close();

	/*d_vertices_.resize(vertices_.size());
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_vertices_), vertices_.data(), vertices_.size() * sizeof(float4), cudaMemcpyHostToDevice));
	d_colors_.resize(colors_.size());
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_colors_), colors_.data(), colors_.size() * sizeof(float4), cudaMemcpyHostToDevice));
	d_normals_.resize(normals_.size());
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_normals_), normals_.data(), normals_.size() * sizeof(float3), cudaMemcpyHostToDevice));*/
}

void wObject::ReadMeshPly(std::string _fileName)
{
	std::ifstream in_file(_fileName);
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
		vertices_.push_back(cv::Vec4f(x, y, z, 1));
		normals_.push_back(cv::Vec3f(nx, ny, nz));
		colors_.push_back(cv::Vec4b(r, g, b, 1));
	}
}

void wObject::WriteMeshObj(std::string _fileName)
{
	std::ofstream out_file(_fileName);
	if (!out_file.is_open()) return;

	for (int i = 0; i < vertices_.size(); i++)
	{
		out_file << "vn " << normals_[i][0] << " " << normals_[i][1] << " " << normals_[i][2] << "\n";
		out_file << "v " << vertices_[i][0] << " " << vertices_[i][1] << " " << vertices_[i][2] << " "
			<< (int)colors_[i][0] << " " << (int)colors_[i][1] << " " << (int)colors_[i][2] << "\n";
	}

	for (int i = 0; i < faceIndices_.size(); i++)
	{
		out_file << "f " << faceIndices_[i][0] + 1 << " " << faceIndices_[i][1] + 1 << " " << faceIndices_[i][2] + 1 << "\n";
	}

	out_file.clear();
	out_file.close();
}

void wObject::WriteMeshPly(std::string _fileName)
{
	std::ofstream out_file(_fileName);
	if (!out_file.is_open()) return;
	out_file << "ply\n";
	out_file << "format ascii 1.0\ncomment VCGLIB generated\n";
	out_file << "element vertex " << vertices_.size() << "\n";
	out_file << "property float x\nproperty float y\nproperty float z\n";
	if (normals_.size() != 0)
	{
		out_file << "property float nx\nproperty float ny\nproperty float nz\n";
	}
	if (colors_.size() != 0)
	{
		out_file << "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n";
	}
	out_file << "element face " << faceIndices_.size() << "\n";
	out_file << "property list uchar int vertex_indices\n";
	out_file << "end_header\n";

	for (int i = 0; i < vertices_.size(); i++)
	{
		out_file << vertices_[i][0] << " " << vertices_[i][1] << " " << vertices_[i][2] << " " 
			<< normals_[i][0] << " " << normals_[i][1] << " " << normals_[i][2] << " "
			<< (int)(colors_[i][0]*255) << " " << (int)(colors_[i][1]*255) << " " << (int)(colors_[i][2]*255) <<" " << 255<< "\n";
	}

	out_file.clear();
	out_file.close();
}

/*void wObject::Transform(std::vector<float> *_m)
{
	assert(_m->size() == 16);

	thrust::device_vector<float>  d_transform_m;
	d_transform_m.resize(_m->size());
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_transform_m), _m->data(), _m->size() * sizeof(float), cudaMemcpyHostToDevice));
	Transform(d_transform_m);
}

__global__ void TransformKernel(float4* _d_vertices, float3* _d_normals, float* _transform_m, int _N)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;

	if (u >= _N) return;

	float4 vertex = _d_vertices[u];
	float4 out_vertex;
	float3 normal = _d_normals[u];
	float3 out_normal;


	out_vertex.x = _transform_m[0] * vertex.x + _transform_m[1] * vertex.y + _transform_m[2] * vertex.z + _transform_m[3] * vertex.w;
	out_vertex.y = _transform_m[4] * vertex.x + _transform_m[5] * vertex.y + _transform_m[6] * vertex.z + _transform_m[7] * vertex.w;
	out_vertex.z = _transform_m[8] * vertex.x + _transform_m[9] * vertex.y + _transform_m[10] * vertex.z + _transform_m[11] * vertex.w;
	out_vertex.w = _transform_m[12] * vertex.x + _transform_m[13] * vertex.y + _transform_m[14] * vertex.z + _transform_m[15] * vertex.w;

	out_normal.x = _transform_m[0] * normal.x + _transform_m[1] * normal.y + _transform_m[2] * normal.z;
	out_normal.y = _transform_m[4] * normal.x + _transform_m[5] * normal.y + _transform_m[6] * normal.z;
	out_normal.z = _transform_m[8] * normal.x + _transform_m[9] * normal.y + _transform_m[10] * normal.z;

	_d_vertices[u] = out_vertex;
}

void wObject::Transform(thrust::device_vector<float> &_d_m)
{
	dim3 block;
	dim3 grid;
	block.x = 256;
	grid.x = (block.x + vertices_.size() - 1) / block.x;

	TransformKernel << <grid, block >> > (RAW_PTR(d_vertices_), RAW_PTR(d_normals_), RAW_PTR(_d_m), vertices_.size());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(vertices_.data(), RAW_PTR(d_vertices_),d_vertices_.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(normals_.data(), RAW_PTR(d_normals_), d_normals_.size() * sizeof(float3), cudaMemcpyDeviceToHost));

	int a = 0;
}*/