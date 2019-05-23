#include "FilterPointCould.h"
#include "KNearestPoint.h"

TBFilterPointCould::TBFilterPointCould()
{
}


TBFilterPointCould::~TBFilterPointCould()
{
}

void TBFilterPointCould::SetTopVertex(std::vector<cv::Vec4f> *_vertex, cv::Mat_<float> *_camera_pose, cv::Mat_<float> *_camera_fxfycxcy, int _width, int _height)
{
	int vertex_num = _vertex->size();

	d_top_vertex_.resize(vertex_num);
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_top_vertex_), _vertex->data(), vertex_num * sizeof(float4), cudaMemcpyHostToDevice));

	d_top_camera_pose_inv_t_.resize(4);
	cv::Mat m = _camera_pose->inv().t();
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_top_camera_pose_inv_t_), m.data, 4 * sizeof(float4), cudaMemcpyHostToDevice));

	memcpy(&h_top_camera_fxfycxcy_, _camera_fxfycxcy->data, 4 * sizeof(float));

	trows_ = _height;
	tcols_ = _width;
	d_top_depth_map_.resize(_width*_height);
	d_top_index_map_.resize(_width*_height);
}

void TBFilterPointCould::SetBottomVertex(std::vector<cv::Vec4f> *_vertex, cv::Mat_<float> *_camera_pose, cv::Mat_<float> *_camera_fxfycxcy, int _width, int _height)
{
	int vertex_num = _vertex->size();

	d_bottom_vertex_.resize(vertex_num);
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_bottom_vertex_), _vertex->data(), vertex_num * sizeof(float4), cudaMemcpyHostToDevice));

	d_bottom_camera_pose_inv_t_.resize(4);
	cv::Mat m = _camera_pose->inv().t();
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_bottom_camera_pose_inv_t_), m.data, 4 * sizeof(float4), cudaMemcpyHostToDevice));

	memcpy(&h_bottom_camera_fxfycxcy_, _camera_fxfycxcy->data, 4 * sizeof(float));

	brows_ = _height;
	bcols_ = _width;
	d_bottom_depth_map_.resize(_width*_height);
	d_bottom_index_map_.resize(_width*_height);
}

__global__ void RepairDepthMapKernel(
	int _max_num,
	float *_d_depth,
	int _width, int _height)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;

	if (u >= _max_num) return;

	int x = u % _width;
	int y = u / _width;

	if (_d_depth[y*_width + x] != 0) return;

	int tx[4] = { 0,1,0,-1 };
	int ty[4] = { 1,0,-1,0 };
	int nx, ny;

	for (int i = 0; i < 4; i++)
	{
		nx = x + tx[i];
		ny = y + ty[i];
		if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
		{
			if (_d_depth[ny*_width + nx] != 0)
			{
				_d_depth[y*_width + x] = _d_depth[ny*_width + nx];
				//atomicExch(&depth[y*width + x], depth[ny*width + nx]);
				return;
			}
		}
	}
}

__global__ void DeleteDepthMapKernel(
	int  _max_num,
	float *_d_depth,
	int *_d_index,
	int _width, int _height)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;

	if (u >= _max_num) return;
	int x = u % _width;
	int y = u / _width;

	if (_d_depth[y*_width + x] == 0) return;

	int tx[4] = { 0,1,0,-1 };
	int ty[4] = { 1,0,-1,0 };
	int nx, ny;

	for (int i = 0; i < 4; i++)
	{
		nx = x + tx[i];
		ny = y + ty[i];
		if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
		{
			if (_d_depth[ny*_width + nx] == 0)
			{
				_d_depth[y*_width + x] = 0;
				_d_index[y*_width + x] = -1;
				return;
			}
		}
	}
}

__global__ void FilterVertexKernel(int _max_num, int *_d_index, float4 *_d_vertex, float4 *_d_out_vertex, int *_out_vertex_num, int _width, int _height)
{
	int u = threadIdx.x + blockDim.x*blockIdx.x;

	if (u >= _max_num) return;

	int x = u % _width;
	int y = u / _width;
	if (_d_index[y*_width + x] == -1) return;

	int i = atomicAdd(_out_vertex_num, 1);
	_d_out_vertex[i] = _d_vertex[_d_index[y*_width + x]];
}

__global__ void FilterTopVertexKernel(int _max_num, int *_d_index, float4 *_d_vertex, float *_d_vertex_nearest_dis, int _bottom_edge_range, float _dis_thresh, float4 *_d_out_vertex, int *_out_vertex_num, int _width, int _height)
{
	int u = threadIdx.x + blockDim.x*blockIdx.x;

	if (u >= _max_num) return;

	int x = u % _width;
	int y = u / _width;
	if (_d_index[y*_width + x] == -1) return;

	if (y > _height - _bottom_edge_range && _d_vertex_nearest_dis[_d_index[y*_width + x]] > _dis_thresh) return;

	int i = atomicAdd(_out_vertex_num, 1);
	_d_out_vertex[i] = _d_vertex[_d_index[y*_width + x]];
}

void TBFilterPointCould::Fileter(int _edge_range, int _dis_thresh)
{
	CudaProjectVertexToDepthImage(d_top_vertex_.size(), RAW_PTR(d_top_vertex_), 
		RAW_PTR(d_top_depth_map_), RAW_PTR(d_top_index_map_), RAW_PTR(d_top_camera_pose_inv_t_), h_top_camera_fxfycxcy_, tcols_, trows_);

	CudaProjectVertexToDepthImage(d_bottom_vertex_.size(), RAW_PTR(d_bottom_vertex_),
		RAW_PTR(d_bottom_depth_map_), RAW_PTR(d_bottom_index_map_), RAW_PTR(d_bottom_camera_pose_inv_t_), h_bottom_camera_fxfycxcy_, bcols_, brows_);


	int block, grid;
	for (int i = 0; i < 3; i++)
	{
		block = 256;
		grid = DivUp(tcols_*trows_, block);
		RepairDepthMapKernel << <grid, block >> > (tcols_*trows_, RAW_PTR(d_top_depth_map_), tcols_, trows_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());

		block = 256;
		grid = DivUp(bcols_*brows_, block);
		RepairDepthMapKernel << <grid, block >> > (bcols_*brows_, RAW_PTR(d_bottom_depth_map_), bcols_, brows_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
	}

	for (int i = 0; i < 6; i++)
	{
		block = 256;
		grid = DivUp(tcols_*trows_, block);
		DeleteDepthMapKernel << <grid, block >> > (tcols_*trows_, RAW_PTR(d_top_depth_map_), RAW_PTR(d_top_index_map_), tcols_, trows_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());

		block = 256;
		grid = DivUp(bcols_*brows_, block);
		DeleteDepthMapKernel << <grid, block >> >(bcols_*brows_, RAW_PTR(d_bottom_depth_map_), RAW_PTR(d_bottom_index_map_), bcols_, brows_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
	}

	cv::Mat_<float> tdepth_map(tcols_, trows_);
	checkCudaErrors(cudaMemcpy(tdepth_map.data, RAW_PTR(d_top_depth_map_), tcols_*trows_ * sizeof(float), cudaMemcpyDeviceToHost));
	cv::Mat_<int> tindex_map(tcols_, trows_);
	checkCudaErrors(cudaMemcpy(tindex_map.data, RAW_PTR(d_top_index_map_), tcols_*trows_ * sizeof(int), cudaMemcpyDeviceToHost));
	cv::Mat_<float> bdepth_map(tcols_, trows_);
	checkCudaErrors(cudaMemcpy(bdepth_map.data, RAW_PTR(d_bottom_depth_map_), bcols_*brows_ * sizeof(float), cudaMemcpyDeviceToHost));
	cv::Mat_<int> bindex_map(tcols_, trows_);
	checkCudaErrors(cudaMemcpy(bindex_map.data, RAW_PTR(d_bottom_index_map_), bcols_*brows_ * sizeof(int), cudaMemcpyDeviceToHost));

	d_bottom_filter_vertex_.resize(d_bottom_vertex_.size());
	d_bottom_filter_vertex_num_.resize(1);
	d_bottom_filter_vertex_num_[0] = 0;
	block = 256;
	grid = DivUp(bcols_*brows_, block);
	FilterVertexKernel << <grid, block >> > (bcols_*brows_, RAW_PTR(d_bottom_index_map_), RAW_PTR(d_bottom_vertex_),
		RAW_PTR(d_bottom_filter_vertex_), RAW_PTR(d_bottom_filter_vertex_num_), bcols_, brows_);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	d_bottom_filter_vertex_.resize(d_bottom_filter_vertex_num_[0]);

	NearestPoint nearestPoint;
	nearestPoint.InitKDTree(RAW_PTR(d_bottom_filter_vertex_), d_bottom_filter_vertex_num_[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	d_top_nearest_idx_.resize(d_top_vertex_.size());
	d_top_nearest_dis_.resize(d_top_vertex_.size());
	nearestPoint.GetKnnResult(RAW_PTR(d_top_vertex_), d_top_vertex_.size(), 1, RAW_PTR(d_top_nearest_idx_), RAW_PTR(d_top_nearest_dis_));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	d_top_filter_vertex_.resize(d_top_vertex_.size());
	d_top_filter_vertex_num_.resize(1);
	d_top_filter_vertex_num_[0] = 0;
	block = 256;
	grid = DivUp(tcols_*trows_, block);
	FilterTopVertexKernel << <grid, block >> >(tcols_*trows_, RAW_PTR(d_top_index_map_), RAW_PTR(d_top_vertex_), RAW_PTR(d_top_nearest_dis_),
		_edge_range, _dis_thresh, RAW_PTR(d_top_filter_vertex_), RAW_PTR(d_top_filter_vertex_num_), tcols_, trows_);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	d_top_filter_vertex_.resize(d_top_filter_vertex_num_[0]);
}