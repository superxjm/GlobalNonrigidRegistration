#pragma once

#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <flann/algorithms/kdtree_cuda_3d_index.h>

// NOTE: You should modify the CudaL2 in the flann library as following
/*
inline __host__ __device__ float dotFloat3(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
struct CudaL2
{

	static float
		__host__ __device__
		axisDist(float a, float b)
	{
		return (a - b)*(a - b);
	}

	static float
		__host__ __device__
		dist(float4 a, float4 b)
	{
		float4 diff = a - b;
		// return dot(diff,diff);原来是这样的
		return dotFloat3(diff, diff);
	}
};
*/

class NearestPoint {
public:
	NearestPoint() : flann_index_{ nullptr } {}

	~NearestPoint()
	{
		delete flann_index_;
		flann_index_ = nullptr;
	}

	void clear()
	{
		if (flann_index_ != nullptr)
		{
			delete flann_index_;
		}
		flann_index_ = nullptr;
	}

	void InitKDTree(float4 *d_target_data, int num_target_data) {
		flann::Matrix<float> matrix_gpu((float*)d_target_data, num_target_data, 4);
		flann::KDTreeCuda3dIndexParams index_params;
		if (num_target_data > 64) {
			index_params["leaf_max_size"] = 64;
		}
		else if (num_target_data > 32) {
			index_params["leaf_max_size"] = 32;
		}
		else if (num_target_data > 16) {
			index_params["leaf_max_size"] = 16;
		}
		else if (num_target_data > 8) {
			index_params["leaf_max_size"] = 8;
		}
		else if (num_target_data > 4) {
			index_params["leaf_max_size"] = 4;
		}
		else if (num_target_data > 2) {
			index_params["leaf_max_size"] = 2;
		}
		else if (num_target_data > 0) {
			index_params["leaf_max_size"] = 1;
		}
		else {
			std::cout << "Warning: invalid data number: " << num_target_data << std::endl;
		}
		index_params["input_is_gpu_float4"] = true;
		flann_index_ = new flann::KDTreeCuda3dIndex<flann::L2<float> >(matrix_gpu, index_params);
		flann_index_->buildIndex();

		params_.matrices_in_gpu_ram = true;
		params_.sorted = true;
	}

	void GetKnnResult(int *d_knn_index, float *d_knn_weight, float4 *d_source_data, int num_source_data, int k) {
		flann::Matrix<int> indices_gpu(d_knn_index, num_source_data, k);
		flann::Matrix<float> dists_gpu(d_knn_weight, num_source_data, k);
		flann::Matrix<float> queries_gpu((float*)d_source_data, num_source_data, 3, sizeof(float) * 4);
		flann_index_->knnSearchGpu(queries_gpu, indices_gpu, dists_gpu, k, params_);
	}

	void GetKnnResult(float4 *d_source_data, int num_source_data, int k, int *d_knn_index, float *d_knn_weight) {
		flann::Matrix<int> indices_gpu(d_knn_index, num_source_data, k);
		flann::Matrix<float> dists_gpu(d_knn_weight, num_source_data, k);
		flann::Matrix<float> queries_gpu((float*)d_source_data, num_source_data, 3, sizeof(float) * 4);
		flann_index_->knnSearchGpu(queries_gpu, indices_gpu, dists_gpu, k, params_);
	}

private:
	flann::KDTreeCuda3dIndex<flann::L2<float> > *flann_index_;
	flann::SearchParams params_;
};


