#ifndef __PCG_SOLVER_H__
#define __PCG_SOLVER_H__

#include <thrust/device_vector.h>
#include <cusparse_v2.h>

#if 0
struct SparseMvWrapper {
	const int *ia;
	const int *ja;
	const float *a;
	const float *x;
	float *res;
	__device__ void BlockReduce64(float *s_tmp_res, const float *s_data, float *res) {
		int warp_id = threadIdx.x >> 5;
		if (threadIdx.x < 2) {
			s_tmp_res[threadIdx.x] = 0.f;
		}

		float value = s_data[threadIdx.x];
		for (int i = 16; i >= 1; i >>= 1) {
			value += __shfl_xor_sync(__activemask(), value, i, 32);
		}
		// lane_id == 31, write result back.
		if ((threadIdx.x & 0x1f) == 31) {
			s_tmp_res[warp_id] = value;
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			*res = s_tmp_res[0] + s_tmp_res[1];
		}
	}

	__device__ void operator()() {
		__shared__ float s_res_tmp[64];
		__shared__ int s_start_idx;
		__shared__ int s_num_row_nnz;
		__shared__ float s_reduce_tmp[2];

		if (threadIdx.x == 0) {
			s_start_idx = ia[blockIdx.x];
			s_num_row_nnz = ia[blockIdx.x + 1] - s_start_idx;
		}
		__syncthreads();

		s_res_tmp[threadIdx.x] = 0.f;

		for (int iter = threadIdx.x; iter < s_num_row_nnz; iter += blockDim.x) {
			s_res_tmp[threadIdx.x] += a[s_start_idx + iter] * x[ja[s_start_idx + iter]];
		}
		__syncthreads();

		BlockReduce64(s_reduce_tmp, s_res_tmp, &(res[blockIdx.x]));
	}
};

class PcgLinearSolver {
public:
	PcgLinearSolver();
	~PcgLinearSolver() {
		cudaFree(buf1);
		buf2.clear();
		cusparseDestroy(cusparseHandle_);
		cusparseDestroyMatDescr(descr_);
	}
	void Init(int length_a);
	void MySolveGPU(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter);
	void SolveGPU(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter);
	void SolveCPU(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter);
	void SolveCPUOpt(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter);
	void SolveCPUOptDouble(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter);

private:
	int row_ = 0;
	const float floatone_ = 1.0f, floatzero_ = 0.0f;
	float alpha_ = 0.0f, beta_ = 0.0f, zr_dot_old = 0.0f, zr_dot = 0.0f;
	float *d_alpha, *d_neg_alpha, *d_beta, *d_zr_dot[2], *buf1;
	thrust::device_vector<float> buf2;
	float *d_p_, *d_omega_, *d_r_, *d_z_;
	thrust::device_vector<float> d_p_vec_, d_omega_vec_, d_r_vec_, d_z_vec_;

	std::vector<float> omega_vec, p_vec, r_vec, z_vec, delta_x_vec, precond_vec;

	std::vector<double> buf3;
	std::vector<double> omega_vec_double, p_vec_double, r_vec_double, z_vec_double, delta_x_vec_double, precond_vec_double;
	double alpha_double = 0.0f, beta_double = 0.0f, zr_dot_old_double = 0.0f, zr_dot_double = 0.0f;

	SparseMvWrapper spmv_wrapper;

	cusparseHandle_t cusparseHandle_;
	cusparseMatDescr_t descr_;
};
#endif

#if 1
struct SparseMvWrapper {
	const int *ia;
	const int *ja;
	const float *a;
	const float *x;
	float *res;
	__device__ void BlockReduce64(float *s_tmp_res, const float *s_data, float *res) {
		int warp_id = threadIdx.x >> 5;
		if (threadIdx.x < 2) {
			s_tmp_res[threadIdx.x] = 0.f;
		}

		float value = s_data[threadIdx.x];
		for (int i = 16; i >= 1; i >>= 1) {
			value += __shfl_xor_sync(__activemask(), value, i, 32);
		}
		// lane_id == 31, write result back.
		if ((threadIdx.x & 0x1f) == 31) {
			s_tmp_res[warp_id] = value;
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			*res = s_tmp_res[0] + s_tmp_res[1];
		}
	}

	__device__ void operator()() {
		__shared__ float s_res_tmp[64];
		__shared__ int s_start_idx;
		__shared__ int s_num_row_nnz;
		__shared__ float s_reduce_tmp[2];

		if (threadIdx.x == 0) {
			s_start_idx = ia[blockIdx.x];
			s_num_row_nnz = ia[blockIdx.x + 1] - s_start_idx;
		}
		__syncthreads();

		s_res_tmp[threadIdx.x] = 0.f;

		for (int iter = threadIdx.x; iter < s_num_row_nnz; iter += blockDim.x) {
			s_res_tmp[threadIdx.x] += a[s_start_idx + iter] * x[ja[s_start_idx + iter]];
		}
		__syncthreads();

		BlockReduce64(s_reduce_tmp, s_res_tmp, &(res[blockIdx.x]));
	}
};

class PcgLinearSolver {
public:
	PcgLinearSolver();
	~PcgLinearSolver() {
		cusparseDestroy(m_cusparseHandle);
		cusparseDestroyMatDescr(m_descr);
		buf.clear();
	}
	void init(int length_a);
	void solveCPUOpt(thrust::device_vector<float>& d_x,
	                 int* d_ia, int* d_ja, float* d_a, int num_nnz,
	                 thrust::device_vector<float>& d_b, thrust::device_vector<float>& preconditioner,
	                 int max_iter);

private:
	int m_row = 0, m_maxRow;
	const float m_floatone = 1.0f, m_floatzero = 0.0f;
	float alpha = 0.0f, beta = 0.0f, zr_dot_old = 0.0f, zr_dot = 0.0f;

	thrust::device_vector<float> buf;
	float *d_p, *d_omega;
	std::vector<float> omega_vec, p_vec, r_vec, z_vec, delta_x_vec, precond_vec;

	SparseMvWrapper spmv_wrapper;

	cusparseHandle_t m_cusparseHandle;
	cusparseMatDescr_t m_descr;
};
#endif

#endif
