#include "PcgSolver.h"

#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <cusparse.h>
#include <helper_cuda.h>
#include <thrust/inner_product.h>

#include "Helpers/InnorealTimer.hpp"
#include "Helpers/xUtils.h"

#if 0
__global__ void SparseMvKernel(SparseMvWrapper spmv_wrapper) {
	spmv_wrapper();
}

PcgLinearSolver::PcgLinearSolver() : cusparseHandle_{ nullptr }, descr_{ nullptr }
{
	cusparseCreate(&cusparseHandle_);
	cusparseCreateMatDescr(&descr_);
	cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);

	row_ = MAX_FRAG_NUM * NODE_NUM_EACH_FRAG * 12;
	checkCudaErrors(cudaMalloc((void **)&buf1, 5 * sizeof(float)));
	d_alpha = buf1;
	d_neg_alpha = buf1 + 1;
	d_beta = buf1 + 2;
	d_zr_dot[0] = buf1 + 3;
	d_zr_dot[1] = buf1 + 4;

	d_p_vec_.resize(row_);
	d_omega_vec_.resize(row_);
	d_r_vec_.resize(row_);
	d_z_vec_.resize(row_);

	buf2.resize(row_ * 4);
	d_p_ = RAW_PTR(buf2);
	d_omega_ = RAW_PTR(buf2) + row_;
	d_r_ = RAW_PTR(buf2) + row_ * 2;
	d_z_ = RAW_PTR(buf2) + row_ * 3;

	omega_vec.resize(row_);
	p_vec.resize(row_);
	r_vec.resize(row_);
	z_vec.resize(row_);
	delta_x_vec.resize(row_, 0);
	precond_vec.resize(row_);

	omega_vec_double.resize(row_);
	p_vec_double.resize(row_);
	r_vec_double.resize(row_);
	z_vec_double.resize(row_);
	delta_x_vec_double.resize(row_, 0);
	precond_vec_double.resize(row_);
}

void PcgLinearSolver::Init(int JTJ_row) {
	row_ = JTJ_row;
	if (row_ > MAX_FRAG_NUM * NODE_NUM_EACH_FRAG * 12)
	{
		std::cout << "node num exceed max node num" << std::endl;
		std::exit(0);
	}
}

__global__ void MyDot64(float *sum, float *vec1, float *vec2, int size)
{
	volatile __shared__ float sharedBuf[64];

	float val = 0;

	for (int i = threadIdx.x; i < size; i += 64) {
		val += vec1[i] * vec2[i];
	}

	sharedBuf[threadIdx.x] = val;
	__syncthreads();

	if (threadIdx.x < 32) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 32];
	}
	if (threadIdx.x < 16) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 16];
	}
	if (threadIdx.x < 8) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 8];
	}
	if (threadIdx.x < 4) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 4];
	}
	if (threadIdx.x < 2) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 2];
	}
	if (threadIdx.x < 1) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 1];
		*sum = sharedBuf[0];
	}
}

__global__ void Dot64BetaZrDot(float *beta, float *sum, float *alpha, float *vec1, float *vec2, int size)
{
	volatile __shared__ float sharedBuf[64];

	float val = 0;

	for (int i = threadIdx.x; i < size; i += 64) {
		val += vec1[i] * vec2[i];
	}

	sharedBuf[threadIdx.x] = val;
	__syncthreads();

	if (threadIdx.x < 32) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 32];
	}
	if (threadIdx.x < 16) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 16];
	}
	if (threadIdx.x < 8) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 8];
	}
	if (threadIdx.x < 4) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 4];
	}
	if (threadIdx.x < 2) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 2];
	}
	if (threadIdx.x < 1) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 1];
		*sum = sharedBuf[0];
		*beta = *sum / *alpha;
	}
}

__global__ void Dot64AlphaNegAlpha(float *sum, float *negsum, float *alpha, float *vec1, float *vec2, int size)
{
	volatile __shared__ float sharedBuf[64];

	float val = 0;

	for (int i = threadIdx.x; i < size; i += 64) {
		val += vec1[i] * vec2[i];
	}

	sharedBuf[threadIdx.x] = val;
	__syncthreads();

	if (threadIdx.x < 32) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 32];
	}
	if (threadIdx.x < 16) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 16];
	}
	if (threadIdx.x < 8) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 8];
	}
	if (threadIdx.x < 4) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 4];
	}
	if (threadIdx.x < 2) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 2];
	}
	if (threadIdx.x < 1) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 1];
		*sum = *alpha / sharedBuf[0];
		*negsum = -(*sum);
	}
}

__global__ void MyReduce64(float *sum, float *vec, int size)
{
	volatile __shared__ float sharedBuf[64];

	float val = 0;

#if 0
	int step = (size + 64 - 1) / 64;
	int start = threadIdx.x * step, end = start + step;
	if (end > size) {
		end = size;
	}
#endif
	for (int i = threadIdx.x; i < size; i += 64) {
		val += vec[i];
	}

	sharedBuf[threadIdx.x] = val;
	__syncthreads();

	if (threadIdx.x < 32) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 32];
	}
	if (threadIdx.x < 16) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 16];
	}
	if (threadIdx.x < 8) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 8];
	}
	if (threadIdx.x < 4) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 4];
	}
	if (threadIdx.x < 2) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 2];
	}
	if (threadIdx.x < 1) {
		sharedBuf[threadIdx.x] += sharedBuf[threadIdx.x + 1];
		*sum = sharedBuf[0];
	}
}

__global__ void MySaxpyKernal(float *vecRes, float *vec1, float *vec2, float *val, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
	{
		return;
	}

	vecRes[idx] = (*val) * vec1[idx] + vec2[idx];
}

__global__ void SaxpyKernalRZ(float *vecRes1, float *vecRes2, float *preconditioner,
	float *vec1, float *vec2, float *val, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
	{
		return;
	}

	float v = (*val) * vec1[idx] + vec2[idx];
	vecRes1[idx] = v;
	vecRes2[idx] = v * preconditioner[idx];
}


__global__ void MyMultiplyKernal(float *vecRes, float *vec1, float *vec2, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
	{
		return;
	}

	vecRes[idx] = vec1[idx] * vec2[idx];
}

void PcgLinearSolver::MySolveGPU(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter) {
	//thrust::copy(d_b.begin(), d_b.end(), d_r_.begin());
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_r_), RAW_PTR(d_b), row_ * sizeof(float), cudaMemcpyDeviceToDevice));

	//thrust::multiplies<float> op_multiplies;
	//thrust::transform(preconditioner.begin(), preconditioner.end(), d_r_.begin(), d_p_.begin(), op_multiplies);

	int block = 256, grid = DivUp(row_, block);
	MyMultiplyKernal << <grid, block >> > (RAW_PTR(d_p_), RAW_PTR(preconditioner), RAW_PTR(d_r_), row_);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//innoreal::InnoRealTimer timer;

	//zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_p_.begin(), 0.0f);
	MyDot64 << <1, 64 >> > (d_zr_dot[0], RAW_PTR(d_r_), RAW_PTR(d_p_), row_);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//float val;
	//checkCudaErrors(cudaMemcpy(&val, d_zr_dot[0], sizeof(float), cudaMemcpyDeviceToHost));
	//std::cout << "val: " << val << std::endl;
	//std::cout << "zr_dot: " << zr_dot << std::endl;
	//timer.TimeStart();
	for (int k = 0; k < max_iter; k++) {
		//zr_dot_old = zr_dot;
		cusparseScsrmv(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE, row_, row_, num_nnz, &floatone_, descr_, d_a, d_ia, d_ja, RAW_PTR(d_p_), &floatzero_, RAW_PTR(d_omega_));
		//alpha_ = zr_dot / thrust::inner_product(d_p_.begin(), d_p_.end(), d_omega_.begin(), 0.0f);
		//alpha_ = zr_dot / thrust::inner_product(d_p_.begin(), d_p_.end(), d_omega_.begin(), 0.0f);
		//timer.TimeStart();
		Dot64AlphaNegAlpha << <1, 64 >> > (d_alpha, d_neg_alpha, d_zr_dot[k % 2], RAW_PTR(d_p_), RAW_PTR(d_omega_), row_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		//timer.TimeEnd();
		//std::cout << "1: " << timer.TimeGap_in_ms() << std::endl;
		//std::cout << "alpha: " << alpha_ << std::endl;
		//thrust::transform(d_p_.begin(), d_p_.end(), d_x.begin(), d_x.begin(), SaxpyFunctor(alpha_));
		//thrust::transform(d_omega_.begin(), d_omega_.end(), d_r_.begin(), d_r_.begin(), SaxpyFunctor(-alpha_));
		//thrust::transform(preconditioner.begin(), preconditioner.end(), d_r_.begin(), d_z_.begin(), op_multiplies);

		//timer.TimeStart();
		MySaxpyKernal << <grid, block >> > (RAW_PTR(d_x), RAW_PTR(d_p_), RAW_PTR(d_x), d_alpha, row_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		//timer.TimeEnd();
		//std::cout << "2: " << timer.TimeGap_in_ms() << std::endl;	

		//timer.TimeStart();
		SaxpyKernalRZ << <grid, block >> > (RAW_PTR(d_r_), RAW_PTR(d_z_), RAW_PTR(preconditioner), RAW_PTR(d_omega_), RAW_PTR(d_r_), d_neg_alpha, row_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		//timer.TimeEnd();
		//std::cout << "3: " << timer.TimeGap_in_ms() << std::endl;

#if 0
		float val2;
		checkCudaErrors(cudaMemcpy(&val2, d_neg_alpha, sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "val: " << val2 << std::endl;

		float val[3000];
		checkCudaErrors(cudaMemcpy(val, RAW_PTR(d_omega_), 3000 * sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 1000; i < 1030; ++i)
		{
			std::cout << val[i] << ", ";
		}
		std::cout << std::endl;

		checkCudaErrors(cudaMemcpy(val, RAW_PTR(d_r_), 3000 * sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 1000; i < 1030; ++i)
		{
			std::cout << val[i] << ", ";
		}
		std::cout << std::endl;
#endif

		//zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_z_.begin(), 0.0f);
		//beta_ = zr_dot / zr_dot_old;
		//timer.TimeStart();
		Dot64BetaZrDot << <1, 64 >> > (d_beta, d_zr_dot[(k + 1) % 2], d_zr_dot[k % 2], RAW_PTR(d_r_), RAW_PTR(d_z_), row_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		//timer.TimeEnd();
		//std::cout << "5: " << timer.TimeGap_in_ms() << std::endl;

		//float val;
		//checkCudaErrors(cudaMemcpy(&val, d_beta, sizeof(float), cudaMemcpyDeviceToHost));
		//std::cout << "val: " << val << std::endl;

		//std::cout << "beta: " << beta_ << std::endl;
		//thrust::transform(d_z_.begin(), d_z_.end(), d_p_.begin(), d_p_.begin(), SxpayFunctor(beta_));
		//timer.TimeStart();

		MySaxpyKernal << <grid, block >> > (RAW_PTR(d_p_), RAW_PTR(d_p_), RAW_PTR(d_z_), d_beta, row_);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		//timer.TimeEnd();
		//std::cout << "6: " << timer.TimeGap_in_ms() << std::endl;
#if 0
		float val[3000];
		checkCudaErrors(cudaMemcpy(val, RAW_PTR(d_p_), 3000 * sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 1000; i < 1030; ++i)
		{
			std::cout << val[i] << ", ";
		}
		std::cout << std::endl;
#endif
	}
	//timer.TimeEnd();
	//std::cout << "pcg iter time: " << timer.TimeGap_in_ms() << std::endl;
}
#if 0
void PcgLinearSolver::SolveGPU(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter) {
#if 1
	thrust::copy(d_b.begin(), d_b.end(), d_r_vec_.begin());

	//std::cout << "row: " << row_ << std::endl;

	thrust::multiplies<float> op_multiplies;
	thrust::transform(preconditioner.begin(), preconditioner.end(), d_r_vec_.begin(), d_p_vec_.begin(), op_multiplies);

#if 0
	thrust::host_vector<float> vec = d_r_;
	thrust::host_vector<float> vec2 = preconditioner;
	thrust::host_vector<float> vec3 = d_p_;
	for (int i = 0; i < vec3.size(); ++i)
	{
		if (isnan(vec3[i]))
		{
			std::cout << vec[i] << ", ";
			std::cout << vec2[i] << ", ";
			std::cout << vec3[i] << std::endl;
			std::cout << "d_p_ ERROR" << std::endl;
			std::exit(0);
		}
	}
	std::cout << "size: " << vec.size() << std::endl;
	std::cout << "size2: " << vec2.size() << std::endl;
#endif

	//innoreal::InnoRealTimer timer;

	zr_dot = thrust::inner_product(d_r_vec_.begin(), d_r_vec_.end(), d_p_vec_.begin(), 0.0f);
	float val;
	//std::cout << "val2: " << zr_dot << std::endl;
	//std::cout << "zr_dot: " << zr_dot << std::endl;
	for (int k = 0; k < max_iter; k++) {
		zr_dot_old = zr_dot;
		cusparseScsrmv(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE, row_, row_, num_nnz, &floatone_, descr_, d_a, d_ia, d_ja, RAW_PTR(d_p_vec_), &floatzero_, RAW_PTR(d_omega_vec_));
		//alpha_ = zr_dot / thrust::inner_product(d_p_.begin(), d_p_.end(), d_omega_.begin(), 0.0f);
		alpha_ = zr_dot / thrust::inner_product(d_p_vec_.begin(), d_p_vec_.end(), d_omega_vec_.begin(), 0.0f);
		//std::cout << "alpha: " << alpha_ << std::endl;
		//timer.TimeStart();
		thrust::transform(d_p_vec_.begin(), d_p_vec_.end(), d_x.begin(), d_x.begin(), SaxpyFunctor(alpha_));
		//timer.TimeEnd();
		//std::cout << "transform time: " << timer.TimeGap_in_ms() << std::endl;
		thrust::transform(d_omega_vec_.begin(), d_omega_vec_.end(), d_r_vec_.begin(), d_r_vec_.begin(), SaxpyFunctor(-alpha_));

		thrust::transform(preconditioner.begin(), preconditioner.end(), d_r_vec_.begin(), d_z_vec_.begin(), op_multiplies);

		zr_dot = thrust::inner_product(d_r_vec_.begin(), d_r_vec_.end(), d_z_vec_.begin(), 0.0f);
		beta_ = zr_dot / zr_dot_old;
		//std::cout << "val2: " << beta_ << std::endl;
		//std::cout << "beta: " << beta_ << std::endl;

		thrust::transform(d_z_vec_.begin(), d_z_vec_.end(), d_p_vec_.begin(), d_p_vec_.begin(), SxpayFunctor(beta_));
#if 0
		float val[3000];
		checkCudaErrors(cudaMemcpy(val, RAW_PTR(d_p_vec_), 3000 * sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 1000; i < 1030; ++i)
		{
			std::cout << val[i] << ", ";
		}
		std::cout << std::endl;
#endif
	}
#endif
}
#endif

void PcgLinearSolver::SolveCPUOpt(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter)
{
	innoreal::InnoRealTimer timer;
	timer.TimeStart();
	float *p1, *p2, *p3;
	checkCudaErrors(cudaMemcpy(r_vec.data(), RAW_PTR(d_b), row_ * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(precond_vec.data(), RAW_PTR(preconditioner), row_ * sizeof(float), cudaMemcpyDeviceToHost));
	timer.TimeEnd();
	//std::cout << "t1: " << timer.TimeGap_in_ms() << std::endl;

	timer.TimeStart();
	p1 = p_vec.data(), p2 = precond_vec.data(), p3 = r_vec.data();
	zr_dot = 0.0f;
	for (int i = 0; i < row_; ++i)
	{
		*p1 = (*(p2++))*(*p3);
		zr_dot += (*(p1++))*(*(p3++));
	}
	spmv_wrapper.ia = d_ia;
	spmv_wrapper.ja = d_ja;
	spmv_wrapper.a = d_a;
	spmv_wrapper.res = d_omega_;

	memset(delta_x_vec.data(), 0, row_ * sizeof(float));
	timer.TimeEnd();
	//std::cout << "t2: " << timer.TimeGap_in_ms() << std::endl;
	double r_val;
	int iterCnt = 0;
	timer.TimeStart();
	for (int k = 0; k < max_iter; k++)
	{
		zr_dot_old = zr_dot;
		//std::cout << "zr_dot" << zr_dot << std::endl;
		checkCudaErrors(cudaMemcpy(d_p_, p_vec.data(), row_ * sizeof(float), cudaMemcpyHostToDevice));

		//spmv_wrapper.x = d_p_;
		//SparseMvKernel << <row_, 64 >> > (spmv_wrapper);
		//checkCudaErrors(cudaDeviceSynchronize());
		cusparseScsrmv(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE, row_, row_, num_nnz, &floatone_, descr_,
			d_a, d_ia, d_ja, RAW_PTR(d_p_), &floatzero_, RAW_PTR(d_omega_));

		checkCudaErrors(cudaMemcpy(omega_vec.data(), d_omega_, row_ * sizeof(float), cudaMemcpyDeviceToHost));

		p1 = p_vec.data(), p2 = omega_vec.data();
		alpha_ = 0.0f;
		for (int i = 0; i < row_; ++i)
		{
			alpha_ += (*(p1++))*(*(p2++));
		}
		alpha_ = zr_dot / alpha_;

#if 0
		for (int i = 0; i < 30; ++i)
		{
			std::cout << p_vec[i] << ", ";
		}
		std::cout << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << omega_vec[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "zr_dot: " << zr_dot << std::endl;
		std::cout << "alpha: " << alpha_ << std::endl;
#endif

		p1 = delta_x_vec.data(), p2 = p_vec.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) += alpha_ * (*(p2++));
		}

#if 0
		std::cout << std::endl << alpha_ << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << delta_x_vec[i] << ", ";
		}
		std::cout << std::endl;
#endif

		p1 = r_vec.data(), p2 = omega_vec.data();
		r_val = 0.0;
		for (int i = 0; i < row_; ++i)
		{
			r_val += (*p1)*(*p1);
			*(p1++) -= alpha_ * (*(p2++));
		}

#if 0
		std::cout << std::endl << alpha_ << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << r_vec[i] << ", ";
		}
		std::cout << std::endl;
#endif

		if (r_val < 1.0e-30)
			break;

		p1 = z_vec.data(), p2 = precond_vec.data(), p3 = r_vec.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) = (*(p2++)) * (*(p3++));
		}

#if 0
		for (int i = 0; i < 30; ++i)
		{
			std::cout << z_vec[i] << ", ";
		}
		std::cout << std::endl;
#endif

		p1 = r_vec.data(), p2 = z_vec.data();
		zr_dot = 0.0f;
		for (int i = 0; i < row_; ++i)
		{
			zr_dot += (*(p1++))*(*(p2++));
		}

		beta_ = zr_dot / zr_dot_old;
		p1 = p_vec.data(), p2 = z_vec.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) = (*(p2++)) + beta_ * (*p1);
		}
		++iterCnt;
#if 0
		for (int i = 1000; i < 1030; ++i)
		{
			std::cout << p_vec[i] << ", ";
		}
		std::cout << std::endl;
#endif
	}
	timer.TimeEnd();
	//std::cout << "t: " << timer.TimeGap_in_ms() << std::endl;
	//printf("PCG iter count: %d\n", iterCnt);
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_x), delta_x_vec.data(), row_ * sizeof(float), cudaMemcpyHostToDevice));
}

void PcgLinearSolver::SolveCPUOptDouble(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter)
{
	std::vector<int> JTJ_ia(row_ + 1);
	std::vector<int> JTJ_ja(num_nnz);
	std::vector<float> JTJ_a(num_nnz);
	checkCudaErrors(cudaMemcpy(JTJ_ia.data(), (d_ia), (row_ + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(JTJ_ja.data(), (d_ja), num_nnz * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(JTJ_a.data(), (d_a), num_nnz * sizeof(float), cudaMemcpyDeviceToHost));
	//std::vector<double> JTb(row_);
	//std::vector<double> precond(row_);
	double JTJ[48][48];
	for (int i = 0; i < JTJ_ia.size() - 1; ++i)
	{
		for (int j = JTJ_ia[i]; j < JTJ_ia[i + 1]; ++j)
		{
			JTJ[i][JTJ_ja[j]] = JTJ_a[j];
		}
	}

	double *p1, *p2, *p3;
	checkCudaErrors(cudaMemcpy(r_vec.data(), RAW_PTR(d_b), row_ * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(precond_vec.data(), RAW_PTR(preconditioner), row_ * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < r_vec.size(); ++i)
	{
		r_vec_double[i] = r_vec[i];
	}
	for (int i = 0; i < precond_vec.size(); ++i)
	{
		precond_vec_double[i] = precond_vec[i];
	}

	p1 = p_vec_double.data(), p2 = precond_vec_double.data(), p3 = r_vec_double.data();
	zr_dot_double = 0.0f;
	for (int i = 0; i < row_; ++i)
	{
		*p1 = (*(p2++))*(*p3);
		zr_dot_double += (*(p1++))*(*(p3++));
	}

	memset(delta_x_vec_double.data(), 0, row_ * sizeof(double));
	double r_val_double;
	int iterCnt = 0;
	for (int k = 0; k < max_iter; k++)
	{
		zr_dot_old_double = zr_dot_double;
		//std::cout << "zr_dot" << zr_dot << std::endl;
#if 0
		for (int i = 0; i < p_vec_double.size(); ++i)
		{
			p_vec[i] = p_vec_double[i];
		}
		checkCudaErrors(cudaMemcpy(d_p_, p_vec.data(), row_ * sizeof(float), cudaMemcpyHostToDevice));
		cusparseScsrmv(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE, row_, row_, num_nnz, &floatone_, descr_,
			d_a, d_ia, d_ja, d_p_, &floatzero_, d_omega_);
		checkCudaErrors(cudaMemcpy(omega_vec.data(), d_omega_, row_ * sizeof(float), cudaMemcpyDeviceToHost));
#endif
		for (int i = 0; i < 48; ++i)
		{
			omega_vec_double[i] = 0.0;
			for (int j = 0; j < 48; ++j)
			{
				omega_vec_double[i] += JTJ[i][j] * p_vec_double[j];
			}
		}

		p1 = p_vec_double.data(), p2 = omega_vec_double.data();
		alpha_double = 0.0;
		for (int i = 0; i < row_; ++i)
		{
			alpha_double += (*(p1++))*(*(p2++));
		}
		alpha_double = zr_dot_double / alpha_double;

#if 0
		for (int i = 0; i < 30; ++i)
		{
			std::cout << p_vec_double[i] << ", ";
		}
		std::cout << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << omega_vec_double[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "zr_dot: " << zr_dot_double << std::endl;
		std::cout << "alpha: " << alpha_double << std::endl;
#endif

		p1 = delta_x_vec_double.data(), p2 = p_vec_double.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) += alpha_double * (*(p2++));
		}

		p1 = r_vec_double.data(), p2 = omega_vec_double.data();
		r_val_double = 0.0;
		for (int i = 0; i < row_; ++i)
		{
			r_val_double += (*p1)*(*p1);
			*(p1++) -= alpha_double * (*(p2++));
		}

#if 0
		std::cout << std::endl << alpha_double << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << r_vec_double[i] << ", ";
		}
		std::cout << std::endl;
#endif

		if (r_val_double < 1.0e-50)
			break;

		p1 = z_vec_double.data(), p2 = precond_vec_double.data(), p3 = r_vec_double.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) = (*(p2++)) * (*(p3++));
		}

#if 0
		for (int i = 0; i < 30; ++i)
		{
			std::cout << z_vec_double[i] << ", ";
		}
		std::cout << std::endl;
#endif

		p1 = r_vec_double.data(), p2 = z_vec_double.data();
		zr_dot_double = 0.0f;
		for (int i = 0; i < row_; ++i)
		{
			zr_dot_double += (*(p1++))*(*(p2++));
		}

		beta_double = zr_dot_double / zr_dot_old_double;
		p1 = p_vec_double.data(), p2 = z_vec_double.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) = (*(p2++)) + beta_double * (*p1);
		}
		++iterCnt;
	}
#if 0
	for (int i = 0; i < delta_x_vec_double.size(); ++i)
	{
		std::cout << delta_x_vec_double[i] << ", ";
	}
	std::cout << std::endl;
#endif
	printf("PCG iter count: %d\n", iterCnt);
	for (int i = 0; i < delta_x_vec_double.size(); ++i)
	{
		delta_x_vec[i] = delta_x_vec_double[i];
	}
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_x), delta_x_vec.data(), row_ * sizeof(float), cudaMemcpyHostToDevice));
}

void PcgLinearSolver::SolveCPU(int *d_ia, int *d_ja, float *d_a, int num_nnz, thrust::device_vector<float> &d_b, thrust::device_vector<float> &d_x, thrust::device_vector<float> &preconditioner, int max_iter) {
#if 0
	//innoreal::InnoRealTimer timer;
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_r_), RAW_PTR(d_b), row_ * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(r_vec.data(), RAW_PTR(d_r_), row_ * sizeof(float), cudaMemcpyDeviceToHost));
#if 0
	thrust::host_vector<float> vec = d_r_;
	thrust::host_vector<float> vec2 = preconditioner;
	thrust::host_vector<float> vec3 = d_p_;
	for (int i = 0; i < vec3.size(); ++i)
	{
		if (isnan(vec3[i]))
		{
			std::cout << vec[i] << ", ";
			std::cout << vec2[i] << ", ";
			std::cout << vec3[i] << std::endl;
			std::cout << "d_p_ ERROR" << std::endl;
			std::exit(0);
		}
	}
	std::cout << "size: " << vec.size() << std::endl;
	std::cout << "size2: " << vec2.size() << std::endl;
#endif
	checkCudaErrors(cudaMemcpy(precond_vec.data(), RAW_PTR(preconditioner), row_ * sizeof(float), cudaMemcpyDeviceToHost));


	//zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_p_.begin(), 0.0f);
	int block = 1024, grid = DivUp(row_, block);
	MyMultiplyKernal << <grid, block >> > (RAW_PTR(d_p_), RAW_PTR(preconditioner), RAW_PTR(d_r_), row_);

	float val[3000];
	checkCudaErrors(cudaMemcpy(val, RAW_PTR(d_p_), 3000 * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 1000; i < 1030; ++i)
	{
		std::cout << val[i] << ", ";
	}
	std::cout << std::endl;

	MyDot64 << <1, 64 >> > (d_zr_dot[0], RAW_PTR(d_r_), RAW_PTR(d_p_), row_);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(&zr_dot, d_zr_dot[0], sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "zr_dot: " << zr_dot << std::endl;
#endif
	float *p1, *p2, *p3;
#if 1
	checkCudaErrors(cudaMemcpy(r_vec.data(), RAW_PTR(d_b), row_ * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(precond_vec.data(), RAW_PTR(preconditioner), row_ * sizeof(float), cudaMemcpyDeviceToHost));

	p1 = p_vec.data(), p2 = precond_vec.data(), p3 = r_vec.data();
	zr_dot = 0.0f;
	for (int i = 0; i < row_; ++i)
	{
		*p1 = (*(p2++))*(*p3);
		zr_dot += (*(p1++))*(*(p3++));
	}

	//checkCudaErrors(cudaMemcpy(RAW_PTR(d_p_), p_vec.data(), row_ * sizeof(float), cudaMemcpyHostToDevice));
#endif
	//std::exit(0);
	memset(delta_x_vec.data(), 0, row_ * sizeof(float));
	for (int k = 0; k < max_iter; k++)
	{
		//std::cout << "zr_dot1: " << zr_dot << std::endl;

		zr_dot_old = zr_dot;
		checkCudaErrors(cudaMemcpy(RAW_PTR(d_p_), p_vec.data(), row_ * sizeof(float), cudaMemcpyHostToDevice));
		cusparseScsrmv(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE, row_, row_, num_nnz, &floatone_, descr_,
			d_a, d_ia, d_ja, RAW_PTR(d_p_), &floatzero_, RAW_PTR(d_omega_));

		//timer.TimeStart();
		//timer.TimeEnd();
		//std::cout << "time memcpy1: " << timer.TimeGap_in_ms() << std::endl;
		//timer.TimeStart();
		checkCudaErrors(cudaMemcpy(omega_vec.data(), RAW_PTR(d_omega_), row_ * sizeof(float), cudaMemcpyDeviceToHost));
		//timer.TimeEnd();
		//std::cout << "time memcpy2: " << timer.TimeGap_in_ms() << std::endl;
		//timer.TimeStart();
		//checkCudaErrors(cudaMemcpy(test.data(), RAW_PTR(testDevice), row_ * 2 * sizeof(float), cudaMemcpyDeviceToHost));
		//timer.TimeEnd();
		//std::cout << "time memcpy3: " << timer.TimeGap_in_ms() << std::endl;

#if 0
		std::cout << "omega_vec:" << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << omega_vec[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "p_vec:" << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << p_vec[i] << ", ";
		}
		std::cout << std::endl;
#endif

		p1 = p_vec.data(), p2 = omega_vec.data();
		alpha_ = 0.0f;
		for (int i = 0; i < row_; ++i)
		{
			alpha_ += (*(p1++))*(*(p2++));
		}
		std::cout << "alpha_:" << alpha_ << std::endl;
		alpha_ = zr_dot / alpha_;	

		p1 = delta_x_vec.data(), p2 = p_vec.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) += alpha_ * (*(p2++));
		}

		std::cout << "r_vec:" << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << r_vec[i] << ", ";
		}
		std::cout << std::endl;

		p1 = r_vec.data(), p2 = omega_vec.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) -= alpha_ * (*(p2++));
		}

		p1 = z_vec.data(), p2 = precond_vec.data(), p3 = r_vec.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) = (*(p2++)) * (*(p3++));
		}

		std::cout << "r_vec:" << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << r_vec[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "z_vec:" << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << z_vec[i] << ", ";
		}
		std::cout << std::endl;
		p1 = r_vec.data(), p2 = z_vec.data();
		zr_dot = 0.0f;
		for (int i = 0; i < row_; ++i)
		{
			zr_dot += (*(p1++))*(*(p2++));
		}
		std::cout << "zr_dot: " << zr_dot << std::endl;

		beta_ = zr_dot / zr_dot_old;
		p1 = p_vec.data(), p2 = z_vec.data();
		for (int i = 0; i < row_; ++i)
		{
			*(p1++) = (*(p2++)) + beta_ * (*p1);
		}
#if 0
		std::cout << "p_vec:" << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << p_vec[i] << ", ";
		}
		std::cout << std::endl;
#endif
	}
#if 0
	for (int i = 0; i < delta_x_vec.size(); ++i)
	{
		std::cout << delta_x_vec[i] << ", ";
	}
	std::cout << std::endl;
#endif

	checkCudaErrors(cudaMemcpy(RAW_PTR(d_x), delta_x_vec.data(), row_ * sizeof(float), cudaMemcpyHostToDevice));
}
#endif

#if 1
__global__ void SparseMvKernel(SparseMvWrapper spmv_wrapper) {
	spmv_wrapper();
}

PcgLinearSolver::PcgLinearSolver() : m_cusparseHandle{ nullptr }, m_descr{ nullptr } 
{
	cusparseCreate(&m_cusparseHandle);
	cusparseCreateMatDescr(&m_descr);
	cusparseSetMatIndexBase(m_descr, CUSPARSE_INDEX_BASE_ZERO);

	m_maxRow = MAX_FRAG_NUM * NODE_NUM_EACH_FRAG * 12;
	buf.resize(m_maxRow * 2);
	d_p = RAW_PTR(buf);
	d_omega = RAW_PTR(buf) + m_maxRow;

	omega_vec.resize(m_maxRow);
	p_vec.resize(m_maxRow);
	r_vec.resize(m_maxRow);
	z_vec.resize(m_maxRow);
	delta_x_vec.resize(m_maxRow, 0);
	precond_vec.resize(m_maxRow);
}

void PcgLinearSolver::init(int row) {
	m_row = row;
	if (m_row > m_maxRow)
	{
		std::cout << "node num exceed max node num" << std::endl;
		std::exit(0);
	}
}

void PcgLinearSolver::solveCPUOpt(thrust::device_vector<float>& d_x,
                                  int* d_ia, int* d_ja, float* d_a, int num_nnz,
                                  thrust::device_vector<float>& d_b, thrust::device_vector<float>& preconditioner,
                                  int max_iter)
{
	std::cout << "row1: " << m_row << std::endl;

	//innoreal::InnoRealTimer timer;
	//timer.TimeStart();
	float *p1, *p2, *p3;
	checkCudaErrors(cudaMemcpy(r_vec.data(), RAW_PTR(d_b), m_row * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(precond_vec.data(), RAW_PTR(preconditioner), m_row * sizeof(float), cudaMemcpyDeviceToHost));
	//timer.TimeEnd();

	//timer.TimeStart();
	p1 = p_vec.data(), p2 = precond_vec.data(), p3 = r_vec.data();
    float factorLM = 1.0f;
	zr_dot = 0.0f;
	for (int i = 0; i < m_row; ++i)
	{
		*p1 = (*(p2++))*(*p3);
		zr_dot += (*(p1++))*(*(p3++));
	}	
	spmv_wrapper.ia = d_ia;
	spmv_wrapper.ja = d_ja;
	spmv_wrapper.a = d_a;	
	spmv_wrapper.res = d_omega;

	memset(delta_x_vec.data(), 0, m_row * sizeof(float));
	//timer.TimeEnd();
	//std::cout << "t2: " << timer.TimeGap_in_ms() << std::endl;
	double r_val;
	int iterCnt = 0;
	//timer.TimeStart();
	for (int k = 0; k < max_iter; k++)
	{
		zr_dot_old = zr_dot;
		//std::cout << "zr_dot" << zr_dot << std::endl;
		checkCudaErrors(cudaMemcpy(d_p, p_vec.data(), m_row * sizeof(float), cudaMemcpyHostToDevice));

		spmv_wrapper.x = d_p;
		SparseMvKernel << <m_row, 64 >> > (spmv_wrapper);
		checkCudaErrors(cudaDeviceSynchronize());
#if 0
		cusparseScsrmv(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_row, m_row, num_nnz, &m_floatone, m_descr,
			d_a, d_ia, d_ja, RAW_PTR(d_p), &m_floatzero, RAW_PTR(d_omega));
#endif

		checkCudaErrors(cudaMemcpy(omega_vec.data(), d_omega, m_row * sizeof(float), cudaMemcpyDeviceToHost));
#if 0
        // LM factor
        p1 = p_vec.data(), p2 = omega_vec.data();
        for (int i = 0; i < m_row; ++i)
        {
            *(p2++) += *(p1++)*factorLM;
        }
#endif
		
		p1 = p_vec.data(), p2 = omega_vec.data();
		alpha = 0.0f;
		for (int i = 0; i < m_row; ++i)
		{
			alpha += (*(p1++))*(*(p2++));
		}
		alpha = zr_dot / alpha;

		p1 = delta_x_vec.data(), p2 = p_vec.data();
		for (int i = 0; i < m_row; ++i)
		{
			*(p1++) += alpha * (*(p2++));
		}

		p1 = r_vec.data(), p2 = omega_vec.data();
		r_val = 0.0;
		for (int i = 0; i < m_row; ++i)
		{
			r_val += (*p1)*(*p1);
			*(p1++) -= alpha * (*(p2++));
		}

		//std::cout << "r_val: " << r_val << std::endl;
#if 1
		//if (r_val < 1.0f)
		if (r_val < 0.1f)
			break;
#endif

		p1 = z_vec.data(), p2 = precond_vec.data(), p3 = r_vec.data();
		for (int i = 0; i < m_row; ++i)
		{
			*(p1++) = (*(p2++)) * (*(p3++));
		}

		p1 = r_vec.data(), p2 = z_vec.data();
		zr_dot = 0.0f;
		for (int i = 0; i < m_row; ++i)
		{
			zr_dot += (*(p1++))*(*(p2++));
		}

		beta = zr_dot / zr_dot_old;
		p1 = p_vec.data(), p2 = z_vec.data();
		for (int i = 0; i < m_row; ++i)
		{
			*(p1) = (*(p2++)) + beta * (*p1);
            ++p1;
		}
		++iterCnt;
	}
	//timer.TimeEnd();
	//std::cout << "t: " << timer.TimeGap_in_ms() << std::endl;
	//printf("PCG iter count: %d\n", iterCnt);
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_x), delta_x_vec.data(), m_row * sizeof(float), cudaMemcpyHostToDevice));
}
#endif
