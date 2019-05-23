#include "Cuda/xLinearSolver.cuh"

#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <helper_cuda.h>

#include "Helpers/xUtils.h"
#include "Helpers/InnorealTimer.hpp"

#define PRE_TRANS_J 1

__global__ void CalcPreCondTermsKernel(float *preCondTerms, int *d_ja_csr, float *d_a_csr, int num_nnz)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= num_nnz)
	{
		return;
	}

	float val = d_a_csr[idx];
	atomicAdd(preCondTerms + d_ja_csr[idx], val * val);
}

__global__ void InversePreCondTerms(float *preCondTerms, int col_J)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if (col >= col_J)
	{
		return;
	}

	preCondTerms[col] = 1.0f / preCondTerms[col];
}

// y = a*x + y;
struct saxpy_functor : public thrust::binary_function<float, float, float>
{
	const float a;

	saxpy_functor(float _a) : a(_a) {}

	__host__ __device__
		float operator()(const float& x, const float& y) const
	{
		return a * x + y;
	}
};

// y = x + a * y;
struct sxpay_functor : public thrust::binary_function<float, float, float>
{
	const float a;

	sxpay_functor(float _a) : a(_a) {}

	__host__ __device__
		float operator()(const float& x, const float& y) const
	{
		return x + a * y;
	}
};

void xLinearSolver::calcPreCondTerms(float *preCondTerms, int *d_ja_csr, float *d_a_csr, int col_J, int num_nnz)
{
	int block = 256;
	int grid = DivUp(num_nnz, block);
	CalcPreCondTermsKernel << <grid, block >> > (preCondTerms, d_ja_csr, d_a_csr, num_nnz);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	grid = DivUp(col_J, block);
	InversePreCondTerms << <grid, block >> > (preCondTerms, col_J);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

void xLinearSolver::init()
{
	cusparseCreate(&m_cusparseHandle);
	cusparseCreateMatDescr(&m_cusparseDescr);
	cusparseSetMatIndexBase(m_cusparseDescr, CUSPARSE_INDEX_BASE_ZERO);
	//cusparseSetMatType(m_cusparseDescr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
	cusparseSetMatType(m_cusparseDescr, CUSPARSE_MATRIX_TYPE_GENERAL);

	cublasCreate_v2(&m_cublasHandle);
	//cusolverSpCreate(&cusolverHandle_);

	/* create the analysis info object for the A matrix */
	cusparseStatus_t cusparseStatus = cusparseCreateSolveAnalysisInfo(&infoA);
	checkCudaErrors(cusparseStatus);

	cusparseCreateSolveAnalysisInfo(&info_u);

	cusparseStatus = cusparseCreateMatDescr(&descrL);
	cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseStatus = cusparseCreateMatDescr(&descrU);
	cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseCreateSolveAnalysisInfo(&info_u);

	cusparseMatDescr_t descrU = 0;
	cusparseStatus = cusparseCreateMatDescr(&descrU);
	cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);

#if 0
// step 1: create a descriptor which contains
	// - matrix M is base-1
	// - matrix L is base-1
	// - matrix L is lower triangular
	// - matrix L has non-unit diagonal 
	cusparseCreateMatDescr(&descr_M);
	cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

	cusparseCreateMatDescr(&descr_L);
	cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

	// step 2: create a empty info structure
	// we need one info for csric02 and two info's for csrsv2
	cusparseCreateCsric02Info(&info_M);
	cusparseCreateCsrsv2Info(&info_L);
	cusparseCreateCsrsv2Info(&info_Lt);
#endif

	//cusparseCreateSolveAnalysisInfo(&inforRt);
	//cusparseCreateSolveAnalysisInfo(&inforR);
	//cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, n, descrR, valR, csrRowPtrR, csrColIndR, inforRt);
	//cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, descrR, valR, csrRowPtrR, csrColIndR, inforR);
}

void xLinearSolver::solvePCG(int *d_ia_csr, int *d_ja_csr, float *d_a_csr,
	int *d_ia_csc, int *d_ja_csc, float *d_a_csc,
	float *d_precond_terms,
	int row_J, int col_J, int num_nnz,
	float *d_b, float *d_x,
	int max_iter)
{
	d_p_.resize(col_J);
	d_r_.resize(col_J);
	d_z_.resize(col_J);
	d_Jp_.resize(row_J);
	d_JtJp_.resize(col_J);
	d_Jtb_.resize(col_J);

	d_LMp_.clear();
	d_LMp_.resize(col_J, 0.0f);
	d_Jx_.resize(row_J);
	d_JtJx_.resize(col_J);

#if PRE_TRANS_J
	cusparseScsr2csc(m_cusparseHandle,
		row_J, col_J, num_nnz,
		d_a_csr, d_ia_csr, d_ja_csr,
		d_a_csc, d_ja_csc, d_ia_csc,
		CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
#endif

	checkCudaErrors(cudaMemset(d_precond_terms, 0, sizeof(float) * col_J));
	calcPreCondTerms(d_precond_terms, d_ja_csr, d_a_csr, col_J, num_nnz);

	thrust::device_ptr<float> d_x_dev_ptr(d_x);
	thrust::device_ptr<float> d_b_dev_ptr(d_b);
	thrust::fill(d_x_dev_ptr, d_x_dev_ptr + col_J, 0.0);

	checkCudaErrors(cusparseScsrmv_mp(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, col_J, row_J,
		num_nnz, &floatone_, m_cusparseDescr,
		d_a_csc, d_ia_csc, d_ja_csc, d_b, &floatzero_, RAW_PTR(d_r_)));
	d_Jtb_ = d_r_;

	thrust::multiplies<float> op_multiplies;
	thrust::plus<float> op_plus;
	thrust::copy(d_r_.begin(), d_r_.end(), d_p_.begin());
	thrust::transform(thrust::device_ptr<float>(d_precond_terms), thrust::device_ptr<float>(d_precond_terms) + col_J,
		d_r_.begin(), d_p_.begin(), op_multiplies);
	zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_p_.begin(), 0.0f);

	float temp = 0.0f, energy0, energy1;
	//energy0 = thrust::inner_product(d_r_.begin(), d_r_.end(), d_r_.begin(), 0.0f);
	int iter = 0;
	while (iter < max_iter)
	{
		zr_dot_old = zr_dot;
		checkCudaErrors(cusparseScsrmv_mp(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, row_J, col_J,
			num_nnz, &floatone_, m_cusparseDescr,
			d_a_csr, d_ia_csr, d_ja_csr, RAW_PTR(d_p_), &floatzero_, RAW_PTR(d_Jp_)));

#if PRE_TRANS_J
		checkCudaErrors(cusparseScsrmv_mp(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, col_J, row_J,
			num_nnz, &floatone_, m_cusparseDescr,
			d_a_csc, d_ia_csc, d_ja_csc, RAW_PTR(d_Jp_), &floatzero_, RAW_PTR(d_JtJp_)));
#else
		checkCudaErrors(cusparseScsrmv(m_cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, row_J, col_J,
			num_nnz, &floatone_, m_cusparseDescr,
			d_a_csr, d_ia_csr, d_ja_csr, RAW_PTR(d_Jp_), &floatzero_, RAW_PTR(d_JtJp_)));
#endif
#if 1
		//for (int i = 0; i < d_LMp_.size(); ++i)//i += 12)
		{
			//d_LMp_[i] = 0.000001;// 0.00010;
			//d_LMp_[i + 9] = 0.00010;
			//d_LMp_[i + 10] = 0.00010;
			//d_LMp_[i + 11] = 0.00010;
		}
		//thrust::transform(d_LMp_.begin(), d_LMp_.end(), d_JtJp_.begin(), d_JtJp_.begin(), sxpay_functor(1.0));
#endif

		temp = thrust::inner_product(d_p_.begin(), d_p_.end(), d_JtJp_.begin(), 0.0f);
		assert(temp > MYEPS || temp < -MYEPS);
		alpha_ = zr_dot / temp;

		thrust::transform(d_p_.begin(), d_p_.end(), d_x_dev_ptr, d_x_dev_ptr, saxpy_functor(alpha_));

#if 1
#if 0
		checkCudaErrors(cusparseScsrmv_mp(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, row_J, col_J,
			num_nnz, &floatone_, m_cusparseDescr,
			d_a_csr, d_ia_csr, d_ja_csr, RAW_PTR(d_x_dev_ptr), &floatzero_, RAW_PTR(d_Jx_)));
		checkCudaErrors(cusparseScsrmv_mp(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, col_J, row_J,
			num_nnz, &floatone_, m_cusparseDescr,
			d_a_csc, d_ia_csc, d_ja_csc, RAW_PTR(d_Jx_), &floatzero_, RAW_PTR(d_JtJx_)));

		float val1 = thrust::inner_product(d_JtJx_.begin(), d_JtJx_.end(), d_x_dev_ptr, 0.0f);
		float val2 = thrust::inner_product(d_Jtb_.begin(), d_Jtb_.end(), d_x_dev_ptr, 0.0f);
		std::cout << "energy " << iter << " : " << val1 * 0.5f - val2 << std::endl;
#endif
		//energy1 = thrust::inner_product(d_r_.begin(), d_r_.end(), d_r_.begin(), 0.0f);
		//std::cout << "resi2: " << iter << " : " << energy1 << std::endl;
		//if (energy1 < 1.0)
		//break;
		//std::cout << "energy " << iter << " : " << energy1 << std::endl;
		//if (energy1 < 1.0f)
		//break;
		//if (energy1 > energy0)
		//{
		//break;
		//}
		//energy0 = energy1;
#endif
		thrust::transform(d_JtJp_.begin(), d_JtJp_.end(), d_r_.begin(), d_r_.begin(), saxpy_functor(-alpha_));

		thrust::transform(thrust::device_ptr<float>(d_precond_terms), thrust::device_ptr<float>(d_precond_terms) + col_J,
			d_r_.begin(), d_z_.begin(), op_multiplies);
		zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_z_.begin(), 0.0f);
		assert(zr_dot_old > MYEPS || zr_dot_old < -MYEPS);
		beta_ = zr_dot / zr_dot_old;
		thrust::transform(d_z_.begin(), d_z_.end(), d_p_.begin(), d_p_.begin(), sxpay_functor(beta_));

		++iter;
	}
	//std::exit(0);
}

__global__ void CalcPreCondTermsJTJKernel(float *preCondTerms, int *ia, int *ja, float *a, int rowJTJ)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= rowJTJ)
	{
		return;
	}

	for (int i = ia[idx]; i < ia[idx + 1]; ++i)
	{
		if (idx == ja[i])
		{
			preCondTerms[idx] = 1.0 / a[i];
			//printf("precond: %f\n", a[i]);
			return;
		}
	}
}

void xLinearSolver::calcPreCondTermsJTJ(float *preCondTerms, int *ia, int *ja, float *a, int rowJTJ)
{
	int block = 256;
	int grid = DivUp(rowJTJ, block);
	CalcPreCondTermsJTJKernel << <grid, block >> > (preCondTerms, ia, ja, a, rowJTJ);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//grid = DivUp(rowJTJ, block);
	//InversePreCondTerms << <grid, block >> > (preCondTerms, rowJTJ);
	//checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cudaGetLastError());
}

void xLinearSolver::incompleteCholesky(int *ia, int *ja, float *a, int rowJTJ, int nnz)
{
#if 0
	// step 3: query how much memory used in csric02 and csrsv2, and allocate the buffer
	cusparseScsric02_bufferSize(m_cusparseHandle, rowJTJ, nnz,
		descr_M, a, ia, ja, info_M, &bufferSize_M);
	cusparseScsrsv2_bufferSize(m_cusparseHandle, trans_L, rowJTJ, nnz,
		descr_L, a, ia, ja, info_L, &pBufferSize_L);
	cusparseScsrsv2_bufferSize(m_cusparseHandle, trans_Lt, rowJTJ, nnz,
		descr_L, a, ia, ja, info_Lt, &pBufferSize_Lt);

	pBufferSize = MAX(bufferSize_M, MAX(pBufferSize_L, pBufferSize_Lt));

	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
	cudaMalloc((void**)&pBuffer, pBufferSize);

	// step 4: perform analysis of incomplete Cholesky on M
	//         perform analysis of triangular solve on L
	//         perform analysis of triangular solve on L' 
	// The lower triangular part of M has the same sparsity pattern as L, so  
	// we can do analysis of csric02 and csrsv2 simultaneously.

	cusparseScsric02_analysis(m_cusparseHandle, rowJTJ, nnz, descr_M,
		a, ia, ja, info_M,
		policy_M, pBuffer);
	cusparseStatus_t status = cusparseXcsric02_zeroPivot(m_cusparseHandle, info_M, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
		printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
	}

	cusparseScsrsv2_analysis(m_cusparseHandle, trans_L, rowJTJ, nnz, descr_L,
		a, ia, ja,
		info_L, policy_L, pBuffer);

	cusparseScsrsv2_analysis(m_cusparseHandle, trans_Lt, rowJTJ, nnz, descr_L,
		a, ia, ja,
		info_Lt, policy_Lt, pBuffer);

	// step 5: M = L * L'
	cusparseScsric02(m_cusparseHandle, rowJTJ, nnz, descr_M,
		a, ia, ja, info_M, policy_M, pBuffer);
	status = cusparseXcsric02_zeroPivot(m_cusparseHandle, info_M, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
		printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
	}
#endif
}

void xLinearSolver::solvePCGJTJIncompleteChol(int *d_row, int *d_col, float *d_val,
	int *d_ia_incomchol, int *d_ja_incomchol, float *d_a_incomchol,
	int N, int col_JTJ, int nz,
	float *d_precond_terms,
	float *rhs, float *d_x,
	int max_iter)
{
	thrust::device_ptr<float> d_x_dev_ptr(d_x);
	thrust::fill(d_x_dev_ptr, d_x_dev_ptr + N, 0.0);
	checkCudaErrors(cudaMalloc((void **)&d_y, N * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_omega, N * sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyDeviceToDevice));

	/* Preconditioned Conjugate Gradient using ILU.
	--------------------------------------------
	Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

	int nzILU0 = 2 * N - 1;

	float *d_valsILU0;
	checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nz * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_zm1, (N) * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_zm2, (N) * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_rm2, (N) * sizeof(float)));	

	cusparseStatus_t cusparseStatus;
	/* Perform the analysis for the Non-Transpose case */
	cusparseStatus = cusparseScsrsv_analysis(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, nz, m_cusparseDescr, d_val, d_row, d_col, infoA);

	checkCudaErrors(cusparseStatus);

	/* Copy A data to ILU0 vals as input*/
	cudaMemcpy(d_valsILU0, d_val, nz * sizeof(float), cudaMemcpyDeviceToDevice);

	/* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */
	cusparseStatus = cusparseScsrilu0(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, m_cusparseDescr, d_valsILU0, d_row, d_col, infoA);

	checkCudaErrors(cusparseStatus);

	/* Create info objects for the ILU0 preconditioner */	
	cusparseStatus = cusparseScsrsv_analysis(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, descrU, d_val, d_row, d_col, info_u);

	/* reset the initial guess of the solution to zero */

	//checkCudaErrors(cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));

	k = 0;
	//cublasSdot(m_cublasHandle, N, d_r, 1, d_r, 1, &r1);

	//thrust::device_ptr<float> d_zm1_dev_ptr(d_zm1);
	//thrust::device_ptr<float> d_p_dev_ptr(d_p);
	//thrust::device_ptr<float> d_r_dev_ptr(d_r);
	//thrust::device_ptr<float> d_omega_dev_ptr(d_omega);

	innoreal::InnoRealTimer timer;

	while (k <= max_iter)
	{
#if 1
		timer.TimeStart();
		// Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
		cusparseStatus = cusparseScsrsv_solve(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrL,
			d_valsILU0, d_row, d_col, infoA, d_r, d_y);
		checkCudaErrors(cusparseStatus);

		// Back Substitution
		cusparseStatus = cusparseScsrsv_solve(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, &floatone, descrU,
			d_valsILU0, d_row, d_col, info_u, d_y, d_zm1);
		checkCudaErrors(cusparseStatus);
		timer.TimeEnd();
		std::cout << "incomplete chol time: " << timer.TimeGap_in_ms() << std::endl;
#endif
		//cublasScopy(m_cublasHandle, N, d_r, 1, d_zm1, 1);

		k++;

		if (k == 1)
		{
			cublasScopy(m_cublasHandle, N, d_zm1, 1, d_p, 1);
		}
		else
		{
			cublasSdot(m_cublasHandle, N, d_r, 1, d_zm1, 1, &numerator);
			cublasSdot(m_cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator);
			beta = numerator / denominator;
			cublasSscal(m_cublasHandle, N, &beta, d_p, 1);
			cublasSaxpy(m_cublasHandle, N, &floatone, d_zm1, 1, d_p, 1);
			//thrust::transform(d_zm1_dev_ptr, d_zm1_dev_ptr+N, d_p_dev_ptr, d_p_dev_ptr, saxpy_functor(floatone));
		}

		cusparseScsrmv(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nzILU0, &floatone, descrU, d_val, d_row, d_col, d_p, &floatzero, d_omega);
		cublasSdot(m_cublasHandle, N, d_r, 1, d_zm1, 1, &numerator);
		cublasSdot(m_cublasHandle, N, d_p, 1, d_omega, 1, &denominator);
		alpha = numerator / denominator;
		cublasSaxpy(m_cublasHandle, N, &alpha, d_p, 1, d_x, 1);
		//thrust::transform(d_p_dev_ptr, d_p_dev_ptr + N, d_x_dev_ptr, d_x_dev_ptr, saxpy_functor(alpha));
		cublasScopy(m_cublasHandle, N, d_r, 1, d_rm2, 1);
		cublasScopy(m_cublasHandle, N, d_zm1, 1, d_zm2, 1);
		nalpha = -alpha;
		cublasSaxpy(m_cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
		//thrust::transform(d_omega_dev_ptr, d_omega_dev_ptr + N, d_r_dev_ptr, d_r_dev_ptr, saxpy_functor(nalpha));
		//cublasSdot(m_cublasHandle, N, d_r, 1, d_r, 1, &r1);
	}

#if 0
	x = (float *)malloc(N * sizeof(float));

	cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* check result */
	err = 0.0;

	for (int i = 0; i < N; i++)
	{
		std::cout << x[i] << ", ";
	}
	std::cout << std::endl;
	std::cout << "----------------" << std::endl;
#endif
}

__global__ void myDot64(float *sum, float *vec1, float *vec2, int size) 
{
	volatile __shared__ float sharedBuf[64];

	float val = 0;

	int step = (size + 64 - 1) / 64;
	int start = threadIdx.x * step, end = start + step;
	if (end > size) {
		end = size;
	}
	for (int i = start; i < end; ++i) {
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

__global__ void myReduce64(float *sum, float *vec, int size) 
{
	volatile __shared__ float sharedBuf[64];

	float val = 0;

	int step = (size + 64 - 1) / 64;
	int start = threadIdx.x * step, end = start + step;
	if (end > size) {
		end = size;
	}
	for (int i = start; i < end; ++i) {
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

void xLinearSolver::solvePCGJTJ(int *d_ia, int *d_ja, float *d_a,
	int row_JTJ, int col_JTJ, int num_nnz,
	float *d_precond_terms,
	float *d_b, float *d_x,
	int max_iter)
{
	d_JtJp_.resize(row_JTJ);
	d_p_.resize(row_JTJ);
	d_r_.resize(row_JTJ);
	d_z_.resize(row_JTJ);

	JtJp_vec.resize(row_JTJ);
	p_vec.resize(row_JTJ);
	r_vec.resize(row_JTJ);
	z_vec.resize(row_JTJ);
	delta_x_vec.clear();
	delta_x_vec.resize(row_JTJ, 0);
	precond_vec.resize(row_JTJ);

	calcPreCondTermsJTJ(d_precond_terms, d_ia, d_ja, d_a, row_JTJ);

	//thrust::device_ptr<float> d_x_dev_ptr(d_x);
	thrust::device_ptr<float> d_b_dev_ptr(d_b);
	thrust::device_ptr<float> precondDevPtr(d_precond_terms);
	//checkCudaErrors(cudaMemset(d_x, 0, row_JTJ * sizeof(float)));
	checkCudaErrors(cudaMemcpy(precond_vec.data(), d_precond_terms, row_JTJ * sizeof(float), cudaMemcpyDeviceToHost));
	
#if 1
	thrust::copy(d_b_dev_ptr, d_b_dev_ptr + row_JTJ, d_r_.begin());
	thrust::multiplies<float> op_multiplies;
	thrust::transform(precondDevPtr, precondDevPtr + row_JTJ, d_r_.begin(), d_p_.begin(), op_multiplies);

	zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_p_.begin(), 0.0f);
	checkCudaErrors(cudaMemcpy(r_vec.data(), RAW_PTR(d_r_), row_JTJ * sizeof(float), cudaMemcpyDeviceToHost));

	//innoreal::InnoRealTimer timer;

	float *p1, *p2, *p3;
	//std::cout << "-------" << std::endl;
	for (int k = 0; k < max_iter; k++)
	{
		zr_dot_old = zr_dot;
		cusparseScsrmv(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, row_JTJ, row_JTJ, num_nnz, &floatone_, m_cusparseDescr,
			d_a, d_ia, d_ja, RAW_PTR(d_p_), &floatzero_, RAW_PTR(d_JtJp_));

		checkCudaErrors(cudaMemcpy(p_vec.data(), RAW_PTR(d_p_), row_JTJ * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(JtJp_vec.data(), RAW_PTR(d_JtJp_), row_JTJ * sizeof(float), cudaMemcpyDeviceToHost));

		//std::cout << "zr_dot1: " << zr_dot << std::endl;
#if 0
		std::cout << "omega_vec:" << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << JtJp_vec[i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "p_vec:" << std::endl;
		for (int i = 0; i < 30; ++i)
		{
			std::cout << p_vec[i] << ", ";
		}
		std::cout << std::endl;
#endif
		
		p1 = p_vec.data(), p2 = JtJp_vec.data();
		alpha_ = 0.0f;
		for (int i = 0; i < row_JTJ; ++i)
		{
			alpha_ += (*(p1++))*(*(p2++));
		}
		//std::cout << "alpha_:" << alpha_ << std::endl;
		alpha_ = zr_dot / alpha_;

		p1 = delta_x_vec.data(), p2 = p_vec.data();
		for (int i = 0; i < row_JTJ; ++i)
		{
			*(p1++) += alpha_ * (*(p2++));
		}

		p1 = r_vec.data(), p2 = JtJp_vec.data();
		for (int i = 0; i < row_JTJ; ++i)
		{
			*(p1++) -= alpha_ * (*(p2++));
		}
		
		p1 = z_vec.data(), p2 = precond_vec.data(), p3 = r_vec.data();
		for (int i = 0; i < row_JTJ; ++i)
		{
			*(p1++) = (*(p2++)) * (*(p3++));
		}
		
		zr_dot = 0.0f;
		p1 = r_vec.data(), p2 = z_vec.data();
		for (int i = 0; i < row_JTJ; ++i)
		{
			zr_dot += (*(p1++))*(*(p2++));
		}

		beta_ = zr_dot / zr_dot_old;
		//thrust::transform(d_z_.begin(), d_z_.end(), d_p_.begin(), d_p_.begin(), sxpay_functor(beta_));
		p1 = p_vec.data(), p2 = z_vec.data();
		for (int i = 0; i < row_JTJ; ++i)
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

		checkCudaErrors(cudaMemcpy(RAW_PTR(d_p_), p_vec.data(), row_JTJ * sizeof(float), cudaMemcpyHostToDevice));
	}
	//std::cout << "-------" << std::endl;

	checkCudaErrors(cudaMemcpy(d_x, delta_x_vec.data(), row_JTJ * sizeof(float), cudaMemcpyHostToDevice));
#if 0
	x = (float *)malloc(row_JTJ * sizeof(float));

	checkCudaErrors(cudaMemcpy(x, d_x, row_JTJ * sizeof(float), cudaMemcpyDeviceToHost));
	/* check result */
	err = 0.0;

	for (int i = 0; i < row_JTJ; i++)
	{
		std::cout << x[i] << ", ";
	}
	std::cout << std::endl;
#endif
#endif
}


#if 0
void xLinearSolver::solvePCGJTJ(int *d_ia, int *d_ja, float *d_a,
	int *d_ia_incomchol, int *d_ja_incomchol, float *d_a_incomchol,
	int row_JTJ, int col_JTJ, int num_nnz,
	float *d_precond_terms,
	float *d_b, float *d_x,
	int max_iter)
{
	d_JtJp_.resize(row_JTJ);
	d_p_.resize(row_JTJ);
	d_r_.resize(row_JTJ);
	d_z_.resize(row_JTJ);

#if 0
	checkCudaErrors(cudaMemcpy(
		d_ia_incomchol,
		d_ia, sizeof(int) * (row_JTJ+1), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(
		d_ja_incomchol,
		d_ja, sizeof(int) * num_nnz, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(
		d_a_incomchol,
		d_a, sizeof(float) * num_nnz, cudaMemcpyDeviceToDevice));
	incompleteCholesky(d_ia_incomchol, d_ja_incomchol, d_a_incomchol, row_JTJ, num_nnz);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	int nnz_incomchol;
	checkCudaErrors(cudaMemcpy(
		&nnz_incomchol,
		d_ia_incomchol + row_JTJ, sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "nnz_incomchol: " << nnz_incomchol << std::endl;
	std::vector<int> iaVec(row_JTJ + 1);
	std::vector<int> jaVec(row_JTJ * row_JTJ);
	std::vector<float> aVec(row_JTJ * row_JTJ);
	checkCudaErrors(cudaMemcpy(
		iaVec.data(),
		d_ia_incomchol, sizeof(int) * (row_JTJ + 1), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(
		jaVec.data(),
		d_ja_incomchol, sizeof(int) * nnz_incomchol, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(
		aVec.data(),
		d_a_incomchol, sizeof(float) * nnz_incomchol, cudaMemcpyDeviceToHost));
	//for (int i = 0; i < iaVec.size(); ++i)
		//std::cout << iaVec[i] << ", ";
	std::cout << std::endl;
	std::cout << "-------------" << std::endl;
	//for (int i = 0; i < jaVec.size(); ++i)
		//std::cout << jaVec[i] << ", ";
	std::cout << std::endl;
	std::cout << "-------------" << std::endl;
	for (int i = 0; i < aVec.size(); ++i)
		std::cout << aVec[i] << ", ";
	std::cout << std::endl;
	std::cout << "-------------" << std::endl;
	std::exit(0);	
#endif

#if 1
	//checkCudaErrors(cudaMemset(d_precond_terms, 0, sizeof(float) * row_JTJ));
	calcPreCondTermsJTJ(d_precond_terms, d_ia, d_ja, d_a, row_JTJ);
#endif

	//printf("1\n");
	thrust::device_ptr<float> d_x_dev_ptr(d_x);
	thrust::device_ptr<float> d_b_dev_ptr(d_b);
	thrust::device_ptr<float> precondDevPtr(d_precond_terms);
	thrust::fill(d_x_dev_ptr, d_x_dev_ptr + row_JTJ, 0.0);

#if 0
	//printf("2\n");
	thrust::copy(d_b_dev_ptr, d_b_dev_ptr + row_JTJ, d_r_.begin());
	//thrust::multiplies<float> op_multiplies;
	//thrust::transform(precondDevPtr, precondDevPtr + row_JTJ, d_r_.begin(), d_p_.begin(), op_multiplies);
	cusparseScsrmv(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, row_JTJ, row_JTJ, nnz_incomchol, &floatone_, m_cusparseDescr,
		d_a_incomchol, d_ia_incomchol, d_ja_incomchol, RAW_PTR(d_r_), &floatzero_, RAW_PTR(d_z_));
	//thrust::copy(d_r_.begin(), d_r_.end(), d_p_.begin());
	zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_p_.begin(), 0.0f);
	//printf("3\n");
	for (int k = 0; k < max_iter; k++)
	{
		zr_dot_old = zr_dot;
		cusparseScsrmv(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, row_JTJ, row_JTJ, num_nnz, &floatone_, m_cusparseDescr,
			d_a, d_ia, d_ja, RAW_PTR(d_p_), &floatzero_, RAW_PTR(d_JtJp_));
		//printf("4\n");
		alpha_ = zr_dot / thrust::inner_product(d_p_.begin(), d_p_.end(), d_JtJp_.begin(), 0.0f);
		thrust::transform(d_p_.begin(), d_p_.end(), d_x_dev_ptr, d_x_dev_ptr, saxpy_functor(alpha_));
		thrust::transform(d_JtJp_.begin(), d_JtJp_.end(), d_r_.begin(), d_r_.begin(), saxpy_functor(-alpha_));
		//printf("5\n");
		//thrust::copy(d_r_.begin(), d_r_.end(), d_z_.begin());
		//thrust::transform(precondDevPtr, precondDevPtr + row_JTJ, d_r_.begin(), d_z_.begin(), op_multiplies);
		cusparseScsrmv(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, row_JTJ, row_JTJ, nnz_incomchol, &floatone_, m_cusparseDescr,
			d_a_incomchol, d_ia_incomchol, d_ja_incomchol, RAW_PTR(d_r_), &floatzero_, RAW_PTR(d_z_));
		zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_z_.begin(), 0.0f);
		beta_ = zr_dot / zr_dot_old;
		//printf("6\n");
		thrust::transform(d_z_.begin(), d_z_.end(), d_p_.begin(), d_p_.begin(), sxpay_functor(beta_));
	}
#endif

#if 1
	innoreal::InnoRealTimer time;

	//printf("2\n");
	thrust::copy(d_b_dev_ptr, d_b_dev_ptr + row_JTJ, d_r_.begin());
	thrust::multiplies<float> op_multiplies;
	thrust::transform(precondDevPtr, precondDevPtr + row_JTJ, d_r_.begin(), d_p_.begin(), op_multiplies);
	//thrust::copy(d_r_.begin(), d_r_.end(), d_p_.begin());
	zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_p_.begin(), 0.0f);
	//printf("3\n");
	//std::cout << "row_JTJ: " << d_r_.size() << std::endl;
	for (int k = 0; k < max_iter; k++) 
	{
		zr_dot_old = zr_dot;
		//time.TimeStart();
		cusparseScsrmv(m_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, row_JTJ, row_JTJ, num_nnz, &floatone_, m_cusparseDescr, 
			d_a, d_ia, d_ja, RAW_PTR(d_p_), &floatzero_, RAW_PTR(d_JtJp_));
		//time.TimeEnd();
		//std::cout << "time mv: " << time.TimeGap_in_ms() << std::endl;
		//printf("4\n");
		//float *resultDevice;
		//float result;
		//cudaMalloc(&resultDevice, sizeof(float));
		//thrust::device_vector<float> testVec(4000, 1);
		///int grid = DivUp(384, 64);
		//time.TimeStart();
		alpha_ = zr_dot / thrust::inner_product(d_p_.begin(), d_p_.end(), d_JtJp_.begin(), 0.0f);
#if 0
		thrust::host_vector<float> d_p_host = d_p_, d_JtJp_host = d_JtJp_;
		std::cout << d_p_host[0] << std::endl;
		std::cout << d_p_host[1] << std::endl;
		std::cout << d_p_host[2] << std::endl;
		std::cout << d_JtJp_host[0] << std::endl;
		std::cout << d_JtJp_host[1] << std::endl;
		std::cout << d_JtJp_host[2] << std::endl;
		std::cout << thrust::inner_product(d_p_.begin(), d_p_.end(), d_JtJp_.begin(), 0.0f) << std::endl;
		std::cout << alpha_ << std::endl;
#endif

		//cublasSdot(this->m_cublasHandle, d_p_.size(),
			//RAW_PTR(d_p_), 1,
			//RAW_PTR(d_JtJp_), 1,
			//&alpha_);
		//alpha_ = zr_dot / alpha_;
		//std::cout << alpha_ << std::endl;
		//myreduce64<<<1, 64>>> (resultDevice, RAW_PTR(testVec), testVec.size());
		//checkCudaErrors(cudaDeviceSynchronize());
		//checkCudaErrors(cudaGetLastError());

		//time.TimeEnd();
		//std::cout << "time trans1: " << time.TimeGap_in_ms() << std::endl;
		//cudaMemcpy(&result, resultDevice, sizeof(float), cudaMemcpyDeviceToHost);
		//std::cout << result << std::endl;
		//std::exit(0);
		//time.TimeStart();
		thrust::transform(d_p_.begin(), d_p_.end(), d_x_dev_ptr, d_x_dev_ptr, saxpy_functor(alpha_));
		//time.TimeEnd();
		//std::cout << "time trans2: " << time.TimeGap_in_ms() << std::endl;
		//time.TimeStart();
		thrust::transform(d_JtJp_.begin(), d_JtJp_.end(), d_r_.begin(), d_r_.begin(), saxpy_functor(-alpha_));
		//time.TimeEnd();
		//std::cout << "time trans3: " << time.TimeGap_in_ms() << std::endl;
		//time.TimeStart();
		//printf("5\n");
		//thrust::copy(d_r_.begin(), d_r_.end(), d_z_.begin());
		thrust::transform(precondDevPtr, precondDevPtr + row_JTJ, d_r_.begin(), d_z_.begin(), op_multiplies);
		//time.TimeEnd();
		//std::cout << "time trans4: " << time.TimeGap_in_ms() << std::endl;
		//time.TimeStart();
		zr_dot = thrust::inner_product(d_r_.begin(), d_r_.end(), d_z_.begin(), 0.0f);
		beta_ = zr_dot / zr_dot_old;
		//printf("6\n");
		thrust::transform(d_z_.begin(), d_z_.end(), d_p_.begin(), d_p_.begin(), sxpay_functor(beta_));
		//time.TimeEnd();
		//std::cout << "time inner: " << time.TimeGap_in_ms() << std::endl;
	}
#if 0
	x = (float *)malloc(row_JTJ * sizeof(float));

	checkCudaErrors(cudaMemcpy(x, d_x, row_JTJ * sizeof(float), cudaMemcpyDeviceToHost));
	/* check result */
	err = 0.0;

	for (int i = 0; i < row_JTJ; i++)
	{
		std::cout << x[i] << ", ";
	}
	std::cout << std::endl;
#endif
#endif
}
#endif

void xLinearSolver::solveChol(int *d_ia_csr, int *d_ja_csr, float *d_a_csr,
	int row_J, int col_J, int num_nnz,
	float *d_b, float *d_x)
{
#if 0
	int *d_ia = d_ia_csr;
	int *d_ja = d_ja_csr;
	float *d_a = d_a_csr;
	thrust::device_ptr<float> d_x_dev_ptr(d_x);
	thrust::fill(d_x_dev_ptr, d_x_dev_ptr + col_, 0.0);

	int baseJtJ, nnzJtJ; // nnzTotalDevHostPtr points to host memory 
	int *nnzTotalDevHostPtr = &nnzJtJ;
	int *csrRowPtrJtJ;
	int *csrColIndJtJ;
	float *csrValJtJ;
	cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST);
	cudaMalloc((void**)&csrRowPtrJtJ, sizeof(int)*(col_J + 1));
	checkCudaErrors(cusparseXcsrgemmNnz(cusparseHandle_,
		CUSPARSE_OPERATION_TRANSPOSE,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		col_J, col_J, row_J,
		descr_, num_nnz, d_ia, d_ja,
		descr_, num_nnz, d_ia, d_ja,
		descr_, csrRowPtrJtJ, nnzTotalDevHostPtr));
	if (NULL != nnzTotalDevHostPtr)
	{
		nnzJtJ = *nnzTotalDevHostPtr;
	}
	else
	{
		cudaMemcpy(&nnzJtJ, csrRowPtrJtJ + col_J, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseJtJ, csrRowPtrJtJ, sizeof(int), cudaMemcpyDeviceToHost);
		nnzJtJ -= baseJtJ;
	}
	cudaMalloc((void**)&csrColIndJtJ, sizeof(int)*nnzJtJ);
	cudaMalloc((void**)&csrValJtJ, sizeof(float)*nnzJtJ);
	checkCudaErrors(cusparseScsrgemm(cusparseHandle_,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		col_J, col_J, row_J,
		descr_, num_nnz, d_a, d_ia, d_ja,
		descr_, num_nnz, d_a, d_ia, d_ja,
		descr_, csrValJtJ, csrRowPtrJtJ, csrColIndJtJ));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	thrust::host_vector<int> csrValVec(col_J);
	for (int i = 0; i < col_J + 1; ++i)
	{
		csrValVec[i] = i;
	}
	int *csrRowPtrLMElem;
	cudaMalloc((void**)&csrRowPtrLMElem, sizeof(int)*(col_J + 1));
	thrust::device_ptr<int> csrRowPtrLMElem_dev_ptr(csrRowPtrLMElem);
	thrust::copy(csrValVec.begin(), csrValVec.begin() + col_J + 1, csrRowPtrLMElem_dev_ptr);
	int *csrColIndLMElem;
	cudaMalloc((void**)&csrColIndLMElem, sizeof(int)*(col_J));
	thrust::device_ptr<int> csrColIndLMElem_dev_ptr(csrColIndLMElem);
	thrust::copy(csrValVec.begin(), csrValVec.begin() + col_J, csrColIndLMElem_dev_ptr);
	float *csrValLMElem;
	cudaMalloc((void**)&csrValLMElem, sizeof(float)*(col_J));
	thrust::device_ptr<float> csrValLMElem_dev_ptr(csrValLMElem);
	thrust::fill(csrValLMElem_dev_ptr, csrValLMElem_dev_ptr + col_J, 0.0f);
	int nnzLMElem = col_J;

	int baseLMJtJ, nnzLMJtJ; // nnzTotalDevHostPtr points to host memory 
	nnzTotalDevHostPtr = &nnzLMJtJ;
	int *csrRowPtrLMJtJ;
	int *csrColIndLMJtJ;
	float *csrValLMJtJ;
	cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST);
	cudaMalloc((void**)&csrRowPtrLMJtJ, sizeof(int)*(col_J + 1));
	cusparseXcsrgeamNnz(cusparseHandle_,
		col_J, col_J,
		descr_, nnzJtJ, csrRowPtrJtJ, csrColIndJtJ,
		descr_, nnzLMElem, csrRowPtrLMElem, csrColIndLMElem,
		descr_, csrRowPtrLMJtJ, nnzTotalDevHostPtr);
	if (NULL != nnzTotalDevHostPtr)
	{
		nnzLMJtJ = *nnzTotalDevHostPtr;
	}
	else {
		cudaMemcpy(&nnzLMJtJ, csrRowPtrLMJtJ + col_J, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseLMJtJ, csrRowPtrLMJtJ, sizeof(int), cudaMemcpyDeviceToHost);
		nnzLMJtJ -= baseLMJtJ;
	}
	cudaMalloc((void**)&csrColIndLMJtJ, sizeof(int)*nnzLMJtJ);
	cudaMalloc((void**)&csrValLMJtJ, sizeof(float)*nnzLMJtJ);
	cusparseScsrgeam(cusparseHandle_, col_J, col_J,
		&floatone_,
		descr_, nnzJtJ, csrValJtJ, csrRowPtrJtJ, csrColIndJtJ,
		&floatone_,
		descr_, nnzLMElem, csrValLMElem, csrRowPtrLMElem, csrColIndLMElem,
		descr_, csrValLMJtJ, csrRowPtrLMJtJ, csrColIndLMJtJ);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	float *Jtb;
	cudaMalloc((void**)&Jtb, sizeof(float)*col_J);
	checkCudaErrors(cusparseScsrmv(cusparseHandle_, CUSPARSE_OPERATION_TRANSPOSE, row_, col_,
		num_nnz, &floatone_, descr_,
		d_a, d_ia, d_ja, d_b, &floatzero_, Jtb));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	int singularity;
	checkCudaErrors(cusolverSpScsrlsvchol(cusolverHandle_,
		col_J, nnzLMJtJ,
		descr_,
		csrValLMJtJ,
		csrRowPtrLMJtJ,
		csrColIndLMJtJ,
		Jtb, 1e-12,
		0, d_x, &singularity));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	//assert(singularity == -1);

	cudaFree(Jtb);

	cudaFree(csrRowPtrJtJ);
	cudaFree(csrColIndJtJ);
	cudaFree(csrValJtJ);
	cudaFree(csrRowPtrLMElem);
	cudaFree(csrColIndLMElem);
	cudaFree(csrValLMElem);
	cudaFree(csrRowPtrLMJtJ);
	cudaFree(csrColIndLMJtJ);
	cudaFree(csrValLMJtJ);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	std::cout << "sigu: " << singularity << std::endl;
	std::cout << "row_J: " << row_J << std::endl;
	std::cout << "col_J: " << col_J << std::endl;
#endif
}
