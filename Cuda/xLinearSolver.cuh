#ifndef __PCG_SOLVER_H__
#define __PCG_SOLVER_H__

#include <thrust/device_vector.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cusolverSp.h>

class xLinearSolver {
public:
	xLinearSolver() : m_cublasHandle(NULL), m_cusparseHandle(NULL), m_cusolverHandle(NULL), m_cusparseDescr(NULL) {}
	~xLinearSolver() { }
	void init();
	void solvePCG(int *d_ia_csr, int *d_ja_csr, float *d_a_csr,
		int *d_ja_csc, int *d_ia_csc, float *d_a_csc,
		float *d_precond_terms,
		int row_J, int col_J, int num_nnz,
		float *d_b, float *d_x,
		int max_iter);
#if 0
	void solvePCGJTJ(int *d_ia, int *d_ja, float *d_a,
		int *d_ia_incomchol, int *d_ja_incomchol, float *d_a_incomchol,
		int row_JTJ, int col_JTJ, int num_nnz,
		float *d_precond_terms,
		float *d_b, float *d_x,
		int max_iter);
#endif
	void solvePCGJTJ(int *d_ia, int *d_ja, float *d_a,
		int row_JTJ, int col_JTJ, int num_nnz,
		float *d_precond_terms,
		float *d_b, float *d_x,
		int max_iter);
	void solveChol(int *d_ia_csr, int *d_ja_csr, float *d_a_csr,
		int row_J, int col_J, int num_nnz,
		float *d_b, float *d_x);
	void solvePCGJTJIncompleteChol(int *d_row, int *d_col, float *d_val,
		int *d_ia_incomchol, int *d_ja_incomchol, float *d_a_incomchol,
		int N, int col_JTJ, int nz,
		float *d_precond_terms,
		float *rhs, float *d_x,
		int max_iter);

private:
	void calcPreCondTerms(float *preCondTerms, int *d_ja_csr, float *d_a_csr, int col_J, int num_nnz);
	void calcPreCondTermsJTJ(float *preCondTerms, int *ia, int *ja, float *a, int rowJTJ);
	void incompleteCholesky(int *ia, int *ja, float *a, int rowJTJ, int nnz);

private:
	const float floatone_ = 1.0f, neg_floatone_ = -1.0f, floatzero_ = 0.0f;
	float alpha_ = 0.0f, beta_ = 0.0f, zr_dot_old = 0.0f, zr_dot = 0.0f;

	thrust::device_vector<float> d_p_, d_Jp_, d_JtJp_, d_r_, d_z_, d_Jtb_;
	thrust::device_vector<float> d_LMp_, d_Jx_, d_JtJx_;

	std::vector<float> JtJp_vec, p_vec, r_vec, z_vec, delta_x_vec, precond_vec;

	cublasHandle_t m_cublasHandle;
	cusparseHandle_t m_cusparseHandle;
	cusolverSpHandle_t m_cusolverHandle;
	cusparseMatDescr_t m_cusparseDescr;

	// for incomplete chol
	const int max_iter = 1000;
	int k, M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
	int qatest = 0;
	const float tol = 1e-12f;
	float *x, *rhs;
	float r0, r1, alpha, beta;
	float *d_x;
	float *d_zm1, *d_zm2, *d_rm2;
	float *d_r, *d_p, *d_omega, *d_y;
	float *val = NULL;
	float *d_valsILU0;
	float *valsILU0;
	float rsum, diff, err = 0.0;
	float qaerr1, qaerr2 = 0.0;
	float dot, numerator, denominator, nalpha;
	const float floatone = 1.0;
	const float floatzero = 0.0;
	cusparseSolveAnalysisInfo_t infoA = 0;
	cusparseSolveAnalysisInfo_t info_u;
	cusparseMatDescr_t descrU = 0;
	cusparseMatDescr_t descrL = 0;
};

#endif
