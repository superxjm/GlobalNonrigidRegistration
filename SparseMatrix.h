#pragma once

#include <thrust/device_vector.h>

class SparseMatrixCsrGpu
{
public:
	SparseMatrixCsrGpu() : m_nnz(0), m_row(0), m_col(0)
	{
	}

	SparseMatrixCsrGpu(int _row, int _col) : m_nnz(0)
	{
		m_row = _row;
		m_col = _col;
	}

	int m_row;
	int m_col;
	int m_nnz;

	thrust::device_vector<int> m_dIa; // row_ptr, size: row_ + 1
	thrust::device_vector<int> m_dJa; // col_ind, size: nnz_
	thrust::device_vector<float> m_dA; // val, size: nnz_
};

void AddVecVec(float* vec1, float* vec2, int length);

void AddVecVecSE3(float* vec1, float* vec2, int length);


