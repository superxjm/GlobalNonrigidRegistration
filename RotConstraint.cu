#include "RotConstraint.h"

#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "InputData.h"
#include "GNSolver.h"
#include "Helpers/UtilsMath.h"

bool RotConstraint::init(GNSolver* gnSolver, float weight)
{
	assert(gnSolver);
	m_gnSolver = gnSolver;
	m_inputData = gnSolver->m_inputData;
	setWeight(weight);
	return true;
}

bool RotConstraint::init()
{
	return true;
}

__global__ void bKernelRot(float* dB,
                           int nodeNum,
                           float3* dVars,
                           float weight)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= nodeNum)
	{
		return;
	}
	int node_seri;
	node_seri = idx * 12;
	int currentIdx = idx * 6;
	float3* R = dVars + 4 * idx;

	dB[currentIdx + 0] = weight * dot(R[0], R[1]);
	dB[currentIdx + 1] = weight * dot(R[0], R[2]);
	dB[currentIdx + 2] = weight * dot(R[1], R[2]);
	dB[currentIdx + 3] = weight * (dot(R[0], R[0]) - 1);
	dB[currentIdx + 4] = weight * (dot(R[1], R[1]) - 1);
	dB[currentIdx + 5] = weight * (dot(R[2], R[2]) - 1);
}

void RotConstraint::b(float3* dVars)
{
	int nodeNum = m_inputData->m_source.m_nodeNum;
	m_dB.resize(nodeNum * 6);

	int block = 512;
	int grid = (block + nodeNum - 1) / block;
	bKernelRot << <grid, block >> >(RAW_PTR(m_dB),
	                                   nodeNum,
	                                   dVars,
	                                   m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void DirectiveJTJAndJTbKernelRot(float* dJTJ_a,
                                            int* dJTJ_ia,
                                            float* dJTb,
                                            float* dB,
                                            int* dNnzPre,
                                            int nodeNum,
                                            float* dVars,
                                            float weight)
{
	__shared__ float s_res[81];
	__shared__ float s_x[12];
	__shared__ float s_J[9 * 6];
	__shared__ float s_JTb[12];
	if (threadIdx.x < 81) {
		s_res[threadIdx.x] = 0.0f;
	}
	if (threadIdx.x < 9 * 6) {
		s_J[threadIdx.x] = 0.0f;
	}
	if (threadIdx.x < 12) {
		s_x[threadIdx.x] = dVars[blockIdx.x * 12 + threadIdx.x];
		s_JTb[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		// c1c2
		s_J[9 * 0 + 0] = s_x[3];
		s_J[9 * 0 + 1] = s_x[4];
		s_J[9 * 0 + 2] = s_x[5];
		s_J[9 * 0 + 3] = s_x[0];
		s_J[9 * 0 + 4] = s_x[1];
		s_J[9 * 0 + 5] = s_x[2];
		// c1c3
		s_J[9 * 1 + 0] = s_x[6];
		s_J[9 * 1 + 1] = s_x[7];
		s_J[9 * 1 + 2] = s_x[8];
		s_J[9 * 1 + 6] = s_x[0];
		s_J[9 * 1 + 7] = s_x[1];
		s_J[9 * 1 + 8] = s_x[2];
		// c2c3
		s_J[9 * 2 + 3] = s_x[6];
		s_J[9 * 2 + 4] = s_x[7];
		s_J[9 * 2 + 5] = s_x[8];
		s_J[9 * 2 + 6] = s_x[3];
		s_J[9 * 2 + 7] = s_x[4];
		s_J[9 * 2 + 8] = s_x[5];
		// c1c1
		s_J[9 * 3 + 0] = 2 * s_x[0];
		s_J[9 * 3 + 1] = 2 * s_x[1];
		s_J[9 * 3 + 2] = 2 * s_x[2];
		// c2c2
		s_J[9 * 4 + 3] = 2 * s_x[3];
		s_J[9 * 4 + 4] = 2 * s_x[4];
		s_J[9 * 4 + 5] = 2 * s_x[5];
		// c3c3
		s_J[9 * 5 + 6] = 2 * s_x[6];
		s_J[9 * 5 + 7] = 2 * s_x[7];
		s_J[9 * 5 + 8] = 2 * s_x[8];
	}
	__syncthreads();

	int row_res = threadIdx.x / 9;
	int col_res = threadIdx.x % 9;
	// reduction
	if (threadIdx.x < 81) {
		float squ_weight = weight * weight;
#pragma unroll
		for (int iter = 0; iter < 6; iter++) {
			s_res[threadIdx.x] += s_J[iter * 9 + row_res] * s_J[iter * 9 + col_res] * squ_weight;
		}
	}
	__syncthreads();

	// write back
	int start_pos;
	if (threadIdx.x < 81) {
		start_pos = dNnzPre[blockIdx.x * nodeNum + blockIdx.x] * 12;
		dJTJ_a[dJTJ_ia[blockIdx.x * 12 + row_res] + start_pos + col_res] += s_res[row_res * 9 + col_res];

	}

	// calculate JTb
	__syncthreads();
	if (threadIdx.x < 9) {
		start_pos = blockIdx.x * 6;

#pragma unroll
		for (int iter = 0; iter < 6; iter++) {
			s_JTb[threadIdx.x] += s_J[iter * 9 + threadIdx.x] * dB[start_pos + iter] * weight;
		}
	}
	__syncthreads();
	// write back
	if (threadIdx.x < 9) {
		int save_start_pos = blockIdx.x * 12;
		dJTb[save_start_pos + threadIdx.x] -= s_JTb[threadIdx.x];
	}
}

void RotConstraint::getJTJAndJTb(float* dJTJ_a, int* dJTJ_ia, thrust::device_vector<float>& dJTb, float3* dVars)
{
	b(dVars);

	int nodeNum = m_inputData->m_source.m_nodeNum;
	int block = 96;
	int grid = nodeNum;
	DirectiveJTJAndJTbKernelRot << <grid, block >> >(dJTJ_a,
	                                                 dJTJ_ia,
	                                                 RAW_PTR(dJTb),
	                                                 RAW_PTR(m_dB),
	                                                 RAW_PTR(m_inputData->m_Iij.m_dNnzPre),
	                                                 nodeNum,
	                                                 reinterpret_cast<float*>(dVars),
	                                                 m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

void RotConstraint::directiveJTJ(float* JTJ_a,
                                    int* JTJ_ia)
{
	
}

void RotConstraint::directiveJTb(thrust::device_vector<float> &JTb)
{
	
}
