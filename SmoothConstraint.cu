#include "SmoothConstraint.h"

#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "InputData.h"
#include "GNSolver.h"
#include "Helpers/UtilsMath.h"

bool SmoothConstraint::init(GNSolver* gnSolver, float weight)
{
	assert(gnSolver);
	m_gnSolver = gnSolver;
	m_inputData = gnSolver->m_inputData;
	setWeight(weight);
	return true;
}

bool SmoothConstraint::init()
{	
	return true;
}

void SmoothConstraint::getJTJAndJTb(float* dJTJ_a, int* dJTJ_ia, thrust::device_vector<float>& dJTb, float3* dVars)
{
	b(dVars);
	directiveJTJ(dJTJ_a, dJTJ_ia);
	directiveJTb(dJTb);
}

__global__ void bKernelSmooth(float* dB,
                              float4* dNodeVec,
                              int nodeNum,
                              int* dNodeRelaIdxVec,
                              float* dNodeRelaWeightVec,
                              float3* dVars,
                              float weight)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= nodeNum * MAX_NEAR_NODE_NUM_NODE)
	{
		return;
	}

	int seriK = idx / MAX_NEAR_NODE_NUM_NODE;
	int seriJ = dNodeRelaIdxVec[idx];
	float weightN2N = dNodeRelaWeightVec[idx] * weight, tmp;
	float3 vvi, nodeK, nodeJ;
	nodeK = make_float3(dNodeVec[seriK]);
	nodeJ = make_float3(dNodeVec[seriJ]);
	vvi = nodeJ - nodeK;

	float3* RtNode = dVars + seriK * 4;

	float3 resi = weightN2N * (((vvi.x * RtNode[0] + vvi.y * RtNode[1] + vvi.z * RtNode[2]) + (nodeK - nodeJ)) +
		(RtNode[3] - *(dVars + seriJ * 4 + 3)));
#if 0
	// 这样数值误差比较大
		weightN2N * (vvi.x * RtNode[0] + vvi.y * RtNode[1] + vvi.z * RtNode[2] +
		(nodeK + RtNode[3]) -
		(nodeJ + *(dVars + seriJ * 4 + 3)));
#endif
	
	dB[3 * idx] = resi.x;
	dB[3 * idx + 1] = resi.y;
	dB[3 * idx + 2] = resi.z;
}

void SmoothConstraint::b(float3* dVars)
{
	int nodeNum = m_inputData->m_source.m_nodeNum;
	m_dB.resize(nodeNum * MAX_NEAR_NODE_NUM_NODE * 3);

	int block = 512;
	int grid = (block + nodeNum * MAX_NEAR_NODE_NUM_NODE - 1) / block;
	bKernelSmooth << <grid, block >> >(RAW_PTR(m_dB),
	                                   RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                   nodeNum,
	                                   RAW_PTR(m_inputData->m_source.m_dNodeRelaIdxVec),
	                                   RAW_PTR(m_inputData->m_source.m_dNodeRelaWeightVec),
	                                   dVars,
	                                   m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
#if 0
	std::vector<float> bVec(m_dB.size());
	checkCudaErrors(cudaMemcpy(bVec.data(), RAW_PTR(m_dB), bVec.size() * sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "b: " << std::endl;
	for (int i = 0; i < bVec.size(); ++i)
	{
		std::cout << bVec[i] << ", ";
	}
	std::cout << "----------------------------" << std::endl;
	std::exit(0);
#endif
}

__global__ void DirectiveJTJKernelSmooth(float* dJtJ_a,
                                         int* dJtJ_ia,
                                         int* dListIij,
                                         int* dOffsetIij,
                                         int* dNzIijCoo,
                                         int* dNnzPre,
                                         int* dDataItemNum,
                                         float4* dNodeVec,
                                         int nodeNum,
                                         int* dNodeRelaIdxVec,
                                         float* dNodeRelaWeightVec,
                                         int matchingPointNum,
                                         float weight)
{
	int nzIndex = dNzIijCoo[blockIdx.x];
	int row = nzIndex / nodeNum;
	int col = nzIndex - row * nodeNum;
	int dataNum = dDataItemNum[blockIdx.x];;
	int num = dOffsetIij[blockIdx.x + 1] - dOffsetIij[blockIdx.x] - dataNum;
	if (0 == num)
	{
		return;
	}

	__shared__ float Jci[32 * 3 * 12];
	__shared__ float Jcj[4 * 3 * 12];
	__shared__ float block_JTJ_res[12 * 12];

	if (threadIdx.x < 144)
	{
		block_JTJ_res[threadIdx.x] = 0.0f;
		Jcj[threadIdx.x] = 0.0f;
	}
#pragma unroll
	for (int iter = 0; iter < 6; iter++)
	{
		Jci[threadIdx.x + iter * 192] = 0.0f;
	}

	__syncthreads();

	int seriK, seriJ, fIdx;
	float4 nodeK, nodeJ, vvi;
	float weight_n2n = 0.0f;
	int* fSet = dListIij + dOffsetIij[blockIdx.x] + dataNum;
	int rowRes, colRes;
	rowRes = threadIdx.x / 12;
	colRes = threadIdx.x % 12;
	int currentIdx = threadIdx.x * 36;
	if (row == col)
	{
		if (threadIdx.x < num)
		{
			fIdx = fSet[threadIdx.x] - matchingPointNum;
			if ((fIdx - (fIdx / MAX_NEAR_NODE_NUM_NODE) * MAX_NEAR_NODE_NUM_NODE) != 0)
			{
				seriK = fIdx / MAX_NEAR_NODE_NUM_NODE;
				weight_n2n = dNodeRelaWeightVec[fIdx] * weight;
				seriJ = dNodeRelaIdxVec[fIdx];
				nodeK = dNodeVec[seriK];
				nodeJ = dNodeVec[seriJ];
				vvi = nodeK - nodeJ;
				if (seriK == row)
				{
					Jci[currentIdx + 0] = vvi.x * weight_n2n;
					Jci[currentIdx + 3] = vvi.y * weight_n2n;
					Jci[currentIdx + 6] = vvi.z * weight_n2n;
					Jci[currentIdx + 9] = -weight_n2n;

					Jci[currentIdx + 12 + 1] = vvi.x * weight_n2n;
					Jci[currentIdx + 12 + 4] = vvi.y * weight_n2n;
					Jci[currentIdx + 12 + 7] = vvi.z * weight_n2n;
					Jci[currentIdx + 12 + 10] = -weight_n2n;

					Jci[currentIdx + 24 + 2] = vvi.x * weight_n2n;
					Jci[currentIdx + 24 + 5] = vvi.y * weight_n2n;
					Jci[currentIdx + 24 + 8] = vvi.z * weight_n2n;
					Jci[currentIdx + 24 + 11] = -weight_n2n;
				}
				else
				{
					Jci[currentIdx + 9] = weight_n2n;
					Jci[currentIdx + 12 + 10] = weight_n2n;
					Jci[currentIdx + 24 + 11] = weight_n2n;
				}
			}
		}
		__syncthreads();
		//reduction
		if (threadIdx.x < 144)
		{
			for (int iter_redu = 0; iter_redu < num * 3; iter_redu++)
			{
				block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + rowRes] * Jci[iter_redu * 12 + colRes];
			}
		}
		__syncthreads();
		//write back
		int start_pos;
		if (threadIdx.x < 144)
		{
			start_pos = dNnzPre[row * nodeNum + col] * 12;
			dJtJ_a[dJtJ_ia[row * 12 + rowRes] + start_pos + colRes] += block_JTJ_res[rowRes * 12 + colRes];
		}
	}
	else
	{
		if (threadIdx.x < num)
		{
			fIdx = fSet[threadIdx.x] - matchingPointNum;
			weight_n2n = - dNodeRelaWeightVec[fIdx] * weight;
			seriK = fIdx / MAX_NEAR_NODE_NUM_NODE;
			seriJ = dNodeRelaIdxVec[fIdx];
			nodeK = dNodeVec[seriK];
			nodeJ = dNodeVec[seriJ];
			vvi = nodeK - nodeJ;
			if (seriK == row)
			{
				Jcj[threadIdx.x * 36 + 9] = weight_n2n;
				Jcj[threadIdx.x * 36 + 12 + 10] = weight_n2n;
				Jcj[threadIdx.x * 36 + 24 + 11] = weight_n2n;

				Jci[threadIdx.x * 36 + 0] = vvi.x * weight_n2n;
				Jci[threadIdx.x * 36 + 3] = vvi.y * weight_n2n;
				Jci[threadIdx.x * 36 + 6] = vvi.z * weight_n2n;
				Jci[threadIdx.x * 36 + 9] = -weight_n2n;

				Jci[threadIdx.x * 36 + 12 + 1] = vvi.x * weight_n2n;
				Jci[threadIdx.x * 36 + 12 + 4] = vvi.y * weight_n2n;
				Jci[threadIdx.x * 36 + 12 + 7] = vvi.z * weight_n2n;
				Jci[threadIdx.x * 36 + 12 + 10] = -weight_n2n;

				Jci[threadIdx.x * 36 + 24 + 2] = vvi.x * weight_n2n;
				Jci[threadIdx.x * 36 + 24 + 5] = vvi.y * weight_n2n;
				Jci[threadIdx.x * 36 + 24 + 8] = vvi.z * weight_n2n;
				Jci[threadIdx.x * 36 + 24 + 11] = -weight_n2n;
			}
			else
			{
				Jci[threadIdx.x * 36 + 9] = weight_n2n;
				Jci[threadIdx.x * 36 + 12 + 10] = weight_n2n;
				Jci[threadIdx.x * 36 + 24 + 11] = weight_n2n;

				Jcj[threadIdx.x * 36 + 0] = vvi.x * weight_n2n;
				Jcj[threadIdx.x * 36 + 3] = vvi.y * weight_n2n;
				Jcj[threadIdx.x * 36 + 6] = vvi.z * weight_n2n;
				Jcj[threadIdx.x * 36 + 9] = -weight_n2n;

				Jcj[threadIdx.x * 36 + 12 + 1] = vvi.x * weight_n2n;
				Jcj[threadIdx.x * 36 + 12 + 4] = vvi.y * weight_n2n;
				Jcj[threadIdx.x * 36 + 12 + 7] = vvi.z * weight_n2n;
				Jcj[threadIdx.x * 36 + 12 + 10] = -weight_n2n;

				Jcj[threadIdx.x * 36 + 24 + 2] = vvi.x * weight_n2n;
				Jcj[threadIdx.x * 36 + 24 + 5] = vvi.y * weight_n2n;
				Jcj[threadIdx.x * 36 + 24 + 8] = vvi.z * weight_n2n;
				Jcj[threadIdx.x * 36 + 24 + 11] = -weight_n2n;
			}
		}
		__syncthreads();
		//reduction
		if (threadIdx.x < 144)
		{
			for (int iter_redu = 0; iter_redu < num * 3; iter_redu++)
			{
				block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + rowRes] * Jcj[iter_redu * 12 + colRes];
			}
		}

		__syncthreads();
		//write back
		int start_pos;
		if (threadIdx.x < 144)
		{
			start_pos = dNnzPre[row * nodeNum + col] * 12;
			dJtJ_a[dJtJ_ia[row * 12 + rowRes] + start_pos + colRes] += block_JTJ_res[rowRes * 12 + colRes];
			start_pos = dNnzPre[col * nodeNum + row] * 12;
			dJtJ_a[dJtJ_ia[col * 12 + rowRes] + start_pos + colRes] += block_JTJ_res[colRes * 12 + rowRes];
		}
	}
}

void SmoothConstraint::directiveJTJ(float* JTJ_a,
                                    int* JTJ_ia)
{
	int block_size = 192; // fixed!!
	int grid_size = m_inputData->m_Iij.m_nnzIij;
	DirectiveJTJKernelSmooth << <grid_size, block_size >> >(JTJ_a,
	                                                        JTJ_ia,
	                                                        RAW_PTR(m_inputData->m_Iij.m_dListIij),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dOffsetIij),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dNzIijCoo),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dNnzPre),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dDataItemNum),
	                                                        RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                                        m_inputData->m_source.m_nodeNum,
	                                                        RAW_PTR(m_inputData->m_source.m_dNodeRelaIdxVec),
	                                                        RAW_PTR(m_inputData->m_source.m_dNodeRelaWeightVec),
	                                                        m_inputData->m_matchingPointNum,
	                                                        m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void DirectiveJTbKernelSmooth(float* dJTb,
                                         float* dB,
                                         int* dListIij,
                                         int* dOffsetIij,
                                         int* dNzIijCoo,
                                         int* dNnzPre,
                                         int* dDataItemNum,
                                         float4* dNodeVec,
                                         int nodeNum,
                                         int* dNodeRelaIdxVec,
                                         float* dNodeRelaWeightVec,
                                         int matchingPointNum,
                                         float weight)
{
	int nzIndex = dNzIijCoo[blockIdx.x];
	int row = nzIndex / nodeNum;
	int col = nzIndex - row * nodeNum;
	if (row != col)
	{
		return;
	}

	int dataNum = dDataItemNum[blockIdx.x];;
	int num = dOffsetIij[blockIdx.x + 1] - dOffsetIij[blockIdx.x] - dataNum;

	__shared__ float JTb_block_tmp[32 * 3 * 12];
	__shared__ float JTb_block_res[12];
#pragma unroll
	for (int iter = 0; iter < 12; iter++)
	{
		JTb_block_tmp[threadIdx.x + 96 * iter] = 0.0f;
	}
	if (threadIdx.x < 12)
	{
		JTb_block_res[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	int seriK, seriJ, fIdx;
	float4 nodeK, nodeJ, vvi;
	float weight_n2n = 0.0f, weight_tmp;
	int* fSet = dListIij + dOffsetIij[blockIdx.x] + dataNum;
	int currentIdx = threadIdx.x * 36;
	if (threadIdx.x < num)
	{
		fIdx = fSet[threadIdx.x] - matchingPointNum;
		if ((fIdx - (fIdx / MAX_NEAR_NODE_NUM_NODE) * MAX_NEAR_NODE_NUM_NODE) != 0)
		{
			seriK = fIdx / MAX_NEAR_NODE_NUM_NODE;
			weight_n2n = - dNodeRelaWeightVec[fIdx] * weight;
			seriJ = dNodeRelaIdxVec[fIdx];
			nodeK = dNodeVec[seriK];
			nodeJ = dNodeVec[seriJ];
			vvi = nodeK - nodeJ;
			if (seriK == row)
			{
				weight_tmp = weight_n2n * dB[fIdx * 3];
				JTb_block_tmp[currentIdx + 0] = vvi.x * weight_tmp;
				JTb_block_tmp[currentIdx + 3] = vvi.y * weight_tmp;
				JTb_block_tmp[currentIdx + 6] = vvi.z * weight_tmp;
				JTb_block_tmp[currentIdx + 9] = -weight_tmp;

				weight_tmp = weight_n2n * dB[fIdx * 3 + 1];
				JTb_block_tmp[currentIdx + 12 + 1] = vvi.x * weight_tmp;
				JTb_block_tmp[currentIdx + 12 + 4] = vvi.y * weight_tmp;
				JTb_block_tmp[currentIdx + 12 + 7] = vvi.z * weight_tmp;
				JTb_block_tmp[currentIdx + 12 + 10] = -weight_tmp;

				weight_tmp = weight_n2n * dB[fIdx * 3 + 2];
				JTb_block_tmp[currentIdx + 24 + 2] = vvi.x * weight_tmp;
				JTb_block_tmp[currentIdx + 24 + 5] = vvi.y * weight_tmp;
				JTb_block_tmp[currentIdx + 24 + 8] = vvi.z * weight_tmp;
				JTb_block_tmp[currentIdx + 24 + 11] = -weight_tmp;
			}
			else
			{
				JTb_block_tmp[currentIdx + 9] = weight_n2n * dB[fIdx * 3];
				JTb_block_tmp[currentIdx + 12 + 10] = weight_n2n * dB[fIdx * 3 + 1];
				JTb_block_tmp[currentIdx + 24 + 11] = weight_n2n * dB[fIdx * 3 + 2];
			}
		}
	}
	__syncthreads();
	// reduction
	if (threadIdx.x < 12)
	{
		for (int iter_redu = 0; iter_redu < num * 3; iter_redu++)
		{
			JTb_block_res[threadIdx.x] += JTb_block_tmp[threadIdx.x + iter_redu * 12];
		}
	}
	__syncthreads();
	// write back
	if (threadIdx.x < 12)
	{
		dJTb[row * 12 + threadIdx.x] -= JTb_block_res[threadIdx.x];
	}
}

void SmoothConstraint::directiveJTb(thrust::device_vector<float>& JTb)
{
	int block_size = 96; // fixed!!
	int grid_size = m_inputData->m_Iij.m_nnzIij;
	DirectiveJTbKernelSmooth << <grid_size, block_size >> >(RAW_PTR(JTb),
	                                                        RAW_PTR(m_dB),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dListIij),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dOffsetIij),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dNzIijCoo),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dNnzPre),
	                                                        RAW_PTR(m_inputData->m_Iij.m_dDataItemNum),
	                                                        RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                                        m_inputData->m_source.m_nodeNum,
	                                                        RAW_PTR(m_inputData->m_source.m_dNodeRelaIdxVec),
	                                                        RAW_PTR(m_inputData->m_source.m_dNodeRelaWeightVec),
	                                                        m_inputData->m_matchingPointNum,
	                                                        m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}
