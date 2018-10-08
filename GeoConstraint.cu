#pragma once

#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "GeoConstraint.h"
#include "GNSolver.h"
#include "InputData.h"
#include "Helpers/UtilsMath.h"
#include <helper_cuda.h>

__forceinline__ __device__ int FindIndex(int* nodeVec, int val, int num)
{
	for (int i = 0; i < num; i++)
	{
		if (nodeVec[i] == val)
		{
			return i;
		}
	}
	return -1;
}

bool GeoConstraint::init(GNSolver* gnSolver, float weight)
{
	assert(gnSolver);
	m_gnSolver = gnSolver;
	m_inputData = gnSolver->m_inputData;
	setWeight(weight);
	return true;
}

bool GeoConstraint::init()
{
	return true;
}

void GeoConstraint::getJTJAndJTb(float* dJTJ_a, int* dJTJ_ia, thrust::device_vector<float>& dJTb, float3* dVars)
{
	b(dVars);
	directiveJTJ(dJTJ_a, dJTJ_ia);
	directiveJTb(dJTb);
}

__global__ void bKernelGeo(float* dB,
                           int* dMatchingPointIndices,
                           int matchingPointNum,
                           float4* dVertexVec,
                           float4* dNormalVec,
                           float weight)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= matchingPointNum)
	{
		return;
	}

	int srcVertexIdx = dMatchingPointIndices[2 * idx];
	int targetVertexIdx = dMatchingPointIndices[2 * idx + 1];

	if (srcVertexIdx < 0 || targetVertexIdx < 0)
	{
		dB[idx] = 0;
		return;
	}

	float4 updatedVertexNormalTarget = dNormalVec[targetVertexIdx];
	float4 residualDiff = dVertexVec[srcVertexIdx] - dVertexVec[targetVertexIdx];

	dB[idx] = weight
		* (updatedVertexNormalTarget.x * residualDiff.x
			+ updatedVertexNormalTarget.y * residualDiff.y
			+ updatedVertexNormalTarget.z * residualDiff.z);
}

void GeoConstraint::b(float3* dVars)
{
	m_dB.resize(m_inputData->m_matchingPointNum);
	int block = 1024;
	int grid = (block + m_inputData->m_matchingPointNum - 1) / block;
	bKernelGeo << <grid, block >> >(RAW_PTR(m_dB),
	                                m_inputData->m_dMatchingPointIndices,
	                                m_inputData->m_matchingPointNum,
	                                RAW_PTR(m_inputData->m_deformed.m_dVertexVec),
	                                RAW_PTR(m_inputData->m_deformed.m_dNormalVec),
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

__global__ void DirectiveJTJKernelGeo(float* dJTJ_a,
                                      int* dJTJ_ia,
                                      int* dListIij,
                                      int* dOffsetIij,
                                      int* dNzIijCoo,
                                      int* dNnzPre,
                                      int* dDataItemNum,
                                      float4* dSrcVertexVec,
                                      int* dVertexRelaIdxVec,
                                      float* dVertexRelaWeightVec,
                                      float4* dNodeVec,
                                      int nodeNum,
                                      float4* dTargetNormalVec,
                                      int* dMatchingPointIndices,
                                      int matchingPointNum,
                                      float weight)
{
	__shared__ float Jci[160 * 12];
	__shared__ float Jcj[160 * 12];
	__shared__ float block_JTJ_res[12 * 12];

	if (threadIdx.x < 144)
	{
		block_JTJ_res[threadIdx.x] = 0;
	}
	__syncthreads();

	int nzIdx = dNzIijCoo[blockIdx.x];
	int row = nzIdx / nodeNum;
	int col = nzIdx - row * nodeNum;
	int num = dDataItemNum[blockIdx.x];
	int* fSet = dListIij + dOffsetIij[blockIdx.x];

	int fIdx;
	float weightV2N, tmp1, tmp2, tmp3;
	float4 vnn, targetNormal;
	int fullTimes = num / int(160);
	int residualNum = num - fullTimes * 160;
	int rowRes, colRes;
	rowRes = threadIdx.x / 12;
	colRes = threadIdx.x - rowRes * 12;
	int currentIdx = threadIdx.x * 12;
#if 1
	for (int iter_ft = 0; iter_ft < fullTimes; iter_ft++)
	{
		fIdx = fSet[threadIdx.x + iter_ft * 160];
		int srcVertexIdx = dMatchingPointIndices[2 * fIdx];
		int targetVertexIdx = dMatchingPointIndices[2 * fIdx + 1];
		if (srcVertexIdx < 0 || targetVertexIdx < 0)
		{
#pragma unroll
			for (int iter = 0; iter < 12; iter++)
			{
				Jci[currentIdx + iter] = 0.0f;
				Jcj[currentIdx + iter] = 0.0f;
			}
		}
		else
		{
			int nodeIdxRow, nodeIdxCol, vertexIdxRow, vertexIdxCol;
			float signRow = 1.0f, signCol = 1.0f;
			int nodeIdxRow1 = FindIndex(dVertexRelaIdxVec + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, row,
			                            MAX_NEAR_NODE_NUM_VERTEX);
			int nodeIdxRow2 = FindIndex(dVertexRelaIdxVec + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, row,
			                            MAX_NEAR_NODE_NUM_VERTEX);
			int nodeIdxCol1 = FindIndex(dVertexRelaIdxVec + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
			                            MAX_NEAR_NODE_NUM_VERTEX);
			int nodeIdxCol2 = FindIndex(dVertexRelaIdxVec + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
			                            MAX_NEAR_NODE_NUM_VERTEX);
			if ((row / NODE_NUM_EACH_FRAG) == (col / NODE_NUM_EACH_FRAG))
			{
				if (nodeIdxRow1 > nodeIdxRow2)
				{
					nodeIdxRow = nodeIdxRow1;
					nodeIdxCol = nodeIdxCol1;
					vertexIdxRow = srcVertexIdx;
					vertexIdxCol = srcVertexIdx;
				}
				else
				{
					nodeIdxRow = nodeIdxRow2;
					nodeIdxCol = nodeIdxCol2;
					vertexIdxRow = targetVertexIdx;
					vertexIdxCol = targetVertexIdx;
				}
			}
			else
			{
				if (nodeIdxRow1 > nodeIdxRow2)
				{
					nodeIdxRow = nodeIdxRow1;
					nodeIdxCol = nodeIdxCol2;
					vertexIdxRow = srcVertexIdx;
					vertexIdxCol = targetVertexIdx;
					signCol = -1.0f;
				}
				else
				{
					nodeIdxRow = nodeIdxRow2;
					nodeIdxCol = nodeIdxCol1;
					vertexIdxRow = targetVertexIdx;
					vertexIdxCol = srcVertexIdx;
					signRow = -1.0f;
				}
			}

			// compute Jcj first.
			targetNormal = dTargetNormalVec[targetVertexIdx];
			targetNormal.w = 0.0f;
			targetNormal = normalize(targetNormal);
			weightV2N = signCol * weight * dVertexRelaWeightVec[vertexIdxCol * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxCol];
			vnn = dSrcVertexVec[vertexIdxCol] -
				dNodeVec[dVertexRelaIdxVec[vertexIdxCol * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxCol]];
			tmp1 = weightV2N * targetNormal.x;
			tmp2 = weightV2N * targetNormal.y;
			tmp3 = weightV2N * targetNormal.z;

			Jcj[currentIdx++] = tmp1 * vnn.x;
			Jcj[currentIdx++] = tmp2 * vnn.x;
			Jcj[currentIdx++] = tmp3 * vnn.x;
			Jcj[currentIdx++] = tmp1 * vnn.y;
			Jcj[currentIdx++] = tmp2 * vnn.y;
			Jcj[currentIdx++] = tmp3 * vnn.y;
			Jcj[currentIdx++] = tmp1 * vnn.z;
			Jcj[currentIdx++] = tmp2 * vnn.z;
			Jcj[currentIdx++] = tmp3 * vnn.z;
			Jcj[currentIdx++] = tmp1;
			Jcj[currentIdx++] = tmp2;
			Jcj[currentIdx] = tmp3;

			// compute Jci
			weightV2N = signRow * weight * dVertexRelaWeightVec[vertexIdxRow * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxRow];
			vnn = dSrcVertexVec[vertexIdxRow] -
				dNodeVec[dVertexRelaIdxVec[vertexIdxRow * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxRow]];
			tmp1 = weightV2N * targetNormal.x;
			tmp2 = weightV2N * targetNormal.y;
			tmp3 = weightV2N * targetNormal.z;

			Jci[currentIdx--] = tmp3;
			Jci[currentIdx--] = tmp2;
			Jci[currentIdx--] = tmp1;
			Jci[currentIdx--] = tmp3 * vnn.z;
			Jci[currentIdx--] = tmp2 * vnn.z;
			Jci[currentIdx--] = tmp1 * vnn.z;
			Jci[currentIdx--] = tmp3 * vnn.y;
			Jci[currentIdx--] = tmp2 * vnn.y;
			Jci[currentIdx--] = tmp1 * vnn.y;
			Jci[currentIdx--] = tmp3 * vnn.x;
			Jci[currentIdx--] = tmp2 * vnn.x;
			Jci[currentIdx] = tmp1 * vnn.x;
		}

		__syncthreads();
		//reduction
		if (threadIdx.x < 144)
		{
#pragma unroll
			for (int iter_redu = 0; iter_redu < 160; iter_redu++)
			{
				block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + rowRes] * Jcj[iter_redu * 12 + colRes];
			}
		}

		__syncthreads();
	}
#endif

	if (threadIdx.x < residualNum)
	{
		fIdx = fSet[threadIdx.x + fullTimes * 160];
		int srcVertexIdx = dMatchingPointIndices[2 * fIdx];
		int targetVertexIdx = dMatchingPointIndices[2 * fIdx + 1];
		if (srcVertexIdx < 0 || targetVertexIdx < 0)
		{
#pragma unroll
			for (int iter = 0; iter < 12; iter++)
			{
				Jci[currentIdx + iter] = 0.0f;
				Jcj[currentIdx + iter] = 0.0f;
			}
		}
		else
		{
			int nodeIdxRow, nodeIdxCol, vertexIdxRow, vertexIdxCol;
			float signRow = 1.0f, signCol = 1.0f;
			int nodeIdxRow1 = FindIndex(dVertexRelaIdxVec + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, row,
			                            MAX_NEAR_NODE_NUM_VERTEX);
			int nodeIdxRow2 = FindIndex(dVertexRelaIdxVec + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, row,
			                            MAX_NEAR_NODE_NUM_VERTEX);
			int nodeIdxCol1 = FindIndex(dVertexRelaIdxVec + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
			                            MAX_NEAR_NODE_NUM_VERTEX);
			int nodeIdxCol2 = FindIndex(dVertexRelaIdxVec + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
			                            MAX_NEAR_NODE_NUM_VERTEX);
			if ((row / NODE_NUM_EACH_FRAG) == (col / NODE_NUM_EACH_FRAG))
			{
				if (nodeIdxRow1 > nodeIdxRow2)
				{
					nodeIdxRow = nodeIdxRow1;
					nodeIdxCol = nodeIdxCol1;
					vertexIdxRow = srcVertexIdx;
					vertexIdxCol = srcVertexIdx;
				}
				else
				{
					nodeIdxRow = nodeIdxRow2;
					nodeIdxCol = nodeIdxCol2;
					vertexIdxRow = targetVertexIdx;
					vertexIdxCol = targetVertexIdx;
				}
			}
			else
			{
				if (nodeIdxRow1 > nodeIdxRow2)
				{
					nodeIdxRow = nodeIdxRow1;
					nodeIdxCol = nodeIdxCol2;
					vertexIdxRow = srcVertexIdx;
					vertexIdxCol = targetVertexIdx;
					signCol = -1.0f;
				}
				else
				{
					nodeIdxRow = nodeIdxRow2;
					nodeIdxCol = nodeIdxCol1;
					vertexIdxRow = targetVertexIdx;
					vertexIdxCol = srcVertexIdx;
					signRow = -1.0f;
				}
			}

			// compute Jcj first.
			targetNormal = dTargetNormalVec[targetVertexIdx];
			targetNormal.w = 0.0f;
			targetNormal = normalize(targetNormal);
			weightV2N = signCol * weight * dVertexRelaWeightVec[vertexIdxCol * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxCol];
			vnn = dSrcVertexVec[vertexIdxCol] -
				dNodeVec[dVertexRelaIdxVec[vertexIdxCol * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxCol]];
			tmp1 = weightV2N * targetNormal.x;
			tmp2 = weightV2N * targetNormal.y;
			tmp3 = weightV2N * targetNormal.z;

			Jcj[currentIdx++] = tmp1 * vnn.x;
			Jcj[currentIdx++] = tmp2 * vnn.x;
			Jcj[currentIdx++] = tmp3 * vnn.x;
			Jcj[currentIdx++] = tmp1 * vnn.y;
			Jcj[currentIdx++] = tmp2 * vnn.y;
			Jcj[currentIdx++] = tmp3 * vnn.y;
			Jcj[currentIdx++] = tmp1 * vnn.z;
			Jcj[currentIdx++] = tmp2 * vnn.z;
			Jcj[currentIdx++] = tmp3 * vnn.z;
			Jcj[currentIdx++] = tmp1;
			Jcj[currentIdx++] = tmp2;
			Jcj[currentIdx] = tmp3;

			// compute Jci
			weightV2N = signRow * weight * dVertexRelaWeightVec[vertexIdxRow * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxRow];
			vnn = dSrcVertexVec[vertexIdxRow] -
				dNodeVec[dVertexRelaIdxVec[vertexIdxRow * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxRow]];
			tmp1 = weightV2N * targetNormal.x;
			tmp2 = weightV2N * targetNormal.y;
			tmp3 = weightV2N * targetNormal.z;

			Jci[currentIdx--] = tmp3;
			Jci[currentIdx--] = tmp2;
			Jci[currentIdx--] = tmp1;
			Jci[currentIdx--] = tmp3 * vnn.z;
			Jci[currentIdx--] = tmp2 * vnn.z;
			Jci[currentIdx--] = tmp1 * vnn.z;
			Jci[currentIdx--] = tmp3 * vnn.y;
			Jci[currentIdx--] = tmp2 * vnn.y;
			Jci[currentIdx--] = tmp1 * vnn.y;
			Jci[currentIdx--] = tmp3 * vnn.x;
			Jci[currentIdx--] = tmp2 * vnn.x;
			Jci[currentIdx] = tmp1 * vnn.x;
		}
	}

	__syncthreads();
	//reduction
	if (threadIdx.x < 144)
	{
		for (int iter_redu = 0; iter_redu < residualNum; iter_redu++)
		{
			block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + rowRes] * Jcj[iter_redu * 12 + colRes];
		}
	}

	__syncthreads();
	// write JTJ_block(i,j) to global memory.
	int startPos;
	if (threadIdx.x < 144)
	{
		if (row == col)
		{
			startPos = dNnzPre[row * nodeNum + col] * 12;
			dJTJ_a[dJTJ_ia[row * 12 + rowRes] + startPos + colRes] += block_JTJ_res[rowRes * 12 + colRes];
		}
		else
		{
			startPos = dNnzPre[row * nodeNum + col] * 12;
			dJTJ_a[dJTJ_ia[row * 12 + rowRes] + startPos + colRes] += block_JTJ_res[rowRes * 12 + colRes];
			startPos = dNnzPre[col * nodeNum + row] * 12;
			dJTJ_a[dJTJ_ia[col * 12 + rowRes] + startPos + colRes] += block_JTJ_res[colRes * 12 + rowRes];
		}
	}
}

void GeoConstraint::directiveJTJ(float* dJTJ_a, int* dJTJ_ia)
{
	//std::cout << m_weight << std::endl;
	//std::exit(0);
	int block = 160; // fixed !!!
	int grid = m_inputData->m_Iij.m_nnzIij; //each CUDA block handle one JTJ block.
	DirectiveJTJKernelGeo << <grid, block >> >(dJTJ_a,
	                                           dJTJ_ia,
	                                           RAW_PTR(m_inputData->m_Iij.m_dListIij),
	                                           RAW_PTR(m_inputData->m_Iij.m_dOffsetIij),
	                                           RAW_PTR(m_inputData->m_Iij.m_dNzIijCoo),
	                                           RAW_PTR(m_inputData->m_Iij.m_dNnzPre),
	                                           RAW_PTR(m_inputData->m_Iij.m_dDataItemNum),
	                                           RAW_PTR(m_inputData->m_source.m_dVertexVec),
	                                           RAW_PTR(m_inputData->m_source.m_dVertexRelaIdxVec),
	                                           RAW_PTR(m_inputData->m_source.m_dVertexRelaWeightVec),
	                                           RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                           m_inputData->m_source.m_nodeNum,
	                                           RAW_PTR(m_inputData->m_deformed.m_dNormalVec),
	                                           m_inputData->m_dMatchingPointIndices,
	                                           m_inputData->m_matchingPointNum,
	                                           m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void DirectiveJTbKernelGeo(float* dJTb,
                                      float* dB,
                                      int* dListIij,
                                      int* dOffsetIij,
                                      int* dNzIijCoo,
                                      int* dDataItemNum,
                                      float4* dSrcVertexVec,
                                      int* dVertexRelaIdxVec,
                                      float* dVertexRelaWeightVec,
                                      float4* dNodeVec,
                                      int nodeNum,
                                      float4* dTargetNormalVec,
                                      int* dMatchingPointIndices,
                                      int matchingPointNum,
                                      float weight)
{
	__shared__ float JTb_block_tmp[12 * 256];
	__shared__ float JTb_block_res[12];
	int nzIdx = dNzIijCoo[blockIdx.x];
	int row = nzIdx / nodeNum;
	int col = nzIdx - row * nodeNum;
	if (row != col)
	{
		return;
	}
	if (threadIdx.x < 12)
	{
		JTb_block_res[threadIdx.x] = 0;
	}
	int num = dDataItemNum[blockIdx.x];
	int fullTimes = num / 256;
	int residualNum = num % 256;
	int currentIdx = threadIdx.x * 12;
	int fIdx = 0;
	int* fSet = dListIij + dOffsetIij[blockIdx.x];
	float4 targetNormal, nodePos = dNodeVec[col], vnn;
	float weight_V2N = 0.0, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;
	int srcVertexIdx, targetVertexIdx;
	int nodeIdxCol1, nodeIdxCol2, nodeIdxCol, vertexIdxCol, signCol = 1.0f;
	for (int iter_ft = 0; iter_ft < fullTimes; iter_ft++)
	{
		fIdx = fSet[threadIdx.x + iter_ft * 256];
		srcVertexIdx = dMatchingPointIndices[2 * fIdx];
		targetVertexIdx = dMatchingPointIndices[2 * fIdx + 1];
		if (srcVertexIdx < 0 || targetVertexIdx < 0)
		{
			JTb_block_tmp[currentIdx + 0] = 0;
			JTb_block_tmp[currentIdx + 1] = 0;
			JTb_block_tmp[currentIdx + 2] = 0;
			JTb_block_tmp[currentIdx + 3] = 0;
			JTb_block_tmp[currentIdx + 4] = 0;
			JTb_block_tmp[currentIdx + 5] = 0;
			JTb_block_tmp[currentIdx + 6] = 0;
			JTb_block_tmp[currentIdx + 7] = 0;
			JTb_block_tmp[currentIdx + 8] = 0;
			JTb_block_tmp[currentIdx + 9] = 0;
			JTb_block_tmp[currentIdx + 10] = 0;
			JTb_block_tmp[currentIdx + 11] = 0;
		}
		else
		{
			nodeIdxCol1 = FindIndex(dVertexRelaIdxVec + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col, MAX_NEAR_NODE_NUM_VERTEX);
			nodeIdxCol2 = FindIndex(dVertexRelaIdxVec + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
			                        MAX_NEAR_NODE_NUM_VERTEX);
			if (nodeIdxCol1 > nodeIdxCol2)
			{
				nodeIdxCol = nodeIdxCol1;
				vertexIdxCol = srcVertexIdx;
				signCol = 1.0f;
			}
			else
			{
				nodeIdxCol = nodeIdxCol2;
				vertexIdxCol = targetVertexIdx;
				signCol = -1.0f;
			}

			targetNormal = dTargetNormalVec[targetVertexIdx];
			targetNormal.w = 0.0f;
			targetNormal = normalize(targetNormal);
			weight_V2N = signCol * weight * dVertexRelaWeightVec[vertexIdxCol * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxCol] * dB[fIdx
			];
			vnn = dSrcVertexVec[vertexIdxCol] - nodePos;
			tmp1 = weight_V2N * targetNormal.x;
			tmp2 = weight_V2N * targetNormal.y;
			tmp3 = weight_V2N * targetNormal.z;	

			JTb_block_tmp[currentIdx + 0] = tmp1 * vnn.x;
			JTb_block_tmp[currentIdx + 1] = tmp2 * vnn.x;
			JTb_block_tmp[currentIdx + 2] = tmp3 * vnn.x;
			JTb_block_tmp[currentIdx + 3] = tmp1 * vnn.y;
			JTb_block_tmp[currentIdx + 4] = tmp2 * vnn.y;
			JTb_block_tmp[currentIdx + 5] = tmp3 * vnn.y;
			JTb_block_tmp[currentIdx + 6] = tmp1 * vnn.z;
			JTb_block_tmp[currentIdx + 7] = tmp2 * vnn.z;
			JTb_block_tmp[currentIdx + 8] = tmp3 * vnn.z;
			JTb_block_tmp[currentIdx + 9] = tmp1;
			JTb_block_tmp[currentIdx + 10] = tmp2;
			JTb_block_tmp[currentIdx + 11] = tmp3;
		}

		__syncthreads();
		//reduction
		if (threadIdx.x < 12)
		{
			for (int iter_redu = 0; iter_redu < 256; iter_redu++)
			{
				JTb_block_res[threadIdx.x] += JTb_block_tmp[threadIdx.x + iter_redu * 12];
			}
		}
		__syncthreads();
	}

	if (threadIdx.x < residualNum)
	{
		fIdx = fSet[threadIdx.x + fullTimes * 256];
		srcVertexIdx = dMatchingPointIndices[2 * fIdx];
		targetVertexIdx = dMatchingPointIndices[2 * fIdx + 1];
		if (srcVertexIdx < 0 || targetVertexIdx < 0)
		{
			JTb_block_tmp[currentIdx + 0] = 0;
			JTb_block_tmp[currentIdx + 1] = 0;
			JTb_block_tmp[currentIdx + 2] = 0;
			JTb_block_tmp[currentIdx + 3] = 0;
			JTb_block_tmp[currentIdx + 4] = 0;
			JTb_block_tmp[currentIdx + 5] = 0;
			JTb_block_tmp[currentIdx + 6] = 0;
			JTb_block_tmp[currentIdx + 7] = 0;
			JTb_block_tmp[currentIdx + 8] = 0;
			JTb_block_tmp[currentIdx + 9] = 0;
			JTb_block_tmp[currentIdx + 10] = 0;
			JTb_block_tmp[currentIdx + 11] = 0;
		}
		else
		{
			nodeIdxCol1 = FindIndex(dVertexRelaIdxVec + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col, MAX_NEAR_NODE_NUM_VERTEX);
			nodeIdxCol2 = FindIndex(dVertexRelaIdxVec + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
			                        MAX_NEAR_NODE_NUM_VERTEX);
			if (nodeIdxCol1 > nodeIdxCol2)
			{
				nodeIdxCol = nodeIdxCol1;
				vertexIdxCol = srcVertexIdx;
				signCol = 1.0f;
			}
			else
			{
				nodeIdxCol = nodeIdxCol2;
				vertexIdxCol = targetVertexIdx;
				signCol = -1.0f;
			}
			targetNormal = dTargetNormalVec[targetVertexIdx];
			targetNormal.w = 0.0f;
			targetNormal = normalize(targetNormal);
			weight_V2N = signCol * weight * dVertexRelaWeightVec[vertexIdxCol * MAX_NEAR_NODE_NUM_VERTEX + nodeIdxCol] * dB[fIdx
			];
			vnn = dSrcVertexVec[vertexIdxCol] - nodePos;
			//printf("vnn: %f %f %f\n", vnn.x, vnn.y, vnn.z);
			tmp1 = weight_V2N * targetNormal.x;
			tmp2 = weight_V2N * targetNormal.y;
			tmp3 = weight_V2N * targetNormal.z;
			//printf("tmp: %f %f %f\n", tmp1, tmp2, tmp3);

			JTb_block_tmp[currentIdx + 0] = tmp1 * vnn.x;
			JTb_block_tmp[currentIdx + 1] = tmp2 * vnn.x;
			JTb_block_tmp[currentIdx + 2] = tmp3 * vnn.x;
			JTb_block_tmp[currentIdx + 3] = tmp1 * vnn.y;
			JTb_block_tmp[currentIdx + 4] = tmp2 * vnn.y;
			JTb_block_tmp[currentIdx + 5] = tmp3 * vnn.y;
			JTb_block_tmp[currentIdx + 6] = tmp1 * vnn.z;
			JTb_block_tmp[currentIdx + 7] = tmp2 * vnn.z;
			JTb_block_tmp[currentIdx + 8] = tmp3 * vnn.z;
			JTb_block_tmp[currentIdx + 9] = tmp1;
			JTb_block_tmp[currentIdx + 10] = tmp2;
			JTb_block_tmp[currentIdx + 11] = tmp3;
		}
	}

	__syncthreads();
	//reduction
	if (threadIdx.x < 12)
	{
		for (int iter_redu = 0; iter_redu < residualNum; iter_redu++)
		{
			JTb_block_res[threadIdx.x] += JTb_block_tmp[threadIdx.x + iter_redu * 12];
		}
	}

	__syncthreads();
	// write_back
	if (threadIdx.x < 12)
	{
		dJTb[col * 12 + threadIdx.x] -= JTb_block_res[threadIdx.x];
	}
}

void GeoConstraint::directiveJTb(thrust::device_vector<float>& JTb)
{
	int block = 256; // fixed!!!
	int grid = m_inputData->m_Iij.m_nnzIij;
	DirectiveJTbKernelGeo << <grid, block >> >(RAW_PTR(JTb),
	                                           RAW_PTR(m_dB),
	                                           RAW_PTR(m_inputData->m_Iij.m_dListIij),
	                                           RAW_PTR(m_inputData->m_Iij.m_dOffsetIij),
	                                           RAW_PTR(m_inputData->m_Iij.m_dNzIijCoo),
	                                           RAW_PTR(m_inputData->m_Iij.m_dDataItemNum),
	                                           RAW_PTR(m_inputData->m_source.m_dVertexVec),
	                                           RAW_PTR(m_inputData->m_source.m_dVertexRelaIdxVec),
	                                           RAW_PTR(m_inputData->m_source.m_dVertexRelaWeightVec),
	                                           RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                           m_inputData->m_source.m_nodeNum,
	                                           RAW_PTR(m_inputData->m_deformed.m_dNormalVec),
	                                           m_inputData->m_dMatchingPointIndices,
	                                           m_inputData->m_matchingPointNum,
	                                           m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
#if 0
	std::vector<float> JTbVec(m_inputData->m_source.m_nodeNum * 12);
	checkCudaErrors(cudaMemcpy(JTbVec.data(), RAW_PTR(JTb), JTbVec.size() * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < JTbVec.size(); ++i)
	{
		std::cout << JTbVec[i] << ", ";
	}
	std::cout << "----------------------------" << std::endl;
	exit(0);
#endif
}
