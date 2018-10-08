#pragma once

#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "PhotoConstraint.h"
#include "GNSolver.h"
#include "InputData.h"
#include "Helpers/UtilsMath.h"

__forceinline__ __device__ int FindIndex(int* node_list, int val, int num)
{
	for (int i = 0; i < num; i++)
	{
		if (node_list[i] == val)
		{
			return i;
		}
	}
	return -1;
}

bool PhotoConstraint::init(GNSolver* gnSolver, float weight)
{
	assert(gnSolver);
	m_gnSolver = gnSolver;
	m_inputData = gnSolver->m_inputData;
	setWeight(weight);
	return true;
}

bool PhotoConstraint::init()
{
	return true;
}

void PhotoConstraint::getJTJAndJTb(float* dJTJ_a, int* dJTJ_ia, thrust::device_vector<float>& dJTb, float3* dVars)
{
	if (m_iter > 0)
	{
		b(dVars);
		directiveJTJ(dJTJ_a, dJTJ_ia);
		directiveJTb(dJTb);
	}
}

__global__ void bKernelPhoto(float* dB,
                             int width,
                             int height,
                             float fx,
                             float fy,
                             float cx,
                             float cy,
                             int* dMatchingPointIndices,
                             int matchingPointNum,
                             float4* dVertexVec,
                             float* dKeyGrayImgs,
                             int grayimgStep,
                             float4* dUpdatedKeyPosesInv,
                             float weight)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= matchingPointNum)
	{
		return;
	}

	int srcVertexInd = dMatchingPointIndices[2 * idx];
	int targetVertexInd = dMatchingPointIndices[2 * idx + 1];

	if (srcVertexInd < 0 || targetVertexInd < 0)
	{
		dB[idx] = 0;
		return;
	}

	float4 *updatedPosSrc, *updatedPosTarget, *updatedNormalSrc, *updatedNormalTarget;
	float4 updatedPosSrcLocalTargetSpace, updatedPosTargetLocalSrcSpace;
	float4 updatedPosSrcLocal, updatedPosTargetLocal;
	float uBiSrc, vBiSrc,
	      uBiSrcTargetSpace, vBiSrcTargetSpace,
	      uBiTarget, vBiTarget,
	      uBiTargetSrcSpace, vBiTargetSrcSpace,
	      coef, valTop, valBottom;
	float biPixSrc, biPixSrcTargetSpace, biPixTarget, biPixTargetSrcSpace;
	int uBi0Src, uBi1Src, vBi0Src, vBi1Src,
	    uBi0SrcTargetSpace, uBi1SrcTargetSpace, vBi0SrcTargetSpace, vBi1SrcTargetSpace,
	    uBi0Target, uBi1Target, vBi0Target, vBi1Target,
	    uBi0TargetSrcSpace, uBi1TargetSrcSpace, vBi0TargetSrcSpace, vBi1TargetSrcSpace;

	updatedPosSrc = dVertexVec + srcVertexInd;
	int fragIndSrc = (int)updatedPosSrc->w;
	updatedPosTarget = dVertexVec + targetVertexInd;
	int fragIndTarget = (int)updatedPosTarget->w;
	float4* updatedPoseInvSrc = dUpdatedKeyPosesInv + fragIndSrc * 4;
	float4* updatedPoseInvTarget = dUpdatedKeyPosesInv + fragIndTarget * 4;

	updatedPosSrcLocalTargetSpace = updatedPosSrc->x * updatedPoseInvTarget[0] + updatedPosSrc->y * updatedPoseInvTarget[1]
		+
		updatedPosSrc->z * updatedPoseInvTarget[2] + updatedPoseInvTarget[3];
	updatedPosTargetLocalSrcSpace = updatedPosTarget->x * updatedPoseInvSrc[0] + updatedPosTarget->y * updatedPoseInvSrc[1]
		+
		updatedPosTarget->z * updatedPoseInvSrc[2] + updatedPoseInvSrc[3];

	float* keyGrayImgSrc = dKeyGrayImgs + fragIndSrc * grayimgStep;
	float* keyGrayImgTarget = dKeyGrayImgs + fragIndTarget * grayimgStep;

	// bilinear intarpolation
	uBiTargetSrcSpace = (updatedPosTargetLocalSrcSpace.x * fx) / updatedPosTargetLocalSrcSpace.z + cx;
	vBiTargetSrcSpace = (updatedPosTargetLocalSrcSpace.y * fy) / updatedPosTargetLocalSrcSpace.z + cy;
#if 0
	if (uBiTargetSrcSpace  < 0 || uBiTargetSrcSpace  > width - 2 || vBiTargetSrcSpace  < 0 || vBiTargetSrcSpace  > height - 2)
	{
		goto Invalid_Matching;
	}
#endif
	uBiTargetSrcSpace = clamp(uBiTargetSrcSpace, (float)0, (float)(width - 2));
	vBiTargetSrcSpace = clamp(vBiTargetSrcSpace, (float)0, (float)(height - 2));
	// bilinear intarpolation
	uBi0TargetSrcSpace = __float2int_rd(uBiTargetSrcSpace);
	uBi1TargetSrcSpace = uBi0TargetSrcSpace + 1;
	vBi0TargetSrcSpace = __float2int_rd(vBiTargetSrcSpace);
	vBi1TargetSrcSpace = vBi0TargetSrcSpace + 1;
	coef = (uBi1TargetSrcSpace - uBiTargetSrcSpace) / (float)(uBi1TargetSrcSpace - uBi0TargetSrcSpace);
	valTop = coef * ((float)*(keyGrayImgSrc + vBi0TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgSrc + vBi0TargetSrcSpace * width + uBi1TargetSrcSpace));
	valBottom = coef * ((float)*(keyGrayImgSrc + vBi1TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgSrc + vBi1TargetSrcSpace * width + uBi1TargetSrcSpace));
	coef = (vBi1TargetSrcSpace - vBiTargetSrcSpace) / (float)(vBi1TargetSrcSpace - vBi0TargetSrcSpace);
	biPixTargetSrcSpace = coef * valTop + (1 - coef) * valBottom;

	uBiSrcTargetSpace = (updatedPosSrcLocalTargetSpace.x * fx) / updatedPosSrcLocalTargetSpace.z + cx;
	vBiSrcTargetSpace = (updatedPosSrcLocalTargetSpace.y * fy) / updatedPosSrcLocalTargetSpace.z + cy;
#if 0
	if (uBiSrcTargetSpace < 0 || uBiSrcTargetSpace > width - 2 || vBiSrcTargetSpace < 0 || vBiSrcTargetSpace > height - 2)
	{
		goto Invalid_Matching;
	}
#endif
	uBiSrcTargetSpace = clamp(uBiSrcTargetSpace, (float)0, (float)(width - 2));
	vBiSrcTargetSpace = clamp(vBiSrcTargetSpace, (float)0, (float)(height - 2));
	// bilinear intarpolation
	uBi0SrcTargetSpace = __float2int_rd(uBiSrcTargetSpace);
	uBi1SrcTargetSpace = uBi0SrcTargetSpace + 1;
	vBi0SrcTargetSpace = __float2int_rd(vBiSrcTargetSpace);
	vBi1SrcTargetSpace = vBi0SrcTargetSpace + 1;
	coef = (uBi1SrcTargetSpace - uBiSrcTargetSpace) / (float)(uBi1SrcTargetSpace - uBi0SrcTargetSpace);
	valTop = coef * ((float)*(keyGrayImgTarget + vBi0SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgTarget + vBi0SrcTargetSpace * width + uBi1SrcTargetSpace));
	valBottom = coef * ((float)*(keyGrayImgTarget + vBi1SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgTarget + vBi1SrcTargetSpace * width + uBi1SrcTargetSpace));
	coef = (vBi1SrcTargetSpace - vBiSrcTargetSpace) / (float)(vBi1SrcTargetSpace - vBi0SrcTargetSpace);
	biPixSrcTargetSpace = coef * valTop + (1 - coef) * valBottom;

	//printf("diff: %f %f, ", biPixSrcTargetSpace, biPixTargetSrcSpace);
	dB[idx] = weight * (biPixSrcTargetSpace - biPixTargetSrcSpace);
	//printf("r: %f %f %f %f %f %f\n", biPixTarget, biPixSrcTargetSpace, middlePixIntensityTargetSpace, biPixSrc, biPixTargetSrcSpace, middlePixIntensitySrcSpace);
}

void PhotoConstraint::b(float3* dVars)
{
	m_dB.resize(m_inputData->m_matchingPointNum);

	int block = 1024;
	int grid = (block + m_inputData->m_matchingPointNum - 1) / block;
	int width = Resolution::getInstance().width(), height = Resolution::getInstance().height();
	bKernelPhoto << <grid, block >> >(RAW_PTR(m_dB),
	                                  width,
	                                  height,
	                                  Intrinsics::getInstance().fx(),
	                                  Intrinsics::getInstance().fy(),
	                                  Intrinsics::getInstance().cx(),
	                                  Intrinsics::getInstance().cy(),
	                                  m_inputData->m_dMatchingPointIndices,
	                                  m_inputData->m_matchingPointNum,
	                                  RAW_PTR(m_inputData->m_deformed.m_dVertexVec),
	                                  m_inputData->m_dKeyGrayImgs,
	                                  width * height,
	                                  m_inputData->m_dUpdatedKeyPosesInv,
	                                  m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__forceinline__ __device__ void CalculateImgDerivative(float* Jc,
                                                       int nodeInd,
                                                       int vertexInd,
                                                       float sign,
                                                       int fragIndTarget,
                                                       int width, int height, float fx, float fy, float cx, float cy,
                                                       float* keyGrayImgsDx, int keyGrayImgsDxStep,
                                                       float* keyGrayImgsDy, int keyGrayImgsDyStep,
                                                       float4* updatedPosesInv,
                                                       float4* originVertexPoses,
                                                       float4* updatedVertexPoses,
                                                       int* vertexToNodeIndices,
                                                       float* vertexToNodeWeights,
                                                       float4* dNodeVec,
                                                       int nodeNum,
                                                       float w_photo)
{
	// Src stands for row, target stands for col here
	float4 *updatedPosSrc, *updatedPosTarget, *updatedNormalSrc, *updatedNormalTarget;
	float4 updatedPosSrcLocalTargetSpace, updatedPosTargetLocalSrcSpace;
	float4 updatedPosSrcLocal, updatedPosTargetLocal;
	float uBiSrc, vBiSrc,
	      uBiSrcTargetSpace, vBiSrcTargetSpace,
	      uBiTarget, vBiTarget,
	      uBiTargetSrcSpace, vBiTargetSrcSpace,
	      coef, valTop, valBottom;
	float biPixSrc, biPixSrcTargetSpace, biPixTarget, biPixTargetSrcSpace;
	int uBi0Src, uBi1Src, vBi0Src, vBi1Src,
	    uBi0SrcTargetSpace, uBi1SrcTargetSpace, vBi0SrcTargetSpace, vBi1SrcTargetSpace,
	    uBi0Target, uBi1Target, vBi0Target, vBi1Target,
	    uBi0TargetSrcSpace, uBi1TargetSrcSpace, vBi0TargetSrcSpace, vBi1TargetSrcSpace;

	updatedPosSrc = updatedVertexPoses + vertexInd;
	float4* updatedPoseInvTarget = updatedPosesInv + fragIndTarget * 4;

	updatedPosSrcLocalTargetSpace = updatedPosSrc->x * updatedPoseInvTarget[0] + updatedPosSrc->y * updatedPoseInvTarget[1]
		+
		updatedPosSrc->z * updatedPoseInvTarget[2] + updatedPoseInvTarget[3];

	float* keyGrayImgDxDeviceTarget = keyGrayImgsDx + fragIndTarget * keyGrayImgsDxStep;
	float* keyGrayImgDyDeviceTarget = keyGrayImgsDy + fragIndTarget * keyGrayImgsDyStep;

	uBiSrcTargetSpace = (updatedPosSrcLocalTargetSpace.x * fx) / updatedPosSrcLocalTargetSpace.z + cx;
	vBiSrcTargetSpace = (updatedPosSrcLocalTargetSpace.y * fy) / updatedPosSrcLocalTargetSpace.z + cy;
	uBiSrcTargetSpace = clamp(uBiSrcTargetSpace, (float)0, (float)(width - 2));
	vBiSrcTargetSpace = clamp(vBiSrcTargetSpace, (float)0, (float)(height - 2));

	// bilinear intarpolation
	uBi0SrcTargetSpace = __float2int_rd(uBiSrcTargetSpace);
	uBi1SrcTargetSpace = uBi0SrcTargetSpace + 1;
	vBi0SrcTargetSpace = __float2int_rd(vBiSrcTargetSpace);
	vBi1SrcTargetSpace = vBi0SrcTargetSpace + 1;

	// compute jacobian geo and photo
	// d_gamma_uv, d_gamm_xyz, d_gamma_Rt
	float3 J_gamma_xyz_src, J_gamma_xyz_target;
	float3 J_gamma_xyz_src_target_space, J_gamma_xyz_target_src_space;
	float3 J_gamma_xyz_global_src, J_gamma_xyz_global_target;
	float3 J_gamma_xyz_global_src_target_space, J_gamma_xyz_global_target_src_space;
	float2 J_gamma_uv_src, J_gamma_uv_src_target_space,
	       J_gamma_uv_target, J_gamma_uv_target_src_space;

	float dx, dy;
#if USE_BILINEAR_TO_CALC_GRAD
	coef = (uBi1SrcTargetSpace - uBi0SrcTargetSpace)*(vBi1SrcTargetSpace - vBi0SrcTargetSpace);
	J_gamma_uv_src_target_space =
		make_float2(-(vBi1SrcTargetSpace - vBiSrcTargetSpace) / coef, -(uBi1SrcTargetSpace - uBiSrcTargetSpace) / coef) *
		((float)*(keyGrayImgTarget + vBi0SrcTargetSpace  * width + uBi0SrcTargetSpace)) +
		make_float2((vBi1SrcTargetSpace - vBiSrcTargetSpace) / coef, -(uBiSrcTargetSpace - uBi0SrcTargetSpace) / coef) *
		((float)*(keyGrayImgTarget + vBi0SrcTargetSpace  * width + uBi1SrcTargetSpace)) +
		make_float2(-(vBiSrcTargetSpace - vBi0SrcTargetSpace) / coef, (uBi1SrcTargetSpace - uBiSrcTargetSpace) / coef) *
		((float)*(keyGrayImgTarget + vBi1SrcTargetSpace  * width + uBi0SrcTargetSpace)) +
		make_float2((vBiSrcTargetSpace - vBi0SrcTargetSpace) / coef, (uBiSrcTargetSpace - uBi0SrcTargetSpace) / coef) *
		((float)*(keyGrayImgTarget + vBi1SrcTargetSpace  * width + uBi1SrcTargetSpace));
#else
	coef = (uBi1SrcTargetSpace - uBiSrcTargetSpace) / (float)(uBi1SrcTargetSpace - uBi0SrcTargetSpace);
	valTop = coef * ((float)*(keyGrayImgDxDeviceTarget + vBi0SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDxDeviceTarget + vBi0SrcTargetSpace * width + uBi1SrcTargetSpace));
	valBottom = coef * ((float)*(keyGrayImgDxDeviceTarget + vBi1SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDxDeviceTarget + vBi1SrcTargetSpace * width + uBi1SrcTargetSpace));
	coef = (vBi1SrcTargetSpace - vBiSrcTargetSpace) / (float)(vBi1SrcTargetSpace - vBi0SrcTargetSpace);
	dx = coef * valTop + (1 - coef) * valBottom;
	coef = (uBi1SrcTargetSpace - uBiSrcTargetSpace) / (float)(uBi1SrcTargetSpace - uBi0SrcTargetSpace);
	valTop = coef * ((float)*(keyGrayImgDyDeviceTarget + vBi0SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDyDeviceTarget + vBi0SrcTargetSpace * width + uBi1SrcTargetSpace));
	valBottom = coef * ((float)*(keyGrayImgDyDeviceTarget + vBi1SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDyDeviceTarget + vBi1SrcTargetSpace * width + uBi1SrcTargetSpace));
	coef = (vBi1SrcTargetSpace - vBiSrcTargetSpace) / (float)(vBi1SrcTargetSpace - vBi0SrcTargetSpace);
	dy = coef * valTop + (1 - coef) * valBottom;
	J_gamma_uv_src_target_space = make_float2(dx, dy);
#endif
	J_gamma_xyz_src_target_space = make_float3(J_gamma_uv_src_target_space.x * fx / updatedPosSrcLocalTargetSpace.z,
	                                           J_gamma_uv_src_target_space.y * fy / updatedPosSrcLocalTargetSpace.z,
	                                           (-J_gamma_uv_src_target_space.x * updatedPosSrcLocalTargetSpace.x * fx -
		                                           J_gamma_uv_src_target_space.y * updatedPosSrcLocalTargetSpace.y * fy) / (
		                                           updatedPosSrcLocalTargetSpace.z * updatedPosSrcLocalTargetSpace.z));
	J_gamma_xyz_global_src_target_space = J_gamma_xyz_src_target_space.x * make_float3(
			updatedPoseInvTarget[0].x, updatedPoseInvTarget[0].y, updatedPoseInvTarget[0].z) +
		J_gamma_xyz_src_target_space.y * make_float3(updatedPoseInvTarget[1].x, updatedPoseInvTarget[1].y,
		                                             updatedPoseInvTarget[1].z) +
		J_gamma_xyz_src_target_space.z * make_float3(updatedPoseInvTarget[2].x, updatedPoseInvTarget[2].y,
		                                             updatedPoseInvTarget[2].z);

	float tmp1, tmp2, tmp3, weight_V2N;
	float4 vnn;
	weight_V2N = sign * w_photo * vertexToNodeWeights[vertexInd * 4 + nodeInd];
	vnn = originVertexPoses[vertexInd] -
		dNodeVec[vertexToNodeIndices[vertexInd * 4 + nodeInd]];

	tmp1 = weight_V2N * J_gamma_xyz_global_src_target_space.x;
	tmp2 = weight_V2N * J_gamma_xyz_global_src_target_space.y;
	tmp3 = weight_V2N * J_gamma_xyz_global_src_target_space.z;

	//printf("%f %f %f", tmp1, tmp2, tmp3);
	Jc[0] = tmp1 * vnn.x;
	Jc[1] = tmp2 * vnn.x;
	Jc[2] = tmp3 * vnn.x;
	Jc[3] = tmp1 * vnn.y;
	Jc[4] = tmp2 * vnn.y;
	Jc[5] = tmp3 * vnn.y;
	Jc[6] = tmp1 * vnn.z;
	Jc[7] = tmp2 * vnn.z;
	Jc[8] = tmp3 * vnn.z;
	Jc[9] = tmp1;
	Jc[10] = tmp2;
	Jc[11] = tmp3;
}

__global__ void DirectiveJTJKernelPhoto(float* dJTJ_a,
                                        int* dJTJ_ia,
                                        int width, int height, float fx, float fy, float cx, float cy,
                                        int* dListIij,
                                        int* dOffsetIij,
                                        int* dNzIijCoo,
                                        int* dNnzPre,
                                        int* dDataItemNum,
                                        float* dKeyGrayImgsDx, int keyGrayImgsDxStep,
                                        float* dKeyGrayImgsDy, int keyGrayImgsDyStep,
                                        float4* dSrcVertexVec,
                                        int* dVertexRelaIdxVec,
                                        float* dVertexRelaWeightVec,
                                        float4* dNodeVec,
                                        int nodeNum,
                                        float4* dDeformedVertexVec,
                                        float4* dUpdatedKeyPosesInv,
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
			int fragIdxOtherRow, fragIdxOtherCol;
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
					fragIdxOtherRow = (int)dDeformedVertexVec[targetVertexIdx].w;
					fragIdxOtherCol = (int)dDeformedVertexVec[targetVertexIdx].w;
				}
				else
				{
					nodeIdxRow = nodeIdxRow2;
					nodeIdxCol = nodeIdxCol2;
					vertexIdxRow = targetVertexIdx;
					vertexIdxCol = targetVertexIdx;
					fragIdxOtherRow = (int)dDeformedVertexVec[srcVertexIdx].w;
					fragIdxOtherCol = (int)dDeformedVertexVec[srcVertexIdx].w;
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
					fragIdxOtherRow = (int)dDeformedVertexVec[targetVertexIdx].w;
					fragIdxOtherCol = (int)dDeformedVertexVec[srcVertexIdx].w;
				}
				else
				{
					nodeIdxRow = nodeIdxRow2;
					nodeIdxCol = nodeIdxCol1;
					vertexIdxRow = targetVertexIdx;
					vertexIdxCol = srcVertexIdx;
					signRow = -1.0f;
					fragIdxOtherRow = (int)dDeformedVertexVec[srcVertexIdx].w;
					fragIdxOtherCol = (int)dDeformedVertexVec[targetVertexIdx].w;
				}
			}

			CalculateImgDerivative(Jcj + currentIdx,
			                       nodeIdxCol,
			                       vertexIdxCol,
			                       signCol,
			                       fragIdxOtherCol,
			                       width, height, fx, fy, cx, cy,
			                       dKeyGrayImgsDx, keyGrayImgsDxStep,
			                       dKeyGrayImgsDy, keyGrayImgsDyStep,
			                       dUpdatedKeyPosesInv,
			                       dSrcVertexVec,
			                       dDeformedVertexVec,
			                       dVertexRelaIdxVec,
			                       dVertexRelaWeightVec,
			                       dNodeVec,
			                       nodeNum,
			                       weight);
			CalculateImgDerivative(Jci + currentIdx,
			                       nodeIdxRow,
			                       vertexIdxRow,
			                       signRow,
			                       fragIdxOtherRow,
			                       width, height, fx, fy, cx, cy,
			                       dKeyGrayImgsDx, keyGrayImgsDxStep,
			                       dKeyGrayImgsDy, keyGrayImgsDyStep,
			                       dUpdatedKeyPosesInv,
			                       dSrcVertexVec,
			                       dDeformedVertexVec,
			                       dVertexRelaIdxVec,
			                       dVertexRelaWeightVec,
			                       dNodeVec,
			                       nodeNum,
			                       weight);
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
			int fragIdxOtherRow, fragIdxOtherCol;
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
					fragIdxOtherRow = (int)dDeformedVertexVec[targetVertexIdx].w;
					fragIdxOtherCol = (int)dDeformedVertexVec[targetVertexIdx].w;
				}
				else
				{
					nodeIdxRow = nodeIdxRow2;
					nodeIdxCol = nodeIdxCol2;
					vertexIdxRow = targetVertexIdx;
					vertexIdxCol = targetVertexIdx;
					fragIdxOtherRow = (int)dDeformedVertexVec[srcVertexIdx].w;
					fragIdxOtherCol = (int)dDeformedVertexVec[srcVertexIdx].w;
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
					fragIdxOtherRow = (int)dDeformedVertexVec[targetVertexIdx].w;
					fragIdxOtherCol = (int)dDeformedVertexVec[srcVertexIdx].w;
				}
				else
				{
					nodeIdxRow = nodeIdxRow2;
					nodeIdxCol = nodeIdxCol1;
					vertexIdxRow = targetVertexIdx;
					vertexIdxCol = srcVertexIdx;
					signRow = -1.0f;
					fragIdxOtherRow = (int)dDeformedVertexVec[srcVertexIdx].w;
					fragIdxOtherCol = (int)dDeformedVertexVec[targetVertexIdx].w;
				}
			}

			CalculateImgDerivative(Jcj + currentIdx,
			                       nodeIdxCol,
			                       vertexIdxCol,
			                       signCol,
			                       fragIdxOtherCol,
			                       width, height, fx, fy, cx, cy,
			                       dKeyGrayImgsDx, keyGrayImgsDxStep,
			                       dKeyGrayImgsDy, keyGrayImgsDyStep,
			                       dUpdatedKeyPosesInv,
			                       dSrcVertexVec,
			                       dDeformedVertexVec,
			                       dVertexRelaIdxVec,
			                       dVertexRelaWeightVec,
			                       dNodeVec,
			                       nodeNum,
			                       weight);
			CalculateImgDerivative(Jci + currentIdx,
			                       nodeIdxRow,
			                       vertexIdxRow,
			                       signRow,
			                       fragIdxOtherRow,
			                       width, height, fx, fy, cx, cy,
			                       dKeyGrayImgsDx, keyGrayImgsDxStep,
			                       dKeyGrayImgsDy, keyGrayImgsDyStep,
			                       dUpdatedKeyPosesInv,
			                       dSrcVertexVec,
			                       dDeformedVertexVec,
			                       dVertexRelaIdxVec,
			                       dVertexRelaWeightVec,
			                       dNodeVec,
			                       nodeNum,
			                       weight);
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

void PhotoConstraint::directiveJTJ(float* dJTJ_a, int* dJTJ_ia)
{
	//std::cout << m_weight << std::endl;
	//std::exit(0);
	int block = 160; // fixed !!!
	int grid = m_inputData->m_Iij.m_nnzIij; //each CUDA block handle one JTJ block.
	int width = Resolution::getInstance().width(), height = Resolution::getInstance().height();
	DirectiveJTJKernelPhoto << <grid, block >> >(dJTJ_a,
	                                             dJTJ_ia,
	                                             width,
	                                             height,
	                                             Intrinsics::getInstance().fx(),
	                                             Intrinsics::getInstance().fy(),
	                                             Intrinsics::getInstance().cx(),
	                                             Intrinsics::getInstance().cy(),
	                                             RAW_PTR(m_inputData->m_Iij.m_dListIij),
	                                             RAW_PTR(m_inputData->m_Iij.m_dOffsetIij),
	                                             RAW_PTR(m_inputData->m_Iij.m_dNzIijCoo),
	                                             RAW_PTR(m_inputData->m_Iij.m_dNnzPre),
	                                             RAW_PTR(m_inputData->m_Iij.m_dDataItemNum),
	                                             m_inputData->m_dKeyGrayImgsDx, width * height,
	                                             m_inputData->m_dKeyGrayImgsDy, width * height,
	                                             RAW_PTR(m_inputData->m_source.m_dVertexVec),
	                                             RAW_PTR(m_inputData->m_source.m_dVertexRelaIdxVec),
	                                             RAW_PTR(m_inputData->m_source.m_dVertexRelaWeightVec),
	                                             RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                             m_inputData->m_source.m_nodeNum,
	                                             RAW_PTR(m_inputData->m_deformed.m_dVertexVec),
	                                             m_inputData->m_dUpdatedKeyPosesInv,
	                                             m_inputData->m_dMatchingPointIndices,
	                                             m_inputData->m_matchingPointNum,
	                                             m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void DirectiveJTbKernelPhoto(float* dJtb,
                                        float* dB,
                                        int width,
                                        int height,
                                        float fx,
                                        float fy,
                                        float cx,
                                        float cy,
                                        int* dListIij,
                                        int* dOffsetIij,
                                        int* dNzIijCoo,
                                        int* dDataItemNum,
                                        float* dKeyGrayImgsDx, int keyGrayImgsDxStep,
                                        float* dKeyGrayImgsDy, int keyGrayImgsDyStep,
                                        float4* dSrcVertexVec,
                                        int* dVertexRelaIdxVec,
                                        float* dVertexRelaWeightVec,
                                        float4* dNodeVec,
                                        int nodeNum,
                                        float4* dDeformedVertexVec,
                                        float4* dUpdatedKeyPosesInv,
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
	int fragIdxOtherCol;
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
			nodeIdxCol1 = FindIndex(dVertexRelaIdxVec + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
			                        MAX_NEAR_NODE_NUM_VERTEX);
			nodeIdxCol2 = FindIndex(dVertexRelaIdxVec + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
			                        MAX_NEAR_NODE_NUM_VERTEX);
			if (nodeIdxCol1 > nodeIdxCol2)
			{
				nodeIdxCol = nodeIdxCol1;
				vertexIdxCol = srcVertexIdx;
				signCol = 1.0f;
				fragIdxOtherCol = (int)dDeformedVertexVec[targetVertexIdx].w;
			}
			else
			{
				nodeIdxCol = nodeIdxCol2;
				vertexIdxCol = targetVertexIdx;
				signCol = -1.0f;
				fragIdxOtherCol = (int)dDeformedVertexVec[srcVertexIdx].w;
			}

#if 1
			CalculateImgDerivative(
				JTb_block_tmp + currentIdx,
				nodeIdxCol,
				vertexIdxCol,
				signCol,
				fragIdxOtherCol,
				width, height, fx, fy, cx, cy,
				dKeyGrayImgsDx, keyGrayImgsDxStep,
				dKeyGrayImgsDy, keyGrayImgsDyStep,
				dUpdatedKeyPosesInv,
				dSrcVertexVec,
				dDeformedVertexVec,
				dVertexRelaIdxVec,
				dVertexRelaWeightVec,
				dNodeVec,
				nodeNum,
				weight);
#endif

#if 1
			float bVal = dB[fIdx];
			JTb_block_tmp[currentIdx + 0] *= bVal;
			JTb_block_tmp[currentIdx + 1] *= bVal;
			JTb_block_tmp[currentIdx + 2] *= bVal;
			JTb_block_tmp[currentIdx + 3] *= bVal;
			JTb_block_tmp[currentIdx + 4] *= bVal;
			JTb_block_tmp[currentIdx + 5] *= bVal;
			JTb_block_tmp[currentIdx + 6] *= bVal;
			JTb_block_tmp[currentIdx + 7] *= bVal;
			JTb_block_tmp[currentIdx + 8] *= bVal;
			JTb_block_tmp[currentIdx + 9] *= bVal;
			JTb_block_tmp[currentIdx + 10] *= bVal;
			JTb_block_tmp[currentIdx + 11] *= bVal;
#endif
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

	__syncthreads();
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
			nodeIdxCol1 = FindIndex(dVertexRelaIdxVec + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
				MAX_NEAR_NODE_NUM_VERTEX);
			nodeIdxCol2 = FindIndex(dVertexRelaIdxVec + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX, col,
				MAX_NEAR_NODE_NUM_VERTEX);
			if (nodeIdxCol1 > nodeIdxCol2)
			{
				nodeIdxCol = nodeIdxCol1;
				vertexIdxCol = srcVertexIdx;
				signCol = 1.0f;
				fragIdxOtherCol = (int)dDeformedVertexVec[targetVertexIdx].w;
			}
			else
			{
				nodeIdxCol = nodeIdxCol2;
				vertexIdxCol = targetVertexIdx;
				signCol = -1.0f;
				fragIdxOtherCol = (int)dDeformedVertexVec[srcVertexIdx].w;
			}

#if 1
			CalculateImgDerivative(
				JTb_block_tmp + currentIdx,
				nodeIdxCol,
				vertexIdxCol,
				signCol,
				fragIdxOtherCol,
				width, height, fx, fy, cx, cy,
				dKeyGrayImgsDx, keyGrayImgsDxStep,
				dKeyGrayImgsDy, keyGrayImgsDyStep,
				dUpdatedKeyPosesInv,
				dSrcVertexVec,
				dDeformedVertexVec,
				dVertexRelaIdxVec,
				dVertexRelaWeightVec,
				dNodeVec,
				nodeNum,
				weight);
#endif

#if 1
			float bVal = dB[fIdx];
			JTb_block_tmp[currentIdx + 0] *= bVal;
			JTb_block_tmp[currentIdx + 1] *= bVal;
			JTb_block_tmp[currentIdx + 2] *= bVal;
			JTb_block_tmp[currentIdx + 3] *= bVal;
			JTb_block_tmp[currentIdx + 4] *= bVal;
			JTb_block_tmp[currentIdx + 5] *= bVal;
			JTb_block_tmp[currentIdx + 6] *= bVal;
			JTb_block_tmp[currentIdx + 7] *= bVal;
			JTb_block_tmp[currentIdx + 8] *= bVal;
			JTb_block_tmp[currentIdx + 9] *= bVal;
			JTb_block_tmp[currentIdx + 10] *= bVal;
			JTb_block_tmp[currentIdx + 11] *= bVal;
#endif
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
		dJtb[col * 12 + threadIdx.x] -= JTb_block_res[threadIdx.x];
	}
}

void PhotoConstraint::directiveJTb(thrust::device_vector<float>& JTb)
{
	int block = 256; // fixed!!!
	int grid = m_inputData->m_Iij.m_nnzIij;
	int width = Resolution::getInstance().width(), height = Resolution::getInstance().height();
	DirectiveJTbKernelPhoto << <grid, block >> >(RAW_PTR(JTb),
	                                             RAW_PTR(m_dB),
	                                             width,
	                                             height,
	                                             Intrinsics::getInstance().fx(),
	                                             Intrinsics::getInstance().fy(),
	                                             Intrinsics::getInstance().cx(),
	                                             Intrinsics::getInstance().cy(),
	                                             RAW_PTR(m_inputData->m_Iij.m_dListIij),
	                                             RAW_PTR(m_inputData->m_Iij.m_dOffsetIij),
	                                             RAW_PTR(m_inputData->m_Iij.m_dNzIijCoo),
	                                             RAW_PTR(m_inputData->m_Iij.m_dDataItemNum),
	                                             m_inputData->m_dKeyGrayImgsDx, width * height,
	                                             m_inputData->m_dKeyGrayImgsDy, width * height,
	                                             RAW_PTR(m_inputData->m_source.m_dVertexVec),
	                                             RAW_PTR(m_inputData->m_source.m_dVertexRelaIdxVec),
	                                             RAW_PTR(m_inputData->m_source.m_dVertexRelaWeightVec),
	                                             RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                             m_inputData->m_source.m_nodeNum,
	                                             RAW_PTR(m_inputData->m_deformed.m_dVertexVec),
	                                             m_inputData->m_dUpdatedKeyPosesInv,
	                                             m_inputData->m_dMatchingPointIndices,
	                                             m_inputData->m_matchingPointNum,
	                                             m_weight);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

