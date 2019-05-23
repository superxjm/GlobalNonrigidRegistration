#ifndef __DEFORMATION_CUDA_FUNCS_2_H__
#define __DEFORMATION_CUDA_FUNCS_2_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "Helpers/xUtils.h"

class JTJBlockStatistics
{
public:
	JTJBlockStatistics()
	{
		
	}
	~JTJBlockStatistics()
	{
		
	}

public:
	int m_nonZeroBlockNum;         // down-triangle
	thrust::host_vector<int> m_nonZeroBlockVec;     // node_num * node_num
	thrust::device_vector<int> m_equIndVecDevice;       // equations list
	thrust::device_vector<int> m_equIndOffsetVecDevice;   // node_num * node_num
	thrust::device_vector<int4> m_nonZeroBlockInfoCooDevice;  // nonZeroBlockNum * (2 + SparsePatternNum), row col num1 num2 ...

	thrust::device_vector<int> m_columnIndEachRowDevice;      // to determin block's position in csr.
};

__global__ void FilterInvalidMatchingPointsKernel(
	int *matchingPoints,
	int matchingPointsNumDescriptor,
	int matchingPointsNumTotal,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	int iter);
void FilterInvalidMatchingPoints(
	int *matchingPointsDevice,
	int matchingPointsNumDescriptor,
	int matchingPointsNumTotal,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	float distThresh1, 
	float distThresh2, 
	float angleThresh,	
	int iter);

__global__ void FindNonzeroJTJBlockVNKernel(int *nonzeroJTJBlock,
	int *matchingPoints,
	int *vertexToNodeIndices,
	int matchingPointsNum,
	int nodeNum);
__global__ void FindNonzeroJTJBlockNNKernel(int *nonzeroJTJBlock,
	int *nodeToNodeIndicesDevice,
	int nodeNum);

__global__ void FillNodeToNodePermutationIndVec(int *regTermNodeIndVec,
	int *regTermEquIndVec,
	int *nodeToNodeIndices,
	int nodeNum,
	int offset);
__global__ void FillVertexToNodePermutationIndVec(int *dataTermNodeIndVec,
	int *dataTermEquIndVec,
	int *matchingPoints,
	int *vertexToNodeIndices,
	int matchingPointsNum);
__global__ void CountEquNumKernel(int *equNumEachNodeVec,
	int *equIndPermutationVec,
	int equPermutationLength);
__global__ void CalcEquIndForNonZeroBlocksKernel(
	int* equIndVec,
	int4* nonZeroBlockInfoCoo,
	int* equIndOffsetVec,
	int *equIndPermutationVec,
	int *equNumEachNodeVec,
	int *exScanEquNumEachNodeVec,
	int nonZeroBlockNum,
	int nodeNum,
	int matchingPointsNum);
__global__ void CountInvalidMatchingPoints(int *invalidMatchingPointNum,
	int *matchingPoints,
	int matchingPointsNum);
void ComputeNonZeroJTJBlockStatistics(
	int &validMathingPointsNum,
	JTJBlockStatistics &blockStatistics,
	int *matchingPointsDevice,
	int matchingPointsNumDescriptor,
	int matchingPointsNumNearest,
	int matchingPointsNumTotal,
	int nodeNum,
	int *nodeToNodeIndicesDevice,
	int *vertexToNodeIndicesDevice);

void ComputeJTJiaAndja(CSRType &JTJ,
	int nodeNum,
	JTJBlockStatistics &blockStatistics);
void ComputeJTJAndJTResidual(CSRType &JTJ, float *JTResidual,
	float *residual,
	JTJBlockStatistics &blockStatistics,
	int width, int height, float fx, float fy, float cx, float cy,
	int *matchingPointsDevice,
	int matchingPointsNumDescriptor,
	int matchingPointsNumNearest,
	int matchingPointsNumTotal,
	int validMatchingPointsNum,
	int nodeNum,
	float4 *originVertexPosesDevice,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	std::pair<float *, int> &keyGrayImgsDevice,
	std::pair<float *, int> &keyGrayImgsDxDevice,
	std::pair<float *, int> &keyGrayImgsDyDevice,
	float4 *updatedKeyPosesInvDevice,
	int *nodeVIndicesDevice,
	int *nodeToNodeIndicesDevice,
	float *nodeToNodeWeightsDevice,
	int *vertexToNodeIndicesDevice,
	float *vertexToNodeWeightsDevice,
	float3 * Rts,
	float w_geo, float w_photo, float w_reg, float w_rot, float w_trans,
	int iter);
void ComputeJTJAndJTResidualGeoTermPointToPlain(CSRType &JTJ, float *JTResidual,
	float *residual,
	JTJBlockStatistics &blockStatistics,
	int *matchingPointsDevice,
	int matchingPointsNumDescriptor,
	int matchingPointsNumNearest,
	int matchingPointsNumTotal,
	int nodeNum,
	float4 *originVertexPosesDevice,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	int *nodeVIndicesDevice,
	int *nodeToNodeIndicesDevice,
	float *nodeToNodeWeightsDevice,
	int *vertexToNodeIndicesDevice,
	float *vertexToNodeWeightsDevice,
	float w_geo);
__forceinline__ __device__ int FindIndex(int *node_list, int val, int num);
__global__ void ComputeJTJGeoTermPointToPlainKernel(
	float *JTJ_a,
	int *JTJ_ia,
	int *equIndVec,
	int *equIndOffsetVec,
	int4 *nonZeroBlockInfoCoo,
	int *columnIndEachRow,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	int *vertexToNodeIndices,
	float *vertexToNodeWeights,
	int *nodeVIndices,
	int nodeNum,
	float w_geo);
__global__ void ComputeResidualGeoTermPointToPlainKernel(
	float *residual,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float w_geo);
__global__ void ComputeJTResidualGeoTermPointToPlainKernel(
	float *JTResidual,
	float *residual,
	int *equIndVec,
	int *equIndOffsetVec,
	int4 *nonZeroBlockInfoCoo,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	int *vertexToNodeIndices,
	float *vertexToNodeWeights,
	int *nodeVIndices,
	int nodeNum,
	float w_geo);

void ComputeJTJAndJTResidualPhotoTerm(CSRType &JTJ, float *JTResidual,
	float *residual,
	JTJBlockStatistics &blockStatistics,
	int width, int height, float fx, float fy, float cx, float cy,
	int *matchingPointsDevice,
	int matchingPointsNumDescriptor,
	int matchingPointsNumNearest,
	int matchingPointsNumTotal,
	int nodeNum,
	float4 *originVertexPosesDevice,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	std::pair<float *, int> &keyGrayImgsDevice,
	std::pair<float *, int> &keyGrayImgsDxDevice,
	std::pair<float *, int> &keyGrayImgsDyDevice,
	float4 *updatedKeyPosesInvDevice,
	int *nodeVIndicesDevice,
	int *nodeToNodeIndicesDevice,
	float *nodeToNodeWeightsDevice,
	int *vertexToNodeIndicesDevice,
	float *vertexToNodeWeightsDevice,
	float w_photo);
__global__ void ComputeJTJPhotoTermKernel(
	float *JTJ_a,
	int *JTJ_ia,
	int width, int height, float fx, float fy, float cx, float cy,
	int *equIndVec,
	int *equIndOffsetVec,
	int4 *nonZeroBlockInfoCoo,
	int *columnIndEachRow,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,	
	float *keyGrayImgsDx, int keyGrayImgsDxStep,
	float *keyGrayImgsDy, int keyGrayImgsDyStep,
	float4 *updatedPosesInv,
	int *vertexToNodeIndices,
	float *vertexToNodeWeights,
	int *nodeVIndices,
	int nodeNum,
	float w_photo);
__global__ void ComputeResidualPhotoTermKernel(
	float *residual,
	int width, int height, float fx, float fy, float cx, float cy,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *updatedVertexPoses,
	float *keyGrayImgs, int keykeyGrayImgsStep,
	float4 *updatedPosesInv,
	float w_photo);
__global__ void ComputeJTResidualPhotoTermKernel(
	float *JTResidual,
	float *residual,
	int width, int height, float fx, float fy, float cx, float cy,
	int *equIndVec,
	int *equIndOffsetVec,
	int4 *nonZeroBlockInfoCoo,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	float *keyGrayImgsDx, int keyGrayImgsDxStep,
	float *keyGrayImgsDy, int keyGrayImgsDyStep,
	float4 *updatedPosesInv,
	int *vertexToNodeIndices,
	float *vertexToNodeWeights,
	int *nodeVIndices,
	int nodeNum,
	float w_photo);
__forceinline__ __device__ void CalculateImgDerivative(
	float *Jc,
	int nodeInd,
	int vertexInd,
	float sign,
	int fragIndTarget,
	int width, int height, float fx, float fy, float cx, float cy,
	float *keyGrayImgsDx, int keyGrayImgsDxStep,
	float *keyGrayImgsDy, int keyGrayImgsDyStep,
	float4 *updatedPosesInv,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	int *vertexToNodeIndices,
	float *vertexToNodeWeights,
	int *nodeVIndices,
	int nodeNum,
	float w_photo);

void ComputeJTJAndJTResidualRegTerm(CSRType &JTJ, float *JTResidual,
	float *residual,
	JTJBlockStatistics &blockStatistics,
	int nodeNum,
	int matchingPointsNum,
	float4 *originVertexPosesDevice,
	float4 *updatedVertexPosesDevice,
	int *nodeVIndicesDevice,
	int *nodeToNodeIndicesDevice,
	float *nodeToNodeWeightsDevice,
	float3 * Rts,
	float w_reg);
__global__ void ComputeJTJRegTermKernel(
	float *JTJ_a,
	int *JTJ_ia,
	int *equIndVec,
	int *equIndOffsetVec,
	int4 *nonZeroBlockInfoCoo,
	int *columnIndEachRow,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	int *nodeVIndices,
	int *nodeToNodeIndices,
	float *nodeToNodeWeights,
	int nodeNum,
	int matchingPointsNum,
	float w_reg);
__global__ void ComputeResidualRegTermKernel(
	float *residual,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	int *nodeVIndices,
	int *nodeToNodeIndices,
	float *nodeToNodeWeights,
	int nodeNum,
	float3 * Rts,
	float w_reg);
__global__ void ComputeJTResidualRegTermKernel(
	float *JTResidual,
	float *residual,
	int *equIndVec,
	int *equIndOffsetVec,
	int4 *nonZeroBlockInfoCoo,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	int *nodeVIndices,
	int *nodeToNodeIndices,
	float *nodeToNodeWeights,
	int nodeNum,
	int matchingPointsNum,
	float w_reg);

void ComputeJTJAndJTResidualRotTerm(CSRType &JTJ, float *JTResidual,
	float *residual,
	JTJBlockStatistics &blockStatistics,
	int nodeNum,
	float3 * Rts,
	float w_rot);
__global__ void ComputeResidualRotTermKernel(
	float *residual,
	int nodeNum,
	float3 * Rts,
	float w_rot);
__global__ void ComputeJTJAndJTResidualRotTermKernel(
	float *JTJ_a,
	float *JTResidual,
	int *JTJ_ia,
	float *residual,
	int nodeNum,
	float * Rts,
	int *columnIndEachRow,
	float w_rot);

#endif

