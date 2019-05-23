#ifndef __DEFORMATION_CUDA_FUNCS_H__
#define __DEFORMATION_CUDA_FUNCS_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <vector>

#include "Helpers/xUtils.h"

void VertexFilter(VBOType *vboDevice, int length, float4 *poseDevice, float fx, float fy, float cx, float cy);
__global__ void VertexFilterKernel(VBOType *vboDevice, int length, float4 *poseDevice, float fx, float fy, float cx, float cy);

void AddNodeIndexBase(int *nodeIndicesDevice, int nodeIndBase, int nodeNumFrag);
__global__ void AddNodeIndexBaseKernel(int *nodeIndices, int nodeIndBase, int nodeNumFrag);

void AddToMatchingPoints(int *matchingPointsFragDevice, int *vertexIndSrcFragDevice,
	int *vertexIndTargetFragDevice, int vertexIndTargetBase, int vertexNumFrag);
__global__ void AddToMatchingPointsKernel(int *matchingPointsFrag, int *vertexIndSrcFrag,
	int *vertexIndTargetFrag, int vertexIndTargetBase, int vertexNumFrag);

void CompressSampledVertex(float4 * sampledPointsDevice, float4 * pointsDevice,
	int * sampledVertexIndices, int sampledVertexNum);
__global__ void CompressSampledVertexKernel(float4 * sampledPointsDevice, float4 * pointsDevice,
	int * sampledVertexIndices, int sampledVertexNum);

void CompressMatchingVertex(float4* matchingVertexDevice, float4 *pointsDevice,
	int *matchingVertexIndices, int matchingVertexNum);
__global__ void CompressMatchingVertexKernel(float4* matchingVertexDevice, float4 *pointsDevice,
	int *matchingVertexIndices, int matchingVertexNum);

void KnnConsistantCheck(int *targetToSrcMatchingIndex, int *srcToTargetMatchingIndex, 
	int vertexIndTargetBase, bool *consistantCheck, int num);
__global__ void KnnConsistantCheckKernel(int *targetToSrcMatchingIndex, int *srcToTargetMatchingIndex,
	int vertexIndTargetBase, bool *consistantCheck, int num);

void CompressNodeIntoTargetPoints(float4 * targetPointsDevice,
	float4 *originVertexPosesDevice,
	int *nodeVIndicesFragDevice,
	int nodeNumFrag);
__global__ void CompressNodeIntoTargetPointsKernel(float4 *targetPoints,
	float4 *originVertexPoses,
	int *nodeVIndicesFrag,
	int nodeNumFrag);

void ComputeDist(float *weightDevice, int *indDevice, int K,
	float4 * srcPointsDevice,
	float4 * targetPointsDevice,
	int srcVertexNum);
__global__ void ComputeDistKernel(float4 *srcPoints, int srcVertexNum,
	float4 * targetPoints, int *indices, int K, float *weight);

void InitializeRts(float3 *Rts, float3 *Rs_transinv);
__global__ void InitializeRtsKernel(float3 *Rts, float3 *Rs_transinv, int nodeNum);

void CreateUpdatedVertexPosesAndNormals(float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	VBOType *vboDevice,
	std::vector<int> &vertexStrideVec,
	int fragInd);
__global__ void CreateUpdatedVertexPosesAndNormalsKernel(VBOType *vboFrag, int vertexNumFrag,
	float4 *updatedVertexPosesFrag, float4 *updatedVertexNormalsFrag);

void CreateVertexPosesAndNormals(float4 *originVertexPosesDevice,
	float4 *updatedVertexPosesDevice,
	float4 *originVertexNormalsDevice,
	float4 *updatedVertexNormalsDevice,
	VBOType *vboDevice,
	std::vector<int> &vertexStrideVec,
	int fragInd);
__global__ void CreateVertexPosesAndNormalsKernel(float4 *originVertexPosesFrag,
	float4 *updatedVertexPosesFrag,
	float4 *originVertexNormalsFrag,
	float4 *updatedVertexNormalsFrag,
	VBOType *vboFrag, int vertexNumFrag);

void UpdateIndMapsPerspective(int *indMapsDevice,
	float *zBufsDevice,
	int width, int height, int fragNum, int vertexNum,
	float fx, float fy, float cx, float cy,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	float4 *updatedKeyPosesInvDevice);
__global__ void UpdateIndMapPerspectiveKernel(int *indMap,
	float *zBufs,
	int width, int height, int vertexNum,
	float fx, float fy, float cx, float cy,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float4 *updatedKeyPoseInv);
__global__ void UpdateIndMapPerspectiveKernelWithLock(int *indMap,
	float *zBufs,
	int width, int height, int vertexNum,
	float fx, float fy, float cx, float cy,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float4 *updatedKeyPoseInv,
	int *lockVec);

void UpdateIndMapsPerspectiveFromVirtualCamera(int *indMapsDevice,
	float *zBufsDevice,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	float4 *virtualCameraPosesInvDevice,
	float4 *virtualCameraFxFyCxCyDevice,
	int vertexNum, int width, int height, 
	int fragNum, int virtualCameraCircle);

__global__ void UpdateIndMapsPerspectiveFromVirtualCameraKernel(
	int *indMap,
	float *zBufs,
	int width, int height, int vertexNum,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float4 *virtualCameraPosesInv,
	float4 *virtualCameraFxFyCxCy,
	int fragNum,
	int virtualCameraCircle);

void UpdateIndMapsPerspectiveFromVirtualCameraAndRepair(int *indMapsDevice,
	float *zBufsDevice,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	float4 *virtualCameraPosesInvDevice,
	float4 *virtualCameraFxFyCxCyDevice,
	int vertexNum, int width, int height,
	int fragNum, int virtualCameraCircle);

void UpdateUpdatedVertexPosesAndNormals(float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float4 *originVertexPoses,
	float4 *originVertexNormals,
	int vertexNum,
	float3 *Rts,
	float3 *Rs_transinv,
	int *nodeVIndicesDevice,
	int *vertexToNodeIndicesDevice,
	float *vertexToNodeWeightsDevice,
	float fx, float fy, float cx, float cy, int width, int height);
__global__ void UpdateUpdatedVertexPosesAndNormalsKernel(float4 *updatedVertexPose,
	float4 *updatedVertexNormal,
	float4 *originVertexPoses,
	float4 *originVertexNormals,
	float3 *Rts,
	float3 *Rs_transinv,
	int vertexNum,
	int *nodeVIndecesDevice,
	int *vertexToNodeIndicesDevice, float *vertexToNodeWeightsDevice,
	float fx, float fy, float cx, float cy, int width, int height);

void FetchColor(std::vector<std::pair<VBOType *, int> > &vboCudaSrcVec,
	std::pair<float4 *, int> &updatedVertexPoses,
	std::pair<float4 *, int> &keyPosesInvDevice,
	int fragNum,
	std::pair<uchar *, int> &imgsDevice,
	float fx, float fy, float cx, float cy, int width, int height);
__global__ void FetchColorKernal(VBOType *vboCudaSrc,
	float4 *updatedVertexPoses,
	float4 *keyPoseInv,
	int vertexNum,
	uchar *img,
	float fx, float fy, float cx, float cy, int width, int height);

void ApplyUpdatedVertexPosesAndNormals(VBOType *vboDevice,
	int vertexNum,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	uchar *keyColorImgsDevice, int keyColorImgsStep,
	float4 *keyPosesInvDevice,
	float fx, float fy, float cx, float cy, int width, int height);
__global__ void ApplyUpdatedVertexPosesAndNormalsKernel(
	VBOType *vbo, int vertexNum,
	float4 *updatedVertexPoses, float4 *updatedVertexNormals,
	uchar *keyColorImgs, int keyColorImgsStep,
	float4 *keyPosesInv,
	float fx, float fy, float cx, float cy, int width, int height);

void DistToWeight(float *vertexToNodeDistDevice, int vertexNum,
	float *nodeToNodeDistDevice, int nodeNum);
__global__ void DistToWeightVertexKernel(float *vertexToNodeDistDevice, float varianceInv, int vertexNum);
__global__ void DistToWeightNodeKernel(float *nodeToNodeDist, int nodeNum);

void UpdateRts(VBOType **vboCudaSrcPtrs, int *nodeVIndices,
	float3 *Rts, int fragNum, float4 *keyPose, float4 *updatedKeyPose);
__global__ void UpdateRtsKernal(VBOType **vboCudaSrcPtrs, int *nodeVIndices,
	float3 *Rts, int nodeNum, float4 *keyPose, float4 *updatedKeyPose);

void AddDeltaRtstoRts(float *x, float *delta_x, int length);
void CalcInvTransRot(float *Rs_transinv, float *Rts, int length);
__global__ void CalcInvTransRotKernal(float *Rts, float *Rs_transinv, int length);

void FindMatchingPointsPerspective(
	int *matchingPointsDevice,
	int *matchingFragsIndicesDevice,
	float4 *updatedVertexPosesDevice,
	std::pair<int *, int> &indMapsDevice,
	float4 *updatedKeyPosesInvDevice,
	int *sampledVertexIndicesDevice,
	int width, int height, float fx, float fy, float cx, float cy,
	int matchingPointsNum);
__global__ void FindMatchingPointsPerspectiveKernel(
	int *matchingPoints,
	int *matchingFragsIndices,
	float4 *updatedVertexPoses,
	int *indMaps, int indMapsStep,
	float4 *updatedKeyPosesInv,
	int *sampledVertexIndices,
	int width, int height, float fx, float fy, float cx, float cy,
	int matchingPointsNum);

void FindMatchingPointsPerspectiveFromVirtualCamera(int *matchingPointsDevice,
	int *matchingFragsIndicesDevice,
	float4 *updatedVertexPosesDevice,
	std::pair<int *, int> &indMapsDevice,
	int *sampledVertexIndicesDevice,
	float *minMatchingPointDist,
	float4 *virtualCameraPosesInvDevice,
	float4 *virtualCameraFxFyCxCyDevice,
	int width, int height, int matchingPointsNum,
	int fragNum, int virtualCameraCircle);
__global__ void FindMatchingPointsPerspectiveFromVirtualCameraKernel(
	int *matchingPoints,
	int *matchingFragsIndices,
	float4 *updatedVertexPoses,
	int *indMaps, int indMapsStep,
	int *sampledVertexIndices,
	float *minMatchingPointDist,
	float4 *virtualCameraPosesInv,
	float4 *virtualCameraFxFyCxCy,
	int width, int height, int matchingPointsNum,
	int fragNum, int virtualCameraCircle);
template <class Type>
__global__ void InitArray(Type *data, Type initdata, int _N)
{
	int vertexPairInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexPairInd >= _N) return;

	data[vertexPairInd] = initdata;
}

static __device__ __forceinline__ void SortNearInd(int *fourNearInd, float *fourNearWeight);

bool EstimateErrorsBeforeRegistration(std::vector<int> &isFragValid,
	std::vector<std::vector<int> > &poseGraph,
	float *residualSum, int *residualNnz, float *residual,
	VBOType *vboDevice, std::vector<int> &vertexStrideVec,
	int loopClosureNum);
__global__ void FragmentInvalid(VBOType *vboFrag, int vertexNumFrag);
__global__ void CalcResidualSumAndNnzKernel(float *residualSum, int *residualNnz, float *residual);

void ComputeJacobianAndResidual(CSRType jacobian, int &row_J, int &col_J, int &nnz, float *residual,
	int width, int height, float fx, float fy, float cx, float cy,
	int *matchingPointsDevice,
	int matchingPointsNumDescriptor,
	int matchingPointsNumNearest,
	int matchingPointsNumTotal,
	int loopClosureNum,
	int nodeNum,
	VBOType *vbo,
	float4 *originVertexPosesDevice,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	std::pair<float *, int> &keyGrayImgsDevice,
	std::pair<float *, int> &keyGrayImgsDxDevice,
	std::pair<float *, int> &keyGrayImgsDyDevice,
	float4 *updatedKeyPosesInvDevice,
	int *nodeVIndicesDevice,
	int *sampledVertexIndicesDevice,
	int *nodeToNodeIndicesDevice,
	float *nodeToNodeWeightsDevice,
	int *vertexToNodeIndicesDevice,
	float *vertexToNodeWeightsDevice,
	float3 * Rts,
	float w_geo, float w_photo, float w_reg, float w_rot, float w_trans,
	int iter);
__global__ void ComputeGeoTermPointToPointKernel(
	int *jacobian_ia, int iaOffset, int *jacobian_ja, float *jacobian_a, float *residual,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	int *vertexToNodeIndices,
	float *vertexToNodeWeights,
	int *nodeVIndices,
	float w_geo,
	int iter);
__global__ void ComputeGeoTermPointToPlainKernel(
	int *jacobian_ia, int iaOffset, int *jacobian_ja, float *jacobian_a, float *residual,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	int *vertexToNodeIndices,
	float *vertexToNodeWeights,
	int *nodeVIndices,
	float w_geo,
	int iter);
__global__ void ComputePhotoTermKernel(
	int *jacobian_ia, int iaOffset, int *jacobian_ja, float *jacobian_a, float *residual,
	int width, int height, float fx, float fy, float cx, float cy,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float *keyGrayImgs, int keykeyGrayImgsStep,
	float *keyGrayImgsDx, int keyGrayImgsDxStep,
	float *keyGrayImgsDy, int keyGrayImgsDyStep,
	float4 *updatedPosesInv,
	int *vertexToNodeIndices,
	float *vertexToNodeWeights,
	int *nodeVIndices,
	float w_photo,
	int iter);
__global__ void ComputeRegTermsKernel(int *jacobian_ja, float *jacobian_a,
	float *residual,
	int nodeNum,
	float3 *Rts,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	int *nodeVIndices,
	int *nodeToNodeIndices,
	float *nodeToNodeWeights,
	float w_reg, float w_rot, float w_trans);

void UpdateCameraPoses(
	float4 *updatedKeyPosesDevice,
	float4 *updatedKeyPosesInvDevice,
	float3 *Rts,
	float *cameraToNodeWeightsDevice,
	float4 *originVertexPosesDevice,
	int *nodeVIndicesDevice,
	float4 *keyPosesDevice,
	int nodeNum);
__global__ void UpdateCameraNodeWeightKernel(float *cameraToNodeWeights,
	float4 *originVertexPoses,
	int *nodeVIndices,
	float4 *keyPoses,
	int nodeNum);
__global__ void UpdateCameraPosesKernel(float4 *updatedKeyPoses,
	float4 *updatedKeyPosesInv,
	float3 *Rts,
	float *cameraToNodeWeight,
	float4 *originVertexPoses,
	int *nodeVIndices,
	float4 *keyPoses,
	int nodeNum);
__device__ __forceinline__ void MatToQuaternion(float3 *matFloat3, float4 *quaternionFloat4);
__device__ __forceinline__ void QuaternionToMat(float3 *mat, float4 *quaternion);

__global__ void BlurKernel(cv::cuda::PtrStepSz<float> bHor,
	cv::cuda::PtrStepSz<float> bVer,
	cv::cuda::PtrStepSz<uchar> grayImg);
__global__ void CalculateBlurScoreKernel(float4 *sumVal,
	cv::cuda::PtrStepSz<float> bHor,
	cv::cuda::PtrStepSz<float> bVer,
	cv::cuda::PtrStepSz<uchar> grayImg);
float CalculateBlurScoreGPU(const cv::cuda::GpuMat &grayImgDevice,
	cv::cuda::GpuMat &bHorDevice,
	cv::cuda::GpuMat &bVerDevice);

#endif