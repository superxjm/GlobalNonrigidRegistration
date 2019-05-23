//#include "stdafx.h"

#include "thrust/device_vector.h"
#include <thrust/transform.h>
#include <thrust/device_ptr.h>

#include "Cuda/xDeformationCudaFuncs.cuh"
#include "Helpers/UtilsMath.h"
#include "Helpers/xUtils.h"
#include "Helpers/InnorealTimer.hpp"
#include "Helpers/xGlobalStats.h"

#define USE_BILINEAR_TO_CALC_GRAD 0

void VertexFilter(VBOType *vboDevice, int length, float4 *poseDevice, float fx, float fy, float cx, float cy)
{
	int block = 256;
	int grid = DivUp(length, block);
	VertexFilterKernel << <grid, block >> > (vboDevice, length, poseDevice, fx, fy, cx, cy);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void VertexFilterKernel(VBOType *vboDevice, int length, float4 *poseDevice, float fx, float fy, float cx, float cy)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= length)
		return;

	float4 vertexPos = vboDevice[idx].posConf;
	float4 camPos = poseDevice[3];
	float3 vertexNormal = make_float3(vboDevice[idx].normalRad);
	float3 vertexCamVec = make_float3(vertexPos - camPos);
	vertexNormal = normalize(vertexNormal);
	vertexCamVec = normalize(vertexCamVec);

	if (fabs(vertexCamVec.x * vertexNormal.x + vertexCamVec.y * vertexNormal.y + vertexCamVec.z * vertexNormal.z) < 0.3f)
	{
		//printf("%f\n", vertexCamVec.x * vertexNormal.x + vertexCamVec.y * vertexNormal.y + vertexCamVec.z * vertexNormal.z);
		vboDevice[idx].colorTime.y = -1;
	}
}

void AddNodeIndexBase(int *nodeIndicesDevice, int nodeIndBase, int nodeNumFrag)
{
	int block = 256;
	int grid = DivUp(nodeNumFrag, block);
	std::cout << "nodeIndBase: " << nodeIndBase << std::endl;

	AddNodeIndexBaseKernel << <grid, block >> > (nodeIndicesDevice, nodeIndBase, nodeNumFrag);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void AddNodeIndexBaseKernel(int *nodeIndices, int nodeIndBase, int nodeNumFrag)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= nodeNumFrag)
		return;

	nodeIndices[idx] = nodeIndices[idx] + nodeIndBase;
}

void AddToMatchingPoints(int *matchingPointsFragDevice, int *vertexIndSrcFragDevice,
	int *vertexIndTargetFragDevice, int vertexIndTargetBase, int vertexNumFrag)
{
	int block = 256;
	int grid = DivUp(vertexNumFrag, block);

	AddToMatchingPointsKernel << <grid, block >> > (matchingPointsFragDevice, vertexIndSrcFragDevice,
		vertexIndTargetFragDevice, vertexIndTargetBase, vertexNumFrag);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void AddToMatchingPointsKernel(int *matchingPointsFrag, int *vertexIndSrcFrag,
	int *vertexIndTargetFrag, int vertexIndTargetBase, int vertexNumFrag)
{
	int vertexIndFrag = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexIndFrag >= vertexNumFrag)
		return;

	matchingPointsFrag[2 * vertexIndFrag] = vertexIndSrcFrag[vertexIndFrag];
	matchingPointsFrag[2 * vertexIndFrag + 1] = vertexIndTargetFrag[vertexIndFrag] + vertexIndTargetBase;
}

void CompressSampledVertex(float4* sampledPointsDevice, float4 * pointsDevice,
	int *sampledVertexIndices, int sampledVertexNum)
{
	//std::cout << sampledVertexNum << std::endl;
	int block = 256;
	int grid = DivUp(sampledVertexNum, block);
	CompressSampledVertexKernel << <grid, block >> > (sampledPointsDevice, pointsDevice,
		sampledVertexIndices, sampledVertexNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CompressSampledVertexKernel(float4 * sampledPointsDevice, float4 * pointsDevice,
	int * sampledVertexIndices, int sampledVertexNum)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= sampledVertexNum)
		return;

	sampledPointsDevice[idx] = pointsDevice[sampledVertexIndices[idx]];
}

void CompressMatchingVertex(float4* matchingVertexDevice, float4 *pointsDevice,
	int *matchingVertexIndices, int matchingVertexNum)
{
	int block = 256;
	int grid = DivUp(matchingVertexNum, block);
	CompressMatchingVertexKernel << <grid, block >> > (matchingVertexDevice, pointsDevice,
		matchingVertexIndices, matchingVertexNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CompressMatchingVertexKernel(float4* matchingVertexDevice, float4 *pointsDevice,
	int *matchingVertexIndices, int matchingVertexNum)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= matchingVertexNum)
		return;

	matchingVertexDevice[idx] = pointsDevice[matchingVertexIndices[idx * 2 + 1]];
}

void KnnConsistantCheck(int *targetToSrcMatchingIndex, int *srcToTargetMatchingIndex, 
	int vertexIndTargetBase, bool *consistantCheck, int num)
{
	int block = 256;
	int grid = DivUp(num, block);
	KnnConsistantCheckKernel << <grid, block >> > (targetToSrcMatchingIndex, srcToTargetMatchingIndex,
		vertexIndTargetBase, consistantCheck, num);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void KnnConsistantCheckKernel(int *targetToSrcMatchingIndex, int *srcToTargetMatchingIndex, 
	int vertexIndTargetBase, bool *consistantCheck, int num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= num) return;

	int srcIndex = targetToSrcMatchingIndex[idx];
	if (srcIndex != idx)
	{
		consistantCheck[idx] = false;
		srcToTargetMatchingIndex[2 * idx] = -1;
		srcToTargetMatchingIndex[2 * idx + 1] = -1;
	}
	else
	{
		consistantCheck[idx] = true;
	}
}

void CompressNodeIntoTargetPoints(float4 * targetPointsDevice,
	float4 *originVertexPosesDevice,
	int *nodeVIndicesFragDevice,
	int nodeNumFrag)
{
	int block = 256;
	int grid = DivUp(nodeNumFrag, block);
	CompressNodeIntoTargetPointsKernel << <grid, block >> > (targetPointsDevice, originVertexPosesDevice, nodeVIndicesFragDevice, nodeNumFrag);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CompressNodeIntoTargetPointsKernel(float4 *targetPoints, 
	float4 *originVertexPoses, 
	int *nodeVIndicesFrag, 
	int nodeNumFrag)
{
	int nodeIndFrag = threadIdx.x + blockIdx.x * blockDim.x;

	if (nodeIndFrag >= nodeNumFrag)
		return;

	targetPoints[nodeIndFrag] = originVertexPoses[nodeVIndicesFrag[nodeIndFrag]];
}

void ComputeDist(float *weightDevice, int *indDevice, int K,
	float4 * srcPointsDevice,
	float4 * targetPointsDevice,
	int srcVertexNum)
{
	int block = 256;
	int grid = DivUp(srcVertexNum, block);
	ComputeDistKernel << <grid, block >> > (srcPointsDevice, srcVertexNum, targetPointsDevice, indDevice, K, weightDevice);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void ComputeDistKernel(float4 *srcPoints, int srcVertexNum,
	float4 * targetPoints, int *indices, int K, float *weight)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= srcVertexNum)
		return;

	float4 *srcPoint, *targetPoint;
	float x, y, z;
	srcPoint = srcPoints + idx;
	for (int i = 0; i < K; ++i)
	{
		targetPoint = targetPoints + indices[idx * K + i];
		x = (srcPoint->x - targetPoint->x);
		y = (srcPoint->y - targetPoint->y);
		z = (srcPoint->z - targetPoint->z);
		*(weight + idx * K + i) = sqrt(x * x + y * y + z * z);
	}
}

void InitializeRts(float3 *Rts, float3 *Rs_transinv)
{
	int nodeNum = NODE_NUM_EACH_FRAG * MAX_FRAG_NUM;

	int block = 256;
	int grid = DivUp(nodeNum, block);

	InitializeRtsKernel << <grid, block >> > (Rts, Rs_transinv, nodeNum);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void InitializeRtsKernel(float3 *Rts, float3 *Rs_transinv, int nodeNum)
{
	int nodeInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (nodeInd >= nodeNum)
		return;

#if 1
	float3 *RtsPtr = (Rts + nodeInd * 4);
	RtsPtr[0].x = 1.0; RtsPtr[0].y = 0.0; RtsPtr[0].z = 0.0;
	RtsPtr[1].x = 0.0; RtsPtr[1].y = 1.0; RtsPtr[1].z = 0.0;
	RtsPtr[2].x = 0.0; RtsPtr[2].y = 0.0; RtsPtr[2].z = 1.0;
	RtsPtr[3].x = 0.0; RtsPtr[3].y = 0.0; RtsPtr[3].z = 0.0;

	float3 *Rs_transinvPtr = (Rs_transinv + nodeInd * 3);
	Rs_transinvPtr[0].x = 1.0; Rs_transinvPtr[0].y = 0.0; Rs_transinvPtr[0].z = 0.0;
	Rs_transinvPtr[1].x = 0.0; Rs_transinvPtr[1].y = 1.0; Rs_transinvPtr[1].z = 0.0;
	Rs_transinvPtr[2].x = 0.0; Rs_transinvPtr[2].y = 0.0; Rs_transinvPtr[2].z = 1.0;
#endif

#if 0
	// For test Jacobian
	float3 *RtsPtr = (Rts + nodeInd * 4);
	int ii = nodeInd;
	RtsPtr[0].x = 1 + ii * 0.002; RtsPtr[0].y = ii * 0.0002; RtsPtr[0].z = ii * 0.0001;
	RtsPtr[1].x = ii * 0.001; RtsPtr[1].y = 1 + ii * 0.003; RtsPtr[1].z = ii * 0.001;
	RtsPtr[2].x = ii * 0.001; RtsPtr[2].y = ii * 0.001; RtsPtr[2].z = 1 + ii * 0.004;
	RtsPtr[3].x = ii * 0.0005; RtsPtr[3].y = ii * 0.0017; RtsPtr[3].z = ii * 0.001;
#endif
}

void CreateUpdatedVertexPosesAndNormals(float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	VBOType *vboDevice,
	std::vector<int> &vertexStrideVec,
	int fragInd)
{
	int vertexNumFrag = vertexStrideVec[fragInd + 1] - vertexStrideVec[fragInd];

	int block = 256, grid = DivUp(vertexNumFrag, block);
	CreateUpdatedVertexPosesAndNormalsKernel << <grid, block >> > (
		vboDevice + vertexStrideVec[fragInd], vertexNumFrag,
		updatedVertexPosesDevice + vertexStrideVec[fragInd],
		updatedVertexNormalsDevice + vertexStrideVec[fragInd]);

	/*
	std::cout << "vertexStrideVec[fragInd]: " << vertexStrideVec[fragInd] << std::endl;
	std::vector<float4> matchingPointsFragVec(100);
	checkCudaErrors(cudaMemcpy(
	matchingPointsFragVec.data(),
	updatedVertexPosesDevice, matchingPointsFragVec.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	for (int n = 0; n < matchingPointsFragVec.size(); ++n)
	{
	std::cout << matchingPointsFragVec[n].x << ", ";
	std::cout << matchingPointsFragVec[n].y << ", ";
	std::cout << matchingPointsFragVec[n].z << ", ";
	std::cout << matchingPointsFragVec[n].w << ", ";
	}
	std::cout << std::endl;
	exit(0);
	*/

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CreateUpdatedVertexPosesAndNormalsKernel(VBOType *vboFrag, int vertexNumFrag,
	float4 *updatedVertexPosesFrag, float4 *updatedVertexNormalsFrag)
{
	int vertexIndFrag = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexIndFrag >= vertexNumFrag)
		return;

	float4 *posConf = &(vboFrag + vertexIndFrag)->posConf;
	float4 *normalRad = &(vboFrag + vertexIndFrag)->normalRad;
	//printf("%f %f %f %f %f %f\n", posConf->x, posConf->y, posConf->z, normalRad->x, normalRad->y, normalRad->z);	
	//printf("%f, ", (vboFrag + vertexIndFrag)->colorTime.y);
	*(updatedVertexPosesFrag + vertexIndFrag) = make_float4(posConf->x, posConf->y, posConf->z, 1.0f);
	*(updatedVertexNormalsFrag + vertexIndFrag) = make_float4(normalRad->x, normalRad->y, normalRad->z, 0.0f);
}

struct float4plus_functor : public thrust::binary_function<float, float, float>
{
	float4plus_functor() {}

	__host__ __device__
		float4 operator()(const float4& x, const float4& y) const
	{
		return x + y;
	}
};

void UpdateIndMapsPerspective(int *indMapsDevice,
	float *zBufsDevice,
	int width, int height, int fragNum, int vertexNum,
	float fx, float fy, float cx, float cy,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	float4 *updatedKeyPosesInvDevice)
{
#if 0
	std::vector<float4> m_updatedVertexPosesVec(100);
	checkCudaErrors(cudaMemcpy(m_updatedVertexPosesVec.data(), updatedVertexPosesDevice,
		m_updatedVertexPosesVec.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	for (int i = 0; i < m_updatedVertexPosesVec.size(); ++i)
	{
		std::cout << m_updatedVertexPosesVec[i].x << ", " <<
			m_updatedVertexPosesVec[i].y << ", " <<
			m_updatedVertexPosesVec[i].z << ", " <<
			m_updatedVertexPosesVec[i].w << std::endl;
	}
	std::cout << std::endl;
	std::exit(0);
	//std::cout << "vertexNum: " << vertexNum << std::endl;
#endif

#if 1
	checkCudaErrors(cudaMemset(indMapsDevice, 0xFF, sizeof(int) * width * height * fragNum));
	checkCudaErrors(cudaMemset(zBufsDevice, 0x00, sizeof(float) * width * height * fragNum));
	int block = 256, grid;
	//std::cout << "vertexNum: " << vertexNum << std::endl;
	grid = DivUp(vertexNum, block);
	UpdateIndMapPerspectiveKernel << <grid, block >> > (
		indMapsDevice,
		zBufsDevice,
		width, height, vertexNum,
		fx, fy, cx, cy,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		updatedKeyPosesInvDevice);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
#endif
#if 0
	checkCudaErrors(cudaMemset(indMapsDevice, 0xFF, sizeof(int) * width * height * fragNum));
	checkCudaErrors(cudaMemset(zBufsDevice, 0x00, sizeof(float) * width * height * fragNum));
	thrust::device_vector<int> lockVec(width * height * fragNum, 1);
	int block = 256, grid;
	//std::cout << "vertexNum: " << vertexNum << std::endl;
	grid = DivUp(vertexNum * 32, block);
	UpdateIndMapPerspectiveKernelWithLock << <grid, block >> > (
		indMapsDevice,
		zBufsDevice,
		width, height, vertexNum,
		fx, fy, cx, cy,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		updatedKeyPosesInvDevice,
		RAW_PTR(lockVec));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
#endif
}

__global__ void UpdateIndMapPerspectiveKernel(
	int *indMap,
	float *zBufs,
	int width, int height, int vertexNum,
	float fx, float fy, float cx, float cy,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float4 *updatedKeyPoseInv)
{
	int vertexInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexInd >= vertexNum)
		return;

	float4 posFragInd = updatedVertexPoses[vertexInd];
	float4 normal = updatedVertexNormals[vertexInd];
	int fragInd = posFragInd.w;
	if (fragInd < 0)
		return;

	float4 *updatedKeyPoseInvFrag = updatedKeyPoseInv + 4 * fragInd;

	///printf("mat: %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n",
	//updatedKeyPoseInvFrag[0].x, updatedKeyPoseInvFrag[1].x, updatedKeyPoseInvFrag[2].x, updatedKeyPoseInvFrag[3].x,
	//updatedKeyPoseInvFrag[0].y, updatedKeyPoseInvFrag[1].y, updatedKeyPoseInvFrag[2].y, updatedKeyPoseInvFrag[3].y,
	//updatedKeyPoseInvFrag[0].z, updatedKeyPoseInvFrag[1].z, updatedKeyPoseInvFrag[2].z, updatedKeyPoseInvFrag[3].z,
	//updatedKeyPoseInvFrag[0].w, updatedKeyPoseInvFrag[1].w, updatedKeyPoseInvFrag[2].w, updatedKeyPoseInvFrag[3].w);

	float4 posLocal = posFragInd.x * updatedKeyPoseInvFrag[0] + posFragInd.y * updatedKeyPoseInvFrag[1] +
		posFragInd.z * updatedKeyPoseInvFrag[2] + updatedKeyPoseInvFrag[3];
	float4 normalLocal = normal.x * updatedKeyPoseInvFrag[0] + normal.y * updatedKeyPoseInvFrag[1] +
		normal.z * updatedKeyPoseInvFrag[2];

	int u = __float2int_rn((posLocal.x * fx) / posLocal.z + cx);
	int v = __float2int_rn((posLocal.y * fy) / posLocal.z + cy);

	if (u < 0 || v < 0 || u >= width || v >= height)
	{
		return;
	}

	float &depth = *(zBufs + width * height * fragInd + v * width + u);

	if ((depth == 0 || posLocal.z < depth) && normalLocal.z > 0)  
	{
		*(indMap + width * height * fragInd + v * width + u) = vertexInd;
		depth = posLocal.z;
	}
}

__global__ void UpdateIndMapPerspectiveKernelWithLock(
	int *indMap,
	float *zBufs,
	int width, int height, int vertexNum,
	float fx, float fy, float cx, float cy,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float4 *updatedKeyPoseInv,
	int *lockVec)
{
	int vertexInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexInd % 32 != 0)
		return;
	vertexInd = vertexInd / 32;
	if (vertexInd >= vertexNum)
		return;

	float4 posFragInd = updatedVertexPoses[vertexInd];
	float4 normal = updatedVertexNormals[vertexInd];
	int fragInd = posFragInd.w;
	if (fragInd < 0)
		return;

	float4 *updatedKeyPoseInvFrag = updatedKeyPoseInv + 4 * fragInd;

	///printf("mat: %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n",
	//updatedKeyPoseInvFrag[0].x, updatedKeyPoseInvFrag[1].x, updatedKeyPoseInvFrag[2].x, updatedKeyPoseInvFrag[3].x,
	//updatedKeyPoseInvFrag[0].y, updatedKeyPoseInvFrag[1].y, updatedKeyPoseInvFrag[2].y, updatedKeyPoseInvFrag[3].y,
	//updatedKeyPoseInvFrag[0].z, updatedKeyPoseInvFrag[1].z, updatedKeyPoseInvFrag[2].z, updatedKeyPoseInvFrag[3].z,
	//updatedKeyPoseInvFrag[0].w, updatedKeyPoseInvFrag[1].w, updatedKeyPoseInvFrag[2].w, updatedKeyPoseInvFrag[3].w);

	float4 posLocal = posFragInd.x * updatedKeyPoseInvFrag[0] + posFragInd.y * updatedKeyPoseInvFrag[1] +
		posFragInd.z * updatedKeyPoseInvFrag[2] + updatedKeyPoseInvFrag[3];
	float4 normalLocal = normal.x * updatedKeyPoseInvFrag[0] + normal.y * updatedKeyPoseInvFrag[1] +
		normal.z * updatedKeyPoseInvFrag[2];

	int u = __float2int_rn((posLocal.x * fx) / posLocal.z + cx);
	int v = __float2int_rn((posLocal.y * fy) / posLocal.z + cy);

	if (u < 0 || v < 0 || u >= width || v >= height)
	{
		return;
	}

#if 1
	int ret;
	do {
		ret = atomicExch(lockVec + width * height * fragInd + v * width + u, 0);
	} while (ret == 0);
#endif

	float &depth = *(zBufs + width * height * fragInd + v * width + u);

	if ((depth == 0 || posLocal.z < depth) && normalLocal.z > 0)
	{
		*(indMap + width * height * fragInd + v * width + u) = vertexInd;
		depth = posLocal.z;
	}

	ret = atomicExch(lockVec + width * height * fragInd + v * width + u, 1);
}

void UpdateIndMapsPerspectiveFromVirtualCamera(int *indMapsDevice,
	float *zBufsDevice,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	float4 *virtualCameraPosesInvDevice,
	float4 *virtualCameraFxFyCxCyDevice,
	int vertexNum, int width, int height,
	int fragNum, int virtualCameraCircle)
{
	checkCudaErrors(cudaMemset(indMapsDevice, 0xFF, sizeof(int) * width * height * fragNum));
	checkCudaErrors(cudaMemset(zBufsDevice, 0x00, sizeof(float) * width * height * fragNum));

	int block = 256, grid;
	//std::cout << "vertexNum: " << vertexNum << std::endl;
	grid = DivUp(vertexNum, block);

	UpdateIndMapsPerspectiveFromVirtualCameraKernel << <grid, block >> > (indMapsDevice, zBufsDevice, 
		width, height, vertexNum, 
		updatedVertexPosesDevice, 
		updatedVertexNormalsDevice, 
		virtualCameraPosesInvDevice,
		virtualCameraFxFyCxCyDevice, 
		fragNum, virtualCameraCircle);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	/*cv::Mat_<float> tmpMat(height, width);
	cv::Mat_<int> tmpMatIdx(height, width);
	for (int i = 0; i < fragNum; i++)
	{
		checkCudaErrors(cudaMemcpy(tmpMat.data, zBufsDevice + i*width*height, width*height * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(tmpMatIdx.data, indMapsDevice + i*width*height, width*height * sizeof(int), cudaMemcpyDeviceToHost));
		int a = 0;
	}*/
}

__global__ void UpdateIndMapsPerspectiveFromVirtualCameraKernel(
	int *indMap,
	float *zBufs,
	int width, int height, int vertexNum, 
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float4 *virtualCameraPosesInv,
	float4 *virtualCameraFxFyCxCy,
	int fragNum,
	int virtualCameraCircle)
{
	int vertexInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexInd >= vertexNum)
		return;

	float4 posFragInd = updatedVertexPoses[vertexInd];
	float4 normal = updatedVertexNormals[vertexInd];
	int fragInd = posFragInd.w;
	if (fragInd < 0)
		return;

	float4 *virtualCameraPosesInvFrag = virtualCameraPosesInv + 4 * fragNum * virtualCameraCircle + 4 * fragInd;
	float4 *virtualCameraFxFyCxCyFrag = virtualCameraFxFyCxCy + fragNum * virtualCameraCircle + fragInd;

	float4 posLocal = posFragInd.x * virtualCameraPosesInvFrag[0] + posFragInd.y * virtualCameraPosesInvFrag[1] +
		posFragInd.z * virtualCameraPosesInvFrag[2] + virtualCameraPosesInvFrag[3];
	float4 normalLocal = normal.x * virtualCameraPosesInvFrag[0] + normal.y * virtualCameraPosesInvFrag[1] +
		normal.z * virtualCameraPosesInvFrag[2];
	
	int u = __float2int_rn((posLocal.x * virtualCameraFxFyCxCyFrag->x) / posLocal.z + virtualCameraFxFyCxCyFrag->z);
	int v = __float2int_rn((posLocal.y * virtualCameraFxFyCxCyFrag->y) / posLocal.z + virtualCameraFxFyCxCyFrag->w);

	if (u < 0 || v < 0 || u >= width || v >= height)
	{
		return;
	}

	float &depth = *(zBufs + width * height * fragInd + v * width + u);

	if ((depth == 0 || posLocal.z < depth) && normalLocal.z < 0) 
	{
		*(indMap + width * height * fragInd + v * width + u) = vertexInd;
		depth = posLocal.z;
	}

}

__global__ void RepairDepthMap(
	float *zBufs,
	int width, int height, int maxNum,
	int fragNum)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;

	if (u >= maxNum) return;

	int fragInd = u / (width*height);
	int x = u % (width*height) % width;
	int y = u % (width*height) / width;

	float *depth = zBufs + width * height * fragInd;

	if (depth[y*width + x] != 0) return;

	int tx[4] = { 0,1,0,-1 };
	int ty[4] = { 1,0,-1,0 };
	int nx, ny;

	for (int i = 0; i < 4; i++)
	{
		nx = x + tx[i];
		ny = y + ty[i];
		if (nx >= 0 && nx < width && ny >= 0 && ny < height)
		{
			if (depth[ny*width + nx] != 0)
			{
				depth[y*width + x] = depth[ny*width + nx];
				//atomicExch(&depth[y*width + x], depth[ny*width + nx]);
				return;
			}
		}
	}
}

__global__ void DeletaDepthMap(float *zBufs,
	int *indMap,
	int width, int height, int maxNum,
	int fragNum)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;

	if (u >= maxNum) return;

	int fragInd = u / (width*height);
	int x = u % (width*height) % width;
	int y = u % (width*height) / width;

	float *depth = zBufs + width * height * fragInd;
	int   *index = indMap + width * height * fragInd;

	if (depth[y*width + x] == 0) return;

	int tx[4] = { 0,1,0,-1 };
	int ty[4] = { 1,0,-1,0 };
	int nx, ny;

	for (int i = 0; i < 4; i++)
	{
		nx = x + tx[i];
		ny = y + ty[i];
		if (nx >= 0 && nx < width && ny >= 0 && ny < height)
		{
			if (depth[ny*width + nx] == 0)
			{
				depth[y*width + x] = 0;
				index[y*width + x] = -1;
				return;
			}
		}
	}
}

void UpdateIndMapsPerspectiveFromVirtualCameraAndRepair(int *indMapsDevice,
	float *zBufsDevice,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	float4 *virtualCameraPosesInvDevice,
	float4 *virtualCameraFxFyCxCyDevice,
	int vertexNum, int width, int height,
	int fragNum, int virtualCameraCircle)
{
	UpdateIndMapsPerspectiveFromVirtualCamera(indMapsDevice,
		zBufsDevice,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		virtualCameraPosesInvDevice,
		virtualCameraFxFyCxCyDevice,
		vertexNum, width, height,
		fragNum, virtualCameraCircle);

	int block = 256;
	int grid = DivUp(fragNum*width*height, block);

	for (int i=0; i<3; i++)
	{
		RepairDepthMap << <grid, block >> >(zBufsDevice, width, height, fragNum*width*height, fragNum);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
	}

	for (int i = 0; i < 7; i++)
	{
		DeletaDepthMap << <grid, block >> >(zBufsDevice, indMapsDevice, width, height, fragNum*width*height, fragNum);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
	}

	/*cv::Mat_<float> tmpMat(height, width);
	cv::Mat_<int> tmpMatIdx(height, width);
	for (int i = 0; i < fragNum; i++)
	{
		checkCudaErrors(cudaMemcpy(tmpMat.data, zBufsDevice + i*width*height, width*height * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(tmpMatIdx.data, indMapsDevice + i*width*height, width*height * sizeof(int), cudaMemcpyDeviceToHost));
		int a = 0;
	}*/
}

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
		float fx, float fy, float cx, float cy, int width, int height)
{
	int block = 256, grid;
	grid = DivUp(vertexNum, block);
	UpdateUpdatedVertexPosesAndNormalsKernel << <grid, block >> > (
		updatedVertexPoses, updatedVertexNormals,
		originVertexPoses, originVertexNormals,
		Rts,
		Rs_transinv,
		vertexNum,
		nodeVIndicesDevice,
		vertexToNodeIndicesDevice, vertexToNodeWeightsDevice,
		fx, fy, cx, cy, width, height);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void UpdateUpdatedVertexPosesAndNormalsKernel(float4 *updatedVertexPose,
	float4 *updatedVertexNormal,
	float4 *originVertexPoses,
	float4 *originVertexNormals,
	float3 *Rts,
	float3 *Rs_transinv,
	int vertexNum,
	int *nodeVIndecesDevice,
	int *vertexToNodeIndicesDevice, float *vertexToNodeWeightsDevice,
	float fx, float fy, float cx, float cy, int width, int height)
{
	int vertexInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexInd >= vertexNum)
		return;

	int *fourNearInd = vertexToNodeIndicesDevice + vertexInd * 4;
	float *fourNearWeight = vertexToNodeWeightsDevice + vertexInd * 4;
	int nearInd;

	float4 vertexPosFragInd = originVertexPoses[vertexInd];
	float4 vertexNormal = originVertexNormals[vertexInd];
	float3 *Rt;
	float3 *R_transinv;
	float4 newVertexPosFragInd = make_float4(0.0, 0.0, 0.0, 0.0);
	float4 newVertexNormal = make_float4(0.0, 0.0, 0.0, 0.0);

	float3 nearNodePose, nearNodeNormal;
	float weight;

	for (int n = 0; n < 4; ++n)
	{
		nearInd = *(fourNearInd + n);
		weight = *(fourNearWeight + n);
		nearNodePose = make_float3(originVertexPoses[nodeVIndecesDevice[nearInd]]);
		nearNodeNormal = normalize(make_float3(originVertexNormals[nodeVIndecesDevice[nearInd]]));
		Rt = Rts + nearInd * 4;
		R_transinv = Rs_transinv + nearInd * 3;

#if 0
		printf("vertexid: %d:\nweight: %f\n%f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n",
			vertexInd,
			weight,
			Rt[0].x, Rt[0].y, Rt[0].z,
			Rt[1].x, Rt[1].y, Rt[1].z,
			Rt[2].x, Rt[2].y, Rt[2].z,
			Rt[3].x, Rt[3].y, Rt[3].z
		);
#endif
		newVertexPosFragInd += make_float4(weight * (Rt[0] * (vertexPosFragInd.x - nearNodePose.x) +
			Rt[1] * (vertexPosFragInd.y - nearNodePose.y) +
			Rt[2] * (vertexPosFragInd.z - nearNodePose.z) +
			nearNodePose +
			Rt[3]));

#if 0	
		printf("weight: %d %d %d %d %d %f %f %f %f %f | %f %f %f %f %f %f %f %f %f\n",
			nearInd,
			*(fourNearInd + 0),
			*(fourNearInd + 1),
			*(fourNearInd + 2),
			*(fourNearInd + 3),
			weight,
			*(fourNearWeight + 0),
			*(fourNearWeight + 1),
			*(fourNearWeight + 2),
			*(fourNearWeight + 3),
			R_transinv[0].x, R_transinv[0].y, R_transinv[0].z,
			R_transinv[1].x, R_transinv[1].y, R_transinv[1].z,
			R_transinv[2].x, R_transinv[2].y, R_transinv[2].z);
#endif
		newVertexNormal += make_float4(weight * (R_transinv[0] * vertexNormal.x +
			R_transinv[1] * vertexNormal.y +
			R_transinv[2] * vertexNormal.z));
	}

	newVertexPosFragInd.w = (updatedVertexPose + vertexInd)->w;
	newVertexNormal = normalize(newVertexNormal);
	*(updatedVertexPose + vertexInd) = newVertexPosFragInd;
	*(updatedVertexNormal + vertexInd) = newVertexNormal;
	/*
	assert(!isnan(vertexPos.x));
	assert(!isnan(vertexPos.y));
	assert(!isnan(vertexPos.z));
	assert(!isnan(vertexNormal.x));
	assert(!isnan(vertexNormal.y));
	assert(!isnan(vertexNormal.z));
	*/	
}

void FetchColor(std::vector<std::pair<VBOType *, int> > &vboCudaSrcVec,
	std::pair<float4 *, int> &updatedVertexPoses,
	std::pair<float4 *, int> &keyPosesInvDevice,
	int fragNum,
	std::pair<uchar *, int> &imgsDevice,
	float fx, float fy, float cx, float cy, int width, int height)
{
	int block = 256, grid;
	uchar *imgsDeviceFrag;
	float4 *updatedVertexPoseFrag;
	float4 *keyPosesInvDeviceFrag;
	VBOType *vboCudaSrcFrag;
	int vertexNumFrag;
	for (int fragInd = 0; fragInd < fragNum; ++fragInd)
	{
		imgsDeviceFrag = imgsDevice.first + fragInd * imgsDevice.second;
		updatedVertexPoseFrag = updatedVertexPoses.first + fragInd * updatedVertexPoses.second;
		keyPosesInvDeviceFrag = keyPosesInvDevice.first + fragInd * keyPosesInvDevice.second;
		vboCudaSrcFrag = vboCudaSrcVec[fragInd].first;
		vertexNumFrag = vboCudaSrcVec[fragInd].second;

		grid = DivUp(vertexNumFrag, block);
		FetchColorKernal << <grid, block >> > (
			vboCudaSrcFrag,
			updatedVertexPoseFrag,
			keyPosesInvDeviceFrag,
			vertexNumFrag,
			imgsDeviceFrag,
			fx, fy, cx, cy, width, height);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void FetchColorKernal(VBOType *vboCudaSrc,
	float4 *updatedVertexPoses,
	float4 *keyPoseInv,
	int vertexNum,
	uchar *img,
	float fx, float fy, float cx, float cy, int width, int height)
{
	int vertexInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexInd >= vertexNum)
		return;

	float4 *posConf = updatedVertexPoses + vertexInd;
	float4 posConfLocal = posConf->x * keyPoseInv[0] + posConf->y * keyPoseInv[1] +
		keyPoseInv->z * keyPoseInv[2] + keyPoseInv[3];

	int u = __float2int_rn((posConfLocal.x * fx) / posConfLocal.z + cx);
	int v = __float2int_rn((posConfLocal.y * fy) / posConfLocal.z + cy);

	float coef;
	float3 valTop, valBottom, val;
	uchar *ptr0, *ptr1;
	int uBi0, uBi1, vBi0, vBi1;
	// bilinear intarpolation
	uBi0 = __float2int_rd(u); uBi1 = uBi0 + 1;
	vBi0 = __float2int_rd(v); vBi1 = vBi0 + 1;
	if (uBi0 < 0 || vBi0 < 0 && uBi1 >= width - 1 && vBi1 >= height - 1)
	{
		return;
	}
	coef = (uBi1 - u) / (float)(uBi1 - uBi0);
	ptr0 = img + 3 * (vBi0 * width + uBi0);
	ptr1 = img + 3 * (vBi0 * width + uBi1);
	valTop = coef * make_float3(*ptr0, *(ptr0 + 1), *(ptr0 + 2)) +
		(1 - coef) * make_float3(*ptr1, *(ptr1 + 1), *(ptr1 + 2));
	ptr0 = img + 3 * (vBi1 * width + uBi0);
	ptr1 = img + 3 * (vBi1 * width + uBi1);
	valBottom = coef * make_float3(*ptr0, *(ptr0 + 1), *(ptr0 + 2)) +
		(1 - coef) * make_float3(*ptr1, *(ptr1 + 1), *(ptr1 + 2));
	coef = (vBi1 - v) / (float)(vBi1 - vBi0);
	val = coef * valTop + (1 - coef) * valBottom;

	uint rgb = 0;
	rgb = (uint)val.z;
	rgb = ((uint)val.y << 8) + rgb;
	rgb = ((uint)val.x << 16) + rgb;
	vboCudaSrc[vertexInd].colorTime.x = rgb;
}

// Just copy the updated poses and normals into the vbo
void ApplyUpdatedVertexPosesAndNormals(VBOType *vboDevice,
	int vertexNum,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	uchar *keyColorImgsDevice, int keyColorImgsStep,
	float4 *keyPosesInvDevice,
	float fx, float fy, float cx, float cy, int width, int height)
{
	int block = 256, grid;

	grid = DivUp(vertexNum, block);
	ApplyUpdatedVertexPosesAndNormalsKernel << <grid, block >> > (
		vboDevice, vertexNum,
		updatedVertexPosesDevice, updatedVertexNormalsDevice,
		keyColorImgsDevice, keyColorImgsStep,
		keyPosesInvDevice,
		fx, fy, cx, cy, width, height);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void ApplyUpdatedVertexPosesAndNormalsKernel(
	VBOType *vbo, int vertexNum,
	float4 *updatedVertexPoses, float4 *updatedVertexNormals,
	uchar *keyColorImgs, int keyColorImgsStep,
	float4 *keyPosesInv,
	float fx, float fy, float cx, float cy, int width, int height)
{
	int vertexInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexInd >= vertexNum)
		return;

	float4 updatedPos = updatedVertexPoses[vertexInd];
	float4 updatedNormal = updatedVertexNormals[vertexInd];

	vbo[vertexInd].posConf.x = updatedPos.x;
	vbo[vertexInd].posConf.y = updatedPos.y;
	vbo[vertexInd].posConf.z = updatedPos.z;
	vbo[vertexInd].normalRad.x = updatedNormal.x;
	vbo[vertexInd].normalRad.y = updatedNormal.y;
	vbo[vertexInd].normalRad.z = updatedNormal.z;

#if 1
	int fragInd = vbo[vertexInd].colorTime.y;
	float4 *keyPoseInv = keyPosesInv + 4 * fragInd;
	uchar *keyColorImg = keyColorImgs + keyColorImgsStep * fragInd;

	float4 posLocal = updatedPos.x * keyPoseInv[0] + updatedPos.y * keyPoseInv[1] +
		updatedPos.z * keyPoseInv[2] + keyPoseInv[3];

	float u = (posLocal.x * fx) / posLocal.z + cx;
	float v = (posLocal.y * fy) / posLocal.z + cy;

	float coef;
	float3 valTop, valBottom, val;
	uchar *ptr0, *ptr1;
	int uBi0, uBi1, vBi0, vBi1;
	// bilinear intarpolation
	uBi0 = __float2int_rd(u); uBi1 = uBi0 + 1;
	vBi0 = __float2int_rd(v); vBi1 = vBi0 + 1;
	if (uBi0 < 0 || vBi0 < 0 && uBi1 >= width - 1 && vBi1 >= height - 1)
	{
		return;
	}
	coef = (uBi1 - u) / (float)(uBi1 - uBi0);
	ptr0 = keyColorImg + 3 * (vBi0 * width + uBi0);
	ptr1 = keyColorImg + 3 * (vBi0 * width + uBi1);
	valTop = coef * make_float3(*ptr0, *(ptr0 + 1), *(ptr0 + 2)) +
		(1 - coef) * make_float3(*ptr1, *(ptr1 + 1), *(ptr1 + 2));
	ptr0 = keyColorImg + 3 * (vBi1 * width + uBi0);
	ptr1 = keyColorImg + 3 * (vBi1 * width + uBi1);
	valBottom = coef * make_float3(*ptr0, *(ptr0 + 1), *(ptr0 + 2)) +
		(1 - coef) * make_float3(*ptr1, *(ptr1 + 1), *(ptr1 + 2));
	coef = (vBi1 - v) / (float)(vBi1 - vBi0);
	val = coef * valTop + (1 - coef) * valBottom;

	uint rgb = 0;
	rgb = (uint)val.z;
	rgb = ((uint)val.y << 8) + rgb;
	rgb = ((uint)val.x << 16) + rgb;
	vbo[vertexInd].colorTime.x = rgb;
#endif
}

void DistToWeight(float *vertexToNodeDistDevice, int vertexNum,
	float *nodeToNodeDistDevice, int nodeNum)
{
	thrust::device_ptr<float> devPtrVertexToNode(vertexToNodeDistDevice);

	float sum = thrust::reduce(devPtrVertexToNode, devPtrVertexToNode + vertexNum * 4, (float)0, thrust::plus<float>());
	assert(sum > MYEPS);
	//std::cout << sum << std::endl;
	float variance = 2 * pow(0.5 * sum / (vertexNum * 4), 2); // variance of gaussian	

	int block = 256;
	int grid = DivUp(vertexNum, block);
	DistToWeightVertexKernel << <grid, block >> > (vertexToNodeDistDevice, 1.0f / variance, vertexNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	grid = DivUp(nodeNum, block);
	DistToWeightNodeKernel << <grid, block >> > (nodeToNodeDistDevice, nodeNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void DistToWeightVertexKernel(float *vertexToNodeDistDevice, float varianceInv, int vertexNum)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= vertexNum)
		return;

	float nearDist0, nearDist1, nearDist2, nearDist3;
	float *distPtr;
	distPtr = vertexToNodeDistDevice + 4 * idx;
	nearDist0 = *distPtr;
	nearDist1 = *(distPtr + 1);
	nearDist2 = *(distPtr + 2);
	nearDist3 = *(distPtr + 3);

	nearDist0 = exp(-nearDist0 * nearDist0 * varianceInv) + MYEPS;
	nearDist1 = exp(-nearDist1 * nearDist1 * varianceInv) + MYEPS;
	nearDist2 = exp(-nearDist2 * nearDist2 * varianceInv) + MYEPS;
	nearDist3 = exp(-nearDist3 * nearDist3 * varianceInv) + MYEPS;

	float sum = nearDist0 + nearDist1 + nearDist2 + nearDist3;

	assert(!isnan(nearDist0 / sum));
	assert(!isnan(nearDist1 / sum));
	assert(!isnan(nearDist2 / sum));
	assert(!isnan(nearDist3 / sum));
	*distPtr = nearDist0 / sum;
	*(distPtr + 1) = nearDist1 / sum;
	*(distPtr + 2) = nearDist2 / sum;
	*(distPtr + 3) = nearDist3 / sum;

#if 0
	* distPtr = 1.0 / 4;
	*(distPtr + 1) = 1.0 / 4;
	*(distPtr + 2) = 1.0 / 4;
	*(distPtr + 3) = 1.0 / 4;
#endif
}

__global__ void DistToWeightNodeKernel(float *nodeToNodeDist, int nodeNum)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= nodeNum)
		return;

	float *distPtr;
	distPtr = nodeToNodeDist + 8 * idx;

	*distPtr = sqrt(1.0 / 8);
	*(distPtr + 1) = sqrt(1.0 / 8);
	*(distPtr + 2) = sqrt(1.0 / 8);
	*(distPtr + 3) = sqrt(1.0 / 8);
	*(distPtr + 4) = sqrt(1.0 / 8);
	*(distPtr + 5) = sqrt(1.0 / 8);
	*(distPtr + 6) = sqrt(1.0 / 8);
	*(distPtr + 7) = sqrt(1.0 / 8);
}

void UpdateRts(VBOType **vboCudaSrcPtrs, int *nodeVIndices,
	float3 *Rts, int fragNum, float4 *keyPose, float4 *updatedKeyPose)
{
	/*
	int nodeNum = NODE_NUM * fragNum;
	int block = 256;
	int grid = DivUp(nodeNum, block);
	UpdateRtsKernal << <grid, block >> > (vboCudaSrcPtrs, nodeVIndices, Rts, nodeNum, keyPose, updatedKeyPose);
	*/
}

__global__ void UpdateRtsKernal(VBOType **vboCudaSrcPtrs, int *nodeVIndices,
	float3 *Rts, int nodeNum, float4 *keyPose, float4 *updatedKeyPose)
{
	/*
	int totalNodeInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (totalNodeInd >= nodeNum)
	return;

	int fragInd = totalNodeInd / NODE_NUM;

	VBOType *vboCudaSrcFrag = vboCudaSrcPtrs[fragInd];
	float4 *posConf = &vboCudaSrcFrag[nodeVIndices[totalNodeInd]].posConf;

	float4 *keyPoseFrag = keyPose + fragInd * 4;
	float4 *updatedKeyPoseFrag = updatedKeyPose + fragInd * 4;

	float3 R_cam[3], t_cam;

	R_cam[0].x = updatedKeyPoseFrag[0].x * keyPoseFrag[0].x + updatedKeyPoseFrag[1].x * keyPoseFrag[1].x + updatedKeyPoseFrag[2].x * keyPoseFrag[2].x;
	R_cam[1].x = updatedKeyPoseFrag[0].x * keyPoseFrag[0].y + updatedKeyPoseFrag[1].x * keyPoseFrag[1].y + updatedKeyPoseFrag[2].x * keyPoseFrag[2].y;
	R_cam[2].x = updatedKeyPoseFrag[0].x * keyPoseFrag[0].z + updatedKeyPoseFrag[1].x * keyPoseFrag[1].z + updatedKeyPoseFrag[2].x * keyPoseFrag[2].z;

	R_cam[0].y = updatedKeyPoseFrag[0].y * keyPoseFrag[0].x + updatedKeyPoseFrag[1].y * keyPoseFrag[1].x + updatedKeyPoseFrag[2].y * keyPoseFrag[2].x;
	R_cam[1].y = updatedKeyPoseFrag[0].y * keyPoseFrag[0].y + updatedKeyPoseFrag[1].y * keyPoseFrag[1].y + updatedKeyPoseFrag[2].y * keyPoseFrag[2].y;
	R_cam[2].y = updatedKeyPoseFrag[0].y * keyPoseFrag[0].z + updatedKeyPoseFrag[1].y * keyPoseFrag[1].z + updatedKeyPoseFrag[2].y * keyPoseFrag[2].z;

	R_cam[0].z = updatedKeyPoseFrag[0].z * keyPoseFrag[0].x + updatedKeyPoseFrag[1].z * keyPoseFrag[1].x + updatedKeyPoseFrag[2].z * keyPoseFrag[2].x;
	R_cam[1].z = updatedKeyPoseFrag[0].z * keyPoseFrag[0].y + updatedKeyPoseFrag[1].z * keyPoseFrag[1].y + updatedKeyPoseFrag[2].z * keyPoseFrag[2].y;
	R_cam[2].z = updatedKeyPoseFrag[0].z * keyPoseFrag[0].z + updatedKeyPoseFrag[1].z * keyPoseFrag[1].z + updatedKeyPoseFrag[2].z * keyPoseFrag[2].z;

	t_cam = -(R_cam[0] * keyPoseFrag[3].x + R_cam[1] * keyPoseFrag[3].y + R_cam[2] * keyPoseFrag[3].z)
	+ make_float3(updatedKeyPoseFrag[3]);

	float3 *RtsFrag = Rts + totalNodeInd * 4;
	RtsFrag[0] = R_cam[0];
	RtsFrag[1] = R_cam[1];
	RtsFrag[2] = R_cam[2];
	RtsFrag[3] = posConf->x * R_cam[0] + posConf->y * R_cam[1] + posConf->z * R_cam[2] + t_cam
	- make_float3(*posConf);
	*/
}

void AddDeltaRtstoRts(float *x, float *delta_x, int length)
{
	thrust::device_ptr<float> x_dev_ptr(x);
	thrust::device_ptr<float> delta_x_dev_ptr(delta_x);

	thrust::plus<float> op_plus;
	thrust::transform(x_dev_ptr, x_dev_ptr + length, delta_x_dev_ptr, x_dev_ptr, op_plus);

#if 0
	thrust::host_vector<float> delta_vec_host(length);
	thrust::copy(delta_x_dev_ptr, delta_x_dev_ptr + length, delta_vec_host.begin());
	for (int i = 0; i < delta_vec_host.size(); ++i)
	{
		delta_vec_host[i] *= scale;// 0.01;// 0.0001;
	}
	thrust::device_vector<float> delta_vec_device = delta_vec_host;
	thrust::transform(x_dev_ptr, x_dev_ptr + length, delta_vec_device.begin(), x_dev_ptr, op_plus);
#endif
}

void CalcInvTransRot(float *Rs_transinv, float *Rts, int length)
{
	int block = 256;
	int grid = DivUp(length, block);
	CalcInvTransRotKernal << <grid, block >> > (Rts, Rs_transinv, length);
}

__global__ void CalcInvTransRotKernal(float *Rts, float *Rs_transinv, int length)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= length)
		return;

	float *R = Rts + 12 * idx;
	float *R_transinv = Rs_transinv + 9 * idx;

	float detM = R[0] * R[4] * R[8] + R[2] * R[3] * R[7] + R[1] * R[5] * R[6]
		- R[2] * R[4] * R[6] - R[1] * R[3] * R[8] - R[0] * R[5] * R[7];

	// inverse tranlation
	R_transinv[0] = +(R[4] * R[8] - R[5] * R[7]) / detM;
	R_transinv[1] = -(R[3] * R[8] - R[5] * R[6]) / detM;
	R_transinv[2] = +(R[3] * R[7] - R[4] * R[6]) / detM;
	R_transinv[3] = -(R[1] * R[8] - R[2] * R[7]) / detM;
	R_transinv[4] = +(R[0] * R[8] - R[2] * R[6]) / detM;
	R_transinv[5] = -(R[0] * R[7] - R[1] * R[6]) / detM;
	R_transinv[6] = +(R[1] * R[5] - R[2] * R[4]) / detM;
	R_transinv[7] = -(R[0] * R[5] - R[2] * R[3]) / detM;
	R_transinv[8] = +(R[0] * R[4] - R[1] * R[3]) / detM;

	/*
	double R_0 = 1.0f;
	double R_1 = 1.0f;
	double R_2 = 2.0f;
	double R_3 = 0.0f;
	double R_4 = 1.0f;
	double R_5 = 3.0f;
	double R_6 = 1.0f;
	double R_7 = 1.0f;
	double R_8 = 1.0f;

	detM = R_0 * R_4 * R_8 + R_2 * R_3 * R_7 + R_1 * R_5 * R_6
	- R_2 * R_4 * R_6 - R_1 * R_3 * R_8 - R_0 * R_5 * R_7;

	double R_transinv_0 = +(R_4 * R_8 - R_5 * R_7) / detM;
	double R_transinv_1 = -(R_3 * R_8 - R_5 * R_6) / detM;
	double R_transinv_2 = +(R_3 * R_7 - R_4 * R_6) / detM;
	double R_transinv_3 = -(R_1 * R_8 - R_2 * R_7) / detM;
	double R_transinv_4= + (R_0 * R_8 - R_2 * R_6) / detM;
	double R_transinv_5 = -(R_0 * R_7 - R_1 * R_6) / detM;
	double R_transinv_6 = +(R_1 * R_5 - R_2 * R_4) / detM;
	double R_transinv_7 = -(R_0 * R_5 - R_2 * R_3) / detM;
	double R_transinv_8 = +(R_0 * R_4 - R_1 * R_3) / detM;

	double val0_d = R_0 * R_transinv_0 + R_1 * R_transinv_1 + R_2 * R_transinv_2;
	double val1_d = R_0 * R_transinv_3 + R_1 * R_transinv_4 + R_2 * R_transinv_5;
	double val2_d = R_0 * R_transinv_6 + R_1 * R_transinv_7 + R_2 * R_transinv_8;
	double val3_d = R_3 * R_transinv_0 + R_4 * R_transinv_1 + R_5 * R_transinv_2;
	double val4_d = R_3 * R_transinv_3 + R_4 * R_transinv_4 + R_5 * R_transinv_5;
	double val5_d = R_3 * R_transinv_6 + R_4 * R_transinv_7 + R_5 * R_transinv_8;
	double val6_d = R_6 * R_transinv_0 + R_7 * R_transinv_1 + R_8 * R_transinv_2;
	double val7_d = R_6 * R_transinv_3 + R_7 * R_transinv_4 + R_8 * R_transinv_5;
	double val8_d = R_6 * R_transinv_6 + R_7 * R_transinv_7 + R_8 * R_transinv_8;

	printf("val: %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", val0_d, val1_d, val2_d, val3_d, val4_d, val5_d, val6_d, val7_d, val8_d);
	*/
}

void FindMatchingPointsPerspective(
	int *matchingPointsDevice,
	int *matchingFragsIndicesDevice,
	float4 *updatedVertexPosesDevice,
	std::pair<int *, int> &indMapsDevice,
	float4 *updatedKeyPosesInvDevice,
	int *sampledVertexIndicesDevice,
	int width, int height, float fx, float fy, float cx, float cy,
	int matchingPointsNum)
{
	int block = 256, grid = DivUp(matchingPointsNum, block);
	FindMatchingPointsPerspectiveKernel << <grid, block >> > (
		matchingPointsDevice,
		matchingFragsIndicesDevice,
		updatedVertexPosesDevice,
		indMapsDevice.first, indMapsDevice.second,
		updatedKeyPosesInvDevice,
		sampledVertexIndicesDevice,
		width, height, fx, fy, cx, cy,
		matchingPointsNum);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void FindMatchingPointsPerspectiveKernel(
	int *matchingPoints,
	int *matchingFragsIndices,
	float4 *updatedVertexPoses,
	int *indMaps, int indMapsStep,
	float4 *updatedKeyPosesInv,
	int *sampledVertexIndices,
	int width, int height, float fx, float fy, float cx, float cy,
	int matchingPointsNum)
{
	int vertexPairInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexPairInd >= matchingPointsNum)
	{
		return;
	}

	int srcFragInd = matchingFragsIndices[2 * (vertexPairInd / SAMPLED_VERTEX_NUM_EACH_FRAG)];
	int targetFragInd = matchingFragsIndices[2 * (vertexPairInd / SAMPLED_VERTEX_NUM_EACH_FRAG) + 1];
	//printf("%d %d, ", srcFragInd, targetFragInd);

	matchingPoints[2 * vertexPairInd + 1] = -1;

	int vertexIndSrc, vertexIndTarget[25];
	float4 *updatedPosSrc, *updatedPosTarget;
	vertexIndSrc = *(sampledVertexIndices + SAMPLED_VERTEX_NUM_EACH_FRAG * srcFragInd + vertexPairInd % SAMPLED_VERTEX_NUM_EACH_FRAG);
	updatedPosSrc = updatedVertexPoses + vertexIndSrc;

	matchingPoints[2 * vertexPairInd] = vertexIndSrc;

	float4 *updatedPoseInvDeviceTarget = updatedKeyPosesInv + 4 * targetFragInd;
	float4 updatedVertexPosLocalTargetSpace = updatedPoseInvDeviceTarget[0] * updatedPosSrc->x +
		updatedPoseInvDeviceTarget[1] * updatedPosSrc->y +
		updatedPoseInvDeviceTarget[2] * updatedPosSrc->z +
		updatedPoseInvDeviceTarget[3];

	int uToTarget = __float2int_rn((updatedVertexPosLocalTargetSpace.x * fx) / updatedVertexPosLocalTargetSpace.z + cx);
	int vToTarget = __float2int_rn((updatedVertexPosLocalTargetSpace.y * fy) / updatedVertexPosLocalTargetSpace.z + cy);

	float dist;
	int steps[4] = { 1, 2, 3, 5 };
	int *indMapDeviceTarget = indMaps + indMapsStep * targetFragInd;
	if (uToTarget < steps[3] || vToTarget < steps[3] || uToTarget >= width - steps[3] || vToTarget >= height - steps[3])
	{
		return;
	}

	int step = steps[0];
	vertexIndTarget[0] = *(indMapDeviceTarget + vToTarget * width + uToTarget);
	vertexIndTarget[1] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget - step));
	vertexIndTarget[2] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget - step));
	vertexIndTarget[3] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget + step));
	vertexIndTarget[4] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget + step));
	vertexIndTarget[5] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget));
	vertexIndTarget[6] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget));
	vertexIndTarget[7] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	vertexIndTarget[8] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	step = steps[1];
	vertexIndTarget[9] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget - step));
	vertexIndTarget[10] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget - step));
	vertexIndTarget[11] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget + step));
	vertexIndTarget[12] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget + step));
	vertexIndTarget[13] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget));
	vertexIndTarget[14] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget));
	vertexIndTarget[15] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	vertexIndTarget[16] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	step = steps[2];
	vertexIndTarget[17] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget - step));
	vertexIndTarget[18] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget - step));
	vertexIndTarget[19] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget + step));
	vertexIndTarget[20] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget + step));
	vertexIndTarget[21] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget));
	vertexIndTarget[22] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget));
	vertexIndTarget[23] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	vertexIndTarget[24] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	step = steps[3];
	vertexIndTarget[25] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget - step));
	vertexIndTarget[26] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget - step));
	vertexIndTarget[27] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget + step));
	vertexIndTarget[28] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget + step));
	vertexIndTarget[29] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget));
	vertexIndTarget[30] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget));
	vertexIndTarget[31] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	vertexIndTarget[32] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));

	int vertexIndTargetNearest = -1;
	float minDist = 1.0e24f;
	for (int i = 0; i < 33; ++i)
	{
		if (vertexIndTarget[i] >= 0)
		{
			updatedPosTarget = updatedVertexPoses + vertexIndTarget[i];
			dist = norm(*updatedPosSrc - *updatedPosTarget);
			if (dist < minDist)
			{
				vertexIndTargetNearest = vertexIndTarget[i];
				minDist = dist;
			}
		}
	}

	if (vertexIndTargetNearest == -1)
	{
		return;
	}

#if 0
	printf("updated src pos: %f %f %f\n", updatedPosSrc->x, updatedPosSrc->y, updatedPosSrc->z);
	printf("updated target pos: %f %f %f\n", updatedPosTarget->x, updatedPosTarget->y, updatedPosTarget->z);
#endif

	matchingPoints[2 * vertexPairInd + 1] = vertexIndTargetNearest;
}


void FindMatchingPointsPerspectiveFromVirtualCamera(int *matchingPointsDevice,
	int *matchingFragsIndicesDevice,
	float4 *updatedVertexPosesDevice,
	std::pair<int *, int> &indMapsDevice,
	int *sampledVertexIndicesDevice,
	float *minMatchingPointDist,
	float4 *virtualCameraPosesInvDevice,
	float4 *virtualCameraFxFyCxCyDevice,
	int width, int height, int matchingPointsNum,
	int fragNum, int virtualCameraCircle)
{
	int block = 256, grid = DivUp(matchingPointsNum, block);

	FindMatchingPointsPerspectiveFromVirtualCameraKernel << <grid, block >> > (matchingPointsDevice,
		matchingFragsIndicesDevice,
		updatedVertexPosesDevice,
		indMapsDevice.first, indMapsDevice.second,
		sampledVertexIndicesDevice,
		minMatchingPointDist,
		virtualCameraPosesInvDevice,
		virtualCameraFxFyCxCyDevice,
		width, height, matchingPointsNum,
		fragNum, virtualCameraCircle);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

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
	int fragNum, int virtualCameraCircle)
{
	int vertexPairInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexPairInd >= matchingPointsNum)
	{
		return;
	}

	int srcFragInd = matchingFragsIndices[2 * (vertexPairInd / SAMPLED_VERTEX_NUM_EACH_FRAG)];
	int targetFragInd = matchingFragsIndices[2 * (vertexPairInd / SAMPLED_VERTEX_NUM_EACH_FRAG) + 1];
	//printf("%d %d, ", srcFragInd, targetFragInd);

	matchingPoints[2 * vertexPairInd + 1] = -1;

	int vertexIndSrc, vertexIndTarget[25];
	float4 *updatedPosSrc, *updatedPosTarget;
	vertexIndSrc = *(sampledVertexIndices + SAMPLED_VERTEX_NUM_EACH_FRAG * srcFragInd + vertexPairInd % SAMPLED_VERTEX_NUM_EACH_FRAG);
	updatedPosSrc = updatedVertexPoses + vertexIndSrc;

	matchingPoints[2 * vertexPairInd] = vertexIndSrc;

	float4 *virtualCameraPosesInvFrag = virtualCameraPosesInv + 4 * fragNum * virtualCameraCircle + 4 * targetFragInd;
	float4 *virtualCameraFxFyCxCyFrag = virtualCameraFxFyCxCy + fragNum * virtualCameraCircle + targetFragInd;

	float4 updatedVertexPosLocalTargetSpace = virtualCameraPosesInvFrag[0] * updatedPosSrc->x +
		virtualCameraPosesInvFrag[1] * updatedPosSrc->y +
		virtualCameraPosesInvFrag[2] * updatedPosSrc->z +
		virtualCameraPosesInvFrag[3];

	int uToTarget = __float2int_rn((updatedVertexPosLocalTargetSpace.x * virtualCameraFxFyCxCyFrag->x) / updatedVertexPosLocalTargetSpace.z + virtualCameraFxFyCxCyFrag->z);
	int vToTarget = __float2int_rn((updatedVertexPosLocalTargetSpace.y * virtualCameraFxFyCxCyFrag->y) / updatedVertexPosLocalTargetSpace.z + virtualCameraFxFyCxCyFrag->w);

	float dist;
	int steps[4] = { 1, 2, 3, 5 };
	int *indMapDeviceTarget = indMaps + indMapsStep * targetFragInd;
	if (uToTarget < steps[3] || vToTarget < steps[3] || uToTarget >= width - steps[3] || vToTarget >= height - steps[3])
	{
		return;
	}

	int step = steps[0];
	vertexIndTarget[0] = *(indMapDeviceTarget + vToTarget * width + uToTarget);
	vertexIndTarget[1] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget - step));
	vertexIndTarget[2] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget - step));
	vertexIndTarget[3] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget + step));
	vertexIndTarget[4] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget + step));
	vertexIndTarget[5] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget));
	vertexIndTarget[6] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget));
	vertexIndTarget[7] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	vertexIndTarget[8] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	step = steps[1];
	vertexIndTarget[9] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget - step));
	vertexIndTarget[10] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget - step));
	vertexIndTarget[11] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget + step));
	vertexIndTarget[12] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget + step));
	vertexIndTarget[13] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget));
	vertexIndTarget[14] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget));
	vertexIndTarget[15] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	vertexIndTarget[16] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	step = steps[2];
	vertexIndTarget[17] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget - step));
	vertexIndTarget[18] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget - step));
	vertexIndTarget[19] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget + step));
	vertexIndTarget[20] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget + step));
	vertexIndTarget[21] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget));
	vertexIndTarget[22] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget));
	vertexIndTarget[23] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	vertexIndTarget[24] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	step = steps[3];
	vertexIndTarget[25] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget - step));
	vertexIndTarget[26] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget - step));
	vertexIndTarget[27] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget + step));
	vertexIndTarget[28] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget + step));
	vertexIndTarget[29] = *(indMapDeviceTarget + (vToTarget - step) * width + (uToTarget));
	vertexIndTarget[30] = *(indMapDeviceTarget + (vToTarget + step) * width + (uToTarget));
	vertexIndTarget[31] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));
	vertexIndTarget[32] = *(indMapDeviceTarget + (vToTarget)* width + (uToTarget + step));

	int vertexIndTargetNearest = -1;
	float minDist = minMatchingPointDist[vertexPairInd];
	for (int i = 0; i < 33; ++i)
	{
		if (vertexIndTarget[i] >= 0)
		{
			updatedPosTarget = updatedVertexPoses + vertexIndTarget[i];
			dist = norm(*updatedPosSrc - *updatedPosTarget);
			if (dist < minDist)
			{
				vertexIndTargetNearest = vertexIndTarget[i];
				minDist = dist;
			}
		}
	}

	if (vertexIndTargetNearest == -1)
	{
		return;
	}

#if 0
	printf("updated src pos: %f %f %f\n", updatedPosSrc->x, updatedPosSrc->y, updatedPosSrc->z);
	printf("updated target pos: %f %f %f\n", updatedPosTarget->x, updatedPosTarget->y, updatedPosTarget->z);
#endif

	matchingPoints[2 * vertexPairInd + 1] = vertexIndTargetNearest;
}

static __device__ __forceinline__ void SortNearInd(int *fourNearInd, float *fourNearWeight)
{
	int tmp1;
	float temp2;
	// 4 elements
#pragma unroll 3
	for (int mini = 0; mini < 3; ++mini)
	{
		for (int ind = 3; ind > mini; --ind)
		{
			if (fourNearInd[ind] < fourNearInd[ind - 1])
			{
				tmp1 = fourNearInd[ind];
				fourNearInd[ind] = fourNearInd[ind - 1];
				fourNearInd[ind - 1] = tmp1;

				temp2 = fourNearWeight[ind];
				fourNearWeight[ind] = fourNearWeight[ind - 1];
				fourNearWeight[ind - 1] = temp2;
			}
		}
	}
}

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
	int iter)
{
	int block = 256, grid;
	int matchingPointsNumRest = matchingPointsNumTotal - matchingPointsNumDescriptor;
	std::cout << "matchingPointsNumDescriptor: " << matchingPointsNumDescriptor << std::endl;
	std::cout << "matchingPointsNumRest: " << matchingPointsNumRest << std::endl;
	int iaOffset = 0, residualOffset = 0;
	innoreal::InnoRealTimer timer;
	timer.TimeStart();

	switch (gs::geoRegistrationType)
	{
#if 1
	case BOTH:
		grid = DivUp(matchingPointsNumDescriptor, block);
		if (grid > 0)
		{
			ComputeGeoTermPointToPointKernel << <grid, block >> > (
				jacobian.m_ia, iaOffset, jacobian.m_ja, jacobian.m_a, residual,
				matchingPointsDevice,
				matchingPointsNumDescriptor,
				originVertexPosesDevice,
				updatedVertexPosesDevice,
				updatedVertexNormalsDevice,
				vertexToNodeIndicesDevice,
				vertexToNodeWeightsDevice,
				nodeVIndicesDevice,
				w_geo,
				iter);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());
			iaOffset += matchingPointsNumDescriptor * (4 * 4 * 6);
			residualOffset += matchingPointsNumDescriptor * 3;

			residual += matchingPointsNumDescriptor * 3;
			jacobian.m_ia += matchingPointsNumDescriptor * 3;
			jacobian.m_a += matchingPointsNumDescriptor * (4 * 4 * 6);
			jacobian.m_ja += matchingPointsNumDescriptor * (4 * 4 * 6);
		}

		grid = DivUp(matchingPointsNumRest, block);
		if (grid > 0)
		{
			ComputeGeoTermPointToPlainKernel << <grid, block >> > (
				jacobian.m_ia, iaOffset, jacobian.m_ja, jacobian.m_a, residual,
				matchingPointsDevice + 2 * matchingPointsNumDescriptor,
				matchingPointsNumRest,
				originVertexPosesDevice,
				updatedVertexPosesDevice,
				updatedVertexNormalsDevice,
				vertexToNodeIndicesDevice,
				vertexToNodeWeightsDevice,
				nodeVIndicesDevice,
				w_geo,
				iter);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());
			iaOffset += matchingPointsNumRest * (4 * 4 * 6);
			residualOffset += matchingPointsNumRest * 1;

			residual += matchingPointsNumRest * 1;
			jacobian.m_ia += matchingPointsNumRest * 1;
			jacobian.m_a += matchingPointsNumRest * (4 * 4 * 6);
			jacobian.m_ja += matchingPointsNumRest * (4 * 4 * 6);
		}
		break;

	case POINT_TO_POINT:
		grid = DivUp(matchingPointsNumDescriptor, block);
		ComputeGeoTermPointToPointKernel << <grid, block >> > (
			jacobian.m_ia, iaOffset, jacobian.m_ja, jacobian.m_a, residual,
			matchingPointsDevice,
			matchingPointsNumDescriptor,
			originVertexPosesDevice,
			updatedVertexPosesDevice,
			updatedVertexNormalsDevice,
			vertexToNodeIndicesDevice,
			vertexToNodeWeightsDevice,
			nodeVIndicesDevice,
			w_geo,
			iter);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		iaOffset += matchingPointsNumDescriptor * (4 * 4 * 6);
		residualOffset += matchingPointsNumDescriptor * 3;

		residual += matchingPointsNumDescriptor * 3;
		jacobian.m_ia += matchingPointsNumDescriptor * 3;
		jacobian.m_a += matchingPointsNumDescriptor * (4 * 4 * 6);
		jacobian.m_ja += matchingPointsNumDescriptor * (4 * 4 * 6);

		grid = DivUp(matchingPointsNumRest, block);
		ComputeGeoTermPointToPointKernel << <grid, block >> > (
			jacobian.m_ia, iaOffset, jacobian.m_ja, jacobian.m_a, residual,
			matchingPointsDevice + 2 * matchingPointsNumDescriptor,
			matchingPointsNumRest,
			originVertexPosesDevice,
			updatedVertexPosesDevice,
			updatedVertexNormalsDevice,
			vertexToNodeIndicesDevice,
			vertexToNodeWeightsDevice,
			nodeVIndicesDevice,
			w_geo,
			iter);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		iaOffset += matchingPointsNumRest * (4 * 4 * 6);
		residualOffset += matchingPointsNumRest * 3;

		residual += matchingPointsNumRest * 3;
		jacobian.m_ia += matchingPointsNumRest * 3;
		jacobian.m_a += matchingPointsNumRest * (4 * 4 * 6);
		jacobian.m_ja += matchingPointsNumRest * (4 * 4 * 6);
		break;

	case POINT_TO_PLAIN:
		grid = DivUp(matchingPointsNumDescriptor, block);
		if (grid > 0)
		{
			ComputeGeoTermPointToPlainKernel << <grid, block >> > (
				jacobian.m_ia, iaOffset, jacobian.m_ja, jacobian.m_a, residual,
				matchingPointsDevice,
				matchingPointsNumDescriptor,
				originVertexPosesDevice,
				updatedVertexPosesDevice,
				updatedVertexNormalsDevice,
				vertexToNodeIndicesDevice,
				vertexToNodeWeightsDevice,
				nodeVIndicesDevice,
				w_geo,
				iter);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaGetLastError());
			iaOffset += matchingPointsNumDescriptor * (4 * 4 * 6);
			residualOffset += matchingPointsNumDescriptor * 1;

			residual += matchingPointsNumDescriptor * 1;
			jacobian.m_ia += matchingPointsNumDescriptor * 1;
			jacobian.m_a += matchingPointsNumDescriptor * (4 * 4 * 6);
			jacobian.m_ja += matchingPointsNumDescriptor * (4 * 4 * 6);
		}

		grid = DivUp(matchingPointsNumRest, block);
		ComputeGeoTermPointToPlainKernel << <grid, block >> > (
			jacobian.m_ia, iaOffset, jacobian.m_ja, jacobian.m_a, residual,
			matchingPointsDevice + 2 * matchingPointsNumDescriptor,
			matchingPointsNumRest,
			originVertexPosesDevice,
			updatedVertexPosesDevice,
			updatedVertexNormalsDevice,
			vertexToNodeIndicesDevice,
			vertexToNodeWeightsDevice,
			nodeVIndicesDevice,
			w_geo,
			iter);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		iaOffset += matchingPointsNumRest * (4 * 4 * 6);
		residualOffset += matchingPointsNumRest * 1;

		residual += matchingPointsNumRest * 1;
		jacobian.m_ia += matchingPointsNumRest * 1;
		jacobian.m_a += matchingPointsNumRest * (4 * 4 * 6);
		jacobian.m_ja += matchingPointsNumRest * (4 * 4 * 6);
		break;
#endif
	}
	timer.TimeEnd();
	std::cout << "Compute geo jacobian time: " << timer.TimeGap_in_ms() << std::endl;

	timer.TimeStart();
	switch (gs::photoRegistrationType)
	{
#if 0
	case PHOTOREGISTATION:	
#if 0
		std::ofstream outFile_1, outFile_2, outFile_3, outFile_4, outFile_5, outFile_6, outFile_7, outFile_8;
		outFile_1.open("D:\\tt2_1.txt");
		outFile_2.open("D:\\tt2_2.txt");
		outFile_3.open("D:\\tt2_3.txt");
		outFile_4.open("D:\\tt2_4.txt");
		outFile_5.open("D:\\tt2_5.txt");
		//outFile_6.open("D:\\tt2_6.txt");
		//outFile_7.open("D:\\tt2_7.txt");
		//outFile_8.open("D:\\tt2_8.txt");
		std::cout << "width: " << width << std::endl;
		std::cout << "height: " << height << std::endl;
		std::cout << "fx: " << fx << std::endl;
		std::cout << "fy: " << fy << std::endl;
		std::cout << "cx: " << cx << std::endl;
		std::cout << "cy: " << cy << std::endl;
		std::vector<float> vec1(640 * 480);
		checkCudaErrors(cudaMemcpy(vec1.data(), keyGrayImgsDevice.first, vec1.size() * sizeof(float), cudaMemcpyDeviceToHost));
		std::vector<float> vec2(640 * 480);
		checkCudaErrors(cudaMemcpy(vec2.data(), keyGrayImgsDxDevice.first, vec2.size() * sizeof(float), cudaMemcpyDeviceToHost));
		std::vector<float> vec3(640 * 480);
		checkCudaErrors(cudaMemcpy(vec3.data(), keyGrayImgsDyDevice.first, vec3.size() * sizeof(float), cudaMemcpyDeviceToHost));
		std::vector<int> vec4(matchingPointsNumTotal * 2);
		checkCudaErrors(cudaMemcpy(vec4.data(), matchingPointsDevice, vec4.size() * sizeof(int), cudaMemcpyDeviceToHost));
		std::vector<float> vec5(16 * 2);
		checkCudaErrors(cudaMemcpy(vec5.data(), updatedKeyPosesInvDevice, vec5.size() * sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 0; i < vec1.size(); ++i)
		{
			outFile_1 << vec1[i] << std::endl;
		}
		for (int i = 0; i < vec2.size(); ++i)
		{
			outFile_2 << vec2[i] << std::endl;
		}
		for (int i = 0; i < vec3.size(); ++i)
		{
			outFile_3 << vec3[i] << std::endl;
		}
		for (int i = 0; i < vec4.size(); ++i)
		{
			outFile_4 << vec4[i] << std::endl;
		}
		for (int i = 0; i < vec5.size(); ++i)
		{
			outFile_5 << vec5[i] << std::endl;
		}
		outFile_1.close();
		outFile_2.close();
		outFile_3.close();
		outFile_4.close();
		outFile_5.close();
		std::exit(0);
#endif
		ComputePhotoTermKernel << <grid, block >> > (
			jacobian.m_ia, iaOffset, jacobian.m_ja, jacobian.m_a, residual,
			width, height, fx, fy, cx, cy,
			matchingPointsDevice,
			matchingPointsNumTotal,
			originVertexPosesDevice,
			updatedVertexPosesDevice,
			updatedVertexNormalsDevice,
			keyGrayImgsDevice.first, keyGrayImgsDevice.second,
			keyGrayImgsDxDevice.first, keyGrayImgsDxDevice.second,
			keyGrayImgsDyDevice.first, keyGrayImgsDyDevice.second,
			updatedKeyPosesInvDevice,
			vertexToNodeIndicesDevice,
			vertexToNodeWeightsDevice,
			nodeVIndicesDevice,
			w_photo,
			iter);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
		iaOffset += matchingPointsNumTotal * 12 * 4 * 2;
		residualOffset += matchingPointsNumTotal * 1;

		residual += matchingPointsNumTotal * 1;
		jacobian.m_ia += matchingPointsNumTotal * 1;
		jacobian.m_a += matchingPointsNumTotal * (12 * 4 * 2);
		jacobian.m_ja += matchingPointsNumTotal * (12 * 4 * 2);
		break;
#endif
	}

	timer.TimeEnd();
	std::cout << "Compute photo jacobian time: " << timer.TimeGap_in_ms() << std::endl;

	std::vector<int> ia;
#if 1
	//r_reg and r_rot		
	timer.TimeStart();
	grid = DivUp(nodeNum, block);
	ComputeRegTermsKernel << <grid, block >> > (
		jacobian.m_ja, jacobian.m_a, residual,
		nodeNum,
		Rts,
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		nodeVIndicesDevice,
		nodeToNodeIndicesDevice,
		nodeToNodeWeightsDevice,
		//2.0f, 2.0f, 0.0f);
		w_reg, w_rot, w_trans);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	for (int ind = 0; ind < nodeNum * 8 * 3; ++ind)
	{
		ia.push_back(iaOffset);
		iaOffset += 5;
	}
	for (int ind = 0; ind < nodeNum; ++ind)
	{
		ia.push_back(iaOffset);
		iaOffset += 6;
		ia.push_back(iaOffset);
		iaOffset += 6;
		ia.push_back(iaOffset);
		iaOffset += 6;
		ia.push_back(iaOffset);
		iaOffset += 3;
		ia.push_back(iaOffset);
		iaOffset += 3;
		ia.push_back(iaOffset);
		iaOffset += 3;
	}
#if 0
	for (int ind = 0; ind < nodeNum; ++ind)
	{
		ia.push_back(iaOffset);
		iaOffset += 1;
		ia.push_back(iaOffset);
		iaOffset += 1;
		ia.push_back(iaOffset);
		iaOffset += 1;
	}
#endif
	timer.TimeEnd();
	std::cout << "Compute reg jacobian time: " << timer.TimeGap_in_ms() << std::endl;

#endif

	// fill meta data
	ia.push_back(iaOffset);
	checkCudaErrors(cudaMemcpy(jacobian.m_ia, ia.data(), ia.size() * sizeof(int), cudaMemcpyHostToDevice));
	row_J = residualOffset + ia.size() - 1;
	col_J = nodeNum * 12;
	nnz = ia.back();
}

template<int CTA_SIZE, typename T>
__device__ __forceinline__ float reduce(volatile T *sharedBuf, int tid)
{
	T val = sharedBuf[tid];

	if (CTA_SIZE >= 1024) { if (tid < 512) sharedBuf[tid] = val = val + sharedBuf[tid + 512]; __syncthreads(); }
	if (CTA_SIZE >= 512) { if (tid < 256) sharedBuf[tid] = val = val + sharedBuf[tid + 256]; __syncthreads(); }
	if (CTA_SIZE >= 256) { if (tid < 128) sharedBuf[tid] = val = val + sharedBuf[tid + 128]; __syncthreads(); }
	if (CTA_SIZE >= 128) { if (tid <  64) sharedBuf[tid] = val = val + sharedBuf[tid + 64]; __syncthreads(); }

	if (tid < 32)
	{
		if (CTA_SIZE >= 64) { sharedBuf[tid] = val = val + sharedBuf[tid + 32]; }
		if (CTA_SIZE >= 32) { sharedBuf[tid] = val = val + sharedBuf[tid + 16]; }
		if (CTA_SIZE >= 16) { sharedBuf[tid] = val = val + sharedBuf[tid + 8]; }
		if (CTA_SIZE >= 8) { sharedBuf[tid] = val = val + sharedBuf[tid + 4]; }
		if (CTA_SIZE >= 4) { sharedBuf[tid] = val = val + sharedBuf[tid + 2]; }
		if (CTA_SIZE >= 2) { sharedBuf[tid] = val = val + sharedBuf[tid + 1]; }
	}
	__syncthreads();
	return sharedBuf[0];
}

bool EstimateErrorsBeforeRegistration(std::vector<int> &isFragValid,
	std::vector<std::vector<int> > &poseGraph,
	float *residualSumDevice, int *residualNnzDevice, float *residual,
	VBOType *vboDevice, std::vector<int> &vertexStrideVec,
	int loopClosureNum)
{
	/*int block = SAMPLED_VERTEX_NUM_EACH_FRAG;
	int grid = DivUp(SAMPLED_VERTEX_NUM_EACH_FRAG * loopClosureNum, block);
	CalcResidualSumAndNnzKernel << <grid, block >> > (
		residualSumDevice,
		residualNnzDevice,
		residual);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	int fragNum = poseGraph.size();
	std::vector<float> residualSumVec(loopClosureNum);
	std::vector<int> residualNnzVec(loopClosureNum);
	std::vector<float> scoreVec(fragNum, 0.0f);
	std::vector<int> cntVec(fragNum, 0);
	checkCudaErrors(cudaMemcpy(residualSumVec.data(), residualSumDevice, loopClosureNum * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(residualNnzVec.data(), residualNnzDevice, loopClosureNum * sizeof(int), cudaMemcpyDeviceToHost));
#if 0
	for (int i = 0; i < residualSumVec.size(); ++i)
	{
		std::cout << residualSumVec[i] << ", ";
	}
	std::cout << std::endl;
	for (int i = 0; i < residualNnzVec.size(); ++i)
	{
		std::cout << residualNnzVec[i] << ", ";
	}
	std::cout << std::endl;
	std::exit(0);
#endif

	int srcFragInd, loopClosureInd = 0;
	for (int targetFragInd = 0; targetFragInd < poseGraph.size(); ++targetFragInd)
	{
		for (int ind = 0; ind < poseGraph[targetFragInd].size(); ++ind)
		{
			srcFragInd = poseGraph[targetFragInd][ind];
			scoreVec[srcFragInd] += residualSumVec[loopClosureInd] / (residualNnzVec[loopClosureInd] + MYEPS);
			++cntVec[srcFragInd];
			scoreVec[targetFragInd] += residualSumVec[loopClosureInd] / (residualNnzVec[loopClosureInd] + MYEPS);
			++cntVec[targetFragInd];
			++loopClosureInd;
		}
	}
	bool needOpt = false;
	for (int i = 0; i < scoreVec.size(); ++i)
	{
		scoreVec[i] /= (cntVec[i] + MYEPS);
		std::cout << "frag " << i << ": " << scoreVec[i] << std::endl;
		std::cout << "frag num: " << fragNum << std::endl;
		if (scoreVec[i] > 0.3)
		{
			needOpt = true;
		}
		if (scoreVec[i] > 0.3)
		{
			--isFragValid[i];
			if (isFragValid[i] == -1)
			{
				// Make fragInd -1
				block = 256;
				int vertexNumFrag = vertexStrideVec[i + 1] - vertexStrideVec[i];
				grid = DivUp(vertexNumFrag, block);
				FragmentInvalid << <grid, block >> > (vboDevice + vertexStrideVec[i], vertexNumFrag);
				checkCudaErrors(cudaDeviceSynchronize());
				checkCudaErrors(cudaGetLastError());
			}
		}
	}
	return needOpt;*/

	return false;
}

__global__ void FragmentInvalid(VBOType *vboFrag, int vertexNumFrag)
{
	int vertexIndFrag = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexIndFrag >= vertexNumFrag)
	{
		return;
	}

	vboFrag[vertexIndFrag].colorTime.y = -1;
}

__global__ void CalcResidualSumAndNnzKernel(float *residualSum, int *residualNnz, float *residual)
{
	/*__shared__ float sumSharedBuf[SAMPLED_VERTEX_NUM_EACH_FRAG];
	__shared__ int nnzSharedBuf[SAMPLED_VERTEX_NUM_EACH_FRAG];

	int tid = threadIdx.x;
	int blockId = blockIdx.x;

	float val0 = *(residual + 3 * SAMPLED_VERTEX_NUM_EACH_FRAG * blockId + 3 * tid);
	float val1 = *(residual + 3 * SAMPLED_VERTEX_NUM_EACH_FRAG * blockId + 3 * tid + 1);
	float val2 = *(residual + 3 * SAMPLED_VERTEX_NUM_EACH_FRAG * blockId + 3 * tid + 2);
	val0 = fabs(val0);
	val1 = fabs(val1);
	val2 = fabs(val2);
	//printf("%f %f %f", val0, val1, val2);
	sumSharedBuf[tid] = val0 + val1 + val2;
	nnzSharedBuf[tid] = (int)(val0 > MYEPS) + (int)(val1 > MYEPS) + (int)(val2 > MYEPS);
	__syncthreads();

	residualSum[blockId] = reduce<SAMPLED_VERTEX_NUM_EACH_FRAG, float>(sumSharedBuf, tid);
	residualNnz[blockId] = reduce<SAMPLED_VERTEX_NUM_EACH_FRAG, int>(nnzSharedBuf, tid);*/
}

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
	int iter)
{
	int vertexPairInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexPairInd >= matchingPointsNum)
	{
		return;
	}

	float *jacobianGeo_a = jacobian_a + vertexPairInd * (4 * 4 * 6);
	int *jacobianGeo_ja = jacobian_ja + vertexPairInd * (4 * 4 * 6);

	*(jacobian_ia + 3 * vertexPairInd) = iaOffset + vertexPairInd * (4 * 4 * 6);
	*(jacobian_ia + 3 * vertexPairInd + 1) = iaOffset + vertexPairInd * (4 * 4 * 6) + 4 * 4 * 2;
	*(jacobian_ia + 3 * vertexPairInd + 2) = iaOffset + vertexPairInd * (4 * 4 * 6) + 4 * 4 * 4;

	int vertexIndSrc, vertexIndTarget;
	float4 posFragIndSrc, posFragIndTarget;
	vertexIndSrc = *(matchingPoints + 2 * vertexPairInd);
	vertexIndTarget = *(matchingPoints + 2 * vertexPairInd + 1);
	if (vertexIndTarget == -1 || vertexIndSrc == -1)
	{
		goto Invalid_Matching;
	}

	posFragIndSrc = originVertexPoses[vertexIndSrc];
	posFragIndTarget = originVertexPoses[vertexIndTarget];
	//printf("%d %d\n", vertexIndTarget, vertexIndSrc);	

	float4 *updatedPosSrc, *updatedPosTarget, *updatedNormalSrc, *updatedNormalTarget;

	updatedPosSrc = updatedVertexPoses + vertexIndSrc;
	updatedNormalSrc = updatedVertexNormals + vertexIndSrc;
	updatedPosTarget = updatedVertexPoses + vertexIndTarget;
	updatedNormalTarget = updatedVertexNormals + vertexIndTarget;

	float dist = norm(*updatedPosSrc - *updatedPosTarget);
	float4 srcToTargetVec = normalize(*updatedPosSrc - *updatedPosTarget);
	*updatedNormalSrc = normalize(*updatedNormalSrc);
	*updatedNormalTarget = normalize(*updatedNormalTarget);
	//printf("%f %f %f\n", updatedNormalSrc->x, updatedNormalSrc->y, updatedNormalSrc->z);
	float distThresh = 0.08f, angleThresh = sin(30.0f * 3.14159254f / 180.f);
	if (iter % 11 == 10)
	{
		distThresh /= 1.1f;
		angleThresh /= 1.1f;
	}
	if (dot(*updatedNormalSrc, *updatedNormalTarget) < 0 || dist > distThresh || norm(cross(*updatedNormalSrc, *updatedNormalTarget)) > angleThresh)
			// ||
			//(dist > distThresh / 50.0 &&
			//(norm(cross(srcToTargetVec, updatedNormalTarget)) > angleThresh &&
			//norm(cross(updatedNormalSrc, srcToTargetVec)) > angleThresh)))
	{
		//goto Invalid_Matching;
	}

	// compute residual geo
	float4 residualDiff = *updatedPosSrc - *updatedPosTarget;
	//printf("residual offset: %d\n", residualOffset);
	residual[3 * vertexPairInd] = -w_geo * residualDiff.x;
	residual[3 * vertexPairInd + 1] = -w_geo * residualDiff.y;
	residual[3 * vertexPairInd + 2] = -w_geo * residualDiff.z;
	//printf("%f %f %f\n", residual[residualOffset + 4 * vertexPairInd], residual[residualOffset + 4 * vertexPairInd + 1], residual[residualOffset + 4 * vertexPairInd + 2]);

	// compute jacobian geo and photo
	my_int4 fourNearIndSrc, fourNearIndTarget;
	my_float4 fourNearWeightSrc, fourNearWeightTarget;
	fourNearIndSrc = *((my_int4 *)(vertexToNodeIndices + vertexIndSrc * 4));
	fourNearWeightSrc = *((my_float4 *)(vertexToNodeWeights + vertexIndSrc * 4));
	fourNearIndTarget = *((my_int4 *)(vertexToNodeIndices + vertexIndTarget * 4));
	fourNearWeightTarget = *((my_float4 *)(vertexToNodeWeights + vertexIndTarget * 4));

	// To sort the 4-nearest node according its index, because CSR must be ordered in each row	
	// bubble sort 
	SortNearInd(fourNearIndSrc.data, fourNearWeightSrc.data);
	SortNearInd(fourNearIndTarget.data, fourNearWeightTarget.data);

#if 0
	printf("%d %d %d %d %d %d %d %d %d %d\n", vertexIndSrc, vertexIndTarget, fourNearIndTarget[0], fourNearIndTarget[1],
		fourNearIndTarget[2], fourNearIndTarget[3], fourNearIndSrc[0], fourNearIndSrc[1],
		fourNearIndSrc[2], fourNearIndSrc[3]);
#endif

	float4 nearPos;
	float4 vertexNodeSrc[4], vertexNodeTarget[4];

	nearPos = originVertexPoses[nodeVIndices[fourNearIndSrc.data[0]]];
	vertexNodeSrc[0] = posFragIndSrc - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndSrc.data[1]]];
	vertexNodeSrc[1] = posFragIndSrc - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndSrc.data[2]]];
	vertexNodeSrc[2] = posFragIndSrc - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndSrc.data[3]]];
	vertexNodeSrc[3] = posFragIndSrc - nearPos;

	nearPos = originVertexPoses[nodeVIndices[fourNearIndTarget.data[0]]];
	vertexNodeTarget[0] = posFragIndTarget - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndTarget.data[1]]];
	vertexNodeTarget[1] = posFragIndTarget - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndTarget.data[2]]];
	vertexNodeTarget[2] = posFragIndTarget - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndTarget.data[3]]];
	vertexNodeTarget[3] = posFragIndTarget - nearPos;

	// Note: targetFragInd < srcFragInd, so target is first	
	float wt, ws, wssx, wssy, wssz, wttx, wtty, wttz;
	int it12, is12;
	float *jacobianGeo_a_4n;
	int *jacobianGeo_ja_4n;
#pragma unroll 4
	for (int n = 0; n < 4; ++n)
	{
		wt = -w_geo * fourNearWeightTarget.data[n];
		ws = w_geo * fourNearWeightSrc.data[n];
		it12 = fourNearIndTarget.data[n] * 12;
		is12 = fourNearIndSrc.data[n] * 12;
		wttx = wt * vertexNodeTarget[n].x;
		wtty = wt * vertexNodeTarget[n].y;
		wttz = wt * vertexNodeTarget[n].z;
		wssx = ws * vertexNodeSrc[n].x;
		wssy = ws * vertexNodeSrc[n].y;
		wssz = ws * vertexNodeSrc[n].z;
		jacobianGeo_a_4n = jacobianGeo_a + 4 * n;
		jacobianGeo_ja_4n = jacobianGeo_ja + 4 * n;

		*(jacobianGeo_a_4n) = wttx;
		*(jacobianGeo_a_4n + 1) = wtty;
		*(jacobianGeo_a_4n + 2) = wttz;
		*(jacobianGeo_a_4n + 3) = wt;

		*(jacobianGeo_a_4n + 0 + 1 * 4 * 4) = wssx;
		*(jacobianGeo_a_4n + 1 + 1 * 4 * 4) = wssy;
		*(jacobianGeo_a_4n + 2 + 1 * 4 * 4) = wssz;
		*(jacobianGeo_a_4n + 3 + 1 * 4 * 4) = ws;

		*(jacobianGeo_a_4n + 0 + 2 * 4 * 4) = wttx;
		*(jacobianGeo_a_4n + 1 + 2 * 4 * 4) = wtty;
		*(jacobianGeo_a_4n + 2 + 2 * 4 * 4) = wttz;
		*(jacobianGeo_a_4n + 3 + 2 * 4 * 4) = wt;

		*(jacobianGeo_a_4n + 0 + 3 * 4 * 4) = wssx;
		*(jacobianGeo_a_4n + 1 + 3 * 4 * 4) = wssy;
		*(jacobianGeo_a_4n + 2 + 3 * 4 * 4) = wssz;
		*(jacobianGeo_a_4n + 3 + 3 * 4 * 4) = ws;

		*(jacobianGeo_a_4n + 0 + 4 * 4 * 4) = wttx;
		*(jacobianGeo_a_4n + 1 + 4 * 4 * 4) = wttx;
		*(jacobianGeo_a_4n + 2 + 4 * 4 * 4) = wttx;
		*(jacobianGeo_a_4n + 3 + 4 * 4 * 4) = wt;

		*(jacobianGeo_a_4n + 0 + 5 * 4 * 4) = wssx;
		*(jacobianGeo_a_4n + 1 + 5 * 4 * 4) = wssx;
		*(jacobianGeo_a_4n + 2 + 5 * 4 * 4) = wssx;
		*(jacobianGeo_a_4n + 3 + 5 * 4 * 4) = ws;

		*(jacobianGeo_ja_4n) = it12;
		*(jacobianGeo_ja_4n + 1) = it12 + 3;
		*(jacobianGeo_ja_4n + 2) = it12 + 6;
		*(jacobianGeo_ja_4n + 3) = it12 + 9;

		*(jacobianGeo_ja_4n + 1 * 4 * 4) = is12;
		*(jacobianGeo_ja_4n + 1 + 1 * 4 * 4) = is12 + 3;
		*(jacobianGeo_ja_4n + 2 + 1 * 4 * 4) = is12 + 6;
		*(jacobianGeo_ja_4n + 3 + 1 * 4 * 4) = is12 + 9;

		*(jacobianGeo_ja_4n + 2 * 4 * 4) = it12 + 1;
		*(jacobianGeo_ja_4n + 1 + 2 * 4 * 4) = it12 + 4;
		*(jacobianGeo_ja_4n + 2 + 2 * 4 * 4) = it12 + 7;
		*(jacobianGeo_ja_4n + 3 + 2 * 4 * 4) = it12 + 10;

		*(jacobianGeo_ja_4n + 3 * 4 * 4) = is12 + 1;
		*(jacobianGeo_ja_4n + 1 + 3 * 4 * 4) = is12 + 4;
		*(jacobianGeo_ja_4n + 2 + 3 * 4 * 4) = is12 + 7;
		*(jacobianGeo_ja_4n + 3 + 3 * 4 * 4) = is12 + 10;

		*(jacobianGeo_ja_4n + 4 * 4 * 4) = it12 + 2;
		*(jacobianGeo_ja_4n + 1 + 4 * 4 * 4) = it12 + 5;
		*(jacobianGeo_ja_4n + 2 + 4 * 4 * 4) = it12 + 8;
		*(jacobianGeo_ja_4n + 3 + 4 * 4 * 4) = it12 + 11;

		*(jacobianGeo_ja_4n + 5 * 4 * 4) = is12 + 2;
		*(jacobianGeo_ja_4n + 1 + 5 * 4 * 4) = is12 + 5;
		*(jacobianGeo_ja_4n + 2 + 5 * 4 * 4) = is12 + 8;
		*(jacobianGeo_ja_4n + 3 + 5 * 4 * 4) = is12 + 11;
	}
	return;
Invalid_Matching:
	residual[3 * vertexPairInd] = 0.0f;
	residual[3 * vertexPairInd + 1] = 0.0f;
	residual[3 * vertexPairInd + 2] = 0.0f;
#pragma unroll 4
	for (int n = 0; n < 4; ++n)
	{
		int n4 = n * 4;
#pragma unroll 4
		for (int eleInd = 0; eleInd < 4; ++eleInd)
		{
			*(jacobianGeo_a) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 1 * 4 * 4) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 2 * 4 * 4) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 3 * 4 * 4) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 4 * 4 * 4) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 5 * 4 * 4) = 0.0f;
			*(jacobianGeo_ja + n4 + eleInd) = n4 + eleInd;
			*(jacobianGeo_ja + n4 + eleInd + 1 * 4 * 4) = n4 + eleInd + 1 * 4 * 4;
			*(jacobianGeo_ja + n4 + eleInd + 2 * 4 * 4) = n4 + eleInd + 2 * 4 * 4;
			*(jacobianGeo_ja + n4 + eleInd + 3 * 4 * 4) = n4 + eleInd + 3 * 4 * 4;
			*(jacobianGeo_ja + n4 + eleInd + 4 * 4 * 4) = n4 + eleInd + 4 * 4 * 4;
			*(jacobianGeo_ja + n4 + eleInd + 5 * 4 * 4) = n4 + eleInd + 5 * 4 * 4;
		}
	}
}

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
	int iter)
{
	int vertexPairInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexPairInd >= matchingPointsNum)
	{
		return;
	}
	
	float *jacobianGeo_a = jacobian_a + vertexPairInd * (4 * 4 * 6);
	int *jacobianGeo_ja = jacobian_ja + vertexPairInd * (4 * 4 * 6);

	*(jacobian_ia + vertexPairInd) = iaOffset + vertexPairInd * (4 * 4 * 6);

	int vertexIndSrc, vertexIndTarget, fragIndSrc, fragIndTarget;
	float4 posFragIndSrc, posFragIndTarget;
	vertexIndSrc = *(matchingPoints + 2 * vertexPairInd);
	vertexIndTarget = *(matchingPoints + 2 * vertexPairInd + 1);
	if (vertexIndTarget == -1 || vertexIndSrc == -1)
	{
		goto Invalid_Matching;
	}
#if 0
	printf("%d vertexIndSrc %d vertexIndTarget %d", vertexPairInd, vertexIndSrc, vertexIndTarget);
#endif

	posFragIndSrc = originVertexPoses[vertexIndSrc];
	posFragIndTarget = originVertexPoses[vertexIndTarget];
	fragIndSrc = (int)posFragIndSrc.w;
	fragIndTarget = (int)posFragIndTarget.w;
	//printf("%d %d\n", vertexIndTarget, vertexIndSrc);	

	float4 updatedPosSrc, updatedPosTarget;
	float4 updatedNormalSrc, updatedNormalTarget;

	updatedPosSrc = updatedVertexPoses[vertexIndSrc];
	updatedNormalSrc = *(updatedVertexNormals + vertexIndSrc);
	updatedPosTarget = updatedVertexPoses[vertexIndTarget];
	updatedNormalTarget = *(updatedVertexNormals + vertexIndTarget);	

	updatedPosSrc.w = 1.0f;
	updatedPosTarget.w = 1.0f;

	float dist = norm(updatedPosSrc - updatedPosTarget);
	float4 srcToTargetVec = normalize(updatedPosSrc - updatedPosTarget);
	updatedNormalSrc = normalize(updatedNormalSrc);
	updatedNormalTarget = normalize(updatedNormalTarget);
	//printf("%f %f %f\n", updatedNormalSrc->x, updatedNormalSrc->y, updatedNormalSrc->z);
	float distThresh = 0.04f, angleThresh = sin(30.0f * 3.14159254f / 180.f);
	if (iter % 11 == 10)
	{
		distThresh /= 1.1f;
		angleThresh /= 1.1f;
	}
	if (dot(updatedNormalSrc, updatedNormalTarget) < 0 || dist > distThresh || norm(cross(updatedNormalSrc, updatedNormalTarget)) > angleThresh
		|| (dist > distThresh / 50.0 &&
		(norm(cross(srcToTargetVec, updatedNormalTarget)) > angleThresh &&
		norm(cross(updatedNormalSrc, srcToTargetVec)) > angleThresh)))
	{
		//goto Invalid_Matching;
	}

	// compute residual geo
	float4 residualDiff = updatedPosSrc - updatedPosTarget;
	//printf("residual offset: %d\n", residualOffset);
	residual[vertexPairInd] = -w_geo * (residualDiff.x * updatedNormalTarget.x
		+ residualDiff.y * updatedNormalTarget.y
		+ residualDiff.z * updatedNormalTarget.z);

#if 0
	printf("residual %f\n", residual);
#endif

	// compute jacobian geo and photo
	my_int4 fourNearIndSrc, fourNearIndTarget;
	my_float4 fourNearWeightSrc, fourNearWeightTarget;
	fourNearIndSrc = *((my_int4 *)(vertexToNodeIndices + vertexIndSrc * 4));
	fourNearWeightSrc = *((my_float4 *)(vertexToNodeWeights + vertexIndSrc * 4));
	fourNearIndTarget = *((my_int4 *)(vertexToNodeIndices + vertexIndTarget * 4));
	fourNearWeightTarget = *((my_float4 *)(vertexToNodeWeights + vertexIndTarget * 4));

#if 0
	printf("pos %d %d %d %d %d %d %d %d\n", 
		fourNearIndSrc.data[0], fourNearIndSrc.data[1], fourNearIndSrc.data[2], fourNearIndSrc.data[3],
		fourNearIndTarget.data[0], fourNearIndTarget.data[1], fourNearIndTarget.data[2], fourNearIndTarget.data[3]);
#endif

	// To sort the 4-nearest node according its index, because CSR must be ordered in each row	
	// bubble sort 
	SortNearInd(fourNearIndSrc.data, fourNearWeightSrc.data);
	SortNearInd(fourNearIndTarget.data, fourNearWeightTarget.data);

#if 0
	printf("%d %d %d %d %d %d %d %d %d %d\n", vertexIndSrc, vertexIndTarget, fourNearIndTarget[0], fourNearIndTarget[1],
		fourNearIndTarget[2], fourNearIndTarget[3], fourNearIndSrc[0], fourNearIndSrc[1],
		fourNearIndSrc[2], fourNearIndSrc[3]);
#endif

	float4 nearPos;
	float4 vertexNodeSrc[4], vertexNodeTarget[4];

	nearPos = originVertexPoses[nodeVIndices[fourNearIndSrc.data[0]]];
	vertexNodeSrc[0] = posFragIndSrc - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndSrc.data[1]]];
	vertexNodeSrc[1] = posFragIndSrc - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndSrc.data[2]]];
	vertexNodeSrc[2] = posFragIndSrc - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndSrc.data[3]]];
	vertexNodeSrc[3] = posFragIndSrc - nearPos;

	nearPos = originVertexPoses[nodeVIndices[fourNearIndTarget.data[0]]];
	vertexNodeTarget[0] = posFragIndTarget - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndTarget.data[1]]];
	vertexNodeTarget[1] = posFragIndTarget - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndTarget.data[2]]];
	vertexNodeTarget[2] = posFragIndTarget - nearPos;
	nearPos = originVertexPoses[nodeVIndices[fourNearIndTarget.data[3]]];
	vertexNodeTarget[3] = posFragIndTarget - nearPos;

	// Note: targetFragInd < srcFragInd, so target is first	
	float wt, ws, wssx, wssy, wssz, wttx, wtty, wttz;
	int it12, is12;
	float *jacobianGeo_a_12n;
	int *jacobianGeo_ja_12n;
#pragma unroll 4
	for (int n = 0; n < 4; ++n)
	{	
		wt = -w_geo * fourNearWeightTarget.data[n];
		ws = w_geo * fourNearWeightSrc.data[n];
		it12 = fourNearIndTarget.data[n] * 12;
		is12 = fourNearIndSrc.data[n] * 12;
		wttx = wt * vertexNodeTarget[n].x;
		wtty = wt * vertexNodeTarget[n].y;
		wttz = wt * vertexNodeTarget[n].z;
		wssx = ws * vertexNodeSrc[n].x;
		wssy = ws * vertexNodeSrc[n].y;
		wssz = ws * vertexNodeSrc[n].z;
		jacobianGeo_a_12n = jacobianGeo_a + 12 * n;
		jacobianGeo_ja_12n = jacobianGeo_ja + 12 * n;

#if 0
		printf("weight: %f %f %f %f %f %f %f %f %f\n",
			updatedNormalTarget.x,
			updatedNormalTarget.y,
			updatedNormalTarget.z,
			wttx,
			wtty,
			wttz,
			wssx,
			wssy,
			wssz);
#endif

		*(jacobianGeo_a_12n) = wttx * updatedNormalTarget.x;
		*(jacobianGeo_a_12n + 1) = wttx * updatedNormalTarget.y;
		*(jacobianGeo_a_12n + 2) = wttx * updatedNormalTarget.z;
		*(jacobianGeo_a_12n + 3) = wtty * updatedNormalTarget.x;
		*(jacobianGeo_a_12n + 4) = wtty * updatedNormalTarget.y;
		*(jacobianGeo_a_12n + 5) = wtty * updatedNormalTarget.z;
		*(jacobianGeo_a_12n + 6) = wttz * updatedNormalTarget.x;
		*(jacobianGeo_a_12n + 7) = wttz * updatedNormalTarget.y;
		*(jacobianGeo_a_12n + 8) = wttz * updatedNormalTarget.z;
		*(jacobianGeo_a_12n + 9) = wt * updatedNormalTarget.x;
		*(jacobianGeo_a_12n + 10) = wt * updatedNormalTarget.y;
		*(jacobianGeo_a_12n + 11) = wt * updatedNormalTarget.z;

		*(jacobianGeo_a_12n + 0 + 3 * 4 * 4) = wssx * updatedNormalTarget.x;
		*(jacobianGeo_a_12n + 1 + 3 * 4 * 4) = wssx * updatedNormalTarget.y;
		*(jacobianGeo_a_12n + 2 + 3 * 4 * 4) = wssx * updatedNormalTarget.z;
		*(jacobianGeo_a_12n + 3 + 3 * 4 * 4) = wssy * updatedNormalTarget.x;
		*(jacobianGeo_a_12n + 4 + 3 * 4 * 4) = wssy * updatedNormalTarget.y;
		*(jacobianGeo_a_12n + 5 + 3 * 4 * 4) = wssy * updatedNormalTarget.z;
		*(jacobianGeo_a_12n + 6 + 3 * 4 * 4) = wssz * updatedNormalTarget.x;
		*(jacobianGeo_a_12n + 7 + 3 * 4 * 4) = wssz * updatedNormalTarget.y;
		*(jacobianGeo_a_12n + 8 + 3 * 4 * 4) = wssz * updatedNormalTarget.z;
		*(jacobianGeo_a_12n + 9 + 3 * 4 * 4) = ws * updatedNormalTarget.x;
		*(jacobianGeo_a_12n + 10 + 3 * 4 * 4) = ws * updatedNormalTarget.y;
		*(jacobianGeo_a_12n + 11 + 3 * 4 * 4) = ws * updatedNormalTarget.z;

		*(jacobianGeo_ja_12n) = it12;
		*(jacobianGeo_ja_12n + 1) = it12 + 1;
		*(jacobianGeo_ja_12n + 2) = it12 + 2;
		*(jacobianGeo_ja_12n + 3) = it12 + 3;
		*(jacobianGeo_ja_12n + 4) = it12 + 4;
		*(jacobianGeo_ja_12n + 5) = it12 + 5;
		*(jacobianGeo_ja_12n + 6) = it12 + 6;
		*(jacobianGeo_ja_12n + 7) = it12 + 7;
		*(jacobianGeo_ja_12n + 8) = it12 + 8;
		*(jacobianGeo_ja_12n + 9) = it12 + 9;
		*(jacobianGeo_ja_12n + 10) = it12 + 10;
		*(jacobianGeo_ja_12n + 11) = it12 + 11;

		*(jacobianGeo_ja_12n + 0 + 3 * 4 * 4) = is12;
		*(jacobianGeo_ja_12n + 1 + 3 * 4 * 4) = is12 + 1;
		*(jacobianGeo_ja_12n + 2 + 3 * 4 * 4) = is12 + 2;
		*(jacobianGeo_ja_12n + 3 + 3 * 4 * 4) = is12 + 3;
		*(jacobianGeo_ja_12n + 4 + 3 * 4 * 4) = is12 + 4;
		*(jacobianGeo_ja_12n + 5 + 3 * 4 * 4) = is12 + 5;
		*(jacobianGeo_ja_12n + 6 + 3 * 4 * 4) = is12 + 6;
		*(jacobianGeo_ja_12n + 7 + 3 * 4 * 4) = is12 + 7;
		*(jacobianGeo_ja_12n + 8 + 3 * 4 * 4) = is12 + 8;
		*(jacobianGeo_ja_12n + 9 + 3 * 4 * 4) = is12 + 9;
		*(jacobianGeo_ja_12n + 10 + 3 * 4 * 4) = is12 + 10;
		*(jacobianGeo_ja_12n + 11 + 3 * 4 * 4) = is12 + 11;
	}
	return;
Invalid_Matching:
	residual[vertexPairInd] = 0.0f;
#pragma unroll 4
	for (int n = 0; n < 4; ++n)
	{
		int n4 = n * 4;
#pragma unroll 4
		for (int eleInd = 0; eleInd < 4; ++eleInd)
		{
			*(jacobianGeo_a) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 1 * 4 * 4) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 2 * 4 * 4) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 3 * 4 * 4) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 4 * 4 * 4) = 0.0f;
			*(jacobianGeo_a + n4 + eleInd + 5 * 4 * 4) = 0.0f;
			*(jacobianGeo_ja + n4 + eleInd) = n4 + eleInd;
			*(jacobianGeo_ja + n4 + eleInd + 1 * 4 * 4) = n4 + eleInd + 1 * 4 * 4;
			*(jacobianGeo_ja + n4 + eleInd + 2 * 4 * 4) = n4 + eleInd + 2 * 4 * 4;
			*(jacobianGeo_ja + n4 + eleInd + 3 * 4 * 4) = n4 + eleInd + 3 * 4 * 4;
			*(jacobianGeo_ja + n4 + eleInd + 4 * 4 * 4) = n4 + eleInd + 4 * 4 * 4;
			*(jacobianGeo_ja + n4 + eleInd + 5 * 4 * 4) = n4 + eleInd + 5 * 4 * 4;
		}
	}
}

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
	int iter)
{
	int vertexPairInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexPairInd >= matchingPointsNum)
	{
		return;
	}

	float *jacobianPhoto_a = jacobian_a + vertexPairInd * (12 * 4 * 2);
	int *jacobianPhoto_ja = jacobian_ja + vertexPairInd * (12 * 4 * 2);

	*(jacobian_ia + vertexPairInd) = iaOffset + vertexPairInd * (12 * 4 * 2);

	int vertexIndSrc, vertexIndTarget, fragIndSrc, fragIndTarget;
	float4 *posFragIndSrc, *posFragIndTarget;
	vertexIndSrc = *(matchingPoints + 2 * vertexPairInd);
	vertexIndTarget = *(matchingPoints + 2 * vertexPairInd + 1);
	if (vertexIndTarget == -1 || vertexIndSrc == -1)
	{
		goto Invalid_Matching;
	}

	posFragIndSrc = originVertexPoses + vertexIndSrc;
	posFragIndTarget = originVertexPoses + vertexIndTarget;
	fragIndSrc = (int)posFragIndSrc->w;
	fragIndTarget = (int)posFragIndTarget->w;

	//printf("%d %d\n", fragIndSrc, fragIndTarget);

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

	float4 *updatedPoseInvSrc = updatedPosesInv + fragIndSrc * 4;
	float4 *updatedPoseInvTarget = updatedPosesInv + fragIndTarget * 4;

	updatedPosSrc = updatedVertexPoses + vertexIndSrc;
	updatedNormalSrc = updatedVertexNormals + vertexIndSrc;
	updatedPosSrcLocalTargetSpace = updatedPosSrc->x * updatedPoseInvTarget[0] + updatedPosSrc->y * updatedPoseInvTarget[1] +
		updatedPosSrc->z * updatedPoseInvTarget[2] + updatedPoseInvTarget[3];

	updatedPosTarget = updatedVertexPoses + vertexIndTarget;
	updatedNormalTarget = updatedVertexNormals + vertexIndTarget;

	updatedPosTargetLocalSrcSpace = updatedPosTarget->x * updatedPoseInvSrc[0] + updatedPosTarget->y * updatedPoseInvSrc[1] +
		updatedPosTarget->z * updatedPoseInvSrc[2] + updatedPoseInvSrc[3];
#if 0
	//float4 middlePos = (*updatedPosSrc + *updatedPosTarget) * 0.5f;
	//middlePos.w = 1.0f;
	printf("middle pos: %f %f %f\n updated src pos: %f %f %f\n updated target pos: %f %f %f\n",
		middlePos.x, middlePos.y, middlePos.z,
		updatedPosSrc->x, updatedPosSrc->y, updatedPosSrc->z,
		updatedPosTarget->x, updatedPosTarget->y, updatedPosTarget->z);
#endif

	float dist = norm(*updatedPosSrc - *updatedPosTarget);
	float4 srcToTargetVec = normalize(*updatedPosSrc - *updatedPosTarget);
	*updatedNormalSrc = normalize(*updatedNormalSrc);
	*updatedNormalTarget = normalize(*updatedNormalTarget);

	float sine = norm(cross(*updatedNormalSrc, *updatedNormalTarget));
	float sine0 = norm(cross(srcToTargetVec, *updatedNormalTarget));
	float sine1 = norm(cross(*updatedNormalSrc, srcToTargetVec));
	//printf("%f %f %f\n", updatedNormalSrc->x, updatedNormalSrc->y, updatedNormalSrc->z);
	float distThresh = 0.02f, angleThresh = sin(30.0f * 3.14159254f / 180.f);
	if (iter % 11 == 10)
	{
		distThresh /= 1.1f;
		angleThresh /= 1.1f;
	}
	if (dot(*updatedNormalSrc, *updatedNormalTarget) < 0 || dist > distThresh || sine > angleThresh ||
		(dist > distThresh / 50.0 && (sine0 > angleThresh || sine1 > angleThresh)))
	{
		//goto Invalid_Matching;
	}

	//printf("%d %d\n", vertexIndSrc, vertexIndTarget);

#if 0
	printf("\npose src: %f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n\n",
		poseDeviceSrc[0].x, poseDeviceSrc[0].y, poseDeviceSrc[0].z,
		poseDeviceSrc[1].x, poseDeviceSrc[1].y, poseDeviceSrc[1].z,
		poseDeviceSrc[2].x, poseDeviceSrc[2].y, poseDeviceSrc[2].z,
		poseDeviceSrc[3].x, poseDeviceSrc[3].y, poseDeviceSrc[3].z);
	printf("\npose target: %f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n\n",
		poseDeviceTarget[0].x, poseDeviceTarget[0].y, poseDeviceTarget[0].z,
		poseDeviceTarget[1].x, poseDeviceTarget[1].y, poseDeviceTarget[1].z,
		poseDeviceTarget[2].x, poseDeviceTarget[2].y, poseDeviceTarget[2].z,
		poseDeviceTarget[3].x, poseDeviceTarget[3].y, poseDeviceTarget[3].z);
#endif

	float *keyGrayImgSrc = keyGrayImgs + fragIndSrc * keykeyGrayImgsStep;
	float *keyGrayImgTarget = keyGrayImgs + fragIndTarget * keykeyGrayImgsStep;
	float *keyGrayImgDxDeviceSrc = keyGrayImgsDx + fragIndSrc * keyGrayImgsDxStep;
	float *keyGrayImgDyDeviceSrc = keyGrayImgsDy + fragIndSrc * keyGrayImgsDyStep;
	float *keyGrayImgDxDeviceTarget = keyGrayImgsDx + fragIndTarget * keyGrayImgsDxStep;
	float *keyGrayImgDyDeviceTarget = keyGrayImgsDy + fragIndTarget * keyGrayImgsDyStep;

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
	uBi0TargetSrcSpace = __float2int_rd(uBiTargetSrcSpace); uBi1TargetSrcSpace = uBi0TargetSrcSpace + 1;
	vBi0TargetSrcSpace = __float2int_rd(vBiTargetSrcSpace); vBi1TargetSrcSpace = vBi0TargetSrcSpace + 1;
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
	//printf("uv: %f %f\n", uBiSrcTargetSpace, vBiSrcTargetSpace);
	// bilinear intarpolation
	uBi0SrcTargetSpace = __float2int_rd(uBiSrcTargetSpace); uBi1SrcTargetSpace = uBi0SrcTargetSpace + 1;
	vBi0SrcTargetSpace = __float2int_rd(vBiSrcTargetSpace); vBi1SrcTargetSpace = vBi0SrcTargetSpace + 1;
	coef = (uBi1SrcTargetSpace - uBiSrcTargetSpace) / (float)(uBi1SrcTargetSpace - uBi0SrcTargetSpace);
	valTop = coef * ((float)*(keyGrayImgTarget + vBi0SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgTarget + vBi0SrcTargetSpace * width + uBi1SrcTargetSpace));
	valBottom = coef * ((float)*(keyGrayImgTarget + vBi1SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgTarget + vBi1SrcTargetSpace * width + uBi1SrcTargetSpace));
	coef = (vBi1SrcTargetSpace - vBiSrcTargetSpace) / (float)(vBi1SrcTargetSpace - vBi0SrcTargetSpace);
	biPixSrcTargetSpace = coef * valTop + (1 - coef) * valBottom;

	//if (fabs(biPixSrcTargetSpace - biPixTargetSrcSpace) > 20.0f)
	{
		//goto Invalid_Matching;
	}

	//printf("diff: %f %f, ", biPixSrcTargetSpace, biPixTargetSrcSpace);
	residual[vertexPairInd] = -w_photo * (biPixSrcTargetSpace - biPixTargetSrcSpace);
	//printf("r: %f %f %f\n", biPixTarget, biPixTargetSrcSpace, residual[vertexPairInd]);

	// compute jacobian geo and photo
	int fourNearIndSrc[4], fourNearIndTarget[4];
	float fourNearWeightSrc[4], fourNearWeightTarget[4];
	int *nearNodePtr = vertexToNodeIndices + vertexIndSrc * 4;
	float *fourNearWeightPtr = vertexToNodeWeights + vertexIndSrc * 4;
	fourNearIndSrc[0] = *(nearNodePtr);
	fourNearIndSrc[1] = *(nearNodePtr + 1);
	fourNearIndSrc[2] = *(nearNodePtr + 2);
	fourNearIndSrc[3] = *(nearNodePtr + 3);
	fourNearWeightSrc[0] = *(fourNearWeightPtr);
	fourNearWeightSrc[1] = *(fourNearWeightPtr + 1);
	fourNearWeightSrc[2] = *(fourNearWeightPtr + 2);
	fourNearWeightSrc[3] = *(fourNearWeightPtr + 3);
	nearNodePtr = vertexToNodeIndices + vertexIndTarget * 4;
	fourNearWeightPtr = vertexToNodeWeights + vertexIndTarget * 4;
	fourNearIndTarget[0] = *(nearNodePtr);
	fourNearIndTarget[1] = *(nearNodePtr + 1);
	fourNearIndTarget[2] = *(nearNodePtr + 2);
	fourNearIndTarget[3] = *(nearNodePtr + 3);
	fourNearWeightTarget[0] = *(fourNearWeightPtr);
	fourNearWeightTarget[1] = *(fourNearWeightPtr + 1);
	fourNearWeightTarget[2] = *(fourNearWeightPtr + 2);
	fourNearWeightTarget[3] = *(fourNearWeightPtr + 3);

	// To sort the 4-nearest node according its index, because CSR must be ordered in each row	
	// bubble sort 
	SortNearInd(fourNearIndSrc, fourNearWeightSrc);
	SortNearInd(fourNearIndTarget, fourNearWeightTarget);

#if 0
	printf("%d %d %d %d %d %d %d %d %d %d\n", vertexIndSrc, vertexIndTarget, fourNearIndTarget[0], fourNearIndTarget[1],
		fourNearIndTarget[2], fourNearIndTarget[3], fourNearIndSrc[0], fourNearIndSrc[1],
		fourNearIndSrc[2], fourNearIndSrc[3]);
#endif

	float4 *nearPos;
	float4 vertexNodeSrc[4], vertexNodeTarget[4];
	for (int n = 0; n < 4; ++n)
	{
		nearPos = &originVertexPoses[nodeVIndices[fourNearIndSrc[n]]];
		vertexNodeSrc[n] = *posFragIndSrc - *nearPos;
	}
	for (int n = 0; n < 4; ++n)
	{
		nearPos = &originVertexPoses[nodeVIndices[fourNearIndTarget[n]]];
		vertexNodeTarget[n] = *posFragIndTarget - *nearPos;
	}

	// d_gamma_uv, d_gamm_xyz, d_gamma_Rt
	float3 J_gamma_xyz_src, J_gamma_xyz_target;
	float3 J_gamma_xyz_src_target_space, J_gamma_xyz_target_src_space;
	float3 J_gamma_xyz_global_src, J_gamma_xyz_global_target;
	float3 J_gamma_xyz_global_src_target_space, J_gamma_xyz_global_target_src_space;
	float2 J_gamma_uv_src, J_gamma_uv_src_target_space,
		J_gamma_uv_target, J_gamma_uv_target_src_space;
#if USE_BILINEAR_TO_CALC_GRAD
	coef = (uBi1TargetSrcSpace - uBi0TargetSrcSpace)*(vBi1TargetSrcSpace - vBi0TargetSrcSpace);
	J_gamma_uv_target_src_space =
		make_float2(-(vBi1TargetSrcSpace - vBiTargetSrcSpace) / coef, -(uBi1TargetSrcSpace - uBiTargetSrcSpace) / coef) *
		((float)*(keyGrayImgSrc + vBi0TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		make_float2((vBi1TargetSrcSpace - vBiTargetSrcSpace) / coef, -(uBiTargetSrcSpace - uBi0TargetSrcSpace) / coef) *
		((float)*(keyGrayImgSrc + vBi0TargetSrcSpace * width + uBi1TargetSrcSpace)) +
		make_float2(-(vBiTargetSrcSpace - vBi0TargetSrcSpace) / coef, (uBi1TargetSrcSpace - uBiTargetSrcSpace) / coef) *
		((float)*(keyGrayImgSrc + vBi1TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		make_float2((vBiTargetSrcSpace - vBi0TargetSrcSpace) / coef, (uBiTargetSrcSpace - uBi0TargetSrcSpace) / coef) *
		((float)*(keyGrayImgSrc + vBi1TargetSrcSpace * width + uBi1TargetSrcSpace));
#else
	float dx, dy;

	coef = (uBi1TargetSrcSpace - uBiTargetSrcSpace) / (float)(uBi1TargetSrcSpace - uBi0TargetSrcSpace);
	valTop = coef * ((float)*(keyGrayImgDxDeviceSrc + vBi0TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDxDeviceSrc + vBi0TargetSrcSpace * width + uBi1TargetSrcSpace));
	valBottom = coef * ((float)*(keyGrayImgDxDeviceSrc + vBi1TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDxDeviceSrc + vBi1TargetSrcSpace * width + uBi1TargetSrcSpace));
	coef = (vBi1TargetSrcSpace - vBiTargetSrcSpace) / (float)(vBi1TargetSrcSpace - vBi0TargetSrcSpace);
	dx = coef * valTop + (1 - coef) * valBottom;
	coef = (uBi1TargetSrcSpace - uBiTargetSrcSpace) / (float)(uBi1TargetSrcSpace - uBi0TargetSrcSpace);
	valTop = coef * ((float)*(keyGrayImgDyDeviceSrc + vBi0TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDyDeviceSrc + vBi0TargetSrcSpace * width + uBi1TargetSrcSpace));
	valBottom = coef * ((float)*(keyGrayImgDyDeviceSrc + vBi1TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDyDeviceSrc + vBi1TargetSrcSpace * width + uBi1TargetSrcSpace));
	coef = (vBi1TargetSrcSpace - vBiTargetSrcSpace) / (float)(vBi1TargetSrcSpace - vBi0TargetSrcSpace);
	dy = coef * valTop + (1 - coef) * valBottom;
	J_gamma_uv_target_src_space = make_float2(dx, dy);
#endif
	J_gamma_xyz_target_src_space = make_float3(J_gamma_uv_target_src_space.x * fx / updatedPosTargetLocalSrcSpace.z,
		J_gamma_uv_target_src_space.y * fy / updatedPosTargetLocalSrcSpace.z,
		(-J_gamma_uv_target_src_space.x * updatedPosTargetLocalSrcSpace.x * fx - J_gamma_uv_target_src_space.y * updatedPosTargetLocalSrcSpace.y * fy) / (updatedPosTargetLocalSrcSpace.z * updatedPosTargetLocalSrcSpace.z));
	J_gamma_xyz_global_target_src_space = J_gamma_xyz_target_src_space.x * make_float3(updatedPoseInvSrc[0].x, updatedPoseInvSrc[0].y, updatedPoseInvSrc[0].z) +
		J_gamma_xyz_target_src_space.y * make_float3(updatedPoseInvSrc[1].x, updatedPoseInvSrc[1].y, updatedPoseInvSrc[1].z) +
		J_gamma_xyz_target_src_space.z * make_float3(updatedPoseInvSrc[2].x, updatedPoseInvSrc[2].y, updatedPoseInvSrc[2].z);

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
		(-J_gamma_uv_src_target_space.x * updatedPosSrcLocalTargetSpace.x * fx - J_gamma_uv_src_target_space.y * updatedPosSrcLocalTargetSpace.y * fy) / (updatedPosSrcLocalTargetSpace.z * updatedPosSrcLocalTargetSpace.z));
	J_gamma_xyz_global_src_target_space = J_gamma_xyz_src_target_space.x * make_float3(updatedPoseInvTarget[0].x, updatedPoseInvTarget[0].y, updatedPoseInvTarget[0].z) +
		J_gamma_xyz_src_target_space.y * make_float3(updatedPoseInvTarget[1].x, updatedPoseInvTarget[1].y, updatedPoseInvTarget[1].z) +
		J_gamma_xyz_src_target_space.z * make_float3(updatedPoseInvTarget[2].x, updatedPoseInvTarget[2].y, updatedPoseInvTarget[2].z);

	/*
	nearNodeIndicesDevice[10 * vertexPairInd + 0] = vertexIndSrc;
	nearNodeIndicesDevice[10 * vertexPairInd + 1] = vertexIndTarget;
	nearNodeIndicesDevice[10 * vertexPairInd + 2] = fourNearIndSrc[0];
	nearNodeIndicesDevice[10 * vertexPairInd + 3] = fourNearIndSrc[1];
	nearNodeIndicesDevice[10 * vertexPairInd + 4] = fourNearIndSrc[2];
	nearNodeIndicesDevice[10 * vertexPairInd + 5] = fourNearIndSrc[3];
	nearNodeIndicesDevice[10 * vertexPairInd + 6] = fourNearIndTarget[0];
	nearNodeIndicesDevice[10 * vertexPairInd + 7] = fourNearIndTarget[1];
	nearNodeIndicesDevice[10 * vertexPairInd + 8] = fourNearIndTarget[2];
	nearNodeIndicesDevice[10 * vertexPairInd + 9] = fourNearIndTarget[3];
	*/
	// Note: targetFragInd < srcFragInd, so target is first	
	for (int n = 0; n < 4; ++n)
	{
		float jacoValTargetSrcSpacePhoto[12] = {
			J_gamma_xyz_global_target_src_space.x * vertexNodeTarget[n].x, J_gamma_xyz_global_target_src_space.y * vertexNodeTarget[n].x, J_gamma_xyz_global_target_src_space.z * vertexNodeTarget[n].x,
			J_gamma_xyz_global_target_src_space.x * vertexNodeTarget[n].y, J_gamma_xyz_global_target_src_space.y * vertexNodeTarget[n].y, J_gamma_xyz_global_target_src_space.z * vertexNodeTarget[n].y,
			J_gamma_xyz_global_target_src_space.x * vertexNodeTarget[n].z, J_gamma_xyz_global_target_src_space.y * vertexNodeTarget[n].z, J_gamma_xyz_global_target_src_space.z * vertexNodeTarget[n].z,
			J_gamma_xyz_global_target_src_space.x, J_gamma_xyz_global_target_src_space.y, J_gamma_xyz_global_target_src_space.z
		};
		float jacoValSrcTargetSpacePhoto[12] = {
			J_gamma_xyz_global_src_target_space.x * vertexNodeSrc[n].x, J_gamma_xyz_global_src_target_space.y * vertexNodeSrc[n].x, J_gamma_xyz_global_src_target_space.z * vertexNodeSrc[n].x,
			J_gamma_xyz_global_src_target_space.x * vertexNodeSrc[n].y, J_gamma_xyz_global_src_target_space.y * vertexNodeSrc[n].y, J_gamma_xyz_global_src_target_space.z * vertexNodeSrc[n].y,
			J_gamma_xyz_global_src_target_space.x * vertexNodeSrc[n].z, J_gamma_xyz_global_src_target_space.y * vertexNodeSrc[n].z, J_gamma_xyz_global_src_target_space.z * vertexNodeSrc[n].z,
			J_gamma_xyz_global_src_target_space.x, J_gamma_xyz_global_src_target_space.y, J_gamma_xyz_global_src_target_space.z
		};
		for (int eleInd = 0; eleInd < 12; ++eleInd)
		{
			*(jacobianPhoto_a + 12 * n + eleInd)
				= -w_photo * fourNearWeightTarget[n] * jacoValTargetSrcSpacePhoto[eleInd];
			*(jacobianPhoto_a + 12 * n + eleInd + 4 * 12)
				= w_photo * fourNearWeightSrc[n] * jacoValSrcTargetSpacePhoto[eleInd];

			*(jacobianPhoto_ja + 12 * n + eleInd) = fourNearIndTarget[n] * 12 + eleInd;
			*(jacobianPhoto_ja + 12 * n + eleInd + 4 * 12) = fourNearIndSrc[n] * 12 + eleInd;
		}
		//printf("%d %d %d %d %d\n", vertexIndTarget, fourNearIndTarget[0], fourNearIndTarget[1], fourNearIndTarget[2], fourNearIndTarget[3]);
		//printf("%d %d %d %d %d\n", vertexIndSrc, fourNearIndSrc[0], fourNearIndSrc[1], fourNearIndSrc[2], fourNearIndSrc[3]);
	}
	return;
Invalid_Matching:
	residual[vertexPairInd] = 0.0f;
	for (int n = 0; n < 4; ++n)
	{
		for (int eleInd = 0; eleInd < 12; ++eleInd)
		{
			*(jacobianPhoto_a + 12 * n + eleInd) = 0.0f;
			*(jacobianPhoto_a + 12 * n + eleInd + 4 * 12) = 0.0f;
			*(jacobianPhoto_ja + 12 * n + eleInd) = n * 12 + eleInd;
			*(jacobianPhoto_ja + 12 * n + eleInd + 4 * 12) = n * 12 + eleInd + 4 * 12;
		}
	}
#if 0
	int vertexPairInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexPairInd >= matchingPointsNum)
	{
		return;
	}

	float *jacobianPhoto_a = jacobian_a + vertexPairInd * (12 * 4 * 2);
	int *jacobianPhoto_ja = jacobian_ja + vertexPairInd * (12 * 4 * 2);

	*(jacobian_ia + vertexPairInd) = iaOffset + vertexPairInd * (12 * 4 * 2);

	int vertexIndSrc, vertexIndTarget, fragIndSrc, fragIndTarget;
	float4 *posFragIndSrc, *posFragIndTarget;
	vertexIndSrc = *(matchingPoints + 2 * vertexPairInd);
	vertexIndTarget = *(matchingPoints + 2 * vertexPairInd + 1);
	if (vertexIndTarget == -1 || vertexIndSrc == -1)
	{
		goto Invalid_Matching;
	}

	posFragIndSrc = originVertexPoses + vertexIndSrc;
	posFragIndTarget = originVertexPoses + vertexIndTarget;
	fragIndSrc = (int)posFragIndSrc->w;
	fragIndTarget = (int)posFragIndTarget->w;

	//printf("%d %d\n", fragIndSrc, fragIndTarget);

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

	float4 *updatedPoseInvSrc = updatedPosesInv + fragIndSrc * 4;
	float4 *updatedPoseInvTarget = updatedPosesInv + fragIndTarget * 4;

	updatedPosSrc = updatedVertexPoses + vertexIndSrc;
	updatedNormalSrc = updatedVertexNormals + vertexIndSrc;
	updatedPosSrcLocalTargetSpace = updatedPosSrc->x * updatedPoseInvTarget[0] + updatedPosSrc->y * updatedPoseInvTarget[1] +
		updatedPosSrc->z * updatedPoseInvTarget[2] + updatedPoseInvTarget[3];

	updatedPosTarget = updatedVertexPoses + vertexIndTarget;
	updatedNormalTarget = updatedVertexNormals + vertexIndTarget;

	updatedPosTargetLocalSrcSpace = updatedPosTarget->x * updatedPoseInvSrc[0] + updatedPosTarget->y * updatedPoseInvSrc[1] +
		updatedPosTarget->z * updatedPoseInvSrc[2] + updatedPoseInvSrc[3];
#if 0
	//float4 middlePos = (*updatedPosSrc + *updatedPosTarget) * 0.5f;
	//middlePos.w = 1.0f;
	printf("middle pos: %f %f %f\n updated src pos: %f %f %f\n updated target pos: %f %f %f\n",
		middlePos.x, middlePos.y, middlePos.z,
		updatedPosSrc->x, updatedPosSrc->y, updatedPosSrc->z,
		updatedPosTarget->x, updatedPosTarget->y, updatedPosTarget->z);
#endif

	float dist = norm(*updatedPosSrc - *updatedPosTarget);
	float4 srcToTargetVec = normalize(*updatedPosSrc - *updatedPosTarget);
	*updatedNormalSrc = normalize(*updatedNormalSrc);
	*updatedNormalTarget = normalize(*updatedNormalTarget);

	float sine = norm(cross(*updatedNormalSrc, *updatedNormalTarget));
	float sine0 = norm(cross(srcToTargetVec, *updatedNormalTarget));
	float sine1 = norm(cross(*updatedNormalSrc, srcToTargetVec));
	//printf("%f %f %f\n", updatedNormalSrc->x, updatedNormalSrc->y, updatedNormalSrc->z);
	float distThresh = 0.02f, angleThresh = sin(30.0f * 3.14159254f / 180.f);
	if (iter % 11 == 10)
	{
		distThresh /= 1.1f;
		angleThresh /= 1.1f;
	}
	if (dot(*updatedNormalSrc, *updatedNormalTarget) < 0 || dist > distThresh || sine > angleThresh ||
		(dist > distThresh / 50.0 && (sine0 > angleThresh || sine1 > angleThresh)))
	{
		//goto Invalid_Matching;
	}

	//printf("%d %d\n", vertexIndSrc, vertexIndTarget);

#if 0
	printf("\npose src: %f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n\n",
		poseDeviceSrc[0].x, poseDeviceSrc[0].y, poseDeviceSrc[0].z,
		poseDeviceSrc[1].x, poseDeviceSrc[1].y, poseDeviceSrc[1].z,
		poseDeviceSrc[2].x, poseDeviceSrc[2].y, poseDeviceSrc[2].z,
		poseDeviceSrc[3].x, poseDeviceSrc[3].y, poseDeviceSrc[3].z);
	printf("\npose target: %f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n\n",
		poseDeviceTarget[0].x, poseDeviceTarget[0].y, poseDeviceTarget[0].z,
		poseDeviceTarget[1].x, poseDeviceTarget[1].y, poseDeviceTarget[1].z,
		poseDeviceTarget[2].x, poseDeviceTarget[2].y, poseDeviceTarget[2].z,
		poseDeviceTarget[3].x, poseDeviceTarget[3].y, poseDeviceTarget[3].z);
#endif

	float *keyGrayImgSrc = keyGrayImgs + fragIndSrc * keykeyGrayImgsStep;
	float *keyGrayImgTarget = keyGrayImgs + fragIndTarget * keykeyGrayImgsStep;
	float *keyGrayImgDxDeviceSrc = keyGrayImgsDx + fragIndSrc * keyGrayImgsDxStep;
	float *keyGrayImgDyDeviceSrc = keyGrayImgsDy + fragIndSrc * keyGrayImgsDyStep;
	float *keyGrayImgDxDeviceTarget = keyGrayImgsDx + fragIndTarget * keyGrayImgsDxStep;
	float *keyGrayImgDyDeviceTarget = keyGrayImgsDy + fragIndTarget * keyGrayImgsDyStep;

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
	uBi0TargetSrcSpace = __float2int_rd(uBiTargetSrcSpace); uBi1TargetSrcSpace = uBi0TargetSrcSpace + 1;
	vBi0TargetSrcSpace = __float2int_rd(vBiTargetSrcSpace); vBi1TargetSrcSpace = vBi0TargetSrcSpace + 1;
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
	//printf("uv: %f %f\n", uBiSrcTargetSpace, vBiSrcTargetSpace);
	// bilinear intarpolation
	uBi0SrcTargetSpace = __float2int_rd(uBiSrcTargetSpace); uBi1SrcTargetSpace = uBi0SrcTargetSpace + 1;
	vBi0SrcTargetSpace = __float2int_rd(vBiSrcTargetSpace); vBi1SrcTargetSpace = vBi0SrcTargetSpace + 1;
	coef = (uBi1SrcTargetSpace - uBiSrcTargetSpace) / (float)(uBi1SrcTargetSpace - uBi0SrcTargetSpace);
	valTop = coef * ((float)*(keyGrayImgTarget + vBi0SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgTarget + vBi0SrcTargetSpace * width + uBi1SrcTargetSpace));
	valBottom = coef * ((float)*(keyGrayImgTarget + vBi1SrcTargetSpace * width + uBi0SrcTargetSpace)) +
		(1 - coef) * ((float)*(keyGrayImgTarget + vBi1SrcTargetSpace * width + uBi1SrcTargetSpace));
	coef = (vBi1SrcTargetSpace - vBiSrcTargetSpace) / (float)(vBi1SrcTargetSpace - vBi0SrcTargetSpace);
	biPixSrcTargetSpace = coef * valTop + (1 - coef) * valBottom;

	//if (fabs(biPixSrcTargetSpace - biPixTargetSrcSpace) > 20.0f)
	{
		//goto Invalid_Matching;
	}

	//printf("diff: %f %f, ", biPixSrcTargetSpace, biPixTargetSrcSpace);
	residual[vertexPairInd] = -w_photo * (biPixSrcTargetSpace - biPixTargetSrcSpace);
	//printf("r: %f %f %f\n", biPixTarget, biPixTargetSrcSpace, residual[vertexPairInd]);

	// compute jacobian geo and photo
	int fourNearIndSrc[4], fourNearIndTarget[4];
	float fourNearWeightSrc[4], fourNearWeightTarget[4];
	int *nearNodePtr = vertexToNodeIndices + vertexIndSrc * 4;
	float *fourNearWeightPtr = vertexToNodeWeights + vertexIndSrc * 4;
	fourNearIndSrc[0] = *(nearNodePtr);
	fourNearIndSrc[1] = *(nearNodePtr + 1);
	fourNearIndSrc[2] = *(nearNodePtr + 2);
	fourNearIndSrc[3] = *(nearNodePtr + 3);
	fourNearWeightSrc[0] = *(fourNearWeightPtr);
	fourNearWeightSrc[1] = *(fourNearWeightPtr + 1);
	fourNearWeightSrc[2] = *(fourNearWeightPtr + 2);
	fourNearWeightSrc[3] = *(fourNearWeightPtr + 3);
	nearNodePtr = vertexToNodeIndices + vertexIndTarget * 4;
	fourNearWeightPtr = vertexToNodeWeights + vertexIndTarget * 4;
	fourNearIndTarget[0] = *(nearNodePtr);
	fourNearIndTarget[1] = *(nearNodePtr + 1);
	fourNearIndTarget[2] = *(nearNodePtr + 2);
	fourNearIndTarget[3] = *(nearNodePtr + 3);
	fourNearWeightTarget[0] = *(fourNearWeightPtr);
	fourNearWeightTarget[1] = *(fourNearWeightPtr + 1);
	fourNearWeightTarget[2] = *(fourNearWeightPtr + 2);
	fourNearWeightTarget[3] = *(fourNearWeightPtr + 3);

	// To sort the 4-nearest node according its index, because CSR must be ordered in each row	
	// bubble sort 
	SortNearInd(fourNearIndSrc, fourNearWeightSrc);
	SortNearInd(fourNearIndTarget, fourNearWeightTarget);

#if 0
	printf("%d %d %d %d %d %d %d %d %d %d\n", vertexIndSrc, vertexIndTarget, fourNearIndTarget[0], fourNearIndTarget[1],
		fourNearIndTarget[2], fourNearIndTarget[3], fourNearIndSrc[0], fourNearIndSrc[1],
		fourNearIndSrc[2], fourNearIndSrc[3]);
#endif

	float4 *nearPos;
	float4 vertexNodeSrc[4], vertexNodeTarget[4];
	for (int n = 0; n < 4; ++n)
	{
		nearPos = &originVertexPoses[nodeVIndices[fourNearIndSrc[n]]];
		vertexNodeSrc[n] = *posFragIndSrc - *nearPos;
	}
	for (int n = 0; n < 4; ++n)
	{
		nearPos = &originVertexPoses[nodeVIndices[fourNearIndTarget[n]]];
		vertexNodeTarget[n] = *posFragIndTarget - *nearPos;
	}

	// d_gamma_uv, d_gamm_xyz, d_gamma_Rt
	float3 J_gamma_xyz_src, J_gamma_xyz_target;
	float3 J_gamma_xyz_src_target_space, J_gamma_xyz_target_src_space;
	float3 J_gamma_xyz_global_src, J_gamma_xyz_global_target;
	float3 J_gamma_xyz_global_src_target_space, J_gamma_xyz_global_target_src_space;
	float2 J_gamma_uv_src, J_gamma_uv_src_target_space,
		J_gamma_uv_target, J_gamma_uv_target_src_space;
#if USE_BILINEAR_TO_CALC_GRAD
	coef = (uBi1TargetSrcSpace - uBi0TargetSrcSpace)*(vBi1TargetSrcSpace - vBi0TargetSrcSpace);
	J_gamma_uv_target_src_space =
		make_float2(-(vBi1TargetSrcSpace - vBiTargetSrcSpace) / coef, -(uBi1TargetSrcSpace - uBiTargetSrcSpace) / coef) *
		((float)*(keyGrayImgSrc + vBi0TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		make_float2((vBi1TargetSrcSpace - vBiTargetSrcSpace) / coef, -(uBiTargetSrcSpace - uBi0TargetSrcSpace) / coef) *
		((float)*(keyGrayImgSrc + vBi0TargetSrcSpace * width + uBi1TargetSrcSpace)) +
		make_float2(-(vBiTargetSrcSpace - vBi0TargetSrcSpace) / coef, (uBi1TargetSrcSpace - uBiTargetSrcSpace) / coef) *
		((float)*(keyGrayImgSrc + vBi1TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		make_float2((vBiTargetSrcSpace - vBi0TargetSrcSpace) / coef, (uBiTargetSrcSpace - uBi0TargetSrcSpace) / coef) *
		((float)*(keyGrayImgSrc + vBi1TargetSrcSpace * width + uBi1TargetSrcSpace));
#else
	float dx, dy;

	coef = (uBi1TargetSrcSpace - uBiTargetSrcSpace) / (float)(uBi1TargetSrcSpace - uBi0TargetSrcSpace);
	valTop = coef * ((float)*(keyGrayImgDxDeviceSrc + vBi0TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDxDeviceSrc + vBi0TargetSrcSpace * width + uBi1TargetSrcSpace));
	valBottom = coef * ((float)*(keyGrayImgDxDeviceSrc + vBi1TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDxDeviceSrc + vBi1TargetSrcSpace * width + uBi1TargetSrcSpace));
	coef = (vBi1TargetSrcSpace - vBiTargetSrcSpace) / (float)(vBi1TargetSrcSpace - vBi0TargetSrcSpace);
	dx = coef * valTop + (1 - coef) * valBottom;
	coef = (uBi1TargetSrcSpace - uBiTargetSrcSpace) / (float)(uBi1TargetSrcSpace - uBi0TargetSrcSpace);
	valTop = coef * ((float)*(keyGrayImgDyDeviceSrc + vBi0TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDyDeviceSrc + vBi0TargetSrcSpace * width + uBi1TargetSrcSpace));
	valBottom = coef * ((float)*(keyGrayImgDyDeviceSrc + vBi1TargetSrcSpace * width + uBi0TargetSrcSpace)) +
		(1 - coef) * ((float)*(keyGrayImgDyDeviceSrc + vBi1TargetSrcSpace * width + uBi1TargetSrcSpace));
	coef = (vBi1TargetSrcSpace - vBiTargetSrcSpace) / (float)(vBi1TargetSrcSpace - vBi0TargetSrcSpace);
	dy = coef * valTop + (1 - coef) * valBottom;
	J_gamma_uv_target_src_space = make_float2(dx, dy);
#endif
	J_gamma_xyz_target_src_space = make_float3(J_gamma_uv_target_src_space.x * fx / updatedPosTargetLocalSrcSpace.z,
		J_gamma_uv_target_src_space.y * fy / updatedPosTargetLocalSrcSpace.z,
		(-J_gamma_uv_target_src_space.x * updatedPosTargetLocalSrcSpace.x * fx - J_gamma_uv_target_src_space.y * updatedPosTargetLocalSrcSpace.y * fy) / (updatedPosTargetLocalSrcSpace.z * updatedPosTargetLocalSrcSpace.z));
	J_gamma_xyz_global_target_src_space = J_gamma_xyz_target_src_space.x * make_float3(updatedPoseInvSrc[0].x, updatedPoseInvSrc[0].y, updatedPoseInvSrc[0].z) +
		J_gamma_xyz_target_src_space.y * make_float3(updatedPoseInvSrc[1].x, updatedPoseInvSrc[1].y, updatedPoseInvSrc[1].z) +
		J_gamma_xyz_target_src_space.z * make_float3(updatedPoseInvSrc[2].x, updatedPoseInvSrc[2].y, updatedPoseInvSrc[2].z);

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
		(-J_gamma_uv_src_target_space.x * updatedPosSrcLocalTargetSpace.x * fx - J_gamma_uv_src_target_space.y * updatedPosSrcLocalTargetSpace.y * fy) / (updatedPosSrcLocalTargetSpace.z * updatedPosSrcLocalTargetSpace.z));
	J_gamma_xyz_global_src_target_space = J_gamma_xyz_src_target_space.x * make_float3(updatedPoseInvTarget[0].x, updatedPoseInvTarget[0].y, updatedPoseInvTarget[0].z) +
		J_gamma_xyz_src_target_space.y * make_float3(updatedPoseInvTarget[1].x, updatedPoseInvTarget[1].y, updatedPoseInvTarget[1].z) +
		J_gamma_xyz_src_target_space.z * make_float3(updatedPoseInvTarget[2].x, updatedPoseInvTarget[2].y, updatedPoseInvTarget[2].z);

	/*
	nearNodeIndicesDevice[10 * vertexPairInd + 0] = vertexIndSrc;
	nearNodeIndicesDevice[10 * vertexPairInd + 1] = vertexIndTarget;
	nearNodeIndicesDevice[10 * vertexPairInd + 2] = fourNearIndSrc[0];
	nearNodeIndicesDevice[10 * vertexPairInd + 3] = fourNearIndSrc[1];
	nearNodeIndicesDevice[10 * vertexPairInd + 4] = fourNearIndSrc[2];
	nearNodeIndicesDevice[10 * vertexPairInd + 5] = fourNearIndSrc[3];
	nearNodeIndicesDevice[10 * vertexPairInd + 6] = fourNearIndTarget[0];
	nearNodeIndicesDevice[10 * vertexPairInd + 7] = fourNearIndTarget[1];
	nearNodeIndicesDevice[10 * vertexPairInd + 8] = fourNearIndTarget[2];
	nearNodeIndicesDevice[10 * vertexPairInd + 9] = fourNearIndTarget[3];
	*/
	// Note: targetFragInd < srcFragInd, so target is first	
	for (int n = 0; n < 4; ++n)
	{
		float jacoValTargetSrcSpacePhoto[12] = {
			J_gamma_xyz_global_target_src_space.x * vertexNodeTarget[n].x, J_gamma_xyz_global_target_src_space.y * vertexNodeTarget[n].x, J_gamma_xyz_global_target_src_space.z * vertexNodeTarget[n].x,
			J_gamma_xyz_global_target_src_space.x * vertexNodeTarget[n].y, J_gamma_xyz_global_target_src_space.y * vertexNodeTarget[n].y, J_gamma_xyz_global_target_src_space.z * vertexNodeTarget[n].y,
			J_gamma_xyz_global_target_src_space.x * vertexNodeTarget[n].z, J_gamma_xyz_global_target_src_space.y * vertexNodeTarget[n].z, J_gamma_xyz_global_target_src_space.z * vertexNodeTarget[n].z,
			J_gamma_xyz_global_target_src_space.x, J_gamma_xyz_global_target_src_space.y, J_gamma_xyz_global_target_src_space.z
		};
		float jacoValSrcTargetSpacePhoto[12] = {
			J_gamma_xyz_global_src_target_space.x * vertexNodeSrc[n].x, J_gamma_xyz_global_src_target_space.y * vertexNodeSrc[n].x, J_gamma_xyz_global_src_target_space.z * vertexNodeSrc[n].x,
			J_gamma_xyz_global_src_target_space.x * vertexNodeSrc[n].y, J_gamma_xyz_global_src_target_space.y * vertexNodeSrc[n].y, J_gamma_xyz_global_src_target_space.z * vertexNodeSrc[n].y,
			J_gamma_xyz_global_src_target_space.x * vertexNodeSrc[n].z, J_gamma_xyz_global_src_target_space.y * vertexNodeSrc[n].z, J_gamma_xyz_global_src_target_space.z * vertexNodeSrc[n].z,
			J_gamma_xyz_global_src_target_space.x, J_gamma_xyz_global_src_target_space.y, J_gamma_xyz_global_src_target_space.z
		};
		for (int eleInd = 0; eleInd < 12; ++eleInd)
		{
			*(jacobianPhoto_a + 12 * n + eleInd)
				= -w_photo * fourNearWeightTarget[n] * jacoValTargetSrcSpacePhoto[eleInd];
			*(jacobianPhoto_a + 12 * n + eleInd + 4 * 12)
				= w_photo * fourNearWeightSrc[n] * jacoValSrcTargetSpacePhoto[eleInd];

			*(jacobianPhoto_ja + 12 * n + eleInd) = fourNearIndTarget[n] * 12 + eleInd;
			*(jacobianPhoto_ja + 12 * n + eleInd + 4 * 12) = fourNearIndSrc[n] * 12 + eleInd;
		}
		//printf("%d %d %d %d %d\n", vertexIndTarget, fourNearIndTarget[0], fourNearIndTarget[1], fourNearIndTarget[2], fourNearIndTarget[3]);
		//printf("%d %d %d %d %d\n", vertexIndSrc, fourNearIndSrc[0], fourNearIndSrc[1], fourNearIndSrc[2], fourNearIndSrc[3]);
	}
	return;
Invalid_Matching:
	residual[vertexPairInd] = 0.0f;
	for (int n = 0; n < 4; ++n)
	{
		for (int eleInd = 0; eleInd < 12; ++eleInd)
		{
			*(jacobianPhoto_a + 12 * n + eleInd) = 0.0f;
			*(jacobianPhoto_a + 12 * n + eleInd + 4 * 12) = 0.0f;
			*(jacobianPhoto_ja + 12 * n + eleInd) = n * 12 + eleInd;
			*(jacobianPhoto_ja + 12 * n + eleInd + 4 * 12) = n * 12 + eleInd + 4 * 12;
		}
	}
#endif
}

__global__ void ComputeRegTermsKernel(
	int *jacobian_ja, float *jacobian_a, float *residual,
	int nodeNum,
	float3 *Rts,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	int *nodeVIndices,
	int *nodeToNodeIndices,
	float *nodeToNodeWeights,
	float w_reg, float w_rot, float w_trans)
{
	int nodeInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (nodeInd >= nodeNum)
		return;

	float *jacobianReg_a = jacobian_a + 8 * 3 * 5 * nodeInd;
	int *jacobianReg_ja = jacobian_ja + 8 * 3 * 5 * nodeInd;
	float *residualReg = residual + 8 * 3 * nodeInd;

	float *jacobianRot_a = jacobian_a + 8 * 3 * 5 * nodeNum + (6 * 3 + 3 * 3) * nodeInd;
	int *jacobianRot_ja = jacobian_ja + 8 * 3 * 5 * nodeNum + (6 * 3 + 3 * 3) * nodeInd;
	float *residualRot = residual + 8 * 3 * nodeNum + 6 * nodeInd;

#if 0
	float *jacobianTrans_a = jacobian_a + 8 * 3 * 5 * nodeNum + (6 * 3 + 3 * 3) * nodeNum + 3 * nodeInd;
	int *jacobianTrans_ja = jacobian_ja + 8 * 3 * 5 * nodeNum + (6 * 3 + 3 * 3) * nodeNum + 3 * nodeInd;
	float *residualTrans = residual + 8 * 3 * nodeNum + 6 * nodeNum + 3 * nodeInd;
#endif

	//printf("%d %d\n", vertexInd, (int *)nodeposConf);
	float3 *R = Rts + nodeInd * 4;
	float3 *ts = Rts + 3, *t = ts + nodeInd * 4;
	int *eightNearInd = nodeToNodeIndices + nodeInd * 8;
	float *eightNearWeight = nodeToNodeWeights + nodeInd * 8;
	int nearInd;

	/*
	printf("%d %d %d %d %d %d %d %d %d\n", nodeInd,
	eightNearInd[0], eightNearInd[1], eightNearInd[2], eightNearInd[3],
	eightNearInd[4], eightNearInd[5], eightNearInd[6], eightNearInd[7]);
	*/

	int fragInd = originVertexPoses[nodeVIndices[nodeInd]].w;
	float3 nodePos = make_float3(originVertexPoses[nodeVIndices[nodeInd]]), nearNodePos;		
	float3 updatedNodePos = make_float3(updatedVertexPoses[nodeVIndices[nodeInd]]);		

	// compute residual reg and jacobian reg	
	float diffx, diffy, diffz, weight;
	float3 resi;
	int stepJaco, stepResi;
#if 0
	float deformValue = 0.0f;
#endif
	for (int n = 0; n < 8; ++n)
	{
		stepResi = 3 * n;
		nearInd = *(eightNearInd + n);
		weight = *(eightNearWeight + n);
		nearNodePos = make_float3(originVertexPoses[nodeVIndices[nearInd]]);
		diffx = nearNodePos.x - nodePos.x;
		diffy = nearNodePos.y - nodePos.y;
		diffz = nearNodePos.z - nodePos.z;
		//printf("%d %d %f %f %f %f %f\n", nodeInd, nearInd, diffx, diffy, diffz, w_reg, weight);
		//printf("%d %d\n", vertexInd, (int *)nodeposConf);
		//printf("%d %d %f %f %f %d %d\n", nodeInd, nearInd, nodeposConf->x, nodeposConf->y, nodeposConf->z,
		//(int *)nearNodeposConf, (int *)nodeposConf);
		
		//printf("%d %d %f %f %f %f %f %f\n", nodeInd, nearInd,
			//nearNodePos.x, nearNodePos.y, nearNodePos.z,
			//nodePos.x, nodePos.y, nodePos.z);
		resi = weight * ((diffx * R[0] + diffy * R[1] + diffz * R[2]) +
			(nodePos - nearNodePos) +
			(*(ts + nodeInd * 4) - *(ts + nearInd * 4)));
#if 0
		resi = weight * (diffx * R[0] + diffy * R[1] + diffz * R[2] +
			(nodePos + *(ts + nodeInd * 4)) -
			(nearNodePos + *(ts + nearInd * 4)));
#endif

		residualReg[stepResi] = -w_reg * resi.x;
		residualReg[stepResi + 1] = -w_reg * resi.y;
		residualReg[stepResi + 2] = -w_reg * resi.z;
		//printf("1 %f %f %f\n", residualReg[stepResi], residualReg[stepResi + 1], residualReg[stepResi + 2]);	
		//printf("-- %f %f %f %f %f %f %f\n", resi.x, resi.y, resi.z, weight * w_reg, (*(ts + nearInd * 4)).x, (*(ts + nearInd * 4)).y, (*(ts + nearInd * 4)).z);
#if 0
		deformValue += (residualReg[stepResi] * residualReg[stepResi] +
			residualReg[stepResi + 1] * residualReg[stepResi + 1] +
			residualReg[stepResi + 2] * residualReg[stepResi + 2]);
#endif

		stepJaco = 15 * n;
		if (nearInd == nodeInd)
		{
			jacobianReg_a[stepJaco] = 0.0f;
			jacobianReg_ja[stepJaco] = nodeInd * 12;
			jacobianReg_a[1 + stepJaco] = 0.0f;
			jacobianReg_ja[1 + stepJaco] = nodeInd * 12 + 3;
			jacobianReg_a[2 + stepJaco] = 0.0f;
			jacobianReg_ja[2 + stepJaco] = nodeInd * 12 + 6;
			jacobianReg_a[3 + stepJaco] = 0.0f;
			jacobianReg_ja[3 + stepJaco] = nodeInd * 12 + 9 - 1;
			jacobianReg_a[4 + stepJaco] = 0.0f;
			jacobianReg_ja[4 + stepJaco] = nodeInd * 12 + 9;

			jacobianReg_a[5 + stepJaco] = 0.0f;
			jacobianReg_ja[5 + stepJaco] = nodeInd * 12 + 1;
			jacobianReg_a[6 + stepJaco] = 0.0f;
			jacobianReg_ja[6 + stepJaco] = nodeInd * 12 + 4;
			jacobianReg_a[7 + stepJaco] = 0.0f;
			jacobianReg_ja[7 + stepJaco] = nodeInd * 12 + 7;
			jacobianReg_a[8 + stepJaco] = 0.0f;
			jacobianReg_ja[8 + stepJaco] = nodeInd * 12 + 10 - 1;
			jacobianReg_a[9 + stepJaco] = 0.0f;
			jacobianReg_ja[9 + stepJaco] = nodeInd * 12 + 10;

			jacobianReg_a[10 + stepJaco] = 0.0f;
			jacobianReg_ja[10 + stepJaco] = nodeInd * 12 + 2;
			jacobianReg_a[11 + stepJaco] = 0.0f;
			jacobianReg_ja[11 + stepJaco] = nodeInd * 12 + 5;
			jacobianReg_a[12 + stepJaco] = 0.0f;
			jacobianReg_ja[12 + stepJaco] = nodeInd * 12 + 8;
			jacobianReg_a[13 + stepJaco] = 0.0f;
			jacobianReg_ja[13 + stepJaco] = nodeInd * 12 + 11 - 1;
			jacobianReg_a[14 + stepJaco] = 0.0f;
			jacobianReg_ja[14 + stepJaco] = nodeInd * 12 + 11;
		}
		else if (nearInd < nodeInd)
		{
			jacobianReg_a[stepJaco] = -1;
			jacobianReg_ja[stepJaco] = nearInd * 12 + 9;
			jacobianReg_a[1 + stepJaco] = diffx;
			jacobianReg_ja[1 + stepJaco] = nodeInd * 12;
			jacobianReg_a[2 + stepJaco] = diffy;
			jacobianReg_ja[2 + stepJaco] = nodeInd * 12 + 3;
			jacobianReg_a[3 + stepJaco] = diffz;
			jacobianReg_ja[3 + stepJaco] = nodeInd * 12 + 6;
			jacobianReg_a[4 + stepJaco] = 1;
			jacobianReg_ja[4 + stepJaco] = nodeInd * 12 + 9;

			jacobianReg_a[5 + stepJaco] = -1;
			jacobianReg_ja[5 + stepJaco] = nearInd * 12 + 10;
			jacobianReg_a[6 + stepJaco] = diffx;
			jacobianReg_ja[6 + stepJaco] = nodeInd * 12 + 1;
			jacobianReg_a[7 + stepJaco] = diffy;
			jacobianReg_ja[7 + stepJaco] = nodeInd * 12 + 4;
			jacobianReg_a[8 + stepJaco] = diffz;
			jacobianReg_ja[8 + stepJaco] = nodeInd * 12 + 7;
			jacobianReg_a[9 + stepJaco] = 1;
			jacobianReg_ja[9 + stepJaco] = nodeInd * 12 + 10;

			jacobianReg_a[10 + stepJaco] = -1;
			jacobianReg_ja[10 + stepJaco] = nearInd * 12 + 11;
			jacobianReg_a[11 + stepJaco] = diffx;
			jacobianReg_ja[11 + stepJaco] = nodeInd * 12 + 2;
			jacobianReg_a[12 + stepJaco] = diffy;
			jacobianReg_ja[12 + stepJaco] = nodeInd * 12 + 5;
			jacobianReg_a[13 + stepJaco] = diffz;
			jacobianReg_ja[13 + stepJaco] = nodeInd * 12 + 8;
			jacobianReg_a[14 + stepJaco] = 1;
			jacobianReg_ja[14 + stepJaco] = nodeInd * 12 + 11;
		}
		else
		{
			jacobianReg_a[stepJaco] = diffx;
			jacobianReg_ja[stepJaco] = nodeInd * 12;
			jacobianReg_a[1 + stepJaco] = diffy;
			jacobianReg_ja[1 + stepJaco] = nodeInd * 12 + 3;
			jacobianReg_a[2 + stepJaco] = diffz;
			jacobianReg_ja[2 + stepJaco] = nodeInd * 12 + 6;
			jacobianReg_a[3 + stepJaco] = 1;
			jacobianReg_ja[3 + stepJaco] = nodeInd * 12 + 9;
			jacobianReg_a[4 + stepJaco] = -1;
			jacobianReg_ja[4 + stepJaco] = nearInd * 12 + 9;

			jacobianReg_a[5 + stepJaco] = diffx;
			jacobianReg_ja[5 + stepJaco] = nodeInd * 12 + 1;
			jacobianReg_a[6 + stepJaco] = diffy;
			jacobianReg_ja[6 + stepJaco] = nodeInd * 12 + 4;
			jacobianReg_a[7 + stepJaco] = diffz;
			jacobianReg_ja[7 + stepJaco] = nodeInd * 12 + 7;
			jacobianReg_a[8 + stepJaco] = 1;
			jacobianReg_ja[8 + stepJaco] = nodeInd * 12 + 10;
			jacobianReg_a[9 + stepJaco] = -1;
			jacobianReg_ja[9 + stepJaco] = nearInd * 12 + 10;

			jacobianReg_a[10 + stepJaco] = diffx;
			jacobianReg_ja[10 + stepJaco] = nodeInd * 12 + 2;
			jacobianReg_a[11 + stepJaco] = diffy;
			jacobianReg_ja[11 + stepJaco] = nodeInd * 12 + 5;
			jacobianReg_a[12 + stepJaco] = diffz;
			jacobianReg_ja[12 + stepJaco] = nodeInd * 12 + 8;
			jacobianReg_a[13 + stepJaco] = 1;
			jacobianReg_ja[13 + stepJaco] = nodeInd * 12 + 11;
			jacobianReg_a[14 + stepJaco] = -1;
			jacobianReg_ja[14 + stepJaco] = nearInd * 12 + 11;
		}
		weight *= w_reg;
		for (int i = 0; i < 15; ++i)
		{
			jacobianReg_a[i + stepJaco] *= weight;
#if 0
			printf("reg: %f\n", jacobianReg_a[i + stepJaco]);
#endif
		}
	}

	//printf("%f\n", deformValue);
#if 0
	if (deformValue > 10)
	{
		printf("%f %d\n", deformValue, nodeInd / NODE_NUM);
	}
#endif	

	// compute residual rot and jacobian rot
	residualRot[0] = -w_rot * dot(R[0], R[1]);
	residualRot[1] = -w_rot * dot(R[0], R[2]);
	residualRot[2] = -w_rot * dot(R[1], R[2]);
	residualRot[3] = -w_rot * (dot(R[0], R[0]) - 1);
	residualRot[4] = -w_rot * (dot(R[1], R[1]) - 1);
	residualRot[5] = -w_rot * (dot(R[2], R[2]) - 1);
	//printf("2 %f %f %f\n", residualRot[0], residualRot[2], residualRot[5]);	

	jacobianRot_a[0] = w_rot * R[1].x;
	jacobianRot_ja[0] = nodeInd * 12;
	jacobianRot_a[1] = w_rot * R[1].y;
	jacobianRot_ja[1] = nodeInd * 12 + 1;
	jacobianRot_a[2] = w_rot * R[1].z;
	jacobianRot_ja[2] = nodeInd * 12 + 2;
	jacobianRot_a[3] = w_rot * R[0].x;
	jacobianRot_ja[3] = nodeInd * 12 + 3;
	jacobianRot_a[4] = w_rot * R[0].y;
	jacobianRot_ja[4] = nodeInd * 12 + 4;
	jacobianRot_a[5] = w_rot * R[0].z;
	jacobianRot_ja[5] = nodeInd * 12 + 5;

	jacobianRot_a[6] = w_rot * R[2].x;
	jacobianRot_ja[6] = nodeInd * 12;
	jacobianRot_a[7] = w_rot * R[2].y;
	jacobianRot_ja[7] = nodeInd * 12 + 1;
	jacobianRot_a[8] = w_rot * R[2].z;
	jacobianRot_ja[8] = nodeInd * 12 + 2;
	jacobianRot_a[9] = w_rot * R[0].x;
	jacobianRot_ja[9] = nodeInd * 12 + 6;
	jacobianRot_a[10] = w_rot * R[0].y;
	jacobianRot_ja[10] = nodeInd * 12 + 7;
	jacobianRot_a[11] = w_rot * R[0].z;
	jacobianRot_ja[11] = nodeInd * 12 + 8;

	jacobianRot_a[12] = w_rot * R[2].x;
	jacobianRot_ja[12] = nodeInd * 12 + 3;
	jacobianRot_a[13] = w_rot * R[2].y;
	jacobianRot_ja[13] = nodeInd * 12 + 4;
	jacobianRot_a[14] = w_rot * R[2].z;
	jacobianRot_ja[14] = nodeInd * 12 + 5;
	jacobianRot_a[15] = w_rot * R[1].x;
	jacobianRot_ja[15] = nodeInd * 12 + 6;
	jacobianRot_a[16] = w_rot * R[1].y;
	jacobianRot_ja[16] = nodeInd * 12 + 7;
	jacobianRot_a[17] = w_rot * R[1].z;
	jacobianRot_ja[17] = nodeInd * 12 + 8;

	jacobianRot_a[18] = w_rot * 2 * R[0].x;
	jacobianRot_ja[18] = nodeInd * 12;
	jacobianRot_a[19] = w_rot * 2 * R[0].y;
	jacobianRot_ja[19] = nodeInd * 12 + 1;
	jacobianRot_a[20] = w_rot * 2 * R[0].z;
	jacobianRot_ja[20] = nodeInd * 12 + 2;

	jacobianRot_a[21] = w_rot * 2 * R[1].x;
	jacobianRot_ja[21] = nodeInd * 12 + 3;
	jacobianRot_a[22] = w_rot * 2 * R[1].y;
	jacobianRot_ja[22] = nodeInd * 12 + 4;
	jacobianRot_a[23] = w_rot * 2 * R[1].z;
	jacobianRot_ja[23] = nodeInd * 12 + 5;

	jacobianRot_a[24] = w_rot * 2 * R[2].x;
	jacobianRot_ja[24] = nodeInd * 12 + 6;
	jacobianRot_a[25] = w_rot * 2 * R[2].y;
	jacobianRot_ja[25] = nodeInd * 12 + 7;
	jacobianRot_a[26] = w_rot * 2 * R[2].z;
	jacobianRot_ja[26] = nodeInd * 12 + 8;

#if 0
	for (int i = 0; i < 6; ++i)
	{
		printf("rot: %f\n", residualRot[i]);
	}
#endif	

#if 0
	residualTrans[0] = w_trans * (nodePos.x + t->x - updatedNodePos.x);
	residualTrans[1] = w_trans * (nodePos.y + t->y - updatedNodePos.y);
	residualTrans[2] = w_trans * (nodePos.z + t->z - updatedNodePos.z);
	
	jacobianTrans_a[0] = w_trans;
	jacobianTrans_ja[0] = nodeInd * 12 + 9;
	jacobianTrans_a[1] = w_trans;
	jacobianTrans_ja[1] = nodeInd * 12 + 10;
	jacobianTrans_a[2] = w_trans;
	jacobianTrans_ja[2] = nodeInd * 12 + 11;
#endif
}

void UpdateCameraPoses(
	float4 *updatedKeyPosesDevice,
	float4 *updatedKeyPosesInvDevice,
	float3 *Rts,
	float *cameraToNodeWeightsDevice,
	float4 *originVertexPosesDevice,
	int *nodeVIndicesDevice,
	float4 *keyPosesDevice,
	int nodeNum)
{
	int block = 256;
	int grid = DivUp(nodeNum, block);
	UpdateCameraNodeWeightKernel << <grid, block >> > (
		cameraToNodeWeightsDevice,
		originVertexPosesDevice,
		nodeVIndicesDevice,
		keyPosesDevice,
		nodeNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = NODE_NUM_EACH_FRAG;
	grid = DivUp(nodeNum, NODE_NUM_EACH_FRAG);
	//std::cout << "block: " << block << std::endl;
	//std::cout << "grid: " << grid << std::endl;
	UpdateCameraPosesKernel << <grid, block,NODE_NUM_EACH_FRAG*sizeof(float) >> > (updatedKeyPosesDevice,
		updatedKeyPosesInvDevice,
		Rts,
		cameraToNodeWeightsDevice,
		originVertexPosesDevice,
		nodeVIndicesDevice,
		keyPosesDevice,
		nodeNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void UpdateCameraNodeWeightKernel(float *cameraToNodeWeights,
	float4 *originVertexPoses,
	int *nodeVIndices,
	float4 *keyPoses,
	int nodeNum)
{
	int nodeInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (nodeInd >= nodeNum)
		return;

	float4 vertexPosFragInd = originVertexPoses[nodeVIndices[nodeInd]];
	int fragInd = (int)vertexPosFragInd.w;

	float4 cameraPose = keyPoses[fragInd * 4 + 3];

	float xDiff = vertexPosFragInd.x - cameraPose.x;
	float yDiff = vertexPosFragInd.y - cameraPose.y;
	float zDiff = vertexPosFragInd.z - cameraPose.z;

	cameraToNodeWeights[nodeInd] =  sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff); //1.0f;
}

__device__ __forceinline__ float reduce64(volatile float* sharedBuf, int tid)
{
	float val = sharedBuf[tid];
	if (tid < 32)
	{
		if (NODE_NUM_EACH_FRAG >= 64) { sharedBuf[tid] = val = val + sharedBuf[tid + 32]; }
		if (NODE_NUM_EACH_FRAG >= 32) { sharedBuf[tid] = val = val + sharedBuf[tid + 16]; }
		if (NODE_NUM_EACH_FRAG >= 16) { sharedBuf[tid] = val = val + sharedBuf[tid + 8]; }
		if (NODE_NUM_EACH_FRAG >= 8) { sharedBuf[tid] = val = val + sharedBuf[tid + 4]; }
		if (NODE_NUM_EACH_FRAG >= 4) { sharedBuf[tid] = val = val + sharedBuf[tid + 2]; }
		if (NODE_NUM_EACH_FRAG >= 2) { sharedBuf[tid] = val = val + sharedBuf[tid + 1]; }
	}
	__syncthreads();
	return sharedBuf[0];
}

extern __shared__ float exSharedBuf[];
__global__ void UpdateCameraPosesKernel(float4 *updatedKeyPoses,
	float4 *updatedKeyPosesInv,
	float3 *Rts,
	float *cameraToNodeWeight,
	float4 *originVertexPoses,
	int *nodeVIndices,
	float4 *keyPoses,
	int nodeNum)
{
	int tid = threadIdx.x;
	int nodeInd = threadIdx.x + blockIdx.x * blockDim.x;
	//volatile __shared__ float sharedBuf[NODE_NUM_EACH_FRAG];
	volatile float* sharedBuf = (float*)exSharedBuf;

	if (nodeInd >= nodeNum)
		return;

	float4 vertexPosFragInd = originVertexPoses[nodeVIndices[nodeInd]];
	int fragInd = (int)vertexPosFragInd.w;
	float3 nodePosition = make_float3(vertexPosFragInd);

	float3 *Rt = Rts + nodeInd * 4;
	float3 nodeRt[4] = { Rt[0], Rt[1], Rt[2], Rt[3] };

	float4 *cameraPos = keyPoses + fragInd * 4 + 3;
	float4 *cameraPose = keyPoses + fragInd * 4;

	float weight;

	// Make the weight Gaussian
	weight = cameraToNodeWeight[nodeInd];
	sharedBuf[tid] = weight;
	float mean = reduce64(sharedBuf, tid) / NODE_NUM_EACH_FRAG;
	//printf("mean: %f\n", mean);

	float varianceInv = 1.0 / (0.5 * mean * mean); // variance of gaussian		
	weight = exp(-weight* weight * varianceInv);
	sharedBuf[tid] = weight;
	float sum = reduce64(sharedBuf, tid);
	weight = weight / sum;
	//printf("weight: %f\n", weight);	

	float3 newPos;
	float3 newRot[3];

	float3 weightedPos = weight *
		(nodeRt[0] * (cameraPos->x - vertexPosFragInd.x) +
			nodeRt[1] * (cameraPos->y - vertexPosFragInd.y) +
			nodeRt[2] * (cameraPos->z - vertexPosFragInd.z) +
			nodePosition + nodeRt[3]);

	sharedBuf[tid] = weightedPos.x;
	newPos.x = reduce64(sharedBuf, tid);
	sharedBuf[tid] = weightedPos.y;
	newPos.y = reduce64(sharedBuf, tid);
	sharedBuf[tid] = weightedPos.z;
	newPos.z = reduce64(sharedBuf, tid);

	sharedBuf[tid] = weight * nodeRt[0].x;
	newRot[0].x = reduce64(sharedBuf, tid);
	sharedBuf[tid] = weight * nodeRt[0].y;
	newRot[0].y = reduce64(sharedBuf, tid);
	sharedBuf[tid] = weight * nodeRt[0].z;
	newRot[0].z = reduce64(sharedBuf, tid);

	sharedBuf[tid] = weight * nodeRt[1].x;
	newRot[1].x = reduce64(sharedBuf, tid);
	sharedBuf[tid] = weight * nodeRt[1].y;
	newRot[1].y = reduce64(sharedBuf, tid);
	sharedBuf[tid] = weight * nodeRt[1].z;
	newRot[1].z = reduce64(sharedBuf, tid);

	sharedBuf[tid] = weight * nodeRt[2].x;
	newRot[2].x = reduce64(sharedBuf, tid);
	sharedBuf[tid] = weight * nodeRt[2].y;
	newRot[2].y = reduce64(sharedBuf, tid);
	sharedBuf[tid] = weight * nodeRt[2].z;
	newRot[2].z = reduce64(sharedBuf, tid);
	/*
	printf("%d nodeRt: %f %f %f\n %f %f %f\n %f %f %f\n", tid,
	newRot[0].x, newRot[0].y, newRot[0].z,
	newRot[1].x, newRot[1].y, newRot[1].z,
	newRot[2].x, newRot[2].y, newRot[2].z);
	*/

	if (tid == 0)
	{
		float4 *updatedKeyPose = updatedKeyPoses + fragInd * 4;
		float4 *updatedKeyPoseInv = updatedKeyPosesInv + fragInd * 4;

		updatedKeyPose[0].x = newRot[0].x * cameraPose[0].x + newRot[1].x * cameraPose[0].y + newRot[2].x * cameraPose[0].z;
		updatedKeyPose[1].x = newRot[0].x * cameraPose[1].x + newRot[1].x * cameraPose[1].y + newRot[2].x * cameraPose[1].z;
		updatedKeyPose[2].x = newRot[0].x * cameraPose[2].x + newRot[1].x * cameraPose[2].y + newRot[2].x * cameraPose[2].z;
		updatedKeyPose[0].y = newRot[0].y * cameraPose[0].x + newRot[1].y * cameraPose[0].y + newRot[2].y * cameraPose[0].z;
		updatedKeyPose[1].y = newRot[0].y * cameraPose[1].x + newRot[1].y * cameraPose[1].y + newRot[2].y * cameraPose[1].z;
		updatedKeyPose[2].y = newRot[0].y * cameraPose[2].x + newRot[1].y * cameraPose[2].y + newRot[2].y * cameraPose[2].z;
		updatedKeyPose[0].z = newRot[0].z * cameraPose[0].x + newRot[1].z * cameraPose[0].y + newRot[2].z * cameraPose[0].z;
		updatedKeyPose[1].z = newRot[0].z * cameraPose[1].x + newRot[1].z * cameraPose[1].y + newRot[2].z * cameraPose[1].z;
		updatedKeyPose[2].z = newRot[0].z * cameraPose[2].x + newRot[1].z * cameraPose[2].y + newRot[2].z * cameraPose[2].z;

		updatedKeyPose[0].w = 0.0f;
		updatedKeyPose[1].w = 0.0f;
		updatedKeyPose[2].w = 0.0f;

		updatedKeyPose[3].x = newPos.x;
		updatedKeyPose[3].y = newPos.y;
		updatedKeyPose[3].z = newPos.z;
		updatedKeyPose[3].w = 1.0f;
	}
}

struct xFloat3
{
	union
	{
		struct { float x, y, z; };
		float data[3];
	};
};

struct xFloat4
{
	union
	{
		struct { float x, y, z, w; };
		float data[4];
	};
};

__device__ __forceinline__ void MatToQuaternion(float3 *matFloat3, float4 *quaternionFloat4)
{
	xFloat3 *mat = (xFloat3 *)matFloat3;
	xFloat4 *quaternion = (xFloat4 *)quaternionFloat4;

	// This algorithm comes from  "Quaternion Calculus and Fast Animation",
	// Ken Shoemake, 1987 SIGGRAPH course notes
	int trace = mat[0].x + mat[1].y + mat[2].z;
	if (trace > 0.0f)
	{
		trace = sqrt(trace + 1.0f);
		quaternion->w = 0.5f * trace;
		trace = 0.5f / trace;
		quaternion->x = (mat[1].z - mat[2].y) * trace;
		quaternion->y = (mat[2].x - mat[0].z) * trace;
		quaternion->z = (mat[0].y - mat[1].x) * trace;
	}
	else
	{
		int i, j, k;
		i = 0;
		if (mat[1].y > mat[0].x)
			i = 1;
		if (mat[2].z > mat[i].data[i])
			i = 2;
		j = (i + 1) % 3;
		k = (j + 1) % 3;

		trace = sqrt(mat[i].data[i] - mat[j].data[j] - mat[k].data[k] + 1.0f);
		quaternion->data[i] = 0.5f * trace;
		trace = 0.5f / trace;
		quaternion->w = (mat[k].data[j] - mat[j].data[k]) * trace;
		quaternion->data[j] = (mat[j].data[i] + mat[i].data[j]) * trace;
		quaternion->data[k] = (mat[k].data[i] + mat[i].data[k]) * trace;
	}
}

__device__ __forceinline__ void QuaternionToMat(float3 *mat, float4 *quaternion)
{
	float qx = quaternion->x, qy = quaternion->y, qz = quaternion->z, qw = quaternion->w;

	mat[0].x = 1 - 2 * qy * qy - 2 * qz * qz;
	mat[1].x = 2 * qx*qy - 2 * qz*qw;
	mat[2].x = 2 * qx*qz + 2 * qy*qw;
	mat[0].y = 2 * qx*qy + 2 * qz*qw;
	mat[1].y = 1 - 2 * qx * qx - 2 * qz * qz;
	mat[2].y = 2 * qy*qz - 2 * qx*qw;
	mat[0].z = 2 * qx*qz - 2 * qy*qw;
	mat[1].z = 2 * qy*qz + 2 * qx*qw;
	mat[2].z = 1 - 2 * qx * qx - 2 * qy * qy;
}

__global__ void BlurKernel(cv::cuda::PtrStepSz<float> bHor,
	cv::cuda::PtrStepSz<float> bVer,
	cv::cuda::PtrStepSz<uchar> grayImg)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < 4 || y < 4 || x >= grayImg.cols - 4 || y >= grayImg.rows - 4)
		return;
	
	float scale = 1.0f / 9.0f;
	bVer.ptr(y)[x] = grayImg.ptr(y)[x - 4] * scale
		+ grayImg.ptr(y)[x - 3] * scale
		+ grayImg.ptr(y)[x - 2] * scale
		+ grayImg.ptr(y)[x - 1] * scale
		+ grayImg.ptr(y)[x] * scale
		+ grayImg.ptr(y)[x + 1] * scale
		+ grayImg.ptr(y)[x + 2] * scale
		+ grayImg.ptr(y)[x + 3] * scale
		+ grayImg.ptr(y)[x + 4] * scale;
	bHor.ptr(y)[x] = grayImg.ptr(y - 4)[x] * scale
		+ grayImg.ptr(y - 3)[x] * scale
		+ grayImg.ptr(y - 2)[x] * scale
		+ grayImg.ptr(y - 1)[x] * scale
		+ grayImg.ptr(y)[x] * scale
		+ grayImg.ptr(y + 1)[x] * scale
		+ grayImg.ptr(y + 2)[x] * scale
		+ grayImg.ptr(y + 3)[x] * scale
		+ grayImg.ptr(y + 4)[x] * scale;
}

__global__ void CalculateBlurScoreKernel(float4 *sumVal,
	cv::cuda::PtrStepSz<float> bHor,
	cv::cuda::PtrStepSz<float> bVer,
	cv::cuda::PtrStepSz<uchar> grayImg)
{
	__shared__ float dFVer[256], dFHor[256], dVVer[256], dVHor[256];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int threadInd = threadIdx.y * blockDim.x + threadIdx.x,
		blockInd = blockIdx.y * gridDim.x + blockIdx.x;

	dFVer[threadInd] = 0.0f;
	dFHor[threadInd] = 0.0f;
	dVVer[threadInd] = 0.0f;
	dVHor[threadInd] = 0.0f;

	if (x < 4 || y < 4 || x >= bHor.cols - 4 || y >= bHor.rows - 4)
		return;

	float fVer = abs((float)grayImg.ptr(y)[x] - (float)grayImg.ptr(y - 1)[x]);
	float fHor = abs((float)grayImg.ptr(y)[x] - (float)grayImg.ptr(y)[x - 1]);

	//printf("%d\n", gridDim.x);

	dFVer[threadInd] = fVer;
	dFHor[threadInd] = fHor;
	dVVer[threadInd] = fmaxf(0, fVer
		- abs(bVer.ptr(y)[x] - bVer.ptr(y - 1)[x]));
	dVHor[threadInd] = fmaxf(0, fHor
		- abs(bHor.ptr(y)[x] - bHor.ptr(y)[x - 1]));

	__syncthreads();

	if (threadInd == 0)
	{
		float sumVal0 = 0.0f, sumVal1 = 0.0f, sumVal2 = 0.0f, sumVal3 = 0.0f;
		for (int i = 0; i < 256; ++i)
		{
			sumVal0 += dFVer[i];
			sumVal1 += dFHor[i];
			sumVal2 += dVVer[i];
			sumVal3 += dVHor[i];	
			//printf("%f, %f, %f, %f\n", dFVer[i], dFHor[i], dVVer[i], dVHor[i]);
		}
		sumVal[blockInd].x = sumVal0;
		sumVal[blockInd].y = sumVal1;
		sumVal[blockInd].z = sumVal2;
		sumVal[blockInd].w = sumVal3;	
	}
}

float CalculateBlurScoreGPU(const cv::cuda::GpuMat &grayImgDevice,
	cv::cuda::GpuMat &bHorDevice,
	cv::cuda::GpuMat &bVerDevice)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(grayImgDevice.cols, block.x), getGridDim(grayImgDevice.rows, block.y));

	BlurKernel << <grid, block >> > (bHorDevice, bVerDevice, grayImgDevice);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	float4 *sumValDevice;
	checkCudaErrors(cudaMalloc(&sumValDevice, sizeof(float4) * grid.x * grid.y));
	checkCudaErrors(cudaMemset(sumValDevice, 0, sizeof(float4) * grid.x * grid.y));

	CalculateBlurScoreKernel << <grid, block >> > (sumValDevice, bHorDevice, bVerDevice, grayImgDevice);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	std::vector<float4> sumValVec(grid.x * grid.y);
	checkCudaErrors(cudaMemcpy(
		sumValVec.data(),
		sumValDevice, sizeof(float4) * sumValVec.size(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(sumValDevice));

	float4 sumVal = { 0.0f, 0.0f, 0.0f, 0.0f };
	int cnt = 0;
	for (int i = 0; i < sumValVec.size(); ++i)
	{
		sumVal += sumValVec[i];
		//printf("%f %f %f %f\n", sumValVec[i].x, sumValVec[i].y, sumValVec[i].z, sumValVec[i].w);
	}

	return MAX(1 - sumVal.z / sumVal.x, 1 - sumVal.w / sumVal.y);
}
