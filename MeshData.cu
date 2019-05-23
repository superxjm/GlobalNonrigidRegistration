#include "MeshData.h"

#include <fstream>
#include <thrust/extrema.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "KNearestPoint.h"
#include "Helpers/xUtils.h"

__global__ void SampleNodeKernel(float4* node, int nodeNum, float4* vertex, int vertexNum,
                                 int step, int startIdx, int baseIdx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nodeNum)
	{
		int sampledIdx = int(startIdx + idx * step);
		sampledIdx = sampledIdx - sampledIdx / vertexNum * vertexNum; // sampledIdx % vertexNum;
		node[idx] = vertex[sampledIdx + baseIdx];
	}
}

__global__ void SampleVertexIdxKernel(int* sampledVertexIdx, int sampledVertexNum, int vertexNum,
                                      int step, int startIdx, int baseIdx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < sampledVertexNum)
	{
		int sampledIdx = int(startIdx + idx * step);
		sampledIdx = sampledIdx - sampledIdx / vertexNum * vertexNum; // sampledIdx % vertexNum;
		sampledVertexIdx[idx] = sampledIdx + baseIdx;
	}
}

__global__ void ComputeV2NDistKernel(float* weight, int* indices,
                                     float4* srcPoints, int srcVertexNum,
                                     float4* targetPoints, int K)
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

__global__ void AddRelaIdxBaseKernel(int* indices, int baseIdx, int num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= num)
		return;

	indices[idx] = indices[idx] + baseIdx;
}

__global__ void DistToWeightV2NKernel(float* vertexToNodeDistDevice, float varianceInv, int vertexNum)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= vertexNum)
		return;

	float nearDist0, nearDist1, nearDist2, nearDist3;
	float* distPtr;
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
}

__global__ void DistToWeightN2NKernel(float* nodeToNodeDist, int nodeNum)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= nodeNum)
		return;

	float* distPtr;
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

MeshData::MeshData()
	: m_vertexNum(0),
	  m_nodeNum(0),
	  m_maxVertexNum(MAX_VERTEX_NUM),
	  m_maxNodeNum(MAX_FRAG_NUM * NODE_NUM_EACH_FRAG)
{
	// Reserve
	m_dVertexVec.reserve(m_maxVertexNum);
	m_dNormalVec.reserve(m_maxVertexNum);

	// Resize
	m_dVertexVec.resize(m_maxVertexNum);
	m_dNormalVec.resize(m_maxVertexNum);
}

MeshData::~MeshData()
{
	m_dVertexVec.clear();
	m_dNormalVec.clear();
}

DeformableMeshData::DeformableMeshData()
	: MeshData(),
	  m_sigma(0)
{
	// Reserve
	m_dVertexVec.reserve(m_maxVertexNum);
	m_dNormalVec.reserve(m_maxVertexNum);
	m_dNodeVec.reserve(m_maxNodeNum);
	m_dVertexRelaIdxVec.reserve(m_maxVertexNum * MAX_NEAR_NODE_NUM_VERTEX);
	m_dVertexRelaWeightVec.reserve(m_maxVertexNum * MAX_NEAR_NODE_NUM_VERTEX);
	m_dNodeRelaIdxVec.reserve(m_maxNodeNum * MAX_NEAR_NODE_NUM_NODE);
	m_dNodeRelaWeightVec.reserve(m_maxNodeNum * MAX_NEAR_NODE_NUM_NODE);

	// Resize
	m_dVertexVec.resize(m_maxVertexNum);
	m_dNormalVec.resize(m_maxVertexNum);
	m_dNodeVec.resize(m_maxNodeNum);
	m_dVertexRelaIdxVec.resize(m_maxVertexNum * MAX_NEAR_NODE_NUM_VERTEX);
	m_dVertexRelaWeightVec.resize(m_maxVertexNum * MAX_NEAR_NODE_NUM_VERTEX);
	m_dNodeRelaIdxVec.resize(m_maxNodeNum * MAX_NEAR_NODE_NUM_NODE);
	m_dNodeRelaWeightVec.resize(m_maxNodeNum * MAX_NEAR_NODE_NUM_NODE);

	m_flannKnn = new NearestPoint;
}

DeformableMeshData::~DeformableMeshData()
{
	m_dVertexVec.clear();
	m_dNormalVec.clear();
	m_dNodeVec.clear();
	m_dVertexRelaIdxVec.clear();
	m_dVertexRelaWeightVec.clear();
	m_dNodeRelaIdxVec.clear();
	m_dNodeRelaWeightVec.clear();

	delete m_flannKnn;
}

FragDeformableMeshData::FragDeformableMeshData()
	: DeformableMeshData(),
	  m_fragNum(0),
	  m_sampledVertexNum(0),
	  m_maxFragNum(MAX_FRAG_NUM),
	  m_maxSampledVertexNum(SAMPLED_VERTEX_NUM_EACH_FRAG * MAX_FRAG_NUM)
{
	// Reserve
	m_vertexStrideVec.reserve(m_maxFragNum);
	m_dSampledVertexIdxVec.reserve(m_maxSampledVertexNum);;
	m_dMatchingPointsIdxVec.reserve(2 * MAX_CLOSURE_NUM_EACH_FRAG * SAMPLED_VERTEX_NUM_EACH_FRAG * MAX_FRAG_NUM);
	m_dNewSampledVertexIdxVec.reserve(SAMPLED_VERTEX_NUM_EACH_FRAG);
	m_dNewNodeVec.reserve(NODE_NUM_EACH_FRAG);

	// Resize
	m_vertexStrideVec.resize(m_maxFragNum);
	m_dSampledVertexIdxVec.resize(m_maxSampledVertexNum);;
	m_dMatchingPointsIdxVec.resize(2 * MAX_CLOSURE_NUM_EACH_FRAG * SAMPLED_VERTEX_NUM_EACH_FRAG * MAX_FRAG_NUM);
	m_dNewSampledVertexIdxVec.resize(SAMPLED_VERTEX_NUM_EACH_FRAG);
	m_dNewNodeVec.resize(NODE_NUM_EACH_FRAG);

	m_vertexStrideVec[0] = 0;
}

FragDeformableMeshData::~FragDeformableMeshData()
{
	m_vertexStrideVec.clear();
	m_dSampledVertexIdxVec.clear();
	m_dMatchingPointsIdxVec.clear();
	m_dNewSampledVertexIdxVec.clear();
	m_dNewNodeVec.clear();
}

void DeformableMeshData::addNewNode(float4* dNewNode, int newNodeNum)
{
	if ((m_nodeNum + newNodeNum) > m_maxNodeNum)
	{
		std::cout << "error: node num exceed limit" << std::endl;
		std::exit(0);
	}

	checkCudaErrors(cudaMemcpy(RAW_PTR(m_dNodeVec) + m_nodeNum,
		dNewNode, sizeof(float4)*newNodeNum, cudaMemcpyDeviceToDevice));
	m_nodeNum += newNodeNum;
}

void FragDeformableMeshData::addNewFrag(int vertexNum)
{
	m_vertexNum = vertexNum;
	++m_fragNum;
	m_vertexStrideVec[m_fragNum] = vertexNum;
}

void FragDeformableMeshData::addNewVertexIdx(int* dNewVertexIdx, int newVertexNum)
{
	if ((m_sampledVertexNum + newVertexNum) > m_maxSampledVertexNum)
	{
		std::cout << "error: sampled vertex num exceed limit" << std::endl;
		std::exit(0);
	}

	checkCudaErrors(cudaMemcpy(RAW_PTR(m_dSampledVertexIdxVec) + m_sampledVertexNum,
		dNewVertexIdx, sizeof(int)*newVertexNum, cudaMemcpyDeviceToDevice));
	m_sampledVertexNum += newVertexNum;
}

void FragDeformableMeshData::sampleNewNodeAndVertexIdx()
{
	const int currentFragIdx = m_fragNum - 1;
	const int vertexBaseIdx = m_vertexStrideVec[currentFragIdx];
	const int vertexNumThisFrag = m_vertexStrideVec[currentFragIdx + 1] - m_vertexStrideVec[currentFragIdx];

	int startIdx = 9999 % vertexNumThisFrag;
	int step = vertexNumThisFrag / (SAMPLED_VERTEX_NUM_EACH_FRAG + MYEPS);
	int block = 256, grid = DivUp(SAMPLED_VERTEX_NUM_EACH_FRAG, block);
	SampleVertexIdxKernel << <grid, block >> >(RAW_PTR(m_dNewSampledVertexIdxVec), SAMPLED_VERTEX_NUM_EACH_FRAG,
	                                           vertexNumThisFrag, step, startIdx, vertexBaseIdx);
	addNewVertexIdx(RAW_PTR(m_dNewSampledVertexIdxVec), SAMPLED_VERTEX_NUM_EACH_FRAG);

	startIdx = 19999 % vertexNumThisFrag;
	step = vertexNumThisFrag / (NODE_NUM_EACH_FRAG + MYEPS);
	block = 256, grid = DivUp(NODE_NUM_EACH_FRAG, block);
	SampleNodeKernel << <grid, block >> >(RAW_PTR(m_dNewNodeVec), NODE_NUM_EACH_FRAG,
	                                      RAW_PTR(m_dVertexVec), vertexNumThisFrag, step, startIdx, vertexBaseIdx);
#if 0
	thrust::host_vector<float4> newNodeVec = m_dNewNodeVec;
	std::cout << "sampled new node: " << std::endl;
	for (int i = 0; i < 16; ++i)
	{
		std::cout << newNodeVec[i].x << " : " << newNodeVec[i].y << " : " << newNodeVec[i].z << std::endl;
	}
#endif
	addNewNode(RAW_PTR(m_dNewNodeVec), NODE_NUM_EACH_FRAG);
#if 0
	thrust::host_vector<float4> nodeVec = m_dNodeVec;
	std::cout << "sampled node: " << std::endl;
	for (int i = 0; i < 32; ++i)
	{
		std::cout << nodeVec[i].x << " : " << nodeVec[i].y << " : " << nodeVec[i].z << std::endl;
	}
#endif
}

void FragDeformableMeshData::getVertexAndNodeRelation()
{
	const int currentFragIdx = m_fragNum - 1;
	const int vertexNumThisFrag = m_vertexStrideVec[currentFragIdx + 1] - m_vertexStrideVec[currentFragIdx];
	const int vertexBaseIdx = m_vertexStrideVec[currentFragIdx];
	const int nodeBaseIdx = currentFragIdx * NODE_NUM_EACH_FRAG;

	try
	{
		m_flannKnn->clear();
		m_flannKnn->InitKDTree(RAW_PTR(m_dNodeVec) + nodeBaseIdx, NODE_NUM_EACH_FRAG);

		m_flannKnn->GetKnnResult(RAW_PTR(m_dVertexRelaIdxVec) + 4 * vertexBaseIdx,
		                         RAW_PTR(m_dVertexRelaWeightVec) + 4 * vertexBaseIdx,
		                         RAW_PTR(m_dVertexVec) + vertexBaseIdx,
		                         vertexNumThisFrag, MAX_NEAR_NODE_NUM_VERTEX);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());

		m_flannKnn->GetKnnResult(RAW_PTR(m_dNodeRelaIdxVec) + 8 * nodeBaseIdx,
		                         RAW_PTR(m_dNodeRelaWeightVec) + 8 * nodeBaseIdx,
		                         RAW_PTR(m_dNodeVec) + nodeBaseIdx,
		                         NODE_NUM_EACH_FRAG, MAX_NEAR_NODE_NUM_NODE);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());
	}
	catch (std::bad_alloc& e)
	{
		std::cout << "Error in KD-tree: Run out of memory" << std::endl;
		exit(0);
	}
	catch (thrust::system_error& e)
	{
		std::cout << "Error in KD-tree: " << e.what() << std::endl;
		exit(0);
	}

	int block = 256, grid = DivUp(vertexNumThisFrag, block);
	// For the distance of the kd-tree may be unaccurate
	ComputeV2NDistKernel << <grid, block >> >(RAW_PTR(m_dVertexRelaWeightVec) + 4 * vertexBaseIdx,
	                                          RAW_PTR(m_dVertexRelaIdxVec) + 4 * vertexBaseIdx,
	                                          RAW_PTR(m_dVertexVec) + vertexBaseIdx, vertexNumThisFrag,
	                                          RAW_PTR(m_dNodeVec) + nodeBaseIdx, MAX_NEAR_NODE_NUM_VERTEX);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());	

#if 0
	std::cout << "-------------" << std::endl;
	std::cout << "vertexBaseIdx: " << vertexBaseIdx << std::endl;
	std::cout << "vertexNumThisFrag: " << vertexNumThisFrag << std::endl;
	std::cout << "-------------" << std::endl;
#endif
	
	// calculate weight exp(||v-g||^2/(2*sigma^2)), then normalize
	const float sum = thrust::reduce(m_dVertexRelaWeightVec.begin() + 4 * vertexBaseIdx,
		m_dVertexRelaWeightVec.begin() + 4 * vertexBaseIdx + 4 * vertexNumThisFrag, (float)0, thrust::plus<float>());
	assert(sum > MYEPS);
	const float variance = 2 * pow(0.5 * sum / (vertexNumThisFrag * 4), 2); // variance of gaussian	

	block = 256, grid = DivUp(4 * vertexNumThisFrag, block);
	AddRelaIdxBaseKernel << <grid, block >> >(RAW_PTR(m_dVertexRelaIdxVec) + 4 * vertexBaseIdx,
		currentFragIdx * NODE_NUM_EACH_FRAG, 4 * vertexNumThisFrag);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	block = 256, grid = DivUp(8 * NODE_NUM_EACH_FRAG, block);
	AddRelaIdxBaseKernel << <grid, block >> >(RAW_PTR(m_dNodeRelaIdxVec) + 8 * nodeBaseIdx,
		currentFragIdx * NODE_NUM_EACH_FRAG, 8 * NODE_NUM_EACH_FRAG);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = 256, grid = DivUp(vertexNumThisFrag, block);
	DistToWeightV2NKernel << <grid, block >> >(RAW_PTR(m_dVertexRelaWeightVec) + 4 * vertexBaseIdx, 1.0f / variance, vertexNumThisFrag);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = 256, grid = DivUp(NODE_NUM_EACH_FRAG, block);
	DistToWeightN2NKernel << <grid, block >> >(RAW_PTR(m_dNodeRelaWeightVec) + 8 * nodeBaseIdx, NODE_NUM_EACH_FRAG);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}


