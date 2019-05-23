#include "Cuda/xDeformationCudaFuncs2.cuh"

#include "Helpers/UtilsMath.h"

__global__ void FilterInvalidMatchingPointsKernel(
	int *matchingPoints,
	int matchingPointsNumDescriptor,
	int matchingPointsNumTotal,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float distThresh1,
	float distThresh2,
	float angleThresh,
	int iter)
{
	int vertexPairInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexPairInd >= matchingPointsNumTotal)
	{
		return;
	}

	int vertexIndSrc, vertexIndTarget;
	float4 posFragIndSrc, posFragIndTarget;
	vertexIndSrc = *(matchingPoints + 2 * vertexPairInd);
	vertexIndTarget = *(matchingPoints + 2 * vertexPairInd + 1);

	if (vertexIndTarget == -1 || vertexIndSrc == -1)
	{
		matchingPoints[2 * vertexPairInd] = -1;
		matchingPoints[2 * vertexPairInd + 1] = -1;
		return;
	}

	float4 updatedPosSrc, updatedPosTarget, updatedNormalSrc, updatedNormalTarget;

	updatedPosSrc = updatedVertexPoses[vertexIndSrc];
	updatedNormalSrc = updatedVertexNormals[vertexIndSrc];
	updatedPosTarget = updatedVertexPoses[vertexIndTarget];
	updatedNormalTarget = updatedVertexNormals[vertexIndTarget];

	updatedPosSrc.w = 1.0f;
	updatedPosTarget.w = 1.0f;

	float dist = norm(updatedPosSrc - updatedPosTarget);
	float4 srcToTargetVec = normalize(updatedPosSrc - updatedPosTarget);
	updatedNormalSrc = normalize(updatedNormalSrc);
	updatedNormalTarget = normalize(updatedNormalTarget);
	//float distThresh1 = 0.04f, distThresh2 = 0.0015f, angleThresh = sin(30.0f * 3.14159254f / 180.f);
	//float distThresh1 = 0.015f, distThresh2 = 0.0015f, angleThresh = sin(25.0f * 3.14159254f / 180.f);
#if 0
	if (iter % 3 == 2)
	{
		distThresh /= 1.2f;
		angleThresh /= 1.2f;
	}
#endif

#if 1
	if (dot(updatedNormalSrc, updatedNormalTarget) < 0
		|| dist > distThresh1
		|| norm(cross(updatedNormalSrc, updatedNormalTarget)) > angleThresh
		|| (dist > distThresh2
			&& (norm(cross(srcToTargetVec, updatedNormalTarget)) > angleThresh
				&& norm(cross(updatedNormalSrc, srcToTargetVec)) > angleThresh)))
	{
		matchingPoints[2 * vertexPairInd] = -1;
		matchingPoints[2 * vertexPairInd + 1] = -1;
		return;
	}
#endif
#if 0
	if (dot(updatedNormalSrc, updatedNormalTarget) < 0
		|| dist > distThresh1
		|| norm(cross(updatedNormalSrc, updatedNormalTarget)) > angleThresh)
	{
		matchingPoints[2 * vertexPairInd] = -1;
		matchingPoints[2 * vertexPairInd + 1] = -1;
		return;
	}
#endif
#if 0
	if (vertexPairInd < matchingPointsNumDescriptor)
	{
		if (dot(updatedNormalSrc, updatedNormalTarget) < 0
			|| dist > distThresh
			|| norm(cross(updatedNormalSrc, updatedNormalTarget)) > angleThresh)
		{
			matchingPoints[2 * vertexPairInd] = -1;
			matchingPoints[2 * vertexPairInd + 1] = -1;
			return;
		}
	}
	else
	{
		if (dot(updatedNormalSrc, updatedNormalTarget) < 0
			|| dist > distThresh
			|| norm(cross(updatedNormalSrc, updatedNormalTarget)) > angleThresh
			|| (dist > distThresh / 50.0
				&& (norm(cross(srcToTargetVec, updatedNormalTarget)) > angleThresh
					&& norm(cross(updatedNormalSrc, srcToTargetVec)) > angleThresh)))
		{
			matchingPoints[2 * vertexPairInd] = -1;
			matchingPoints[2 * vertexPairInd + 1] = -1;
			return;
		}
	}
#endif

	return;
}
void FilterInvalidMatchingPoints(
	int *matchingPointsDevice,
	int matchingPointsNumDescriptor,
	int matchingPointsNumTotal,
	float4 *updatedVertexPosesDevice,
	float4 *updatedVertexNormalsDevice,
	float distThresh1,
	float distThresh2,
	float angleThresh,
	int iter)
{
	int block = 256;
	int grid = DivUp(matchingPointsNumTotal, block);
	FilterInvalidMatchingPointsKernel << < grid, block >> > (matchingPointsDevice,
		matchingPointsNumDescriptor,
		matchingPointsNumTotal,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		distThresh1,
		distThresh2,
		angleThresh,
		iter);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void FindNonzeroJTJBlockVNKernel(int *nonzeroJTJBlock,
	int *matchingPoints,
	int *vertexToNodeIndices,
	int matchingPointsNum,
	int nodeNum)
{
	int vertexPairInd = blockDim.x * blockIdx.x + threadIdx.x;

	if (vertexPairInd >= matchingPointsNum)
	{
		return;
	}

	int vertexIndSrc = *(matchingPoints + 2 * vertexPairInd);
	int vertexIndTarget = *(matchingPoints + 2 * vertexPairInd + 1);
	if (vertexIndSrc < 0 || vertexIndTarget < 0)
	{
		return;
	}

	int *nodeIndexSrc = vertexToNodeIndices + 4 * vertexIndSrc;
	int *nodeIndexTarget = vertexToNodeIndices + 4 * vertexIndTarget;

	int nodeInd[8];
	nodeInd[0] = nodeIndexSrc[0];
	nodeInd[1] = nodeIndexSrc[1];
	nodeInd[2] = nodeIndexSrc[2];
	nodeInd[3] = nodeIndexSrc[3];
	nodeInd[4] = nodeIndexTarget[0];
	nodeInd[5] = nodeIndexTarget[1];
	nodeInd[6] = nodeIndexTarget[2];
	nodeInd[7] = nodeIndexTarget[3];

#pragma unroll
	for (int i = 0; i < 8; ++i)
	{
#pragma unroll
		for (int j = 0; j < 8; ++j)
		{
			nonzeroJTJBlock[nodeInd[i] * nodeNum + nodeInd[j]] = 1;
			nonzeroJTJBlock[nodeInd[j] * nodeNum + nodeInd[i]] = 1;
		}
	}
}

__global__ void FindNonzeroJTJBlockNNKernel(int *nonzeroJTJBlock,
	int *nodeToNodeIndicesDevice,
	int nodeNum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= nodeNum * 8)
	{
		return;
	}

	int nodeInd = idx / 8;
	int otherNodeInd = nodeToNodeIndicesDevice[idx];

	nonzeroJTJBlock[nodeInd * nodeNum + otherNodeInd] = 1;
	nonzeroJTJBlock[otherNodeInd * nodeNum + nodeInd] = 1;

}

__global__ void FillNodeToNodePermutationIndVec(int *regTermNodeIndVec,
	int *regTermEquIndVec,
	int *nodeToNodeIndices,
	int nodeNum,
	int offset)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= nodeNum)
	{
		return;
	}

#pragma unroll
	for (int iter = 0; iter < 8; iter++)
	{
		regTermNodeIndVec[idx * 15 + iter] = nodeToNodeIndices[idx * 8 + iter];
	}
#pragma unroll
	for (int iter = 0; iter < 7; iter++)
	{
		regTermNodeIndVec[idx * 15 + 8 + iter] = idx;
	}
#pragma unroll
	for (int iter = 0; iter < 8; iter++)
	{
		regTermEquIndVec[idx * 15 + iter] = offset + idx * 8 + iter;
	}
#pragma unroll
	for (int iter = 0; iter < 7; iter++)
	{
		regTermEquIndVec[idx * 15 + 8 + iter] = offset + idx * 8 + iter + 1;
	}
}

__global__ void FillVertexToNodePermutationIndVec(int *dataTermNodeIndVec,
	int *dataTermEquIndVec,
	int *matchingPoints,
	int *vertexToNodeIndices,
	int matchingPointsNum)
{
	int vertexPairInd = blockDim.x * blockIdx.x + threadIdx.x;

	if (vertexPairInd >= matchingPointsNum)
	{
		return;
	}

	int tmp = vertexPairInd << 3;
	int cnt = tmp;
	dataTermEquIndVec[cnt++] = vertexPairInd;
	dataTermEquIndVec[cnt++] = vertexPairInd;
	dataTermEquIndVec[cnt++] = vertexPairInd;
	dataTermEquIndVec[cnt++] = vertexPairInd;
	dataTermEquIndVec[cnt++] = vertexPairInd;
	dataTermEquIndVec[cnt++] = vertexPairInd;
	dataTermEquIndVec[cnt++] = vertexPairInd;
	dataTermEquIndVec[cnt] = vertexPairInd;

	int vertexIndSrc = *(matchingPoints + 2 * vertexPairInd);
	int vertexIndTarget = *(matchingPoints + 2 * vertexPairInd + 1);
	if (vertexIndSrc < 0 || vertexIndTarget < 0)
	{
		cnt = tmp;
		dataTermNodeIndVec[cnt++] = -1;
		dataTermNodeIndVec[cnt++] = -1;
		dataTermNodeIndVec[cnt++] = -1;
		dataTermNodeIndVec[cnt++] = -1;
		dataTermNodeIndVec[cnt++] = -1;
		dataTermNodeIndVec[cnt++] = -1;
		dataTermNodeIndVec[cnt++] = -1;
		dataTermNodeIndVec[cnt] = -1;
		return;
	}
	int *nodeIndSrc = vertexToNodeIndices + 4 * vertexIndSrc;
	int *nodeIndTarget = vertexToNodeIndices + 4 * vertexIndTarget;

	cnt = tmp;
	dataTermNodeIndVec[cnt++] = nodeIndTarget[0];
	dataTermNodeIndVec[cnt++] = nodeIndTarget[1];
	dataTermNodeIndVec[cnt++] = nodeIndTarget[2];
	dataTermNodeIndVec[cnt++] = nodeIndTarget[3];
	dataTermNodeIndVec[cnt++] = nodeIndSrc[0];
	dataTermNodeIndVec[cnt++] = nodeIndSrc[1];
	dataTermNodeIndVec[cnt++] = nodeIndSrc[2];
	dataTermNodeIndVec[cnt] = nodeIndSrc[3];
}

__global__ void CountInvalidMatchingPoints(int *invalidMatchingPointNum,
	int *matchingPoints,
	int matchingPointsNum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= matchingPointsNum)
	{
		return;
	}

	if (matchingPoints[2 * idx] < 0 || matchingPoints[2 * idx + 1] < 0)
	{
		atomicAdd(invalidMatchingPointNum, 1);
	}
}

__global__ void CountEquNumKernel(int *equNumEachNodeVec,
	int *equIndPermutationVec,
	int equPermutationLength)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= equPermutationLength)
	{
		return;
	}

	atomicAdd(&equNumEachNodeVec[equIndPermutationVec[idx]], 1);
}

__global__ void CalcEquIndForNonZeroBlocksKernel(
	int* equIndVec,
	int4* nonZeroBlockInfoCoo,
	int* equIndOffsetVec,
	int *equIndPermutationVec,
	int *equNumEachNodeVec,
	int *exScanEquNumEachNodeVec,
	int nonZeroBlockNum,
	int nodeNum,
	int matchingPointsNum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= nonZeroBlockNum)
	{
		return;
	}

	int i, j, offset, num_i, num_j;
	int *start_i_set, *start_j_set;
	i = nonZeroBlockInfoCoo[idx].x;
	j = nonZeroBlockInfoCoo[idx].y;
	offset = equIndOffsetVec[i * nodeNum + j];

	start_i_set = equIndPermutationVec + exScanEquNumEachNodeVec[i];
	start_j_set = equIndPermutationVec + exScanEquNumEachNodeVec[j];
	num_i = equNumEachNodeVec[i];
	num_j = equNumEachNodeVec[j];

	int iter_i = 0, iter_j = 0, count_data_term = 0, count_all_term = 0, i_val, j_val;
	for (; iter_i < num_i && iter_j < num_j;) {
		i_val = start_i_set[iter_i];
		j_val = start_j_set[iter_j];
		if (i_val == j_val) {
			equIndVec[offset + count_all_term] = i_val;
			if (i_val < matchingPointsNum) {
				++count_data_term;
			}
			count_all_term++;
			iter_i++;
			iter_j++;
		}
		else if (i_val < j_val) {
			iter_i++;
		}
		else {
			iter_j++;
		}
	}
	nonZeroBlockInfoCoo[idx].z = count_data_term;
	nonZeroBlockInfoCoo[idx].w = count_all_term - count_data_term;
}

void ComputeNonZeroJTJBlockStatistics(
	int &validMatchingPointsNum,
	JTJBlockStatistics &blockStatistics,
	int *matchingPointsDevice,
	int matchingPointsNumDescriptor,
	int matchingPointsNumNearest,
	int matchingPointsNumTotal,
	int nodeNum,
	int *nodeToNodeIndicesDevice,
	int *vertexToNodeIndicesDevice)
{
	int *invalidMatchingPointNumDevice;
	int invalidMatchingPointNum;
	checkCudaErrors(cudaMalloc((void**)&invalidMatchingPointNumDevice, sizeof(int)));
	checkCudaErrors(cudaMemset(invalidMatchingPointNumDevice, 0, sizeof(int)));
	int block = 256;
	int grid = DivUp(matchingPointsNumTotal, block);
	if (grid > 0)
	{
		CountInvalidMatchingPoints << < grid, block >> > (invalidMatchingPointNumDevice,
			matchingPointsDevice,
			matchingPointsNumTotal);
	}
	checkCudaErrors(cudaMemcpy(&invalidMatchingPointNum, invalidMatchingPointNumDevice,
		sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(invalidMatchingPointNumDevice));
	validMatchingPointsNum = matchingPointsNumTotal - invalidMatchingPointNum;
	std::cout << "validMatchingPointsNum: " << validMatchingPointsNum << std::endl;
	std::cout << "invalidMatchingPointNum: " << invalidMatchingPointNum << std::endl;

	thrust::device_vector<int> nonZeroJTJBlockDevice(nodeNum * nodeNum, 0);

	block = 512;
	grid = DivUp(matchingPointsNumTotal, block);
	if (grid > 0)
	{
		FindNonzeroJTJBlockVNKernel << < grid, block >> > (RAW_PTR(nonZeroJTJBlockDevice),
			matchingPointsDevice, vertexToNodeIndicesDevice, matchingPointsNumTotal, nodeNum);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

#if 0
	std::vector<int> vec(nodeNum * 8);
	checkCudaErrors(cudaMemcpy(vec.data(), nodeToNodeIndicesDevice,
		vec.size() * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < vec.size(); ++i)
	{
		std::cout << i / 8 << " : " << vec[i] << std::endl;
	}
	std::cout << std::endl;
	std::exit(0);
#endif

#if 1
	grid = DivUp(nodeNum * 8, block);
	FindNonzeroJTJBlockNNKernel << < grid, block >> > (RAW_PTR(nonZeroJTJBlockDevice),
		nodeToNodeIndicesDevice, nodeNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
#endif

	// 7 because the 8 nearest node must contain the node itself
	int vertexToNodeEquPermutationLength = matchingPointsNumTotal * 8,
		nodeToNodeEquPermutationLength = nodeNum * 8 + nodeNum * 7;
	int equPermutationLength = vertexToNodeEquPermutationLength + nodeToNodeEquPermutationLength;
	thrust::device_vector<int> equIndPermutationVecDevice(equPermutationLength);
	thrust::device_vector<int> nodeIndPermutationVecDevice(equPermutationLength);

	grid = DivUp(matchingPointsNumTotal, block);
	if (grid > 0)
	{
		FillVertexToNodePermutationIndVec << < grid, block >> > (
			RAW_PTR(nodeIndPermutationVecDevice),
			RAW_PTR(equIndPermutationVecDevice),
			matchingPointsDevice,
			vertexToNodeIndicesDevice,
			matchingPointsNumTotal);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

#if 0
	std::vector<int> vec(vertexToNodeEquPermutationLength);
	checkCudaErrors(cudaMemcpy(vec.data(), RAW_PTR(nodeIndPermutationVecDevice),
		vec.size() * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < vec.size(); ++i)
	{
		std::cout << vec[i] << " : ";
	}
	std::cout << std::endl;
	std::exit(0);
#endif

#if 1
	grid = DivUp(nodeNum, block);
	FillNodeToNodePermutationIndVec << < grid, block >> > (
		RAW_PTR(nodeIndPermutationVecDevice) + vertexToNodeEquPermutationLength,
		RAW_PTR(equIndPermutationVecDevice) + vertexToNodeEquPermutationLength,
		nodeToNodeIndicesDevice, nodeNum, matchingPointsNumTotal);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
#endif

	thrust::stable_sort_by_key(nodeIndPermutationVecDevice.begin(), nodeIndPermutationVecDevice.end(),
		equIndPermutationVecDevice.begin());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

#if 0
	std::cout << invalidMatchingPointNum << std::endl;
	std::vector<int> vec1(equPermutationLength);
	std::vector<int> vec2(equPermutationLength);
	checkCudaErrors(cudaMemcpy(vec1.data(), RAW_PTR(nodeIndPermutationVecDevice),
		vec1.size() * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vec2.data(), RAW_PTR(equIndPermutationVecDevice),
		vec2.size() * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < vec1.size(); ++i)
	{
		std::cout << vec1[i] << " : " << vec2[i] << std::endl;
		if (i == invalidMatchingPointNum * 8 - 1)
		{
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
	std::cout << validMatchingPointsNum << std::endl;
	std::exit(0);
#endif

	thrust::device_vector<int> equNumEachNodeVecDevice(nodeNum, 0),
		exScanEquNumEachNodeVecDevice(nodeNum);
	grid = DivUp(equPermutationLength - invalidMatchingPointNum * 8, block);
	CountEquNumKernel << < grid, block >> > (RAW_PTR(equNumEachNodeVecDevice),
		RAW_PTR(nodeIndPermutationVecDevice) + invalidMatchingPointNum * 8,
		equPermutationLength - invalidMatchingPointNum * 8);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	thrust::exclusive_scan(equNumEachNodeVecDevice.begin(), equNumEachNodeVecDevice.end(), exScanEquNumEachNodeVecDevice.begin());

#if 0
	std::vector<int> vec(nodeNum);
	checkCudaErrors(cudaMemcpy(vec.data(), RAW_PTR(equNumEachNodeVecDevice),
		vec.size() * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < vec.size(); ++i)
	{
		std::cout << vec[i] << " : ";
	}
	std::cout << std::endl;
	checkCudaErrors(cudaMemcpy(vec.data(), RAW_PTR(exScanEquNumEachNodeVecDevice),
		vec.size() * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < vec.size(); ++i)
	{
		std::cout << vec[i] << " : ";
	}
	std::cout << std::endl;
	std::exit(0);
#endif

	blockStatistics.m_nonZeroBlockVec = nonZeroJTJBlockDevice;
#if 0
	for (int i = 0; i < nodeNum; i++)
	{
		for (int j = 0; j < nodeNum; j++)
		{
			std::cout << blockStatistics.m_nonZeroBlockVec[i * nodeNum + j] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::exit(0);
#endif
	int nnz = thrust::reduce(nonZeroJTJBlockDevice.begin(), nonZeroJTJBlockDevice.end());
	blockStatistics.m_nonZeroBlockNum = (nnz + nodeNum) / 2;

	thrust::host_vector<int> equNumEachNodeVecHost = equNumEachNodeVecDevice, columnIndEachRowHost;
	thrust::host_vector<int4> nonZeroBlockInfoCooHost(blockStatistics.m_nonZeroBlockNum);
	std::vector<int> equIndOffsetVecHost(nodeNum * nodeNum);

	int count = 0, offset = 0;
	for (int i = 0; i < nodeNum; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			if (blockStatistics.m_nonZeroBlockVec[i * nodeNum + j])
			{
				nonZeroBlockInfoCooHost[count].x = i;
				nonZeroBlockInfoCooHost[count].y = j;
				nonZeroBlockInfoCooHost[count].z = -1;
				nonZeroBlockInfoCooHost[count].w = -1;
				equIndOffsetVecHost[i * nodeNum + j] = offset;
				equIndOffsetVecHost[j * nodeNum + i] = offset;
				offset += std::min(equNumEachNodeVecHost[i], equNumEachNodeVecHost[j]);
				++count;
			}
		}
	}
	if (count != blockStatistics.m_nonZeroBlockNum) {
		printf("Error, count <--> nnz_Iij: %d <--> %d\n", count, blockStatistics.m_nonZeroBlockNum);
		std::exit(0);
	}
	else {
		//printf("non-zero JTJ blocks: %d\n", nnz);
	}
#if 0
	std::cout << blockStatistics.m_nonZeroBlockNum << std::endl;
	std::cout << count << std::endl;
	std::exit(0);
#endif

	columnIndEachRowHost.resize(nodeNum * nodeNum, 0);
	for (int i = 0; i < nodeNum; i++)
	{
		count = 0;
		for (int j = 0; j < nodeNum; j++)
		{
			if (blockStatistics.m_nonZeroBlockVec[i * nodeNum + j])
			{
				columnIndEachRowHost[i * nodeNum + j] = count;
				count++;
			}
		}
	}
	blockStatistics.m_columnIndEachRowDevice = columnIndEachRowHost;
	blockStatistics.m_equIndOffsetVecDevice = equIndOffsetVecHost;
	blockStatistics.m_nonZeroBlockInfoCooDevice = nonZeroBlockInfoCooHost;
	blockStatistics.m_equIndVecDevice.resize(offset);

	block = 64;
	grid = DivUp(blockStatistics.m_nonZeroBlockNum, block);
	CalcEquIndForNonZeroBlocksKernel << < grid, block >> > (
		RAW_PTR(blockStatistics.m_equIndVecDevice),
		RAW_PTR(blockStatistics.m_nonZeroBlockInfoCooDevice),
		RAW_PTR(blockStatistics.m_equIndOffsetVecDevice),
		RAW_PTR(equIndPermutationVecDevice) + invalidMatchingPointNum * 8,
		RAW_PTR(equNumEachNodeVecDevice),
		RAW_PTR(exScanEquNumEachNodeVecDevice),
		blockStatistics.m_nonZeroBlockNum,
		nodeNum,
		matchingPointsNumTotal);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

#if 0
	std::vector<int> nodeToNodeIndicesDeviceVec(nodeNum * 8);
	checkCudaErrors(cudaMemcpy(nodeToNodeIndicesDeviceVec.data(), nodeToNodeIndicesDevice,
		nodeToNodeIndicesDeviceVec.size() * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < nodeToNodeIndicesDeviceVec.size(); ++i)
	{
		std::cout << i / 8 << " : "
			<< nodeToNodeIndicesDeviceVec[i] << " : "
			<< i << std::endl;
	}
	std::cout << "------------------ " << std::endl;
	std::cout << matchingPointsNumTotal << std::endl;
	std::vector<int4> vec(blockStatistics.m_nonZeroBlockNum);
	checkCudaErrors(cudaMemcpy(vec.data(), RAW_PTR(blockStatistics.m_nonZeroBlockInfoCooDevice),
		vec.size() * sizeof(int4), cudaMemcpyDeviceToHost));
	std::cout << vec.size() << std::endl;
	thrust::host_vector<int> equIndVec = blockStatistics.m_equIndVecDevice;
	for (int i = 0; i < vec.size(); ++i)
	{
		std::cout << vec[i].x << " : "
			<< vec[i].y << " : "
			<< vec[i].z << " : "
			<< vec[i].w << std::endl;
		for (int j = 0; j < vec[i].w; ++j)
		{
			int *ptr = RAW_PTR(equIndVec) + equIndOffsetVecHost[vec[i].x * nodeNum + vec[i].y] + vec[i].z;
			std::cout << ptr[j] - matchingPointsNumTotal << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::exit(0);
#endif

	return;
}

void ComputeJTJiaAndja(CSRType &JTJ,
	int nodeNum,
	JTJBlockStatistics &blockStatistics)
{
	int variableNum = nodeNum * 12;
	int nonZeroNum = (blockStatistics.m_nonZeroBlockNum * 2 - nodeNum) * 144;

	std::vector<int> JTJiaVec(variableNum + 1);
	std::vector<int>  JTJjaVec(nonZeroNum);
	thrust::host_vector<int> &nonZeroBlockVec = blockStatistics.m_nonZeroBlockVec;
	int offset = 0, countCol = 0, countRow = 1, countNnz = 0;
	JTJiaVec[0] = 0;
	for (int i = 0; i < nodeNum; ++i) {
		offset = 0;
		countNnz = 0;
		// update first row of 12 rows
		for (int j = 0; j < nodeNum; j++) {
			if (nonZeroBlockVec[i * nodeNum + j]) {
				for (int k = 0; k < 12; ++k) {
					JTJjaVec[countCol++] = countNnz * 12 + k;
				}
				offset += 12;
			}
			++countNnz;
		}
		for (int t = 0; t < 11 * offset; ++t) {
			JTJjaVec[countCol] = JTJjaVec[countCol - offset];
			++countCol;
		}
		// update each row of JTJ_ia
		for (int k = 0; k < 12; ++k) {
			JTJiaVec[countRow] = JTJiaVec[countRow - 1] + offset;
			++countRow;
		}
	}
	checkCudaErrors(cudaMemcpy(JTJ.m_ia, JTJiaVec.data(), JTJiaVec.size() * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(JTJ.m_ja, JTJjaVec.data(), JTJjaVec.size() * sizeof(int), cudaMemcpyHostToDevice));

	return;
}

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
	int iter)
{
#if 1
	ComputeJTJAndJTResidualGeoTermPointToPlain(JTJ, JTResidual,
		residual,
		blockStatistics,
		matchingPointsDevice,
		matchingPointsNumDescriptor,
		matchingPointsNumNearest,
		matchingPointsNumTotal,
		nodeNum,
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		nodeVIndicesDevice,
		nodeToNodeIndicesDevice,
		nodeToNodeWeightsDevice,
		vertexToNodeIndicesDevice,
		vertexToNodeWeightsDevice,
		w_geo);
#endif

	//if (iter >= 0)
	{
#if 0
		ComputeJTJAndJTResidualPhotoTerm(JTJ, JTResidual,
			residual,
			blockStatistics,
			width, height, fx, fy, cx, cy,
			matchingPointsDevice,
			matchingPointsNumDescriptor,
			matchingPointsNumNearest,
			matchingPointsNumTotal,
			nodeNum,
			originVertexPosesDevice,
			updatedVertexPosesDevice,
			updatedVertexNormalsDevice,
			keyGrayImgsDevice,
			keyGrayImgsDxDevice,
			keyGrayImgsDyDevice,
			updatedKeyPosesInvDevice,
			nodeVIndicesDevice,
			nodeToNodeIndicesDevice,
			nodeToNodeWeightsDevice,
			vertexToNodeIndicesDevice,
			vertexToNodeWeightsDevice,
			w_photo);
#endif
	}

#if 1
	ComputeJTJAndJTResidualRegTerm(JTJ, JTResidual,
		residual,
		blockStatistics,
		nodeNum,
		matchingPointsNumTotal,
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		nodeVIndicesDevice,
		nodeToNodeIndicesDevice,
		nodeToNodeWeightsDevice,
		Rts,
		//2.0f);
		w_reg);
#endif

#if 1
	ComputeJTJAndJTResidualRotTerm(JTJ, JTResidual,
		residual,
		blockStatistics,
		nodeNum,
		Rts,
		//2.0f);
		w_rot);
#endif

	return;
}

void ComputeJTJAndJTResidualRotTerm(CSRType &JTJ, float *JTResidual,
	float *residual,
	JTJBlockStatistics &blockStatistics,
	int nodeNum,
	float3 * Rts,
	float w_rot)
{
	int block = 512;
	int grid = DivUp(nodeNum, block);
	ComputeResidualRotTermKernel << <grid, block >> > (
		residual,
		nodeNum,
		Rts,
		w_rot);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = 96;
	grid= nodeNum;
	ComputeJTJAndJTResidualRotTermKernel << <grid, block >> > (
		JTJ.m_a,
		JTResidual,
		JTJ.m_ia,
		residual,
		nodeNum,
		(float *)Rts,
		RAW_PTR(blockStatistics.m_columnIndEachRowDevice),
		w_rot);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void ComputeResidualRotTermKernel(
	float *residual,
	int nodeNum,
	float3 * Rts,
	float w_rot)
{
	int iNode = blockDim.x * blockIdx.x + threadIdx.x;
	int ind = 0;
	if (iNode < nodeNum) {
		int node_seri;
		node_seri = iNode * 12;
		ind = iNode * 7;
		float3 *R = Rts + 4 * iNode;
	
		residual[ind + 0] = w_rot * dot(R[0], R[1]);
		residual[ind + 1] = w_rot * dot(R[0], R[2]);
		residual[ind + 2] = w_rot * dot(R[1], R[2]);
		residual[ind + 3] = w_rot * (dot(R[0], R[0]) - 1);
		residual[ind + 4] = w_rot * (dot(R[1], R[1]) - 1);
		residual[ind + 5] = w_rot * (dot(R[2], R[2]) - 1);
		//residual[ind + 6] = 0.0f;

#if 0
		if (_TURN_DETA_ON) {
			d_b[ind + 6] = w_rot * (x[node_seri + 0] * (x[node_seri + 4] * x[node_seri + 8] - x[node_seri + 5] * x[node_seri + 7])
				+ x[node_seri + 1] * (x[node_seri + 5] * x[node_seri + 6] - x[node_seri + 3] * x[node_seri + 8])
				+ x[node_seri + 2] * (x[node_seri + 3] * x[node_seri + 7] - x[node_seri + 4] * x[node_seri + 6])
				- 1.0);
		}
		else {
			d_b[ind + 6] = 0;
		}
#endif
	}
}

__global__ void ComputeJTJAndJTResidualRotTermKernel(
	float *JTJ_a,
	float *JTResidual,
	int *JTJ_ia,
	float *residual,
	int nodeNum,
	float * Rts,
	int *columnIndEachRow,
	float w_rot)
{
	__shared__ float s_res[81];
	__shared__ float s_x[12];
	__shared__ float s_J[9 * 7];
	__shared__ float s_JTb[12];
	if (threadIdx.x < 81) {
		s_res[threadIdx.x] = 0.0f;
	}
	if (threadIdx.x < 63) {
		s_J[threadIdx.x] = 0.0f;
	}
	if (threadIdx.x < 12) {
		s_x[threadIdx.x] = Rts[blockIdx.x * 12 + threadIdx.x];
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
#if 0
		// det A
		s_J[9 * 6 + 0] = s_x[4] * s_x[8] - s_x[5] * s_x[7];
		s_J[9 * 6 + 1] = s_x[5] * s_x[6] - s_x[3] * s_x[8];
		s_J[9 * 6 + 2] = s_x[3] * s_x[7] - s_x[6] * s_x[4];
		s_J[9 * 6 + 3] = s_x[2] * s_x[7] - s_x[1] * s_x[8];
		s_J[9 * 6 + 4] = s_x[0] * s_x[8] - s_x[2] * s_x[6];
		s_J[9 * 6 + 5] = s_x[1] * s_x[6] - s_x[0] * s_x[7];
		s_J[9 * 6 + 6] = s_x[1] * s_x[5] - s_x[2] * s_x[4];
		s_J[9 * 6 + 7] = s_x[2] * s_x[3] - s_x[0] * s_x[5];
		s_J[9 * 6 + 8] = s_x[0] * s_x[4] - s_x[1] * s_x[3];
#endif
	}
	__syncthreads();

	int row_res = threadIdx.x / 9;
	int col_res = threadIdx.x % 9;
	// reduction
	if (threadIdx.x < 81) {
		float squ_weight = w_rot * w_rot;
#pragma unroll
		for (int iter = 0; iter < 7; iter++) {
			s_res[threadIdx.x] += s_J[iter * 9 + row_res] * s_J[iter * 9 + col_res] * squ_weight;
		}
	}
	__syncthreads();

	// write back
	int start_pos;
	if (threadIdx.x < 81) {
		start_pos = columnIndEachRow[blockIdx.x * nodeNum + blockIdx.x] * 12;
		JTJ_a[JTJ_ia[blockIdx.x * 12 + row_res] + start_pos + col_res] += s_res[row_res * 9 + col_res];

	}

	// calculate JTb
	__syncthreads();
	if (threadIdx.x < 9) {
		start_pos = blockIdx.x * 7;

#pragma unroll
		for (int iter = 0; iter < 7; iter++) {
			s_JTb[threadIdx.x] += s_J[iter * 9 + threadIdx.x] * residual[start_pos + iter] * w_rot;
		}
	}
	__syncthreads();
	// write back
	if (threadIdx.x < 9) {
		int save_start_pos = blockIdx.x * 12;
		JTResidual[save_start_pos + threadIdx.x] -= s_JTb[threadIdx.x];
	}
}

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
	float w_reg)
{
	int block = 192; // fixed !!!
	int grid = blockStatistics.m_nonZeroBlockNum; 
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	ComputeJTJRegTermKernel << <grid, block >> > (
		JTJ.m_a,
		JTJ.m_ia,
		RAW_PTR(blockStatistics.m_equIndVecDevice),
		RAW_PTR(blockStatistics.m_equIndOffsetVecDevice),
		RAW_PTR(blockStatistics.m_nonZeroBlockInfoCooDevice),
		RAW_PTR(blockStatistics.m_columnIndEachRowDevice),
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		nodeVIndicesDevice,
		nodeToNodeIndicesDevice,
		nodeToNodeWeightsDevice,
		nodeNum,
		matchingPointsNum,
		w_reg);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());	

	block = 512;
	grid = DivUp(nodeNum * 8, block);
	ComputeResidualRegTermKernel << <grid, block >> > (
		residual,
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		nodeVIndicesDevice,
		nodeToNodeIndicesDevice,
		nodeToNodeWeightsDevice, 
		nodeNum,
		Rts,
		w_reg);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = 96; // fixed!!!
	grid = blockStatistics.m_nonZeroBlockNum;
	ComputeJTResidualRegTermKernel << <grid, block >> > (
		JTResidual,
		residual,
		RAW_PTR(blockStatistics.m_equIndVecDevice),
		RAW_PTR(blockStatistics.m_equIndOffsetVecDevice),
		RAW_PTR(blockStatistics.m_nonZeroBlockInfoCooDevice),
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		nodeVIndicesDevice,
		nodeToNodeIndicesDevice,
		nodeToNodeWeightsDevice,
		nodeNum,
		matchingPointsNum,
		w_reg);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

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
	float w_reg)
{
	int num = nonZeroBlockInfoCoo[blockIdx.x].w;
	if (num <= 0) {
		return;
	}

	int row = nonZeroBlockInfoCoo[blockIdx.x].x;
	int col = nonZeroBlockInfoCoo[blockIdx.x].y;
	int data_num = nonZeroBlockInfoCoo[blockIdx.x].z;

	//if (row != 0 || col != 0)
		//return;

	// assume the node pairs num of any diagnal block will not exceed 32
	__shared__ float Jci[32 * 3 * 12];
	__shared__ float Jcj[4 * 3 * 12];
	__shared__ float block_JTJ_res[12 * 12];

	if (threadIdx.x < 144) {
		block_JTJ_res[threadIdx.x] = 0.0f;
		Jcj[threadIdx.x] = 0.0f;
	}
#pragma unroll
	for (int iter = 0; iter < 6; iter++) {
		Jci[threadIdx.x + iter * 192] = 0.0f;
	}

	__syncthreads();

	int seri_k, seri_j, f_no;
	float4 node_k, node_j, vvi;
	float weight_n2n = 0.0f;
	int *f_set = equIndVec + equIndOffsetVec[row * nodeNum + col] + data_num;
	int row_res, col_res;
	row_res = threadIdx.x / 12;
	col_res = threadIdx.x % 12;

	if (row == col) {
		if (threadIdx.x < num) {
			f_no = f_set[threadIdx.x] - matchingPointsNum;
			if ((f_no - (f_no / 8) * 8) != 0) {
				seri_k = f_no / 8;
				weight_n2n = nodeToNodeWeights[f_no] * w_reg;
				seri_j = nodeToNodeIndices[f_no];
				node_k = originVertexPoses[nodeVIndices[seri_k]];
				node_j = originVertexPoses[nodeVIndices[seri_j]];
				vvi = node_k - node_j;
				if (seri_k == row) {
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
				else {
					Jci[threadIdx.x * 36 + 9] = weight_n2n;
					Jci[threadIdx.x * 36 + 12 + 10] = weight_n2n;
					Jci[threadIdx.x * 36 + 24 + 11] = weight_n2n;
				}
			}
		}
		__syncthreads();
		//reduction
		if (threadIdx.x < 144) {
			for (int iter_redu = 0; iter_redu < num * 3; iter_redu++) {
				block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + row_res] * Jci[iter_redu * 12 + col_res];
			}
		}
		__syncthreads();
		//write back
		int start_pos;
		if (threadIdx.x < 144) {
			start_pos = columnIndEachRow[row * nodeNum + col] * 12;
			JTJ_a[JTJ_ia[row * 12 + row_res] + start_pos + col_res] += block_JTJ_res[row_res * 12 + col_res];
		}
	}
	else 
	{
		if (threadIdx.x < num) {
			f_no = f_set[threadIdx.x] - matchingPointsNum;
			seri_k = f_no / 8;
			weight_n2n = nodeToNodeWeights[f_no] * w_reg;
			seri_j = nodeToNodeIndices[f_no];
			node_k = originVertexPoses[nodeVIndices[seri_k]];
			node_j = originVertexPoses[nodeVIndices[seri_j]];
			vvi = node_k - node_j;
			if (seri_k == row) {
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
			else {
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
		if (threadIdx.x < 144) {
			for (int iter_redu = 0; iter_redu < num * 3; iter_redu++) {
				block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + row_res] * Jcj[iter_redu * 12 + col_res];
			}
		}

		__syncthreads();
		//write back
		int start_pos;
		if (threadIdx.x < 144) {
			start_pos = columnIndEachRow[row * nodeNum + col] * 12;
			JTJ_a[JTJ_ia[row * 12 + row_res] + start_pos + col_res] += block_JTJ_res[row_res * 12 + col_res];

			start_pos = columnIndEachRow[col * nodeNum + row] * 12;
			JTJ_a[JTJ_ia[col * 12 + row_res] + start_pos + col_res] += block_JTJ_res[col_res * 12 + row_res];
		}
	}
}

__global__ void ComputeResidualRegTermKernel(
	float *residual,
	float4 *originVertexPoses,
	float4 *updatedVertexPoses,
	int *nodeVIndices,
	int *nodeToNodeIndices,
	float *nodeToNodeWeights,
	int nodeNum,
	float3 * Rts,
	float w_reg)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= nodeNum * 8)
	{
		return;
	}

	int seri_k = idx / 8;
	int seri_j = nodeToNodeIndices[idx];
	float weight_N2N = nodeToNodeWeights[idx] * w_reg, tmp;
	float3 vvi, node_k, node_j;
	node_k = make_float3(originVertexPoses[nodeVIndices[seri_k]]);
	node_j = make_float3(originVertexPoses[nodeVIndices[seri_j]]);
	vvi = node_j - node_k;

	float3 *RtNode = Rts + seri_k * 4;

	float3 resi = weight_N2N *
		((vvi.x * RtNode[0] + vvi.y * RtNode[1] + vvi.z * RtNode[2]) + (node_k - node_j) + (RtNode[3] - *(Rts + seri_j * 4 + 3)));
#if 0
		weight_N2N * (vvi.x * RtNode[0] + vvi.y * RtNode[1] + vvi.z * RtNode[2] +
		(node_k + RtNode[3]) -
		(node_j + *(Rts + seri_j * 4 + 3)));
#endif
	//printf("%f %f %f\n", resi.x, resi.y, resi.x);
	residual[3 * idx] = resi.x;
	residual[3 * idx + 1] = resi.y;
	residual[3 * idx + 2] = resi.z;
}

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
	float w_reg)
{
	int row = nonZeroBlockInfoCoo[blockIdx.x].x;
	int col = nonZeroBlockInfoCoo[blockIdx.x].y;
	
	if (row != col) {
		return;
	}

	int data_num = nonZeroBlockInfoCoo[blockIdx.x].z;
	int num = nonZeroBlockInfoCoo[blockIdx.x].w;
	__shared__ float JTb_block_tmp[32 * 3 * 12];
	__shared__ float JTb_block_res[12];
#pragma unroll
	for (int iter = 0; iter < 12; iter++) {
		JTb_block_tmp[threadIdx.x + 96 * iter] = 0.0f;
	}
	if (threadIdx.x < 12) {
		JTb_block_res[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	int seri_k, seri_j, f_no;
	float4 node_k, node_j, vvi;
	float weight_n2n = 0.0f, weight_tmp;
	int *f_set = equIndVec + equIndOffsetVec[row * nodeNum + col] + data_num;
	if (threadIdx.x < num) {
		f_no = f_set[threadIdx.x] - matchingPointsNum;
		if ((f_no - (f_no / 8) * 8) != 0) {
			seri_k = f_no / 8;
			weight_n2n = - nodeToNodeWeights[f_no] * w_reg;
			seri_j = nodeToNodeIndices[f_no];
			node_k = originVertexPoses[nodeVIndices[seri_k]];
			node_j = originVertexPoses[nodeVIndices[seri_j]];
			vvi = node_k - node_j;
			if (seri_k == row) {
				weight_tmp = weight_n2n * residual[f_no * 3];
				JTb_block_tmp[threadIdx.x * 36 + 0] = vvi.x * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 3] = vvi.y * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 6] = vvi.z * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 9] = -weight_tmp;

				weight_tmp = weight_n2n * residual[f_no * 3 + 1];
				JTb_block_tmp[threadIdx.x * 36 + 12 + 1] = vvi.x * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 12 + 4] = vvi.y * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 12 + 7] = vvi.z * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 12 + 10] = -weight_tmp;

				weight_tmp = weight_n2n * residual[f_no * 3 + 2];
				JTb_block_tmp[threadIdx.x * 36 + 24 + 2] = vvi.x * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 24 + 5] = vvi.y * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 24 + 8] = vvi.z * weight_tmp;
				JTb_block_tmp[threadIdx.x * 36 + 24 + 11] = -weight_tmp;
			}
			else {
				JTb_block_tmp[threadIdx.x * 36 + 9] = weight_n2n * residual[f_no * 3];
				JTb_block_tmp[threadIdx.x * 36 + 12 + 10] = weight_n2n * residual[f_no * 3 + 1];
				JTb_block_tmp[threadIdx.x * 36 + 24 + 11] = weight_n2n * residual[f_no * 3 + 2];
			}
		}
	}
	__syncthreads();
	// reduction
	if (threadIdx.x < 12) {
		for (int iter_redu = 0; iter_redu < num * 3; iter_redu++) {
			JTb_block_res[threadIdx.x] += JTb_block_tmp[threadIdx.x + iter_redu * 12];
		}
	}
	__syncthreads();
	// write back
	if (threadIdx.x < 12) {
		JTResidual[row * 12 + threadIdx.x] -= JTb_block_res[threadIdx.x];
	}
}

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
	float w_photo)
{
	int block = 160; // fixed !!!
	int grid = blockStatistics.m_nonZeroBlockNum; //each CUDA block handle one JTJ block.
	ComputeJTJPhotoTermKernel << <grid, block >> > (
		JTJ.m_a,
		JTJ.m_ia,
		width, height, fx, fy, cx, cy,
		RAW_PTR(blockStatistics.m_equIndVecDevice),
		RAW_PTR(blockStatistics.m_equIndOffsetVecDevice),
		RAW_PTR(blockStatistics.m_nonZeroBlockInfoCooDevice),
		RAW_PTR(blockStatistics.m_columnIndEachRowDevice),
		matchingPointsDevice,
		matchingPointsNumTotal,
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		keyGrayImgsDxDevice.first, keyGrayImgsDxDevice.second,
		keyGrayImgsDyDevice.first, keyGrayImgsDyDevice.second,
		updatedKeyPosesInvDevice,
		vertexToNodeIndicesDevice,
		vertexToNodeWeightsDevice,
		nodeVIndicesDevice,
		nodeNum,
		w_photo);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = 1024;
	grid = DivUp(matchingPointsNumTotal, block);
	ComputeResidualPhotoTermKernel << <grid, block >> > (
		residual,
		width, height, fx, fy, cx, cy,
		matchingPointsDevice,
		matchingPointsNumTotal,
		updatedVertexPosesDevice,
		keyGrayImgsDevice.first, keyGrayImgsDevice.second,
		updatedKeyPosesInvDevice,
		w_photo);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

#if 1
	block = 256; // fixed!!!
	grid = blockStatistics.m_nonZeroBlockNum;
	ComputeJTResidualPhotoTermKernel << <grid, block >> > (
		JTResidual,
		residual,
		width, height, fx, fy, cx, cy,
		RAW_PTR(blockStatistics.m_equIndVecDevice),
		RAW_PTR(blockStatistics.m_equIndOffsetVecDevice),
		RAW_PTR(blockStatistics.m_nonZeroBlockInfoCooDevice),
		matchingPointsDevice,
		matchingPointsNumTotal,
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		keyGrayImgsDxDevice.first, keyGrayImgsDxDevice.second,
		keyGrayImgsDyDevice.first, keyGrayImgsDyDevice.second,
		updatedKeyPosesInvDevice,
		vertexToNodeIndicesDevice,
		vertexToNodeWeightsDevice,
		nodeVIndicesDevice,
		nodeNum,
		w_photo);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

#endif
}

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
	float w_photo)
{
	__shared__ float Jci[160 * 12];
	__shared__ float Jcj[160 * 12];
	__shared__ float block_JTJ_res[12 * 12];

	if (threadIdx.x < 144) {
		block_JTJ_res[threadIdx.x] = 0;
	}
	__syncthreads();

	int row = nonZeroBlockInfoCoo[blockIdx.x].x;
	int col = nonZeroBlockInfoCoo[blockIdx.x].y;
	int num = nonZeroBlockInfoCoo[blockIdx.x].z;

	int f_no;
	int *f_set = equIndVec + equIndOffsetVec[row * nodeNum + col];
	float weight_V2N, tmp1, tmp2, tmp3;
	float4 vnn, nor_tar;
	int full_times = num / int(160);
	int residual_num = num % 160;
	int row_res, col_res;
	row_res = threadIdx.x / 12;
	col_res = threadIdx.x % 12;
	int ind_thread = threadIdx.x * 12;

	int nodeIndRow, nodeIndCol, vertexIndRow, vertexIndCol;
	int fragIndOtherRow, fragIndOtherCol;

	for (int iter_ft = 0; iter_ft < full_times; iter_ft++)
	{
		f_no = f_set[threadIdx.x + iter_ft * 160];
		int srcVertexInd = matchingPoints[2 * f_no];
		int targetVertexInd = matchingPoints[2 * f_no + 1];
		if (srcVertexInd < 0 || targetVertexInd < 0)
		{
#pragma unroll
			for (int iter = 0; iter < 12; iter++)
			{
				Jci[threadIdx.x * 12 + iter] = 0.0f;
			}
		}
		else
		{
			float signRow = 1.0f, signCol = 1.0f;
			int nodeIndRow1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, row, 4);
			int nodeIndRow2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, row, 4);
			int nodeIndCol1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, col, 4);
			int nodeIndCol2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, col, 4);
			if ((row / NODE_NUM_EACH_FRAG) == (col / NODE_NUM_EACH_FRAG))
			{
				if (nodeIndRow1 > nodeIndRow2)
				{
					nodeIndRow = nodeIndRow1;
					nodeIndCol = nodeIndCol1;
					vertexIndRow = srcVertexInd;
					vertexIndCol = srcVertexInd;
					fragIndOtherRow = (int)updatedVertexPoses[targetVertexInd].w;
					fragIndOtherCol = (int)updatedVertexPoses[targetVertexInd].w;
				}
				else
				{
					nodeIndRow = nodeIndRow2;
					nodeIndCol = nodeIndCol2;
					vertexIndRow = targetVertexInd;
					vertexIndCol = targetVertexInd;
					fragIndOtherRow = (int)updatedVertexPoses[srcVertexInd].w;
					fragIndOtherCol = (int)updatedVertexPoses[srcVertexInd].w;
				}
			}
			else
			{
				if (nodeIndRow1 > nodeIndRow2)
				{
					nodeIndRow = nodeIndRow1;
					nodeIndCol = nodeIndCol2;
					vertexIndRow = srcVertexInd;
					vertexIndCol = targetVertexInd;
					signCol = -1.0f;
					fragIndOtherRow = (int)updatedVertexPoses[targetVertexInd].w;
					fragIndOtherCol = (int)updatedVertexPoses[srcVertexInd].w;
				}
				else
				{
					nodeIndRow = nodeIndRow2;
					nodeIndCol = nodeIndCol1;
					vertexIndRow = targetVertexInd;
					vertexIndCol = srcVertexInd;
					signRow = -1.0f;
					fragIndOtherRow = (int)updatedVertexPoses[srcVertexInd].w;
					fragIndOtherCol = (int)updatedVertexPoses[targetVertexInd].w;
				}
			}
#if 0
			printf("num: %d %d %d | %d %d %d %d | %d %d %d %d | %d %d %d %d | %d\n",
				num, row, col,
				srcVertexInd, targetVertexInd, nodeIndRow, nodeIndCol,
				nodeIndRow1, nodeIndRow2, nodeIndCol1, nodeIndCol2,
				nodeIndRow, nodeIndCol, vertexIndRow, vertexIndCol,
				columnIndEachRow[row * nodeNum + col]);
#endif	

			CalculateImgDerivative(
				Jcj + ind_thread,
				nodeIndCol,
				vertexIndCol,
				signCol,
				fragIndOtherCol,
				width, height, fx, fy, cx, cy,
				keyGrayImgsDx, keyGrayImgsDxStep,
				keyGrayImgsDy, keyGrayImgsDyStep,
				updatedPosesInv,
				originVertexPoses,
				updatedVertexPoses,
				vertexToNodeIndices,
				vertexToNodeWeights,
				nodeVIndices,
				nodeNum,
				w_photo);
			CalculateImgDerivative(
				Jci + ind_thread,
				nodeIndRow,
				vertexIndRow,
				signRow,
				fragIndOtherRow,
				width, height, fx, fy, cx, cy,
				keyGrayImgsDx, keyGrayImgsDxStep,
				keyGrayImgsDy, keyGrayImgsDyStep,
				updatedPosesInv,
				originVertexPoses,
				updatedVertexPoses,
				vertexToNodeIndices,
				vertexToNodeWeights,
				nodeVIndices,
				nodeNum,
				w_photo);
		}

		__syncthreads();
		//reduction
		if (threadIdx.x < 144) {
#pragma unroll
			for (int iter_redu = 0; iter_redu < 160; iter_redu++) {
				block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + row_res] * Jcj[iter_redu * 12 + col_res];
			}
		}

		__syncthreads();
	}

	//printf("hehe srcVertexInd targetVertexInd %d %d %d %d \n", row_res, col_res, threadIdx.x, residual_num);

	if (threadIdx.x < residual_num) {
		f_no = f_set[threadIdx.x + full_times * 160];
		int srcVertexInd = matchingPoints[2 * f_no];
		int targetVertexInd = matchingPoints[2 * f_no + 1];
		//printf("srcVertexInd targetVertexInd %d %d %d %d\n", srcVertexInd, targetVertexInd, row_res, col_res);
		if (srcVertexInd < 0 || targetVertexInd < 0)
		{
#pragma unroll
			for (int iter = 0; iter < 12; iter++) {
				Jci[threadIdx.x * 12 + iter] = 0.0f;
			}
		}
		else {
			float signRow = 1.0f, signCol = 1.0f;
			int nodeIndRow1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, row, 4);
			int nodeIndRow2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, row, 4);
			int nodeIndCol1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, col, 4);
			int nodeIndCol2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, col, 4);
			if ((row / NODE_NUM_EACH_FRAG) == (col / NODE_NUM_EACH_FRAG))
			{
				if (nodeIndRow1 > nodeIndRow2)
				{
					nodeIndRow = nodeIndRow1;
					nodeIndCol = nodeIndCol1;
					vertexIndRow = srcVertexInd;
					vertexIndCol = srcVertexInd;
					fragIndOtherRow = (int)updatedVertexPoses[targetVertexInd].w;
					fragIndOtherCol = (int)updatedVertexPoses[targetVertexInd].w;
				}
				else
				{
					nodeIndRow = nodeIndRow2;
					nodeIndCol = nodeIndCol2;
					vertexIndRow = targetVertexInd;
					vertexIndCol = targetVertexInd;
					fragIndOtherRow = (int)updatedVertexPoses[srcVertexInd].w;
					fragIndOtherCol = (int)updatedVertexPoses[srcVertexInd].w;
				}
			}
			else
			{
				if (nodeIndRow1 > nodeIndRow2)
				{
					nodeIndRow = nodeIndRow1;
					nodeIndCol = nodeIndCol2;
					vertexIndRow = srcVertexInd;
					vertexIndCol = targetVertexInd;
					signCol = -1.0f;
					fragIndOtherRow = (int)updatedVertexPoses[targetVertexInd].w;
					fragIndOtherCol = (int)updatedVertexPoses[srcVertexInd].w;
				}
				else
				{
					nodeIndRow = nodeIndRow2;
					nodeIndCol = nodeIndCol1;
					vertexIndRow = targetVertexInd;
					vertexIndCol = srcVertexInd;
					signRow = -1.0f;
					fragIndOtherRow = (int)updatedVertexPoses[srcVertexInd].w;
					fragIndOtherCol = (int)updatedVertexPoses[targetVertexInd].w;
				}
			}
#if 0
			printf("num: %d %d %d | %d %d %d %d | %d %d %d %d | %d %d %d %d | %d\n",
				num, row, col,
				srcVertexInd, targetVertexInd, nodeIndRow, nodeIndCol,
				nodeIndRow1, nodeIndRow2, nodeIndCol1, nodeIndCol2,
				nodeIndRow, nodeIndCol, vertexIndRow, vertexIndCol,
				columnIndEachRow[row * nodeNum + col]);
#endif	

			CalculateImgDerivative(
				Jcj + ind_thread,
				nodeIndCol,
				vertexIndCol,
				signCol,
				fragIndOtherCol,
				width, height, fx, fy, cx, cy,
				keyGrayImgsDx, keyGrayImgsDxStep,
				keyGrayImgsDy, keyGrayImgsDyStep,
				updatedPosesInv,
				originVertexPoses,
				updatedVertexPoses,
				vertexToNodeIndices,
				vertexToNodeWeights,
				nodeVIndices,
				nodeNum,
				w_photo);
			CalculateImgDerivative(
				Jci + ind_thread,
				nodeIndRow,
				vertexIndRow,
				signRow,
				fragIndOtherRow,
				width, height, fx, fy, cx, cy,
				keyGrayImgsDx, keyGrayImgsDxStep,
				keyGrayImgsDy, keyGrayImgsDyStep,
				updatedPosesInv,
				originVertexPoses,
				updatedVertexPoses,
				vertexToNodeIndices,
				vertexToNodeWeights,
				nodeVIndices,
				nodeNum,
				w_photo);
			}
		}

	__syncthreads();
	//reduction
	if (threadIdx.x < 144) {
		for (int iter_redu = 0; iter_redu < residual_num; iter_redu++) {
			block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + row_res] * Jcj[iter_redu * 12 + col_res];
		}
	}

	__syncthreads();
	// write JTJ_block(i,j) to global memory.
	int start_pos;
	if (threadIdx.x < 144) {
		if (row == col) {
			start_pos = columnIndEachRow[row * nodeNum + col] * 12;
			JTJ_a[JTJ_ia[row * 12 + row_res] + start_pos + col_res] += block_JTJ_res[row_res * 12 + col_res];
		}
		else {
			start_pos = columnIndEachRow[row * nodeNum + col] * 12;
			JTJ_a[JTJ_ia[row * 12 + row_res] + start_pos + col_res] += block_JTJ_res[row_res * 12 + col_res];
			start_pos = columnIndEachRow[col * nodeNum + row] * 12;
			JTJ_a[JTJ_ia[col * 12 + row_res] + start_pos + col_res] += block_JTJ_res[col_res * 12 + row_res];
		}
	}
	}

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
	float4 *updatedPoseInvTarget = updatedPosesInv + fragIndTarget * 4;

	updatedPosSrcLocalTargetSpace = updatedPosSrc->x * updatedPoseInvTarget[0] + updatedPosSrc->y * updatedPoseInvTarget[1] +
		updatedPosSrc->z * updatedPoseInvTarget[2] + updatedPoseInvTarget[3];
#if 0
	//float4 middlePos = (*updatedPosSrc + *updatedPosTarget) * 0.5f;
	//middlePos.w = 1.0f;
	printf("middle pos: %f %f %f\n updated src pos: %f %f %f\n updated target pos: %f %f %f\n",
		middlePos.x, middlePos.y, middlePos.z,
		updatedPosSrc->x, updatedPosSrc->y, updatedPosSrc->z,
		updatedPosTarget->x, updatedPosTarget->y, updatedPosTarget->z);
#endif

	float *keyGrayImgDxDeviceTarget = keyGrayImgsDx + fragIndTarget * keyGrayImgsDxStep;
	float *keyGrayImgDyDeviceTarget = keyGrayImgsDy + fragIndTarget * keyGrayImgsDyStep;

	uBiSrcTargetSpace = (updatedPosSrcLocalTargetSpace.x * fx) / updatedPosSrcLocalTargetSpace.z + cx;
	vBiSrcTargetSpace = (updatedPosSrcLocalTargetSpace.y * fy) / updatedPosSrcLocalTargetSpace.z + cy;
	uBiSrcTargetSpace = clamp(uBiSrcTargetSpace, (float)0, (float)(width - 2));
	vBiSrcTargetSpace = clamp(vBiSrcTargetSpace, (float)0, (float)(height - 2));
#if 0
	if (uBiSrcTargetSpace < 0 || uBiSrcTargetSpace > width - 2 || vBiSrcTargetSpace < 0 || vBiSrcTargetSpace > height - 2)
	{
		Jc[0] = 0;
		Jc[1] = 0;
		Jc[2] = 0;
		Jc[3] = 0;
		Jc[4] = 0;
		Jc[5] = 0;
		Jc[6] = 0;
		Jc[7] = 0;
		Jc[8] = 0;
		Jc[9] = 0;
		Jc[10] = 0;
		Jc[11] = 0;
		return;
	}
#endif
	// bilinear intarpolation
	uBi0SrcTargetSpace = __float2int_rd(uBiSrcTargetSpace); uBi1SrcTargetSpace = uBi0SrcTargetSpace + 1;
	vBi0SrcTargetSpace = __float2int_rd(vBiSrcTargetSpace); vBi1SrcTargetSpace = vBi0SrcTargetSpace + 1;

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
		(-J_gamma_uv_src_target_space.x * updatedPosSrcLocalTargetSpace.x * fx - J_gamma_uv_src_target_space.y * updatedPosSrcLocalTargetSpace.y * fy) / (updatedPosSrcLocalTargetSpace.z * updatedPosSrcLocalTargetSpace.z));
	J_gamma_xyz_global_src_target_space = J_gamma_xyz_src_target_space.x * make_float3(updatedPoseInvTarget[0].x, updatedPoseInvTarget[0].y, updatedPoseInvTarget[0].z) +
		J_gamma_xyz_src_target_space.y * make_float3(updatedPoseInvTarget[1].x, updatedPoseInvTarget[1].y, updatedPoseInvTarget[1].z) +
		J_gamma_xyz_src_target_space.z * make_float3(updatedPoseInvTarget[2].x, updatedPoseInvTarget[2].y, updatedPoseInvTarget[2].z);

	float tmp1, tmp2, tmp3, weight_V2N;
	float4 vnn;
	weight_V2N = sign * w_photo * vertexToNodeWeights[vertexInd * 4 + nodeInd];
	vnn = originVertexPoses[vertexInd] -
		originVertexPoses[nodeVIndices[vertexToNodeIndices[vertexInd * 4 + nodeInd]]];

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

__global__ void ComputeResidualPhotoTermKernel(
	float *residual,
	int width, int height, float fx, float fy, float cx, float cy,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *updatedVertexPoses,
	float *keyGrayImgs, int keykeyGrayImgsStep,
	float4 *updatedPosesInv,
	float w_photo)
{
	int matchingPairInd = blockDim.x * blockIdx.x + threadIdx.x;
	if (matchingPairInd >= matchingPointsNum)
	{
		return;
	}

	int srcVertexInd = matchingPoints[2 * matchingPairInd];
	int targetVertexInd = matchingPoints[2 * matchingPairInd + 1];

	if (srcVertexInd < 0 || targetVertexInd < 0)
	{
		residual[matchingPairInd] = 0;
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

	updatedPosSrc = updatedVertexPoses + srcVertexInd;
	int fragIndSrc = (int)updatedPosSrc->w;
	updatedPosTarget = updatedVertexPoses + targetVertexInd;
	int fragIndTarget = (int)updatedPosTarget->w;
	float4 *updatedPoseInvSrc = updatedPosesInv + fragIndSrc * 4;
	float4 *updatedPoseInvTarget = updatedPosesInv + fragIndTarget * 4;

	updatedPosSrcLocalTargetSpace = updatedPosSrc->x * updatedPoseInvTarget[0] + updatedPosSrc->y * updatedPoseInvTarget[1] +
		updatedPosSrc->z * updatedPoseInvTarget[2] + updatedPoseInvTarget[3];
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

	float *keyGrayImgSrc = keyGrayImgs + fragIndSrc * keykeyGrayImgsStep;
	float *keyGrayImgTarget = keyGrayImgs + fragIndTarget * keykeyGrayImgsStep;

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

	//printf("diff: %f %f, ", biPixSrcTargetSpace, biPixTargetSrcSpace);
	residual[matchingPairInd] = w_photo * (biPixSrcTargetSpace - biPixTargetSrcSpace);
	//printf("r: %f %f %f %f %f %f\n", biPixTarget, biPixSrcTargetSpace, middlePixIntensityTargetSpace, biPixSrc, biPixTargetSrcSpace, middlePixIntensitySrcSpace);
}

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
	float w_photo)
{
	int blockInd = blockIdx.x;
	int row = nonZeroBlockInfoCoo[blockInd].x;
	int col = nonZeroBlockInfoCoo[blockInd].y;
	if (row != col) {
		return;
	}
	__shared__ float JTb_block_tmp[12 * 256];
	__shared__ float JTb_block_res[12];
	int threadInd = threadIdx.x;
	if (threadInd < 12) {
		JTb_block_res[threadInd] = 0;
	}
	int num = nonZeroBlockInfoCoo[blockInd].z;
	int nodeInd = col;

	int full_times = num / 256;
	int residual_num = num % 256;
	int f_no = 0, corr_no = 0, node_offset = 0;
	int *f_set = equIndVec + equIndOffsetVec[row * nodeNum + col];
	float4 nor_tar, node_pos = originVertexPoses[nodeVIndices[nodeInd]], vnn;
	float weight_V2N = 0.0, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;
	int srcVertexInd, targetVertexInd;
	int nodeIndCol1, nodeIndCol2, nodeIndCol, vertexIndCol, signCol = 1.0f;
	int ind_thread = threadInd * 12;
	int fragIndOtherCol;
	for (int iter_ft = 0; iter_ft < full_times; iter_ft++) {
		f_no = f_set[threadInd + iter_ft * 256];
		//printf("%d %d %d %d %d\n", row, col, f_no, srcVertexInd, targetVertexInd);
		srcVertexInd = matchingPoints[2 * f_no];
		targetVertexInd = matchingPoints[2 * f_no + 1];
		if (srcVertexInd < 0 || targetVertexInd < 0)
		{
			JTb_block_tmp[threadInd * 12 + 0] = 0;
			JTb_block_tmp[threadInd * 12 + 1] = 0;
			JTb_block_tmp[threadInd * 12 + 2] = 0;
			JTb_block_tmp[threadInd * 12 + 3] = 0;
			JTb_block_tmp[threadInd * 12 + 4] = 0;
			JTb_block_tmp[threadInd * 12 + 5] = 0;
			JTb_block_tmp[threadInd * 12 + 6] = 0;
			JTb_block_tmp[threadInd * 12 + 7] = 0;
			JTb_block_tmp[threadInd * 12 + 8] = 0;
			JTb_block_tmp[threadInd * 12 + 9] = 0;
			JTb_block_tmp[threadInd * 12 + 10] = 0;
			JTb_block_tmp[threadInd * 12 + 11] = 0;
		}
		else {
			nodeIndCol1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, col, 4);
			nodeIndCol2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, col, 4);
			if (nodeIndCol1 > nodeIndCol2)
			{
				nodeIndCol = nodeIndCol1;
				vertexIndCol = srcVertexInd;
				signCol = 1.0f;
				fragIndOtherCol = (int)updatedVertexPoses[targetVertexInd].w;
			}
			else
			{
				nodeIndCol = nodeIndCol2;
				vertexIndCol = targetVertexInd;
				signCol = -1.0f;
				fragIndOtherCol = (int)updatedVertexPoses[srcVertexInd].w;
			}

#if 1
			CalculateImgDerivative(
				JTb_block_tmp + ind_thread,
				nodeIndCol,
				vertexIndCol,
				signCol,
				fragIndOtherCol,
				width, height, fx, fy, cx, cy,
				keyGrayImgsDx, keyGrayImgsDxStep,
				keyGrayImgsDy, keyGrayImgsDyStep,
				updatedPosesInv,
				originVertexPoses,
				updatedVertexPoses,
				vertexToNodeIndices,
				vertexToNodeWeights,
				nodeVIndices,
				nodeNum,
				w_photo);
#endif

#if 1
			JTb_block_tmp[ind_thread + 0] *= residual[f_no];
			JTb_block_tmp[ind_thread + 1] *= residual[f_no];
			JTb_block_tmp[ind_thread + 2] *= residual[f_no];

			JTb_block_tmp[ind_thread + 3] *= residual[f_no];
			JTb_block_tmp[ind_thread + 4] *= residual[f_no];
			JTb_block_tmp[ind_thread + 5] *= residual[f_no];

			JTb_block_tmp[ind_thread + 6] *= residual[f_no];
			JTb_block_tmp[ind_thread + 7] *= residual[f_no];
			JTb_block_tmp[ind_thread + 8] *= residual[f_no];

			JTb_block_tmp[ind_thread + 9] *= residual[f_no];
			JTb_block_tmp[ind_thread + 10] *= residual[f_no];
			JTb_block_tmp[ind_thread + 11] *= residual[f_no];
#endif
		}

		__syncthreads();
		//reduction
		if (threadInd < 12)
		{
			for (int iter_redu = 0; iter_redu < 256; iter_redu++)
			{
				JTb_block_res[threadInd] += JTb_block_tmp[threadInd + iter_redu * 12];
			}
		}
		__syncthreads();
	}

	__syncthreads();
	if (threadInd < residual_num) {
		f_no = f_set[threadInd + full_times * 256];
		srcVertexInd = matchingPoints[2 * f_no];
		targetVertexInd = matchingPoints[2 * f_no + 1];
		//printf("%d %d %d %d %d\n", row, col, f_no, srcVertexInd, targetVertexInd);
		if (srcVertexInd < 0 || targetVertexInd < 0)
		{
			JTb_block_tmp[threadInd * 12 + 0] = 0;
			JTb_block_tmp[threadInd * 12 + 1] = 0;
			JTb_block_tmp[threadInd * 12 + 2] = 0;
			JTb_block_tmp[threadInd * 12 + 3] = 0;
			JTb_block_tmp[threadInd * 12 + 4] = 0;
			JTb_block_tmp[threadInd * 12 + 5] = 0;
			JTb_block_tmp[threadInd * 12 + 6] = 0;
			JTb_block_tmp[threadInd * 12 + 7] = 0;
			JTb_block_tmp[threadInd * 12 + 8] = 0;
			JTb_block_tmp[threadInd * 12 + 9] = 0;
			JTb_block_tmp[threadInd * 12 + 10] = 0;
			JTb_block_tmp[threadInd * 12 + 11] = 0;
		}
		else {
			nodeIndCol1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, col, 4);
			nodeIndCol2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, col, 4);
			if (nodeIndCol1 > nodeIndCol2)
			{
				nodeIndCol = nodeIndCol1;
				vertexIndCol = srcVertexInd;
				signCol = 1.0f;
				fragIndOtherCol = (int)updatedVertexPoses[targetVertexInd].w;
			}
			else
			{
				nodeIndCol = nodeIndCol2;
				vertexIndCol = targetVertexInd;
				signCol = -1.0f;
				fragIndOtherCol = (int)updatedVertexPoses[srcVertexInd].w;
			}

#if 1
			CalculateImgDerivative(
				JTb_block_tmp + ind_thread,
				nodeIndCol,
				vertexIndCol,
				signCol,
				fragIndOtherCol,
				width, height, fx, fy, cx, cy,
				keyGrayImgsDx, keyGrayImgsDxStep,
				keyGrayImgsDy, keyGrayImgsDyStep,
				updatedPosesInv,
				originVertexPoses,
				updatedVertexPoses,
				vertexToNodeIndices,
				vertexToNodeWeights,
				nodeVIndices,
				nodeNum,
				w_photo);
#endif

#if 1
			JTb_block_tmp[ind_thread + 0] *= residual[f_no];
			JTb_block_tmp[ind_thread + 1] *= residual[f_no];
			JTb_block_tmp[ind_thread + 2] *= residual[f_no];

			JTb_block_tmp[ind_thread + 3] *= residual[f_no];
			JTb_block_tmp[ind_thread + 4] *= residual[f_no];
			JTb_block_tmp[ind_thread + 5] *= residual[f_no];

			JTb_block_tmp[ind_thread + 6] *= residual[f_no];
			JTb_block_tmp[ind_thread + 7] *= residual[f_no];
			JTb_block_tmp[ind_thread + 8] *= residual[f_no];

			JTb_block_tmp[ind_thread + 9] *= residual[f_no];
			JTb_block_tmp[ind_thread + 10] *= residual[f_no];
			JTb_block_tmp[ind_thread + 11] *= residual[f_no];
#endif
		}
	}

	__syncthreads();
	//reduction
	if (threadInd < 12) {
		for (int iter_redu = 0; iter_redu < residual_num; iter_redu++) {
			JTb_block_res[threadInd] += JTb_block_tmp[threadInd + iter_redu * 12];
		}
	}

	__syncthreads();
	// write_back
	if (threadInd < 12) {
		JTResidual[col * 12 + threadInd] -= JTb_block_res[threadInd];
	}
}

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
	float w_geo)
{
	//std::cout << w_geo << std::endl;
	//std::exit(0);
	int block = 160; // fixed !!!
	int grid = blockStatistics.m_nonZeroBlockNum; //each CUDA block handle one JTJ block.
	ComputeJTJGeoTermPointToPlainKernel << <grid, block >> > (
		JTJ.m_a,
		JTJ.m_ia,
		RAW_PTR(blockStatistics.m_equIndVecDevice),
		RAW_PTR(blockStatistics.m_equIndOffsetVecDevice),
		RAW_PTR(blockStatistics.m_nonZeroBlockInfoCooDevice),
		RAW_PTR(blockStatistics.m_columnIndEachRowDevice),
		matchingPointsDevice,
		matchingPointsNumTotal,
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		vertexToNodeIndicesDevice,
		vertexToNodeWeightsDevice,
		nodeVIndicesDevice,
		nodeNum,
		w_geo);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = 1024;
	grid = DivUp(matchingPointsNumTotal, block);
	ComputeResidualGeoTermPointToPlainKernel << <grid, block >> > (
		residual,
		matchingPointsDevice,
		matchingPointsNumTotal,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		w_geo);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = 256; // fixed!!!
	grid = blockStatistics.m_nonZeroBlockNum;
	ComputeJTResidualGeoTermPointToPlainKernel << <grid, block >> > (
		JTResidual,
		residual,
		RAW_PTR(blockStatistics.m_equIndVecDevice),
		RAW_PTR(blockStatistics.m_equIndOffsetVecDevice),
		RAW_PTR(blockStatistics.m_nonZeroBlockInfoCooDevice),
		matchingPointsDevice,
		matchingPointsNumTotal,
		originVertexPosesDevice,
		updatedVertexPosesDevice,
		updatedVertexNormalsDevice,
		vertexToNodeIndicesDevice,
		vertexToNodeWeightsDevice,
		nodeVIndicesDevice,
		nodeNum,
		w_geo);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__forceinline__ __device__ int FindIndex(int *node_list, int val, int num) {
	for (int i = 0; i < num; i++) {
		if (node_list[i] == val) {
			return i;
		}
	}
	return -1;
}

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
	float w_geo)
{
	__shared__ float Jci[160 * 12];
	__shared__ float Jcj[160 * 12];
	__shared__ float block_JTJ_res[12 * 12];

	if (threadIdx.x < 144) {
		block_JTJ_res[threadIdx.x] = 0;
	}
	__syncthreads();

	int row = nonZeroBlockInfoCoo[blockIdx.x].x;
	int col = nonZeroBlockInfoCoo[blockIdx.x].y;
	int num = nonZeroBlockInfoCoo[blockIdx.x].z;

#if 0
	if (row != 27 || col != 27)
		return;
	printf("num: %d\n", num);
#endif

	int f_no;
	int *f_set = equIndVec + equIndOffsetVec[row * nodeNum + col];
	float weight_V2N, tmp1, tmp2, tmp3;
	float4 vnn, nor_tar;
	int full_times = num / int(160);
	int residual_num = num % 160;
	int row_res, col_res;
	row_res = threadIdx.x / 12;
	col_res = threadIdx.x % 12;
	int ind_thread = threadIdx.x * 12;

	for (int iter_ft = 0; iter_ft < full_times; iter_ft++)
	{
		f_no = f_set[threadIdx.x + iter_ft * 160];
		int srcVertexInd = matchingPoints[2 * f_no];
		int targetVertexInd = matchingPoints[2 * f_no + 1];
		if (srcVertexInd < 0 || targetVertexInd < 0)
		{
#pragma unroll
			for (int iter = 0; iter < 12; iter++)
			{
				Jci[threadIdx.x * 12 + iter] = 0.0f;
			}
		}
		else
		{
			int nodeIndRow, nodeIndCol, vertexIndRow, vertexIndCol;
			float signRow = 1.0f, signCol = 1.0f;
			int nodeIndRow1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, row, 4);
			int nodeIndRow2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, row, 4);
			int nodeIndCol1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, col, 4);
			int nodeIndCol2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, col, 4);
			if ((row / NODE_NUM_EACH_FRAG) == (col / NODE_NUM_EACH_FRAG))
			{
				if (nodeIndRow1 > nodeIndRow2)
				{
					nodeIndRow = nodeIndRow1;
					nodeIndCol = nodeIndCol1;
					vertexIndRow = srcVertexInd;
					vertexIndCol = srcVertexInd;
				}
				else
				{
					nodeIndRow = nodeIndRow2;
					nodeIndCol = nodeIndCol2;
					vertexIndRow = targetVertexInd;
					vertexIndCol = targetVertexInd;
				}
			}
			else
			{
				if (nodeIndRow1 > nodeIndRow2)
				{
					nodeIndRow = nodeIndRow1;
					nodeIndCol = nodeIndCol2;
					vertexIndRow = srcVertexInd;
					vertexIndCol = targetVertexInd;
					signCol = -1.0f;
				}
				else
				{
					nodeIndRow = nodeIndRow2;
					nodeIndCol = nodeIndCol1;
					vertexIndRow = targetVertexInd;
					vertexIndCol = srcVertexInd;
					signRow = -1.0f;
				}
			}
#if 0
			printf("num: %d %d %d | %d %d %d %d | %d %d %d %d | %d %d %d %d | %d\n",
				num, row, col,
				srcVertexInd, targetVertexInd, nodeIndRow, nodeIndCol,
				nodeIndRow1, nodeIndRow2, nodeIndCol1, nodeIndCol2,
				nodeIndRow, nodeIndCol, vertexIndRow, vertexIndCol,
				columnIndEachRow[row * nodeNum + col]);
#endif	

			// compute Jcj first, col.
			nor_tar = updatedVertexNormals[targetVertexInd];
			nor_tar.w = 0.0f;
			nor_tar = normalize(nor_tar);
			weight_V2N = signCol * w_geo * vertexToNodeWeights[vertexIndCol * 4 + nodeIndCol];
			vnn = originVertexPoses[vertexIndCol] -
				originVertexPoses[nodeVIndices[vertexToNodeIndices[vertexIndCol * 4 + nodeIndCol]]];

			tmp1 = weight_V2N * nor_tar.x;
			tmp2 = weight_V2N * nor_tar.y;
			tmp3 = weight_V2N * nor_tar.z;

			Jcj[ind_thread++] = tmp1 * vnn.x;
			Jcj[ind_thread++] = tmp2 * vnn.x;
			Jcj[ind_thread++] = tmp3 * vnn.x;
			Jcj[ind_thread++] = tmp1 * vnn.y;
			Jcj[ind_thread++] = tmp2 * vnn.y;
			Jcj[ind_thread++] = tmp3 * vnn.y;
			Jcj[ind_thread++] = tmp1 * vnn.z;
			Jcj[ind_thread++] = tmp2 * vnn.z;
			Jcj[ind_thread++] = tmp3 * vnn.z;
			Jcj[ind_thread++] = tmp1;
			Jcj[ind_thread++] = tmp2;
			Jcj[ind_thread] = tmp3;

#if 0
			printf("|||%f %f %f %f %f %f %f %f %f %f %f %f\n",
				tmp1 * vnn.x, tmp1 * vnn.y, tmp1 * vnn.z,
				tmp2 * vnn.x, tmp2 * vnn.y, tmp2 * vnn.z,
				tmp3 * vnn.x, tmp3 * vnn.y, tmp3 * vnn.z,
				tmp1, tmp2, tmp3);
#endif

			// compute Jci, row
			weight_V2N = signRow * w_geo * vertexToNodeWeights[vertexIndRow * 4 + nodeIndRow];
			vnn = originVertexPoses[vertexIndRow] -
				originVertexPoses[nodeVIndices[vertexToNodeIndices[vertexIndRow * 4 + nodeIndRow]]];
			tmp1 = weight_V2N * nor_tar.x;
			tmp2 = weight_V2N * nor_tar.y;
			tmp3 = weight_V2N * nor_tar.z;

			Jci[ind_thread--] = tmp3;
			Jci[ind_thread--] = tmp2;
			Jci[ind_thread--] = tmp1;
			Jci[ind_thread--] = tmp3 * vnn.z;
			Jci[ind_thread--] = tmp2 * vnn.z;
			Jci[ind_thread--] = tmp1 * vnn.z;
			Jci[ind_thread--] = tmp3 * vnn.y;
			Jci[ind_thread--] = tmp2 * vnn.y;
			Jci[ind_thread--] = tmp1 * vnn.y;
			Jci[ind_thread--] = tmp3 * vnn.x;
			Jci[ind_thread--] = tmp2 * vnn.x;
			Jci[ind_thread] = tmp1 * vnn.x;
		}

		__syncthreads();
		//reduction
		if (threadIdx.x < 144) {
#pragma unroll
			for (int iter_redu = 0; iter_redu < 160; iter_redu++) {
				block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + row_res] * Jcj[iter_redu * 12 + col_res];
			}
		}

		__syncthreads();
	}

	//printf("hehe srcVertexInd targetVertexInd %d %d %d %d \n", row_res, col_res, threadIdx.x, residual_num);

	if (threadIdx.x < residual_num) {
		f_no = f_set[threadIdx.x + full_times * 160];
		int srcVertexInd = matchingPoints[2 * f_no];
		int targetVertexInd = matchingPoints[2 * f_no + 1];
		//printf("srcVertexInd targetVertexInd %d %d %d %d\n", srcVertexInd, targetVertexInd, row_res, col_res);
		if (srcVertexInd < 0 || targetVertexInd < 0)
		{
#pragma unroll
			for (int iter = 0; iter < 12; iter++) {
				Jci[threadIdx.x * 12 + iter] = 0.0f;
			}
		}
		else {
			int nodeIndRow, nodeIndCol, vertexIndRow, vertexIndCol;
			float signRow = 1.0f, signCol = 1.0f;
			int nodeIndRow1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, row, 4);
			int nodeIndRow2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, row, 4);
			int nodeIndCol1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, col, 4);
			int nodeIndCol2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, col, 4);
			//printf("%d %d %d %d\n", nodeIndRow1, nodeIndRow2, nodeIndCol1, nodeIndCol2);
			if ((row / NODE_NUM_EACH_FRAG) == (col / NODE_NUM_EACH_FRAG))
			{
				if (nodeIndRow1 > nodeIndRow2)
				{
					nodeIndRow = nodeIndRow1;
					nodeIndCol = nodeIndCol1;
					vertexIndRow = srcVertexInd;
					vertexIndCol = srcVertexInd;
				}
				else
				{
					nodeIndRow = nodeIndRow2;
					nodeIndCol = nodeIndCol2;
					vertexIndRow = targetVertexInd;
					vertexIndCol = targetVertexInd;
				}
			}
			else
			{
				if (nodeIndRow1 > nodeIndRow2)
				{
					nodeIndRow = nodeIndRow1;
					nodeIndCol = nodeIndCol2;
					vertexIndRow = srcVertexInd;
					vertexIndCol = targetVertexInd;
					signCol = -1.0f;
				}
				else
				{
					nodeIndRow = nodeIndRow2;
					nodeIndCol = nodeIndCol1;
					vertexIndRow = targetVertexInd;
					vertexIndCol = srcVertexInd;
					signRow = -1.0f;
			}
		}
#if 0
			printf("num: %d %d %d | %d %d %d %d | %d %d %d %d | %d %d %d %d | %d\n",
				num, row, col,
				srcVertexInd, targetVertexInd, nodeIndRow, nodeIndCol,
				nodeIndRow1, nodeIndRow2, nodeIndCol1, nodeIndCol2,
				nodeIndRow, nodeIndCol, vertexIndRow, vertexIndCol,
				columnIndEachRow[row * nodeNum + col]);
#endif	

			// compute Jcj first, col.
			nor_tar = updatedVertexNormals[targetVertexInd];
			nor_tar.w = 0.0f;
			nor_tar = normalize(nor_tar);
			weight_V2N = signCol * w_geo * vertexToNodeWeights[vertexIndCol * 4 + nodeIndCol];
			vnn = originVertexPoses[vertexIndCol] -
				originVertexPoses[nodeVIndices[vertexToNodeIndices[vertexIndCol * 4 + nodeIndCol]]];

			tmp1 = weight_V2N * nor_tar.x;
			tmp2 = weight_V2N * nor_tar.y;
			tmp3 = weight_V2N * nor_tar.z;

			Jcj[ind_thread++] = tmp1 * vnn.x;
			Jcj[ind_thread++] = tmp2 * vnn.x;
			Jcj[ind_thread++] = tmp3 * vnn.x;
			Jcj[ind_thread++] = tmp1 * vnn.y;
			Jcj[ind_thread++] = tmp2 * vnn.y;
			Jcj[ind_thread++] = tmp3 * vnn.y;
			Jcj[ind_thread++] = tmp1 * vnn.z;
			Jcj[ind_thread++] = tmp2 * vnn.z;
			Jcj[ind_thread++] = tmp3 * vnn.z;
			Jcj[ind_thread++] = tmp1;
			Jcj[ind_thread++] = tmp2;
			Jcj[ind_thread] = tmp3;

#if 0
			printf("|||%f %f %f %f %f %f %f %f %f %f %f %f\n",
				tmp1 * vnn.x, tmp1 * vnn.y, tmp1 * vnn.z,
				tmp2 * vnn.x, tmp2 * vnn.y, tmp2 * vnn.z,
				tmp3 * vnn.x, tmp3 * vnn.y, tmp3 * vnn.z,
				tmp1, tmp2, tmp3);
#endif

			// compute Jci, row
			weight_V2N = signRow * w_geo * vertexToNodeWeights[vertexIndRow * 4 + nodeIndRow];
			vnn = originVertexPoses[vertexIndRow] -
				originVertexPoses[nodeVIndices[vertexToNodeIndices[vertexIndRow * 4 + nodeIndRow]]];
			tmp1 = weight_V2N * nor_tar.x;
			tmp2 = weight_V2N * nor_tar.y;
			tmp3 = weight_V2N * nor_tar.z;

			Jci[ind_thread--] = tmp3;
			Jci[ind_thread--] = tmp2;
			Jci[ind_thread--] = tmp1;
			Jci[ind_thread--] = tmp3 * vnn.z;
			Jci[ind_thread--] = tmp2 * vnn.z;
			Jci[ind_thread--] = tmp1 * vnn.z;
			Jci[ind_thread--] = tmp3 * vnn.y;
			Jci[ind_thread--] = tmp2 * vnn.y;
			Jci[ind_thread--] = tmp1 * vnn.y;
			Jci[ind_thread--] = tmp3 * vnn.x;
			Jci[ind_thread--] = tmp2 * vnn.x;
			Jci[ind_thread] = tmp1 * vnn.x;
	}
}

	__syncthreads();
	//reduction
	if (threadIdx.x < 144) {
		for (int iter_redu = 0; iter_redu < residual_num; iter_redu++) {
			block_JTJ_res[threadIdx.x] += Jci[iter_redu * 12 + row_res] * Jcj[iter_redu * 12 + col_res];
		}
	}

	__syncthreads();
	// write JTJ_block(i,j) to global memory.
	int start_pos;
	if (threadIdx.x < 144) {
		if (row == col) {
			start_pos = columnIndEachRow[row * nodeNum + col] * 12;
			JTJ_a[JTJ_ia[row * 12 + row_res] + start_pos + col_res] += block_JTJ_res[row_res * 12 + col_res];
		}
		else {
			start_pos = columnIndEachRow[row * nodeNum + col] * 12;
			JTJ_a[JTJ_ia[row * 12 + row_res] + start_pos + col_res] += block_JTJ_res[row_res * 12 + col_res];
			start_pos = columnIndEachRow[col * nodeNum + row] * 12;
			JTJ_a[JTJ_ia[col * 12 + row_res] + start_pos + col_res] += block_JTJ_res[col_res * 12 + row_res];
		}
	}
}

__global__ void ComputeResidualGeoTermPointToPlainKernel(
	float *residual,
	int *matchingPoints,
	int matchingPointsNum,
	float4 *updatedVertexPoses,
	float4 *updatedVertexNormals,
	float w_geo)
{
	int matchingPairInd = blockDim.x * blockIdx.x + threadIdx.x;
	if (matchingPairInd >= matchingPointsNum)
	{
		return;
	}

	int srcVertexInd = matchingPoints[2 * matchingPairInd];
	int targetVertexInd = matchingPoints[2 * matchingPairInd + 1];

	if (srcVertexInd < 0 || targetVertexInd < 0)
	{
		residual[matchingPairInd] = 0;
		return;
	}

	float4 updatedVertexNormalTarget = updatedVertexNormals[targetVertexInd];
	float4 residualDiff = updatedVertexPoses[srcVertexInd] - updatedVertexPoses[targetVertexInd];

	residual[matchingPairInd] = w_geo
		* (updatedVertexNormalTarget.x * residualDiff.x
			+ updatedVertexNormalTarget.y * residualDiff.y
			+ updatedVertexNormalTarget.z * residualDiff.z);
}

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
	float w_geo)
{
	int blockInd = blockIdx.x;
	int row = nonZeroBlockInfoCoo[blockInd].x;
	int col = nonZeroBlockInfoCoo[blockInd].y;
	if (row != col) {
		return;
	}
	__shared__ float JTb_block_tmp[12 * 256];
	__shared__ float JTb_block_res[12];
	int threadInd = threadIdx.x;
	if (threadInd < 12) {
		JTb_block_res[threadInd] = 0;
	}
	int num = nonZeroBlockInfoCoo[blockInd].z;
	int nodeInd = col;

	int full_times = num / 256;
	int residual_num = num % 256;
	int f_no = 0, corr_no = 0, node_offset = 0;
	int *f_set = equIndVec + equIndOffsetVec[row * nodeNum + col];
	float4 nor_tar, node_pos = originVertexPoses[nodeVIndices[nodeInd]], vnn;
	float weight_V2N = 0.0, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;
	int srcVertexInd, targetVertexInd;
	int nodeIndCol1, nodeIndCol2, nodeIndCol, vertexIndCol, signCol = 1.0f;
	for (int iter_ft = 0; iter_ft < full_times; iter_ft++) {
		f_no = f_set[threadInd + iter_ft * 256];
		//printf("%d %d %d %d %d\n", row, col, f_no, srcVertexInd, targetVertexInd);
		srcVertexInd = matchingPoints[2 * f_no];
		targetVertexInd = matchingPoints[2 * f_no + 1];
		if (srcVertexInd < 0 || targetVertexInd < 0)
		{
			JTb_block_tmp[threadInd * 12 + 0] = 0;
			JTb_block_tmp[threadInd * 12 + 1] = 0;
			JTb_block_tmp[threadInd * 12 + 2] = 0;
			JTb_block_tmp[threadInd * 12 + 3] = 0;
			JTb_block_tmp[threadInd * 12 + 4] = 0;
			JTb_block_tmp[threadInd * 12 + 5] = 0;
			JTb_block_tmp[threadInd * 12 + 6] = 0;
			JTb_block_tmp[threadInd * 12 + 7] = 0;
			JTb_block_tmp[threadInd * 12 + 8] = 0;
			JTb_block_tmp[threadInd * 12 + 9] = 0;
			JTb_block_tmp[threadInd * 12 + 10] = 0;
			JTb_block_tmp[threadInd * 12 + 11] = 0;
		}
		else {

			nodeIndCol1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, col, 4);
			nodeIndCol2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, col, 4);
			if (nodeIndCol1 > nodeIndCol2)
			{
				nodeIndCol = nodeIndCol1;
				vertexIndCol = srcVertexInd;
				signCol = 1.0f;
			}
			else
			{
				nodeIndCol = nodeIndCol2;
				vertexIndCol = targetVertexInd;
				signCol = -1.0f;
			}

			nor_tar = updatedVertexNormals[targetVertexInd];
			weight_V2N = signCol * w_geo * vertexToNodeWeights[vertexIndCol * 4 + nodeIndCol] * residual[f_no];
			vnn = originVertexPoses[vertexIndCol] -
				node_pos;

			tmp1 = weight_V2N * nor_tar.x;
			tmp2 = weight_V2N * nor_tar.y;
			tmp3 = weight_V2N * nor_tar.z;

			JTb_block_tmp[threadInd * 12 + 0] = tmp1 * vnn.x;
			JTb_block_tmp[threadInd * 12 + 1] = tmp2 * vnn.x;
			JTb_block_tmp[threadInd * 12 + 2] = tmp3 * vnn.x;

			JTb_block_tmp[threadInd * 12 + 3] = tmp1 * vnn.y;
			JTb_block_tmp[threadInd * 12 + 4] = tmp2 * vnn.y;
			JTb_block_tmp[threadInd * 12 + 5] = tmp3 * vnn.y;

			JTb_block_tmp[threadInd * 12 + 6] = tmp1 * vnn.z;
			JTb_block_tmp[threadInd * 12 + 7] = tmp2 * vnn.z;
			JTb_block_tmp[threadInd * 12 + 8] = tmp3 * vnn.z;

			JTb_block_tmp[threadInd * 12 + 9] = tmp1;
			JTb_block_tmp[threadInd * 12 + 10] = tmp2;
			JTb_block_tmp[threadInd * 12 + 11] = tmp3;
		}

		__syncthreads();
		//reduction
		if (threadInd < 12)
		{
			for (int iter_redu = 0; iter_redu < 256; iter_redu++)
			{
				JTb_block_res[threadInd] += JTb_block_tmp[threadInd + iter_redu * 12];
			}
		}
		__syncthreads();
	}

	__syncthreads();
	if (threadInd < residual_num) {
		f_no = f_set[threadInd + full_times * 256];
		srcVertexInd = matchingPoints[2 * f_no];
		targetVertexInd = matchingPoints[2 * f_no + 1];
		//printf("%d %d %d %d %d\n", row, col, f_no, srcVertexInd, targetVertexInd);
		if (srcVertexInd < 0 || targetVertexInd < 0)
		{
			JTb_block_tmp[threadInd * 12 + 0] = 0;
			JTb_block_tmp[threadInd * 12 + 1] = 0;
			JTb_block_tmp[threadInd * 12 + 2] = 0;
			JTb_block_tmp[threadInd * 12 + 3] = 0;
			JTb_block_tmp[threadInd * 12 + 4] = 0;
			JTb_block_tmp[threadInd * 12 + 5] = 0;
			JTb_block_tmp[threadInd * 12 + 6] = 0;
			JTb_block_tmp[threadInd * 12 + 7] = 0;
			JTb_block_tmp[threadInd * 12 + 8] = 0;
			JTb_block_tmp[threadInd * 12 + 9] = 0;
			JTb_block_tmp[threadInd * 12 + 10] = 0;
			JTb_block_tmp[threadInd * 12 + 11] = 0;
		}
		else {
			nodeIndCol1 = FindIndex(vertexToNodeIndices + srcVertexInd * 4, col, 4);
			nodeIndCol2 = FindIndex(vertexToNodeIndices + targetVertexInd * 4, col, 4);
			if (nodeIndCol1 > nodeIndCol2)
			{
				nodeIndCol = nodeIndCol1;
				vertexIndCol = srcVertexInd;
				signCol = 1.0f;
			}
			else
			{
				nodeIndCol = nodeIndCol2;
				vertexIndCol = targetVertexInd;
				signCol = -1.0f;
			}
			//printf("%d %d\n", vertexToNodeIndices[vertexIndCol * 4 + nodeIndCol], nodeInd);

			nor_tar = updatedVertexNormals[targetVertexInd];
			weight_V2N = signCol * w_geo * vertexToNodeWeights[vertexIndCol * 4 + nodeIndCol] * residual[f_no];
			vnn = originVertexPoses[vertexIndCol] -
				node_pos;

			tmp1 = weight_V2N * nor_tar.x;
			tmp2 = weight_V2N * nor_tar.y;
			tmp3 = weight_V2N * nor_tar.z;
			JTb_block_tmp[threadInd * 12 + 0] = tmp1 * vnn.x;
			JTb_block_tmp[threadInd * 12 + 1] = tmp2 * vnn.x;
			JTb_block_tmp[threadInd * 12 + 2] = tmp3 * vnn.x;

			JTb_block_tmp[threadInd * 12 + 3] = tmp1 * vnn.y;
			JTb_block_tmp[threadInd * 12 + 4] = tmp2 * vnn.y;
			JTb_block_tmp[threadInd * 12 + 5] = tmp3 * vnn.y;

			JTb_block_tmp[threadInd * 12 + 6] = tmp1 * vnn.z;
			JTb_block_tmp[threadInd * 12 + 7] = tmp2 * vnn.z;
			JTb_block_tmp[threadInd * 12 + 8] = tmp3 * vnn.z;

			JTb_block_tmp[threadInd * 12 + 9] = tmp1;
			JTb_block_tmp[threadInd * 12 + 10] = tmp2;
			JTb_block_tmp[threadInd * 12 + 11] = tmp3;
		}
	}

	__syncthreads();
	//reduction
	if (threadInd < 12) {
		for (int iter_redu = 0; iter_redu < residual_num; iter_redu++) {
			JTb_block_res[threadInd] += JTb_block_tmp[threadInd + iter_redu * 12];
		}
	}

	__syncthreads();
	// write_back
	if (threadInd < 12) {
		JTResidual[col * 12 + threadInd] -= JTb_block_res[threadInd];
	}
}

