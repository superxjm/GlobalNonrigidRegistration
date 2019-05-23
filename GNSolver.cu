#include "GNSolver.h"

//#include "DataPointPlane.h"
//#include "SmoothConstraint.h"
#include "Constraint.h"

#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include "InputData.h"
#include "Helpers/InnorealTimer.hpp"
#include "GeoConstraint.h"
#include "PhotoConstraint.h"
#include "SmoothConstraint.h"
#include "RotConstraint.h"
#include "PcgSolver.h"
#include "Helpers/UtilsMath.h"

__global__ void InitVarkernel(float* vars, int varNum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < varNum)
	{
		vars[idx] = (idx % 4 == 0) ? 1.0f : 0.0f;
	}
}

GNSolver::GNSolver(InputData* input_data, const SolverPara& para)
	: m_inputData{nullptr}, m_pcgLinearSolver{nullptr}, m_varsNum{0}, m_iter{0}, m_residual{0}
{
	assert(input_data);
	m_inputData = input_data;
	initCons(para);

	m_dInitVarPatternVec.resize(NODE_NUM_EACH_FRAG * 12);
	int block = 256;
	int grid = DivUp(NODE_NUM_EACH_FRAG * 12, block);
	InitVarkernel << < grid, block >> >(RAW_PTR(m_dInitVarPatternVec), NODE_NUM_EACH_FRAG * 12);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	m_maxVarNum = MAX_FRAG_NUM * NODE_NUM_EACH_FRAG * 12;
	m_dJTb.resize(m_maxVarNum);
	m_dPreconditioner.resize(m_maxVarNum);
	m_dVars.resize(m_maxVarNum);
	m_dRsTransInv.resize(m_maxVarNum);
	m_dDelta.resize(m_maxVarNum);
	m_dCameraToNodeWeightVec.resize(m_inputData->m_source.m_maxNodeNum);

	m_pcgLinearSolver = new PcgLinearSolver;
}

GNSolver::~GNSolver()
{
	std::vector<Constraint *>::iterator it;
	for (it = m_cons.begin(); it != m_cons.end(); ++it)
	{
		delete *it;
	}

	m_dInitVarPatternVec.clear();
	m_dJTb.clear();
	m_dPreconditioner.clear();
	m_dVars.clear();
	m_dRsTransInv.clear();
	m_dDelta.clear();
	m_dCameraToNodeWeightVec.clear();

	if (m_pcgLinearSolver != nullptr) delete m_pcgLinearSolver;
}

bool GNSolver::initVars()
{
#if 1
	m_varsNum = m_inputData->getVarsNum(); // 12 variables each
	m_dJTJ.m_row = m_varsNum;
	m_dJTJ.m_col = m_varsNum;

	if (m_varsNum > m_maxVarNum)
	{
		std::cout << "error: var num exceed limit" << std::endl;
		std::exit(0);
	}

	//timer.TimeStart();
	//initialize variables
	const int varsNumEachFrag = m_inputData->getVarsNumEachFrag();
	//std::cout << varsNumEachFrag << std::endl;
	//std::cout << m_varsNum << std::endl;
	//std::cout << m_dInitVarPatternVec.size() << std::endl;
	checkCudaErrors(cudaMemcpy(RAW_PTR(m_dVars) + m_varsNum - varsNumEachFrag, RAW_PTR(m_dInitVarPatternVec),
		varsNumEachFrag * sizeof(float), cudaMemcpyDeviceToDevice));
#if 0
	int block = 512, grid = (m_varsNum + block - 1) / block;
	InitVarkernel << < grid, block >> > (RAW_PTR(m_dVars), m_varsNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
#endif
	//timer.TimeEnd();
	//printf("time of init vars: %f \n", timer.TimeGap_in_ms());

	m_pcgLinearSolver->init(m_varsNum);
#endif

	return true;
}

bool GNSolver::initCons()
{
	for (int i = 0; i < m_cons.size(); ++i)
	{
		m_cons[i]->init();
	}
	return true;
}

bool GNSolver::initCons(const SolverPara& para)
{
	m_cons.clear();
	if (para.m_hasDataCons1)
	{
		GeoConstraint* pNPC = new GeoConstraint;
		pNPC->init(this, para.m_dataConsWeight1);
		m_cons.push_back(pNPC);
	}

	if (para.m_hasDataCons2)
	{
		PhotoConstraint* pNPC = new PhotoConstraint;
		pNPC->init(this, para.m_dataConsWeight2);
		m_cons.push_back(pNPC);
	}

	if (para.m_hasSmoothCons)
	{
		SmoothConstraint* pNPC = new SmoothConstraint;
		pNPC->init(this, para.m_smoothConsWeight);
		m_cons.push_back(pNPC);
	}

	if (para.m_hasRotCons)
	{
		RotConstraint* pNPC = new RotConstraint;
		pNPC->init(this, para.m_rotConsWeight);
		m_cons.push_back(pNPC);
	}

	return true;
}

struct InitJaOfJTJWrapper
{
	int num_node;
	int num_nnz_block;

	int* jtj_ja;
	int* coo;
	int* nnz_pre;
	int* row_ptr;

	__device__ void operator()()
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < num_nnz_block)
		{
			// 填充下三角中的非零块
			int index = coo[idx];
			int seri_i = index / num_node;
			int seri_j = index - seri_i * num_node;
			int num_pre_row = nnz_pre[index];
			int num_pre_all = row_ptr[seri_i];
			int num_nnz_this_row = row_ptr[seri_i + 1] - num_pre_all;
			for (int iter_row = 0; iter_row < 12; ++iter_row)
			{
				for (int iter_col = 0; iter_col < 12; ++iter_col)
				{
					jtj_ja[num_pre_all * 144 + iter_row * num_nnz_this_row * 12 + num_pre_row * 12 + iter_col] = seri_j * 12 +
						iter_col;
				}
			}
			// 填充上三角中的非零块
			int tmp = seri_i;
			seri_i = seri_j;
			seri_j = tmp;
			index = seri_i * num_node + seri_j;
			num_pre_row = nnz_pre[index];
			num_pre_all = row_ptr[seri_i];
			num_nnz_this_row = row_ptr[seri_i + 1] - num_pre_all;
			for (int iter_row = 0; iter_row < 12; ++iter_row)
			{
				for (int iter_col = 0; iter_col < 12; ++iter_col)
				{
					jtj_ja[num_pre_all * 144 + iter_row * num_nnz_this_row * 12 + num_pre_row * 12 + iter_col] = seri_j * 12 +
						iter_col;
				}
			}
		}
	}
};

__global__ void InitJaOfJTJKernel(InitJaOfJTJWrapper ijoj_wrapper)
{
	ijoj_wrapper();
}

void GNSolver::initJtj()
{
	int num_node = m_inputData->m_source.m_nodeNum;
	int num_nnz_block = m_inputData->m_Iij.m_nnzIij; // 下三角中的非零元素个数.
	int all_nnz = num_nnz_block * 2 - num_node; // 所有的非零元素个数。
	m_dJTJ.m_nnz = all_nnz * 144;
	m_dJTJ.m_dJa.resize(m_dJTJ.m_nnz);
	m_dJTJ.m_dIa.resize(num_node * 12 + 1);
	m_dJTJ.m_dA.resize(m_dJTJ.m_nnz);

	// 在cpu端填充jtj_ia，与GPU端填充jtj_ja并行。
	thrust::host_vector<int> ia(num_node * 12 + 1);
	thrust::host_vector<int> h_row_ptr = m_inputData->m_Iij.m_dRowPtr;

	InitJaOfJTJWrapper ijoj_wrapper;
	ijoj_wrapper.num_node = num_node;
	ijoj_wrapper.num_nnz_block = num_nnz_block;
	ijoj_wrapper.jtj_ja = RAW_PTR(m_dJTJ.m_dJa);
	ijoj_wrapper.coo = RAW_PTR(m_inputData->m_Iij.m_dNzIijCoo);
	ijoj_wrapper.nnz_pre = RAW_PTR(m_inputData->m_Iij.m_dNnzPre);
	ijoj_wrapper.row_ptr = RAW_PTR(m_inputData->m_Iij.m_dRowPtr);
	int block_size = 256;
	int grid_size = (num_nnz_block + block_size - 1) / block_size;
	InitJaOfJTJKernel << < grid_size, block_size >> >(ijoj_wrapper);

	int offset = 0;
	int nnz_each_row;
	int counter = 0;
	for (int iter = 0; iter < num_node; ++iter)
	{
		nnz_each_row = (h_row_ptr[iter + 1] - h_row_ptr[iter]) * 12;
		for (int iter_inner = 0; iter_inner < 12; ++iter_inner)
		{
			ia[counter] = offset;
			counter++;
			offset += nnz_each_row;
		}
	}
	ia[counter] = offset;
	m_dJTJ.m_dIa = ia;
	assert(counter == ia.size() - 1);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void ExtractPreconditioner(float* preCondTerms, int* ia, int* ja, float* a, int rowJTJ)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= rowJTJ)
	{
		return;
	}

	for (int i = ia[idx]; i < ia[idx + 1]; ++i)
	{
		if (idx == ja[i])
		{
            //a[i] += 0.001f;
			preCondTerms[idx] = 1.0 / a[i];// (a[i] + 1e-24);
			return;
		}
	}
}

static void AddDeltaVarstoVars(thrust::device_vector<float> &dVars, thrust::device_vector<float> &dDeltaVars, int length)
{
	thrust::plus<float> opPlus;
#if 0
	thrust::transform(dVars.begin(), dVars.begin() + length, dDeltaVars.begin(), dVars.begin(), opPlus);
#else
	thrust::transform(dVars.begin() + NODE_NUM_EACH_FRAG * 12, dVars.begin() + length,
	                  dDeltaVars.begin() + NODE_NUM_EACH_FRAG * 12, dVars.begin() + NODE_NUM_EACH_FRAG * 12, opPlus);
#endif
}

static __global__ void CalcInvTransRotKernal(float *dVars, float *dRsTransInv, int length)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= length)
		return;

	float *R = dVars + 12 * idx;
	float *R_transinv = dRsTransInv + 9 * idx;

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
}

static void CalcInvTransRot(float *dRsTransInv, float *dVars, int length)
{
	int block = 256;
	int grid = DivUp(length, block);
	CalcInvTransRotKernal << <grid, block >> > (dVars, dRsTransInv, length);
}

static __global__ void UpdateVertexPosesAndNormalsKernel(float4* dDeformedVertexVec,
                                                         float4 * dDeformedNormalVec,
                                                         float4 * dSrcdVertexVec,
                                                         float4 * dSrcNormalVec,
                                                         float3 * dVars,
                                                         float3 * dRsTransInv,
                                                         int vertexNum,
                                                         float4* dNodeVec,
                                                         int* vertexToNodeIndicesDevice,
                                                         float* vertexToNodeWeightsDevice)
{
	int vertexInd = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexInd >= vertexNum)
		return;

	int *nearIdxvec = vertexToNodeIndicesDevice + vertexInd * MAX_NEAR_NODE_NUM_VERTEX;
	float *nearWeightVec = vertexToNodeWeightsDevice + vertexInd * MAX_NEAR_NODE_NUM_VERTEX;
	int nearIdx;

	float4 vertexPosFragInd = dSrcdVertexVec[vertexInd];
	float4 vertexNormal = dSrcNormalVec[vertexInd];
	float3 *Rt;
	float3 *RTransinv;
	float4 newVertexPosFragInd = make_float4(0.0, 0.0, 0.0, 0.0);
	float4 newVertexNormal = make_float4(0.0, 0.0, 0.0, 0.0);

	float3 nearNodePose, nearNodeNormal;
	float nearWeight;

	for (int n = 0; n < MAX_NEAR_NODE_NUM_VERTEX; ++n)
	{
		nearIdx = *(nearIdxvec + n);
		nearWeight = *(nearWeightVec + n);
		nearNodePose = make_float3(dNodeVec[nearIdx]);
		Rt = dVars + nearIdx * 4;
		RTransinv = dRsTransInv + nearIdx * 3;

		newVertexPosFragInd += make_float4(nearWeight * (Rt[0] * (vertexPosFragInd.x - nearNodePose.x) +
			Rt[1] * (vertexPosFragInd.y - nearNodePose.y) +
			Rt[2] * (vertexPosFragInd.z - nearNodePose.z) +
			nearNodePose +
			Rt[3]));

		newVertexNormal += make_float4(nearWeight * (RTransinv[0] * vertexNormal.x +
			RTransinv[1] * vertexNormal.y +
			RTransinv[2] * vertexNormal.z));
	}

	newVertexPosFragInd.w = (dDeformedVertexVec + vertexInd)->w;
	newVertexNormal = normalize(newVertexNormal);
	*(dDeformedVertexVec + vertexInd) = newVertexPosFragInd;
	*(dDeformedNormalVec + vertexInd) = newVertexNormal;
	/*
	assert(!isnan(vertexPos.x));
	assert(!isnan(vertexPos.y));
	assert(!isnan(vertexPos.z));
	assert(!isnan(vertexNormal.x));
	assert(!isnan(vertexNormal.y));
	assert(!isnan(vertexNormal.z));
	*/
}

static void UpdateVertexPosesAndNormals(float4* dDeformedVertexVec,
                                        float4* dDeformedNormalVec,
                                        float4* dSrcdVertexVec,
                                        float4* dSrcNormalVec,
                                        float3* dVars,
                                        float3* dRsTransInv,
                                        int vertexNum,
                                        float4* dNodeVec,
                                        int* vertexToNodeIndicesDevice,
                                        float* vertexToNodeWeightsDevice)
{
	int block = 256, grid;
	grid = DivUp(vertexNum, block);
	UpdateVertexPosesAndNormalsKernel << <grid, block >> >(dDeformedVertexVec,
	                                                              dDeformedNormalVec,
	                                                              dSrcdVertexVec,
	                                                              dSrcNormalVec,
	                                                              dVars,
	                                                              dRsTransInv,
	                                                              vertexNum,
	                                                              dNodeVec,
	                                                              vertexToNodeIndicesDevice,
	                                                              vertexToNodeWeightsDevice);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

static __global__ void ApplyUpdatedVertexPosesAndNormalsKernel(VBOType* dVboVec,
                                                        float4* dDeformedVertexVec,
                                                        float4* dDeformedNormalVec,
                                                        int vertexNum,
                                                        uchar* dKeyColorImgs, int keyColorImgsStep,
                                                        float4* m_dKeyPosesInv,
                                                        float fx, float fy, float cx, float cy, int width, int height)
{
	int vertexIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexIdx >= vertexNum)
		return;

	float4 updatedPos = dDeformedVertexVec[vertexIdx];
	float4 updatedNormal = dDeformedNormalVec[vertexIdx];

	dVboVec[vertexIdx].posConf.x = updatedPos.x;
	dVboVec[vertexIdx].posConf.y = updatedPos.y;
	dVboVec[vertexIdx].posConf.z = updatedPos.z;
	dVboVec[vertexIdx].normalRad.x = updatedNormal.x;
	dVboVec[vertexIdx].normalRad.y = updatedNormal.y;
	dVboVec[vertexIdx].normalRad.z = updatedNormal.z;

	if (dKeyColorImgs != nullptr)
	{
		int fragInd = dVboVec[vertexIdx].colorTime.y;
		float4 *keyPoseInv = m_dKeyPosesInv + 4 * fragInd;
		uchar *keyColorImg = dKeyColorImgs + keyColorImgsStep * fragInd;

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
		dVboVec[vertexIdx].colorTime.x = rgb;
	}
}

void GNSolver::updateVboVec(VBOType* dVboCuda)
{
	int width = Resolution::getInstance().width(), height = Resolution::getInstance().height();
	int vertexNum = m_inputData->m_source.m_vertexNum;
	int block = 256;
	int grid = DivUp(vertexNum, block);
	ApplyUpdatedVertexPosesAndNormalsKernel << <grid, block >> >(dVboCuda,
	                                                             RAW_PTR(m_inputData->m_deformed.m_dVertexVec),
	                                                             RAW_PTR(m_inputData->m_deformed.m_dNormalVec),
	                                                             vertexNum,
	                                                             m_inputData->m_dKeyColorImgs, width * height * 3,
	                                                             m_inputData->m_dUpdatedKeyPosesInv,
	                                                             Intrinsics::getInstance().fx(),
	                                                             Intrinsics::getInstance().fy(),
	                                                             Intrinsics::getInstance().cx(),
	                                                             Intrinsics::getInstance().cy(),
	                                                             width, height);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

static __global__ void UpdateCameraNodeWeightKernel(float* dCameraToNodeWeights,
                                                    float4* dNodeVec,
                                                    float4* dKeyPoses,
                                                    int nodeNum)
{
	int nodeIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (nodeIdx >= nodeNum)
		return;

	float4 vertexPosFragIdx = dNodeVec[nodeIdx];
	int fragIdx = (int)vertexPosFragIdx.w;

	float4 cameraPose = dKeyPoses[fragIdx * 4 + 3];

	float xDiff = vertexPosFragIdx.x - cameraPose.x;
	float yDiff = vertexPosFragIdx.y - cameraPose.y;
	float zDiff = vertexPosFragIdx.z - cameraPose.z;

	dCameraToNodeWeights[nodeIdx] = sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff); //1.0f;
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
static __global__ void UpdateCameraPosesKernel(float4* dUpdatedKeyPoses,
                                               float4* dUpdatedKeyPosesInv,
                                               float3* dVars,
                                               float* dCameraToNodeWeightVec,
                                               float4* dNodeVec,
                                               float4* dKeyPoses,
                                               int nodeNum)
{
	int tid = threadIdx.x;
	int nodeIdx = threadIdx.x + blockIdx.x * blockDim.x;
	//__shared__ float sharedBuf[NODE_NUM_EACH_FRAG];
	float* sharedBuf = (float*)exSharedBuf;

	if (nodeIdx >= nodeNum)
		return;

	float4 vertexPosFragIdx = dNodeVec[nodeIdx];
	int fragInd = (int)vertexPosFragIdx.w;
	float3 nodePosition = make_float3(vertexPosFragIdx);

	float3 *Rt = dVars + nodeIdx * 4;
	float3 nodeRt[4] = { Rt[0], Rt[1], Rt[2], Rt[3] };

	float4 *cameraPos = dKeyPoses + fragInd * 4 + 3;
	float4 *cameraPose = dKeyPoses + fragInd * 4;

	float weight;

	// Make the weight Gaussian
	weight = dCameraToNodeWeightVec[nodeIdx];
	sharedBuf[tid] = weight;
	float mean = reduce64(sharedBuf, tid) / NODE_NUM_EACH_FRAG;

	float varianceInv = 1.0 / (0.5 * mean * mean); // variance of gaussian		
	weight = exp(-weight* weight * varianceInv);
	sharedBuf[tid] = weight;
	float sum = reduce64(sharedBuf, tid);
	weight = weight / sum;

	float3 newPos;
	float3 newRot[3];

	float3 weightedPos = weight *
	(nodeRt[0] * (cameraPos->x - vertexPosFragIdx.x) +
		nodeRt[1] * (cameraPos->y - vertexPosFragIdx.y) +
		nodeRt[2] * (cameraPos->z - vertexPosFragIdx.z) +
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
		float4 *updatedKeyPose = dUpdatedKeyPoses + fragInd * 4;
		float4 *updatedKeyPoseInv = dUpdatedKeyPosesInv + fragInd * 4;

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

void GNSolver::updateCameraPoses()
{
	int nodeNum = m_inputData->m_source.m_nodeNum;
	int block = 256;
	int grid = DivUp(nodeNum, block);
	UpdateCameraNodeWeightKernel << <grid, block >> >(RAW_PTR(m_dCameraToNodeWeightVec),
	                                                  RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                                  m_inputData->m_dKeyPoses,
	                                                  nodeNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	block = NODE_NUM_EACH_FRAG;
	grid = DivUp(nodeNum, NODE_NUM_EACH_FRAG);
	UpdateCameraPosesKernel << <grid, block,NODE_NUM_EACH_FRAG*sizeof(float) >> >(m_inputData->m_dUpdatedKeyPoses,
	                                             m_inputData->m_dUpdatedKeyPosesInv,
	                                             (float3*)RAW_PTR(m_dVars),
	                                             RAW_PTR(m_dCameraToNodeWeightVec),
	                                             RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                             m_inputData->m_dKeyPoses,
	                                             nodeNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	int fragIdx = m_inputData->m_source.m_fragNum - 1;
	std::vector<xMatrix4f> updatedKeyPoses(fragIdx + 1),
		updatedKeyPosesSVD(fragIdx + 1),
		updatedKeyPosesInvSVD(fragIdx + 1);
	checkCudaErrors(cudaMemcpy(updatedKeyPoses.data(),
		m_inputData->m_dUpdatedKeyPoses, sizeof(float4) * 4 * (fragIdx + 1), cudaMemcpyDeviceToHost));
#pragma omp for
	for (int i = 0; i < updatedKeyPoses.size(); ++i)
	{
		updatedKeyPosesSVD[i] = updatedKeyPoses[i].orthogonalization();
		updatedKeyPosesInvSVD[i] = updatedKeyPosesSVD[i].inverse();
		m_inputData->m_keyPoses[i] = updatedKeyPosesSVD[i];
	}
	checkCudaErrors(cudaMemcpy(m_inputData->m_dUpdatedKeyPoses,
		updatedKeyPosesSVD.data(), sizeof(float4) * 4 * (fragIdx + 1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_inputData->m_dUpdatedKeyPosesInv,
		updatedKeyPosesInvSVD.data(), sizeof(float4) * 4 * (fragIdx + 1), cudaMemcpyHostToDevice));
#if 0
	for (int i = 0; i < updatedKeyPoses.size(); ++i)
	{
		updatedKeyPosesSVD[i].print();
		updatedKeyPosesInvSVD[i].print();
	}
#endif
}

__global__ void ToMatKernel(int* _ia, int* _ja, float* _da, float* _mat, int _N, int _width, int _height)
{
	int u = blockDim.x * blockIdx.x + threadIdx.x;

	if (u >= _N) return;

	int num = _ia[u + 1] - _ia[u];
	int offset = _ia[u];
	for (int i = 0; i < num; i++)
	{
		_mat[u*_width + _ja[offset + i]] = _da[offset + i];
	}
}

cv::Mat ToMat(SparseMatrixCsrGpu* _sm)
{
	thrust::device_vector<float> d_mat(_sm->m_row*_sm->m_col, 0);
	int block = 256;
	int grid = (block + _sm->m_row - 1) / block;
	ToMatKernel << <grid, block >> > (RAW_PTR(_sm->m_dIa), RAW_PTR(_sm->m_dJa), RAW_PTR(_sm->m_dA), RAW_PTR(d_mat),
		_sm->m_row, _sm->m_col, _sm->m_row);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	cv::Mat h_mat(_sm->m_row, _sm->m_col, CV_32FC1);
	checkCudaErrors(cudaMemcpy(h_mat.data, RAW_PTR(d_mat), _sm->m_row*_sm->m_col * sizeof(float), cudaMemcpyDeviceToHost));
	return h_mat;
}

bool GNSolver::next(int iter)
{
	innoreal::InnoRealTimer timer;
	//timer.TimeStart();
	checkCudaErrors(cudaMemset(RAW_PTR(m_dDelta), 0, m_dDelta.size() * sizeof(float)));
	checkCudaErrors(cudaMemset(RAW_PTR(m_dJTb), 0, m_dJTb.size() * sizeof(float)));
	checkCudaErrors(cudaMemset(RAW_PTR(m_dJTJ.m_dA), 0, m_dJTJ.m_dA.size() * sizeof(float)));
	//timer.TimeEnd();
	//printf("time of reset JTb,JTJ,delta: %f ms \n", timer.TimeGap_in_ms());
	// Accumulate JTJ and JTb
	std::vector<Constraint *>::iterator it;
	double totalTime = 0.0;

	//timer.TimeStart();
	for (it = m_cons.begin(); it != m_cons.end(); ++it)
	{
		Constraint* pCon = *it;
		pCon->m_iter = iter;
		//timer.TimeStart();
		pCon->getJTJAndJTb(RAW_PTR(m_dJTJ.m_dA), RAW_PTR(m_dJTJ.m_dIa), m_dJTb,
		                   reinterpret_cast<float3*>(RAW_PTR(m_dVars)));
		//timer.TimeEnd();
		//totalTime += timer.TimeGap_in_ms();
		//printf("time of JTJ in %s : %f ms \n", pCon->ctype(), timer.TimeGap_in_ms());
	}
	//timer.TimeEnd();
	//std::cout << "JTJ time: " << timer.TimeGap_in_ms() << std::endl;


#if 0
	//cv::Mat tmp = ToMat(&m_dJTJ);
	//cv::Mat tmp_eigenvalue, tmp_eigenvector;
	//cv::eigen(tmp, tmp_eigenvalue, tmp_eigenvector);

	float jtb_sum = thrust::reduce(m_dJTb.begin(), m_dJTb.end());
	std::cout << "Jtb sum: " << jtb_sum << "\n";
	std::cout << "Nan JtJ Num: " << CheckNanVertex(RAW_PTR(m_dJTJ.m_dA), m_dJTJ.m_dA.size()) << "\n";
	std::cout << "Nan Vars Num: " << CheckNanVertex(RAW_PTR(m_dVars), m_varsNum) << "\n";
	std::cout << "Nan Delta Num: " << CheckNanVertex(RAW_PTR(m_dDelta), m_varsNum) << "\n";
	thrust::copy(m_dVars.begin(), m_dVars.begin() + 20, std::ostream_iterator<float>(std::cout, " ")); printf("\n");
	thrust::copy(m_dDelta.begin(), m_dDelta.begin() + 20, std::ostream_iterator<float>(std::cout, " ")); printf("\n");

	float val;
	for (it = m_cons.begin(); it != m_cons.end(); ++it) {
		Constraint *pCon = *it;
		val = pCon->val(reinterpret_cast<float3*>(RAW_PTR(m_dVars)));
		printf("before opt: GPU,%s:%.10f \n", pCon->ctype(), val);
		//m_residual += val;
	}
#endif

	int block_size = 512;
	int grid_size = (m_varsNum + block_size - 1) / block_size;
	ExtractPreconditioner << < grid_size, block_size >> >(RAW_PTR(m_dPreconditioner),
	                                                      RAW_PTR(m_dJTJ.m_dIa),
	                                                      RAW_PTR(m_dJTJ.m_dJa),
	                                                      RAW_PTR(m_dJTJ.m_dA),
	                                                      m_varsNum);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	int nnzSize = (m_inputData->m_Iij.m_nnzIij * 2 - m_inputData->m_source.m_nodeNum) * 144;
	//std::cout << "nnzSize2: " << nnzSize << std::endl;
	//timer.TimeStart();
	m_pcgLinearSolver->solveCPUOpt(m_dDelta, RAW_PTR(m_dJTJ.m_dIa), RAW_PTR(m_dJTJ.m_dJa), RAW_PTR(m_dJTJ.m_dA), nnzSize,
	                               m_dJTb, m_dPreconditioner, 100);
	/*m_pcgLinearSolver->SolveCPUOpt(RAW_PTR(m_dJTJ.m_dIa), RAW_PTR(m_dJTJ.m_dJa), RAW_PTR(m_dJTJ.m_dA), m_dJTJ.m_nnz, m_dJTb,
		m_dDelta, m_dPreconditioner, 1000);*/

	//timer.TimeEnd();
	//std::cout << "PCG time: " << timer.TimeGap_in_ms() << std::endl;

	
#if 0
	std::cout << "delta: " <<std::endl;
	std::vector<float> deltaRtsVec(m_inputData->m_source.m_nodeNum * 12);
	checkCudaErrors(cudaMemcpy(deltaRtsVec.data(), RAW_PTR(m_dDelta), deltaRtsVec.size() * sizeof(float), cudaMemcpyDeviceToHost));
	for (int nn = 0; nn < deltaRtsVec.size(); ++nn)
	{
		printf("%f, ", deltaRtsVec[nn]);
	}
	std::cout << std::endl;
#endif

#if 1
	//timer.TimeStart();
	AddDeltaVarstoVars(m_dVars, m_dDelta, m_varsNum);

#if 0
	std::cout << "Nan Vars Num: " << CheckNanVertex(RAW_PTR(m_dVars), m_varsNum) << "\n";
	std::cout << "Nan Delta Num: " << CheckNanVertex(RAW_PTR(m_dDelta), m_varsNum) << "\n";
	thrust::copy(m_dVars.begin(), m_dVars.begin() + 20, std::ostream_iterator<float>(std::cout, " ")); printf("\n");
	thrust::copy(m_dDelta.begin(), m_dDelta.begin() + 20, std::ostream_iterator<float>(std::cout, " ")); printf("\n");

	for (it = m_cons.begin(); it != m_cons.end(); ++it) {
		Constraint *pCon = *it;
		val = pCon->val(reinterpret_cast<float3*>(RAW_PTR(m_dVars)));
		printf("before opt: GPU,%s:%.10f \n", pCon->ctype(), val);
		//m_residual += val;
	}
#endif

	CalcInvTransRot((float *)RAW_PTR(m_dRsTransInv), (float *)RAW_PTR(m_dVars), m_inputData->m_source.m_nodeNum);
	UpdateVertexPosesAndNormals(RAW_PTR(m_inputData->m_deformed.m_dVertexVec),
	                                   RAW_PTR(m_inputData->m_deformed.m_dNormalVec),
	                                   RAW_PTR(m_inputData->m_source.m_dVertexVec),
	                                   RAW_PTR(m_inputData->m_source.m_dNormalVec),
	                                   (float3 *)RAW_PTR(m_dVars),
	                                   (float3 *)RAW_PTR(m_dRsTransInv),
	                                   m_inputData->m_source.m_vertexNum,
	                                   RAW_PTR(m_inputData->m_source.m_dNodeVec),
	                                   RAW_PTR(m_inputData->m_source.m_dVertexRelaIdxVec),
	                                   RAW_PTR(m_inputData->m_source.m_dVertexRelaWeightVec));
	//timer.TimeEnd();
	//std::cout << "update vertex time: " << timer.TimeGap_in_ms() << std::endl;

	//std::cout << "Nan Deform Vertex Num: " << CheckNanVertex(RAW_PTR(m_inputData->m_deformed.m_dVertexVec), m_inputData->m_deformed.m_vertexNum) << "\n";

#if 1
	//timer.TimeStart();
	updateCameraPoses();
	//timer.TimeEnd();
	//std::cout << "update camera time: " << timer.TimeGap_in_ms() << std::endl;
#endif
#endif

	return true;
}
