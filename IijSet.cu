#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include "Helpers/xUtils.h"
#include "InputData.h"

struct FillIijInDataTermWrapper
{
	int IijNumEach;
	int matchingPointNum;
	int *matchingPointIndices;
	int* vertexRela;
	int* nzIndex;
	int* functionId;
	int nodeNum;

	__device__ void operator()()
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= matchingPointNum)
		{
			return;
		}

		// fill function_id
		for (int iter = 0; iter < IijNumEach; ++iter)
		{
			functionId[idx * IijNumEach + iter] = idx;
		}

		int srcVertexIdx = *(matchingPointIndices + 2 * idx);
		int targetVertexIdx = *(matchingPointIndices + 2 * idx + 1);

		if (srcVertexIdx < 0 || targetVertexIdx < 0)
		{
			for (int iter = 0; iter < IijNumEach; ++iter)
			{
				nzIndex[idx * IijNumEach + iter] = -1;
			}
			return;
		}

		int i, j, index_Iij;
		int cnt = 0;
		int* srcRela = vertexRela + srcVertexIdx * MAX_NEAR_NODE_NUM_VERTEX;
		int* targetRela = vertexRela + targetVertexIdx * MAX_NEAR_NODE_NUM_VERTEX;	
		// fill Iij coo
		for (int row = 0; row < MAX_NEAR_NODE_NUM_VERTEX; ++row)
		{
			i = srcRela[row];
			for (int col = row; col < MAX_NEAR_NODE_NUM_VERTEX; ++col)
			{
				j = srcRela[col];
				if (i < j)
				{
					index_Iij = j * nodeNum + i;
				}
				else
				{
					index_Iij = i * nodeNum + j;
				}

				nzIndex[idx * IijNumEach + cnt] = index_Iij;
				++cnt;
			}
		}	
		for (int row = 0; row < MAX_NEAR_NODE_NUM_VERTEX; ++row)
		{
			i = targetRela[row];
			for (int col = row; col < MAX_NEAR_NODE_NUM_VERTEX; ++col)
			{
				j = targetRela[col];
				if (i < j)
				{
					index_Iij = j * nodeNum + i;
				}
				else
				{
					index_Iij = i * nodeNum + j;
				}

				nzIndex[idx * IijNumEach + cnt] = index_Iij;
				++cnt;
			}
		}
		for (int row = 0; row < MAX_NEAR_NODE_NUM_VERTEX; ++row)
		{
			i = targetRela[row];
			for (int col = 0; col < MAX_NEAR_NODE_NUM_VERTEX; ++col)
			{
				j = srcRela[col];
				index_Iij = j * nodeNum + i;
				nzIndex[idx * IijNumEach + cnt] = index_Iij;
				++cnt;
			}
		}
	}
};

__global__ void FillIijInDataTermKernel(FillIijInDataTermWrapper fidt_wrapper)
{
	fidt_wrapper();
}

struct FillIijInSmoothTermWrapper
{
	int IijNumEach;
	int matchingPointNum;
	int* nodeRela;
	int* nzIndex;
	int* functionId;
	int nodeNum;

	__device__ void operator()()
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < nodeNum)
		{
			int i = idx, j, IiiIdx;
			int cnt = 0;
			int* rela = nodeRela + idx * MAX_NEAR_NODE_NUM_NODE;
			int funcIdx = idx * MAX_NEAR_NODE_NUM_NODE + matchingPointNum;
			// fill Iij coo, node与自己共八个连接，第一个node与其连接的node共七个连接。
			// 每个node与自己的连接产生一个非零块。
			// 每个node与非自己的连接在下三角矩阵中产生三个非零块。
			IiiIdx = i * nodeNum + i;
			nzIndex[idx * IijNumEach + cnt] = IiiIdx;
			functionId[idx * IijNumEach + cnt] = funcIdx;
			funcIdx++;
			cnt++;

			int IijIdx;
			for (int iter = 1; iter < MAX_NEAR_NODE_NUM_NODE; ++iter)
			{
				nzIndex[idx * IijNumEach + cnt] = IiiIdx;
				functionId[idx * IijNumEach + cnt] = funcIdx;
				cnt++;

				j = rela[iter];
				IijIdx = j * nodeNum + j;
				nzIndex[idx * IijNumEach + cnt] = IijIdx;
				functionId[idx * IijNumEach + cnt] = funcIdx;
				cnt++;

				if (i < j)
				{
					IijIdx = j * nodeNum + i;
				}
				else
				{
					IijIdx = i * nodeNum + j;
				}
				nzIndex[idx * IijNumEach + cnt] = IijIdx;
				functionId[idx * IijNumEach + cnt] = funcIdx;
				cnt++;

				funcIdx++;
			}
		}
	}
};

__global__ void FillIijInSmoothTermKernel(FillIijInSmoothTermWrapper fist_wrapper)
{
	fist_wrapper();
}

struct SetOffsetFlagWrapper
{
	int num_element;
	int* Iij_list;
	int* offset_flag;

	__device__ void operator()()
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x + 1;
		if (idx < num_element)
		{
			if (Iij_list[idx] != Iij_list[idx - 1])
			{
				offset_flag[idx] = idx;
			}
		}
	}
};

__global__ void SetOffsetFlagKernel(SetOffsetFlagWrapper sof_wrapper)
{
	sof_wrapper();
}

struct SetCOOFlagWrapper
{
	int num_element;
	int* nz_index;
	int* coo_flag;

	__device__ void operator()()
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x + 1;
		if (idx < num_element)
		{
			if (nz_index[idx] != nz_index[idx - 1])
			{
				coo_flag[idx] = nz_index[idx];
			}
		}
	}
};

__global__ void SetCOOFlagKernel(SetCOOFlagWrapper scf_wrapper)
{
	scf_wrapper();
}

struct SetNzFlagWrapper
{
	int num_coo_ele;
	int* Iij_coo;
	bool* nz_flag;
	int num_node;

	__device__ void operator()()
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < num_coo_ele)
		{
			int index = Iij_coo[idx];
			int seri_i = index / num_node;
			int seri_j = index - seri_i * num_node;
			nz_flag[index] = true;
			nz_flag[seri_j * num_node + seri_i] = true;
		}
	}
};

__global__ void SetNzFlagKernel(SetNzFlagWrapper snf_wrapper)
{
	snf_wrapper();
}

struct GetNnzPreWrapper
{
	bool* nz_flag;
	int* nnz_pre;
	int* row_ptr;
	int num_node;

	// 每个block包含一个warp， 通过warp内协作进行prefix_sum
	__device__ void operator()()
	{
		__shared__ int s_pre[1];
		if (threadIdx.x == 0)
		{
			s_pre[0] = 0;
			nnz_pre[num_node * blockIdx.x] = 0;
		}
		//__syncwarp();

		bool* flag_ptr = nz_flag + num_node * blockIdx.x;
		// nnz_pre_ptr需要的是exclude scan, 加一个偏移在include scan上实现。
		int* nnz_pre_ptr = nnz_pre + num_node * blockIdx.x + 1;
		int value, lane_id;
		// 循环内为warp内include scan（prefix sum）
		for (int iter = threadIdx.x; iter < num_node - 1; iter += blockDim.x)
		{
			lane_id = threadIdx.x & 0x1f;
			value = flag_ptr[iter];
			for (int i = 1; i <= 32; i *= 2)
			{
				int n = __shfl_up_sync(__activemask(), value, i, 32);
				if ((lane_id & 31) >= i)
				{
					value += n;
				}
			}
			nnz_pre_ptr[iter] = s_pre[0] + value;
			if (threadIdx.x == blockDim.x - 1)
			{
				s_pre[0] += value;
			}
			__syncwarp();
		}

		if (threadIdx.x == 0)
		{
			if (flag_ptr[num_node - 1])
			{
				row_ptr[blockIdx.x] = nnz_pre_ptr[num_node - 2] + 1;
			}
			else
			{
				row_ptr[blockIdx.x] = nnz_pre_ptr[num_node - 2];
			}
		}
	}
};

__global__ void GetNnzPreKernel(GetNnzPreWrapper gnp_wrapper)
{
	gnp_wrapper();
}

struct GetDataTermNumWrapper
{
	int nnzLowTri;
	int matchingPointNum;

	int* funcList;
	int* funcOffset;
	int* numDataTerm;

	__device__ void operator()()
	{
		__shared__ int* s_func_list_ptr;
		__shared__ int s_num_fun;
		__shared__ int s_res_each_warp[4]; // block_size = 128
		if (blockIdx.x < nnzLowTri)
		{
			if (threadIdx.x == 0)
			{
				s_func_list_ptr = funcList + funcOffset[blockIdx.x];
				s_num_fun = funcOffset[blockIdx.x + 1] - funcOffset[blockIdx.x];
			}
			if (threadIdx.x < 4)
			{
				s_res_each_warp[threadIdx.x] = 0;
			}
			__syncthreads();

			bool is_data_term;
			unsigned vote_result;
			int warp_id = threadIdx.x / 32;
			// warp内每个线程根据自己的数据（is_data_term）进行投票，然后统计投票结果。
			for (int iter = threadIdx.x; iter < s_num_fun; iter += blockDim.x)
			{
				is_data_term = (s_func_list_ptr[iter] < matchingPointNum);
				vote_result = __ballot_sync(__activemask(), is_data_term);
				if (threadIdx.x - warp_id * 32 == 0 && vote_result > 0)
				{
					// first thread of each warp
					s_res_each_warp[warp_id] += int(log2f(float(vote_result) + 1.f));
				}
				__syncthreads();
			}
			if (threadIdx.x == 0)
			{
				numDataTerm[blockIdx.x] = s_res_each_warp[0] + s_res_each_warp[1] + s_res_each_warp[2] + s_res_each_warp[3];
			}
		}
	}

	__device__ void operator()(int type)
	{
		__shared__ int s_res[1];
		if (blockIdx.x < nnzLowTri)
		{
			if (threadIdx.x == 0)
			{
				s_res[0] = 0;
			}
			int* func_list_ptr = funcList + funcOffset[blockIdx.x];
			int num_fun = funcOffset[blockIdx.x + 1] - funcOffset[blockIdx.x];

			__syncthreads();

			bool is_data_term;
			int vote_result;
			int warp_id = threadIdx.x / 32;
			// 块内每个线程根据自己的数据（is_data_term）进行投票，然后统计投票结果。
			for (int iter = threadIdx.x; iter < num_fun; iter += blockDim.x)
			{
				is_data_term = (func_list_ptr[iter] < matchingPointNum);
				vote_result = __syncthreads_count(is_data_term);
				if (threadIdx.x == 0)
				{
					s_res[0] += vote_result;
				}
			}
			if (threadIdx.x == 0)
			{
				numDataTerm[blockIdx.x] = s_res[0];
			}
		}
	}
};

__global__ void GetDataTermNumKernel(GetDataTermNumWrapper gdtn_wrapper)
{
	gdtn_wrapper();
	//  gdtn_wrapper(1);
}

__global__ void CountInvalidElements(int *cnt, int *elem, int elemNum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= elemNum)
	{
		return;
	}

	if (elem[idx] == -1)
	{
		atomicAdd(cnt, 1);
	}
}

void Iij::getIijSet(int* matchingPointIndices, int matchingPointNum, FragDeformableMeshData &source)
{
#if 0
	thrust::device_vector<int> dInvalidMatchingPointNum(1, 0);
	int invalidMatchingPointNum, validMatchingPointNum;
	checkCudaErrors(cudaMemset(RAW_PTR(dInvalidMatchingPointNum), 0, sizeof(int)));
	int block = 256;
	int grid = DivUp(matchingPointNum, block);
	if (grid > 0)
	{
		CountInvalidMatchingPoints << < grid, block >> >(RAW_PTR(dInvalidMatchingPointNum),
		                                                 matchingPointIndices,
		                                                 matchingPointNum);
	}
	checkCudaErrors(cudaMemcpy(&invalidMatchingPointNum, RAW_PTR(dInvalidMatchingPointNum),
		sizeof(int), cudaMemcpyDeviceToHost));
	validMatchingPointNum = matchingPointNum - invalidMatchingPointNum;
#endif

	int num_node = source.m_nodeNum, num_vertex = source.m_vertexNum;
	// 只需存储下三角，下三角元素个数：(m*m+m)/2
	int num_Iij_each_data_function = (MAX_NEAR_NODE_NUM_VERTEX * MAX_NEAR_NODE_NUM_VERTEX + MAX_NEAR_NODE_NUM_VERTEX) / 2
		+ (MAX_NEAR_NODE_NUM_VERTEX * MAX_NEAR_NODE_NUM_VERTEX + MAX_NEAR_NODE_NUM_VERTEX) / 2
		+ (MAX_NEAR_NODE_NUM_VERTEX * MAX_NEAR_NODE_NUM_VERTEX);
	// 36个。每个点对包含4个node，形成36个下三角中的非零块。
	int num_Iij_each_smooth_function = (MAX_NEAR_NODE_NUM_NODE - 1) * 3 + 1;
	// 22个。第一个node与其连接的node共七个, 每个连接包含两个node，形成三个下三角中的非零块。第一个node与自身连接形成一个下三角的非零块。
	int len_Iij = matchingPointNum * num_Iij_each_data_function + num_node * num_Iij_each_smooth_function;
	
	thrust::device_vector<int> dListIijTmp(len_Iij);	
	thrust::device_vector<int> nonZeroIdxTmp(len_Iij);
	// 根据data项确定JTJ中的非零块的位置及其对应函数序号。
	int block = 256, grid = (matchingPointNum + block - 1) / block;
	FillIijInDataTermWrapper fidt_wrapper;
	fidt_wrapper.IijNumEach = num_Iij_each_data_function;
	fidt_wrapper.matchingPointNum = matchingPointNum;
	fidt_wrapper.matchingPointIndices = matchingPointIndices;
	fidt_wrapper.vertexRela = RAW_PTR(source.m_dVertexRelaIdxVec);
	fidt_wrapper.nzIndex = RAW_PTR(nonZeroIdxTmp);
	fidt_wrapper.functionId = RAW_PTR(dListIijTmp);
	fidt_wrapper.nodeNum = num_node;
	if (grid > 0)
	{
		FillIijInDataTermKernel << < grid, block >> > (fidt_wrapper);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	// 根据smooth项确定JTJ中的非零块的位置及其对应函数序号。
	block = 64, grid = (num_node + block - 1) / block;
	FillIijInSmoothTermWrapper fist_wrapper;
	fist_wrapper.IijNumEach = num_Iij_each_smooth_function;
	fist_wrapper.matchingPointNum = matchingPointNum;
	fist_wrapper.nodeRela = RAW_PTR(source.m_dNodeRelaIdxVec);
	fist_wrapper.nzIndex = RAW_PTR(nonZeroIdxTmp) + matchingPointNum * num_Iij_each_data_function;
	fist_wrapper.functionId = RAW_PTR(dListIijTmp) + matchingPointNum * num_Iij_each_data_function;
	fist_wrapper.nodeNum = num_node;
	FillIijInSmoothTermKernel << < grid, block >> >(fist_wrapper);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	thrust::stable_sort_by_key(nonZeroIdxTmp.begin(), nonZeroIdxTmp.end(), dListIijTmp.begin());

#if 1
	thrust::device_vector<int> dCntInvalid(1, 0);
	block = 256, grid = (len_Iij + block - 1) / block;
	CountInvalidElements << < grid, block >> >(RAW_PTR(dCntInvalid), RAW_PTR(nonZeroIdxTmp), len_Iij);
	int cntInvalid;
	checkCudaErrors(cudaMemcpy(&cntInvalid, RAW_PTR(dCntInvalid), sizeof(int),
		cudaMemcpyDeviceToHost));
#endif

	len_Iij -= cntInvalid;
	m_dListIij.resize(len_Iij);
	thrust::device_vector<int> nonZeroIdx(len_Iij);
	checkCudaErrors(cudaMemcpy(RAW_PTR(m_dListIij), RAW_PTR(dListIijTmp) + cntInvalid, len_Iij * sizeof(int),
		cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(RAW_PTR(nonZeroIdx), RAW_PTR(nonZeroIdxTmp) + cntInvalid, len_Iij * sizeof(int),
		cudaMemcpyDeviceToDevice));

	thrust::device_vector<int> temp_array(len_Iij);

	// calculate offset
	checkCudaErrors(cudaMemset(RAW_PTR(temp_array), 0, len_Iij * sizeof(int)));
	block = 256, grid = (len_Iij + block - 1) / block;
	SetOffsetFlagWrapper sof_wrapper;
	sof_wrapper.num_element = len_Iij;
	sof_wrapper.Iij_list = RAW_PTR(nonZeroIdx);
	sof_wrapper.offset_flag = RAW_PTR(temp_array);
	SetOffsetFlagKernel << < grid, block >> >(sof_wrapper);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	
	auto new_end = thrust::remove(temp_array.begin() + 1, temp_array.end(), int(0)); // 保留第一位的0，表示其偏移。
	int nnz_Iij = new_end - temp_array.begin();

	m_nnzIij = nnz_Iij;
	//printf("nnzIij = %d \n", nnzIij);

	m_dOffsetIij.resize(nnz_Iij + 1);
	temp_array[nnz_Iij] = len_Iij; // exclude_scan 最后一位填充为总长度。 num = offset[idx+1] - offset[idx];
	checkCudaErrors(cudaMemcpy(RAW_PTR(m_dOffsetIij), RAW_PTR(temp_array), (nnz_Iij + 1) * sizeof(int),
		cudaMemcpyDeviceToDevice));

	// calculate d_nz_Iij_coo, node_num*node_num稀疏矩阵的(i,j)块coo编码为i×node_num+j。
	checkCudaErrors(cudaMemset(RAW_PTR(temp_array), 0, len_Iij * sizeof(int)));
	block = 256, grid = (len_Iij + block - 1) / block;
	SetCOOFlagWrapper scf_wrapper;
	scf_wrapper.num_element = len_Iij;
	scf_wrapper.nz_index = RAW_PTR(nonZeroIdx);
	scf_wrapper.coo_flag = RAW_PTR(temp_array);
	SetCOOFlagKernel << < grid, block >> >(scf_wrapper);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	new_end = thrust::remove(temp_array.begin() + 1, temp_array.end(), 0); // 保留第一位的0，表示其偏移。
	nnz_Iij = new_end - temp_array.begin();

	m_dNzIijCoo.resize(nnz_Iij);
	checkCudaErrors(cudaMemcpy(RAW_PTR(m_dNzIijCoo), RAW_PTR(temp_array), nnz_Iij * sizeof(int),
		cudaMemcpyDeviceToDevice));

	// 创建一个NxN的标志位矩阵，非零块的位置标记为1.
	thrust::device_vector<bool> d_nz_flag(num_node * num_node, 0);
	SetNzFlagWrapper snf_wrapper;
	snf_wrapper.num_coo_ele = nnz_Iij;
	snf_wrapper.Iij_coo = RAW_PTR(m_dNzIijCoo);
	snf_wrapper.nz_flag = RAW_PTR(d_nz_flag);
	snf_wrapper.num_node = num_node;
	block = 256, grid = (nnz_Iij + block - 1) / block;
	SetNzFlagKernel << < grid, block >> >(snf_wrapper);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	thrust::device_vector<int> d_row_ptr(num_node);
	m_dRowPtr.resize(num_node + 1); // 每一行的非零块个数
	m_dNnzPre.resize(num_node * num_node);
	//checkCudaErrors(cudaMemset(RAW_PTR(m_Iij.d_nnz_pre), 0, m_Iij.d_nnz_pre.size() * sizeof(int)));
	GetNnzPreWrapper gnp_wrapper;
	gnp_wrapper.nz_flag = RAW_PTR(d_nz_flag);
	gnp_wrapper.nnz_pre = RAW_PTR(m_dNnzPre);
	gnp_wrapper.row_ptr = RAW_PTR(d_row_ptr);
	gnp_wrapper.num_node = num_node;
	block = 32, grid = num_node;
	GetNnzPreKernel << < grid, block >> >(gnp_wrapper);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	// row_ptr, JTJ (非零块)稀疏矩阵csr中的row_ptr（ia）.
	thrust::inclusive_scan(d_row_ptr.begin(), d_row_ptr.end(), m_dRowPtr.begin() + 1);

	// 计算数据项方程的个数 d_data_item_num, 用于区分数data项jtj方程个数与smooth项方程的个数，方程编号大于顶点个数的为smooth项方程。
	GetDataTermNumWrapper gdtn_wrapper;
	gdtn_wrapper.nnzLowTri = nnz_Iij;
	gdtn_wrapper.matchingPointNum = matchingPointNum;
	gdtn_wrapper.funcList = RAW_PTR(m_dListIij);
	gdtn_wrapper.funcOffset = RAW_PTR(m_dOffsetIij);
	m_dDataItemNum.resize(nnz_Iij, 0);
	gdtn_wrapper.numDataTerm = RAW_PTR(m_dDataItemNum);
	block = 128, grid = nnz_Iij;
	GetDataTermNumKernel << < grid, block >> >(gdtn_wrapper);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}
