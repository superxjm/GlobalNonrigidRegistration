#pragma once

#include "MeshData.h"

#include "Helpers/xUtils.h"

// 用于存储JTJ矩阵稀疏结构相关信息的数据结构。
struct Iij
{
	int m_nnzIij; // nnz block, down-triangle of Iij.
	thrust::device_vector<int> m_dListIij; // equations list
	thrust::device_vector<int> m_dOffsetIij;
	// nz_index offset of each non-zero block; assert(nnzIij == d_offset_Iij.size())
	thrust::device_vector<int> m_dNzIijCoo;
	//  non-zero position in JTJ. index = seri_i * num_node + seri_j; assert(nnzIij == d_nz_Iij_coo.size())
	thrust::device_vector<int> m_dNnzPre;
	// nnz before this node in the row, to determine block's position in csr. size: num_node * num_node
	thrust::device_vector<int> m_dDataItemNum; // number of data term equations for each nnz block. size: nnzIij
	thrust::device_vector<int> m_dIndDiagonal;
	// node_num*12 record index of diagonal elements of JTJ for extracting preconditioner(doing in rotation DirectiveJTJAndJTb).
	thrust::device_vector<int> m_dRowPtr;
	// row_ptr of JTJ matrix.  size: node_num + 1, exclude scan of nnz block of each row of JTJ.
	void clear()
	{
		m_dListIij.clear();
		m_dOffsetIij.clear();
		m_dNzIijCoo.clear();
		m_dIndDiagonal.clear();
		m_dNnzPre.clear();
		m_dDataItemNum.clear();
		m_dRowPtr.clear();
	}

	void getIijSet(int* matchingPointIndices, int matchingPointNum, FragDeformableMeshData &source);
};

class InputData
{
public:
	InputData();;
	~InputData();

	void prepareData(VBOType* vboDevice,
	                 int vertexNum,
	                 int fragIdx);
	void getIijSet(int* matchingPointIndices, int matchingPointNum);

	int getSrcVertexNum() const { return m_source.m_vertexNum; }
	int getSrcNodeNum() const { return m_source.m_nodeNum; }
	int getVarsNum() const { return m_source.m_nodeNum * 12; }
	int getVarsNumEachFrag() const { return NODE_NUM_EACH_FRAG * 12; }

	FragDeformableMeshData m_source;
	MeshData m_deformed;
	uchar* m_dKeyColorImgs = nullptr;
	float* m_dKeyGrayImgs;
	float* m_dKeyGrayImgsDx;
	float* m_dKeyGrayImgsDy;
	float4 *m_dKeyPoses;
	xMatrix4f *m_keyPoses;
	float4 *m_dUpdatedKeyPoses;
	float4 *m_dUpdatedKeyPosesInv;

	int* m_dMatchingPointIndices;
	int m_matchingPointNum;
	Iij m_Iij;
};


