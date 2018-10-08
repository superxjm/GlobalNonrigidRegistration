#pragma once

#include <thrust/device_vector.h>
#include "KNearestPoint.h"

class MeshData
{
public:
	MeshData();
	~MeshData();;

	thrust::device_vector<float4> m_dVertexVec;
	thrust::device_vector<float4> m_dNormalVec;

	int m_vertexNum;
	int m_nodeNum;
	int m_maxVertexNum;
	int m_maxNodeNum;
};

class DeformableMeshData : public MeshData
{
public:
	DeformableMeshData();
	~DeformableMeshData();;
	
	void addNewNode(float4* dNewNode, int newNodenum);

	thrust::device_vector<float4> m_dNodeVec;
	thrust::device_vector<int> m_dVertexRelaIdxVec;
	thrust::device_vector<float> m_dVertexRelaWeightVec;
	thrust::device_vector<int> m_dNodeRelaIdxVec;
	thrust::device_vector<float> m_dNodeRelaWeightVec;
	float m_sigma;

	NearestPoint* m_flannKnn;
};

class FragDeformableMeshData : public DeformableMeshData
{
public:
	FragDeformableMeshData();
	~FragDeformableMeshData();;

	void addNewFrag(int vertexNum);
	void sampleNewNodeAndVertexIdx();
	void getVertexAndNodeRelation();
	void addNewVertexIdx(int *dNewVertexIdx, int newVertexNum);

	int m_fragNum;
	int m_maxFragNum;
	int m_sampledVertexNum;
	int m_maxSampledVertexNum;
	std::vector<int> m_vertexStrideVec;
	thrust::device_vector<int> m_dSampledVertexIdxVec;
	thrust::device_vector<int> m_dMatchingPointsIdxVec;

	thrust::device_vector<int> m_dNewSampledVertexIdxVec;
	thrust::device_vector<float4> m_dNewNodeVec;
};

