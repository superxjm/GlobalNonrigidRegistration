#include "InputData.h"

#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <fstream>

#include "Helpers/xUtils.h"
#include "Helpers/UtilsMath.h"

__global__ void CreateVertexPosesAndNormalsKernel(float4 *originVertexPosesFrag,
	float4 *updatedVertexPosesFrag,
	float4 *originVertexNormalsFrag,
	float4 *updatedVertexNormalsFrag,
	VBOType *vboFrag, int vertexNumFrag)
{
	int vertexIndFrag = threadIdx.x + blockIdx.x * blockDim.x;

	if (vertexIndFrag >= vertexNumFrag)
		return;

	float4 *posConf = &(vboFrag + vertexIndFrag)->posConf;
	float4 *normalRad = &(vboFrag + vertexIndFrag)->normalRad;
	float4 *colorTime = &(vboFrag + vertexIndFrag)->colorTime;

	*(originVertexPosesFrag + vertexIndFrag) = make_float4(posConf->x, posConf->y, posConf->z, colorTime->y);
	*(updatedVertexPosesFrag + vertexIndFrag) = make_float4(posConf->x, posConf->y, posConf->z, colorTime->y);

	*(originVertexNormalsFrag + vertexIndFrag) = make_float4(normalRad->x, normalRad->y, normalRad->z, 0.0f);
	*(updatedVertexNormalsFrag + vertexIndFrag) = make_float4(normalRad->x, normalRad->y, normalRad->z, 0.0f);
}

void CreateVertexPosesAndNormals(float4 *originVertexPosesDevice,
	float4 *updatedVertexPosesDevice,
	float4 *originVertexNormalsDevice,
	float4 *updatedVertexNormalsDevice,
	VBOType *vboDevice,
	std::vector<int> &vertexStrideVec,
	int fragInd)
{
	int vertexNumFrag = vertexStrideVec[fragInd + 1] - vertexStrideVec[fragInd];

	int block = 256, grid = DivUp(vertexNumFrag, block);
	CreateVertexPosesAndNormalsKernel << <grid, block >> > (
		originVertexPosesDevice + vertexStrideVec[fragInd],
		updatedVertexPosesDevice + vertexStrideVec[fragInd],
		originVertexNormalsDevice + vertexStrideVec[fragInd],
		updatedVertexNormalsDevice + vertexStrideVec[fragInd],
		vboDevice + vertexStrideVec[fragInd], vertexNumFrag);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

#if 0
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
#endif
}

InputData::InputData()
{
}

InputData::~InputData()
{
}

void InputData::prepareData(VBOType* vboDevice,
                            int vertexNum,
                            int fragIdx)
{		
	m_deformed.m_vertexNum = vertexNum;
	m_source.addNewFrag(vertexNum);

	CreateVertexPosesAndNormals(RAW_PTR(m_source.m_dVertexVec),
		RAW_PTR(m_deformed.m_dVertexVec),
		RAW_PTR(m_source.m_dNormalVec),
		RAW_PTR(m_deformed.m_dNormalVec),
		vboDevice,
		m_source.m_vertexStrideVec,
		fragIdx);
	m_source.sampleNewNodeAndVertexIdx();
	m_source.getVertexAndNodeRelation();
}

void InputData::getIijSet(int* matchingPointIndices, int matchingPointNum)
{
	m_Iij.getIijSet(matchingPointIndices, matchingPointNum, m_source);
}


