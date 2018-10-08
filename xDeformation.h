#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
//#include <Eigen/Eigen>

//#include "xDeformation/Cuda/xLinearSolver.cuh"
#include "xDeformation/Cuda/xDeformationCudaFuncs2.cuh"
#include "Helpers/xUtils.h"
#include "Helpers/xGlobalStats.h"
//#include "SiftGPU/xSift.h"
#include "xDeformation/KNearestPoint.h"

class GNSolver;
class InputData;

class xDeformation
{
public:
	xDeformation(int& fragIdx, VBOType *dVboCuda);
	void addData(const cv::Mat& colorImg,
	             const cv::Mat& fullColorImg,
	             const cv::Mat_<uchar>& grayImg,
	             const cv::cuda::GpuMat& dGrayImg,
	             xMatrix4f pose);
	void addDataWithKeyFrame(const cv::Mat& colorImg,
	                         const cv::Mat& rawDepthImg,
	                         const cv::Mat& grayImg,
	                         xMatrix4f pose);
	void deform(xMatrix4f *latestPose, VBOType* vboDevice, int vertexNum, int keyFrameIdxEachFrag = -1);

	void savePly(const char *fileDir, int fragIdx);
	void saveModel();
	int getSrcVertexNum();
	int getSrcNodeNum();
	int getFragNum();
	void getVertexStrideVe(std::vector<int>& vertexStrideVec);
	void getDeformedVertices(std::vector<float4>& deformedVertexVec);
	void getDeformedNormals(std::vector<float4>& deformedVertexVec);
	void getMatchingPointIndices(std::vector<int>& matchingPointsVec);

private:
	void prepareData(int vertexNum);
	void prepareDataWithKeyFrame(int vertexNum, int keyFrameIdxEachFrag);

	void allocEssentialCUDAMemory();

	void doOptimize(int iterNum);

	void updatePoseGraph();
	
	void findMatchingKNN();
	void findMatchingPerspective();
	void findMatchingPoints()
	{
		switch (gs::geoMatchingType)
		{
		case KNN:
			findMatchingKNN();
			break;
		case PERSPECTIVE:
			findMatchingPerspective();
			break;
		case NO_GEOMATCHING:
			break;
		}
		if (m_matchingPointsNumDescriptor == 0 && m_matchingPointsNumNearest == 0)
		{
			std::cout << "ERROR: no matching points" << std::endl;
			std::exit(0);
		}
	}

public:
	// For frames
	std::vector<cv::Mat> m_grayImgVec;
	std::vector<cv::Mat> m_colorImgVec;	
	std::vector<cv::Mat> m_fullColorImgVec;	
	std::vector<cv::Mat> m_keyFullColorImgVec;	
	std::vector<float> m_blurScoreVec;
	std::vector<xMatrix4f> m_poseVec;

    std::vector<cv::Mat> m_depthImgVec;
    std::vector<cv::Mat> m_keyDepthImgVec;

	// For calc blur score
	//cv::cuda::GpuMat m_grayImgDevice;
	cv::cuda::GpuMat m_dVerticalBlurImg;
	cv::cuda::GpuMat m_dHorizontalBlurImg;

	// For fragments
	std::vector<int> m_isFragValid;
	std::vector<std::vector<int> > m_poseGraph;
	std::vector<std::vector<int> > m_poseGraphInv;
	std::vector<xMatrix4f> m_keyPoseVec;
	std::vector<xMatrix4f> m_keyPoseInvVec;

	// Device
	VBOType *m_dVboCuda;

	std::pair<uchar *, int> m_dKeyColorImgs;
	std::pair<float *, int> m_dKeyGrayImgs;
	std::pair<float *, int> m_dKeyGrayImgsDx;
	std::pair<float *, int> m_dKeyGrayImgsDy;
	cv::Mat m_kxRow, m_kyRow, m_kxCol, m_kyCol;

	// For matching points
	std::pair<int *, int> m_dIdxMaps;
	std::pair<float *, int> m_dIdxMapZBufs;
	int *m_dMatchingFragIndices;
	int *m_dMatchingPointIndices;

	int m_matchingPointNum;
	int m_matchingPointsNumDescriptor;
	int m_matchingPointsNumNearest;

	// For camera update
	float4 *m_dKeyPoses;
	float4 *m_dUpdatedKeyPoses;
	float4 *m_dUpdatedKeyPosesInv;

	int& m_fragIdx;
	int m_loopClosureNum;
	int m_width;
	int m_height;

	GNSolver* m_gnSolver;
	InputData *m_inputData;
};


