#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
//#include <Eigen/Eigen>

#include "Cuda/xLinearSolver.cuh"
#include "Cuda/xDeformationCudaFuncs2.cuh"
#include "Helpers/xUtils.h"
#include "Helpers/xGlobalStats.h"
//#include "SiftGPU/xSift.h"
#include "KNearestPoint.h"

class GNSolver;
class InputData;

class xDeformation
{
public:
	xDeformation(int& fragIdx, VBOType *dVboCuda);
	~xDeformation();
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
	void findMatchingPoints(std::vector<cv::Mat_<float>> &_camera_pose,
		std::vector<cv::Mat_<float>> &_camera_fxfycxcy,
		int _method, float _dist_thresh1, float _dist_thresh2, float _angle_thresh,
		int _width, int _height);
	void deformToghter(int iter_num, VBOType* vboDevice, std::vector<int> sumVertexNum, std::vector<cv::Mat_<float>> &_camera_pose,
		std::vector<cv::Mat_<float>> &_camera_fxfycxcy,
		int _method, float _dist_thresh1, float _dist_thresh2, float _angle_thresh,
		int _width, int _height);

	void savePly(const char *fileDir, int fragIdx);
	void saveModel();
	int getSrcVertexNum();
	int getSrcNodeNum();
	int getFragNum();
	void getVertexStrideVe(std::vector<int>& vertexStrideVec);
	void getDeformedVertices(std::vector<float4>& deformedVertexVec);
	void getDeformedNormals(std::vector<float4>& deformedVertexVec);
	void getMatchingPointIndices(std::vector<int>& matchingPointsVec);

public:
	void prepareData(int vertexNum);
private:
	void prepareDataWithKeyFrame(int vertexNum, int keyFrameIdxEachFrag);

	void allocEssentialCUDAMemory();

	void doOptimize(int iterNum, float _dist_thresh1, float _dist_thresh2, float _angle_thresh);

	void updatePoseGraph();

	void setVirtualCameraInformation(std::string fileName);
	void setVirtualCameraInformation(std::vector<cv::Mat_<float>> cameraPose, 
		std::vector<cv::Mat_<float>> cameraFxFyCxCy, 
		int width, int height);
	void setVirtualCameraInformation(int virtualCameraCircleNum, 
		int *virtualWidth, int *virtualHeight, 
		float4 *virtualCameraPoses, 
		float4 *virtualCameraPosesInv, 
		float4 *virtualCameraFxFyCxCy);

	void findMatchingKNNWithConsistentCheck();
	void findMatchingKNN();
	void findMatchingPerspective();
	void findMatchingPerspectiveFromVirtualCamera();
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
		case KNNWITHCONSISTANTCHECK:
			findMatchingKNNWithConsistentCheck();
			break;
		case PERSPECTIVEFORMVIRTUAL:
			findMatchingPerspectiveFromVirtualCamera();
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
	

	float *m_dMatchingPointDist;
	std::pair<int *, int>     m_dVirtualIdxMaps;
	std::pair<float *, int>   m_dVirtualIdxMapZBufs;
	int m_virtualCameraCircleNum;
	std::vector<int> m_virtualWidth, m_virtualHeight;
	thrust::device_vector<float4> m_dVirtualCameraPoses;
	thrust::device_vector<float4> m_dVirtualCameraPosesInv;
	thrust::device_vector<float4> m_dVirtualCameraFxFyCxCy;


	int& m_fragIdx;
	int m_loopClosureNum;
	int m_width;
	int m_height;

	GNSolver* m_gnSolver;
	InputData *m_inputData;

	std::vector<cv::Vec4f> last_deformed_vertex_;
};


