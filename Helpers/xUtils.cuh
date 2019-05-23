#pragma once

//#include "Helpers/containers/device_array.hpp"
#include "Helpers/UtilsMath.h"
#include <opencv2/opencv.hpp>

//void BilateralFilter(const DeviceArray2D<ushort>& src, DeviceArray2D<ushort>& dst, ushort max_distance_mm);
void BilateralFilter(cv::cuda::GpuMat& dst, const cv::cuda::GpuMat& src);

void VMapToDepthMap(cv::cuda::GpuMat& dDepthMap, cv::cuda::GpuMat& dVMapFloat4);

void RegisterDepthImg(cv::cuda::GpuMat& dRawDepthImg, cv::cuda::GpuMat& dRawDepthImgBuffer,
                      float cxDepth, float cyDepth, float fxDepth, float fyDepth,
                      float3 RX, float3 RY, float3 RZ, float3 t,
                      float cxColor, float cyColor, float fxColor, float fyColor);

void CvGpuMatDownload(uchar* dst, cv::cuda::GpuMat& src);

void CvGpuMatUpload(cv::cuda::GpuMat& dst, uchar* src);

void PruneDepth(cv::cuda::GpuMat& dRawDepthImg,
                cv::cuda::GpuMat& dFilteredDepthImg,
                cv::cuda::GpuMat& dRawDepthImg32F,
                cv::cuda::GpuMat& dFilteredDepthImg32F,
                cv::cuda::GpuMat& dPruneMat);
void FilterDepthUsingDepth(cv::cuda::GpuMat& dRawDepthImg,
						   cv::cuda::GpuMat& dFilteredDepthImg,
						   cv::cuda::GpuMat& dRawDepthImg32F,
						   cv::cuda::GpuMat& dFilteredDepthImg32F,
						   float depthThresh);

