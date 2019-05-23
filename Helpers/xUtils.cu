#include "xUtils.cuh"

#include "xUtils.h"

namespace GlobalParameter
{
	__device__ int d_node_num_each_frag;
	int h_node_num_each_frag;

	__host__ __device__ int GetNodeNumEachFrag()
	{
#if defined(__CUDA_ARCH__)
		return d_node_num_each_frag;
#else
		return h_node_num_each_frag;
#endif
	}

	__host__ void SetNodeNumEachFrag(int node_num)
	{
		h_node_num_each_frag = node_num;
		cudaMemcpyToSymbol(d_node_num_each_frag, &h_node_num_each_frag, sizeof(int));
	}

	__device__ int d_sampled_vertex_num_each_frag;
	int h_sampled_vertex_num_each_frag;

    __host__ __device__ int GetSampledVertexNumEachFrag()
	{
#if defined(__CUDA_ARCH__)
		return d_sampled_vertex_num_each_frag;
#else
		return h_sampled_vertex_num_each_frag;
#endif
	}

	__host__ void SetSampledVertexNumEachFrag(int vertex_num)
	{
		h_sampled_vertex_num_each_frag = vertex_num;
		cudaMemcpyToSymbol(d_sampled_vertex_num_each_frag, &h_sampled_vertex_num_each_frag, sizeof(int));
	}
}

#if 0
__global__ void BilateralKernel(const PtrStepSz<ushort> src,
	PtrStep<ushort> dst,
	float sigma_space2_inv_half, 
	float sigma_color2_inv_half,
	ushort max_distance_mm)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= src.cols || y >= src.rows)
		return;

	int value = src.ptr(y)[x];
	if (value > max_distance_mm)
	{
		dst.ptr(y)[x] = 0;
		return;
	}

	const int R = 6;       //static_cast<int>(sigma_space * 1.5);
	const int D = R * 2 + 1;
	
	int tx = min(x - D / 2 + D, src.cols - 1);
	int ty = min(y - D / 2 + D, src.rows - 1);

	float sum1 = 0;
	float sum2 = 0;

	for (int cy = max(y - D / 2, 0); cy < ty; ++cy)
	{
		for (int cx = max(x - D / 2, 0); cx < tx; ++cx)
		{
			int tmp = src.ptr(cy)[cx];

			float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
			float color2 = (value - tmp) * (value - tmp);

			float weight = __expf(-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

			sum1 += tmp * weight;
			sum2 += weight;
		}
	}

	int res = __float2int_rn(sum1 / sum2);
	dst.ptr(y)[x] = max(0, min(res, SHRT_MAX));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BilateralKernel(const DeviceArray2D<ushort>& src, DeviceArray2D<ushort>& dst, ushort max_distance_mm)
{
	dim3 block(32, 8);
	dim3 grid(DivUp(src.cols(), block.x), DivUp(src.rows(), block.y));

	const float sigma_space2_inv_half = 0.024691358; // 0.5 / (sigma_space * sigma_space)
	const float sigma_color2_inv_half = 0.000555556; // 0.5 / (sigma_color * sigma_color)
	cudaFuncSetCacheConfig(BilateralKernel, cudaFuncCachePreferL1);
	BilateralKernel << <grid, block >> >(src, dst, sigma_space2_inv_half, sigma_color2_inv_half, max_distance_mm);

	checkCudaErrors(cudaGetLastError());
};
#endif

__global__ void CheckNanVertexKernel(float4 *_d_vertex, int *_nan_num, int _N)
{
	int u = blockDim.x*blockIdx.x + threadIdx.x;

	if (u >= _N) return;

	if (isnan(_d_vertex[u].x))
	{
		atomicAdd(_nan_num, 1);
	}
}

int CheckNanVertex(float4 *_d_vertex, int _N)
{
	dim3 block(256);
	dim3 grid(DivUp(_N, block.x));
	thrust::device_vector<int> nan_num(1, 0);
	CheckNanVertexKernel << <grid, block >> > (_d_vertex, RAW_PTR(nan_num), _N);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	return nan_num[0];
}

__global__ void CheckNanVertexKernel(float *_d_vertex, int *_nan_num, int _N)
{
	int u = blockDim.x*blockIdx.x + threadIdx.x;

	if (u >= _N) return;

	if (isnan(_d_vertex[u]))
	{
		atomicAdd(_nan_num, 1);
	}
}

int CheckNanVertex(float *_d_vertex, int _N)
{
	dim3 block(256);
	dim3 grid(DivUp(_N, block.x));
	thrust::device_vector<int> nan_num(1, 0);
	CheckNanVertexKernel << <grid, block >> > (_d_vertex, RAW_PTR(nan_num), _N);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	return nan_num[0];
}

__global__ void BilateralKernel(cv::cuda::PtrStep<ushort> dst,
                                const cv::cuda::PtrStepSz<ushort> src,
                                float sigma_space2_inv_half,
                                float sigma_color2_inv_half)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= src.cols || y >= src.rows)
		return;

	int value = src.ptr(y)[x];

	const int R = 6;       //static_cast<int>(sigma_space * 1.5);
	const int D = R * 2 + 1;

	int tx = min(x - D / 2 + D, src.cols - 1);
	int ty = min(y - D / 2 + D, src.rows - 1);

	float sum1 = 0;
	float sum2 = 0;

	for (int cy = max(y - D / 2, 0); cy < ty; ++cy)
	{
		for (int cx = max(x - D / 2, 0); cx < tx; ++cx)
		{
			int tmp = src.ptr(cy)[cx];

			float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
			float color2 = (value - tmp) * (value - tmp);

			float weight = __expf(-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

			sum1 += tmp * weight;
			sum2 += weight;
		}
	}

	int res = __float2int_rn(sum1 / sum2);
	dst.ptr(y)[x] = max(0, min(res, SHRT_MAX));
}

void BilateralFilter(cv::cuda::GpuMat& dst, const cv::cuda::GpuMat& src)
{
	dim3 block(32, 8);
	dim3 grid(DivUp(src.cols, block.x), DivUp(src.rows, block.y));

	const float sigma_space2_inv_half = 0.024691358; // 0.5 / (sigma_space * sigma_space)
	const float sigma_color2_inv_half = 0.000555556; // 0.5 / (sigma_color * sigma_color)
	cudaFuncSetCacheConfig(BilateralKernel, cudaFuncCachePreferL1);
	BilateralKernel << <grid, block >> >(dst, src, sigma_space2_inv_half, sigma_color2_inv_half);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void VMapToDepthMapKernel(cv::cuda::PtrStep<float> dDepthMap,
                                     cv::cuda::PtrStepSz<float4> dVMapFloat4)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= dVMapFloat4.cols || y >= dVMapFloat4.rows)
		return;

	dDepthMap.ptr(y)[x] = dVMapFloat4.ptr(y)[x].z;
}

void VMapToDepthMap(cv::cuda::GpuMat& dDepthMap, cv::cuda::GpuMat& dVMapFloat4)
{
	dim3 block(32, 8);
	dim3 grid(DivUp(dVMapFloat4.cols, block.x), DivUp(dVMapFloat4.rows, block.y));

	VMapToDepthMapKernel << <grid, block >> >(dDepthMap, dVMapFloat4);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void RegisterDepthImgKernel(cv::cuda::PtrStep<ushort> dRawDepthImgBuffer,
                            cv::cuda::PtrStepSz<ushort> dRawDepthImg,
                            float cxDepth, float cyDepth, float fxDepth, float fyDepth,
                            float3 RX, float3 RY, float3 RZ, float3 t,
                            float cxColor, float cyColor, float fxColor, float fyColor)
{
	int c = threadIdx.x + blockIdx.x * blockDim.x;
	int r = threadIdx.y + blockIdx.y * blockDim.y;

	if (c >= dRawDepthImg.cols || r >= dRawDepthImg.rows)
		return;

	float3 pos;
	int u, v;
	pos.z = dRawDepthImg.ptr(r)[c];
	pos.x = (c - cxDepth) / fxDepth * pos.z;
	pos.y = (r - cyDepth) / fyDepth * pos.z;
			
	pos = RX * pos.x + RY * pos.y + RZ * pos.z + t;

	u = pos.x / pos.z * fxColor + cxColor + 0.5;
	v = pos.y / pos.z * fyColor + cyColor + 0.5;

	if (u >= 0 && u < dRawDepthImg.cols && v >= 0 && v < dRawDepthImg.rows)
	{
		if (pos.z < dRawDepthImgBuffer.ptr(v)[u])
		{
			dRawDepthImgBuffer.ptr(v)[u] = pos.z;
		}
	}
}

__global__ void RegisterDepthImgWriteBackKernel(cv::cuda::PtrStep<ushort> dRawDepthImg,
                                     cv::cuda::PtrStepSz<ushort> dRawDepthImgBuffer)
{
	int c = threadIdx.x + blockIdx.x * blockDim.x;
	int r = threadIdx.y + blockIdx.y * blockDim.y;

	if (c >= dRawDepthImgBuffer.cols || r >= dRawDepthImgBuffer.rows)
		return;

	if ((int)dRawDepthImgBuffer.ptr(r)[c] == 65535)
	{
		dRawDepthImg.ptr(r)[c] = 0;
	}
	else
	{
		dRawDepthImg.ptr(r)[c] = dRawDepthImgBuffer.ptr(r)[c];
	}
}

void RegisterDepthImg(cv::cuda::GpuMat& dRawDepthImg, cv::cuda::GpuMat& dRawDepthImgBuffer,
                      float cxDepth, float cyDepth, float fxDepth, float fyDepth,
                      float3 RX, float3 RY, float3 RZ, float3 t,
                      float cxColor, float cyColor, float fxColor, float fyColor)
{
	checkCudaErrors(cudaMemset(dRawDepthImgBuffer.data, -1, dRawDepthImgBuffer.step * dRawDepthImgBuffer.rows));

	dim3 block(32, 8);
	dim3 grid(DivUp(dRawDepthImg.cols, block.x), DivUp(dRawDepthImg.rows, block.y));

	RegisterDepthImgKernel << <grid, block >> >(dRawDepthImgBuffer, dRawDepthImg,
	                                            cxDepth, cyDepth, fxDepth, fyDepth,
	                                            RX, RY, RZ, t,
	                                            cxColor, cyColor, fxColor, fyColor);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	RegisterDepthImgWriteBackKernel << <grid, block >> >(dRawDepthImg, dRawDepthImgBuffer);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void CvGpuMatDownloadKernel(uchar* dst, uchar *src, int rows, int cols, int colsInByte, int stepInByte, int elemSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= cols || y >= rows)
		return;

	for (int i = 0; i < elemSize; ++i)
	{
		dst[y * colsInByte + x * elemSize + i] = src[y * stepInByte + x * elemSize + i];
	}
}

void CvGpuMatDownload(uchar* dst, cv::cuda::GpuMat& src)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(src.cols, block.x), getGridDim(src.rows, block.y));

	int elemSize = src.elemSize1();
	CvGpuMatDownloadKernel << <grid, block >> >(dst, src.data, src.rows, src.cols, src.cols * elemSize, src.step, elemSize);
}

__global__ void CvGpuMatUploadKernel(uchar* dst, uchar *src, int rows, int cols, int colsInByte, int stepInByte, int elemSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= cols || y >= rows)
		return;

	for (int i = 0; i < elemSize; ++i)
	{
		dst[y * stepInByte + x * elemSize + i] = src[y * colsInByte + x * elemSize + i];
	}
}

void CvGpuMatUpload(cv::cuda::GpuMat& dst, uchar* src)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(dst.cols, block.x), getGridDim(dst.rows, block.y));

	int elemSize = dst.elemSize1();
	CvGpuMatUploadKernel << <grid, block >> >(dst.data, src, dst.rows, dst.cols, dst.cols * elemSize, dst.step, elemSize);
}

__global__ void PruneDepthKernel(cv::cuda::PtrStep<ushort> dRawDepthImg,
                                 cv::cuda::PtrStep<ushort> dFilteredDepthImg,
                                 cv::cuda::PtrStep<float> dRawDepthImg32F,
                                 cv::cuda::PtrStep<float> dFilteredDepthImg32F,
                                 cv::cuda::PtrStepSz<uchar> dPruneMat)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dPruneMat.cols || y >= dPruneMat.rows)
		return;

	if (dFilteredDepthImg32F.ptr(y)[x] < MYEPS)
	{
		dRawDepthImg.ptr(y)[x] = 0;
		dFilteredDepthImg.ptr(y)[x] = 0;
		dRawDepthImg32F.ptr(y)[x] = 0.0;
	}
#if 1
	if (dPruneMat.ptr(y)[x] == 255)
	{
		dRawDepthImg.ptr(y)[x] = 0;
		dFilteredDepthImg.ptr(y)[x] = 0;
		dRawDepthImg32F.ptr(y)[x] = 0.0;
		dFilteredDepthImg32F.ptr(y)[x] = 0.0;
	}
#endif
}

void PruneDepth(cv::cuda::GpuMat& dRawDepthImg,
                cv::cuda::GpuMat& dFilteredDepthImg,
                cv::cuda::GpuMat& dRawDepthImg32F,
                cv::cuda::GpuMat& dFilteredDepthImg32F,
                cv::cuda::GpuMat& dPruneMat)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(dFilteredDepthImg.cols, block.x), getGridDim(dFilteredDepthImg.rows, block.y));

	PruneDepthKernel << <grid, block >> >(dRawDepthImg, dFilteredDepthImg, dRawDepthImg32F, dFilteredDepthImg32F, dPruneMat);
}

__global__ void FilterDepthUsingDepthKernel(cv::cuda::PtrStepSz<ushort> dRawDepthImg,
											cv::cuda::PtrStep<ushort> dFilteredDepthImg,
											cv::cuda::PtrStep<float> dRawDepthImg32F,
											cv::cuda::PtrStep<float> dFilteredDepthImg32F,
											float depthThresh)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dRawDepthImg.cols || y >= dRawDepthImg.rows)
		return;

	if (dRawDepthImg.ptr(y)[x] > depthThresh || dRawDepthImg.ptr(y)[x] < 400.0f)
	{
		dRawDepthImg.ptr(y)[x] = 0;
		dFilteredDepthImg.ptr(y)[x] = 0;
	}
	if (dRawDepthImg32F.ptr(y)[x] > depthThresh / 1000.0f || dRawDepthImg32F.ptr(y)[x] < 400.0f / 1000.0f)
	{
		dRawDepthImg32F.ptr(y)[x] = 0.0f;
		dFilteredDepthImg32F.ptr(y)[x] = 0.0f;
	}
}

void FilterDepthUsingDepth(cv::cuda::GpuMat& dRawDepthImg,
						   cv::cuda::GpuMat& dFilteredDepthImg,
						   cv::cuda::GpuMat& dRawDepthImg32F,
						   cv::cuda::GpuMat& dFilteredDepthImg32F,
						   float depthThresh)
{
	dim3 block(32, 8);
	dim3 grid(getGridDim(dFilteredDepthImg.cols, block.x), getGridDim(dFilteredDepthImg.rows, block.y));

	FilterDepthUsingDepthKernel << <grid, block >> > (dRawDepthImg, dFilteredDepthImg, dRawDepthImg32F, dFilteredDepthImg32F, depthThresh);
}