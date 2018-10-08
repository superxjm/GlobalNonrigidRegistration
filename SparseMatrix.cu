#include "SparseMatrix.h"

#include <cusparse_v2.h>
#include <helper_cuda.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>

__global__ void AddVecVecKernel(float* vec1, float* vec2, int length)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < length)
	{
		vec1[idx] += vec2[idx];
	}
}

void AddVecVec(float* vec1, float* vec2, int length)
{
	int block = 512, grid = (block + length - 1) / block;
	AddVecVecKernel << <grid, block >> >(vec1, vec2, length);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void AddVecVecSE3Kernel(float* vec1, float* vec2, int length, bool doOrtho)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= length)
	{
		return;
	}
	double rx, ry, rz, tx, ty, tz, theta;

	rx = vec2[6 * idx];
	ry = vec2[6 * idx + 1];
	rz = vec2[6 * idx + 2];
	tx = vec2[6 * idx + 3];
	ty = vec2[6 * idx + 4];
	tz = vec2[6 * idx + 5];

	theta = sqrt(rx * rx + ry * ry + rz * rz);
	float* vars_ptr = vec1 + 12 * idx;
	//printf("theta: %f %f %f %f, ", theta, rx, ry, rz);
	if (theta >= DBL_EPSILON)
	{
		const double I[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

		float vars_tmp[9];
		vars_tmp[0] = vars_ptr[0];
		vars_tmp[1] = vars_ptr[1];
		vars_tmp[2] = vars_ptr[2];
		vars_tmp[3] = vars_ptr[3];
		vars_tmp[4] = vars_ptr[4];
		vars_tmp[5] = vars_ptr[5];
		vars_tmp[6] = vars_ptr[6];
		vars_tmp[7] = vars_ptr[7];
		vars_tmp[8] = vars_ptr[8];

		double c = cos(theta);
		double s = sin(theta);
		double c1 = 1. - c;
		double itheta = theta ? 1. / theta : 0.;

		rx *= itheta;
		ry *= itheta;
		rz *= itheta;

		double rrt[] = {rx * rx, rx * ry, rx * rz, rx * ry, ry * ry, ry * rz, rx * rz, ry * rz, rz * rz};
		double _r_x_[] = {0, -rz, ry, rz, 0, -rx, -ry, rx, 0};

		double R[9];
		for (int k = 0; k < 9; k++)
		{
			R[k] = c * I[k] + c1 * rrt[k] + s * _r_x_[k];
		}

		vars_ptr[0] = R[0] * vars_tmp[0] + R[1] * vars_tmp[3] + R[2] * vars_tmp[6];
		vars_ptr[1] = R[0] * vars_tmp[1] + R[1] * vars_tmp[4] + R[2] * vars_tmp[7];
		vars_ptr[2] = R[0] * vars_tmp[2] + R[1] * vars_tmp[5] + R[2] * vars_tmp[8];

		vars_ptr[3] = R[3] * vars_tmp[0] + R[4] * vars_tmp[3] + R[5] * vars_tmp[6];
		vars_ptr[4] = R[3] * vars_tmp[1] + R[4] * vars_tmp[4] + R[5] * vars_tmp[7];
		vars_ptr[5] = R[3] * vars_tmp[2] + R[4] * vars_tmp[5] + R[5] * vars_tmp[8];

		vars_ptr[6] = R[6] * vars_tmp[0] + R[7] * vars_tmp[3] + R[8] * vars_tmp[6];
		vars_ptr[7] = R[6] * vars_tmp[1] + R[7] * vars_tmp[4] + R[8] * vars_tmp[7];
		vars_ptr[8] = R[6] * vars_tmp[2] + R[7] * vars_tmp[5] + R[8] * vars_tmp[8];

		if (doOrtho)
		{
			double factor1, factor2, norm1, norm2;
			norm1 = vars_ptr[0] * vars_ptr[0] + vars_ptr[1] * vars_ptr[1] + vars_ptr[2] * vars_ptr[2];

			factor1 = (vars_ptr[3] * vars_ptr[0] + vars_ptr[4] * vars_ptr[1] + vars_ptr[5] * vars_ptr[2]) / norm1;

			vars_ptr[3] = vars_ptr[3] - factor1 * vars_ptr[0];
			vars_ptr[4] = vars_ptr[4] - factor1 * vars_ptr[1];
			vars_ptr[5] = vars_ptr[5] - factor1 * vars_ptr[2];

			norm2 = vars_ptr[3] * vars_ptr[3] + vars_ptr[4] * vars_ptr[4] + vars_ptr[5] * vars_ptr[5];
			factor1 = (vars_ptr[6] * vars_ptr[0] + vars_ptr[7] * vars_ptr[1] + vars_ptr[8] * vars_ptr[2]) / norm1;
			factor2 = (vars_ptr[6] * vars_ptr[3] + vars_ptr[7] * vars_ptr[4] + vars_ptr[8] * vars_ptr[5]) / norm2;

			vars_ptr[6] = vars_ptr[6] - factor1 * vars_ptr[0] - factor2 * vars_ptr[3];
			vars_ptr[7] = vars_ptr[7] - factor1 * vars_ptr[1] - factor2 * vars_ptr[4];;
			vars_ptr[8] = vars_ptr[8] - factor1 * vars_ptr[2] - factor2 * vars_ptr[5];;

			norm1 = sqrt(vars_ptr[0] * vars_ptr[0] + vars_ptr[1] * vars_ptr[1] + vars_ptr[2] * vars_ptr[2]);
			vars_ptr[0] /= norm1;
			vars_ptr[1] /= norm1;
			vars_ptr[2] /= norm1;
			norm1 = sqrt(vars_ptr[3] * vars_ptr[3] + vars_ptr[4] * vars_ptr[4] + vars_ptr[5] * vars_ptr[5]);
			vars_ptr[3] /= norm1;
			vars_ptr[4] /= norm1;
			vars_ptr[5] /= norm1;
			norm1 = sqrt(vars_ptr[6] * vars_ptr[6] + vars_ptr[7] * vars_ptr[7] + vars_ptr[8] * vars_ptr[8]);
			vars_ptr[6] /= norm1;
			vars_ptr[7] /= norm1;
			vars_ptr[8] /= norm1;
		}
	}

	vars_ptr[9] += tx;
	vars_ptr[10] += ty;
	vars_ptr[11] += tz;
}

void AddVecVecSE3(float* dVec1, float* dVec2, int length)
{
	int block = 256, grid = (block + length - 1) / block;
	AddVecVecSE3Kernel << <grid, block >> >(dVec1, dVec2, length, false);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}
