//#include "stdafx.h"

#include "Helpers/xUtils.h"
#include "Helpers/UtilsMath.h"

#include <Eigen/Eigen>
#include <Eigen/Eigenvalues> 

void PCA3x3(float3& zAxis, float *data)
{
	Eigen::Matrix3f covMat(data);
	//std::cout << "covMat: " << covMat << std::endl;

	Eigen::EigenSolver<Eigen::Matrix3f> es(covMat);
	Eigen::Matrix3f V = es.pseudoEigenvectors();
	Eigen::Matrix3f D = es.pseudoEigenvalueMatrix();
#if 0
	std::cout << "The pseudo-eigenvalue matrix D is:" << std::endl << D << std::endl;
	std::cout << "The pseudo-eigenvector matrix V is:" << std::endl << V << std::endl;
	//std::exit(0);
#endif
	if (D(2, 2) < D(0, 0) && D(2, 2) < D(1, 1))
	{
		zAxis.x = V(0, 2);
		zAxis.y = V(1, 2);
		zAxis.z = V(2, 2);
		return;
	}
	if (D(1, 1) < D(0, 0) && D(1, 1) < D(2, 2))
	{
		zAxis.x = V(0, 1);
		zAxis.y = V(1, 1);
		zAxis.z = V(2, 1);
		return;
	}
	if (D(0, 0) < D(1, 1) && D(0, 0) < D(2, 2))
	{
		zAxis.x = V(0, 0);
		zAxis.y = V(1, 0);
		zAxis.z = V(2, 0);
		return;
	}
	
#if 0
	eigenVectors[0].x = V(0, 0);
	eigenVectors[0].y = V(1, 0);
	eigenVectors[0].z = V(2, 0);

	eigenVectors[1].x = V(0, 1);
	eigenVectors[1].y = V(1, 1);
	eigenVectors[1].z = V(2, 1);

	eigenVectors[2].x = V(0, 2);
	eigenVectors[2].y = V(1, 2);
	eigenVectors[2].z = V(2, 2);
#endif
#if 0
	eigenVectors[0].x = V(0, 0);
	eigenVectors[0].y = V(0, 1);
	eigenVectors[0].z = V(0, 2);

	eigenVectors[1].x = V(1, 0);
	eigenVectors[1].y = V(1, 1);
	eigenVectors[1].z = V(1, 2);

	eigenVectors[2].x = V(2, 0);
	eigenVectors[2].y = V(2, 1);
	eigenVectors[2].z = V(2, 2);
#endif
}

xMatrix4f::xMatrix4f()
{
	
}

xMatrix4f::xMatrix4f(float *data)
{
	memcpy(this->data(), data, 4 * sizeof(float4));
}

float *xMatrix4f::data()
{
	return reinterpret_cast<float *>(&(m_data[0]));
}

float4 xMatrix4f::col(int idx)
{
	return m_data[idx];
}

xMatrix4f xMatrix4f::inverse()
{
	Eigen::Matrix4f tmp, tmpInv;
	xMatrix4f res;
	memcpy(tmp.data(), this->data(), 4 * sizeof(float4));
	tmpInv = tmp.inverse();
	memcpy(res.data(), tmpInv.data(), 4 * sizeof(float4));

	return res;
}

xMatrix4f xMatrix4f::orthogonalization()
{
	Eigen::Matrix4f mat, matSVD;
	xMatrix4f res;
	memcpy(mat.data(), this->data(), 4 * sizeof(float4));
	matSVD = mat;
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(mat.topLeftCorner(3, 3), Eigen::ComputeFullV | Eigen::ComputeFullU);
	matSVD.topLeftCorner(3, 3) = svd.matrixU() * svd.matrixV().transpose();
	memcpy(res.data(), matSVD.data(), 4 * sizeof(float4));

	return res;
}

xMatrix4f operator*(xMatrix4f &a, xMatrix4f &b)
{
	xMatrix4f res;
	res.m_data[0] = a.m_data[0] * b.m_data[0].x + a.m_data[1] * b.m_data[0].y
		+ a.m_data[2] * b.m_data[0].z + a.m_data[3] * b.m_data[0].w;
	res.m_data[1] = a.m_data[0] * b.m_data[1].x + a.m_data[1] * b.m_data[1].y
		+ a.m_data[2] * b.m_data[1].z + a.m_data[3] * b.m_data[1].w;
	res.m_data[2] = a.m_data[0] * b.m_data[2].x + a.m_data[1] * b.m_data[2].y
		+ a.m_data[2] * b.m_data[2].z + a.m_data[3] * b.m_data[2].w;
	res.m_data[3] = a.m_data[0] * b.m_data[3].x + a.m_data[1] * b.m_data[3].y
		+ a.m_data[2] * b.m_data[3].z + a.m_data[3] * b.m_data[3].w;
	return res;
}

void xMatrix4f::print()
{
	printf("Matrix4f:\n");
	printf("%.15f, %.15f, %.15f, %.15f\n", this->m_data[0].x, this->m_data[1].x, this->m_data[2].x, this->m_data[3].x);
	printf("%.15f, %.15f, %.15f, %.15f\n", this->m_data[0].y, this->m_data[1].y, this->m_data[2].y, this->m_data[3].y);
	printf("%.15f, %.15f, %.15f, %.15f\n", this->m_data[0].z, this->m_data[1].z, this->m_data[2].z, this->m_data[3].z);
	printf("%.15f, %.15f, %.15f, %.15f\n", this->m_data[0].w, this->m_data[1].w, this->m_data[2].w, this->m_data[3].w);
}