#pragma once

#include <cassert>
#include <opencv2/opencv.hpp>
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <thrust/device_vector.h>
//#include <pangolin/gl/glplatform.h>

#define CPU_REGISTER 0
#define USE_XTION 1
#define USE_STRUCTURE_SENSOR 0

#define FRAG_SIZE 50

#define MAX_CLOSURE_NUM_EACH_FRAG 8
#define MAX_FRAG_NUM 10
//#define NODE_NUM_EACH_FRAG 256
#define NODE_NUM_EACH_FRAG GlobalParameter::GetNodeNumEachFrag()
//#define NODE_NUM_EACH_FRAG 8
//#define SAMPLED_VERTEX_NUM_EACH_FRAG 32768
#define SAMPLED_VERTEX_NUM_EACH_FRAG GlobalParameter::GetSampledVertexNumEachFrag()
#define MAX_VERTEX_NUM_EACH_FRAG 450000
#define MAX_VERTEX_NUM 3000000
#define MAX_NEAR_NODE_NUM_NODE 8  
#define MAX_NEAR_NODE_NUM_VERTEX 4

#define MYEPS 1.0e-24
#define RAW_PTR(thrust_device_vector) thrust::raw_pointer_cast(&(thrust_device_vector)[0])

namespace GlobalParameter
{
	extern __host__ __device__ int GetNodeNumEachFrag();
	extern __host__ void SetNodeNumEachFrag(int node_num);

	extern __host__ __device__ int GetSampledVertexNumEachFrag();
	extern __host__ void SetSampledVertexNumEachFrag(int vertex_num);
}

/*#define xCheckGlDieOnError() pangolin::_xCheckGlDieOnError( __FILE__, __LINE__ );
namespace pangolin {
	inline void _xCheckGlDieOnError(const char *sFile, const int nLine)
	{
		GLenum glError = glGetError();
		if (glError != GL_NO_ERROR) {
			pango_print_error("OpenGL Error: %s (%d)\n", glErrorString(glError), glError);
			pango_print_error("In: %s, line %d\n", sFile, nLine);
			std::exit(0);
		}
	}
}*/

struct xMatrix4f
{
	xMatrix4f();
	xMatrix4f(float *data);

	float *data();
	float4 col(int idx);
	xMatrix4f inverse();
	xMatrix4f orthogonalization();
	void print();
	friend xMatrix4f operator*(xMatrix4f &a, xMatrix4f &b);

	float4 m_data[4];
};

void PCA3x3(float3& zAxis, float *data);

struct Box
{
	int m_left, m_right, m_top, m_bottom;
	float m_score;
	Box() : m_score(0.0f) {}
	Box(int left, int right, int top, int bottom) : m_left(left), m_right(right), m_top(top), m_bottom(bottom), m_score(0.0f) {}
};

class Resolution
{
public:
	static const Resolution& getInstance(int depthWidth = 0, int depthHeight = 0,
		int colorWidth = 0, int colorHeight = 0,
		int fullColorWidth = 0, int fullColorHeight = 0,
		float scale = 1.0f)
	{
		static const Resolution instance((int)(depthWidth * scale),
		                                 (int)(depthHeight * scale),
		                                 (int)(colorWidth * scale),
		                                 (int)(colorHeight * scale),
		                                 (int)(fullColorWidth * scale),
		                                 (int)(fullColorHeight * scale));
		return instance;
	}

	const int& width() const
	{
		return m_colorWidth;
	}

	const int& height() const
	{
		return m_colorHeight;
	}

	const int& cols() const
	{
		return m_colorWidth;
	}

	const int& rows() const
	{
		return m_colorHeight;
	}

	const int& depthWidth() const
	{
		return m_depthWidth;
	}

	const int& depthHeight() const
	{
		return m_depthHeight;
	}

	const int& depthCols() const
	{
		return m_depthWidth;
	}

	const int& depthRows() const
	{
		return m_depthHeight;
	}

	const int& numPixels() const
	{
		return m_colorNumPixels;
	}

	const int& fullColorWidth() const
	{
		return m_fullColorWidth;
	}

	const int& fullColorHeight() const
	{
		return m_fullColorHeight;
	}

	const int& fullColorCols() const
	{
		return m_fullColorWidth;
	}

	const int& fullColorRows() const
	{
		return m_fullColorHeight;
	}

	const int& fullColorNumPixels() const
	{
		return m_fullColorNumPixels;
	}

private:
	Resolution(int depthWidth, int depthHeight,
		int colorWidth, int colorHeight,
		int fullColorWidth, int fullColorHeight)
		: m_depthWidth(depthWidth),
		m_depthHeight(depthHeight),
		m_colorWidth(colorWidth),
		m_colorHeight(colorHeight),
		m_fullColorWidth(fullColorWidth),
		m_fullColorHeight(fullColorHeight),
		m_colorNumPixels(colorWidth * colorHeight),
		m_fullColorNumPixels(fullColorWidth * fullColorHeight)
	{
		assert(m_colorWidth > 0 && m_colorHeight > 0 && "You haven't initialised the Resolution class!");
	}

	const int m_colorWidth;
	const int m_colorHeight;
	const int m_depthWidth;
	const int m_depthHeight;
	const int m_fullColorWidth;
	const int m_fullColorHeight;
	const int m_colorNumPixels;
	const int m_fullColorNumPixels;
};

class Intrinsics
{
public:
	static const Intrinsics & getInstance(float fx = 0, float fy = 0, float cx = 0, float cy = 0, float imgScale = 1.0f)
	{
		static const Intrinsics instance(fx * imgScale, fy * imgScale, cx * imgScale, cy * imgScale);
		return instance;
	}

	const float & fx() const
	{
		return fx_;
	}

	const float & fy() const
	{
		return fy_;
	}

	const float & cx() const
	{
		return cx_;
	}

	const float & cy() const
	{
		return cy_;
	}

private:
	Intrinsics(float fx, float fy, float cx, float cy)
		: fx_(fx),
		fy_(fy),
		cx_(cx),
		cy_(cy)
	{
		assert(fx != 0 && fy != 0 && "You haven't initialised the Intrinsics class!");
	}

	const float fx_, fy_, cx_, cy_;
};

struct CSRType
{
	int * m_ia;
	int * m_ja;
	float * m_a;
};

struct CSCType
{
	int * m_ia;
	int * m_ja;
	float * m_a;
};

struct VBOType
{
	float4 posConf;
	float4 colorTime;
	float4 normalRad;
};

struct __builtin_align__(16) my_int4
{
	union
	{
		int data[4];
	};
};

struct __builtin_align__(16) my_float4
{
	union
	{
		float data[4];
	};
};

inline float CalculateBlurScore(const cv::Mat_<uchar> &img)
{
	cv::Mat_<float> hV, hH, bVer, bHor;
	cv::Mat_<float> dFVer(img.size()), dFHor(img.size()), dBVer(img.size()), dBHor(img.size()), dVVer(img.size()), dVHor(img.size());
	hV = cv::Mat_<float>::ones(1, 9)*(1.0 / 9.0);
	hH = hV.t();
	cv::filter2D(img, bVer, bVer.depth(), hV);
	cv::filter2D(img, bHor, bHor.depth(), hH);
	for (int y = 1; y < img.rows; ++y)
	{
		for (int x = 1; x < img.cols; ++x)
		{
#if 0
			if (img[y][x] == 0 || img[y - 1][x] == 0 || img[y][x - 1] == 0)
			{
				dFVer[y][x] = dFHor[y][x] = dBVer[y][x] = dBHor[y][x] = 0;
				continue;
			}
#endif
			dFVer[y][x] = abs(img[y][x] - img[y - 1][x]);
			dFHor[y][x] = abs(img[y][x] - img[y][x - 1]);
			dBVer[y][x] = abs(bVer[y][x] - bVer[y - 1][x]);
			dBHor[y][x] = abs(bHor[y][x] - bHor[y][x - 1]);
		}
	}
	for (int y = 1; y < img.rows; ++y)
	{
		for (int x = 1; x < img.cols; ++x)
		{
			dVVer[y][x] = MAX(0, dFVer[y][x] - dBVer[y][x]);
			dVHor[y][x] = MAX(0, dFHor[y][x] - dBHor[y][x]);	
		}
	}
	double sFVer = 0.0, sFHor = 0.0, sVVer = 0.0, sVHor = 0.0;
	for (int y = 1; y < img.rows; ++y)
	{
		for (int x = 1; x < img.cols; ++x)
		{
			sFVer += dFVer[y][x];
			sFHor += dFHor[y][x];
			sVVer += dVVer[y][x];
			sVHor += dVHor[y][x];
		}
	}
	/*
	std::cout << sVVer << std::endl;
	std::cout << sFVer << std::endl;
	std::cout << sVHor << std::endl;
	std::cout << sFHor << std::endl;
	*/
	
	return MAX(1 - sVVer / sFVer, 1 - sVHor / sFHor);
}

inline int IsFrag(const int64_t &time)
{
	if (time >= FRAG_SIZE && (time % FRAG_SIZE) == 0)
		return 1;
	return 0;
}

static inline int DivUp(int total, int grain) { return (total + grain - 1) / grain; }

static inline int getGridDim(int x, int y)
{
	return (x + y - 1) / y;
}

struct ColorFrameInfo
{
	int rows;
	int cols;
	int cbCompressedLength;
	int crCompressedLength;
	int frameIdx;
};

struct DepthFrameInfo
{
	int rows;
	int cols;
	int msbCompressedLength; // big end half
	int lsbCompressedLength; // small end half
	int frameIdx;
};

struct ImuMsg
{
	double timeStamp;
	double3 acc;
	double3 gyr;
};

struct DepthColorFrameInfo
{
	int keyFrameIdxEachFrag;//0 for non-key frame
	int colorRows;
	int colorCols;
	int depthRows;
	int depthCols;
	int cbCompressedLength;
	int crCompressedLength;
	int msbCompressedLength;
	int lsbCompressedLength;
	int frameIdx;
	int imuMeasurementSize;
};

struct FullColorFrameInfo
{
	int keyFrameNum;
	int frameIdx;
	int colorRows;
	int colorCols;
	int colorBytePerRow;
};

#define HEAD_SIZE 1024
typedef std::vector<ImuMsg> ImuMeasurements;
typedef double3 Gravity;

int CheckNanVertex(float4 *_d_vertex, int _N);
int CheckNanVertex(float *_d_vertex, int _N);