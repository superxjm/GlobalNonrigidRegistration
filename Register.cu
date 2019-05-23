#include "Register.h"


Register::Register()
{
	deform_ = nullptr;
}


Register::~Register()
{
}

void Register::CreateDeformation(int _vbo_id, std::vector<int> _sum_vertex_num)
{
	if (deform_ != nullptr)
	{
		delete deform_;
		cudaGraphicsUnmapResources(1, &vbo_CUDA_, 0);
	}
	
	vbo_id_ = _vbo_id;
	sum_vertex_num_ = _sum_vertex_num;
	CalcEachFragColor();

	imgScale = 1.0f; dispScal = 2.0f;
	depthWidth = 640; depthHeight = 480; colorWidth = 640; colorHeight = 480;
	fx = 525.0; fy = 525.0; cx = 319.5; cy = 239.5;
	Resolution::getInstance(depthWidth, depthHeight, colorWidth, colorHeight, imgScale);
	Intrinsics::getInstance(fx, fy, cx, cy, imgScale);
	color_.create(colorHeight, colorWidth, CV_8UC3);
	full_color_.create(colorHeight, colorWidth, CV_8UC3);
	gray_.create(colorHeight, colorWidth);
	d_gray_ = cv::cuda::GpuMat(colorHeight, colorWidth, CV_8UC1);

	cudaGraphicsGLRegisterBuffer(&vbo_CUDA_, _vbo_id, cudaGraphicsMapFlagsNone);
	
	cudaGraphicsMapResources(1, &vbo_CUDA_, 0);
	size_t num_bytes;// = _vertex_num * sizeof(VBOType);
	cudaGraphicsResourceGetMappedPointer((void**)&vbo_, &num_bytes, vbo_CUDA_);

	deform_ = new xDeformation(frag_id_, vbo_);

	cv::Mat_<float> mat;
	mat = mat.eye(4, 4);
	xMatrix4f m(reinterpret_cast<float*>(mat.data));

	for (int i = 0; i<_sum_vertex_num.size() - 1; i++)
		deform_->addData(color_, full_color_, gray_, d_gray_, m);

	for (int i = 0; i < _sum_vertex_num.size() - 1; i++)
	{
		frag_id_ = i;
		deform_->prepareData(_sum_vertex_num[i + 1]);
	}

	//cudaGraphicsUnmapResources(1, &vbo_CUDA_, 0);
}

/*void Register::RegisterRun(int _frag_id, std::vector<int> _sum_vertex_num)
{

	cv::Mat_<float> mat;
	mat = mat.eye(4,4);
	xMatrix4f m(reinterpret_cast<float*>(mat.data));

	//cudaGraphicsGLRegisterBuffer(&vbo_CUDA_, vbo_id_, cudaGraphicsMapFlagsNone);
	//size_t num_bytes;
	//cudaGraphicsResourceGetMappedPointer((void**)&vbo_, &num_bytes, vbo_CUDA_);

	frag_id_ = _frag_id;
	deform_->addData(color_, full_color_, gray_, d_gray_, m);
	//deform_->addDataWithKeyFrame(color_, full_color_, gray_, m);
	deform_->deform(&m, vbo_, _sum_vertex_num[frag_id_ + 1], 0);

	//cudaGraphicsUnmapResources(1, &vbo_CUDA_, 0);
	CopyData();
}*/

void Register::FindCorrespondencePoints(std::vector<cv::Mat_<float>> &_camera_pose,
	std::vector<cv::Mat_<float>> &_camera_fxfycxcy,
	int _method, float _dist_thresh1, float _dist_thresh2, float _angle_thresh,
	int _width, int _height)
{
	deform_->findMatchingPoints(_camera_pose, _camera_fxfycxcy, _method, _dist_thresh1, _dist_thresh2, _angle_thresh, _width, _height);

	CopyData();
}

void Register::RegisterRun(int _iter_num, std::vector<int> _sum_vertex_num,
	std::vector<cv::Mat_<float>> &_camera_pose,
	std::vector<cv::Mat_<float>> &_camera_fxfycxcy,
	int _method, float _dist_thresh1, float _dist_thresh2, float _angle_thresh,
	int _width, int _height)
{
	if (deform_ == nullptr) return;

	deform_->deformToghter(_iter_num, vbo_, _sum_vertex_num, _camera_pose, _camera_fxfycxcy, 
		_method, _dist_thresh1, _dist_thresh2, _angle_thresh, _width, _height);

	CopyData();
}

void Register::CalcEachFragColor()
{
	cv::Vec4b constantColor[10] = { cv::Vec4b(255, 128,128,255), cv::Vec4b(255, 255,128,255),  cv::Vec4b(128, 255,128,255), cv::Vec4b(0, 128,255,255),cv::Vec4b(128, 0,64,255),
		cv::Vec4b(128, 64,64,255), cv::Vec4b(0, 128,128,255), cv::Vec4b(128, 255,64,255), cv::Vec4b(128, 128,128,255), cv::Vec4b(192, 192,192,255) };

	h_color_.resize(sum_vertex_num_.back());
	for (int i = 0; i < sum_vertex_num_.size() - 1; i++)
	{
		for (int j = sum_vertex_num_[i]; j < sum_vertex_num_[i + 1]; j++)
		{
			h_color_[j] = constantColor[i % 10];
		}
	}
}

void Register::CopyData()
{
	h_nodes_.resize(deform_->m_inputData->m_source.m_nodeNum);
	checkCudaErrors(cudaMemcpy(h_nodes_.data(), RAW_PTR(deform_->m_inputData->m_source.m_dNodeVec), deform_->m_inputData->m_source.m_nodeNum * sizeof(float4), cudaMemcpyDeviceToHost));
	for (int i = 0; i < h_nodes_.size(); i++)
	{
		h_nodes_[i][3] = 1;
	}

	h_deformed_.resize(deform_->m_inputData->m_source.m_vertexNum);
	checkCudaErrors(cudaMemcpy(h_deformed_.data(), RAW_PTR(deform_->m_inputData->m_deformed.m_dVertexVec), deform_->m_inputData->m_source.m_vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));
	for (int i = 0; i < h_deformed_.size(); i++)
	{
		h_deformed_[i][3] = 1;
	}

	h_source_ = deform_->last_deformed_vertex_;
	for (int i = 0; i < h_source_.size(); i++)
	{
		h_source_[i][3] = 1;
	}


	std::vector<int> h_sample_index(deform_->m_inputData->m_source.m_sampledVertexNum);
	checkCudaErrors(cudaMemcpy(h_sample_index.data(), RAW_PTR(deform_->m_inputData->m_source.m_dSampledVertexIdxVec), h_sample_index.size() * sizeof(int), cudaMemcpyDeviceToHost));
	h_deformed_sample_vertex_.resize(h_sample_index.size());
	h_deformed_sample_color_.resize(h_sample_index.size());
	for (int i = 0; i < h_sample_index.size(); i++)
	{
		h_deformed_sample_vertex_[i] = h_deformed_[h_sample_index[i]];
		h_deformed_sample_color_[i] = h_color_[h_sample_index[i]];
	}
	h_source_sample_vertex_.resize(h_sample_index.size());
	h_source_sample_color_.resize(h_sample_index.size());
	for (int i = 0; i < h_sample_index.size(); i++)
	{
		h_source_sample_vertex_[i] = h_source_[h_sample_index[i]];
		h_source_sample_color_[i] = h_color_[h_sample_index[i]];
	}

	h_source2deformed_vertex_.resize(h_source_.size() + h_deformed_.size());
	h_source2deformed_index_.resize(h_source_.size() + h_deformed_.size());
	h_source2deformed_color_.resize(h_source_.size() + h_deformed_.size());
	for (int i = 0; i < h_source_.size(); i++)
	{
		h_source2deformed_vertex_[2 * i] = h_source_[i];
		h_source2deformed_vertex_[2 * i + 1] = h_deformed_[i];
		h_source2deformed_color_[2 * i] = h_color_[i];
		h_source2deformed_color_[2 * i + 1] = h_color_[i];
		h_source2deformed_index_[2 * i] = 2 * i;
		h_source2deformed_index_[2 * i + 1] = 2 * i + 1;
	}


	h_corr_vertex_.resize(deform_->m_matchingPointNum * 2);
	h_corr_index_.resize(deform_->m_matchingPointNum * 2);
	h_corr_color_.resize(deform_->m_matchingPointNum * 2);
	checkCudaErrors(cudaMemcpy(h_corr_index_.data(), deform_->m_dMatchingPointIndices, deform_->m_matchingPointNum * 2 * sizeof(int), cudaMemcpyDeviceToHost));
	int num = 0;
	for (int i = 0; i < deform_->m_matchingPointNum; i++)
	{
		if (h_corr_index_[2 * i] != -1 && h_corr_index_[2 * i + 1] != -1)
		{
			//h_corr_vertex_[2 * num] = h_source_[h_corr_index_[2 * i]];
			//h_corr_vertex_[2 * num + 1] = h_source_[h_corr_index_[2 * i + 1]];
			h_corr_vertex_[2 * num] = h_deformed_[h_corr_index_[2 * i]];
			h_corr_vertex_[2 * num + 1] = h_deformed_[h_corr_index_[2 * i + 1]];
			h_corr_color_[2 * num] = h_color_[h_corr_index_[2 * i]];
			h_corr_color_[2 * num + 1] = h_color_[h_corr_index_[2 * i + 1]];
			num++;
		}
	}
	h_corr_vertex_.resize(num * 2);
	h_corr_index_.resize(num * 2);
	h_corr_color_.resize(num * 2);
	for (int i = 0; i < num; i++)
	{
		h_corr_index_[2 * i] = 2 * i;
		h_corr_index_[2 * i + 1] = 2 * i + 1;
	}
	//printf("corr num: %d\n", num);


	h_deformed_normal_.resize(deform_->m_inputData->m_source.m_vertexNum);
	checkCudaErrors(cudaMemcpy(h_deformed_normal_.data(), RAW_PTR(deform_->m_inputData->m_deformed.m_dNormalVec), deform_->m_inputData->m_source.m_vertexNum * sizeof(float4), cudaMemcpyDeviceToHost));


}

std::vector<cv::Vec4f>& Register::get_nodes()
{
	return h_nodes_;
}

std::vector<cv::Vec4f>& Register::get_deformed()
{
	return h_deformed_;
}

std::vector<cv::Vec4b>& Register::get_deformed_color()
{
	return h_color_;
}

std::vector<cv::Vec4f>& Register::get_deformed_normal()
{
	return h_deformed_normal_;
}

std::vector<cv::Vec4f>& Register::get_source()
{
	return h_source_;
}

std::vector<cv::Vec4b>& Register::get_source_color()
{
	return h_color_;
}

std::vector<cv::Vec4f>& Register::get_deformed_sample_vertex()
{
	return h_deformed_sample_vertex_;
}

std::vector<cv::Vec4b>& Register::get_deformed_sample_color()
{
	return h_deformed_sample_color_;
}

std::vector<cv::Vec4f>& Register::get_source_sample_vertex()
{
	return h_source_sample_vertex_;
}

std::vector<cv::Vec4b>& Register::get_source_sample_color()
{
	return h_source_sample_color_;
}

std::vector<cv::Vec4f>& Register::get_source2deformed_vertex()
{
	/*h_source2deformed_vertex_.resize(h_source_.size() + h_deformed_.size());
	h_source2deformed_index_.resize(h_source_.size() + h_deformed_.size());
	h_source2deformed_color_.resize(h_source_.size() + h_deformed_.size());

	memcpy(h_source2deformed_vertex_.data(), h_source_.data(), h_source_.size() * sizeof(float4));
	memcpy(h_source2deformed_vertex_.data() + h_source_.size(), h_deformed_.data(), h_deformed_.size() * sizeof(float4));*/

	/*for (int i = 0; i < h_source_.size(); i++)
	{
		h_source2deformed_index_[2 * i] = i;
		h_source2deformed_index_[2 * i + 1] = h_source_.size() + i;
		h_source2deformed_color_[2 * i] = cv::Vec4b(0, 255, 0, 255);
		h_source2deformed_color_[2 * i + 1] = cv::Vec4b(0, 0, 255, 255);
	}*/

	/*h_source2deformed_vertex_.resize(h_deformed_.size());
	h_source2deformed_index_.resize(h_deformed_.size());
	h_source2deformed_color_.resize(h_deformed_.size());

	memcpy(h_source2deformed_vertex_.data(), h_deformed_.data(), h_deformed_.size() * sizeof(float4));
	
	for (int i = 0; i < h_deformed_.size(); i++)
	{
		h_source2deformed_color_[i] = cv::Vec4b(0, 0, 255, 255);
	}*/

	return h_source2deformed_vertex_;
}

std::vector<int>& Register::get_source2deformed_index()
{
	return h_source2deformed_index_;
}

std::vector<cv::Vec4b>& Register::get_source2deformed_color()
{
	return h_source2deformed_color_;
}

std::vector<cv::Vec4f>& Register::get_corr_vertex()
{
	return h_corr_vertex_;
}

std::vector<cv::Vec4b>& Register::get_corr_color()
{
	return h_corr_color_;
}

std::vector<int>& Register::get_corr_index()
{
	return h_corr_index_;
}