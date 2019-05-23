#pragma once
#include "Display/DrawObject.h"
#include "Helpers/xUtils.h"

class MyPointsObject :
	public DrawObject
{
public:
	MyPointsObject(int _id, QGLWidget *_openglWindow);
	~MyPointsObject();
    
	void SetVertex(std::vector<VBOType> &_vbo);
	int GetVertexBufferId() { return vertexBuffer_->bufferId(); }

	void SetPosition(std::vector<float> &_vertices) override {}
	void SetPosition(std::vector<cv::Vec4f> &_vertices) override  {}
	void SetNormal(std::vector<cv::Vec4f> &_normals) override  {}
	void SetColor(std::vector<cv::Vec4b> &_colors) override  {}

	void SetPositionToBuffer() override;
	void SetNormalToBuffer() override;
	void SetColorToBuffer() override;

	void PhongDraw() override;
	void ColorDraw() override;
	void RenderVertexIdDraw() override;

	void SetPointSize(float _pointSize) { pointSize_ = _pointSize; }
	cv::Vec4f get_point(int _i) override
	{
		cv::Vec4f p;
		p[0] = vbo_[_i].posConf.x;
		p[1] = vbo_[_i].posConf.y;
		p[2] = vbo_[_i].posConf.z;
		p[3] = 1;
		return p;
	}

	int get_vertex_num() { return num_vertex_; }

private:
	float pointSize_;

	QOpenGLBuffer *vertexBuffer_;

	std::vector<VBOType> vbo_;
};

