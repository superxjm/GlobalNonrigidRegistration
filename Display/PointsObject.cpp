#include "PointsObject.h"



PointsObject::PointsObject(int _id, QGLWidget *_openglWindow)
	:DrawObject(_id, GL_POINTS, _openglWindow)
{
	pointSize_ = 1.0f;
}


PointsObject::~PointsObject()
{
}

void PointsObject::ColorDraw()
{
	if (num_vertex_ == 0) return;

	VAO_->bind();
	glPointSize(pointSize_);
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	glPointSize(1.0f);
	VAO_->release();
}

void PointsObject::PhongDraw()
{
	if (num_vertex_ == 0) return;

	PhongVAO_->bind();
	glPointSize(pointSize_);
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	glPointSize(1.0f);
	PhongVAO_->release();
}

void PointsObject::RenderVertexIdDraw()
{
	if (num_vertex_ == 0) return;

	renderVertexIdProgram_->setUniformValue(renderVertexIdObjectIdHandle_, id_);

	RRTVAO_->bind();
	glPointSize(1.5f);
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	glPointSize(1.0f);
	RRTVAO_->release();
}

void PointsObject::SetPointSize(float _pointSize)
{
	pointSize_ = _pointSize;
}