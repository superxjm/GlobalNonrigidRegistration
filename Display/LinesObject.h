#pragma once
#include <GL/glut.h>
#include <QGLWidget>
#include <QVector3D>
#include <QVector4D>
#include <QMatrix4x4>
#include <QMatrix3x3>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFramebufferObject>
#include <QGLShaderProgram>
#include "opencv2/core/core.hpp"
#include "DrawObject.h"

class LinesObject :	public DrawObject
{
public:
	LinesObject(int _id, QGLWidget *_openglWindow);
	~LinesObject();

	void SetIndex(std::vector<int>& _indices) override
	{
		indices_ = _indices;
	}

	void SetIndexToBuffer() override
	{
		//printf("%d\n", indices_.size());
		indexBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicRead);
		indexBuffer_->bind();
		indexBuffer_->allocate(indices_.data(), indices_.size() * sizeof(GLuint));
		indexBuffer_->release();
	}

	void PhongDraw() override;
	void ColorDraw() override;

private:
	QOpenGLBuffer          *indexBuffer_;
	std::vector<int>        indices_;
};

