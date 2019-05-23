#pragma once
#include <GL/glut.h>
#include <QGLWidget>
#include <QVector3D>
#include <QVector4D>
#include <QMatrix4x4>
#include <QMatrix3x3>
#include <QOpenGLBuffer>
#include <QGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFramebufferObject>
#include "opencv2/core/core.hpp"
#include "DrawObject.h"

class PointsObject: public DrawObject
{
public:
	PointsObject(int _id, QGLWidget *_openglWindow);
	~PointsObject();

	void SetPointSize(float _pointSize);

	void PhongDraw() override;
	void ColorDraw() override;
	void RenderVertexIdDraw() override;

private:
	float pointSize_;
};

