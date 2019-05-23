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
#include "opencv2/core/core.hpp"

class TrianglesObject
{
public:
	TrianglesObject(QGLWidget *_openglWindow);
	~TrianglesObject();

	void SetPosition(std::vector<cv::Vec4f> &_vertices);
	void SetNormal(std::vector<cv::Vec4f> &_normals);
	void SetColor(std::vector<cv::Vec3b> &_colors);

private:
	QOpenGLBuffer *vertexPosBuffer_;
	QOpenGLBuffer *vertexColorBuffer_;
	QOpenGLVertexArrayObject *VAO_;

	QOpenGLBuffer *PhongVertexPosBuffer_;
	QOpenGLBuffer *PhongVertexNorBuffer_;
	QOpenGLVertexArrayObject *PhongVAO_;

	QOpenGLBuffer *indexBuffer_;
};

