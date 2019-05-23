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



class DrawObject
{
public:
	DrawObject(int _id, int _type);
	DrawObject(int _id, int _type, QGLWidget *_openglWindow);
	~DrawObject();
	
	void SetProgram(QGLShaderProgram *_phongProgram, QGLShaderProgram *_colorProgram, QGLShaderProgram *_renderVertexIdProgram);
	void SetHandle(int _colorPositionHandle, int _colorColorHandle,
		int _phongPositionHandle, int _phongNormalHandle,
		int _renderVertexIdPositionHandle, int _renderVertexIdObjectIdHandle);
	
	virtual void SetPosition(std::vector<float> &_vertices);
	virtual void SetPosition(std::vector<cv::Vec4f> &_vertices);
	virtual void SetNormal(std::vector<cv::Vec4f> &_normals);
	virtual void SetColor(std::vector<cv::Vec4b> &_colors);
	virtual void SetPositionToBuffer();
	virtual void SetNormalToBuffer();
	virtual void SetColorToBuffer();
	void SetRenderFlag(bool _flag) { renderFlag_ = _flag; }

	virtual void SetIndex(std::vector<int>& _indices) {}
	virtual void SetIndexToBuffer() {}

	virtual void PhongDraw() {};
	virtual void ColorDraw() {};
	virtual void RenderVertexIdDraw() {};

	virtual cv::Vec4f get_point(int _i) { assert(_i >= 0); assert(_i < num_vertex_); return vertices_[_i]; };
	int get_id() { return id_; }
	bool get_render_flag() { return renderFlag_; }

public:
	float orthoLeft, orthoRight, orthoBottom, orthoTop, orthoNear, orthoFar;

protected:

	int id_;

	//QOpenGLBuffer *indexBuffer_;

	QOpenGLBuffer *vertexPosBuffer_;
	QOpenGLBuffer *vertexColorBuffer_;
	QOpenGLVertexArrayObject *VAO_;
	QGLShaderProgram *colorProgram_;
	int colorPositionHandle_;
	int colorColorHandle_;

	QOpenGLBuffer *PhongVertexPosBuffer_;
	QOpenGLBuffer *PhongVertexNorBuffer_;
	QOpenGLVertexArrayObject *PhongVAO_;
	QGLShaderProgram *phongProgram_;
	int phongPositionHandle_;
	int phongNormalHandle_;

	QOpenGLBuffer *RRTvertexPosBuffer_;
	QOpenGLVertexArrayObject *RRTVAO_;
	QGLShaderProgram *renderVertexIdProgram_;
	int renderVertexIdPositionHandle_;
	int renderVertexIdObjectIdHandle_;

	std::vector<cv::Vec4f>  vertices_;
	std::vector<cv::Vec4f>  normals_;
	std::vector<cv::Vec4b>  colors_;
	int num_vertex_;

	//std::vector<int>        indices_;

	int type_;
	bool renderFlag_;
};

