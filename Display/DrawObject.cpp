#include "DrawObject.h"

DrawObject::DrawObject(int _id, int _type) {

	id_ = _id;
	type_ = _type;
	renderFlag_ = true;
	num_vertex_ = 0;
	vertices_.clear();
	normals_.clear();
	colors_.clear();

	orthoLeft = 100000;
	orthoRight = -100000;
	orthoBottom = 100000;
	orthoTop = -100000;
	orthoNear = 100000;
	orthoFar = -100000;

	renderFlag_ = true;
};

DrawObject::DrawObject(int _id, int _type, QGLWidget *_openglWindow)
{
	id_ = _id;
	type_ = _type;

	VAO_ = new QOpenGLVertexArrayObject(_openglWindow);
	vertexPosBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	vertexColorBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	VAO_->create();
	vertexPosBuffer_->create();
	vertexColorBuffer_->create();

	PhongVAO_ = new QOpenGLVertexArrayObject(_openglWindow);
	PhongVertexPosBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	PhongVertexNorBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	PhongVAO_->create();
	PhongVertexPosBuffer_->create();
	PhongVertexNorBuffer_->create();

	RRTVAO_ = new QOpenGLVertexArrayObject(_openglWindow);
	RRTvertexPosBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	RRTVAO_->create();
	RRTvertexPosBuffer_->create();

	//indexBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer);
	//indexBuffer_->create();

	num_vertex_ = 0;
	vertices_.clear();
	normals_.clear();
	colors_.clear();

	orthoLeft = 100000;
	orthoRight = -100000;
	orthoBottom = 100000;
	orthoTop = -100000;
	orthoNear = 100000;
	orthoFar = -100000;

	renderFlag_ = true;
}


DrawObject::~DrawObject()
{
	if (vertexPosBuffer_ != nullptr) delete vertexPosBuffer_;
	if (vertexColorBuffer_ != nullptr) delete vertexColorBuffer_;
	if (VAO_ != nullptr) delete VAO_;
	if (PhongVertexPosBuffer_ != nullptr) delete PhongVertexPosBuffer_;
	if (PhongVertexNorBuffer_ != nullptr) delete PhongVertexNorBuffer_;
	if (PhongVAO_ != nullptr) delete PhongVAO_;
	if (RRTvertexPosBuffer_ != nullptr) delete RRTvertexPosBuffer_;
	if (RRTVAO_ != nullptr) delete RRTVAO_;
}

void DrawObject::SetProgram(QGLShaderProgram *_phongProgram, QGLShaderProgram *_colorProgram, QGLShaderProgram *_renderVertexIdProgram)
{
	phongProgram_ = _phongProgram;
	colorProgram_ = _colorProgram;
	renderVertexIdProgram_ = _renderVertexIdProgram;
}

void DrawObject::SetHandle(int _colorPositionHandle, int _colorColorHandle, int _phongPositionHandle, int _phongNormalHandle,
	int _renderVertexIdPositionHandle, int _renderVertexIdObjectIdHandle)
{
	colorPositionHandle_ = _colorPositionHandle;
	colorColorHandle_ = _colorColorHandle;
	phongPositionHandle_ = _phongPositionHandle;
	phongNormalHandle_ = _phongNormalHandle;
	renderVertexIdPositionHandle_ = _renderVertexIdPositionHandle;
	renderVertexIdObjectIdHandle_ = _renderVertexIdObjectIdHandle;
}

void DrawObject::SetPosition(std::vector<float> &_vertices)
{
	num_vertex_ = _vertices.size() / 4;
	vertices_.resize(num_vertex_);
	memcpy(vertices_.data(), _vertices.data(), num_vertex_*4*sizeof(float));

	SetPosition(vertices_);
}

void DrawObject::SetPosition(std::vector<cv::Vec4f> &_vertices)
{
	num_vertex_ = _vertices.size();
	vertices_ = _vertices;

	orthoLeft = 100000;
	orthoRight = -100000;
	orthoBottom = 100000;
	orthoTop = -100000;
	orthoNear = 100000;
	orthoFar = -100000;
	for (int i = 0; i < _vertices.size(); i++)
	{
		if (_vertices[i][0] < orthoLeft) orthoLeft = _vertices[i][0];
		if (_vertices[i][0] > orthoRight) orthoRight = _vertices[i][0];
		if (_vertices[i][1] < orthoBottom) orthoBottom = _vertices[i][1];
		if (_vertices[i][1] > orthoTop) orthoTop = _vertices[i][1];
		if (_vertices[i][2] < orthoNear) orthoNear = _vertices[i][2];
		if (_vertices[i][2] > orthoFar) orthoFar = _vertices[i][2];
	}
}

void DrawObject::SetPositionToBuffer()
{
	if (vertices_.size() == 0) return;

	VAO_->bind();
	vertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	vertexPosBuffer_->bind();
	vertexPosBuffer_->allocate(vertices_.data(), num_vertex_ * 4 * sizeof(GLfloat));
	colorProgram_->enableAttributeArray(colorPositionHandle_);
	colorProgram_->setAttributeBuffer(colorPositionHandle_, GL_FLOAT, 0, 4);
	vertexPosBuffer_->release();
	VAO_->release();
	

	PhongVAO_->bind();
	PhongVertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	PhongVertexPosBuffer_->bind();
	PhongVertexPosBuffer_->allocate(vertices_.data(), num_vertex_ * 4 * sizeof(GLfloat));
	phongProgram_->enableAttributeArray(phongPositionHandle_);
	phongProgram_->setAttributeBuffer(phongPositionHandle_, GL_FLOAT, 0, 4);
	PhongVertexPosBuffer_->release();
	PhongVAO_->release();

	RRTVAO_->bind();
	RRTvertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	RRTvertexPosBuffer_->bind();
	RRTvertexPosBuffer_->allocate(vertices_.data(), num_vertex_ * 4 * sizeof(GLfloat));
	renderVertexIdProgram_->enableAttributeArray(renderVertexIdPositionHandle_);
	renderVertexIdProgram_->setAttributeArray(renderVertexIdPositionHandle_, GL_FLOAT, 0, 4);
	RRTvertexPosBuffer_->release();
	RRTVAO_->release();
}

void DrawObject::SetNormal(std::vector<cv::Vec4f> &_normals)
{
	normals_ = _normals;
}

void DrawObject::SetNormalToBuffer()
{
	if (vertices_.size() == 0) return;
	if (normals_.size() == 0) return;

	PhongVAO_->bind();
	PhongVertexNorBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	PhongVertexNorBuffer_->bind();
	PhongVertexNorBuffer_->allocate(normals_.data(), num_vertex_ * 4 * sizeof(GLfloat));
	phongProgram_->enableAttributeArray(phongNormalHandle_);
	phongProgram_->setAttributeBuffer(phongNormalHandle_, GL_FLOAT, 0, 4);
	PhongVertexNorBuffer_->release();
	PhongVAO_->release();
}

void DrawObject::SetColor(std::vector<cv::Vec4b> &_colors)
{
	colors_ = _colors;
}

void DrawObject::SetColorToBuffer()
{
	if (vertices_.size() == 0) return;
	if (colors_.size() == 0) return;

	VAO_->bind();
	vertexColorBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicRead);
	vertexColorBuffer_->bind();
	vertexColorBuffer_->allocate(colors_.data(), num_vertex_ * 4 * sizeof(GLubyte));
	colorProgram_->enableAttributeArray(colorColorHandle_);
	colorProgram_->setAttributeBuffer(colorColorHandle_, GL_UNSIGNED_BYTE, 0, 4);
	vertexColorBuffer_->release();
	VAO_->release();
}