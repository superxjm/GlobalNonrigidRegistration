#include "MyPointsObject.h"



MyPointsObject::MyPointsObject(int _id, QGLWidget *_openglWindow)
	:DrawObject(_id, GL_POINTS)
{
	VAO_ = new QOpenGLVertexArrayObject(_openglWindow);
	VAO_->create();

	PhongVAO_ = new QOpenGLVertexArrayObject(_openglWindow);
	PhongVAO_->create();

	RRTVAO_ = new QOpenGLVertexArrayObject(_openglWindow);
	RRTVAO_->create();

	vertexBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	vertexBuffer_->create();

	pointSize_ = 1.0f;
}



MyPointsObject::~MyPointsObject()
{
	if (vertexBuffer_ != nullptr) delete vertexBuffer_;
}

void MyPointsObject::SetVertex(std::vector<VBOType> &_vbo)
{
	vbo_ = _vbo;
	num_vertex_ = vbo_.size();

	for (int i = 0; i < num_vertex_; i++)
	{
		if (_vbo[i].posConf.x < orthoLeft) orthoLeft = _vbo[i].posConf.x;
		if (_vbo[i].posConf.x > orthoRight) orthoRight = _vbo[i].posConf.x;
		if (_vbo[i].posConf.y < orthoBottom) orthoBottom = _vbo[i].posConf.y;
		if (_vbo[i].posConf.y > orthoTop) orthoTop = _vbo[i].posConf.y;
		if (_vbo[i].posConf.z < orthoNear) orthoNear = _vbo[i].posConf.z;
		if (_vbo[i].posConf.z > orthoFar) orthoFar = _vbo[i].posConf.z;
	}

	vertexBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	vertexBuffer_->bind();
	vertexBuffer_->allocate(vbo_.data(), num_vertex_ * sizeof(VBOType));
	vertexBuffer_->release();
}

void MyPointsObject::SetPositionToBuffer()
{
	if (vbo_.size() == 0) return;

	VAO_->bind();
	vertexBuffer_->bind();
	colorProgram_->enableAttributeArray(colorPositionHandle_);
	colorProgram_->setAttributeBuffer(colorPositionHandle_, GL_FLOAT, 0, 3, sizeof(VBOType));
	vertexBuffer_->release();
	VAO_->release();

	PhongVAO_->bind();
	vertexBuffer_->bind();
	phongProgram_->enableAttributeArray(phongPositionHandle_);
	phongProgram_->setAttributeBuffer(phongPositionHandle_, GL_FLOAT, 0, 3, sizeof(VBOType));
	vertexBuffer_->release();
	PhongVAO_->release();

	RRTVAO_->bind();
	vertexBuffer_->bind();
	renderVertexIdProgram_->enableAttributeArray(renderVertexIdPositionHandle_);
	renderVertexIdProgram_->setAttributeBuffer(renderVertexIdPositionHandle_, GL_FLOAT, 0, 3, sizeof(VBOType));
	vertexBuffer_->release();
	RRTVAO_->release();
}

void MyPointsObject::SetNormalToBuffer()
{
	if (vbo_.size() == 0) return;

	PhongVAO_->bind();
	vertexBuffer_->bind();
	phongProgram_->enableAttributeArray(phongNormalHandle_);
	phongProgram_->setAttributeBuffer(phongNormalHandle_, GL_FLOAT, 32, 3, sizeof(VBOType));
	vertexBuffer_->release();
	PhongVAO_->release();
}

void MyPointsObject::SetColorToBuffer()
{
	if (vbo_.size() == 0) return;

	VAO_->bind();
	vertexBuffer_->bind();
	colorProgram_->enableAttributeArray(colorColorHandle_);
	colorProgram_->setAttributeBuffer(colorColorHandle_, GL_UNSIGNED_BYTE, 16, 4, sizeof(VBOType));
	vertexBuffer_->release();
	VAO_->release();

}

void MyPointsObject::PhongDraw()
{
	if (num_vertex_ == 0) return;

	PhongVAO_->bind();
	glPointSize(pointSize_);
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	glPointSize(1.0f);
	PhongVAO_->release();
}

void MyPointsObject::ColorDraw()
{
	if (num_vertex_ == 0) return;

	VAO_->bind();
	glPointSize(pointSize_);
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	glPointSize(1.0f);
	VAO_->release();
}

void MyPointsObject::RenderVertexIdDraw()
{
	if (num_vertex_ == 0) return;

	renderVertexIdProgram_->setUniformValue(renderVertexIdObjectIdHandle_, id_);

	RRTVAO_->bind();
	glPointSize(1.5f);
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	glPointSize(1.0f);
	RRTVAO_->release();
}