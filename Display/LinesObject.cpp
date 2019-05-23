#include "LinesObject.h"



LinesObject::LinesObject(int _id, QGLWidget *_openglWindow)
	:DrawObject(_id, GL_LINES, _openglWindow)
{
	indexBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer);
	indexBuffer_->create();
}


LinesObject::~LinesObject()
{
	if (indexBuffer_ != nullptr) delete indexBuffer_;
}

void LinesObject::ColorDraw()
{
	if (num_vertex_ == 0) return;
	if (indices_.size() == 0) return;

	VAO_->bind();
	indexBuffer_->bind();
	//glDrawArrays(GL_LINES, 0, num_vertex_);
	glDrawElements(GL_LINES, indices_.size(), GL_UNSIGNED_INT, 0);
	indexBuffer_->release();
	VAO_->release();
}

void LinesObject::PhongDraw()
{
	if (num_vertex_ == 0) return;
	if (indices_.size() == 0) return;

	PhongVAO_->bind();
	indexBuffer_->bind();
	//glDrawArrays(GL_LINES, 0, num_vertex_);
	glDrawElements(GL_LINES, indices_.size(), GL_UNSIGNED_INT, 0);
	indexBuffer_->release();
	PhongVAO_->release();
}