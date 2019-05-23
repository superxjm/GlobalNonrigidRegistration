#include "TrianglesObject.h"



TrianglesObject::TrianglesObject(QGLWidget *_openglWindow)
{
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

	indexBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer);
	indexBuffer_->create();
}

TrianglesObject::~TrianglesObject()
{
}

void TrianglesObject::SetPosition(std::vector<cv::Vec4f> &_vertices)
{

}
void TrianglesObject::SetNormal(std::vector<cv::Vec4f> &_normals)
{

}
void TrianglesObject::SetColor(std::vector<cv::Vec3b> &_colors)
{

}



