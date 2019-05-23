#include "OpenGLWindow.h"

OpenGLWindow::OpenGLWindow(QWidget *parent)
	: QGLWidget(parent)
{
	Init();
}

OpenGLWindow::~OpenGLWindow()
{
	if (vertexPosBuffer_ != nullptr) delete vertexPosBuffer_;
	if (vertexColorBuffer_ != nullptr) delete vertexColorBuffer_;
	if (indexBuffer_ != nullptr) delete indexBuffer_;
	if (VAO_ != nullptr) delete VAO_;
	if (RRTvertexPosBuffer_ != nullptr) delete RRTvertexPosBuffer_;
	if (RRTVAO_ != nullptr) delete RRTVAO_;
	if (PhongVertexPosBuffer_ != nullptr) delete PhongVertexPosBuffer_;
	if (PhongVertexNorBuffer_ != nullptr) delete PhongVertexNorBuffer_;
	if (PhongIndexBuffer_ != nullptr) delete PhongIndexBuffer_;
	if (PhongVAO_ != nullptr) delete PhongVAO_;
	if (arcBall_ != nullptr) delete arcBall_;
	if (vertexIdFramebufferObject != nullptr) delete vertexIdFramebufferObject;
}

void OpenGLWindow::Init()
{
	drawObject_.clear();

	selectedIndex_ = -1;
	showMode_ = 0;
	shaderMode_ = 0;
	pickMode_ = 1;
	arcBall_ = new ArcBallT(this->size().width(), this->size().height());

	//num_vertex_ = 0;
	num_face_ = 0;

	orthoLeft = 100000;
	orthoRight = -100000;
	orthoBottom = 100000;
	orthoTop = -100000;
	orthoNear = 100000;
	orthoFar = -100000;
	xInitTrans = 0;
	yInitTrans = 0;
	zInitTrans = 0;
	initScale = 1;

	xTrans = 0;
	yTrans = 0;
	zTrans = 0;
	scale = 1;
	rotateMatrix.setToIdentity();


	mModelMatrix.setToIdentity();
	mViewMatrix.setToIdentity();
	mProjectionMatrix.setToIdentity();
}

void OpenGLWindow::initializeGL()
{
	glFrontFace(GL_CCW);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glShadeModel(GL_SMOOTH);
	glClearColor(0.4f, 0.4f, 0.6f, 1.0f);
	glClearDepth(1.0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	initShader();
	vLightPosition = QVector3D(0.0, 0.0, 1.0);
	vAmbientColor = QVector4D(0.2, 0.2, 0.2, 1.0);
	vDiffuseColor = QVector4D(0.6, 0.6, 0.6, 1.0);
	vSpecularColor = QVector4D(0.5, 0.5, 0.5, 1.0);
	fDepthChang = 0;

	//init buffer
	VAO_ = new QOpenGLVertexArrayObject(this);
	vertexPosBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	vertexColorBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	indexBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer);
	VAO_->create();
	vertexPosBuffer_->create();
	vertexColorBuffer_->create();
	indexBuffer_->create();

	RRTVAO_ = new QOpenGLVertexArrayObject(this);
	RRTvertexPosBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	RRTVAO_->create();
	RRTvertexPosBuffer_->create();

	PhongVAO_ = new QOpenGLVertexArrayObject(this);
	PhongVertexPosBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	PhongVertexNorBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
	PhongIndexBuffer_ = new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer);
	PhongVAO_->create();
	PhongVertexPosBuffer_->create();
	PhongVertexNorBuffer_->create();
	PhongIndexBuffer_->create();

	vertexIdFramebufferObject = new QOpenGLFramebufferObject(this->size().width(), this->size().height(), QOpenGLFramebufferObject::Attachment::Depth, GL_TEXTURE_2D, GL_RGBA8);
	vertexIdFramebufferObject->release();

}

void OpenGLWindow::initShader()
{
	QGLShader *vshader_phong = new QGLShader(QGLShader::Vertex, this);
	vshader_phong->compileSourceFile("./Display/VertexPhong.txt");
	QGLShader *fshader_phong = new QGLShader(QGLShader::Fragment, this);
	fshader_phong->compileSourceFile("./Display/FragmentPhong.txt");
	phongProgram.addShader(vshader_phong);
	phongProgram.addShader(fshader_phong);
	phongProgram.bindAttributeLocation("a_Position", 0);
	phongProgram.bindAttributeLocation("a_Normal", 1);
	phongProgram.link();
	phongPositionHandle = phongProgram.attributeLocation("a_Position");
	phongNormalHandle = phongProgram.attributeLocation("a_Normal");
	phongModelMatrixHandle = phongProgram.uniformLocation("u_ModelMatrix");
	phongViewMatrixHandle = phongProgram.uniformLocation("u_ViewMatrix");
	phongProjectionMatrixHandle = phongProgram.uniformLocation("u_ProjectionMatrix");
	phongNormalMatrixHandle = phongProgram.uniformLocation("u_NormalMatrix");
	phongLightPositionHandle = phongProgram.uniformLocation("u_LightPosition");
	phongAmbientColorHandle = phongProgram.uniformLocation("u_AmbientColor");
	phongDiffuseColorHandle = phongProgram.uniformLocation("u_DiffuseColor");
	phongSpecularColorHandle = phongProgram.uniformLocation("u_SpecularColor");
	phongDepthChangeHandle = phongProgram.uniformLocation("u_depthChange");
	phongProgram.removeAllShaders();

	QGLShader *vshader_color = new QGLShader(QGLShader::Vertex, this);
	vshader_color->compileSourceFile("./Display/VertexColor.txt");
	QGLShader *fshader_color = new QGLShader(QGLShader::Fragment, this);
	fshader_color->compileSourceFile("./Display/FragmentColor.txt");
	colorProgram.addShader(vshader_color);
	colorProgram.addShader(fshader_color);
	colorProgram.bindAttributeLocation("a_Position", 0);
	colorProgram.bindAttributeLocation("a_Color", 1);
	colorProgram.link();
	colorPositionHandle = colorProgram.attributeLocation("a_Position");
	colorColorHandle = colorProgram.attributeLocation("a_Color");
	colorMVPMatrixHandle = colorProgram.uniformLocation("u_MVPMatrix");
	colorDepthChangeHandle = colorProgram.uniformLocation("u_depthChange");
	colorProgram.removeAllShaders();

	QGLShader *vshader_vertexId = new QGLShader(QGLShader::Vertex, this);
	vshader_vertexId->compileSourceFile("./Display/VertexRenderVertexId.txt");
	QGLShader *fshader_vertexId = new QGLShader(QGLShader::Fragment, this);
	fshader_vertexId->compileSourceFile("./Display/FragmentRenderVertexId.txt");
	renderVertexIdProgram.addShader(vshader_vertexId);
	renderVertexIdProgram.addShader(fshader_vertexId);
	renderVertexIdProgram.bindAttributeLocation("a_Position", 0);
	//renderVertexIdProgram.bindAttributeLocation("a_VertexId", 1);
	renderVertexIdProgram.link();
	renderVertexIdPositionHandle = renderVertexIdProgram.attributeLocation("a_Position");
	//renderVertexIdVertexIdHandle = renderVertexIdProgram.attributeLocation("a_VertexId");
	renderVertexIdMVPMatrixHandle = renderVertexIdProgram.uniformLocation("u_MVPMatrix");
	renderVertexIdObjectIdHandle = renderVertexIdProgram.uniformLocation("u_ObjectId");
	renderVertexIdProgram.removeAllShaders();

}

void OpenGLWindow::resizeGL(int width, int height)
{
	w = width;
	h = height;
	glViewport(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (w > h)
		glOrtho(-static_cast<GLdouble>(w) / h, static_cast<GLdouble>(w) / h, -1.0, 1.0, -5000.0, 5000.0);
	else
		glOrtho(-1.0, 1.0, -static_cast<GLdouble>(h) / w, static_cast<GLdouble>(h) / w, -5000.0, 5000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 1000, 0, 0, -1, 0, 1, 0);

	mProjectionMatrix.setToIdentity();
	if (w > h)
		mProjectionMatrix.ortho(-static_cast<GLdouble>(w) / h, static_cast<GLdouble>(w) / h, -1.0, 1.0, -5000.0, 5000.0);
	else
		mProjectionMatrix.ortho(-1.0, 1.0, -static_cast<GLdouble>(h) / w, static_cast<GLdouble>(h) / w, -5000.0, 5000.0);


	mViewMatrix.setToIdentity();
	mViewMatrix.lookAt(QVector3D(0, 0, 1000), QVector3D(0, 0, -1), QVector3D(0, 1, 0));

	//qDebug() << w << " " << h << " " << this->size();
	arcBall_->setBounds(w, h);
	delete vertexIdFramebufferObject;
	vertexIdFramebufferObject = new QOpenGLFramebufferObject(this->size().width(), this->size().height(), QOpenGLFramebufferObject::Attachment::Depth);
	vertexIdFramebufferObject->release();
}

void OpenGLWindow::RenderVertexIdToTexture()
{
	renderVertexIdProgram.setUniformValue(renderVertexIdMVPMatrixHandle, (mProjectionMatrix * mViewMatrix * mModelMatrix));

	/*RRTVAO_->bind();
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	RRTVAO_->release();*/
	for (int i = 0; i < drawObject_.size(); i++)
	{
		if (drawObject_[i]->get_render_flag())
			drawObject_[i]->RenderVertexIdDraw();
	}
}

void OpenGLWindow::DrawSelectedPoint()
{
	if (selectedIndex_ == -1) return;
	if (selectedObjectIndex_ == -1) return;

	cv::Vec4f p;
	for (int i = 0; i < drawObject_.size(); i++)
	{
		if (selectedObjectIndex_ == drawObject_[i]->get_id())
		{
			p = drawObject_[i]->get_point(selectedIndex_);
		}
	}

	qDebug() << "position: " <<p[0] << " " << p[1] << " " << " " << p[2] << "\n";

	glDisable(GL_LIGHTING);
	glPointSize(3.0f);
	glColor3f(1.0f, 1.0f, 0.0f);
	glBegin(GL_POINTS);
	glVertex3f(p[0], p[1], p[2]);
	glEnd();

	glPointSize(1.0f);
	glColor3f(0.0f, 0.0f, 0.0f);

}

/*void OpenGLWindow::ChangePointsObjectColor(int _id, std::vector<int>& _indices, cv::Vec4b _color)
{

	cv::Vec4f p;
	for (int i = 0; i < pointsObject_.size(); i++)
	{
		if (pointsObject_[i]->get_id() == _id)
		{
			glDisable(GL_LIGHTING);
			glPointSize(3.0f);
			glColor3f(_color[0]/255.0f, _color[1]/255.0f, _color[2]/255.0f);
			glBegin(GL_POINTS);
			for (int j = 0; j < _indices.size(); j++)
			{
				p = pointsObject_[i]->get_point(_indices[j]);
				glVertex3f(p[0], p[1], p[2]);
			}
			glEnd();
		}
	}

	glPointSize(1.0f);
	glColor3f(0.0f, 0.0f, 0.0f);
}*/

void OpenGLWindow::DrawPoints()
{
	switch (shaderMode_)
	{
	case 0:
		colorProgram.bind();
		DrawColorPoints();
		colorProgram.release();
		break;
	case 1:
		phongProgram.bind();
		DrawPhongPoints();
		phongProgram.release();
		break;
	}

	DrawSelectedPoint();
}

void OpenGLWindow::DrawColorPoints()
{
	colorProgram.setUniformValue(colorMVPMatrixHandle, (mProjectionMatrix * mViewMatrix * mModelMatrix));
	colorProgram.setUniformValue(colorDepthChangeHandle, 0.f);

	/*VAO_->bind();
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	VAO_->release();*/
	for (int i = 0; i < drawObject_.size(); i++)
	{
		if (drawObject_[i]->get_render_flag())
			drawObject_[i]->ColorDraw();
	}
}

void OpenGLWindow::DrawPhongPoints()
{
	phongProgram.setUniformValue(phongModelMatrixHandle, mModelMatrix);
	phongProgram.setUniformValue(phongViewMatrixHandle, mViewMatrix);
	phongProgram.setUniformValue(phongProjectionMatrixHandle, mProjectionMatrix);
	phongProgram.setUniformValue(phongNormalMatrixHandle, mNormalMatrix);
	phongProgram.setUniformValue(phongLightPositionHandle, vLightPosition);
	phongProgram.setUniformValue(phongAmbientColorHandle, vAmbientColor);
	phongProgram.setUniformValue(phongDiffuseColorHandle, vDiffuseColor);
	phongProgram.setUniformValue(phongSpecularColorHandle, vSpecularColor);
	phongProgram.setUniformValue(phongDepthChangeHandle, 0.f);

	/*PhongVAO_->bind();
	glDrawArrays(GL_POINTS, 0, num_vertex_);
	PhongVAO_->release();*/
	for (int i = 0; i < drawObject_.size(); i++)
	{
		if (drawObject_[i]->get_render_flag())
			drawObject_[i]->PhongDraw();
	}
}

void OpenGLWindow::DrawLines()
{
	switch (shaderMode_)
	{
	case 0:
		colorProgram.bind();
		DrawColorLines();
		colorProgram.release();
		break;
	case 1:
		phongProgram.bind();
		DrawPhongLines();
		phongProgram.release();
		break;
	}

	DrawSelectedPoint();
}

void OpenGLWindow::DrawColorLines()
{
	colorProgram.setUniformValue(colorMVPMatrixHandle, (mProjectionMatrix * mViewMatrix * mModelMatrix));
	colorProgram.setUniformValue(colorDepthChangeHandle, -0.00001f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	VAO_->bind();
	glDrawElements(GL_TRIANGLES, num_face_ * 3, GL_UNSIGNED_INT, 0);
	VAO_->release();
}

void OpenGLWindow::DrawPhongLines()
{
	phongProgram.setUniformValue(phongModelMatrixHandle, mModelMatrix);
	phongProgram.setUniformValue(phongViewMatrixHandle, mViewMatrix);
	phongProgram.setUniformValue(phongProjectionMatrixHandle, mProjectionMatrix);
	phongProgram.setUniformValue(phongNormalMatrixHandle, mNormalMatrix);
	phongProgram.setUniformValue(phongLightPositionHandle, vLightPosition);
	phongProgram.setUniformValue(phongAmbientColorHandle, vAmbientColor);
	phongProgram.setUniformValue(phongDiffuseColorHandle, vDiffuseColor);
	phongProgram.setUniformValue(phongSpecularColorHandle, vSpecularColor);
	phongProgram.setUniformValue(phongDepthChangeHandle, 0.f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	PhongVAO_->bind();
	glDrawElements(GL_TRIANGLES, num_face_ * 3, GL_UNSIGNED_INT, 0);
	PhongVAO_->release();
}

void OpenGLWindow::DrawTriangles()
{
	switch (shaderMode_)
	{
	case 0:
		colorProgram.bind();
		DrawColorTriangles();
		colorProgram.release();
		break;
	case 1:
		phongProgram.bind();
		DrawPhongTriangles();
		phongProgram.release();
		break;
	}

	DrawSelectedPoint();
}

void OpenGLWindow::DrawColorTriangles()
{
	colorProgram.setUniformValue(colorMVPMatrixHandle, (mProjectionMatrix * mViewMatrix * mModelMatrix));
	colorProgram.setUniformValue(colorDepthChangeHandle, -0.00001f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	VAO_->bind();
	glDrawElements(GL_TRIANGLES, num_face_ * 3, GL_UNSIGNED_INT, 0);
	VAO_->release();
}

void OpenGLWindow::DrawPhongTriangles()
{
	phongProgram.setUniformValue(phongModelMatrixHandle, mModelMatrix);
	phongProgram.setUniformValue(phongViewMatrixHandle, mViewMatrix);
	phongProgram.setUniformValue(phongProjectionMatrixHandle, mProjectionMatrix);
	phongProgram.setUniformValue(phongNormalMatrixHandle, mNormalMatrix);
	phongProgram.setUniformValue(phongLightPositionHandle, vLightPosition);
	phongProgram.setUniformValue(phongAmbientColorHandle, vAmbientColor);
	phongProgram.setUniformValue(phongDiffuseColorHandle, vDiffuseColor);
	phongProgram.setUniformValue(phongSpecularColorHandle, vSpecularColor);
	phongProgram.setUniformValue(phongDepthChangeHandle, 0.f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	PhongVAO_->bind();
	glDrawElements(GL_TRIANGLES, num_face_ * 3, GL_UNSIGNED_INT, 0);
	PhongVAO_->release();
}

void OpenGLWindow::Render()
{
	QMatrix4x4 rMatrix(arcBall_->Transform.M);
	rotateMatrix = rMatrix;
	rotateMatrix = rotateMatrix.transposed();
	mModelMatrix.setToIdentity();
	mModelMatrix *= rotateMatrix;
	mModelMatrix.scale(scale, scale, scale);
	mModelMatrix.translate(xTrans, yTrans, zTrans);
	mModelMatrix.scale(initScale, initScale, initScale);
	mModelMatrix.translate(xInitTrans, yInitTrans, zInitTrans);
	mNormalMatrix = (mViewMatrix * mModelMatrix).normalMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(mModelMatrix.data());

	switch (showMode_)
	{
	case 0:
		DrawPoints();
		break;
	case 1:
		//DrawLines();
		break;
	case 2:
		//DrawTriangles();
		break;
	case 3:
		//DrawPickPoints();
		break;
	case 4:
		renderVertexIdProgram.bind();
		vertexIdFramebufferObject->bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		RenderVertexIdToTexture();
		vertexIdFramebufferObject->release();
		renderVertexIdProgram.release();
		break;
	}

	glPopMatrix();
}

void OpenGLWindow::paintGL()
{
	//glFrontFace(GL_CCW);
	//glEnable(GL_CULL_FACE);
	glDisable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	glShadeModel(GL_SMOOTH);
	glClearColor(0.4f, 0.4f, 0.6f, 1.0f);
	glClearDepth(1.0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	Render();
}

/*void OpenGLWindow::SetVertex(std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4b> &_colors, 
	std::vector<cv::Vec3f> &_normals, std::vector<cv::Vec3i> &_faceIndices)
{
	vertices_ = _vertices;

	//将mesh放入opengl的buffer中

	num_vertex_ = _vertices.size();
	num_face_ = _faceIndices.size();

	
	//std::vector<GLuint> _faces;
	//for (int i = 0; i < num_face_; i++)
	//{
	//	_faces.push_back((GLuint)_faceIndices[i][0]);
	//	_faces.push_back((GLuint)_faceIndices[i][1]);
	//	_faces.push_back((GLuint)_faceIndices[i][2]);
	//}

	VAO_->bind();
	vertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	vertexPosBuffer_->bind();
	vertexPosBuffer_->allocate(_vertices.data(), num_vertex_ * 4 * sizeof(GLfloat));
	colorProgram.enableAttributeArray(colorPositionHandle);
	colorProgram.setAttributeBuffer(colorPositionHandle, GL_FLOAT, 0, 4);
	vertexPosBuffer_->release();
	vertexColorBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicRead);
	vertexColorBuffer_->bind();	
	vertexColorBuffer_->allocate(_colors.data(), num_vertex_ * 4 * sizeof(GLubyte));
	colorProgram.enableAttributeArray(colorColorHandle);
	colorProgram.setAttributeBuffer(colorColorHandle, GL_UNSIGNED_BYTE, 0, 4);
	vertexColorBuffer_->release();
	indexBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicRead);
	indexBuffer_->bind();
	indexBuffer_->allocate(_faceIndices.data(), num_face_ * 3 * sizeof(GLuint));
	//indexBuffer_->release();
	VAO_->release();

	RRTVAO_->bind();
	RRTvertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	RRTvertexPosBuffer_->bind();
	RRTvertexPosBuffer_->allocate(_vertices.data(), num_vertex_ * 4 * sizeof(GLfloat));
	renderVertexIdProgram.enableAttributeArray(renderVertexIdPositionHandle);
	renderVertexIdProgram.setAttributeArray(renderVertexIdPositionHandle, GL_FLOAT, 0, 4);
	RRTvertexPosBuffer_->release();
	RRTVAO_->release();

	PhongVAO_->bind();
	PhongVertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	PhongVertexPosBuffer_->bind();
	PhongVertexPosBuffer_->allocate(_vertices.data(), num_vertex_ * 4 * sizeof(GLfloat));
	phongProgram.enableAttributeArray(phongPositionHandle);
	phongProgram.setAttributeBuffer(phongPositionHandle, GL_FLOAT, 0, 4);
	PhongVertexPosBuffer_->release();
	PhongVertexNorBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	PhongVertexNorBuffer_->bind();
	PhongVertexNorBuffer_->allocate(_normals.data(), num_vertex_ * 3 * sizeof(GLfloat));
	phongProgram.enableAttributeArray(phongNormalHandle);
	phongProgram.setAttributeBuffer(phongNormalHandle, GL_FLOAT, 0, 3);
	PhongVertexNorBuffer_->release();
	PhongIndexBuffer_->bind();
	PhongIndexBuffer_->allocate(_faceIndices.data(), num_face_ * 3 * sizeof(GLuint));
	PhongVAO_->release();

	

	//归一化模型坐标

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

	poid[0] = (orthoLeft + orthoRight) / 2;
	poid[1] = (orthoBottom + orthoTop) / 2;
	poid[2] = (orthoNear + orthoFar) / 2;

	xInitTrans = -poid[0];
	yInitTrans = -poid[1];
	zInitTrans = -poid[2];
	float max_len = 0.0f;
	if (orthoRight - orthoLeft > max_len) { max_len = orthoRight - orthoLeft; }
	if (orthoTop - orthoBottom > max_len) { max_len = orthoTop - orthoBottom; }
	if (orthoFar - orthoNear > max_len) { max_len = orthoFar - orthoNear; }
	initScale = 2.0f / max_len;

	updateGL();
}*/

/*void OpenGLWindow::SetVertex(std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals)
{

	vertices_ = _vertices;

	//将mesh放入opengl的buffer中

	num_vertex_ = _vertices.size();

	VAO_->bind();
	vertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	vertexPosBuffer_->bind();
	vertexPosBuffer_->allocate(_vertices.data(), num_vertex_ * 4 * sizeof(GLfloat));
	colorProgram.enableAttributeArray(colorPositionHandle);
	colorProgram.setAttributeBuffer(colorPositionHandle, GL_FLOAT, 0, 4);
	vertexPosBuffer_->release();
	VAO_->release();

	RRTVAO_->bind();
	RRTvertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	RRTvertexPosBuffer_->bind();
	RRTvertexPosBuffer_->allocate(_vertices.data(), num_vertex_ * 4 * sizeof(GLfloat));
	renderVertexIdProgram.enableAttributeArray(renderVertexIdPositionHandle);
	renderVertexIdProgram.setAttributeArray(renderVertexIdPositionHandle, GL_FLOAT, 0, 4);
	RRTvertexPosBuffer_->release();
	RRTVAO_->release();

	PhongVAO_->bind();
	PhongVertexPosBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	PhongVertexPosBuffer_->bind();
	PhongVertexPosBuffer_->allocate(_vertices.data(), num_vertex_ * 4 * sizeof(GLfloat));
	phongProgram.enableAttributeArray(phongPositionHandle);
	phongProgram.setAttributeBuffer(phongPositionHandle, GL_FLOAT, 0, 4);
	PhongVertexPosBuffer_->release();
	PhongVertexNorBuffer_->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	PhongVertexNorBuffer_->bind();
	PhongVertexNorBuffer_->allocate(_normals.data(), num_vertex_ * 4 * sizeof(GLfloat));
	phongProgram.enableAttributeArray(phongNormalHandle);
	phongProgram.setAttributeBuffer(phongNormalHandle, GL_FLOAT, 0, 4);
	PhongVertexNorBuffer_->release();
	PhongVAO_->release();



	//归一化模型坐标

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

	poid[0] = (orthoLeft + orthoRight) / 2;
	poid[1] = (orthoBottom + orthoTop) / 2;
	poid[2] = (orthoNear + orthoFar) / 2;

	xInitTrans = -poid[0];
	yInitTrans = -poid[1];
	zInitTrans = -poid[2];
	float max_len = 0.0f;
	if (orthoRight - orthoLeft > max_len) { max_len = orthoRight - orthoLeft; }
	if (orthoTop - orthoBottom > max_len) { max_len = orthoTop - orthoBottom; }
	if (orthoFar - orthoNear > max_len) { max_len = orthoFar - orthoNear; }
	initScale = 2.0f / max_len;

	updateGL();
}*/

void OpenGLWindow::UpdateModelMatrix()
{
	orthoLeft = 100000;
	orthoRight = -100000;
	orthoBottom = 100000;
	orthoTop = -100000;
	orthoNear = 100000;
	orthoFar = -100000;
	for (int i = 0; i < drawObject_.size(); i++)
	{
		if (drawObject_[i]->orthoLeft < orthoLeft) orthoLeft = drawObject_[i]->orthoLeft;
		if (drawObject_[i]->orthoRight > orthoRight) orthoRight = drawObject_[i]->orthoRight;
		if (drawObject_[i]->orthoBottom < orthoBottom) orthoBottom = drawObject_[i]->orthoBottom;
		if (drawObject_[i]->orthoTop > orthoTop) orthoTop = drawObject_[i]->orthoTop;
		if (drawObject_[i]->orthoNear < orthoNear) orthoNear = drawObject_[i]->orthoNear;
		if (drawObject_[i]->orthoFar > orthoFar) orthoFar = drawObject_[i]->orthoFar;
	}

	poid[0] = (orthoLeft + orthoRight) / 2;
	poid[1] = (orthoBottom + orthoTop) / 2;
	poid[2] = (orthoNear + orthoFar) / 2;

	xInitTrans = -poid[0];
	yInitTrans = -poid[1];
	zInitTrans = -poid[2];
	float max_len = 0.0f;
	if (orthoRight - orthoLeft > max_len) { max_len = orthoRight - orthoLeft; }
	if (orthoTop - orthoBottom > max_len) { max_len = orthoTop - orthoBottom; }
	if (orthoFar - orthoNear > max_len) { max_len = orthoFar - orthoNear; }
	initScale = 2.0f / max_len;
}

void OpenGLWindow::AddObject(DrawObject *_drawObject)
{
	_drawObject->SetProgram(&phongProgram, &colorProgram, &renderVertexIdProgram);
	_drawObject->SetHandle(colorPositionHandle, colorColorHandle,
		phongPositionHandle, phongNormalHandle,
		renderVertexIdPositionHandle, renderVertexIdObjectIdHandle);
	_drawObject->SetPositionToBuffer();
	_drawObject->SetNormalToBuffer();
	_drawObject->SetColorToBuffer();
	_drawObject->SetIndexToBuffer();

	drawObject_.push_back(_drawObject);

	UpdateModelMatrix();
	updateGL();
}
/*
void OpenGLWindow::AddPointsObject(int _id)
{
	pointsObject_.push_back(new PointsObject(_id, this));
	pointsObject_.back()->SetProgram(&phongProgram, &colorProgram, &renderVertexIdProgram);
	pointsObject_.back()->SetHandle(colorPositionHandle, colorColorHandle,
		phongPositionHandle, phongNormalHandle,
		renderVertexIdPositionHandle, renderVertexIdObjectIdHandle);


}

void OpenGLWindow::AddPointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals, float _pointSize)
{
	pointsObject_.push_back(new PointsObject(_id, this));
	pointsObject_.back()->SetProgram(&phongProgram, &colorProgram, &renderVertexIdProgram);
	pointsObject_.back()->SetHandle(colorPositionHandle, colorColorHandle, 
		phongPositionHandle, phongNormalHandle, 
		renderVertexIdPositionHandle, renderVertexIdObjectIdHandle);
	pointsObject_.back()->SetPosition(_vertices);
	pointsObject_.back()->SetNormal(_normals);
	pointsObject_.back()->SetPointSize(_pointSize);

	UpdateModelMatrix();

	updateGL();
}

void OpenGLWindow::AddPointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4b> _colors, float _pointSize)
{
	pointsObject_.push_back(new PointsObject(_id, this));
	pointsObject_.back()->SetProgram(&phongProgram, &colorProgram, &renderVertexIdProgram);
	pointsObject_.back()->SetHandle(colorPositionHandle, colorColorHandle,
		phongPositionHandle, phongNormalHandle,
		renderVertexIdPositionHandle, renderVertexIdObjectIdHandle);
	pointsObject_.back()->SetPosition(_vertices);
	pointsObject_.back()->SetColor(_colors);
	pointsObject_.back()->SetPointSize(_pointSize);

	UpdateModelMatrix();

	updateGL();
}

void OpenGLWindow::AddPointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals, std::vector<cv::Vec4b> _colors, float _pointSize)
{
	pointsObject_.push_back(new PointsObject(_id, this));
	pointsObject_.back()->SetProgram(&phongProgram, &colorProgram, &renderVertexIdProgram);
	pointsObject_.back()->SetHandle(colorPositionHandle, colorColorHandle,
		phongPositionHandle, phongNormalHandle,
		renderVertexIdPositionHandle, renderVertexIdObjectIdHandle);
	pointsObject_.back()->SetPosition(_vertices);
	pointsObject_.back()->SetNormal(_normals);
	pointsObject_.back()->SetColor(_colors);
	pointsObject_.back()->SetPointSize(_pointSize);

	UpdateModelMatrix();

	updateGL();

}

void OpenGLWindow::ChangePointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals, float _pointSize)
{
	for (int i = 0; i < pointsObject_.size(); i++)
	{
		if (pointsObject_[i]->get_id() == _id)
		{
			pointsObject_[i]->SetPosition(_vertices);
			pointsObject_[i]->SetNormal(_normals);
			pointsObject_[i]->SetPointSize(_pointSize);
		}
	}

	UpdateModelMatrix();

	updateGL();
}

void OpenGLWindow::ChangePointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4b> _colors, float _pointSize)
{
	for (int i = 0; i < pointsObject_.size(); i++)
	{
		if (pointsObject_[i]->get_id() == _id)
		{
			pointsObject_[i]->SetPosition(_vertices);
			pointsObject_[i]->SetColor(_colors);
			pointsObject_[i]->SetPointSize(_pointSize);
		}
	}

	UpdateModelMatrix();

	updateGL();
}

void OpenGLWindow::ChangePointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals, std::vector<cv::Vec4b> _colors, float _pointSize)
{
	for (int i = 0; i < pointsObject_.size(); i++)
	{
		if (pointsObject_[i]->get_id() == _id)
		{
			pointsObject_[i]->SetPosition(_vertices);
			pointsObject_[i]->SetNormal(_normals);
			pointsObject_[i]->SetColor(_colors);
			pointsObject_[i]->SetPointSize(_pointSize);
		}
	}

	UpdateModelMatrix();

	updateGL();

}
*/
void OpenGLWindow::mousePressEvent(QMouseEvent *event)
{

	if (event->buttons() & Qt::RightButton)
	{
		arcBall_->isClicked = true;
		arcBall_->MousePt.s.X = event->pos().x();
		arcBall_->MousePt.s.Y = event->pos().y();
		arcBall_->upstate();
		lastMouseButton = Qt::RightButton;
		lastMousePos = event->pos();
	}

	if (event->buttons() & Qt::MidButton)
	{
		lastMousePos = event->pos();
	}

	if (event->buttons() & Qt::LeftButton)
	{
		selectedRect_[0] = 
		lastMouseButton = Qt::LeftButton;
		lastMousePos = event->pos();
	}

	updateGL();
}

void OpenGLWindow::mouseReleaseEvent(QMouseEvent *event)
{
	if (lastMouseButton == (int)Qt::RightButton)
	{
		lastMouseButton = -1;
		arcBall_->isClicked = false;
		arcBall_->MousePt.s.X = event->pos().x();
		arcBall_->MousePt.s.Y = event->pos().y();
		arcBall_->upstate();
	}

	if (lastMouseButton == (int)Qt::MidButton)
	{
	}

	if (lastMouseButton == (int)Qt::LeftButton)
	{

	}

	updateGL();
}

void OpenGLWindow::mouseMoveEvent(QMouseEvent *event)
{
	if (event->buttons() & Qt::MidButton)
	{
		QMatrix4x4 mouseMatrix;
		mouseMatrix.setToIdentity();
		mouseMatrix.translate(-xTrans, -yTrans, -zTrans);
		mouseMatrix.scale(1 / scale, 1 / scale, 1 / scale);
		mouseMatrix *= rotateMatrix.inverted();
		mouseMatrix.translate((event->pos().x() - lastMousePos.x()) / (float)w, -(event->pos().y() - lastMousePos.y()) / (float)h, 0);
		mouseMatrix *= rotateMatrix;
		mouseMatrix.scale(scale, scale, scale);
		mouseMatrix.translate(xTrans, yTrans, zTrans);
		QVector4D v = mouseMatrix.column(3);
		xTrans += v[0];
		yTrans += v[1];
		zTrans += v[2];
		lastMousePos = event->pos();
	}

	if (event->buttons() & Qt::RightButton)
	{
		arcBall_->MousePt.s.X = event->pos().x();
		arcBall_->MousePt.s.Y = event->pos().y();
		arcBall_->upstate();
	}

	if (event->buttons() & Qt::LeftButton)
	{

	}

	updateGL();
}

void OpenGLWindow::wheelEvent(QWheelEvent *event)
{
	if (event->delta() > 0) scale = scale + scale * 0.1;
	else scale = scale - scale* 0.1;
	updateGL();
}

void OpenGLWindow::mouseDoubleClickEvent(QMouseEvent *event)
{
	if (event->button() != Qt::LeftButton) return;

	/*if (pickMode_ == 0)
	{
		const int MaxSize = 512;
		GLuint buffer[MaxSize];
		GLint viewport[4];
		GLint hits;

		glGetIntegerv(GL_VIEWPORT, viewport);
		glSelectBuffer(MaxSize, buffer);
		glRenderMode(GL_SELECT);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		gluPickMatrix((GLdouble)(event->x()), (GLdouble)(viewport[3] - event->y()), 10, 10, viewport);
		int w = width(), h = height();
		if (w > h)
			glOrtho(-static_cast<GLdouble>(w) / h, static_cast<GLdouble>(w) / h, -1.0, 1.0, -5000.0, 5000.0);
		else
			glOrtho(-1.0, 1.0, -static_cast<GLdouble>(h) / w, static_cast<GLdouble>(h) / w, -5000.0, 5000.0);

		glInitNames();
		int lastShowMode = showMode_;
		showMode_ = 3;
		Render();
		showMode_ = lastShowMode;

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glFlush();

		hits = glRenderMode(GL_RENDER);
		qDebug() << "hits: " << hits;
		if (hits > 0)
		{
			unsigned int i, j;
			GLuint names, *ptr, minZ, *ptrNames, numberOfNames;

			ptr = (GLuint *)buffer;
			minZ = 0xffffffff;

			for (i = 0; i < hits; i++) {
				names = *ptr;
				ptr++;
				if (*ptr < minZ) {
					numberOfNames = names;
					minZ = *ptr;
					ptrNames = ptr + 2;
				}

				ptr += names + 2;
			}

			ptr = ptrNames;
			qDebug() << "num: " << numberOfNames;
			for (j = 0; j < numberOfNames; j++, ptr++) {
				qDebug() << *ptr;
				selectedIndex_ = *ptr;
			}
			if (numberOfNames == 0) selectedIndex_ = -1;
		}
		else selectedIndex_ = -1;
	}*/

	if (pickMode_ == 1)
	{
		int lastShowMode = showMode_;
		showMode_ = 4;
		Render();
		QImage tmpImage = vertexIdFramebufferObject->toImage();
		QImage image(tmpImage.constBits(), tmpImage.width(), tmpImage.height(), QImage::Format_ARGB32);

		//printf("%d\n", image.format());
		//image.save("RRTVertexID.png");
		selectedIndex_ = -1;
		selectedObjectIndex_ = -1;
		int s = event->pos().x()-1;
		int t = event->pos().y()-1;
		for (int i = 0; i < 3; i++, s++)
			for (int j = 0; j < 3; j++, t++)
			{
				if (selectedIndex_ != -1) break;
				if (s<0 || s>w) continue;
				if (t<0 || t>h) continue;
				QColor rgb = image.pixelColor(s, t);
				int index = rgb.red() + rgb.green() * 256 + rgb.blue() * 256 * 256;

				int objectIndex = rgb.alpha();
				if (index >= 0 && objectIndex >= 0 && objectIndex < 255)
				{
					qDebug() << "selected: " << objectIndex << " " << index;
					selectedIndex_ = index;
					selectedObjectIndex_ = objectIndex;
					emit SendSelectedIndex(selectedObjectIndex_, selectedIndex_);
				}
				else
				{
					emit SendSelectedIndex(selectedObjectIndex_, selectedIndex_);
					qDebug() << "selected non";
				}
			}
		showMode_ = lastShowMode;
	}

	updateGL();
}

void OpenGLWindow::keyPressEvent(QKeyEvent *event)
{
	switch (event->key())
	{
	case Qt::Key_F1:
		shaderMode_ = 0;
		break;
	case Qt::Key_F2:
		shaderMode_ = 1;
		break;
	case Qt::Key_1:
		showMode_ = 0;
		break;
	case Qt::Key_2:
		showMode_ = 1;
		break;
	case Qt::Key_3:
		showMode_ = 2;
		break;
	case Qt::Key_R:
		scale = 1;
		xTrans = 0;
		yTrans = 0;
		zTrans = 0;
		Matrix4fSetIdentity(&arcBall_->Transform);
		Matrix3fSetIdentity(&arcBall_->LastRot);
		Matrix3fSetIdentity(&arcBall_->ThisRot);
		break;
	}

	updateGL();
}