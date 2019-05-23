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
#include <QMouseEvent>
#include <deque>
#include "opencv2/core/core.hpp"
#include "ArcBall.h"
#include "PointsObject.h"
#include "LinesObject.h"

class OpenGLWindow : public QGLWidget
{
	Q_OBJECT

public:
	OpenGLWindow(QWidget *parent = 0);
	~OpenGLWindow();

	void initializeGL();
	void resizeGL(int width, int height);
	void paintGL();
	void initShader();
	void Init();

protected:
	void Render();
	void DrawPoints();
	void DrawColorPoints();
	void DrawPhongPoints();
	void DrawLines();
	void DrawColorLines();
	void DrawPhongLines();
	void DrawTriangles();
	void DrawColorTriangles();
	void DrawPhongTriangles();
	void DrawSelectedPoint();
	void RenderVertexIdToTexture();

	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mouseDoubleClickEvent(QMouseEvent *event);
	void wheelEvent(QWheelEvent *event);
	void keyPressEvent(QKeyEvent *event);

public:
	//void ChangePointsObjectColor(int _id, std::vector<int>& _indices, cv::Vec4b _color);

	//void SetVertex(std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4b> &_colors, std::vector<cv::Vec3f> &_normals,std::vector<cv::Vec3i> &_faceIndices);
	//void SetVertex(std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals);
	void AddObject(DrawObject *_drawObject);
	DrawObject* GetDrawObject(int _id)
	{
		int index = IdToIndex(_id);
		if (index == -1) return nullptr;
		return drawObject_[index];
	}
	/*void AddPointsObject(int _id);
	void AddPointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals, std::vector<cv::Vec4b> _colors, float _pointSize=1.0f);
	void AddPointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals, float _pointSize=1.0f);
	void AddPointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4b> _colors, float _pointSize=1.0f);
	void ChangePointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals, std::vector<cv::Vec4b> _colors, float _pointSize = 1.0f);
	void ChangePointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4f> &_normals, float _pointSize = 1.0f);
	void ChangePointsObject(int _id, std::vector<cv::Vec4f> &_vertices, std::vector<cv::Vec4b> _colors, float _pointSize = 1.0f);*/

	int IdToIndex(int _id)
	{
		for (int i = 0; i < drawObject_.size(); i++)
			if (drawObject_[i]->get_id() == _id) return i;
		
		return -1;
	}

	void Update(std::vector<int> &_ids)
	{
		int index;
		for (int i = 0; i < _ids.size(); i++)
		{
			//printf("update id: %d\n", _ids[i]);
			index = IdToIndex(_ids[i]);
			if (index != -1)
			{
				drawObject_[index]->SetPositionToBuffer();
				drawObject_[index]->SetNormalToBuffer();
				drawObject_[index]->SetColorToBuffer();
				drawObject_[index]->SetIndexToBuffer();
			}
		}

		UpdateModelMatrix();
		updateGL();
	}
	void UpdateModelMatrix();

public:

	//std::vector<cv::Vec4f> vertices_;

	int tot_num_vertex_;
	//int num_vertex_;
	int num_face_;

	QOpenGLBuffer *vertexPosBuffer_;
	QOpenGLBuffer *vertexColorBuffer_;
	QOpenGLBuffer *indexBuffer_;
	QOpenGLVertexArrayObject *VAO_;

	QOpenGLBuffer *RRTvertexPosBuffer_;
	QOpenGLVertexArrayObject *RRTVAO_;

	QOpenGLBuffer *PhongVertexPosBuffer_;
	QOpenGLBuffer *PhongVertexNorBuffer_;
	QOpenGLBuffer *PhongIndexBuffer_;
	QOpenGLVertexArrayObject *PhongVAO_;

	QVector3D vLightPosition;
	QVector4D vAmbientColor;
	QVector4D vDiffuseColor;
	QVector4D vSpecularColor;
	float fDepthChang;

	QMatrix4x4 mModelMatrix;
	QMatrix4x4 mViewMatrix;
	QMatrix4x4 mProjectionMatrix;
	QMatrix3x3 mNormalMatrix;

	QGLShaderProgram phongProgram;
	int phongPositionHandle;
	int phongNormalHandle;
	int phongModelMatrixHandle;
	int phongViewMatrixHandle;
	int phongProjectionMatrixHandle;
	int phongNormalMatrixHandle;
	int phongLightPositionHandle;
	int phongAmbientColorHandle;
	int phongDiffuseColorHandle;
	int phongSpecularColorHandle;
	int phongDepthChangeHandle;

	QGLShaderProgram colorProgram;
	int colorPositionHandle;
	int colorColorHandle;
	int colorMVPMatrixHandle;
	int colorDepthChangeHandle;

	QGLShaderProgram renderVertexIdProgram;
	int renderVertexIdPositionHandle;
	int renderVertexIdMVPMatrixHandle;
	int renderVertexIdObjectIdHandle;

	cv::Vec4f poid;
	float orthoLeft, orthoRight, orthoBottom, orthoTop, orthoNear, orthoFar;
	float initScale;
	float xInitTrans;
	float yInitTrans;
	float zInitTrans;
	float xTrans;
	float yTrans;
	float zTrans;
	float scale;
	QMatrix4x4 rotateMatrix;
	ArcBallT *arcBall_;
	QPoint lastMousePos;
	int lastMouseButton;

	int w, h;

	int showMode_;
	int shaderMode_;
	int pickMode_;


	cv::Vec4f selectedRect_;
	int selectedObjectIndex_;
	int selectedIndex_;
	std::vector<int> selectedIndices_;

	QOpenGLFramebufferObject *vertexIdFramebufferObject;

	//std::deque<PointsObject*> pointsObject_;
	std::deque<DrawObject*>   drawObject_;

signals:
	void SendSelectedIndex(int _objectIndex, int _vertexIndex);
};
