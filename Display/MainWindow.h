#pragma once

#include <QWidget>
#include <QMainWindow>
#include <QPushButton>
#include <QCheckBox>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QLineEdit>
#include "ImageWindow.h"
#include "OpenGLWindow.h"
#include "Helpers/xUtils.h"
#include "MyPointsObject.h"
#include "Register.h"
#include "MeshPointsIO.h"
#include "Config.h"
#include "SpinWidget.h"
#include "WeightWidget.h"
#include "ComboBoxWidget.h"
#include "LabelEditLineWidget.h"

class MainWindow : public QMainWindow
{
	Q_OBJECT
public:

public:
	MainWindow(QWidget *parent=0);
	~MainWindow();

	void CreateImageWindow();
	void CreateOpenGLWindow();
	void CreateMenu();
	void CreateButton();
	void CreateConnect();
	void CreateConfig();


private:
	std::vector<cv::Vec4b> colors_;
	std::vector<cv::Vec4b>& GenerateColor(int _num, cv::Vec4b _color);

private:

	QVBoxLayout *mainLayout_;
	QWidget *mainWidget_;

	QMenu *fileMenu_;
	QAction *openAction_;

	ComboBoxWidget      *methodComboBoxWidget_;
	LabelEditLineWidget *distThresh1Widget_;
	LabelEditLineWidget *distThresh2Widget_;
	LabelEditLineWidget *angleThreshWidget_;
	SpinWidget  *nodeNumEachFragSpinWidget_;
	SpinWidget  *sampleVertexNumEachFragSpinWidget_;
	WeightWidget *weightWidget_;
	//QLineEdit   *nodeNumEachFragLineEdit_;
	//QLineEdit   *vertexNumEachFragLineEdit_;
	QPushButton *createRigisterDeformationButton_;
	SpinWidget  *iterNumSpinWidget_;
	//QLineEdit   *iterNumLineEdit_;
	//QPushButton *acceptDataStepByStepButton_;
	QPushButton *findCorrespondencePointsButton_;
	QPushButton *acceptDataToghterButton_;
	QPushButton *saveDataButton_;

	QCheckBox *sourceCheckBox_;
	QCheckBox *deformedCheckBox_;
	QCheckBox *nodeCheckBox_;
	QCheckBox *sourceSampleCheckBox_;
	QCheckBox *deformedSampleCheckBox_;
	QCheckBox *resultCheckBox_;
	QCheckBox *corrLinesCheckBox_;
	QCheckBox *s2dLinesCheckBox_;


	ImageWindow *imageWindow_;
	OpenGLWindow *openGLWindow_;
	Config       *config_;


	int time_;
	Register *register_;
	std::vector<int> sum_vertex_num_;
	std::vector<VBOType> vbo_;

	void MethodChangedSlot(int _index);

public slots:
    //void AcceptDataStepByStep();
    void CreateRegisterDeformation();
	void FindCorrespondencePoints();
	void AcceptDataToghter();
	void SourceCheckBoxSlot(int _state);
	void DeformedCheckBoxSlot(int _state);
	void NodeCheckBoxSlot(int _state);
	void SourceSampleCheckBoxSlot(int _state);
	void DeformedSampleCheckBoxSlot(int _state);
	void ResultCheckBoxSlot(int _state);
	void CorrLinesCheckBoxSlot(int _state);
	void S2DLinesCheckBoxSlot(int _state);
	void SaveSlot();

	void OpenActionSlot();

};
