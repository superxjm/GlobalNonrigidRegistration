#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	CreateImageWindow();
	CreateOpenGLWindow();
	CreateMenu();
	CreateButton();
	CreateConnect();
	CreateConfig();

	register_ = new Register();
}

MainWindow::~MainWindow()
{
	if (imageWindow_!=nullptr) delete imageWindow_;
	if (openGLWindow_ != nullptr) delete openGLWindow_;
	printf("finish delete\n");
}

void MainWindow::CreateImageWindow()
{
	imageWindow_ = new ImageWindow(2);
	imageWindow_->setWindowTitle("ImageWindow");
	imageWindow_->resize(800, 600);
	imageWindow_->show();
}

void MainWindow::CreateOpenGLWindow()
{
	openGLWindow_ = new OpenGLWindow();
	openGLWindow_->resize(900, 800);
	openGLWindow_->setWindowTitle("OpenGLWindow");
	openGLWindow_->setFocusPolicy(Qt::TabFocus);
	openGLWindow_->show();

	MyPointsObject *mp = new MyPointsObject(0, openGLWindow_);
	openGLWindow_->AddObject(mp);

	PointsObject *np = new PointsObject(1, openGLWindow_);
	np->SetPointSize(4.0);
	openGLWindow_->AddObject(np);

	PointsObject *sp = new PointsObject(3, openGLWindow_);
	//sp->SetPointSize(4.0);
	openGLWindow_->AddObject(sp);

	PointsObject *dp = new PointsObject(2, openGLWindow_);
	openGLWindow_->AddObject(dp);

	LinesObject *sdl = new LinesObject(4, openGLWindow_);
	openGLWindow_->AddObject(sdl);

	LinesObject *corrl = new LinesObject(5, openGLWindow_);
	openGLWindow_->AddObject(corrl);

	PointsObject *dsp = new PointsObject(6, openGLWindow_);
	openGLWindow_->AddObject(dsp);

	PointsObject *ssp = new PointsObject(7, openGLWindow_);
	openGLWindow_->AddObject(ssp);
}

void MainWindow::CreateMenu()
{
	
	fileMenu_ = this->menuBar()->addMenu(tr("File"));
	openAction_ = fileMenu_->addAction(tr("Open"));

}

void MainWindow::CreateButton()
{
	mainWidget_ = new QWidget();
	mainLayout_ = new QVBoxLayout();

	nodeNumEachFragSpinWidget_ = new SpinWidget("NodeNum", 8, 1024, true);
	sampleVertexNumEachFragSpinWidget_ = new SpinWidget("SampleVertexNum", 1024, 65536, true);
	mainLayout_->addWidget(nodeNumEachFragSpinWidget_);
	mainLayout_->addWidget(sampleVertexNumEachFragSpinWidget_);
	weightWidget_ = new WeightWidget(tr("0.15"), tr("3.0"), tr("3.0"));
	mainLayout_->addWidget(weightWidget_);
	//nodeNumEachFragLineEdit_ = new QLineEdit("32");
	//vertexNumEachFragLineEdit_ = new QLineEdit("32768");
	//mainLayout_->addWidget(nodeNumEachFragLineEdit_);
	//mainLayout_->addWidget(vertexNumEachFragLineEdit_);
	createRigisterDeformationButton_ = new QPushButton(tr("Create Register"));
	mainLayout_->addWidget(createRigisterDeformationButton_);

	std::vector<std::string> methodString;
	methodString.push_back("KNN");
	methodString.push_back("PERSPECTIVE");
	methodString.push_back("KNN With ConsistantCheck");
	methodComboBoxWidget_ = new ComboBoxWidget("FindCorrespondencePointMethod", &methodString);
	mainLayout_->addWidget(methodComboBoxWidget_);
	
	distThresh1Widget_ = new LabelEditLineWidget("DistThresh1", "0.015");
	distThresh2Widget_ = new LabelEditLineWidget("DistThresh2", "0.0015");
	angleThreshWidget_ = new LabelEditLineWidget("AngleThresh", "25.0");
	findCorrespondencePointsButton_ = new QPushButton(tr("Find Correspondence Points"));
	mainLayout_->addWidget(distThresh1Widget_);
	mainLayout_->addWidget(distThresh2Widget_);
	mainLayout_->addWidget(angleThreshWidget_);
	mainLayout_->addWidget(findCorrespondencePointsButton_);

	iterNumSpinWidget_ = new SpinWidget("IterNum", 6, 20, false);
	//iterNumLineEdit_ = new QLineEdit("10");
	//mainLayout_->addWidget(iterNumLineEdit_);
	mainLayout_->addWidget(iterNumSpinWidget_);

	//acceptDataStepByStepButton_ = new QPushButton(tr("AcceptDataStepByStep"));
	acceptDataToghterButton_ = new QPushButton(tr("Run Register"));
	saveDataButton_ = new QPushButton(tr("Save Data"));
	//mainLayout_->addWidget(acceptDataStepByStepButton_);
	mainLayout_->addWidget(acceptDataToghterButton_);
	mainLayout_->addWidget(saveDataButton_);


	sourceCheckBox_ = new QCheckBox(tr("Source"));
	deformedCheckBox_ = new QCheckBox(tr("Deformed"));
	nodeCheckBox_ = new QCheckBox(tr("Node"));
	sourceSampleCheckBox_ = new QCheckBox(tr("Source Sample"));
	deformedSampleCheckBox_ = new QCheckBox(tr("Deformed Sample"));
	resultCheckBox_ = new QCheckBox(tr("Result"));
	corrLinesCheckBox_ = new QCheckBox(tr("Corr Lines"));
	s2dLinesCheckBox_ = new QCheckBox(tr("Source to Deformed"));
	sourceCheckBox_->setChecked(true);
	deformedCheckBox_->setChecked(true);
	nodeCheckBox_->setChecked(true);
	sourceSampleCheckBox_->setChecked(true);
	deformedSampleCheckBox_->setChecked(true);
	resultCheckBox_->setChecked(true);
	corrLinesCheckBox_->setChecked(true);
	s2dLinesCheckBox_->setChecked(true);

	mainLayout_->addWidget(sourceCheckBox_);
	mainLayout_->addWidget(deformedCheckBox_);
	mainLayout_->addWidget(nodeCheckBox_);
	mainLayout_->addWidget(sourceSampleCheckBox_);
	mainLayout_->addWidget(deformedSampleCheckBox_);
	mainLayout_->addWidget(resultCheckBox_);
	mainLayout_->addWidget(corrLinesCheckBox_);
	mainLayout_->addWidget(s2dLinesCheckBox_);

	mainWidget_->setLayout(mainLayout_);
	this->setCentralWidget(mainWidget_);

}

void MainWindow::CreateConnect()
{
	connect(openAction_, SIGNAL(triggered()), this, SLOT(OpenActionSlot()));

	//connect(acceptDataStepByStepButton_, SIGNAL(clicked()), this, SLOT(AcceptDataStepByStep()));
	connect(createRigisterDeformationButton_, SIGNAL(clicked()), this, SLOT(CreateRegisterDeformation()));
	connect(findCorrespondencePointsButton_, SIGNAL(clicked()), this, SLOT(FindCorrespondencePoints()));
	connect(acceptDataToghterButton_, SIGNAL(clicked()), this, SLOT(AcceptDataToghter()));
	connect(saveDataButton_, SIGNAL(clicked()), this, SLOT(SaveSlot()));
	connect(sourceCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(SourceCheckBoxSlot(int)));
	connect(deformedCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(DeformedCheckBoxSlot(int)));
	connect(nodeCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(NodeCheckBoxSlot(int)));
	connect(sourceSampleCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(SourceSampleCheckBoxSlot(int)));
	connect(deformedSampleCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(DeformedSampleCheckBoxSlot(int)));
	connect(resultCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(ResultCheckBoxSlot(int)));
	connect(corrLinesCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(CorrLinesCheckBoxSlot(int)));
	connect(s2dLinesCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(S2DLinesCheckBoxSlot(int)));
	
	//methodComboBoxWidget_->CreateConnect(&MainWindow::MethodChangedSlot, this);
	//test(&MainWindow::MethodChangedSlot);
}

void MainWindow::CreateConfig()
{
	config_ = new Config();
}

/*void MainWindow::AcceptDataStepByStep()
{
	register_->RegisterRun(time_, sum_vertex_num_);
    //openGLWindow_->updateGL();

	openGLWindow_->GetDrawObject(1)->SetPosition(register_->get_nodes());
	openGLWindow_->GetDrawObject(1)->SetColor(GenerateColor(register_->get_nodes().size(), cv::Vec4f(255,255,255,255)));
	openGLWindow_->GetDrawObject(3)->SetPosition(register_->get_source());
	openGLWindow_->GetDrawObject(3)->SetColor(register_->get_source_color());
	openGLWindow_->GetDrawObject(2)->SetPosition(register_->get_deformed());
	openGLWindow_->GetDrawObject(2)->SetColor(register_->get_deformed_color());
	openGLWindow_->GetDrawObject(4)->SetPosition(register_->get_source2deformed_vertex());
	openGLWindow_->GetDrawObject(4)->SetColor(register_->get_source2deformed_color());
	openGLWindow_->GetDrawObject(4)->SetIndex(register_->get_source2deformed_index());
	if (time_ != 0)
	{
		openGLWindow_->GetDrawObject(5)->SetPosition(register_->get_corr_vertex());
		openGLWindow_->GetDrawObject(5)->SetColor(register_->get_corr_color());
		openGLWindow_->GetDrawObject(5)->SetIndex(register_->get_corr_index());
	}
	openGLWindow_->GetDrawObject(6)->SetPosition(register_->get_deformed_sample_vertex());
	openGLWindow_->GetDrawObject(6)->SetColor(register_->get_deformed_sample_color());
	openGLWindow_->GetDrawObject(7)->SetPosition(register_->get_source_sample_vertex());
	openGLWindow_->GetDrawObject(7)->SetColor(register_->get_source_sample_color());

	std::vector<int> ids;
	ids.push_back(1);
	ids.push_back(2);
	ids.push_back(3);
	ids.push_back(4);
	if (time_ != 0) ids.push_back(5);
	ids.push_back(6);
	ids.push_back(7);
	openGLWindow_->Update(ids);

	time_++;
}*/

void MainWindow::FindCorrespondencePoints()
{
	int method = methodComboBoxWidget_->CurrentIndex();
	float distThresh1 = distThresh1Widget_->Text().toFloat();
	float distThresh2 = distThresh2Widget_->Text().toFloat();
	float angleThresh = angleThreshWidget_->Text().toFloat();

	switch (method)
	{
	case 0:
		qDebug() << "Find Correspondence Point Method: Knn";
		break;
	case 1:
		qDebug() << "Find Correspondence Point Method: Perspective";
		break;
	}
	qDebug() << "DistThresh1: " << distThresh1;
	qDebug() << "DistThresh2: " << distThresh2;
	qDebug() << "AngleThresh: " << angleThresh;

	register_->FindCorrespondencePoints(config_->get_camera_pose(), config_->get_camera_fxfycxcy(),
		method, distThresh1, distThresh2, angleThresh,
		config_->get_camera_width(), config_->get_camera_height());

	openGLWindow_->GetDrawObject(1)->SetPosition(register_->get_nodes());
	openGLWindow_->GetDrawObject(1)->SetColor(GenerateColor(register_->get_nodes().size(), cv::Vec4f(255, 255, 255, 255)));
	openGLWindow_->GetDrawObject(3)->SetPosition(register_->get_source());
	openGLWindow_->GetDrawObject(3)->SetColor(register_->get_source_color());
	openGLWindow_->GetDrawObject(2)->SetPosition(register_->get_deformed());
	openGLWindow_->GetDrawObject(2)->SetColor(register_->get_deformed_color());
	openGLWindow_->GetDrawObject(4)->SetPosition(register_->get_source2deformed_vertex());
	openGLWindow_->GetDrawObject(4)->SetColor(register_->get_source2deformed_color());
	openGLWindow_->GetDrawObject(4)->SetIndex(register_->get_source2deformed_index());
	openGLWindow_->GetDrawObject(5)->SetPosition(register_->get_corr_vertex());
	openGLWindow_->GetDrawObject(5)->SetColor(register_->get_corr_color());
	openGLWindow_->GetDrawObject(5)->SetIndex(register_->get_corr_index());
	openGLWindow_->GetDrawObject(6)->SetPosition(register_->get_deformed_sample_vertex());
	openGLWindow_->GetDrawObject(6)->SetColor(register_->get_deformed_sample_color());
	openGLWindow_->GetDrawObject(7)->SetPosition(register_->get_source_sample_vertex());
	openGLWindow_->GetDrawObject(7)->SetColor(register_->get_source_sample_color());

	std::vector<int> ids;
	ids.push_back(1);
	ids.push_back(2);
	ids.push_back(3);
	ids.push_back(4);
	ids.push_back(5);
	ids.push_back(6);
	ids.push_back(7);
	openGLWindow_->Update(ids);

	time_ = sum_vertex_num_.size() - 1;
}

void MainWindow::AcceptDataToghter()
{
	if (config_->get_vbo().size() == 0) return;

	int method = methodComboBoxWidget_->CurrentIndex();
	float distThresh1 = distThresh1Widget_->Text().toFloat();
	float distThresh2 = distThresh2Widget_->Text().toFloat();
	float angleThresh = angleThreshWidget_->Text().toFloat();

	switch (method)
	{
	case 0:
		qDebug() << "Find Correspondence Point Method: Knn";
		break;
	case 1:
		qDebug() << "Find Correspondence Point Method: Perspective";
		break;
	}
	qDebug() << "DistThresh1: " << distThresh1;
	qDebug() << "DistThresh2: " << distThresh2;
	qDebug() << "AngleThresh: " << angleThresh;

	int  iter_num = iterNumSpinWidget_->Text().toInt();
	if (iter_num!=0)
	register_->RegisterRun(iter_num, sum_vertex_num_, 
		config_->get_camera_pose(), config_->get_camera_fxfycxcy(), 
		method, distThresh1, distThresh2, angleThresh,
		config_->get_camera_width(), config_->get_camera_height());

	openGLWindow_->GetDrawObject(1)->SetPosition(register_->get_nodes());
	openGLWindow_->GetDrawObject(1)->SetColor(GenerateColor(register_->get_nodes().size(), cv::Vec4f(255, 255, 255, 255)));
	openGLWindow_->GetDrawObject(3)->SetPosition(register_->get_source());
	openGLWindow_->GetDrawObject(3)->SetColor(register_->get_source_color());
	openGLWindow_->GetDrawObject(2)->SetPosition(register_->get_deformed());
	openGLWindow_->GetDrawObject(2)->SetColor(register_->get_deformed_color());
	openGLWindow_->GetDrawObject(4)->SetPosition(register_->get_source2deformed_vertex());
	openGLWindow_->GetDrawObject(4)->SetColor(register_->get_source2deformed_color());
	openGLWindow_->GetDrawObject(4)->SetIndex(register_->get_source2deformed_index());
	openGLWindow_->GetDrawObject(5)->SetPosition(register_->get_corr_vertex());
	openGLWindow_->GetDrawObject(5)->SetColor(register_->get_corr_color());
	openGLWindow_->GetDrawObject(5)->SetIndex(register_->get_corr_index());
	openGLWindow_->GetDrawObject(6)->SetPosition(register_->get_deformed_sample_vertex());
	openGLWindow_->GetDrawObject(6)->SetColor(register_->get_deformed_sample_color());
	openGLWindow_->GetDrawObject(7)->SetPosition(register_->get_source_sample_vertex());
	openGLWindow_->GetDrawObject(7)->SetColor(register_->get_source_sample_color());

	std::vector<int> ids;
	ids.push_back(1);
	ids.push_back(2);
	ids.push_back(3);
	ids.push_back(4);
	ids.push_back(5);
	ids.push_back(6);
	ids.push_back(7);
	openGLWindow_->Update(ids);

	time_ = sum_vertex_num_.size() - 1;
}

std::vector<cv::Vec4b>& MainWindow::GenerateColor(int _num, cv::Vec4b _color)
{
	colors_.resize(_num);
	for (int i = 0; i < _num; i++)
		colors_[i] = _color;

	return colors_;
}

void MainWindow::CreateRegisterDeformation()
{
	if (config_->get_vbo().size() == 0) return;

	vbo_ = config_->get_vbo();
	sum_vertex_num_ = config_->get_sum_vertex_num();

	
	int  node_num = nodeNumEachFragSpinWidget_->Text().toInt();
	int  vertex_num = sampleVertexNumEachFragSpinWidget_->Text().toInt();
	GlobalParameter::SetNodeNumEachFrag(node_num);
	GlobalParameter::SetSampledVertexNumEachFrag(vertex_num);
	gs::weightGeo = weightWidget_->GeoText().toFloat();
	gs::weightReg = weightWidget_->RegText().toFloat();
	gs::weightRot = weightWidget_->RotText().toFloat();
	register_->CreateDeformation(((MyPointsObject*)openGLWindow_->GetDrawObject(0))->GetVertexBufferId(), sum_vertex_num_);

	time_ = sum_vertex_num_.size() - 1;
}

void MainWindow::SourceCheckBoxSlot(int _state)
{
	if (_state == 2)
	{
		openGLWindow_->GetDrawObject(3)->SetRenderFlag(true);
		openGLWindow_->updateGL();
	}
	else
	{
		openGLWindow_->GetDrawObject(3)->SetRenderFlag(false);
		openGLWindow_->updateGL();
	}
}

void MainWindow::DeformedCheckBoxSlot(int _state)
{
	if (_state == 2)
	{
		openGLWindow_->GetDrawObject(2)->SetRenderFlag(true);
		openGLWindow_->updateGL();
	}
	else
	{
		openGLWindow_->GetDrawObject(2)->SetRenderFlag(false);
		openGLWindow_->updateGL();
	}
}

void MainWindow::S2DLinesCheckBoxSlot(int _state)
{
	if (_state == 2)
	{
		openGLWindow_->GetDrawObject(4)->SetRenderFlag(true);
		openGLWindow_->updateGL();
	}
	else
	{
		openGLWindow_->GetDrawObject(4)->SetRenderFlag(false);
		openGLWindow_->updateGL();
	}
}

void MainWindow::CorrLinesCheckBoxSlot(int _state)
{
	if (_state == 2)
	{
		openGLWindow_->GetDrawObject(5)->SetRenderFlag(true);
		openGLWindow_->updateGL();
	}
	else
	{
		openGLWindow_->GetDrawObject(5)->SetRenderFlag(false);
		openGLWindow_->updateGL();
	}
}

void MainWindow::ResultCheckBoxSlot(int _state)
{
	if (_state == 2)
	{
		openGLWindow_->GetDrawObject(0)->SetRenderFlag(true);
		openGLWindow_->updateGL();
	}
	else
	{
		openGLWindow_->GetDrawObject(0)->SetRenderFlag(false);
		openGLWindow_->updateGL();
	}
}

void MainWindow::NodeCheckBoxSlot(int _state)
{
	if (_state == 2)
	{
		openGLWindow_->GetDrawObject(1)->SetRenderFlag(true);
		openGLWindow_->updateGL();
	}
	else
	{
		openGLWindow_->GetDrawObject(1)->SetRenderFlag(false);
		openGLWindow_->updateGL();
	}
}

void MainWindow::SourceSampleCheckBoxSlot(int _state)
{
	if (_state == 2)
	{
		openGLWindow_->GetDrawObject(7)->SetRenderFlag(true);
		openGLWindow_->updateGL();
	}
	else
	{
		openGLWindow_->GetDrawObject(7)->SetRenderFlag(false);
		openGLWindow_->updateGL();
	}
}

void MainWindow::DeformedSampleCheckBoxSlot(int _state)
{
	if (_state == 2)
	{
		openGLWindow_->GetDrawObject(6)->SetRenderFlag(true);
		openGLWindow_->updateGL();
	}
	else
	{
		openGLWindow_->GetDrawObject(6)->SetRenderFlag(false);
		openGLWindow_->updateGL();
	}
}

void MainWindow::SaveSlot()
{
	if (config_->get_vbo().size() == 0) return;

	std::vector<cv::Vec4f> vertex;
	std::vector<cv::Vec4f> normal;
	std::vector<cv::Vec4b> color;
	std::vector<cv::Vec4b> tot_color;
	std::vector<cv::Vec4f> tot_vertex = register_->get_deformed();
	std::vector<cv::Vec4f> tot_normal = register_->get_deformed_normal();

	tot_color.resize(tot_vertex.size());
	for (int i = 0; i < tot_color.size(); i++)
	{
		memcpy(&tot_color[i], &vbo_[i].colorTime.x, sizeof(cv::Vec4b));
		tot_color[i][3] = 255;
	}

	int num;
	for (int i = 0; i < time_; i++)
	{
		num = sum_vertex_num_[i + 1] - sum_vertex_num_[i];
		vertex.resize(num);
		memcpy(vertex.data(), tot_vertex.data() + sum_vertex_num_[i], num * sizeof(cv::Vec4f));
		color.resize(num);
		memcpy(color.data(), tot_color.data() + sum_vertex_num_[i], num * sizeof(cv::Vec4b));
		normal.resize(num);
		memcpy(normal.data(), tot_normal.data() + sum_vertex_num_[i], num * sizeof(cv::Vec4f));

		WritePointsToPly<float, float, uchar>("out_deformed_00000" + std::to_string(i) + ".ply", num, 
			reinterpret_cast<float*>(vertex.data()), 4, 
			reinterpret_cast<float*>(normal.data()), 4, 
			reinterpret_cast<uchar*>(color.data()), 4);
	}
}

void MainWindow::OpenActionSlot()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), ".", tr("Config Xml(*.xml)"));
	std::string st = std::string((const char*)fileName.toLocal8Bit());
	//std::cout << st << "\n";
	config_->ReadConfig(st);

	((MyPointsObject*)openGLWindow_->GetDrawObject(0))->SetVertex(config_->get_vbo());
	std::vector<int> ids;
	ids.push_back(0);
	openGLWindow_->Update(ids);

	//CreateRegisterDeformation();
}

void MainWindow::MethodChangedSlot(int _index)
{
	qDebug() << "MainWindow: " << _index;
}