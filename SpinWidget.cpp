#include "SpinWidget.h"

SpinWidget::SpinWidget(std::string _name, int _min, int _max, bool _flag, QWidget *parent)
	: QWidget(parent)
{
	label_ = new QLabel();
	QString st(QString::fromLocal8Bit(_name.c_str()));
	label_->setText(st);
	spinBox_ = new QSpinBox();
	spinBox_->setValue(_min);
	if (_flag) spinBox_->setSingleStep(_min);
	else spinBox_->setSingleStep(1);
	spinBox_->setMinimum(_min);
	spinBox_->setMaximum(_max); 
	//slider_ = new QSlider(Qt::Horizontal);
	//slider_->setMinimum(8);
	//slider_->setMaximum(1024);

	mainWidget_ = new QWidget();
	mainLayout_ = new QHBoxLayout(mainWidget_);
	mainLayout_->addWidget(label_);
	mainLayout_->addWidget(spinBox_);
	//mainLayout_->addWidget(slider_);

	stackLayout_ = new QStackedLayout(this);
	stackLayout_->addWidget(mainWidget_);
	stackLayout_->setCurrentIndex(0);

	if(_flag) connect(spinBox_, SIGNAL(valueChanged(int)), this, SLOT(ChangeSingleStepSlot(int)));
}

SpinWidget::~SpinWidget()
{
	if (label_ != nullptr) delete label_;
	if (spinBox_ != nullptr) delete spinBox_;
	//if (slider_ != nullptr) delete slider_;
	if (mainLayout_ != nullptr) delete mainLayout_;
	if (mainWidget_ != nullptr) delete mainWidget_;
	if (stackLayout_ != nullptr) delete stackLayout_;
}

void SpinWidget::ChangeSingleStepSlot(int _change)
{
	spinBox_->setSingleStep(_change);
	//slider_->setValue(_change);
	//slider_->setSingleStep(_change);
}