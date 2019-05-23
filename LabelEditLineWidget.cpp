#include "LabelEditLineWidget.h"

LabelEditLineWidget::LabelEditLineWidget(std::string _name, std::string _text, QWidget *parent)
	: QWidget(parent)
{
	mainWidget_ = new QWidget();
	mainLayout_ = new QHBoxLayout(mainWidget_);

	label_ = new QLabel(QString::fromLocal8Bit(_name.c_str()));
	lineEdit_ = new QLineEdit(QString::fromLocal8Bit(_text.c_str()));
	lineEdit_->setMinimumWidth(80);
	lineEdit_->setMaximumWidth(160);

	mainLayout_->addWidget(label_);
	mainLayout_->addWidget(lineEdit_);

	stackLayout_ = new QStackedLayout(this);
	stackLayout_->addWidget(mainWidget_);
	stackLayout_->setCurrentIndex(0);

	connect(lineEdit_, SIGNAL(editingFinished()), this, SLOT(EditingFinishedChanged()));
	//connect(lineEdit_, SIGNAL(returnPressed()), this, SLOT(LineEditReturnPressed()));
	fn_ = nullptr;
	fnBelong_ = nullptr;
}

LabelEditLineWidget::~LabelEditLineWidget()
{
	if (label_ != nullptr) delete label_;
	if (lineEdit_ != nullptr) delete lineEdit_;

	if (mainLayout_ != nullptr) delete mainLayout_;
	if (mainWidget_ != nullptr) delete mainWidget_;
	if (stackLayout_ != nullptr) delete stackLayout_;
}

void LabelEditLineWidget::EditingFinishedChanged()
{
	//qDebug() << lineEdit_->text();
	if (fn_ != nullptr && fnBelong_ != nullptr)
	{
		(fnBelong_->*fn_)(lineEdit_->text());
	}
	
}

void LabelEditLineWidget::LineEditReturnPressed()
{
	//qDebug() << lineEdit_->text();
	lineEdit_->clearFocus();
}

void LabelEditLineWidget::CreateConnect(mFuncString _fn, FnBelong *_fnBelong)
{
	fn_ = _fn;
	fnBelong_ = _fnBelong;
}

QString LabelEditLineWidget::Text()
{
	return lineEdit_->text();
}