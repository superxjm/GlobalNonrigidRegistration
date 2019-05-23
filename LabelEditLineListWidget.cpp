#include "LabelEditLineListWidget.h"

LabelEditLineListWidget::LabelEditLineListWidget(QWidget *parent)
	: QWidget(parent)
{
	mainWidget_ = new QWidget();
	mainLayout_ = new QHBoxLayout(mainWidget_);
	
	stackLayout_ = new QStackedLayout(this);
	stackLayout_->addWidget(mainWidget_);
	stackLayout_->setCurrentIndex(0);
}

LabelEditLineListWidget::~LabelEditLineListWidget()
{
	for (int i = 0; i < labelEditLines_.size(); i++)
	{
		if (labelEditLines_[i] != nullptr) delete labelEditLines_[i];
	}

	if (mainLayout_ != nullptr) delete mainLayout_;
	if (mainWidget_ != nullptr) delete mainWidget_;
	if (stackLayout_ != nullptr) delete stackLayout_;
}
