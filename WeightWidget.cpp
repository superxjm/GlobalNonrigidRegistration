#include "WeightWidget.h"

WeightWidget::WeightWidget(QString geoWeight, QString regWeight, QString rotWeight, QWidget *parent)
	: QWidget(parent)
{

	//this->setMaximumWidth(60);

	mainWidget_ = new QWidget();
	mainLayout_ = new QHBoxLayout(mainWidget_);
	
	geoLabel_ = new QLabel(tr("GeoWeight"));
	geoLineEdit_ = new QLineEdit(geoWeight);
	regLabel_ = new QLabel(tr("RegWeight"));
	regLineEdit_ = new QLineEdit(regWeight);
	rotLabel_ = new QLabel(tr("RotWeight"));
	rotLineEdit_ = new QLineEdit(rotWeight);

	mainLayout_->addWidget(geoLabel_);
	mainLayout_->addWidget(geoLineEdit_);
	mainLayout_->addWidget(regLabel_);
	mainLayout_->addWidget(regLineEdit_);
	mainLayout_->addWidget(rotLabel_);
	mainLayout_->addWidget(rotLineEdit_);

	stackLayout_ = new QStackedLayout(this);
	stackLayout_->addWidget(mainWidget_);
	stackLayout_->setCurrentIndex(0);
}

WeightWidget::~WeightWidget()
{
	if (geoLabel_ != nullptr) delete geoLabel_;
	if (geoLineEdit_ != nullptr) delete geoLineEdit_;
	if (regLabel_ != nullptr) delete regLabel_;
	if (regLineEdit_ != nullptr) delete regLineEdit_;
	if (rotLabel_ != nullptr) delete rotLabel_;
	if (rotLineEdit_ != nullptr) delete rotLineEdit_;

	if (mainLayout_ != nullptr) delete mainLayout_;
	if (mainWidget_ != nullptr) delete mainWidget_;
	if (stackLayout_ != nullptr) delete stackLayout_;
}
