#include "ImageWidget.h"

ImageWidget::ImageWidget(int _imageWidgetId, int _imageWidth, int _imageHeight, int _imageWidgetWidth, int _imageWidgetHeight, QImage::Format _imageFormat, QWidget *parent)
	: QWidget(parent)
{
	imageWidgetId_ = _imageWidgetId;
	imageWidth_ = _imageWidth;
	imageHeight_ = _imageHeight;
	imageWidgetWidth_ = _imageWidgetWidth;
	imageWidgetHeight_ = _imageWidgetHeight;
	imageFormat_ = _imageFormat;

	imageLabel_ = new QLabel();
	image_ = new QImage(_imageWidth, _imageHeight, _imageFormat);

	imageLabel_->resize(imageWidgetWidth_, imageWidgetHeight_);
	//imageLabel_->setMaximumSize(imageWidgetWidth_, imageWidgetHeight_);
	imageLabel_->setMinimumSize(imageWidgetWidth_, imageWidgetHeight_);
	imageLabel_->setScaledContents(true);

	mainWidget_ = new QWidget();
	mainLayout_ = new QVBoxLayout(mainWidget_);
	mainLayout_->addWidget(imageLabel_);

	stackLayout_ = new QStackedLayout(this);
	stackLayout_->addWidget(mainWidget_);
	stackLayout_->setCurrentIndex(0);
}

ImageWidget::~ImageWidget()
{
	if (image_ != nullptr) delete image_;
	if (imageLabel_ != nullptr) delete imageLabel_;
	if (mainLayout_ != nullptr) delete mainLayout_;
	if (mainWidget_ != nullptr) delete mainWidget_;
	if (stackLayout_ != nullptr) delete stackLayout_;
}

void ImageWidget::SetImage(const uchar *data)
{
	QImage tmpImg;
	switch (imageFormat_)
	{
	case QImage::Format::Format_RGB888:
		memcpy(image_->bits(), data, imageWidth_*imageHeight_ * 3);
		tmpImg = image_->rgbSwapped();
	    memcpy(image_->bits(), tmpImg.bits(), imageWidth_*imageHeight_ * 3);
		break;
	case QImage::Format::Format_Grayscale8:
		memcpy(image_->bits(), data, imageWidth_*imageHeight_);
		break;
	}
	
	imageLabel_->setPixmap(QPixmap::fromImage(image_->scaled(imageWidgetWidth_, imageWidgetHeight_, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)));
	//imageLabel_->setPixmap(QPixmap::fromImage(image_->scaled(imageWidgetWidth_, imageWidgetHeight_)));

}
