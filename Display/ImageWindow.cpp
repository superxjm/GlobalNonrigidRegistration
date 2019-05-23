#include "ImageWindow.h"

ImageWindow::ImageWindow(int _cols, QWidget *parent)
	: QWidget(parent)
{
	cols_ = _cols;
	mainWidget_ = new QWidget();
	mainScrollArea_ = new QScrollArea();
	mainScrollArea_->setWidget(mainWidget_);
	mainScrollArea_->setWidgetResizable(true);
	mainLayout_ = new QGridLayout(mainWidget_);

	stackLayout_ = new QStackedLayout(this);
	stackLayout_->addWidget(mainScrollArea_);
	stackLayout_->setCurrentIndex(0);
}

ImageWindow::~ImageWindow()
{
	for (int i = 0; i < imageWidgets_.size(); i++)
	{
		delete imageWidgets_[i];
	}
	if (mainLayout_ != nullptr) delete mainLayout_;
	if (mainWidget_ != nullptr) delete mainWidget_;
	if (mainScrollArea_ != nullptr) delete mainScrollArea_;
	if (stackLayout_ != nullptr) delete stackLayout_;
}

void ImageWindow::AddImageWidget(int _imageWidgetId, QImage::Format _imageFormat, int _imageWidth, int _imageHeight, int _imageWidgetWidth, int _imageWidgetHeight)
{
	imageWidgets_.push_back(new ImageWidget(_imageWidgetId, _imageWidth, _imageHeight, _imageWidgetWidth, _imageWidgetHeight, _imageFormat));
	//imageWidgets_.back()->SetImage(data);
	imageWidgets_.back()->setFocusPolicy(Qt::ClickFocus);
	mainLayout_->addWidget(imageWidgets_.back(), (imageWidgets_.size() - 1) / cols_, (imageWidgets_.size() - 1) % cols_);
}

void ImageWindow::AddImageWidget(uchar* _data, int _imageWidgetId, QImage::Format _imageFormat, int _imageWidth, int _imageHeight, int _imageWidgetWidth, int _imageWidgetHeight)
{
	imageWidgets_.push_back(new ImageWidget(_imageWidgetId, _imageWidth, _imageHeight, _imageWidgetWidth, _imageWidgetHeight, _imageFormat));
	imageWidgets_.back()->SetImage(_data);
	imageWidgets_.back()->setFocusPolicy(Qt::ClickFocus);
	mainLayout_->addWidget(imageWidgets_.back(), (imageWidgets_.size() - 1) / cols_, (imageWidgets_.size() - 1) % cols_);
}

void ImageWindow::SetImageWidgetData(int _imageWidgetId, const uchar *_data)
{
	imageWidgets_[ImageWidgetIdToIndex(_imageWidgetId)]->SetImage(_data);
}

int ImageWindow::ImageWidgetIdToIndex(int _imageWidgetId)
{
	for (int i = 0; i < imageWidgets_.size(); i++)
	{
		if (imageWidgets_[i]->GetImageWidgetId() == _imageWidgetId)
		{
			return i;
		}
	}
}