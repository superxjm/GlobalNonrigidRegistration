#pragma once

#include <QWidget>
#include <QDebug>
#include <QPainter>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QVBoxLayout>
#include <QStackedLayout>

class ImageWidget : public QWidget
{
	Q_OBJECT

public:
	ImageWidget(int _imageWidgetId, int _imageWidth, int _imageHeight, int _imageWidgetWidth, int _imageWidgetHeight, QImage::Format _imageFormat, QWidget *parent=0);
	~ImageWidget();

	void SetImage(const uchar *data);
	int GetImageWidgetId() { return imageWidgetId_; }

private:
	QStackedLayout *stackLayout_;
	QVBoxLayout *mainLayout_;
	QWidget *mainWidget_;
	QLabel *imageLabel_;
	QImage::Format imageFormat_;
	QImage *image_;
	QImage img;

	int imageWidgetId_;
	int imageWidgetWidth_;
	int imageWidgetHeight_;
	int imageWidth_;
	int imageHeight_;

};
