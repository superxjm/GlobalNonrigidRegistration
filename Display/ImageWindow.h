#pragma once

#include <QWidget>
#include <QVector>
#include <QGridLayout>
#include "ImageWidget.h"
#include <QScrollArea>

class ImageWindow : public QWidget
{
	Q_OBJECT

public:
	ImageWindow(int _cols, QWidget *parent = 0);
	~ImageWindow();

	void AddImageWidget(int _imageWidgetId, QImage::Format _imageFormat, int _imageWidth, int _imageHeight, int _imageWidgetWidth, int _imageWidgetHeight);
	void AddImageWidget(uchar* _data, int _imageWidgetId, QImage::Format _imageFormat, int _imageWidth, int _imageHeight, int _imageWidgetWidth, int _imageWidgetHeight);
	void SetImageWidgetData(int _imageWidgetId, const uchar *_data);

	int ImageWidgetIdToIndex(int _imageWidgetId);
	
private:

	int cols_;
	QGridLayout *mainLayout_;
	QStackedLayout *stackLayout_;
	QScrollArea *mainScrollArea_;
	QWidget *mainWidget_;
	QVector<ImageWidget*> imageWidgets_;
};
