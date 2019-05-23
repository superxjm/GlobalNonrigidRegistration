#pragma once

#include <QWidget>
#include <QWidget>
#include <QStackedLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include "ConnectFunctionPointer.h"
#include <QVector>
#include "LabelEditLineWidget.h"

class LabelEditLineListWidget : public QWidget
{
	Q_OBJECT

public:
	LabelEditLineListWidget(QWidget *parent=0);
	~LabelEditLineListWidget();

private:

	QStackedLayout *stackLayout_;
	QHBoxLayout *mainLayout_;
	QWidget *mainWidget_;

private:
	QVector<LabelEditLineWidget*> labelEditLines_;
};
