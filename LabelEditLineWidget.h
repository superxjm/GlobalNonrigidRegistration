#pragma once

#include <QWidget>
#include <QStackedLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QDebug>
#include "ConnectFunctionPointer.h"

class LabelEditLineWidget : public QWidget
{
	Q_OBJECT

public:
	LabelEditLineWidget(std::string _name, std::string _text,QWidget *parent=0);
	~LabelEditLineWidget();

	QString Text();
	void CreateConnect(mFuncString _fn, FnBelong *_fnBelong);

private:
	QStackedLayout *stackLayout_;
	QHBoxLayout *mainLayout_;
	QWidget *mainWidget_;

	QLabel *label_;
	QLineEdit *lineEdit_;

	FnBelong     *fnBelong_;
	mFuncString   fn_;

private slots:
    void EditingFinishedChanged();
	void LineEditReturnPressed();
};
