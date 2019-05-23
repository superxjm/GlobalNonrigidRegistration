#pragma once
#include <QWidget>
#include <QDebug>
#include <QComboBox>
#include <QLabel>
#include <QStackedLayout>
#include <QHBoxLayout>
#include <QString>
#include <vector>
#include <string>
#include "ConnectFunctionPointer.h"

//class MainWindow;
//typedef void (MainWindow::*ActivatedIndexFunc)(int text);
//typedef void (MainWindow::*ActivatedStringFunc)(QString text);

class ComboBoxWidget : public QWidget
{
	Q_OBJECT

public:
	typedef void(ComboBoxWidget::*TestFunc)(int text);

public:
	ComboBoxWidget(std::string _name, std::vector<std::string> *_items, QWidget *parent = 0);
	~ComboBoxWidget();

	int CurrentIndex();

	void CreateConnect(mFuncInt _fn, FnBelong* _fnBelong);
	void CreateConnect1(TestFunc _testFn);

private:
	QStackedLayout *stackLayout_;
	QHBoxLayout *mainLayout_;
	QWidget *mainWidget_;

	QLabel    *label_;
	QComboBox *comboBox_;
	
	FnBelong *fnBelong_;
	mFuncInt  fn_;

private slots:
    void ComboBoxWidgetItemIndexChnaged(int _index);
	void TestSlot(int _index);

signals:
	void ItemIndexChangedSignals(int _index);
};

