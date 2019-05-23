#pragma once

#include <QWidget>
#include <QSpinBox>
#include <QSlider>
#include <QStackedLayout>
#include <QHBoxLayout>
#include <QLabel>

class SpinWidget : public QWidget
{
	Q_OBJECT

public:
	SpinWidget(std::string _name, int _min, int _max, bool _flag, QWidget *parent=0);
	~SpinWidget();

	QString Text() { return spinBox_->text(); };

private:
	QStackedLayout *stackLayout_;
	QHBoxLayout *mainLayout_;
	QWidget *mainWidget_;
	
	QLabel *label_;
	QSpinBox *spinBox_;
	//QSlider *slider_;

public slots:
    void ChangeSingleStepSlot(int _change);
};
