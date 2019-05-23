#pragma once

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QStackedLayout>
#include <QHBoxLayout>

class WeightWidget : public QWidget
{
	Q_OBJECT

public:
	WeightWidget(QString geoWeight, QString regWeight, QString rotWeight, QWidget *parent=0);
	~WeightWidget();

	QString GeoText() { return geoLineEdit_->text(); }
	QString RegText() { return regLineEdit_->text(); }
	QString RotText() { return rotLineEdit_->text(); }

private:
	QStackedLayout *stackLayout_;
	QHBoxLayout *mainLayout_;
	QWidget *mainWidget_;

	QLabel *geoLabel_;
	QLabel *regLabel_;
	QLabel *rotLabel_;
	QLineEdit *geoLineEdit_;
	QLineEdit *regLineEdit_;
	QLineEdit *rotLineEdit_;

};
