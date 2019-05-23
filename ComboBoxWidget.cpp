#include "ComboBoxWidget.h"



ComboBoxWidget::ComboBoxWidget(std::string _name, std::vector<std::string> *_items, QWidget *parent)
	:QWidget(parent)
{

	mainWidget_ = new QWidget();
	mainLayout_ = new QHBoxLayout(mainWidget_);

	label_ = new QLabel(QString::fromLocal8Bit(_name.c_str()));
	comboBox_ = new QComboBox();

	for (int i = 0; i < _items->size(); i++)
	{
		//comboBox_->addItem(tr("aaaa"));
		comboBox_->addItem(QString::fromLocal8Bit((*_items)[i].c_str()));
	}

	mainLayout_->addWidget(label_);
	mainLayout_->addWidget(comboBox_);

	stackLayout_ = new QStackedLayout(this);
	stackLayout_->addWidget(mainWidget_);
	stackLayout_->setCurrentIndex(0);

	//comboBox_->currentIndex();
	connect(comboBox_, SIGNAL(activated(int)), this, SLOT(ComboBoxWidgetItemIndexChnaged(int)));
	//CreateConnect1(&ComboBoxWidget::TestSlot);
	fnBelong_ = nullptr;
	fn_ = nullptr;
}

ComboBoxWidget::~ComboBoxWidget()
{
	if (label_ != nullptr)  delete label_;
	if (comboBox_ != nullptr) delete comboBox_;

	if (mainLayout_ != nullptr) delete mainLayout_;
	if (mainWidget_ != nullptr) delete mainWidget_;
	if (stackLayout_ != nullptr) delete stackLayout_;
}

void ComboBoxWidget::ComboBoxWidgetItemIndexChnaged(int _index)
{
	//qDebug() << "ComboBox: " << _index;
	if (fnBelong_ != nullptr && fn_ != nullptr)
	{
		(fnBelong_->*fn_)(_index);
	}
}

void ComboBoxWidget::CreateConnect(mFuncInt _fn, FnBelong* _fnBelong)
{
	fn_ = _fn;
	fnBelong_ = _fnBelong;
	//(_fnBelong->*_fn)(29);
	//connect(comboBox_, static_cast<void (QComboBox::*)(int)>(&QComboBox::activated), _fnBelong, _fn);
	//connect(this, &ComboBoxWidget::ItemIndexChangedSignals, this, ComboBoxWidget::ComboBoxWidgetItemIndexChnaged);
	//connect(comboBox_, &QComboBox::activated, (QObject*)_fnBelong, SLOT(_fn(int)));
	//connect(comboBox_, SIGNAL(activated(int)), (QObject*)_fnBelong, SLOT(_fn(int)));
	//connect(sourceCheckBox_, SIGNAL(stateChanged(int)), this, SLOT(SourceCheckBoxSlot(int)));
}

int ComboBoxWidget::CurrentIndex()
{
	return comboBox_->currentIndex();
}

void ComboBoxWidget::CreateConnect1(TestFunc _testFn)
{
	int a = 0;
	connect(comboBox_, static_cast<void (QComboBox::*)(int)>(&QComboBox::activated), this, _testFn);
}

void ComboBoxWidget::TestSlot(int _index)
{
	qDebug() << "Test " << _index;
}