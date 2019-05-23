#pragma once
#include <QString>

class MainWindow;
typedef MainWindow FnBelong;
typedef void (FnBelong::*mFunc)();
typedef void (FnBelong::*mFuncInt)(int _index);
typedef void (FnBelong::*mFuncString)(QString text);