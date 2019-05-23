openGLWindow_ = new OpenGLWindow();
openGLWindow_->resize(w, h);
openGLWindow_->setWindowTitle("OpenGLWindow");
openGLWindow_->setFocusPolicy(Qt::TabFocus);//Qt::ClickFocus
openGLWindow_->show(); //一定要show一下用于初始化

openGLWindow_->SetVertexBuffer() //传入模型数据,具体参数看函数定义
操作:
F1:颜色 F2:phong光照模型
1:点 2:线 3:面
r:复原
鼠标右键双击拾取点,左键旋转,中建平移

依赖Qt和OpenCV