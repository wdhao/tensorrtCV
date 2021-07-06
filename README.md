## 项目介绍

* 零代码：只需配置文件（json文件）和权重文件即可生成engin文件，实现trt部署。
* 自动化生成配置文件：由pt模型文件可以自动化生成json文件。
* 可视化网络：便于查验和原始网络（比如pytorch）的区别。
* debug教程：方便对比trt输出和pytorch模型输出的区别，从而方便定位部署上的问题。

# 项目结构

```
tensorrtCV
    |
    |- src
    |  |
    |  |- plugin
    |  |  |-xxx.h xxx.cu
    |  |-xxx.cpp
    |  |-xxx.h
    |- example
    |  |
    |  |-ddrnet
    |  | |-main.cpp
    |
    |  |-yolov5
    |  | |-main.cpp
    |
    |- model
    |  |-xxx.json
```

* `src` 为tensort项目的主代码目录。其中包含`plugin`目录存放啊各种plugin
* example目录中为不同网络的demo代码
* model目录中为各种网络的json文件。

## 编译运行

### win/linux

统一使用cmake管理，**需要在CMakeLists手动修改 OpenCV_DIR TENSORRT_DIR **

编译命令：

```makefile
mkdir build
cd build
cmake .. 
// win下指定vs编译
// cmake .. -G "Visual Studio 15 2017 Win64"
make
// win下vs打开tensorrtF.sln编译运行
```

### 自动化生成json文件

* [pytorch-classification](https://github.com/AlfengYuan/pytorch-classification)

## Comming Soon

- [ ] 小白系列教程
- [ ] json网络可视化

**欢迎各位同学PR模型配置(json文件)和新功能。**
**另外，请关注我的微信公众号（CV加速器），定期有直播讲解整个工程和集中回答问题。**

