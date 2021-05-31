## 项目介绍

* 零代码：只需配置文件（json文件）和权重文件即可生成engin文件，实现trt部署。
* 自动化生成配置文件：由pt模型文件可以自动化生成json文件。
* 可视化网络：便于查验和原始网络（比如pytorch）的区别。
* debug教程：方便对比trt输出和pytorch模型输出的区别，从而方便定位部署上的问题。

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

### 运行

```
tensorrtF.exe [-e / -t] [jsonPath] [imgPath]
# -e 为生成eging模式
# -t 为测试模式
```

### 自动化生成json文件



### debug教程

**1： 首先在json文件中标记output层。**

比如查看第一层卷积层的输出是否和pytorch相同，在第一层的json文件中加入output标志：

![image-20210531104549416](https://user-images.githubusercontent.com/20653176/120133816-688fae00-c1ff-11eb-8d3d-ed4a8901187b.png)



注意：

* outputBlobName可以有多个：但是要满足标记output的层的个数和outputBlobName个数相同。debug模式下建议只有一个。
* outputSize：debug模式下建议写1000，这样无需每次修改。

**2 : 保存infer输出**

engine前向时，在infer前向函数后添加代码，将结果保存到txt中：

```c++
std::ofstream ofile;  
ofile.setf(std::ios::fixed, std::ios::floatfield);
ofile.precision(4);
ofile.open("trt_out.txt");
for (int i=0; i< 1000; ++i)
{
    ofile << prob[i];
    ofile << "\n";
    //std::cout << prob[i] << std::endl;
}
ofile.close();
```

**3： 保存pytorch对应层的输出**

在pytorch代码对应层的位置添加代码

```python
import numpy as np

d = x.detach().cpu().numpy().flatten().reshape(-1,1) # x为要对比层的输出
np.savetxt("py_out.txt", d[:1000], fmt="%.4f")
```

**4： 对比**

可人工对比，或者利用脚本对比：

```python
import os
import numpy as np
def read_file(file):
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        #print(line)
        try:
            data.append(float(line.strip()))
        except ValueError:
            print(line)
    return data

def compare_txt(file1, file2， delta=0.001):
    data1 = read_file(file1)
    data2 = read_file(file2)
    assert len(data1)==len(data2)
    diff_count = 0
    for i in range(len(data1)):
        d1, d2 = data1[i], data2[i]
        if abs(d1-d2) > delta:
            #print("line %d:%f %f"%(i, d1, d2))
            diff_count += 1
    return diff_count, len(data1)

if __name__ =="__main__":
    d1 = "trt_out.txt"
    d2 = "py_out.txt"
    diff_count, total_count = compare_txt(d1, d2)
    print("error count %d"%diff_count)
    print("total count %d"%total_count)
```



## Comming Soon

- [ ] json网络可视化
- [ ] 常见网络的json文件





**欢迎各位同学PR模型配置(json文件)和新功能。**
**另外，请关注我的微信公众号（CV加速器），定期有直播讲解整个工程和集中回答问题。**

