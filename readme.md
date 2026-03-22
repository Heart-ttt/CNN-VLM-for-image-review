# 面向校园网高并发场景的CNN-VLM级联图像审核系统设计与轻量化研究
YOLO11+MobileNetV4+Qwen-VL

## 版本日志在 [CHANGELOG](./CHANGELOG.md) 中查看 

# Prerequisites

## 环境要求:[requirements](environment/requirements.txt)

> 
> `python==3.10.14`
> 
> 需要`conda虚拟环境`; `YOLO`; `ollama`; 以及`streamlib`交互工具
> 
> [Pytorch-GPU测试代码](environment/test_cuda.py)
>
> [YOLO测试代码](environment/test_yolo11.py)  YOLO模型在`./Code`文件夹下(已写好路径)
>
> [ollama官网](https://ollama.com/) | [ollama-python github](https://github.com/ollama/ollama-python)<br>
> Windows: `irm https://ollama.com/install.ps1 | iex`粘贴到PowerShell下载<br>
> Linux: `curl -fsSL https://ollama.com/install.sh | sh`
> 
> 安装完成后Windows需要打开客户端Linux需要启动服务`sudo ollama serve`<br>
> 并拉取VLM`ollama.pull('qwen3-vl:2b')`在运行测试<br>
> [ollama测试代码](environment/test_ollama.py)

## YOLO model training

1. 下载**赌博违规训练数据集**`https://images.cv/download/casino/1013`
2. 使用labelimg进行标注，标注成YOLO格式
3. 按照8:2划分训练集与验证集
4. 使用`yolov26n`作为基准模型进行训练

## MobileNetV4 model training
1. 下载**COCO训练数据集**`http://images.cocodataset.org/zips/train2017.zip`！数据集为16G注意空间
2. 下载**json文件**`http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
3. 解压文件放入train文件夹中**annotations_trainval2017.zip中只需要放`instances_train2017.json`**
4. train目录如下
```tree
├─train_mobilenetv4_coco.py
├─COCO
│  ├─annotations
│  │    └─instances_train2017.json
│  └─train2017
│
└─dataset(自动生成)
    └─train
        ├─complex
        └─simple
```
5. 运行`train_mobilenetv4_coco.py`

# Usage
# WebInterface