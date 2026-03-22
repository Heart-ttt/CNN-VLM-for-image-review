# 技术栈

# 基于华为信创平台的“云边端”协同信息处理的智慧工地
## CANN

## CUDA依赖

### ATC

## CNN

## 模型量化策略

# 基于4G机器视觉与毫米波雷达的网约车驾驶员危险驾驶行为综合检测系统

### DLIB

### RK3588+RKNN-Toolkit2
系统采用 Rockchip RK3588 作为主控芯片，配合 RKNN-Toolkit2 进行模型部署，实现高性能的边缘 AI 推理。
- RK3588 硬件特性：
    - 算力：内置 6 TOPS 算力的 NPU（神经网络处理器），支持 INT4/INT8/INT16/FP16 混合运算。
    - 架构：4核 Cortex-A76 + 4核 Cortex-A55 大八核 CPU， Mali-G610 MP4 GPU。
    - 视频处理：支持 8K@60fps 视频编解码，可同时处理多路高清摄像头输入。
- RKNN-Toolkit2 工作流：
    - 模型转换：将 PyTorch/TensorFlow 训练的模型（如 YOLOv8, MobileNet）转换为 .rknn 格式。
    - 量化优化：执行 PTQ（训练后量化）或 QAT（感知量化训练），将模型权重从 FP32 压缩至 INT8，在精度损失<1% 的前提下提升 3-5 倍推理速度。
    - 异构调度：通过 C++/Python API 调用 NPU 资源，实现视频流解码（RGA/MPP）与 AI 推理的流水线并行。
### R60ABD1毫米波雷达

## MQTT协议

## 阿里云IoT平台

## YOLO损失函数与mAP公式写法

---



## Transformer

## 注意力机制

## json

## dockor