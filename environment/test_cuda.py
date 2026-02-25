import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ 成功检测到 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存总量: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} GB")
    # 简单测试矩阵乘法
    x = torch.rand(5, 3).cuda()
    y = torch.rand(3, 5).cuda()
    z = torch.mm(x, y)
    print("✅ GPU 计算测试通过！")
else:
    print("❌ 未检测到 GPU,将使用 CPU。")