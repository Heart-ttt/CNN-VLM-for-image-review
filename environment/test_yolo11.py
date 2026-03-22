import cv2
import torch
from ultralytics import YOLO
import os

img_path = "../pictures/278.jpg"
# img_path = "../pictures/test_image.png"
def test_yolo11n(img_path):
    print("="*30)
    print("开始测试 YOLO11n ...")
    print("="*30)

    # 1. 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"✅ GPU 检测到: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        device = "cuda"
    else:
        print("⚠️ 未检测到 GPU，将使用 CPU 运行 (速度较慢)")
        device = "cpu"

    # 2. 加载模型
    # 'yolo11n.pt' 会自动下载如果本地不存在
    # 也可以指定本地路径: model = YOLO("path/to/yolo11n.pt")
    print("\n🔄 正在加载模型 yolo11n.pt ...")
    try:
        # model = YOLO("../Code/yolo11n.pt")
        model = YOLO("../Code/model/gambling.pt")
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 将模型移动到指定设备 (可选，ultralytics 通常会自动处理，但显式指定更稳妥)
    model.to(device)

    # 3. 准备测试图片
    # 如果没有图片，Ultralytics 自带了一些测试图，或者我们创建一个简单的黑色图片
    # 移动到开头 
    
    # 如果本地没有 bus.jpg，尝试使用 ultralytics 自带的示例或创建一个虚拟图像
    if not os.path.exists(img_path):
        print(f"\n⚠️ 未找到 {img_path}，正在使用 Ultralytics 内置资产或生成测试图...")
        # 这里直接使用 ultralytics 的 predict 接口，它内部有处理逻辑
        # 为了演示，我们直接运行预测，Ultralytics 会在第一次运行时自动下载示例图或使用默认逻辑
        # 但为了代码清晰，我们假设用户有一张图，或者我们手动创建一个
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "YOLO11n Test", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        img_path = "dummy_test.jpg"
        cv2.imwrite(img_path, dummy_img)
        print(f"   已生成测试图片: {img_path}")

    # 4. 执行推理 (Inference)
    print(f"\n🚀 开始在 [{device}] 上进行推理...")
    
    # conf: 置信度阈值, iou: IOU 阈值, verbose: 是否打印详细信息
    results = model.predict(
        source=img_path, 
        conf=0.25, 
        iou=0.45, 
        device=device, 
        verbose=True
    )

    # 5. 处理结果
    result = results[0] # 取第一张图片的结果
    
    print("\n📊 检测结果:")
    if len(result.boxes) == 0:
        print("   未检测到任何目标。")
    else:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            print(f"   - 类别: {class_name}, 置信度: {confidence:.4f}")

    # 6. 保存结果图片
    output_path = "./output/yolo11n_result.png"
    result.save(filename=output_path)
    print(f"\n✅ 结果已保存至: {output_path}")
    
    # 如果需要弹窗显示 (仅在本地有GUI环境且安装了 opencv-python 非 headless 版时有效)
    # 如果是服务器/headless 环境，请注释掉下面几行
    try:
        # 读取保存的结果图
        res_img = cv2.imread(output_path)
        if res_img is not None:
            cv2.imshow("YOLO11n Detection", res_img)
            print("\n💡 按任意键关闭预览窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n⚠️ 无法显示图片窗口 (可能是 headless 环境): {e}")

    print("\n" + "="*30)
    print("测试完成!")
    print("="*30)

if __name__ == "__main__":
    test_yolo11n(img_path)