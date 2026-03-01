from ultralytics import YOLO
import json
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

class YOLO_Dec:
    """
    YOLO检测器
    仅检测以下类别：
    - knife
    - baseball bat
    - scissors
    - wine glass
    """

    TARGET_CLASSES = ["knife", "baseball bat", "scissors", "wine glass"]

    def __init__(self, model_path: str):
        """
        初始化模型
        :param model_path: YOLO模型文件路径(如 yolov8n.pt 或自训练模型)
        """
        self.model = YOLO(model_path)

    def detect(self, image_path: str) -> str:
        """
        检测图片
        :param image_path: 图片路径
        :return: JSON字符串 {"category": , "confidence": }
        """

        results = self.model(image_path, verbose=False)

        best_category = None
        best_confidence = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                category_name = result.names[cls_id]

                if category_name in self.TARGET_CLASSES:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_category = category_name

        output = {
            "category": best_category,
            "confidence": round(best_confidence, 4)
        }

        return json.dumps(output)


class Classifier:
    """
    用于 simple / complex 图像复杂度分类
    """

    def __init__(self, weight_path=None, device="cuda"):
        """
        :param weight_path: 训练好的权重路径 best_mobilenetv4.pth
        :param device: "cuda" 或 "cpu"
        """

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 创建模型
        self.model = timm.create_model(
            "mobilenetv4_conv_small",
            pretrained=False,
            num_classes=2
        )

        if weight_path:
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # 类别映射（按ImageFolder顺序）
        self.class_map = {
            0: "complex",
            1: "simple"
        }

    def predict(self, image_path):
        """
        单张图片预测
        :return: dict {"category": , "confidence": }
        """

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        return {
            "category": self.class_map[pred.item()],
            "confidence": round(conf.item(), 4)
        }

    def predict_batch(self, image_paths):
        """
        批量预测
        :image_paths: 要检测的文件夹目录
        """

        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            images.append(img)

        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

        results = []
        for i in range(len(image_paths)):
            results.append({
                "image": image_paths[i],
                "category": self.class_map[preds[i].item()],
                "confidence": round(confs[i].item(), 4)
            })

        return results

if __name__ == "__main__":
    vlm = YOLO_Dec("yolo11n.pt")
    vlm.detect("../pictures/test_image.png")      
    classfy = Classifier("")
    result = classfy.predict("../pictures/test_image.png")
    print(result)
    