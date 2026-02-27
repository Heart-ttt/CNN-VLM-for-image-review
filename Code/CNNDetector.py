from ultralytics import YOLO
import json


class CNN_Dec:
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
        :param model_path: YOLO模型文件路径（如 yolov8n.pt 或自训练模型）
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

if __name__ == "__main__":
    vlm = CNN_Dec("yolo11n.pt")
    vlm.detect("image.png")      