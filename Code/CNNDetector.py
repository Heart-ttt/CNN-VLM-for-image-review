from ultralytics import YOLO
import json
import torch
import timm
from torchvision import transforms
from PIL import Image
import easyocr
import ahocorasick
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import List, Tuple, Dict

class YOLODetector:
    """
    YOLO违规物体检测器
    """

    TARGET_CLASSES = [
        "knife",
        "baseball bat",
        "scissors",
        "wine glass"
    ]

    def __init__(self, model_path: str, device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = YOLO(model_path)

    def detect(self, image_path: str) -> dict:

        results = self.model(
            image_path,
            device=self.device,
            verbose=False
        )

        best_category = None
        best_confidence = 0.0

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                category_name = result.names[cls_id]

                if category_name in self.TARGET_CLASSES:

                    if confidence > best_confidence:

                        best_confidence = confidence
                        best_category = category_name

        return {
            "category": best_category,
            "confidence": round(best_confidence, 4)
        }


class ImgClassifier:
    """
    图像复杂度分类器
    simple / complex
    """

    def __init__(self, weight_path=None, device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = timm.create_model(
            "mobilenetv4_conv_small",
            pretrained=False,
            num_classes=2
        )

        if weight_path:

            checkpoint = torch.load(
                weight_path,
                map_location=self.device,
                weights_only=True
            )

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.class_map = {
            0: "complex",
            1: "simple"
        }

    def preprocess(self, image_path):

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        return img.unsqueeze(0).to(self.device)

    def predict(self, image_path):

        img = self.preprocess(image_path)

        with torch.inference_mode():

            outputs = self.model(img)

            probs = torch.softmax(outputs, dim=1)

            conf, pred = torch.max(probs, 1)

        return {
            "category": self.class_map[pred.item()],
            "confidence": round(conf.item(), 4)
        }

    def predict_batch(self, image_paths):

        images = []

        for path in image_paths:

            img = Image.open(path).convert("RGB")
            img = self.transform(img)

            images.append(img)

        images = torch.stack(images).to(self.device)

        with torch.inference_mode():

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

class TextAuditPipeline:
    """
    文字合规化判定
    """
    def __init__(self, langs=['ch_sim', 'en'], use_gpu=True):
        """
        初始化审核流水线
        :param langs: 需要支持的语言列表，['ch_sim', 'en'] 表示简体中文和英文
        :param use_gpu: 是否启用 GPU 加速
        """
        # 1. 初始化 EasyOCR
        # 首次运行会自动下载模型，建议预先下载好放入服务器
        print(f"Initializing EasyOCR with GPU={use_gpu}...")
        self.reader = easyocr.Reader(langs, gpu=use_gpu)
        
        # 2. 初始化规则引擎 (AC 自动机)
        self.automaton = ahocorasick.Automaton()
        self._load_violation_keywords()
        self.automaton.make_automaton()
        
        # 3. 初始化 DistilBERT 语义模型
        # 使用针对中文情感/毒性分类微调过的模型，这里以通用中文模型为例
        # 实际生产中建议使用自有违规语料库微调后的模型
        print("Initializing DistilBERT...")
        self.model_name = "uer/distilbert-base-chinese-cluecorpussmall" # 或者自定义微调模型路径
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.nlp_model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.nlp_model.eval()
        
        # 移动模型到 GPU (如果可用)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.nlp_model.to(self.device)
        
        # 阈值配置
        self.RISK_THRESHOLD = 0.85  # 高于此值直接拦截
        self.UNCERTAIN_THRESHOLD = 0.45 # 低于此值放行，中间值送入 VLM

    def _load_violation_keywords(self):
        """加载违规关键词库"""
        # 示例关键词，实际应从数据库或配置文件加载成千上万个词
        keywords = [
            "办证", "发票", "赌博", "色情", "暴力", "反动", "作弊", 
            "代考", "枪支", "毒品", "特定敏感人名", "特定违规短语"
        ]
        for idx, keyword in enumerate(keywords):
            self.automaton.add_word(keyword, (idx, keyword))

    def extract_text(self, image_path: str) -> str:
        """使用 EasyOCR 提取图片文字"""
        try:
            # result 格式: [([坐标], 文本, 置信度), ...]
            results = self.reader.readtext(image_path, detail=0) # detail=0 只返回文本列表
            full_text = " ".join(results)
            return full_text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def rule_filter(self, text: str) -> Tuple[bool, List[str]]:
        """
        使用 AC 自动机进行关键词匹配
        :return: (是否命中, 命中的关键词列表)
        """
        hits = []
        for end_index, (origin_index, value) in self.automaton.iter(text):
            hits.append(value)
        
        if hits:
            return True, list(set(hits)) # 去重
        return False, []

    def semantic_analysis(self, text: str) -> Tuple[float, str]:
        """
        使用 DistilBERT 进行语义风险评分
        :return: (风险分数 0-1, 判定标签)
        """
        if not text or len(text) < 2:
            return 0.0, "SAFE"
            
        # 截断过长的文本 (DistilBERT 最大长度通常为 512)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.nlp_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # 假设 label 1 是 "违规/风险", label 0 是 "安全"
            # 注意：具体 label 映射取决于您使用的模型训练时的定义
            risk_score = probs[0][1].item() 
            label = "RISK" if risk_score > 0.5 else "SAFE"
            
        return risk_score, label

    def audit(self, image_path: str) -> Dict:
        """
        完整审核流程
        :return: 包含决策结果、原因、风险分数的字典
        """
        # Step 1: OCR 提取
        text = self.extract_text(image_path)
        if not text:
            return {"decision": "PASS", "reason": "No text detected", "text": ""}
        
        # Step 2: 规则过滤 (最快)
        is_hit, hits = self.rule_filter(text)
        if is_hit:
            return {
                "decision": "BLOCK", 
                "reason": f"Hit keywords: {hits}", 
                "text": text,
                "stage": "Rule Engine"
            }
        
        # Step 3: 语义分析 (较快)
        risk_score, label = self.semantic_analysis(text)
        
        if risk_score >= self.RISK_THRESHOLD:
            return {
                "decision": "BLOCK",
                "reason": f"High semantic risk (Score: {risk_score:.4f})",
                "text": text,
                "stage": "DistilBERT"
            }
        elif risk_score <= self.UNCERTAIN_THRESHOLD:
            return {
                "decision": "PASS",
                "reason": f"Low semantic risk (Score: {risk_score:.4f})",
                "text": text,
                "stage": "DistilBERT"
            }
        else:
            # Step 4: 模糊地带，升级至 Qwen3-VL (最慢但最准)
            # 这里仅返回标记，实际系统中会在此处调用 Qwen3-VL 接口
            return {
                "decision": "REVIEW_VLM", 
                "reason": f"Uncertain semantic risk (Score: {risk_score:.4f}), escalating to VLM",
                "text": text,
                "stage": "Escalation"
            }
        

if __name__ == "__main__":

    # yolo = YOLODetector("yolo11n.pt")
    # result = yolo.detect("../pictures/test_image.png")
    # print("YOLO:", result)

    # classifier = ImgClassifier("best_mobilenetv4.pth")
    # result = classifier.predict("../pictures/test_image.png")

    pipeline = TextAuditPipeline(langs=['ch_sim', 'en'])
    result = pipeline.audit("../pictures/text.png")
    print(f"决策: {result['decision']}")
    print(f"原因: {result['reason']}")
    print(f"阶段: {result['stage']}")