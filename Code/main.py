import io
import base64
from typing import List, Optional
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import ollama

# 阈值配置
CNN_HIGH_THRESH = 0.85
VLM_GIGH_THRESH = 0.85
VLM_LOW_THRESH = 0.30

# 模型路径
CNN_MODEL_PATH = "yolov10n.pt"  
VLM_MODEL_NAME = "qwen3-vl:2b" 

# 违规提示词
VLM_PROMPT = """
你是一名校园网络安全审计员。请分析这张图片是否包含以下违规内容：
1. 色情低俗 ( nudity, sexual acts)
2. 暴力血腥 ( violence, blood, weapons)
3. 政治敏感 ( sensitive political symbols, protests)
4. 赌博毒品 ( gambling, drugs)

如果包含上述任何一项，请回答 "VIOLATION: [具体原因]"。
如果图片安全，请回答 "SAFE"。
如果图片模糊或无法确定，请回答 "UNCERTAIN"。
只输出结论，不要废话。
"""

# 初始化模型web
# app = FastAPI(title="Campus Net Image Auditor")

# 加载CNN模型 (启动时加载一次)
try:
    cnn_model = YOLO(CNN_MODEL_PATH)
    print(f"✅ CNN Model {CNN_MODEL_PATH} loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load CNN model: {e}")
    cnn_model = None
    
def run_cnn_inference(img: Image.Image) -> tuple[float, str]:
    """
    运行CNN识别
    返回: (max_confidence, label)
    """
    if not cnn_model:
        raise HTTPException(status_code=500, detail="CNN Model not loaded")
    
    # YOLO推理
    results = cnn_model(img, verbose=False)
    result = results[0]
    
    max_conf = 0.0
    detected_label = "None"
    
    # 遍历所有检测结果，取置信度最高的违规类
    # 假设你的模型训练了特定的违规类别 (如: 'porn', 'violence')
    # 如果没有特定训练，这里可以使用预训练的通用类别作为演示，实际毕设需训练自定义数据集
    for box in result.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label_name = result.names[cls_id]
        
        if conf > max_conf:
            max_conf = conf
            detected_label = label_name
            
    return max_conf, detected_label    

async def run_vlm_inference(img: Image.Image) -> dict:
    """
    运行VLM (Ollama) 进行二次研判
    """
    # 将图片转换为 base64 供 Ollama 使用
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    try:
        response = ollama.chat(
            model=VLM_MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': VLM_PROMPT,
                'images': [img_base64]
            }]
        )
        content = response['message']['content'].strip()
        
        # 解析 VLM 返回
        if "VIOLATION" in content.upper():
            return {"status": "VIOLATION", "reason": content, "source": "VLM"}
        elif "UNCERTAIN" in content.upper():
            return {"status": "REVIEW", "reason": content, "source": "VLM"}
        else:
            return {"status": "SAFE", "reason": content, "source": "VLM"}
            
    except Exception as e:
        # VLM 失败降级处理，建议转人工
        return {"status": "REVIEW", "reason": f"VLM Error: {str(e)}", "source": "SYSTEM_ERROR"}