# main.py

import streamlit as st
from WebInterface import DetectionUI

app = DetectionUI()

app.init_models(
    yolo_model_path="yolo11n.pt",
    classifier_weight_path="best_mobilenetv4.pth",
    vlm_model_name="qwen3-VL:2b",
    prompt_path="LLMprompt.txt"
)

app.render()