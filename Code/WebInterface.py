import streamlit as st
import tempfile
import time
import json
import os
from datetime import datetime
from PIL import Image
import numpy as np

from CNNDetector import YOLODetector, ImgClassifier
from VLMAnalyzer import VLM_Dec


class WebInterface:

    def __init__(self):

        self.yolo = YOLODetector("model/first-gambling.pt")
        self.classifier = ImgClassifier("model/classify-new.pth")
        self.vlm = VLM_Dec("qwen3-VL:2b", "LLMprompt.txt")

        os.makedirs("output", exist_ok=True)

        self.result_file = "output/result.txt"

        if "counter" not in st.session_state:
            st.session_state.counter = 0

    # ==========================
    # 保存结果
    # ==========================
    def save_result(self, line):

        with open(self.result_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ==========================
    # 读取历史结果
    # ==========================
    def load_results(self):

        if not os.path.exists(self.result_file):
            return []

        with open(self.result_file, "r", encoding="utf-8") as f:
            return f.readlines()

    # ==========================
    # UI
    # ==========================
    def render(self):

        st.set_page_config(layout="wide")

        st.title("违规图像检测系统")

        left, middle, right = st.columns([2, 5, 4])

        # ======================
        # 左 控制区
        # ======================
        with left:

            st.subheader("控制区")

            uploaded_file = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png"]
            )

            mode = st.radio(
                "检测模式",
                options=[
                    "CNN初筛",
                    "VLM检测",
                    "级联检测"
                ]
            )

            st.write("")

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                start = st.button("开始检测")

            error_box = st.empty()

        # ======================
        # 中 图片预览
        # ======================
        with middle:

            st.subheader("图片预览")

            image_box = st.empty()

        # ======================
        # 右 结果
        # ======================
        with right:

            st.subheader("检测结果")

            progress_box = st.empty()
            time_box = st.empty()
            result_box = st.empty()

            history = self.load_results()

            if history:
                result_box.text("".join(history))

        # ======================
        # 处理图片
        # ======================
        image_bytes = None
        image_path = None

        if uploaded_file is not None:

            image_bytes = uploaded_file.getvalue()

            img = Image.open(uploaded_file)

            image_box.image(
                img,
                use_container_width=True
            )

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(image_bytes)
            image_path = tmp.name
            tmp.close()

        # ======================
        # 点击检测
        # ======================
        if start:

            if image_path is None:

                error_box.error("请先上传图片")

                return

            try:

                start_time = time.time()

                start_time_str = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")

                progress = progress_box.progress(0)

                violation = False
                category = "无"
                confidence = 0
                reason = ""

                # ======================
                # CNN初筛
                # ======================
                if mode == "CNN初筛":

                    progress.progress(30)

                    yolo_result = self.yolo.detect(image_path)

                    progress.progress(100)

                    if yolo_result["category"]:

                        violation = True
                        category = yolo_result["category"]
                        confidence = yolo_result["confidence"]
                        reason = "YOLO检测到违规"

                    else:

                        violation = False
                        category = "无"
                        reason = "YOLO未检测到违规"

                # ======================
                # VLM检测
                # ======================
                elif mode == "VLM检测":

                    progress.progress(30)

                    vlm_raw = self.vlm.detect(image_path)

                    progress.progress(70)

                    vlm_json = json.loads(vlm_raw)

                    violation = vlm_json["violation"]
                    category = vlm_json["category"]
                    confidence = vlm_json["confidence"]
                    reason = vlm_json["reason"]

                    progress.progress(100)

                # ======================
                # 级联检测
                # ======================
                else:

                    progress.progress(20)

                    yolo_result = self.yolo.detect(image_path)

                    progress.progress(50)

                    if yolo_result["category"]:

                        vlm_raw = self.vlm.detect(image_path)

                        vlm_json = json.loads(vlm_raw)

                        violation = True
                        category = vlm_json["category"]
                        confidence = vlm_json["confidence"]
                        reason = vlm_json["reason"]

                    else:

                        violation = False
                        category = "无"
                        reason = "YOLO未检测到违规"

                    progress.progress(100)

                end_time = time.time()

                cost_time = round(end_time - start_time, 3)

                # ======================
                # 结果
                # ======================
                st.session_state.counter += 1

                result_line = (
                    f"{st.session_state.counter};"
                    f"{start_time_str};"
                    f"{violation};"
                    f"{mode};"
                    f"{cost_time}s;"
                    f"{category};"
                    f"{confidence};"
                    f"{reason}"
                )

                self.save_result(result_line)

                history = self.load_results()

                result_box.text("".join(history))

                time_box.info(f"检测耗时: {cost_time} 秒")

            except Exception as e:

                error_box.error(f"检测失败: {str(e)}")


if __name__ == "__main__":
    # streamlit run WebInterface.py
    app = WebInterface()

    app.render()