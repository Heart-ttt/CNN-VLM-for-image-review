# webface.py
import streamlit as st
import os
import time
import shutil
import requests
import json
from PIL import Image
from io import BytesIO

from CNNDetector import CNN_Dec
from VLMAnalyzer import VLM_Dec
from preprocess import Preprocess


class WebFace:

    def __init__(self):
        self.cnn = CNN_Dec("yolo11n.pt")
        self.vlm = VLM_Dec("qwen3-VL:2b", "LLMprompt.txt")
        self.preprocess = Preprocess()

        # 初始化输出目录
        self.base_output = "Output"
        self.cnn_output = os.path.join(self.base_output, "CNNresult")
        self.vlm_output = os.path.join(self.base_output, "VLMresult")
        self.log_file = os.path.join(self.base_output, "detect_log.txt")

        self.__init_output_dirs()

    def __init_output_dirs(self):
        if not os.path.exists(self.base_output):
            os.makedirs(self.base_output)

        if not os.path.exists(self.cnn_output):
            os.makedirs(self.cnn_output)

        if not os.path.exists(self.vlm_output):
            os.makedirs(self.vlm_output)

    def write_log(self, content):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(content + "\n")

    def save_cnn_result(self, image_path, category):
        category_dir = os.path.join(self.cnn_output, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        shutil.copy(image_path, category_dir)

    def save_vlm_result(self, image_path):
        shutil.copy(image_path, self.vlm_output)

    def load_url_image(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        temp_path = "temp_url.jpg"
        img.save(temp_path)
        return temp_path

    def process_image(self, image_path, mode):
        start_time = time.time()

        # 预处理
        processed_path = self.preprocess.process(image_path)

        result_data = {}

        # 单 VLM 模式
        if mode == "单VLM处理":
            vlm_result = self.vlm.detect(processed_path)
            self.save_vlm_result(processed_path)

            result_data["mode"] = "VLM"
            result_data["vlm_result"] = vlm_result

            log_content = f"[VLM] {image_path} -> {vlm_result}"

        # CNN + VLM 模式
        else:
            cnn_result = json.loads(self.cnn.detect(processed_path))
            category = cnn_result["category"]

            if category is not None:
                # CNN识别成功
                self.save_cnn_result(processed_path, category)

                result_data["mode"] = "CNN"
                result_data["cnn_result"] = cnn_result

                log_content = f"[CNN] {image_path} -> {cnn_result}"

            else:
                # 继续VLM检测
                vlm_result = self.vlm.detect(processed_path)
                self.save_vlm_result(processed_path)

                result_data["mode"] = "VLM(After CNN Fail)"
                result_data["vlm_result"] = vlm_result

                log_content = f"[CNN->VLM] {image_path} -> {vlm_result}"

        end_time = time.time()
        result_data["time_cost"] = round(end_time - start_time, 3)

        # 写入日志
        self.write_log(log_content)

        return result_data

    def process_folder(self, folder_path, mode):
        results = {}

        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                full_path = os.path.join(folder_path, file)
                results[file] = self.process_image(full_path, mode)

        return results

    def run(self):
        st.title("🚨 VLM + YOLO 违规图片检测系统")

        mode = st.radio(
            "选择检测模式",
            ["单VLM处理", "CNN + VLM处理"]
        )

        input_type = st.radio(
            "选择输入方式",
            ["单张图片上传", "文件夹路径", "URL图片"]
        )

        # 单张图片
        if input_type == "单张图片上传":
            uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                with open("temp_upload.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.image(uploaded_file, caption="上传图片")

                if st.button("开始检测"):
                    result = self.process_image("temp_upload.jpg", mode)
                    st.json(result)

        # 文件夹
        elif input_type == "文件夹路径":
            folder_path = st.text_input("输入文件夹路径")

            if st.button("开始检测"):
                result = self.process_folder(folder_path, mode)
                st.json(result)

        # URL
        elif input_type == "URL图片":
            url = st.text_input("输入图片URL")

            if st.button("开始检测"):
                image_path = self.load_url_image(url)
                st.image(image_path, caption="URL图片")
                result = self.process_image(image_path, mode)
                st.json(result)


if __name__ == "__main__":
    # streamlit run WebInterface.py
    app = WebFace()
    app.run()