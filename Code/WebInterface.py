import streamlit as st
import os
import tempfile
import json

from CNNDetector import YOLO_Dec, Classifier
from Preprocess import preprocess
from VLMAnalyzer import VLM_Dec


class WebFace:
    """
    Streamlit 界面类
    """

    def __init__(self):
        self.yolo = None
        self.classifier = None
        self.vlm = None
        self.preprocessor = preprocess()

    # ==========================
    # 初始化模型接口（供 main 调用）
    # ==========================
    def init_models(
        self,
        yolo_model_path: str,
        classifier_weight_path: str,
        vlm_model_name: str,
        prompt_path: str = "LLMprompt.txt",
    ):
        self.yolo = YOLO_Dec(yolo_model_path)
        self.classifier = Classifier(classifier_weight_path)
        self.vlm = VLM_Dec(vlm_model_name, prompt_path)

    # ==========================
    # 主界面渲染
    # ==========================
    def render(self):
        st.title("违规图像检测系统")

        st.sidebar.header("检测模式")
        mode = st.sidebar.radio(
            "选择检测流程",
            [
                "YOLO初筛",
                "复杂度分类(CNN)",
                "VLM检测",
                "级联检测(YOLO→CNN→VLM)"
            ]
        )

        uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image_path = self.__save_temp_file(uploaded_file)

            st.image(image_path, caption="上传图片", use_column_width=True)

            if st.button("开始检测"):
                result = self.handle_detection(mode, image_path)
                st.subheader("检测结果")
                st.write(result)

    # ==========================
    # 检测调度接口
    # ==========================
    def handle_detection(self, mode: str, image_path: str):
        """
        根据模式调用不同检测流程
        """

        # 预处理
        processed_path = self.preprocessor.process(image_path)

        if mode == "YOLO初筛":
            return self.yolo.detect(processed_path)

        elif mode == "复杂度分类(CNN)":
            return self.classifier.predict(processed_path)

        elif mode == "VLM检测":
            return self.vlm.detect(processed_path)

        elif mode == "级联检测(YOLO→CNN→VLM)":

            # 1️⃣ YOLO检测
            yolo_result = json.loads(self.yolo.detect(processed_path))
            if yolo_result["category"] is not None:
                return {
                    "stage": "YOLO",
                    "result": yolo_result
                }

            # 2️⃣ 复杂度分类
            cnn_result = self.classifier.predict(processed_path)
            if cnn_result["category"] == "simple":
                return {
                    "stage": "CNN",
                    "result": cnn_result
                }

            # 3️⃣ VLM检测
            vlm_result = self.vlm.detect(processed_path)

            return {
                "stage": "VLM",
                "result": vlm_result
            }

    # ==========================
    # 临时文件保存
    # ==========================
    def __save_temp_file(self, uploaded_file):
        """
        将上传文件保存为临时文件
        """
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            return tmp.name

if __name__ == "__main__":
    # streamlit run WebInterface.py
    app = WebFace()
    app.init_models(
    yolo_model_path="yolo11n.pt",
    classifier_weight_path="best_mobilenetv4.pth",
    vlm_model_name="qwen3-VL:2b",
    prompt_path="LLMprompt.txt"
    )

    app.render()