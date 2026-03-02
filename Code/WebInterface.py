import streamlit as st
import os
import tempfile
import json
import time

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
    # 初始化模型
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
    # 主界面
    # ==========================
    def render(self):

        st.set_page_config(
            page_title="图片检测系统",
            page_icon="🚀", 
            layout="wide",
            initial_sidebar_state="expanded",       # (可选) 侧边栏初始状态: "expanded", "collapsed", "auto"
            menu_items=None,                     # (可选) 自定义右上角菜单
            )
        # st.title("违规图像检测系统")

        # 页面三分区
        left_col, mid_col, right_col = st.columns([1, 2, 2])

        # ==========================
        # 左侧控制区
        # ==========================
        with left_col:

            st.subheader("检测控制")

            mode = st.radio(
                "选择检测模式",
                [
                    "YOLO初筛",
                    "复杂度分类(CNN)",
                    "VLM检测",
                    "级联检测(YOLO→CNN→VLM)"
                ]
            )

            st.markdown("---")

            # 按钮居中
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                start_btn = st.button("开始检测", use_container_width=True)

            error_placeholder = st.empty()

        # ==========================
        # 中间图片展示区
        # ==========================
        with mid_col:

            # st.subheader("图片展示")

            uploaded_file = st.file_uploader(
                "上传图片",
                type=["jpg", "png", "jpeg"]
            )

            image_path = None

            if uploaded_file is not None:
                image_path = self.__save_temp_file(uploaded_file)
                st.image(image_path, use_column_width=True)

        # ==========================
        # 右侧结果展示区
        # ==========================
        with right_col:

            st.subheader("检测结果")

            result_placeholder = st.empty()
            metric_placeholder = st.empty()

        # ==========================
        # 按钮点击逻辑
        # ==========================
        if start_btn:

            # 错误检查
            if not image_path:
                error_placeholder.error("请先上传图片")
                return

            if self.yolo is None or self.classifier is None or self.vlm is None:
                error_placeholder.error("模型未初始化，请先调用 init_models()")
                return

            error_placeholder.empty()

            result, preprocess_time, stage_times, total_time = \
                self.run_detection(mode, image_path, result_placeholder)

            # 显示耗时
            with right_col:
                st.subheader("耗时统计")

                col1, col2 = st.columns(2)
                col1.metric("预处理耗时(s)", round(preprocess_time, 4))
                col2.metric("总耗时(s)", round(total_time, 4))

                st.markdown("#### 阶段耗时")
                for stage, t in stage_times.items():
                    st.write(f"{stage}: {round(t,4)} s")

    # ==========================
    # 核心检测流程
    # ==========================
    def run_detection(self, mode, image_path, result_placeholder):

        progress = result_placeholder.progress(0)
        total_start = time.time()

        # 预处理
        start = time.time()
        processed_path = self.preprocessor.process(image_path)
        preprocess_time = time.time() - start
        progress.progress(20)

        stage_times = {}
        result = None

        if mode == "YOLO初筛":

            start = time.time()
            result = json.loads(self.yolo.detect(processed_path))
            stage_times["YOLO"] = time.time() - start
            progress.progress(100)

        elif mode == "复杂度分类(CNN)":

            start = time.time()
            result = self.classifier.predict(processed_path)
            stage_times["CNN"] = time.time() - start
            progress.progress(100)

        elif mode == "VLM检测":

            start = time.time()
            result = self.vlm.detect(processed_path)
            stage_times["VLM"] = time.time() - start
            progress.progress(100)

        elif mode == "级联检测(YOLO→CNN→VLM)":

            # YOLO
            start = time.time()
            yolo_result = json.loads(self.yolo.detect(processed_path))
            stage_times["YOLO"] = time.time() - start
            progress.progress(40)

            if yolo_result["category"] is not None:
                result = {"stage": "YOLO", "result": yolo_result}

            else:
                # CNN
                start = time.time()
                cnn_result = self.classifier.predict(processed_path)
                stage_times["CNN"] = time.time() - start
                progress.progress(70)

                if cnn_result["category"] == "simple":
                    result = {"stage": "CNN", "result": cnn_result}
                else:
                    # VLM
                    start = time.time()
                    vlm_result = self.vlm.detect(processed_path)
                    stage_times["VLM"] = time.time() - start
                    progress.progress(100)

                    result = {"stage": "VLM", "result": vlm_result}

        total_time = time.time() - total_start

        result_placeholder.json(result)

        return result, preprocess_time, stage_times, total_time

    # ==========================
    # 临时文件保存
    # ==========================
    def __save_temp_file(self, uploaded_file):
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