import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from CNNDetector import YOLODetector, ImgClassifier
from VLMAnalyzer import VLM_Dec

try:
    import torch
except Exception:
    torch = None


class WebInterface:
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def __init__(self):
        self.yolo = YOLODetector("model/first-gambling.pt")
        self.classifier = ImgClassifier("model/classify-new.pth")
        self.vlm = VLM_Dec("qwen3-VL:2b", "LLMprompt.txt")

        os.makedirs("output", exist_ok=True)
        self.result_file = "output/result.jsonl"

        if "counter" not in st.session_state:
            st.session_state.counter = len(self.load_results())

        if "latest_payload" not in st.session_state:
            st.session_state.latest_payload = None

        if "latest_operation_meta" not in st.session_state:
            st.session_state.latest_operation_meta = None

    def save_result(self, result: Dict[str, Any]) -> None:
        with open(self.result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def load_results(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.result_file):
            return []

        results: List[Dict[str, Any]] = []
        with open(self.result_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return results

    def _inject_styles(self) -> None:
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(180deg, #f5f7fb 0%, #eef4ff 100%);
            }
            .block-container {
                padding-top: 2.75rem;
                padding-bottom: 1.5rem;
            }
            .hero {
                padding: 1.5rem 1.7rem;
                border-radius: 24px;
                background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
                color: white;
                box-shadow: 0 20px 40px rgba(15, 23, 42, 0.16);
                margin-top: 0.35rem;
                margin-bottom: 1.1rem;
            }
            .hero h1 {
                margin: 0;
                font-size: 2rem;
                font-weight: 800;
                letter-spacing: 0.01em;
                line-height: 1.35;
            }
            .section-title {
                font-size: 1.05rem;
                font-weight: 800;
                color: #0f172a;
                margin: 0.1rem 0 0.85rem 0;
            }
            .subtle-text {
                color: #64748b;
                font-size: 0.92rem;
                margin-bottom: 0.9rem;
            }
            .section-note {
                padding: 0.85rem 0.95rem;
                border-radius: 14px;
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                color: #475569;
                margin-bottom: 0.8rem;
            }
            .result-card {
                padding: 1rem 1.05rem;
                border-radius: 18px;
                color: #111827;
                margin-top: 0.5rem;
                margin-bottom: 0.9rem;
            }
            .result-block {
                background: #fff1f2;
                border: 1px solid #fecdd3;
            }
            .result-pass {
                background: #f0fdf4;
                border: 1px solid #bbf7d0;
            }
            .result-title {
                font-size: 1.1rem;
                font-weight: 800;
                margin-bottom: 0.35rem;
            }
            .result-sub {
                color: #475569;
                font-size: 0.95rem;
                line-height: 1.6;
            }
            .operation-card {
                background: #eaf2ff;
                border: 1px solid #c7dbff;
                border-radius: 18px;
                padding: 1rem 1.05rem;
                margin-bottom: 1rem;
            }
            .operation-line {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: center;
                margin-bottom: 0.55rem;
                flex-wrap: wrap;
            }
            .operation-line:last-child {
                margin-bottom: 0.85rem;
            }
            .operation-label {
                font-size: 0.92rem;
                color: #475569;
            }
            .operation-value {
                font-size: 1rem;
                font-weight: 800;
                color: #0f172a;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
            }
            .metric-item {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 14px;
                padding: 0.75rem 0.85rem;
            }
            .metric-label {
                font-size: 0.82rem;
                color: #64748b;
                margin-bottom: 0.15rem;
            }
            .metric-value {
                font-size: 1rem;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.4;
                word-break: break-word;
            }
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.8rem;
                margin: 0.7rem 0 1rem 0;
            }
            .summary-item {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 0.85rem 0.95rem;
            }
            .summary-label {
                font-size: 0.84rem;
                color: #64748b;
                margin-bottom: 0.2rem;
            }
            .summary-value {
                font-size: 1.1rem;
                font-weight: 800;
                color: #0f172a;
                word-break: break-word;
            }
            .history-note {
                color: #64748b;
                font-size: 0.85rem;
                margin-bottom: 0.45rem;
            }
            .stButton > button {
                border-radius: 14px;
                height: 3rem;
                font-weight: 700;
                border: 0;
                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.16);
            }
            .stDownloadButton > button {
                border-radius: 12px;
                font-weight: 700;
            }
            @media (max-width: 1200px) {
                .metric-grid,
                .summary-grid {
                    grid-template-columns: 1fr;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def _clean_json_text(self, raw: str) -> str:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            return match.group(0)

        return cleaned

    def _parse_vlm_json(self, raw: str) -> Dict[str, Any]:
        cleaned = self._clean_json_text(raw)

        try:
            data = json.loads(cleaned)
        except Exception:
            return {
                "violation": False,
                "category": None,
                "confidence": 0.0,
                "reason": "VLM返回非标准JSON",
                "raw": raw,
            }

        violation = data.get("violation")
        if violation is None and "blocked" in data:
            violation = bool(data.get("blocked"))
        if violation is None and "decision" in data:
            decision = str(data.get("decision", "")).upper()
            violation = decision in {"BLOCK", "REJECT", "VIOLATION", "INTERCEPT"}
        if violation is None:
            violation = False

        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        return {
            "violation": bool(violation),
            "category": data.get("category"),
            "confidence": round(confidence, 4),
            "reason": data.get("reason") or data.get("message") or "",
            "raw": raw,
        }

    def _make_result(
        self,
        file_name: str,
        mode: str,
        blocked: bool,
        reason: str,
        stage: str,
        elapsed_seconds: float,
        category: Optional[str] = None,
        confidence: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        st.session_state.counter += 1
        return {
            "id": st.session_state.counter,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": file_name,
            "mode": mode,
            "decision": "BLOCK" if blocked else "PASS",
            "blocked": blocked,
            "reason": reason,
            "stage": stage,
            "category": category,
            "confidence": round(float(confidence or 0.0), 4),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "details": details or {},
        }

    def _cuda_available(self) -> bool:
        return bool(torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available())

    def _reset_peak_gpu_memory(self) -> None:
        if not self._cuda_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _read_gpu_memory_mb(self) -> Tuple[Optional[float], Optional[float]]:
        if not self._cuda_available():
            return None, None
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            current = round(float(torch.cuda.memory_allocated()) / (1024 ** 2), 2)
            peak = round(float(torch.cuda.max_memory_allocated()) / (1024 ** 2), 2)
            return current, peak
        except Exception:
            return None, None

    def _run_cnn_detection(self, image_path: str, file_name: str) -> Dict[str, Any]:
        start_time = time.time()
        yolo_result = self.yolo.detect(image_path)
        elapsed_seconds = time.time() - start_time

        blocked = bool(yolo_result.get("category"))
        reason = "YOLO拦截" if blocked else "YOLO放行"

        return self._make_result(
            file_name=file_name,
            mode="CNN初筛",
            blocked=blocked,
            reason=reason,
            stage="YOLO",
            elapsed_seconds=elapsed_seconds,
            category=yolo_result.get("category"),
            confidence=yolo_result.get("confidence", 0.0),
            details={
                "yolo": yolo_result,
                "text_audit_pipeline_used": False,
            },
        )

    def _run_vlm_detection(self, image_path: str, file_name: str) -> Dict[str, Any]:
        start_time = time.time()
        vlm_raw = self.vlm.detect(image_path)
        vlm_result = self._parse_vlm_json(vlm_raw)
        elapsed_seconds = time.time() - start_time

        blocked = vlm_result["violation"]
        reason = "VLM拦截" if blocked else "VLM放行"

        return self._make_result(
            file_name=file_name,
            mode="VLM检测",
            blocked=blocked,
            reason=reason,
            stage="Qwen3-VL",
            elapsed_seconds=elapsed_seconds,
            category=vlm_result.get("category"),
            confidence=vlm_result.get("confidence", 0.0),
            details={
                "vlm_reason": vlm_result.get("reason"),
                "vlm_raw": vlm_result.get("raw"),
            },
        )

    def _run_cascade_detection(self, image_path: str, file_name: str) -> Dict[str, Any]:
        start_time = time.time()

        yolo_result = self.yolo.detect(image_path)
        if yolo_result.get("category"):
            elapsed_seconds = time.time() - start_time
            return self._make_result(
                file_name=file_name,
                mode="级联检测",
                blocked=True,
                reason="YOLO拦截",
                stage="YOLO",
                elapsed_seconds=elapsed_seconds,
                category=yolo_result.get("category"),
                confidence=yolo_result.get("confidence", 0.0),
                details={
                    "yolo": yolo_result,
                    "text_audit_pipeline_used": False,
                },
            )

        classifier_result = self.classifier.predict(image_path)
        if classifier_result.get("category") == "simple":
            elapsed_seconds = time.time() - start_time
            return self._make_result(
                file_name=file_name,
                mode="级联检测",
                blocked=False,
                reason="YOLO放行，MobileNet放行",
                stage="MobileNetV4",
                elapsed_seconds=elapsed_seconds,
                category=classifier_result.get("category"),
                confidence=classifier_result.get("confidence", 0.0),
                details={
                    "yolo": yolo_result,
                    "classifier": classifier_result,
                    "text_audit_pipeline_used": False,
                },
            )

        vlm_raw = self.vlm.detect(image_path)
        vlm_result = self._parse_vlm_json(vlm_raw)
        elapsed_seconds = time.time() - start_time
        blocked = vlm_result["violation"]
        reason = f"YOLO放行，VLM{'拦截' if blocked else '放行'}"

        return self._make_result(
            file_name=file_name,
            mode="级联检测",
            blocked=blocked,
            reason=reason,
            stage="Qwen3-VL",
            elapsed_seconds=elapsed_seconds,
            category=vlm_result.get("category") or classifier_result.get("category"),
            confidence=vlm_result.get("confidence", 0.0),
            details={
                "yolo": yolo_result,
                "classifier": classifier_result,
                "vlm_reason": vlm_result.get("reason"),
                "vlm_raw": vlm_result.get("raw"),
                "text_audit_pipeline_used": False,
            },
        )

    def _run_detection(self, image_path: str, file_name: str, mode: str) -> Dict[str, Any]:
        if mode == "CNN初筛":
            return self._run_cnn_detection(image_path, file_name)
        if mode == "VLM检测":
            return self._run_vlm_detection(image_path, file_name)
        return self._run_cascade_detection(image_path, file_name)

    def _scan_folder_images(self, folder_path: str, recursive: bool) -> List[Path]:
        folder = Path(folder_path).expanduser()
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"不是有效文件夹: {folder}")

        iterator = folder.rglob("*") if recursive else folder.glob("*")
        image_paths = [p for p in iterator if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES]
        image_paths.sort(key=lambda p: p.name.lower())
        return image_paths

    def _make_batch_payload(
        self,
        folder_path: str,
        mode: str,
        recursive: bool,
        results: List[Dict[str, Any]],
        elapsed_seconds: float,
    ) -> Dict[str, Any]:
        blocked_count = sum(1 for item in results if item.get("blocked"))
        passed_count = len(results) - blocked_count

        return {
            "type": "batch",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "folder_path": str(Path(folder_path).expanduser().resolve()),
            "recursive": recursive,
            "mode": mode,
            "total_files": len(results),
            "blocked_count": blocked_count,
            "passed_count": passed_count,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "results": results,
        }

    def _build_operation_meta(
        self,
        source_mode: str,
        mode: str,
        results: List[Dict[str, Any]],
        memory_samples_mb: Optional[List[float]] = None,
        peak_memory_mb: Optional[float] = None,
        f1_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        elapsed_samples = [float(item.get("elapsed_seconds", 0.0)) for item in results if item.get("elapsed_seconds") is not None]
        avg_elapsed = round(mean(elapsed_samples), 4) if elapsed_samples else None

        cleaned_memory = [float(item) for item in (memory_samples_mb or []) if item is not None]
        avg_memory = round(mean(cleaned_memory), 2) if cleaned_memory else None
        peak_memory = round(float(peak_memory_mb), 2) if peak_memory_mb is not None else None

        return {
            "source_mode": source_mode,
            "mode": mode,
            "f1_score": round(float(f1_score), 4) if f1_score is not None else None,
            "avg_inference_seconds": avg_elapsed,
            "avg_memory_mb": avg_memory,
            "peak_memory_mb": peak_memory,
        }

    def _format_metric_value(self, value: Optional[float], suffix: str = "") -> str:
        if value is None:
            return "N/A"
        return f"{value}{suffix}"

    def _render_operation_overview(self, operation_meta: Optional[Dict[str, Any]]) -> None:
        if operation_meta is None:
            operation_meta = {
                "source_mode": "-",
                "mode": "-",
                "f1_score": None,
                "avg_inference_seconds": None,
                "avg_memory_mb": None,
                "peak_memory_mb": None,
            }

        st.markdown(
            f"""
            <div class="operation-card">
                <div class="operation-line">
                    <div class="operation-label">检测入口</div>
                    <div class="operation-value">{operation_meta.get('source_mode', '-')}</div>
                </div>
                <div class="operation-line">
                    <div class="operation-label">检测模式</div>
                    <div class="operation-value">{operation_meta.get('mode', '-')}</div>
                </div>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">F1-score</div>
                        <div class="metric-value">{self._format_metric_value(operation_meta.get('f1_score'))}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">平均推理时间</div>
                        <div class="metric-value">{self._format_metric_value(operation_meta.get('avg_inference_seconds'), ' s')}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">平均 / 峰值显存占用</div>
                        <div class="metric-value">{self._format_metric_value(operation_meta.get('avg_memory_mb'), ' MB')} / {self._format_metric_value(operation_meta.get('peak_memory_mb'), ' MB')}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if operation_meta.get("f1_score") is None:
            st.caption(" ")

    def _render_single_result(self, result: Dict[str, Any]) -> None:
        card_class = "result-block" if result.get("blocked") else "result-pass"
        title = "已拦截" if result.get("blocked") else "已放行"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-title">{title}</div>
                <div class="result-sub">
                    文件：{result.get('file_name', '-')}<br>
                    原因：{result.get('reason', '-')}<br>
                    阶段：{result.get('stage', '-')} ｜ 类别：{result.get('category') or '无'} ｜ 耗时：{result.get('elapsed_seconds', 0)}s
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "下载本次 JSON 结果",
            data=json_bytes,
            file_name=f"result_{result.get('id', 'latest')}.json",
            mime="application/json",
            use_container_width=True,
        )

    def _render_batch_result(self, payload: Dict[str, Any]) -> None:
        st.markdown(
            """
            <div class="result-card result-pass">
                <div class="result-title">批量检测完成</div>
                <div class="result-sub">已按 JSON 聚合输出本次文件夹检测结果，可直接下载。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">检测模式</div>
                    <div class="summary-value">{payload.get('mode', '-')}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">图片总数</div>
                    <div class="summary-value">{payload.get('total_files', 0)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">拦截数量</div>
                    <div class="summary-value">{payload.get('blocked_count', 0)}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">放行数量</div>
                    <div class="summary-value">{payload.get('passed_count', 0)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"目录：{payload.get('folder_path', '-')}")
        st.caption(f"总耗时：{payload.get('elapsed_seconds', 0)}s")

        table_data = []
        for item in payload.get("results", []):
            table_data.append(
                {
                    "文件": item.get("file_name"),
                    "决策": item.get("decision"),
                    "原因": item.get("reason"),
                    "阶段": item.get("stage"),
                    "类别": item.get("category") or "无",
                    "耗时(s)": item.get("elapsed_seconds"),
                }
            )

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
            height=min(320, 42 * (len(table_data) + 1) + 6),
        )

        json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "下载批量 JSON 结果",
            data=json_bytes,
            file_name="batch_detect_result.json",
            mime="application/json",
            use_container_width=True,
        )

    def _render_latest_payload(self, payload: Optional[Dict[str, Any]]) -> None:
        if not payload:
            st.info("完成检测后，这里会展示最新结果，并支持下载 JSON。")
            return

        if payload.get("type") == "batch":
            self._render_batch_result(payload)
            return

        self._render_single_result(payload)

    def _render_history(self, history: List[Dict[str, Any]]) -> None:
        st.markdown("#### 历史记录")
        if not history:
            st.info("暂无检测记录")
            return

        table_data = []
        for item in reversed(history[-10:]):
            table_data.append(
                {
                    "ID": item.get("id"),
                    "时间": item.get("timestamp"),
                    "文件": item.get("file_name"),
                    "模式": item.get("mode"),
                    "决策": item.get("decision"),
                    "原因": item.get("reason"),
                    "耗时(s)": item.get("elapsed_seconds"),
                }
            )

        st.markdown('<div class="history-note">仅展示最近 10 条记录，表格高度已限制，避免超出当前页面显示区域。</div>', unsafe_allow_html=True)
        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
            height=min(320, 42 * (len(table_data) + 1) + 6),
        )

    def render(self) -> None:
        st.set_page_config(
            page_title="违规图像检测系统",
            page_icon="🛡️",
            layout="wide",
        )
        self._inject_styles()

        st.markdown(
            """
            <div class="hero">
                <h1>🛡️ 违规图像检测系统</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left_col, right_col = st.columns([1, 1.45], gap="large")

        with left_col:
            st.markdown('<div class="section-title">检测控制台</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="subtle-text">支持单图检测与文件夹批量检测，结果统一保存为 JSON / JSONL。</div>',
                unsafe_allow_html=True,
            )

            source_mode = st.radio(
                "检测入口",
                options=["单图检测", "文件夹批量"],
                horizontal=True,
            )

            mode = st.radio(
                "检测模式",
                options=["CNN初筛", "VLM检测", "级联检测"],
                horizontal=True,
                help="CNN初筛仅使用 YOLO；级联检测：YOLO -> MobileNetV4 -> VLM。",
            )

            uploaded_file = None
            folder_path = ""
            recursive = False

            if source_mode == "单图检测":
                uploaded_file = st.file_uploader(
                    "上传图片",
                    type=["jpg", "jpeg", "png"],
                    help="支持 JPG / JPEG / PNG",
                )
                if uploaded_file is not None:
                    file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 2)
                    st.caption(f"文件名：{uploaded_file.name}")
                    st.caption(f"文件大小：{file_size_kb} KB")
            else:
                st.markdown(
                    '<div class="section-note">输入服务器或本机可访问的文件夹路径，系统将批量检测其中所有 JPG / JPEG / PNG 图片。</div>',
                    unsafe_allow_html=True,
                )
                folder_path = st.text_input(
                    "文件夹路径",
                    placeholder="例如：D:/images 或 /data/images",
                ).strip()
                recursive = st.checkbox("包含子目录", value=False)

            start = st.button("开始检测", type="primary", use_container_width=True)
            error_box = st.empty()

        with right_col:
            st.markdown('<div class="section-title">检测结果</div>', unsafe_allow_html=True)
            status_box = st.empty()
            progress_box = st.empty()
            self._render_operation_overview(st.session_state.get("latest_operation_meta"))
            self._render_latest_payload(st.session_state.get("latest_payload"))
            self._render_history(self.load_results())

        if not start:
            return

        try:
            if source_mode == "单图检测":
                if uploaded_file is None:
                    error_box.error("请先上传图片")
                    return

                suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
                tmp_path = None
                memory_samples_mb: List[float] = []

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    self._reset_peak_gpu_memory()
                    progress = progress_box.progress(10)
                    status_box.info("模型检测中，请稍候...")
                    result = self._run_detection(tmp_path, uploaded_file.name, mode)
                    current_mem_mb, peak_mem_mb = self._read_gpu_memory_mb()
                    if current_mem_mb is not None:
                        memory_samples_mb.append(current_mem_mb)
                    elif peak_mem_mb is not None:
                        memory_samples_mb.append(peak_mem_mb)
                    progress.progress(100)

                    self.save_result(result)
                    st.session_state.latest_payload = result
                    st.session_state.latest_operation_meta = self._build_operation_meta(
                        source_mode=source_mode,
                        mode=mode,
                        results=[result],
                        memory_samples_mb=memory_samples_mb,
                        peak_memory_mb=peak_mem_mb,
                        f1_score=None,
                    )
                    status_box.success("检测完成")
                    st.rerun()
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            else:
                if not folder_path:
                    error_box.error("请先输入文件夹路径")
                    return

                image_paths = self._scan_folder_images(folder_path, recursive)
                if not image_paths:
                    error_box.error("该文件夹下未找到 JPG / JPEG / PNG 图片")
                    return

                batch_start = time.time()
                results: List[Dict[str, Any]] = []
                memory_samples_mb: List[float] = []
                peak_memory_samples_mb: List[float] = []
                progress = progress_box.progress(0)

                for index, image_path in enumerate(image_paths, start=1):
                    self._reset_peak_gpu_memory()
                    status_box.info(f"批量检测中 ({index}/{len(image_paths)}): {image_path.name}")
                    result = self._run_detection(str(image_path), image_path.name, mode)
                    self.save_result(result)
                    results.append(result)

                    current_mem_mb, peak_mem_mb = self._read_gpu_memory_mb()
                    if current_mem_mb is not None:
                        memory_samples_mb.append(current_mem_mb)
                    elif peak_mem_mb is not None:
                        memory_samples_mb.append(peak_mem_mb)
                    if peak_mem_mb is not None:
                        peak_memory_samples_mb.append(peak_mem_mb)

                    progress.progress(int(index / len(image_paths) * 100))

                payload = self._make_batch_payload(
                    folder_path=folder_path,
                    mode=mode,
                    recursive=recursive,
                    results=results,
                    elapsed_seconds=time.time() - batch_start,
                )
                st.session_state.latest_payload = payload
                st.session_state.latest_operation_meta = self._build_operation_meta(
                    source_mode=source_mode,
                    mode=mode,
                    results=results,
                    memory_samples_mb=memory_samples_mb,
                    peak_memory_mb=max(peak_memory_samples_mb) if peak_memory_samples_mb else None,
                    f1_score=None,
                )
                status_box.success(f"批量检测完成，共处理 {len(results)} 张图片")
                st.rerun()

        except Exception as e:
            error_box.error(f"检测失败: {str(e)}")


if __name__ == "__main__":
    app = WebInterface()
    app.render()
