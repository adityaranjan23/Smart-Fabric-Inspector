from __future__ import annotations

import tempfile
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from smartfabric.config import DEFAULT_MODEL_PATH, OUTPUT_DIR, SAMPLE_DIR
from smartfabric.detector import Detection, SmartFabricDetector
from smartfabric.reporting import csv_bytes, detections_to_frame, pdf_bytes, summarize
from smartfabric.visuals import summary_chart


st.set_page_config(page_title="SmartFabric Inspector", page_icon="SF", layout="wide")


def inject_theme(mode: str) -> None:
    dark = mode == "Dark"
    bg = "#08111F" if dark else "#F5F7FA"
    panel = "#0F1B2D" if dark else "#FFFFFF"
    panel_alt = "#13233A" if dark else "#EEF6F6"
    text = "#E6EDF5" if dark else "#102A43"
    muted = "#91A4B7" if dark else "#5F6F7E"
    border = "#26364B" if dark else "#D8E2EA"
    accent = "#0EA5A5"
    accent_2 = "#2563EB"
    shadow = "0 18px 60px rgba(0, 0, 0, 0.28)" if dark else "0 18px 45px rgba(16, 42, 67, 0.10)"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg};
            color: {text};
        }}
        .block-container {{
            padding-top: 1.3rem;
            max-width: 1440px;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{ color: {text}; letter-spacing: 0; }}
        section[data-testid="stSidebar"] > div {{
            background: {panel};
            border: 1px solid {border};
        }}
        section[data-testid="stSidebar"] {{
            border-right: 1px solid {border};
        }}
        div[data-testid="stExpander"] {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 8px;
        }}
        button[kind="primary"], .stDownloadButton button, .stButton button {{
            border-radius: 8px;
            border: 1px solid {accent};
            background: {accent};
            color: white;
            font-weight: 650;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.45rem;
            border-bottom: 1px solid {border};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 2.75rem;
            border-radius: 8px 8px 0 0;
            padding: 0 1rem;
            background: {panel_alt};
            border: 1px solid {border};
            border-bottom: 0;
        }}
        .stTabs [aria-selected="true"] {{
            background: {panel};
            border-top: 3px solid {accent};
        }}
        [data-testid="stFileUploader"] section {{
            border: 1.5px dashed {accent};
            border-radius: 8px;
            background: {panel_alt};
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid {border};
            border-radius: 8px;
            overflow: hidden;
        }}
        .sf-muted {{ color: {muted}; }}
        .sf-header {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items: end;
            padding: 1.35rem 1.4rem;
            background: {panel};
            border: 1px solid {border};
            border-radius: 8px;
            box-shadow: {shadow};
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }}
        .sf-header:before {{
            content: "";
            position: absolute;
            inset: 0;
            border-left: 5px solid {accent};
            pointer-events: none;
        }}
        .sf-kicker {{
            color: {accent};
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }}
        .sf-header h1 {{
            margin: 0;
            font-size: 2.35rem;
            line-height: 1.05;
        }}
        .sf-header p {{
            margin: 0.45rem 0 0;
            max-width: 780px;
            color: {muted};
        }}
        .sf-header-actions {{
            display: flex;
            gap: 0.55rem;
            justify-content: flex-end;
            flex-wrap: wrap;
        }}
        .sf-status {{
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
            background: {"#12372F" if dark else "#DFF8EF"};
            color: {"#A7F3D0" if dark else "#057A55"};
            border: 1px solid {"#1F6F5B" if dark else "#9AE6B4"};
        }}
        .sf-status-warn {{
            background: {"#392B12" if dark else "#FFF3CD"};
            color: {"#FCD34D" if dark else "#8A5A00"};
            border-color: {"#7C5B1C" if dark else "#F6D365"};
        }}
        .sf-chip {{
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            border: 1px solid {border};
            color: {muted};
            background: {panel_alt};
            font-size: 0.84rem;
            font-weight: 700;
        }}
        .sf-panel {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 1rem;
            margin: 0.6rem 0 1rem;
        }}
        .sf-section-title {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            margin: 1.1rem 0 0.55rem;
        }}
        .sf-section-title h3 {{
            margin: 0;
            font-size: 1.05rem;
        }}
        .sf-section-title span {{
            color: {muted};
            font-size: 0.86rem;
        }}
        .sf-metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.8rem 0 1rem;
        }}
        .sf-metric {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 0.95rem;
            min-height: 104px;
            box-shadow: {"none" if dark else "0 10px 28px rgba(16, 42, 67, 0.06)"};
        }}
        .sf-metric-label {{
            color: {muted};
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .sf-metric-value {{
            color: {text};
            font-size: 1.75rem;
            line-height: 1.1;
            font-weight: 850;
            margin-top: 0.35rem;
        }}
        .sf-metric-note {{
            color: {muted};
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }}
        .sf-empty {{
            border: 1px dashed {border};
            background: {panel_alt};
            border-radius: 8px;
            padding: 1.4rem;
            color: {muted};
            margin-top: 0.8rem;
        }}
        .sf-empty strong {{ color: {text}; }}
        @media (max-width: 860px) {{
            .sf-header {{ grid-template-columns: 1fr; }}
            .sf-header h1 {{ font-size: 1.8rem; }}
            .sf-header-actions {{ justify-content: flex-start; }}
            .sf-metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        @media (max-width: 520px) {{
            .sf-metric-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_detector(model_path: str, task: str) -> SmartFabricDetector:
    resolved = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
    return SmartFabricDetector(resolved, task=task)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    pil_image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_report(df: pd.DataFrame, key_prefix: str) -> None:
    summary = summarize(df)
    metrics_grid(
        [
            ("Total defects", summary["total_defects"], "Objects above threshold"),
            ("Highest risk", summary["highest_risk"], "Severity rule summary"),
            ("Mean confidence", summary["mean_confidence"], "Average model confidence"),
            ("Defect types", len(summary["types"]), "Unique classes detected"),
        ]
    )

    chart = summary_chart(df)
    if chart:
        section_header("Inspection analytics", "Distribution by class and severity")
        st.image(chart, use_container_width=True)

    section_header("Detection log", "Downloadable quality-control evidence")
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download CSV",
        csv_bytes(df),
        "smartfabric_report.csv",
        "text/csv",
        use_container_width=True,
        key=f"{key_prefix}_download_csv",
    )
    c2.download_button(
        "Download PDF",
        pdf_bytes(df),
        "smartfabric_report.pdf",
        "application/pdf",
        use_container_width=True,
        key=f"{key_prefix}_download_pdf",
    )


def metrics_grid(items: list[tuple[str, object, str]]) -> None:
    cards = "\n".join(
        f"""
        <div class="sf-metric">
          <div class="sf-metric-label">{escape(str(label))}</div>
          <div class="sf-metric-value">{escape(str(value))}</div>
          <div class="sf-metric-note">{escape(str(note))}</div>
        </div>
        """
        for label, value, note in items
    )
    st.markdown(f'<div class="sf-metric-grid">{cards}</div>', unsafe_allow_html=True)


def section_header(title: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="sf-section-title">
          <h3>{escape(title)}</h3>
          <span>{escape(note)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="sf-empty">
          <strong>{escape(title)}</strong><br>
          {escape(body)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_image(
    detector: SmartFabricDetector,
    image: np.ndarray,
    confidence: float,
    iou: float,
    imgsz: int,
    max_det: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    annotated, detections = detector.predict(image, confidence=confidence, iou=iou, imgsz=imgsz, max_det=max_det)
    return annotated, detections_to_frame(detections)


def process_video(
    detector: SmartFabricDetector,
    video_path: Path,
    confidence: float,
    iou: float,
    frame_stride: int,
    imgsz: int,
    max_det: int,
) -> tuple[Path, pd.DataFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = OUTPUT_DIR / "annotated_video.mp4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_detections: list[Detection] = []
    frame_id = 0
    last_annotated: np.ndarray | None = None
    progress = st.progress(0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % frame_stride == 0:
            annotated, detections = detector.predict(frame, confidence=confidence, iou=iou, imgsz=imgsz, max_det=max_det)
            for detection in detections:
                detection.frame_id = frame_id
            all_detections.extend(detections)
            last_annotated = annotated
        writer.write(last_annotated if last_annotated is not None else frame)
        frame_id += 1
        if frame_id % 10 == 0:
            progress.progress(min(frame_id / total, 1.0))

    progress.empty()
    cap.release()
    writer.release()
    return output_path, detections_to_frame(all_detections)


def sample_images() -> list[Path]:
    return sorted([*SAMPLE_DIR.glob("*.jpg"), *SAMPLE_DIR.glob("*.png"), *SAMPLE_DIR.glob("*.jpeg")])


mode = st.sidebar.radio("Appearance", ["Light", "Dark"], horizontal=True)
inject_theme(mode)

st.markdown(
    """
    <div class="sf-header">
      <div>
        <div class="sf-kicker">Textile quality console</div>
        <h1>SmartFabric Inspector</h1>
        <p>Real-time fabric defect detection with YOLO inference, visual evidence, and export-ready inspection reports.</p>
      </div>
      <div class="sf-header-actions">
        <span class="sf-chip">YOLO ready</span>
        <span class="sf-chip">CPU/GPU auto</span>
        <span class="sf-chip">CSV/PDF reports</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Inspection settings")
    model_path = st.text_input("Model checkpoint", value=str(DEFAULT_MODEL_PATH))
    task = st.selectbox("Model task", ["detect", "segment"], help="Use segment when loading a YOLO segmentation checkpoint.")
    preset = st.segmented_control("Speed preset", ["Fast", "Balanced", "Quality"], default="Fast")
    preset_imgsz = {"Fast": 320, "Balanced": 416, "Quality": 640}[preset]
    preset_stride = {"Fast": 6, "Balanced": 3, "Quality": 1}[preset]
    confidence = st.slider(
        "Confidence threshold",
        0.01,
        0.95,
        0.05,
        0.01,
        help="Lower values are useful for early or lightly trained models; raise this after production training.",
    )
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    inference_size = st.select_slider("Inference size", options=[256, 320, 416, 512, 640], value=preset_imgsz)
    frame_stride = st.slider("Video frame stride", 1, 12, preset_stride)
    max_det = st.slider("Max detections per frame", 10, 300, 50, 10)

detector = load_detector(model_path, task)
status = f"Model loaded on {detector.device}" if detector.is_loaded else "Demo mode: model checkpoint not found"
status_class = "sf-status" if detector.is_loaded else "sf-status sf-status-warn"
st.markdown(f"<span class='{status_class}'>{escape(status)}</span>", unsafe_allow_html=True)

tab_live, tab_upload, tab_demo = st.tabs(["Live webcam", "Upload inspection", "Demo samples"])

with tab_live:
    section_header("Live inspection", "Browser capture for quick checks, workstation webcam for short real-time runs")
    camera_file = st.camera_input("Capture fabric image")
    if camera_file:
        image = read_uploaded_image(camera_file)
        annotated, df = process_image(detector, image, confidence, iou, inference_size, max_det)
        st.image(bgr_to_rgb(annotated), caption="Inspection overlay", use_container_width=True)
        render_report(df, "live_capture")
    else:
        empty_state("Waiting for a fabric frame", "Capture from the browser camera to generate the first inspection overlay and report.")

    with st.expander("Local real-time webcam"):
        seconds = st.slider("Run duration (seconds)", 3, 60, 10)
        if st.button("Start local webcam", use_container_width=True):
            placeholder = st.empty()
            table_placeholder = st.empty()
            cap = cv2.VideoCapture(0)
            all_detections: list[Detection] = []
            frame_id = 0
            if not cap.isOpened():
                st.error("Could not open local webcam device 0.")
            else:
                import time

                deadline = time.time() + seconds
                while time.time() < deadline:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    annotated, detections = detector.predict(
                        frame,
                        confidence=confidence,
                        iou=iou,
                        imgsz=inference_size,
                        max_det=max_det,
                    )
                    for detection in detections:
                        detection.frame_id = frame_id
                    all_detections.extend(detections)
                    placeholder.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)
                    frame_id += 1
                cap.release()
                table_placeholder.dataframe(detections_to_frame(all_detections), use_container_width=True, hide_index=True)

with tab_upload:
    section_header("Batch inspection", "Inspect a single image or process a video stream")
    uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])
    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            image = read_uploaded_image(uploaded)
            annotated, df = process_image(detector, image, confidence, iou, inference_size, max_det)
            c1, c2 = st.columns(2)
            c1.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
            c2.image(bgr_to_rgb(annotated), caption="Detected defects", use_container_width=True)
            render_report(df, "upload_image")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = Path(tmp.name)
            with st.spinner("Processing video..."):
                out_path, df = process_video(detector, tmp_path, confidence, iou, frame_stride, inference_size, max_det)
            st.video(str(out_path))
            render_report(df, "upload_video")
    else:
        empty_state("Drop an inspection asset", "Upload a fabric image or production-line video to review detections and export a report.")

with tab_demo:
    section_header("Demo samples", "Use extracted dataset images while the trained checkpoint is unavailable")
    samples = sample_images()
    if not samples:
        empty_state("No sample images found", "Run scripts/prepare_dataset.py to extract demo samples from the attached dataset.")
    else:
        selected = st.selectbox("Sample fabric image", samples, format_func=lambda path: path.name)
        image = cv2.imread(str(selected))
        annotated, df = process_image(detector, image, confidence, iou, inference_size, max_det)
        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(image), caption="Sample", use_container_width=True)
        c2.image(bgr_to_rgb(annotated), caption="Inspection overlay", use_container_width=True)
        render_report(df, "demo_sample")
