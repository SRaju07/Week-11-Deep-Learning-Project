import os
import shutil
import tempfile
from pathlib import Path

# Set YOLO configuration directory before importing Ultralytics
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

import time
from collections import Counter, deque

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO

# Limit PyTorch CPU thread allocation to prevent cloud quota throttling
torch.set_num_threads(1)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="RoadSense AI: Real-Time Road Object Perception & Collision Risk Warning System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
        
        header[data-testid="stHeader"] {
            display: none !important;
        }

        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }

        .nav-hud {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 22px !important;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            height: auto !important;
            overflow: visible !important;
        }
        .nav-title {
            font-size: 1.25rem !important;
            font-weight: 800 !important;
            color: #0f172a !important;
            letter-spacing: -0.01em;
            line-height: 1.5 !important;
            margin: 0 !important;
            padding: 0 !important;
            display: block !important;
        }
        .nav-subtitle {
            font-size: 0.78rem !important;
            color: #64748b !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            margin-top: 4px !important;
        }
        .core-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            background: #e0f2fe;
            color: #0369a1;
            border: 1px solid #bae6fd;
            white-space: nowrap;
        }

        .telemetry-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
            border-left: 4px solid #0284c7;
            margin-bottom: 8px;
            min-height: 85px;
        }
        .telemetry-label {
            font-size: 0.70rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
        }
        .telemetry-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.45rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 2px;
        }
        .telemetry-alert {
            border-left-color: #ef4444 !important;
            background: #fef2f2 !important;
        }
        .telemetry-alert .telemetry-value {
            color: #b91c1c !important;
        }
        .telemetry-ped {
            border-left-color: #f59e0b !important;
        }
        .telemetry-ped .telemetry-value {
            color: #b45309 !important;
        }

        .danger-banner {
            background: #fee2e2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 10px 14px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.88rem;
            text-align: center;
            margin-bottom: 12px;
            box-shadow: 0 2px 6px rgba(239, 68, 68, 0.08);
        }

        div.stButton > button {
            background-color: #ffffff;
            color: #ef4444;
            border: 1.5px solid #fca5a5;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.82rem;
            padding: 8px 18px;
            width: 100%;
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            background-color: #ef4444;
            color: #ffffff;
            border-color: #ef4444;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Dynamic Model Loading & Device Tuning
# ---------------------------------------------------------
CANDIDATE_PATHS = [
    r"D:\python1\Week-11-Deep-Learning-Project\bdd100k\runs\hazard_detector\weights\best.pt",
    "runs/hazard_detector/weights/best.pt",
    "weights/best.pt",
    "best.pt",
    "yolo26n.pt",
    "yolo11n.pt",
    "yolov8n.pt",
    "yolo11s.pt",
]

model_path = "yolo26n.pt"
for candidate in CANDIDATE_PATHS:
    if os.path.isfile(candidate):
        model_path = candidate
        break

@st.cache_resource
def load_yolo_engine(path: str):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        loaded = YOLO(path)
    except Exception:
        try:
            loaded = YOLO("yolo11n.pt")
        except Exception:
            loaded = YOLO("yolov8n.pt")
    return loaded, dev

model, device = load_yolo_engine(model_path)
track_history = {}

# Keep input dimensions modest for cloud CPU performance
INFER_SIZE = 480 if "cuda" in device else 320

# ---------------------------------------------------------
# Sidebar Controls & Sensitivity
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Detection Sensitivity")
    conf_thresh = st.slider("Confidence Cutoff", 0.05, 1.00, 0.35, 0.05)
    iou_thresh = st.slider("IoU NMS Overlap", 0.20, 0.90, 0.45, 0.05)
    enable_tracking = st.toggle("ByteTrack Kinematics", value=True)
    frame_skip = st.slider("Stream Frame Skip Rate", 1, 10, 4)

    st.markdown("---")
    st.subheader("⚠️ Hazard Distance Thresholds")
    area_high_risk = st.number_input("High-Risk Area Threshold (px²)", value=50000, step=5000)
    area_med_risk = st.number_input("Medium-Risk Area Threshold (px²)", value=15000, step=2500)
    dist_danger_h = st.number_input("Proximity Danger Height (px)", value=300, step=25)
    dist_caution_h = st.number_input("Proximity Caution Height (px)", value=150, step=25)

# Top Header Bar
st.markdown(
    f"""
    <div class="nav-hud">
        <div>
            <div class="nav-title">🚗 RoadSense AI: Real-Time Perception and Collision Warning System</div>
            <div class="nav-subtitle">Multi-Tier Collision Guard • Kinematic TTC Tracking • Ground Plane BEV</div>
        </div>
        <div class="core-badge">ENGINE: {device.upper()}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Bird's-Eye View (BEV) Ground Radar
# ---------------------------------------------------------
def render_light_bev(tracked_objects, frame_w, frame_h, radar_dim=(320, 320)):
    radar = np.full((radar_dim[1], radar_dim[0], 3), 250, dtype=np.uint8)
    cx, cy = radar_dim[0] // 2, radar_dim[1] - 32

    for r in [60, 120, 180, 240]:
        cv2.ellipse(radar, (cx, cy), (r, r), 0, 180, 360, (220, 228, 238), 1, cv2.LINE_AA)
        cv2.putText(radar, f"{r//6}m", (cx + 6, cy - r + 12), cv2.FONT_HERSHEY_PLAIN, 0.65, (140, 155, 175), 1)

    cv2.line(radar, (cx, cy), (cx - 125, cy - 250), (225, 230, 240), 1, cv2.LINE_AA)
    cv2.line(radar, (cx, cy), (cx + 125, cy - 250), (225, 230, 240), 1, cv2.LINE_AA)

    cv2.rectangle(radar, (cx - 10, cy - 18), (cx + 10, cy + 8), (2, 132, 199), -1)
    cv2.putText(radar, "EGO", (cx - 12, cy + 22), cv2.FONT_HERSHEY_PLAIN, 0.75, (2, 132, 199), 1)

    for obj in tracked_objects:
        norm_x = (obj['center_x'] - (frame_w / 2)) / (frame_w / 2)
        rel_dist = np.clip(1.0 - (obj['bbox_bottom'] / frame_h), 0.05, 1.0)

        rx = int(cx + (norm_x * 125))
        ry = int(cy - (rel_dist * 220))
        rx = np.clip(rx, 10, radar_dim[0] - 10)
        ry = np.clip(ry, 10, radar_dim[1] - 10)

        color = (30, 30, 220) if obj['is_hazard'] else (210, 130, 0)
        cv2.circle(radar, (rx, ry), 5, color, -1, cv2.LINE_AA)
        cv2.circle(radar, (rx, ry), 8, color, 1, cv2.LINE_AA)

        lbl = f"#{obj['id']}" if obj['id'] is not None else obj['cls'][:3]
        cv2.putText(radar, lbl, (rx + 8, ry + 3), cv2.FONT_HERSHEY_PLAIN, 0.65, (30, 41, 59), 1)

    return radar

# ---------------------------------------------------------
# HUD Reticles & Bounding Boxes
# ---------------------------------------------------------
def draw_hud_box(frame, x1, y1, x2, y2, color, label, is_ped=False):
    line_len = int(min(x2 - x1, y2 - y1) * 0.25)
    t = 3 if is_ped else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, t)
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, t)
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, t)
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, t)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.42, 1)
    tag_y1 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 8, y1), (255, 255, 255), -1)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 8, y1), color, 1)
    cv2.putText(frame, label, (x1 + 4, max(12, y1 - 4)), font, 0.42, (15, 23, 42), 1, cv2.LINE_AA)

# ---------------------------------------------------------
# Perception & Risk Logic
# ---------------------------------------------------------
def classify_risk(w, h, cls_name):
    area = w * h
    is_pedestrian = any(kw in cls_name.lower() for kw in {"person", "pedestrian", "rider"})

    effective_danger_h = dist_danger_h * 0.65 if is_pedestrian else dist_danger_h
    effective_caution_h = dist_caution_h * 0.65 if is_pedestrian else dist_caution_h

    if h >= effective_danger_h or area >= area_high_risk:
        proximity = "DANGER"
    elif h >= effective_caution_h or area >= area_med_risk:
        proximity = "CAUTION"
    else:
        proximity = "SAFE"

    if is_pedestrian:
        if proximity == "DANGER" or area >= (area_med_risk * 0.4):
            risk_level = "CRITICAL PEDESTRIAN"
            color = (0, 0, 220)
        else:
            risk_level = "PED ALERT"
            color = (0, 140, 240)
    else:
        if proximity == "DANGER" or area >= area_high_risk:
            risk_level = "HIGH RISK"
            color = (0, 0, 220)
        elif proximity == "CAUTION" or area >= area_med_risk:
            risk_level = "MED RISK"
            color = (0, 175, 240)
        else:
            risk_level = "LOW RISK"
            color = (0, 190, 80)

    return risk_level, proximity, color, is_pedestrian

def process_frame(frame, tracker=False):
    fh, fw = frame.shape[:2]
    detected_classes = []
    critical_threats = []
    ped_count = 0
    tracked_metadata = []

    with torch.no_grad():
        if tracker:
            results = model.track(
                frame,
                persist=True,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=INFER_SIZE,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
        else:
            results = model.predict(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=INFER_SIZE,
                verbose=False,
            )[0]

    ts = time.time()
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = max(0, x2 - x1), max(0, y2 - y1)
            cls_id = int(box.cls[0])
            cls_name = results.names.get(cls_id, f"cls_{cls_id}").capitalize()
            track_id = int(box.id[0]) if (box.id is not None) else None

            risk, proximity, color, is_ped = classify_risk(w, h, cls_name)

            ttc = None
            if track_id is not None:
                if track_id not in track_history:
                    track_history[track_id] = deque(maxlen=5)
                track_history[track_id].append((y2, ts))

                if len(track_history[track_id]) >= 3:
                    dy = track_history[track_id][-1][0] - track_history[track_id][0][0]
                    dt = track_history[track_id][-1][1] - track_history[track_id][0][1]
                    if dt > 0 and dy > 8:
                        approach_rate = (dy / fh * 45.0) / dt
                        if approach_rate > 0.4:
                            est_dist = max(2.5, 45.0 * (1.0 - (y2 / fh)))
                            ttc = est_dist / approach_rate

            if is_ped:
                ped_count += 1
            if "HIGH" in risk or "CRITICAL" in risk or proximity == "DANGER" or (ttc is not None and ttc < 2.5):
                critical_threats.append(cls_name)

            detected_classes.append(cls_name)
            id_str = f" #{track_id}" if track_id is not None else ""
            ttc_str = f" | TTC:{ttc:.1f}s" if ttc else ""
            label = f"{cls_name.upper()}{id_str} | {proximity}{ttc_str}"

            draw_hud_box(frame, x1, y1, x2, y2, color, label, is_ped)

            tracked_metadata.append({
                "id": track_id,
                "cls": cls_name,
                "center_x": (x1 + x2) // 2,
                "bbox_bottom": y2,
                "is_hazard": "HIGH" in risk or "CRITICAL" in risk or proximity == "DANGER" or (ttc and ttc < 2.5),
            })

    bev_radar = render_light_bev(tracked_metadata, fw, fh)
    return frame, bev_radar, len(tracked_metadata), ped_count, critical_threats

# ---------------------------------------------------------
# Stream Selection & Execution
# ---------------------------------------------------------
input_mode = st.radio("Perception Stream Selection:", ["📹 Video Perception Feed", "🖼️ Single Frame Inspection"], horizontal=True)

if input_mode == "📹 Video Perception Feed":
    uploaded_video = st.file_uploader("Upload Driving Sequence (.mp4, .mov, .avi, .mkv)", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video:
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, uploaded_video.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video_path)

        alert_placeholder = st.empty()

        col1, col2, col3, col4 = st.columns(4)
        c_fps = col1.empty()
        c_peds = col2.empty()
        c_threats = col3.empty()
        c_targets = col4.empty()

        st.markdown("---")

        stream_view, radar_view = st.columns([3.4, 1.4])

        with stream_view:
            st_frame = st.empty()
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            stop_processing = st.checkbox("⏹ Stop Pipeline Stream", value=False)

        with radar_view:
            st.caption("Top-Down Ground Radar (BEV)")
            st_radar = st.empty()

        frame_counter = 0

        while cap.isOpened() and not stop_processing:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            orig_h, orig_w = frame.shape[:2]
            target_w = 480
            if orig_w > target_w:
                scale = target_w / orig_w
                frame = cv2.resize(frame, (target_w, int(orig_h * scale)), interpolation=cv2.INTER_AREA)

            start_time = time.time()
            annotated_frame, bev_radar, total_objs, ped_cnt, crit_threats = process_frame(
                frame, tracker=enable_tracking
            )

            infer_fps = 1.0 / max(0.001, (time.time() - start_time))
            num_critical = len(crit_threats)
            threat_str = ", ".join([f"{k} ({v})" for k, v in Counter(crit_threats).items()]) if crit_threats else "None"

            if frame_counter % (frame_skip * 3) == 0:
                if num_critical > 0:
                    alert_placeholder.markdown(
                        f"<div class='danger-banner'>⚠️ IMMINENT COLLISION THREAT: {num_critical} HAZARD(S) [{threat_str.upper()}]</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    alert_placeholder.empty()

                c_fps.markdown(
                    f"""<div class="telemetry-card"><div class="telemetry-label">Perception FPS</div><div class="telemetry-value" style="color: #0284c7;">{infer_fps:.1f}</div></div>""",
                    unsafe_allow_html=True,
                )
                c_peds.markdown(
                    f"""<div class="telemetry-card telemetry-ped"><div class="telemetry-label">Pedestrians Detected</div><div class="telemetry-value">{ped_cnt}</div></div>""",
                    unsafe_allow_html=True,
                )
                c_threats.markdown(
                    f"""<div class="telemetry-card {'telemetry-alert' if num_critical > 0 else ''}"><div class="telemetry-label">Critical Hazards</div><div class="telemetry-value">{num_critical}</div></div>""",
                    unsafe_allow_html=True,
                )
                c_targets.markdown(
                    f"""<div class="telemetry-card"><div class="telemetry-label">Tracked Units</div><div class="telemetry-value">{total_objs}</div></div>""",
                    unsafe_allow_html=True,
                )

            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            st_radar.image(cv2.cvtColor(bev_radar, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

            # Prevents pinning the CPU at 100%
            time.sleep(0.035)

        cap.release()
        try:
            os.remove(temp_video_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            pass

elif input_mode == "🖼️ Single Frame Inspection":
    uploaded_image = st.file_uploader("Upload Inspection Still (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Executing Spatial Pass..."):
            annotated_frame, bev_radar, total_objs, ped_cnt, crit_threats = process_frame(frame, tracker=False)

        num_critical = len(crit_threats)
        threat_str = ", ".join([f"{k} ({v})" for k, v in Counter(crit_threats).items()]) if crit_threats else "None"

        if num_critical > 0:
            st.markdown(
                f"<div class='danger-banner'>⚠️ COLLISION WARNING: {num_critical} IMMINENT HAZARD(S) [{threat_str.upper()}]</div>",
                unsafe_allow_html=True,
            )

        h1, h2, h3 = st.columns(3)
        h1.markdown(f"""<div class="telemetry-card"><div class="telemetry-label">Targets Detected</div><div class="telemetry-value">{total_objs}</div></div>""", unsafe_allow_html=True)
        h2.markdown(f"""<div class="telemetry-card telemetry-ped"><div class="telemetry-label">Pedestrians Detected</div><div class="telemetry-value">{ped_cnt}</div></div>""", unsafe_allow_html=True)
        h3.markdown(f"""<div class="telemetry-card {'telemetry-alert' if num_critical > 0 else ''}"><div class="telemetry-label">Critical Hazards</div><div class="telemetry-value">{num_critical}</div></div>""", unsafe_allow_html=True)

        col_cam, col_rad = st.columns([3, 1.4])
        with col_cam:
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Ego Perspective", width="stretch")
        with col_rad:
            st.image(cv2.cvtColor(bev_radar, cv2.COLOR_BGR2RGB), caption="Top-Down Ground Radar (BEV)", width="stretch")