import gradio as gr
import cv2
import torch
import json
import subprocess
import numpy as np
import sys
import os
from pathlib import Path

# ── config ────────────────────────────────────────────────
BASE_DIR      = r"D:\Dev\Animal Detection Live\wildlife-detector"
YOLOV5_PATH   = os.path.join(BASE_DIR, "yolov5")
MODEL_PATH    = os.path.join(BASE_DIR, "best.pt")
CLASSMAP_PATH = os.path.join(BASE_DIR, "class_map.json")
DOWNLOAD_PATH = os.path.join(BASE_DIR, "demo_video.mp4")
OUTPUT_PATH   = os.path.join(BASE_DIR, "output.mp4")

IUCN_COLORS = {
    "Critically Endangered": (50,  50,  220),
    "Endangered":            (50,  130, 220),
    "Vulnerable":            (50,  200, 200),
    "Near Threatened":       (50,  180, 50),
    "Least Concern":         (50,  210, 50),
    "Data Deficient":        (180, 180, 180),
    "Not Evaluated":         (80,  200, 120),  # green instead of grey
}

# ── setup ─────────────────────────────────────────────────
if not Path(YOLOV5_PATH).exists():
    os.system(f'git clone https://github.com/ultralytics/yolov5.git "{YOLOV5_PATH}" -q')

sys.path.insert(0, YOLOV5_PATH)

ckpt     = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
md_model = ckpt['model'].float().eval()
if torch.cuda.is_available():
    md_model = md_model.cuda()
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CPU only")

with open(CLASSMAP_PATH) as f:
    class_map = json.load(f)

print("✓ Model ready")

# ── helpers ───────────────────────────────────────────────
def download_video(youtube_url):
    try:
        result = subprocess.run(
            ["yt-dlp",
             "-f", "best[height<=480][ext=mp4]/best[height<=480]/best",
             "--download-sections", "*0:00-02:00",
             "-o", DOWNLOAD_PATH,
             youtube_url],
            capture_output=True, text=True, timeout=180
        )
        if Path(DOWNLOAD_PATH).exists():
            return DOWNLOAD_PATH, "✓ Downloaded"
        return None, f"Download failed: {result.stderr[:300]}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def detect_frame(frame, conf_threshold=0.35):
    h, w = frame.shape[:2]
    img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_tensor  = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
    img_tensor  = img_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        output = md_model(img_tensor)[0]

    preds      = output[0]
    obj_conf   = preds[:, 4]
    scores_all = preds[:, 5:]
    cls_scores, cls_ids = scores_all.max(dim=1)
    scores     = obj_conf * cls_scores
    mask       = scores > conf_threshold
    scores     = scores[mask]
    boxes      = preds[mask, :4]
    cls_ids    = cls_ids[mask]

    detections = []
    for i in range(len(scores)):
        cx, cy, bw, bh = boxes[i].tolist()
        x1 = max(0, int((cx - bw/2) / 512 * w))
        y1 = max(0, int((cy - bh/2) / 512 * h))
        x2 = min(w, int((cx + bw/2) / 512 * w))
        y2 = min(h, int((cy + bh/2) / 512 * h))

        cls_id   = int(cls_ids[i])
        conf     = float(scores[i])
        info     = class_map.get(str(cls_id), {})
        sci_name = info.get("scientific_name", "Unknown")
        common   = info.get("common_name", "Unknown")
        iucn     = info.get("iucn_status", "Not Evaluated")
        color    = IUCN_COLORS.get(iucn, (80, 200, 120))

        # clean thin box
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # rounded pill label — single line, compact
        # replace the label drawing section inside detect_frame
        font      = cv2.FONT_HERSHEY_SIMPLEX
        line1     = f"{sci_name}  {conf:.2f}"
        line2     = common[:35] if common and common != "Unknown" else ""

        scale1, scale2 = 0.5, 0.38
        thick1, thick2 = 1, 1

        (tw1, th1), bl1 = cv2.getTextSize(line1, font, scale1, thick1)
        (tw2, th2), bl2 = cv2.getTextSize(line2, font, scale2, thick2) if line2 else (0, 0, 0)

        pad_x, pad_y = 8, 5
        gap          = 4  # gap between line1 and line2

        box_w = max(tw1, tw2) + pad_x * 2 + 3
        box_h = th1 + (th2 + gap if line2 else 0) + pad_y * 2

        # position above box, below if no space
        if y1 - box_h - 4 >= 0:
            lx = max(0, min(x1, w - box_w))
            ly = y1 - box_h - 4
        else:
            lx = max(0, min(x1, w - box_w))
            ly = y2 + 4

        # semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + box_w, ly + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # left color accent bar
        cv2.rectangle(frame, (lx, ly), (lx + 3, ly + box_h), color, -1)

        # line 1 — scientific name + confidence
        cv2.putText(frame, line1,
            (lx + pad_x + 2, ly + pad_y + th1),
            font, scale1, (240, 240, 240), thick1, cv2.LINE_AA)

        # line 2 — common name
        if line2:
            cv2.putText(frame, line2,
                (lx + pad_x + 2, ly + pad_y + th1 + gap + th2),
                font, scale2, color, thick2, cv2.LINE_AA)

        detections.append(f"**{sci_name}** (*{common}*) `{conf:.2f}` | {iucn}")

    return frame, detections

def process_video(video_path, conf_threshold, every_n, progress=gr.Progress()):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    frame_count  = 0
    detect_count = 0
    all_species  = set()

    PAUSE_FRAMES = int(fps * 1.5)  # pause for 2 seconds on detection

    progress(0, desc="Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress(frame_count / max(total_frames, 1), desc=f"Frame {frame_count}/{total_frames}")

        if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < 40:
            out.write(frame)
            continue

        if frame_count % int(every_n) == 0:
            annotated, last_detections = detect_frame(frame, conf_threshold)

            if last_detections:
                detect_count += 1
                for d in last_detections:
                    species = d.split("**")[1]
                    all_species.add(species)

                # write the annotated frame N times = pause effect
                for _ in range(PAUSE_FRAMES):
                    out.write(annotated)
            else:
                out.write(frame)
        else:
            out.write(frame)

    cap.release()
    out.release()

    summary = (
        f"✓ **{frame_count}** frames processed\n\n"
        f"✓ **{detect_count}** detection events\n\n"
        f"✓ **{len(all_species)}** unique species\n\n"
        f"---\n\n**Species found:**\n\n"
        + "\n\n".join(f"- {s}" for s in sorted(all_species))
    )

    return OUTPUT_PATH, summary

def download_and_process(youtube_url, conf_threshold, every_n, progress=gr.Progress()):
    if not youtube_url.strip():
        return None, "Please enter a YouTube URL."
    progress(0, desc="Downloading video...")
    video_path, msg = download_video(youtube_url)
    if not video_path:
        return None, msg
    progress(0.1, desc="Running detection...")
    return process_video(video_path, conf_threshold, every_n, progress)

def process_uploaded(video_file, conf_threshold, every_n, progress=gr.Progress()):
    if video_file is None:
        return None, "Please upload a video."
    return process_video(video_file, conf_threshold, every_n, progress)

# ── UI ────────────────────────────────────────────────────
with gr.Blocks(title="Wildlife Detector", theme=gr.themes.Base(primary_hue="green")) as demo:

    gr.Markdown("# 🦁 Wildlife Detection\n**Fine-tuned MegaDetector · 500 species · mAP50: 0.637**")

    with gr.Tabs():

        with gr.Tab("📥 YouTube Video"):
            gr.Markdown("Download a YouTube video and run detection on it.")
            yt_url = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                value="https://www.youtube.com/watch?v=o50N3-OaGdM"
            )
            with gr.Row():
                yt_conf    = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence")
                yt_every_n = gr.Slider(1, 30, value=15, step=1, label="Inference every N frames")
            with gr.Row():
                yt_btn  = gr.Button("⬇ Download & Detect", variant="primary", size="lg")
                yt_stop = gr.Button("⏹ Stop")
            with gr.Row():
                yt_video = gr.Video(label="Detected output")
                yt_text  = gr.Markdown()
            yt_event = yt_btn.click(
                fn=download_and_process,
                inputs=[yt_url, yt_conf, yt_every_n],
                outputs=[yt_video, yt_text]
            )
            yt_stop.click(fn=None, cancels=[yt_event])

        with gr.Tab("📂 Upload Video"):
            gr.Markdown("Upload your own video file.")
            up_video = gr.Video(label="Upload video")
            with gr.Row():
                up_conf    = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence")
                up_every_n = gr.Slider(1, 30, value=15, step=1, label="Inference every N frames")
            with gr.Row():
                up_btn  = gr.Button("▶ Run Detection", variant="primary", size="lg")
                up_stop = gr.Button("⏹ Stop")
            with gr.Row():
                up_out_video = gr.Video(label="Detected output")
                up_text      = gr.Markdown()
            up_event = up_btn.click(
                fn=process_uploaded,
                inputs=[up_video, up_conf, up_every_n],
                outputs=[up_out_video, up_text]
            )
            up_stop.click(fn=None, cancels=[up_event])

    gr.Markdown("---\nCSE 546 · Spring 2026 · University at Buffalo")

demo.launch(inbrowser=True, allowed_paths=[BASE_DIR])
