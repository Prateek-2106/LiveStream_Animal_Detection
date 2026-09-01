# WildLens 🦌📷

**Real-time wildlife detection and species identification system**, built for CSE 546 (University at Buffalo). WildLens detects animals in video/livestream footage and classifies them across 500 vertebrate species.

🔗 **Live demo**: [Prateek2106/Wildlife-Detector on HuggingFace Spaces](https://huggingface.co/spaces/Prateek2106/Wildlife-Detector)

---

## Overview

WildLens combines a fine-tuned detector with a species classifier to identify wildlife from video input — whether it's an uploaded clip, a livestream, or a YouTube video. The goal: give researchers, conservationists, or hobbyists a fast, low-friction way to identify what's on camera without manual review.

## Features

- 🎯 **Detection + classification pipeline** fine-tuned from **MegaDetector v5a**, extended to identify **500 vertebrate species**
- 📹 **Multiple input modes**: upload a video, point it at a livestream, or drop in a **YouTube URL** (via `yt-dlp`) for automatic download-and-detect
- 🖥️ **Gradio-based UI** — runs locally or deployed on HuggingFace Spaces
- ⚡ Trained and tuned for real-time-friendly inference

## Results

| Metric | Value |
|---|---|
| Species covered | 500 |
| Training set | ~92K images |
| Validation set | ~8K images |
| mAP50 | ~0.637 |

The dataset was auto-labeled using MegaDetector v5a and augmented into a flat 500-class structure for training.

## Repo Structure

```
├── Live_animal_detection.ipynb   # Training / experimentation notebook
├── app.py                        # Gradio app entry point
├── Sample Videos/                # Example input clips
├── output.mp4                    # Sample detection output
├── WildLifeWatch_Poster_v2.pptx  # Project poster (CSE Demo Day)
└── requirements.txt
```

## Getting Started

```bash
git clone https://github.com/<your-username>/LiveStream_Animal_Detection.git
cd LiveStream_Animal_Detection
pip install -r requirements.txt
python app.py
```

This launches the Gradio app locally, where you can upload a video, paste a YouTube link, or point it at a livestream source.

## Tech Stack

- **Detection base**: MegaDetector v5a
- **Classification**: Fine-tuned on a custom 500-species dataset
- **UI**: Gradio
- **Video ingestion**: yt-dlp
- **Training hardware**: Windows / RTX 5070 (local), with cloud experimentation during development

## Background

Earlier iterations explored a YOLOv8m-based segmentation approach before pivoting to the MegaDetector-based detect-then-classify pipeline, which gave stronger species-level accuracy. Presented at **UB CSE Demo Day**.

## Roadmap

- [ ] Dockerize the app
- [ ] CI/CD via GitHub Actions
- [ ] Explore PySpark for the dataset augmentation/split pipeline

## Acknowledgments

Built on top of [MegaDetector](https://github.com/microsoft/CameraTraps) for the base detection model.

---

*CSE 546 course project — University at Buffalo*
