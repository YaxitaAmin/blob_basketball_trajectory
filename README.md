# 🏀 Basketball Shot Trajectory Tracker
**MSML640 Computer Vision Project | Yaxita Amin | Spring 2026**

A classical computer vision application that tracks a basketball's flight path in real-world gym footage, visualizes the arc trajectory, and computes shot metrics like release angle and arc height — using **zero deep learning**.

---

## What It Does

Given a basketball video, the tracker:
- Detects the ball each frame using **HSV color filtering + blob detection**
- Falls back to **Shi-Tomasi corner detection** when the ball is blurred or occluded
- Predicts position during occlusion using a **Kalman filter**
- Segments individual shots using a **state machine** (IDLE → ASCENDING → DESCENDING → COMPLETE)
- Draws a **parabola fit** over the detected arc
- Reports shot metrics: release angle, arc height, flight time, horizontal displacement

---

## Demo

![trajectory demo](assets/demo.png)

---

## How to Run

### 1. Install dependencies
```bash
pip install opencv-python numpy
```

### 2. Run the tracker
```bash
python main.py --input your_video.mp4 --output output.mp4 --show
```

### Available flags
| Flag | Description |
|------|-------------|
| `--input` | Path to input video (required) |
| `--output` | Path to save output video (default: `output_tracked.mp4`) |
| `--show` | Show live window while processing |
| `--debug` | Show HSV mask thumbnail in corner |
| `--no-bg-sub` | Disable background subtraction (recommended for indoor gym videos) |

### Example
```bash
python main.py --input gym_shot.mp4 --output tracked.mp4 --show --debug --no-bg-sub
```

---

## Project Structure

```
├── main.py                  # Entry point — full pipeline
├── hsv_filter.py            # HSV color masking for orange ball detection
├── blob_detector.py         # Contour + corner fallback detection with ROI
├── background_subtractor.py # MOG2 motion mask to reject static false positives
├── kalman_tracker.py        # Kalman filter for smooth tracking + occlusion prediction
├── trajectory.py            # Smoothing, outlier rejection, metrics computation
├── shot_segmenter.py        # State machine for shot segmentation + validation
├── visualizer.py            # Drawing trajectory, parabola fit, HUD, debug mask
```

---

## Core Algorithms (Classical CV Only)

| Algorithm | Purpose |
|-----------|---------|
| Gaussian Blur | Reduce noise before color segmentation |
| HSV Color Filtering | Isolate orange ball pixels |
| Contour Detection | Find circular blob candidates |
| Circularity Check | Reject non-circular false positives |
| Shi-Tomasi Corner Detection | Fallback when blob detection fails |
| MOG2 Background Subtraction | Remove static background noise |
| Kalman Filter | Predict ball position during occlusion |
| Polynomial Fitting (deg=2) | Fit parabola to tracked arc |

---

## Real-World Challenges Addressed

- **Motion blur** — Gaussian preprocessing + corner fallback
- **False positives** — ROI masking (ceiling/floor blackout), saturation threshold, circularity scoring
- **Occlusion** — Kalman filter prediction for up to 8 missed frames
- **Shot direction** — Absolute angle calculation handles left-to-right and right-to-left shots
- **Indoor lighting** — HSV ranges tuned specifically for gym fluorescent lighting

---

## Sample Output Metrics

```
✅ Shot #1 @ frame 212
  release_angle_deg: 21.3°
  arc_height_px: 703 px
  flight_time_sec: 4.37 s
  horizontal_displacement: 566 px
  total_frames_tracked: 67
```

---

## AI Usage Disclosure
---
Claude (Anthropic) was used for debugging assistance and code formatting only. All function logic, algorithm design, pipeline architecture, and parameter tuning decisions were conceived and implemented by Yaxita Amin.
---

## Files Included

- `/src` — All Python source files
- `/data` — Sample test video
- `proposal.pdf` — Original project proposal
- `presentation.pdf` — Final slide deck
