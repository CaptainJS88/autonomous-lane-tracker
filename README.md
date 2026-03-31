# Autonomous Lane Tracker Playground

This project is my lightweight lane-detection pipeline for experimenting with classical computer vision on local road images and videos.

The basic flow is:
1. read image/video frames from `Images/` and `Videos/`
2. run preprocessing (grayscale + blur + Canny + ROI masking)
3. detect lane boundaries with Hough lines and visualize overlays

I kept it simple on purpose so it's easy to tweak thresholds, ROI geometry, and line-fitting behavior.

## What each file does

- `LanePerception/lane_detector.py`: full end-to-end script that:
  - runs lane detection on a sample image
  - processes video frame-by-frame
  - writes `Videos/output_lanes.mp4`
  - includes perspective-warp + sliding-window lane search experiments
- `LanePerception/Images/`: sample road images used for testing.
- `LanePerception/Videos/`: input clips and processed output video.
- `LanePerception/Samples/`: extra assets for quick local experiments.
- `LanePerception/README.md`: module-level notes and portfolio context.

## Requirements

- Python 3
- `pip`
- Python packages: `opencv-python`, `numpy`, `matplotlib`, `Pillow`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy matplotlib pillow
```

## Run

From the project root:

```bash
python LanePerception/lane_detector.py
```

## Inputs and outputs

- Inputs:
  - `LanePerception/Images/test_lane.jpg`
  - `LanePerception/Videos/solidWhiteRight.mp4`
- Outputs:
  - visual debug plots (Matplotlib windows)
  - real-time video preview window (OpenCV)
  - `LanePerception/Videos/output_lanes.mp4`

## Notes

- If a window does not appear, confirm your environment supports GUI rendering for OpenCV/Matplotlib.
- For quick experiments, start by tuning Canny thresholds and ROI points in `lane_detector.py`.
- The script currently combines multiple experiments in one file, which is intentional for rapid iteration.
