# Automated Optical Inspection (AOI) for Fasteners

An open-source, real-time Computer Vision system designed to automate the quality control of industrial fasteners (bolts). Using a standard webcam and OpenCV, this system detects, measures, and classifies bolts based on dimensional accuracy and thread presence.

## Features
* **Real-Time Processing:** Operates at 30+ FPS on a standard CPU.
* **Dimensional Verification:** Measures object length with a customizable +/- 5mm tolerance using a pixel-to-millimeter calibration factor.
* **Thread Detection:** Uses a novel 1D signal roughness algorithm to distinguish between threaded bolts and smooth shanks without the need for heavy Deep Learning models.
* **Shadow Mitigation:** Utilizes Adaptive Gaussian Thresholding to ignore thick shadows cast by large bolts.
* **Binary Output:** Simple "OK" (Green) or "NOT OK" (Red) visual feedback.

---

## How It Works
The inspection pipeline follows a strict "Double-Check" logic gate:
1. **Size Check:** The system isolates the bolt using morphological closing, finds the bounding box, and calculates the physical length in mm. If the length is outside the +/- 5mm tolerance, it fails immediately.
2. **Texture Check:** If the size is correct, the system extracts a vertical strip down the center of the bolt, applies CLAHE contrast enhancement, and calculates the intensity variance. A zig-zag variance indicates threads; a flat variance indicates a smooth/defective bolt.

---
