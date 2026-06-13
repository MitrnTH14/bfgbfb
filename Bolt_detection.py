#!/usr/bin/env python3

import cv2
import numpy as np
import time
from collections import deque

# ================= USER CONFIGURATION =================

# --- 1. CALIBRATION ---
# CHANGE THIS to the real length of your small bolt!
REAL_LENGTH_MM = 60.0  
REF_PIXELS     = 354.0 

# Scale Factor
PX_PER_MM = REF_PIXELS / REAL_LENGTH_MM

# --- 2. TARGETS (Converted to mm) ---
TARGETS_MM = {
    "SMALL":  (354 / PX_PER_MM, 69 / PX_PER_MM),
    "MEDIUM": (460 / PX_PER_MM, 65 / PX_PER_MM),
    "LARGE":  (254 / PX_PER_MM, 140 / PX_PER_MM)
}

# --- 3. THE RULES ---
TOLERANCE_MM = 5.0    # +/- 5mm allowed
THREAD_THRESH = 8.0   # Sensitivity

# --- 4. CAMERA & BOX ---
CAMERA_INDEX = 0      
BOX_SIZE_W = 180      # <--- TINY WIDTH (Requires precision)
BOX_SIZE_H = 360      # <--- TINY HEIGHT (Just barely fits Small Bolt)
SCAN_WIDTH_PCT = 0.25 
GAMMA_VALUE = 1.5     
# ======================================================

class Stabilizer:
    def __init__(self, max_len):
        self.len_buffer = deque(maxlen=max_len)
        self.dia_buffer = deque(maxlen=max_len)
    def add(self, length, diameter):
        self.len_buffer.append(length)
        self.dia_buffer.append(diameter)
    def get_stable_values(self):
        if len(self.len_buffer) < 1: return 0, 0
        stable_len = np.median(self.len_buffer)
        stable_dia = np.median(self.dia_buffer)
        return stable_len, stable_dia
    def reset(self):
        self.len_buffer.clear()
        self.dia_buffer.clear()

stabilizer = Stabilizer(15)

# Global Variables
ok_buffer  = 0
final_display_text = "READY"
box_color = (100, 100, 100)
latch_frames = 8

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gamma_table = np.array([((i / 255.0) ** GAMMA_VALUE) * 255 for i in np.arange(0, 256)]).astype("uint8")

def analyze_bolt_tiny(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- 1. SIZE ---
    blur_shape = cv2.GaussianBlur(gray, (7, 7), 0)
    binary = cv2.adaptiveThreshold(blur_shape, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    kernel = np.ones((9,9), np.uint8) 
    solid_shape = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    solid_shape = cv2.morphologyEx(solid_shape, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(solid_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    px_len = 0
    px_dia = 0
    bolt_rect = None
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            x, y, width, height = cv2.boundingRect(c)
            px_len = height 
            px_dia = width   
            bolt_rect = (x, y, width, height)

    # --- 2. THREADS ---
    darkened = cv2.LUT(gray, gamma_table)
    enhanced = clahe.apply(darkened)
    
    crop_w = int(w * SCAN_WIDTH_PCT)
    start_x = int(w * (0.5 - SCAN_WIDTH_PCT/2)) 
    strip = enhanced[:, start_x : start_x + crop_w]
    
    activity = 0
    if strip.size > 0 and px_len > 0:
        profile = np.mean(strip, axis=1)
        activity = np.mean(np.abs(np.diff(profile)))

    return px_len, px_dia, activity, bolt_rect

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened(): exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 

print(f"--- TINY BOX MODE ---")
print(f"Box: {BOX_SIZE_W}x{BOX_SIZE_H}")
print(f"Align bolt perfectly centered!")

last_print = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    c_x, c_y = w // 2, h // 2
    
    # TINY ROI
    x1 = max(0, c_x - (BOX_SIZE_W // 2))
    x2 = min(w, c_x + (BOX_SIZE_W // 2))
    y1 = max(0, c_y - (BOX_SIZE_H // 2))
    y2 = min(h, c_y + (BOX_SIZE_H // 2))
    roi = frame[y1:y2, x1:x2]
    
    # 1. ANALYZE
    r_len, r_dia, r_act, bolt_rect = analyze_bolt_tiny(roi)
    
    # 2. STABILIZE
    if r_len > 20:
        stabilizer.add(r_len, r_dia)
        s_len_px, s_dia_px = stabilizer.get_stable_values()
    else:
        stabilizer.reset()
        s_len_px, s_dia_px = 0, 0

    # 3. CONVERT TO MM
    mm_len = s_len_px / PX_PER_MM
    mm_dia = s_dia_px / PX_PER_MM
    
    bolt_present = mm_len > 10.0
    is_valid_size = False
    
    # 4. CHECK TARGETS (+/- 5mm)
    if bolt_present:
        for name, (t_len, t_dia) in TARGETS_MM.items():
            if abs(mm_len - t_len) <= TOLERANCE_MM:
                is_valid_size = True
                break
        
        # Shadow exception for Large Bolt
        if not is_valid_size and mm_dia > 25.0:
             target_L = TARGETS_MM["LARGE"][0]
             if abs(mm_len - target_L) <= 10.0:
                 is_valid_size = True

    # 5. DECISION
    is_threaded = r_act >= THREAD_THRESH
    is_pass = bolt_present and is_valid_size and is_threaded

    temp_text = "READY"
    if not bolt_present:
        temp_text = "PLACE BOLT"
        box_color = (100, 100, 100)
        ok_buffer = 0
    else:
        if is_pass:
            temp_text = "OK"
            temp_color = (0, 255, 0)
            ok_buffer = min(ok_buffer + 1, latch_frames)
        else:
            temp_text = "NOT OK"
            temp_color = (0, 0, 255)
            ok_buffer = 0

    if ok_buffer >= latch_frames:
        final_display_text = "OK"
        box_color = (0, 255, 0)
    elif temp_text == "NOT OK":
        final_display_text = "NOT OK"
        box_color = (0, 0, 255)
    elif temp_text == "PLACE BOLT":
        final_display_text = "..."
        box_color = (100, 100, 100)
        
    # 6. VISUALS
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    
    if bolt_present and bolt_rect is not None:
        bx, by, bw, bh = bolt_rect
        cv2.rectangle(frame, (x1+bx, y1+by), (x1+bx+bw, y1+by+bh), box_color, 2)
    
    text_size = cv2.getTextSize(final_display_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, final_display_text, (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, box_color, 5)

    if bolt_present:
        cv2.putText(frame, f"L: {mm_len:.1f} mm", (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"D: {mm_dia:.1f} mm", (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("QC Station (Tiny)", frame)

    if time.time() - last_print > 0.5:
        if bolt_present:
            print(f"Status: {final_display_text:<10} | {mm_len:.1f}mm | Thr: {is_threaded}")
        last_print = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
