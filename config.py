"""
config.py
---------
Central constants: model/asset paths, detection thresholds, colors,
and runtime defaults. No logic here — only values other modules import.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------
MODEL_PATH = ROOT_DIR / "face_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
ASSETS_DIR = ROOT_DIR / "assets"
SUNGLASSES_PNG_PATH = ROOT_DIR / "ironman_sunglasses.png"

RECOGNITION_PANEL_PATH = ASSETS_DIR / "recognition_panel.png"

# Reticle styles — cycle through these with keys 1/2/3/4 in main.py.
RETICLE_PATHS = [
    ASSETS_DIR / "reticle_arrows.png",
    ASSETS_DIR / "reticle_rings.png",
    ASSETS_DIR / "reticle_dot.png",
    ASSETS_DIR / "reticle_brackets.png",
]
RETICLE_SIZE_SCALE = 0.9   # reticle width as a multiple of eye-to-eye span
RETICLE_SPIN_SPEED = 1.5   # degrees rotated per frame

FACE_SCAN_MASK_PATH = ASSETS_DIR / "face_scan_mask.png"
MASK_ALPHA = 0.45            # overall transparency of the mask overlay
MASK_PADDING_SCALE = 1.18    # mask size relative to the face bounding box

# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --------------------------------------------------------------------------
# Runtime defaults
# --------------------------------------------------------------------------
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
SMOOTHING_ALPHA = 0.55
DOT_RADIUS = 1
LINE_THICKNESS = 1
WINDOW_NAME = "Face Mesh AR - Professional Edition"

# Sunglasses sizing: width as a multiple of the eye-to-eye (outer corner)
# distance, and a small vertical offset (in fractions of that distance)
# to nudge the glasses up/down relative to the eye line.
SUNGLASSES_WIDTH_SCALE = 2.6
SUNGLASSES_Y_OFFSET_SCALE = 0.05

# --------------------------------------------------------------------------
# Colors (BGR — OpenCV convention)
# --------------------------------------------------------------------------
COLOR_MESH_DOTS = (180, 220, 255)
COLOR_FACE_OVAL = (255, 255, 255)
COLOR_EYES = (0, 220, 255)
COLOR_IRIS = (0, 140, 255)
COLOR_EYEBROWS = (200, 160, 255)
COLOR_LIPS = (120, 80, 255)
COLOR_NOSE = (180, 255, 180)
COLOR_HUD_TEXT = (255, 255, 255)
COLOR_HUD_BG = (30, 30, 30)
COLOR_BBOX = (0, 255, 120)

# Fallback vector visor (used only if the PNG asset is missing)
COLOR_VISOR_RIM = (20, 25, 170)
COLOR_VISOR_LENS = (15, 12, 12)
COLOR_VISOR_GOLD = (40, 170, 235)
COLOR_VISOR_SHINE = (255, 255, 255)

# Sparkle / confetti particles
SPARKLE_COLORS = [
    (60, 220, 60), (40, 60, 230), (235, 170, 40),
    (220, 120, 30), (230, 60, 200), (230, 200, 40),
]
NUM_SPARKLES = 36

# Scanning / facial-recognition HUD
COLOR_SCAN_LINE = (255, 230, 0)
COLOR_SCAN_BRACKET = (255, 255, 255)
COLOR_RETICLE = (255, 230, 0)
COLOR_BADGE = (210, 210, 210)
COLOR_BADGE_ACCENT = (255, 230, 0)