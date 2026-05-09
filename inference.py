import os
import glob
import torch
import cv2
import random   # ← for selecting random images

from src.model import DinoSegmentationModel
from src.utils import visualize_prediction


# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = r"C:\Users\sandr\Downloads\dinooooo\LoveDA\Test\Test\Urban\images_png"

# Use absolute path to avoid saving outside folder
OUTPUT_DIR = os.path.join(os.getcwd(), "inference_outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose decoder type (experiment control)
DECODER = "dpt"   # "simple" or "dpt"


# -----------------------------
# LOAD MODEL
# -----------------------------
model = DinoSegmentationModel(
    num_classes=7,
    weights=None,               # do NOT load pretrained again
    decoder_type=DECODER        # important: match training
).to(DEVICE)

print("Using decoder:", DECODER)

# Load trained weights (this contains full model)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()


# -----------------------------
# COLORS (for visualization)
# -----------------------------
colors = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128)
]


# -----------------------------
# LOAD IMAGES
# -----------------------------
# Only pick image files (important)
image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))

print("Exists?", os.path.exists(INPUT_DIR))
print("Total images found:", len(image_paths))


# -----------------------------
# SELECT ONLY 4 RANDOM IMAGES
# -----------------------------
# Instead of processing entire dataset
selected_images = random.sample(image_paths, min(4, len(image_paths)))

print("Selected images:", [os.path.basename(p) for p in selected_images])


# -----------------------------
# INFERENCE LOOP
# -----------------------------
for image_path in selected_images:

    print("Processing:", os.path.basename(image_path))

    # --- load image ---
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- resize (must match training size) ---
    image_resized = cv2.resize(image, (384, 384))

    # --- normalize (VERY IMPORTANT: same as training) ---
    mean = torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1)
    std  = torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1)

    image_tensor = torch.tensor(image_resized / 255.0).permute(2, 0, 1).float()
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # --- model prediction ---
    with torch.no_grad():
        output = model(image_tensor)

    # --- save visualization ---
    save_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))

    visualize_prediction(
        image_tensor[0].cpu(),
        output[0].cpu(),
        colors,
        save_path=save_path
    )

    print(f"Saved: {save_path}")