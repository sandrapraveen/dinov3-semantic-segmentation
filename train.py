import sys
sys.path.append(".")  # allows importing from src/ when running script directly

import torch
from torch.utils.data import DataLoader

# --- project modules ---
from src.dataset import LoveDADataset
from src.model import DinoSegmentationModel
from src.engine import train_one_epoch, validate
from src.utils import SaveBestModelIOU, save_checkpoint, visualize_prediction

# --- config (single control point for experiments) ---
from config import CONFIG
import os


# =====================================================
# EXPERIMENT SETUP (AUTO NAMING)
# =====================================================
# creates experiment name based on config
# ensures reproducibility and no overwriting
EXP_NAME = f"{CONFIG['decoder']}_ft{CONFIG['fine_tune']}_b{CONFIG['use_boundary_loss']}"

# create output directory for this experiment
OUTPUT_DIR = os.path.join("outputs", EXP_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Saving outputs to:", OUTPUT_DIR)


# =====================================================
# CONFIG (READ FROM CONFIG FILE)
# =====================================================
DATA_PATH = CONFIG["data_path"]     # dataset root
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = CONFIG["batch_size"]   # training batch size
EPOCHS = CONFIG["epochs"]           # number of epochs


# =====================================================
# DATASET LOADING
# =====================================================
# creates dataset objects (handles loading + transforms)
train_dataset = LoveDADataset(DATA_PATH, split="Train")
val_dataset = LoveDADataset(DATA_PATH, split="Val")

# DataLoader handles batching + shuffling
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =====================================================
# MODEL INITIALIZATION
# =====================================================
# config controls:
# ✔ decoder type (simple / dpt)
# ✔ fine-tuning (freeze / train backbone)
# ✔ pretrained weights

model = DinoSegmentationModel(
    num_classes=CONFIG["num_classes"],
    weights=CONFIG["weights_path"],     # pretrained DINO weights
    fine_tune=CONFIG["fine_tune"],      # freeze or train backbone
    decoder_type=CONFIG["decoder"]      # select decoder head
).to(DEVICE)


# =====================================================
# SANITY CHECK (VERY IMPORTANT)
# =====================================================
# ensures model forward pass works before training
print("Decoder:", CONFIG["decoder"])

model.eval()

with torch.no_grad():
    images, masks = next(iter(train_loader))  # take one batch

    print("Input shape:", images.shape)

    images = images.to(DEVICE)

    out = model(images)

    print("Output shape:", out.shape)
    # expected: (B, num_classes, H, W)


# =====================================================
# UTILITIES
# =====================================================
# saves best model based on validation mIoU
save_best = SaveBestModelIOU(
    os.path.join(OUTPUT_DIR, "best_model.pth")
)

# segmentation loss (baseline)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

# optimizer (learning rate from config)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])


# =====================================================
# COLOR MAP (FOR VISUALIZATION)
# =====================================================
# maps class index → RGB color
colors = [
    (0, 0, 0),        # background
    (128, 0, 0),      # class 1
    (0, 128, 0),      # class 2
    (128, 128, 0),    # class 3
    (0, 0, 128),      # class 4
    (128, 0, 128),    # class 5
    (0, 128, 128),    # class 6
]


# =====================================================
# TRAINING LOOP
# =====================================================
for epoch in range(EPOCHS):

    # -------------------------
    # TRAINING STEP
    # -------------------------
    # forward → loss → backward → update
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        DEVICE
    )

    # -------------------------
    # VALIDATION STEP
    # -------------------------
    # evaluate model performance
    val_loss, scores = validate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"mIoU: {scores['mean_iou']:.4f}"
    )

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    # saves model if mIoU improves
    save_best(scores["mean_iou"], epoch, model)

    # -------------------------
    # SAVE CHECKPOINT
    # -------------------------
    # saves model + optimizer for resume
    save_checkpoint(
        model,
        optimizer,
        epoch,
        path=os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch}.pth")
    )

    # -------------------------
    # VISUALIZE PREDICTION
    # -------------------------
    # useful for qualitative analysis
    model.eval()

    with torch.no_grad():
        img, mask = val_dataset[0]

        img_input = img.unsqueeze(0).to(DEVICE)
        output = model(img_input)

        visualize_prediction(
            img.cpu(),
            output[0].cpu(),
            colors,
            save_path=os.path.join(OUTPUT_DIR, f"pred_epoch_{epoch}.jpg")
        )