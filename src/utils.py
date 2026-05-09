import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# --- import config so everything is consistent ---
from config import CONFIG


# =====================================================
# NORMALIZATION (NOW FROM CONFIG)
# =====================================================
# Ensures training, inference, and visualization all use SAME values
DINO_MEAN = CONFIG["mean"]
DINO_STD  = CONFIG["std"]


# =====================================================
# MODEL SAVING UTILITIES
# =====================================================
class SaveBestModelIOU:
    """
    Saves model when validation mIoU improves.

    Also stores CONFIG for reproducibility.
    """
    def __init__(self, save_path="best_model.pth"):
        self.best_iou = 0.0
        self.save_path = save_path

    def __call__(self, current_iou, epoch, model):
        if current_iou > self.best_iou:
            self.best_iou = current_iou

            print(f"\nSaving BEST model at epoch {epoch+1} | mIoU={current_iou:.4f}")
            print(f"Saved to: {self.save_path}\n")

            # Save model + config (VERY IMPORTANT for experiments)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": CONFIG,
                "best_iou": current_iou,
                "epoch": epoch,
            }, self.save_path)


class SaveBestModelLoss:
    """
    Saves model when validation loss improves.
    Useful if optimizing for loss instead of mIoU.
    """
    def __init__(self, save_path="best_model_loss.pth"):
        self.best_loss = float("inf")
        self.save_path = save_path

    def __call__(self, current_loss, epoch, model):
        if current_loss < self.best_loss:
            self.best_loss = current_loss

            print(f"\nSaving BEST model at epoch {epoch+1} | Loss={current_loss:.4f}")
            print(f"Saved to: {self.save_path}\n")

            torch.save({
                "model_state_dict": model.state_dict(),
                "config": CONFIG,
                "best_loss": current_loss,
                "epoch": epoch,
            }, self.save_path)


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    """
    Saves checkpoint for resuming training later.

    Includes:
    ✔ model weights
    ✔ optimizer state
    ✔ epoch number
    ✔ config (for reproducibility)
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": CONFIG,
    }, path)

    print(f"Checkpoint saved: {path}")


# =====================================================
# VISUALIZATION UTILITIES
# =====================================================
def denormalize_image(img):
    """
    Convert normalized tensor → displayable image

    Input:
        tensor (3, H, W)

    Output:
        numpy image (H, W, 3) in range [0,1]
    """
    img = img.permute(1, 2, 0).cpu().numpy()

    # reverse normalization
    img = img * DINO_STD + DINO_MEAN

    img = np.clip(img, 0, 1)
    return img


def decode_segmap(mask, colors):
    """
    Convert class mask → RGB image

    mask: (H, W)
    colors: list of (R, G, B)
    """
    h, w = mask.shape
    seg_map = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        seg_map[mask == i] = color

    return seg_map


def overlay_segmentation(image, seg_map):
    """
    Overlay segmentation map on original image

    image: (H, W, 3) RGB [0,1]
    seg_map: (H, W, 3)
    """
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    seg_map = cv2.cvtColor(seg_map, cv2.COLOR_RGB2BGR)

    # blend image + segmentation
    overlay = cv2.addWeighted(image, 0.7, seg_map, 0.3, 0)
    return overlay


def visualize_prediction(image, output, colors, save_path=None):
    """
    Creates visualization of model prediction.

    image: tensor (3, H, W)
    output: model output (C, H, W)
    """

    # -----------------------------
    # GET PREDICTED CLASS MAP
    # -----------------------------
    pred = torch.argmax(output, dim=0).cpu().numpy()

    # -----------------------------
    # DENORMALIZE INPUT IMAGE
    # -----------------------------
    image = denormalize_image(image)

    # -----------------------------
    # CREATE COLOR SEGMENTATION MAP
    # -----------------------------
    seg_map = decode_segmap(pred, colors)

    # -----------------------------
    # OVERLAY SEGMENTATION ON IMAGE
    # -----------------------------
    overlay = overlay_segmentation(image, seg_map)

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    if save_path is not None:
        cv2.imwrite(save_path, overlay)

        # save colored mask (recommended)
        mask_path = save_path.replace(".png", "_mask.png")
        cv2.imwrite(mask_path, seg_map)

        print(f"Saved visualization: {save_path}")
        print(f"Saved mask: {mask_path}")

    return overlay


# =====================================================
# INFERENCE HELPER
# =====================================================
def predict_single_image(model, image, device):
    """
    Runs model on a single image.

    image: tensor (1, 3, H, W)
    """
    model.eval()

    with torch.no_grad():
        output = model(image.to(device))

    return output