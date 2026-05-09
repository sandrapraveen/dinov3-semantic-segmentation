import torch
from tqdm import tqdm

# --- metrics ---
from src.metrics import SegmentationMetrics

# --- config (controls behavior of training) ---
from config import CONFIG

# --- optional boundary loss ---
from src.losses.boundary_loss import boundary_loss


# =====================================================
# TRAIN FUNCTION
# =====================================================
# Performs one full epoch of training
#
# Flow:
#   input → model → prediction → loss → backprop → update
#
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # enable training mode (dropout, BN updates)

    total_loss = 0

    # tqdm → progress bar for batches
    for images, masks in tqdm(loader):

        # move data to GPU / CPU
        images = images.to(device)
        masks = masks.to(device)

        # clear previous gradients
        optimizer.zero_grad()

        # -----------------------------
        # FORWARD PASS
        # -----------------------------
        outputs = model(images)  
        # shape: (B, num_classes, H, W)

        # -----------------------------
        # LOSS COMPUTATION
        # -----------------------------

        # 1. Cross-Entropy Loss (main segmentation loss)
        ce_loss = criterion(outputs, masks)

        # 2. Optional Boundary-Aware Loss
        if CONFIG["use_boundary_loss"]:
            # computes edge difference between prediction and GT
            b_loss = boundary_loss(outputs, masks)

            # combine both losses
            loss = ce_loss + CONFIG["boundary_weight"] * b_loss
        else:
            # only CE loss
            loss = ce_loss

        # -----------------------------
        # BACKPROPAGATION
        # -----------------------------
        loss.backward()       # compute gradients
        optimizer.step()      # update model weights

        total_loss += loss.item()

    # return average loss over all batches
    return total_loss / len(loader)



# =====================================================
# VALIDATION FUNCTION
# =====================================================
# No gradient updates → only evaluation
#
# Flow:
#   input → model → prediction → loss + metrics
#
def validate(model, loader, criterion, device):
    model.eval()  # disable training behavior (dropout, BN updates)

    # metrics object (mIoU, accuracy, etc.)
    metrics = SegmentationMetrics(num_classes=CONFIG["num_classes"])

    total_loss = 0

    # no gradient computation → faster + less memory
    with torch.no_grad():
        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            # -----------------------------
            # FORWARD PASS
            # -----------------------------
            outputs = model(images)

            # -----------------------------
            # LOSS COMPUTATION
            # -----------------------------

            ce_loss = criterion(outputs, masks)

            if CONFIG["use_boundary_loss"]:
                b_loss = boundary_loss(outputs, masks)
                loss = ce_loss + CONFIG["boundary_weight"] * b_loss
            else:
                loss = ce_loss

            total_loss += loss.item()

            # -----------------------------
            # METRICS UPDATE
            # -----------------------------
            # converts logits → predictions internally
            metrics.update(outputs, masks)

    # compute final metrics (mIoU, etc.)
    scores = metrics.get_scores()

    return total_loss / len(loader), scores