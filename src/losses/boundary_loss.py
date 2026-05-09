import torch
import torch.nn.functional as F


def _sobel_edges(x):
    # x: (B, 1, H, W)
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1,-2,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    return torch.sqrt(gx**2 + gy**2)


def boundary_loss(logits, target):
    """
    logits: (B, C, H, W)
    target: (B, H, W)
    """

    # predicted labels
    pred = torch.argmax(logits, dim=1, keepdim=True).float()  # (B,1,H,W)
    target = target.unsqueeze(1).float()                      # (B,1,H,W)

    pred_edges = _sobel_edges(pred)
    target_edges = _sobel_edges(target)

    return F.l1_loss(pred_edges, target_edges)