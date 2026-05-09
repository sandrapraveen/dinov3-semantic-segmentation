import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true != self.ignore_index)

        label_true = label_true[mask]
        label_pred = label_pred[mask]

        hist = np.bincount(
            self.num_classes * label_true.astype(int) + label_pred.astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        return hist

    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1)

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        for p, t in zip(preds, targets):
            self.confusion_matrix += self._fast_hist(t.flatten(), p.flatten())

    def get_scores(self):
        hist = self.confusion_matrix
        eps = 1e-10

        intersection = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - intersection

        # --- metrics ---
        overall_acc = intersection.sum() / (hist.sum() + eps)

        per_class_acc = intersection / (hist.sum(axis=1) + eps)
        mean_class_acc = np.nanmean(per_class_acc)

        per_class_iou = intersection / (union + eps)

        valid = union > 0
        mean_iou = np.mean(per_class_iou[valid])

        freq = hist.sum(axis=1) / (hist.sum() + eps)
        fw_iou = (freq[freq > 0] * per_class_iou[freq > 0]).sum()

        return {
            "overall_acc": overall_acc,
            "mean_class_acc": mean_class_acc,
            "per_class_acc": per_class_acc,
            "per_class_iou": per_class_iou,
            "mean_iou": mean_iou,
            "freq_weighted_iou": fw_iou,
            "confusion_matrix": hist
        }