import numpy as np
from scipy.optimize import linear_sum_assignment


def boxes_array(boxes):
    a = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not np.isfinite(a).all() or (a[:, 2:] <= a[:, :2]).any():
        raise ValueError('Boxes must be finite, positive-area xyxy coordinates')
    return a


def iou_matrix(a, b):
    a, b = boxes_array(a), boxes_array(b)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(hi-lo, 0).prod(axis=2)
    area_a = (a[:, 2:]-a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:]-b[:, :2]).prod(axis=1)
    return intersection / (area_a[:, None]+area_b[None, :]-intersection)


def match_boxes(a, b, threshold):
    """Maximum-cardinality gated assignment, then maximize IoU within valid edges."""
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    ious = iou_matrix(a, b)
    if not ious.size:
        return []
    # A forbidden match costs more than every possible valid match combined.
    cost = np.where(ious >= threshold, 1-ious, min(ious.shape)+1)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if ious[r, c] >= threshold]
