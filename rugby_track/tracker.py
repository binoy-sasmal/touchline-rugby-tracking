from dataclasses import dataclass
import numpy as np
from .geometry import boxes_array, match_boxes


@dataclass
class Track:
    identity: int
    box: np.ndarray
    velocity: np.ndarray
    observed_at: int

    def predicted(self, frame):
        predicted = self.box + self.velocity * (frame-self.observed_at)
        predicted[2:] = np.maximum(predicted[2:], predicted[:2]+1)
        return predicted


class Tracker:
    """Constant-velocity IoU association, with a second pass for weak detections.

    Not an implementation of ByteTrack/SORT; no learned appearance or Kalman filter.
    Lost predictions are used for association only, never exported as observations.
    """
    def __init__(self, high=0.5, low=0.15, iou=0.15, max_age=10):
        if not 0 <= low <= high <= 1 or not 0 < iou <= 1 or max_age < 0:
            raise ValueError('Invalid tracker parameters')
        self.high, self.low, self.iou, self.max_age = high, low, iou, max_age
        self.tracks = {}
        self.next_id = 1
        self.last_frame = -1

    def reset(self):
        self.tracks.clear()
        # Do not reuse IDs across scenes in one run.

    def update(self, frame, detections):
        if frame <= self.last_frame:
            raise ValueError('Frame indices must increase strictly')
        self.last_frame = frame
        for ident in list(self.tracks):
            if frame-self.tracks[ident].observed_at > self.max_age:
                del self.tracks[ident]
        boxes_array([d['box'] for d in detections])
        if any(not np.isfinite(d['score']) or not 0 <= d['score'] <= 1 for d in detections):
            raise ValueError('Invalid confidence')
        strong = [i for i,d in enumerate(detections) if d['score'] >= self.high]
        weak = [i for i,d in enumerate(detections) if self.low <= d['score'] < self.high]
        remaining = list(self.tracks)
        assigned = {}
        for candidates in [strong, weak]:
            pairs = match_boxes([self.tracks[i].predicted(frame) for i in remaining],
                                [detections[i]['box'] for i in candidates], self.iou)
            used = set()
            for r,c in pairs:
                ident, di = remaining[r], candidates[c]
                track = self.tracks[ident]
                box = np.asarray(detections[di]['box'],dtype=float)
                measured = (box-track.box)/(frame-track.observed_at)
                track.velocity = 0.5*track.velocity+0.5*measured
                track.box, track.observed_at = box, frame
                assigned[di] = ident
                used.add(ident)
            remaining = [i for i in remaining if i not in used]
        for di in strong:
            if di not in assigned:
                ident = self.next_id
                self.next_id += 1
                self.tracks[ident] = Track(ident,np.array(detections[di]['box'],dtype=float),np.zeros(4),frame)
                assigned[di] = ident
        return [dict(detections[i],track_id=ident) for i,ident in sorted(assigned.items())]
