from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TrackedClip:
    track_id: int
    start_frame: int
    bboxes: list[torch.Tensor] = field(default_factory=list)  # each (4,) xyxy
    masks: list[torch.Tensor] = field(default_factory=list)  # each (H, W) bool

    @property
    def end_frame(self) -> int:
        return self.start_frame + len(self.bboxes) - 1

    @property
    def frame_count(self) -> int:
        return len(self.bboxes)

    def frame_indices(self) -> list[int]:
        return list(range(self.start_frame, self.start_frame + len(self.bboxes)))


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU matrix between two sets of boxes (xyxy format). No CPU sync.
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) IoU matrix
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    if n == 0 or m == 0:
        return torch.zeros((n, m), device=boxes1.device)

    b1 = boxes1.unsqueeze(1)  # (N, 1, 4)
    b2 = boxes2.unsqueeze(0)  # (1, M, 4)

    inter_x1 = torch.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = torch.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = torch.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    return inter_area / union_area.clamp(min=1e-6)


def merge_overlapping_boxes(
    bboxes: torch.Tensor, masks: torch.Tensor, iou_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge overlapping bboxes within a single frame using GPU ops.
    bboxes: (K, 4) xyxy
    masks: (K, H, W) bool
    Returns merged (N, 4) bboxes and (N, H, W) masks where N <= K
    """
    n = bboxes.shape[0]
    if n <= 1:
        return bboxes, masks

    iou_matrix = compute_iou_matrix(bboxes, bboxes)
    adjacency = iou_matrix > iou_threshold

    labels = torch.arange(n, device=bboxes.device)
    for _ in range(n):
        for i in range(n):
            neighbors = adjacency[i].nonzero(as_tuple=True)[0]
            if neighbors.numel() > 0:
                min_label = labels[neighbors].min()
                if min_label < labels[i]:
                    labels[i] = min_label

    unique_labels = labels.unique()
    merged_bboxes = []
    merged_masks = []

    for label in unique_labels:
        group_mask = labels == label
        group_boxes = bboxes[group_mask]
        x1 = group_boxes[:, 0].min()
        y1 = group_boxes[:, 1].min()
        x2 = group_boxes[:, 2].max()
        y2 = group_boxes[:, 3].max()
        merged_bboxes.append(torch.stack([x1, y1, x2, y2]))
        merged_masks.append(masks[group_mask].any(dim=0))

    return torch.stack(merged_bboxes), torch.stack(merged_masks)


class ClipTracker:
    def __init__(self, max_clip_size: int, iou_threshold: float = 0.3):
        self.max_clip_size = max_clip_size
        self.iou_threshold = iou_threshold
        self.active_clips: dict[int, TrackedClip] = {}
        self.next_track_id = 0
        self.last_frame_boxes: torch.Tensor | None = None  # (T, 4) stacked boxes
        self.track_ids: list[int] = []  # track_id for each row in last_frame_boxes

    def update(
        self, frame_idx: int, bboxes: torch.Tensor, masks: torch.Tensor
    ) -> tuple[list[TrackedClip], set[int]]:
        """
        Update tracker with detections from a new frame.
        
        bboxes: (K, 4) xyxy format
        masks: (K, H, W) bool
        
        Returns:
            ended_clips: clips that ended this frame (max size or no match)
            active_track_ids: track ids that cover this frame
        """
        if bboxes.shape[0] > 0:
            bboxes, masks = merge_overlapping_boxes(bboxes, masks, self.iou_threshold)

        ended_clips: list[TrackedClip] = []
        active_track_ids: set[int] = set()

        if bboxes.shape[0] == 0:
            for track_id in self.track_ids:
                ended_clips.append(self.active_clips.pop(track_id))
            self.last_frame_boxes = None
            self.track_ids = []
            return ended_clips, active_track_ids

        n_detections = bboxes.shape[0]
        det_to_track: dict[int, int] = {}
        matched_track_indices: set[int] = set()

        if self.last_frame_boxes is not None and len(self.track_ids) > 0:
            iou_matrix = compute_iou_matrix(bboxes, self.last_frame_boxes)
            iou_cpu = iou_matrix.cpu().numpy()
            n_tracks = len(self.track_ids)
            matched_det_cpu = [False] * n_detections

            for _ in range(min(n_detections, n_tracks)):
                best_iou = self.iou_threshold
                best_det, best_track = -1, -1
                for di in range(n_detections):
                    if matched_det_cpu[di]:
                        continue
                    for ti in range(n_tracks):
                        if ti in matched_track_indices:
                            continue
                        if iou_cpu[di, ti] > best_iou:
                            best_iou = iou_cpu[di, ti]
                            best_det, best_track = di, ti
                if best_det < 0:
                    break
                matched_det_cpu[best_det] = True
                matched_track_indices.add(best_track)
                det_to_track[best_det] = best_track

        for det_idx, track_idx in det_to_track.items():
            track_id = self.track_ids[track_idx]
            clip = self.active_clips[track_id]
            clip.bboxes.append(bboxes[det_idx])
            clip.masks.append(masks[det_idx])
            active_track_ids.add(track_id)

            if clip.frame_count >= self.max_clip_size:
                ended_clips.append(clip)
                del self.active_clips[track_id]

        for track_idx, track_id in enumerate(self.track_ids):
            if track_idx not in matched_track_indices and track_id in self.active_clips:
                ended_clips.append(self.active_clips.pop(track_id))

        for det_idx in range(n_detections):
            if det_idx not in det_to_track:
                track_id = self.next_track_id
                self.next_track_id += 1
                clip = TrackedClip(
                    track_id=track_id,
                    start_frame=frame_idx,
                    bboxes=[bboxes[det_idx]],
                    masks=[masks[det_idx]],
                )
                self.active_clips[track_id] = clip
                active_track_ids.add(track_id)

        new_boxes = []
        new_track_ids = []
        for track_id in active_track_ids:
            clip = self.active_clips.get(track_id)
            if clip:
                new_boxes.append(clip.bboxes[-1])
                new_track_ids.append(track_id)

        if new_boxes:
            self.last_frame_boxes = torch.stack(new_boxes)
            self.track_ids = new_track_ids
        else:
            self.last_frame_boxes = None
            self.track_ids = []

        return ended_clips, active_track_ids

    def flush(self) -> list[TrackedClip]:
        """End all active clips and return them."""
        clips = list(self.active_clips.values())
        self.active_clips.clear()
        self.last_frame_boxes = None
        self.track_ids = []
        return clips
