"""
nnUNetPredictor that feeds each sliding-window tile's normalized position to
the network, matching the patch-position conditioning used during training.
"""
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def patch_center_pos(sl, image_shape) -> torch.Tensor:
    """Normalized patch-center coordinates in [-1, 1] for a sliding-window slicer."""
    norm = []
    for s, dim in zip(sl[1:], image_shape):
        lo = max(0, s.start if s.start is not None else 0)
        hi = min(dim, s.stop if s.stop is not None else dim)
        norm.append(2.0 * ((lo + hi) / 2.0) / max(float(dim), 1.0) - 1.0)
    return torch.tensor(norm, dtype=torch.float32)


class PositionAwarePredictor(nnUNetPredictor):
    """
    Tiles are consumed in slicer order (single producer, FIFO queue), so the
    positions are precomputed per call and handed to the network one tile at
    a time. Positions are not flipped for mirror TTA: they describe where the
    tile sits in the volume, which mirroring doesn't change.
    """
    def _internal_predict_sliding_window_return_logits(self, data, slicers, do_on_device=True):
        image_shape = data.shape[1:]
        self._tile_positions = iter([patch_center_pos(sl, image_shape) for sl in slicers])
        return super()._internal_predict_sliding_window_return_logits(data, slicers, do_on_device)

    def _internal_maybe_mirror_and_predict(self, x):
        pos = next(self._tile_positions)
        self.network.current_pos = pos.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1)
        try:
            return super()._internal_maybe_mirror_and_predict(x)
        finally:
            self.network.current_pos = None
