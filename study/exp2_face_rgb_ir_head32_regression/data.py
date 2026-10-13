"""Decode synchronized RGB+IR packets and apply the same view to all channels."""

from collections import OrderedDict

import torch
import torch.nn.functional as functional

from study.common.face_video import decode_indexed_frame
from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.config import BRIGHTNESS_DELTA, CONTRAST_DELTA, CROP_SCALE, IMAGE_SIZE
from .config import MEAN, STD


class RGBIRDataset(AllFramesDataset):
    def __init__(self, frame_index, records, ir_index, **kwargs):
        super().__init__(frame_index, records, **kwargs)
        self.ir_index = ir_index
        self._ir_decoders = OrderedDict()

    def __getstate__(self):
        state = super().__getstate__()
        state["_ir_decoders"] = OrderedDict()
        return state

    def _decode(self, global_frame_index, video_row):
        cached = self._decoded_cache.pop(global_frame_index, None)
        if cached is not None:
            self._decoded_cache[global_frame_index] = cached
            return cached
        position = self.record_index_videos[video_row]
        rgb = decode_indexed_frame(self.index, position, global_frame_index,
                                   self._handle(str(self.index.video_paths[position])), self._ffv1_decoders)
        ir = decode_indexed_frame(self.ir_index, position, global_frame_index,
                                  self._handle(str(self.ir_index.video_paths[position])), self._ir_decoders)
        if not torch.equal(ir[0], ir[1]) or not torch.equal(ir[1], ir[2]):
            raise ValueError("Expected a true grayscale IR crop; refusing an arbitrary color-channel substitution")
        tensor = torch.cat((rgb, ir[:1]), dim=0)
        self._decoded_cache[global_frame_index] = tensor
        while len(self._decoded_cache) > getattr(self, "decode_cache_frames", 16):
            self._decoded_cache.popitem(last=False)
        return tensor

    def close(self):
        super().close()
        if hasattr(self, "_ir_decoders"):
            self._ir_decoders.clear()


def prepare_images(images, view_codes, interpolation, device, *unused):
    if images.shape[1] != 4:
        raise ValueError("Expected uint8 RGB+IR input")
    images = images.to(device, non_blocking=True).float().div_(255.)
    codes = view_codes.to(device, non_blocking=True)
    if codes.ndim == 2:
        images = images.repeat_interleave(codes.shape[1], dim=0)
        codes = codes.flatten()
    flip = codes.eq(1)
    if flip.any():
        images[flip] = torch.flip(images[flip], dims=(-1,))
    brightness = codes.eq(3)
    if brightness.any():
        images[brightness] = (images[brightness]*(1+BRIGHTNESS_DELTA)).clamp_(0, 1)
    contrast = codes.eq(4)
    if contrast.any():
        selected = images[contrast]
        means = selected.mean(dim=(-2, -1), keepdim=True)
        images[contrast] = ((selected-means)*(1+CONTRAST_DELTA)+means).clamp_(0, 1)
    crop = codes.eq(2)
    output = torch.empty((len(images), 4, IMAGE_SIZE, IMAGE_SIZE), dtype=images.dtype, device=device)
    full = ~crop
    if full.any():
        output[full] = images[full] if images.shape[-2:] == (IMAGE_SIZE, IMAGE_SIZE) else functional.interpolate(
            images[full], size=(IMAGE_SIZE, IMAGE_SIZE), mode=interpolation, align_corners=False, antialias=True)
    if crop.any():
        height, width = images.shape[-2:]
        h, w = max(8, round(height*CROP_SCALE)), max(8, round(width*CROP_SCALE))
        top, left = (height-h)//2, (width-w)//2
        output[crop] = functional.interpolate(images[crop, :, top:top+h, left:left+w],
            size=(IMAGE_SIZE, IMAGE_SIZE), mode=interpolation, align_corners=False, antialias=True)
    mean = output.new_tensor(MEAN).view(1, 4, 1, 1)
    std = output.new_tensor(STD).view(1, 4, 1, 1)
    return ((output-mean)/std).contiguous(memory_format=torch.channels_last)
