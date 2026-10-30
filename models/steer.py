"""Activation steering for RQ4: add alpha times a direction to one decoder layer's output.

The probes index layers like `output_hidden_states`: index 0 is the embeddings and
index L is the output of decoder block L - 1. So steering probe layer L hooks
block L - 1. The vector is added at every position, prompt and generated tokens.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


def decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot find the decoder layers of this model")


@contextmanager
def steer(model, layer: int, vector, alpha: float):
    """Within the block, block `layer - 1` outputs h + alpha * vector."""
    if alpha == 0:
        yield
        return
    if layer < 1:
        raise ValueError("layer 0 is the embedding layer; steer layer 1 or above")
    block = decoder_layers(model)[layer - 1]
    delta = torch.as_tensor(np.asarray(vector), dtype=torch.float32, device=model.device) * alpha

    def hook(module, inputs, output):
        # Depending on the transformers version a block returns a tuple or a tensor.
        if isinstance(output, tuple):
            return (output[0] + delta.to(output[0].dtype),) + tuple(output[1:])
        return output + delta.to(output.dtype)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def random_direction(like, seed: int) -> np.ndarray:
    """A Gaussian direction with the same norm as `like`."""
    like = np.asarray(like, dtype=np.float32)
    vector = np.random.default_rng(seed).standard_normal(like.shape).astype(np.float32)
    return vector * (np.linalg.norm(like) / np.linalg.norm(vector))
