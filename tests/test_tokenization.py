"""Protect tokenizer position, channel and RVQ layout independently of model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emg_gpt.tokenization import tokenize_frames


@pytest.mark.parametrize("n_positions,batch_size", [(256, 1), (256, 16), (37, 3)])
def test_independent_patches_use_last_learned_position(n_positions, batch_size):
    # More than one storage block catches accidental dependence on frame/block index.
    emg = np.arange(16 * 1000, dtype=np.float32).reshape(16, 1000)

    class Tokenizer:
        encoder = SimpleNamespace(time_embed=torch.empty(n_positions, 2))
        frames_seen = 0

        def eval(self):
            return self

        def encode(self, patches, temporal, spatial):
            batch = len(patches)
            assert patches.shape == (batch, 16, 1, 200)
            assert temporal.shape == spatial.shape == (batch, 16)
            assert torch.all(temporal == self.encoder.time_embed.shape[0] - 1)
            assert torch.equal(spatial, torch.arange(16).expand(batch, -1))
            frames = torch.arange(self.frames_seen, self.frames_seen + batch)
            for index, frame in enumerate(frames.tolist()):
                np.testing.assert_array_equal(
                    patches[index, :, 0], emg[:, frame * 40 : frame * 40 + 200]
                )
            indices = []
            for branch in range(4):
                # Independent expected axes: scale, batch, channel, one time patch.
                values = (
                    torch.arange(6)[:, None, None] * 1000
                    + frames[None, :, None] * 16
                    + torch.arange(16)[None, None, :]
                    + branch * 10000
                )
                indices.append(values.reshape(6, -1))
            self.frames_seen += batch
            return None, indices, None, None

    tokenizer = Tokenizer()
    codes = tokenize_frames(tokenizer, emg, 18, batch_size=batch_size, device=torch.device("cpu"))
    expected = (
        np.arange(18)[:, None, None, None] * 16
        + np.arange(16)[None, :, None, None]
        + np.arange(4)[None, None, :, None] * 10000
        + np.arange(4)[None, None, None, :] * 1000
    )
    assert codes.dtype == np.uint16
    np.testing.assert_array_equal(codes, expected)
