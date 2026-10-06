"""Channel convention used by the released NeuroRVQ EMG tokenizer."""

from __future__ import annotations

import numpy as np

GLOBAL_EMG_CHANNELS = np.asarray(
    [
        b"c1",
        b"c10",
        b"c11",
        b"c12",
        b"c13",
        b"c14",
        b"c15",
        b"c16",
        b"c2",
        b"c3",
        b"c4",
        b"c5",
        b"c6",
        b"c7",
        b"c8",
        b"c9",
    ],
    dtype="S4",
)

# Compatibility alias used by the original NeuroRVQ inference helper.
ch_names_global = GLOBAL_EMG_CHANNELS
