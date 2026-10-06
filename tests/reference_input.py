"""Deterministic synthetic EMG; no participant recordings are distributed."""

import numpy as np


def synthetic_emg():
    # Integer mixing makes the input independent of random-generator versions.
    bits = np.arange(14000 * 16, dtype=np.uint32) + np.uint32(20261006)
    bits = (bits ^ (bits >> 16)) * np.uint32(0x7FEB352D)
    bits = (bits ^ (bits >> 15)) * np.uint32(0x846CA68B)
    bits ^= bits >> 16
    values = (bits & 65535).astype(np.float32) - 32768
    return (values / 512).reshape(14000, 16)


def synthetic_boundary():
    return (np.arange(1, 21, dtype=np.float32) / 40)[None]
