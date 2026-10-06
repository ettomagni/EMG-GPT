"""Offline hand-pose inference with pretrained EMG-GPT models."""

__version__ = "0.1.0"

__all__ = ["PosePredictor", "PosePrediction", "plan_windows"]


def __getattr__(name):
    # Keep model-submodule imports acyclic and package metadata lightweight.
    if name in __all__:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(name)
