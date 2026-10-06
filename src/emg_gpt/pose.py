"""Selected Regression and Tracking decoders; no training dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .model import EMGFrameGPT

PoseHeadMode = Literal["regression_lstm", "tracking_lstm"]
BackboneMode = Literal["frozen", "top", "full"]
TrackingUpsampleMode = Literal["repeat"]
TemporalFeatureMode = Literal["hidden"]


@dataclass(frozen=True)
class PoseModelConfig:
    n_joints: int = 20
    pose_steps_per_frame: int = 2
    head_mode: PoseHeadMode = "regression_lstm"
    head_hidden: int = 256
    head_layers: int = 2
    dropout: float = 0.1
    tracking_feature_dim: int = 64
    tracking_left_context_frames: int = 25
    tracking_velocity_scale: float = 0.01
    tracking_upsample_mode: TrackingUpsampleMode = "repeat"
    regression_feature_dim: int = 64
    regression_left_context_frames: int = 25
    regression_position_steps: int = 12
    regression_output_scale: float = 0.01
    temporal_feature_mode: TemporalFeatureMode = "hidden"
    backbone_mode: BackboneMode = "frozen"
    unfreeze_last_n: int = 2

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class StateConditionedTrackingHead(nn.Module):
    """Token-rate features to a 50 Hz pose trajectory with pose feedback.

    This is the token-adapted analogue of emg2pose's state-conditioned LSTM:
    GPT features are held causally to the pose rate, then each angular increment
    is conditioned on the pose predicted at the preceding rollout step.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_joints: int,
        feature_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        left_context_frames: int,
        pose_steps_per_frame: int,
        velocity_scale: float,
        upsample_mode: TrackingUpsampleMode,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("Tracking-head dimensions must be positive")
        if left_context_frames < 1:
            raise ValueError("tracking_lstm requires at least one left-context token frame")
        if pose_steps_per_frame < 1 or velocity_scale <= 0:
            raise ValueError("Tracking pose rate and velocity scale must be positive")
        if upsample_mode != "repeat":
            raise ValueError(f"Unknown tracking upsample mode: {upsample_mode}")
        self.n_joints = int(n_joints)
        self.left_context_frames = int(left_context_frames)
        self.pose_steps_per_frame = int(pose_steps_per_frame)
        self.velocity_scale = float(velocity_scale)
        self.upsample_mode = upsample_mode
        self.feature_proj = nn.Linear(d_model, feature_dim)
        self.rnn = nn.LSTM(
            input_size=feature_dim + n_joints,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.LeakyReLU(), nn.Linear(hidden_size, n_joints))

    def _expand_features(self, hidden: torch.Tensor) -> torch.Tensor:
        scored_hidden = self.feature_proj(hidden[:, self.left_context_frames :])
        return scored_hidden.repeat_interleave(self.pose_steps_per_frame, dim=1)

    def forward(self, hidden: torch.Tensor, initial_pose: torch.Tensor) -> torch.Tensor:
        recurrent_state = None
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden [batch,time,dim], got {hidden.shape}")
        if initial_pose.shape != (hidden.shape[0], self.n_joints):
            raise ValueError(
                f"Expected initial_pose {(hidden.shape[0], self.n_joints)}, got {tuple(initial_pose.shape)}"
            )
        if hidden.shape[1] <= self.left_context_frames:
            raise ValueError(
                f"Input has {hidden.shape[1]} token frames but tracking left context is {self.left_context_frames}"
            )
        features = self._expand_features(hidden)
        rollout_steps = features.shape[1]
        previous_pose = initial_pose
        trajectory = []
        for step_ix in range(rollout_steps):
            decoder_input = torch.cat((features[:, step_ix], previous_pose), dim=-1).unsqueeze(1)
            decoded, recurrent_state = self.rnn(decoder_input, recurrent_state)
            angular_increment = self.out(decoded[:, 0]) * self.velocity_scale
            previous_pose = previous_pose + angular_increment
            trajectory.append(previous_pose)
        prediction = torch.stack(trajectory, dim=1)
        return prediction


class StateConditionedRegressionHead(nn.Module):
    """Token adaptation of the official vEMG2Pose Regression decoder.

    The first rollout steps predict absolute angles. Later steps integrate the
    predicted angular velocity. No ground-truth initial pose is supplied.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_joints: int,
        feature_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        left_context_frames: int,
        pose_steps_per_frame: int,
        position_steps: int,
        output_scale: float,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("Regression-head dimensions must be positive")
        if left_context_frames < 1:
            raise ValueError("regression_lstm requires at least one left-context token frame")
        if pose_steps_per_frame < 1 or position_steps < 1 or output_scale <= 0:
            raise ValueError("Regression rollout settings must be positive")
        self.n_joints = int(n_joints)
        self.left_context_frames = int(left_context_frames)
        self.pose_steps_per_frame = int(pose_steps_per_frame)
        self.position_steps = int(position_steps)
        self.output_scale = float(output_scale)
        self.feature_proj = nn.Linear(d_model, feature_dim)
        self.rnn = nn.LSTM(
            input_size=feature_dim + n_joints,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.LeakyReLU(), nn.Linear(hidden_size, 2 * n_joints))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden [batch,time,dim], got {hidden.shape}")
        if hidden.shape[1] <= self.left_context_frames:
            raise ValueError(
                f"Input has {hidden.shape[1]} token frames but regression left context is {self.left_context_frames}"
            )
        scored = self.feature_proj(hidden[:, self.left_context_frames :])
        rollout_steps = scored.shape[1] * self.pose_steps_per_frame
        features = F.interpolate(
            scored.transpose(1, 2), size=rollout_steps, mode="linear", align_corners=True
        ).transpose(1, 2)
        previous_pose = torch.zeros(
            hidden.shape[0], self.n_joints, dtype=features.dtype, device=features.device
        )
        recurrent_state = None
        trajectory = []
        for step_ix in range(rollout_steps):
            decoder_input = torch.cat((features[:, step_ix], previous_pose), dim=-1).unsqueeze(1)
            decoded, recurrent_state = self.rnn(decoder_input, recurrent_state)
            output = self.out(decoded[:, 0]) * self.output_scale
            position, velocity = output.chunk(2, dim=-1)
            previous_pose = position if step_ix < self.position_steps else previous_pose + velocity
            trajectory.append(previous_pose)
        return torch.stack(trajectory, dim=1)


class EMGPoseModel(nn.Module):
    """Complete adapted backbone and 20-joint decoder; recurrent state resets per window."""

    def __init__(self, backbone: EMGFrameGPT, config: PoseModelConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        if config.temporal_feature_mode != "hidden":
            raise ValueError("Only hidden-state pose decoders are supported")
        common = dict(
            d_model=backbone.config.d_model,
            n_joints=config.n_joints,
            hidden_size=config.head_hidden,
            num_layers=config.head_layers,
            dropout=config.dropout,
            pose_steps_per_frame=config.pose_steps_per_frame,
        )
        if config.head_mode == "regression_lstm":
            self.pose_head = StateConditionedRegressionHead(
                **common,
                feature_dim=config.regression_feature_dim,
                left_context_frames=config.regression_left_context_frames,
                position_steps=config.regression_position_steps,
                output_scale=config.regression_output_scale,
            )
        elif config.head_mode == "tracking_lstm" and config.tracking_upsample_mode == "repeat":
            self.pose_head = StateConditionedTrackingHead(
                **common,
                feature_dim=config.tracking_feature_dim,
                left_context_frames=config.tracking_left_context_frames,
                velocity_scale=config.tracking_velocity_scale,
                upsample_mode="repeat",
            )
        else:
            raise ValueError("Expected the selected Regression or Tracking LSTM decoder")

    @property
    def requires_initial_pose(self) -> bool:
        return self.config.head_mode == "tracking_lstm"

    def forward(
        self, tokens: torch.Tensor, initial_pose: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.backbone.forward_hidden(tokens)
        if self.requires_initial_pose:
            if initial_pose is None:
                raise ValueError("Tracking requires an explicit boundary pose")
            return self.pose_head(hidden, initial_pose)
        if initial_pose is not None:
            raise ValueError("Regression does not accept an initial pose")
        return self.pose_head(hidden)
