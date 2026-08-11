"""Neural network architectures for TS-JEPA."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic 2D residual block with optional spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ContextEncoder(nn.Module):
    """
    Convolutional ResNet encoder for CartPole frames.

    Input:  (B, 3, 64, 128)
    Output: (B, 256)
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 128)
            features = self.layer3(self.layer2(self.layer1(self.stem(dummy))))
            flat_dim = features.numel()

        self.projection = nn.Linear(flat_dim, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        return self.projection(x)


class Predictor(nn.Module):
    """
    MLP that predicts a sequence of future embeddings from the current embedding and actions.

    Input:  embedding (B, 256), future actions (B, Kp, 1)
    Output: (B, Kp, 256)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        action_horizon: int = 5,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_horizon = action_horizon

        input_dim = embedding_dim + action_horizon
        output_dim = embedding_dim * action_horizon
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embedding: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        if future_actions.ndim == 3:
            actions = future_actions.flatten(start_dim=1)
        else:
            actions = future_actions

        if actions.shape[-1] != self.action_horizon:
            raise ValueError(
                f"Expected {self.action_horizon} future actions, got shape {tuple(actions.shape)}"
            )

        batch_size = embedding.shape[0]
        x = torch.cat([embedding, actions], dim=-1)
        x = self.mlp(x)
        return x.view(batch_size, self.action_horizon, self.embedding_dim)


class SemanticActor(nn.Module):
    """
    MLP policy head that maps embeddings to continuous control commands.

    Input:  (B, 256)
    Output: (B, 1)
    """

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)


def _verify_shapes() -> None:
    batch_size = 4
    action_horizon = 5

    images = torch.randn(batch_size, 3, 64, 128)
    embedding = torch.randn(batch_size, 256)
    future_actions = torch.randn(batch_size, action_horizon, 1)

    encoder = ContextEncoder()
    predictor = Predictor(action_horizon=action_horizon)
    actor = SemanticActor()

    with torch.no_grad():
        layer3_features = encoder.layer3(
            encoder.layer2(encoder.layer1(encoder.stem(images)))
        )
    print(
        "ContextEncoder layer3 output:",
        tuple(layer3_features.shape),
        f"flattened dim={layer3_features.numel() // batch_size}",
    )

    encoded = encoder(images)
    predicted = predictor(embedding, future_actions)
    action = actor(embedding)

    assert encoded.shape == (batch_size, 256), f"encoder: {encoded.shape}"
    assert predicted.shape == (batch_size, action_horizon, 256), f"predictor: {predicted.shape}"
    assert action.shape == (batch_size, 1), f"actor: {action.shape}"

    print("ContextEncoder:", tuple(images.shape), "->", tuple(encoded.shape))
    print(
        "Predictor:",
        (batch_size, 256),
        "+",
        tuple(future_actions.shape),
        "->",
        tuple(predicted.shape),
    )
    print("SemanticActor:", tuple(embedding.shape), "->", tuple(action.shape))
    print("All shape checks passed.")


if __name__ == "__main__":
    _verify_shapes()
