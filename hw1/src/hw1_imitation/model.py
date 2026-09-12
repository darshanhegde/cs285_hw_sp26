"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers: list[nn.Module] = []
        layers.append(nn.Linear(state_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.net(state)
        return out.view(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_chunk = self.forward(state)
        return torch.mean(torch.sum((action_chunk - pred_chunk) ** 2, dim=(1, 2)))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.forward(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        chunk_dim = chunk_size * action_dim
        in_dim = state_dim + chunk_dim + 1  # state, noisy chunk, and tau

        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], chunk_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action_chunk_tau: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        flat_chunk = action_chunk_tau.reshape(state.shape[0], -1)
        net_input = torch.cat([state, flat_chunk, tau.view(state.shape[0], 1)], dim=-1)
        out = self.net(net_input)
        return out.view(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(state.shape[0], device=state.device, dtype=state.dtype)
        tau_expanded = tau.view(-1, 1, 1)
        noisy_chunk = tau_expanded * action_chunk + (1 - tau_expanded) * noise

        pred_velocity = self.forward(state, noisy_chunk, tau)
        target_velocity = action_chunk - noise
        return torch.mean(torch.sum((pred_velocity - target_velocity) ** 2, dim=(1, 2)))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        action_chunk = torch.randn(
            batch_size, self.chunk_size, self.action_dim,
            device=state.device, dtype=state.dtype,
        )
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = torch.full((batch_size,), i * dt, device=state.device, dtype=state.dtype)
            velocity = self.forward(state, action_chunk, tau)
            action_chunk = action_chunk + dt * velocity
        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
