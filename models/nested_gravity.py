from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class NestedGravitationalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        gravity_dim: int = 16,
        num_centers: int = 8,
        num_parent_centers: int = 2,
        local_window: int = 16,
        dropout: float = 0.0,
        epsilon: float = 0.1,
        gravity_power: float = 1.0,
        force_clip: float = 5.0,
        temperature: float = 0.5,
        ema_alpha: float = 0.05,
        use_local_gravity: bool = True,
        use_nesting: bool = True,
        use_nested_bridges: bool = False,
        bridge_strength: float = 1.0,
        use_repulsion: bool = False,
        **_: Dict,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gravity_dim = gravity_dim
        self.num_centers = num_centers
        self.num_parent_centers = num_parent_centers
        self.local_window = local_window
        self.epsilon = epsilon
        self.gravity_power = gravity_power
        self.force_clip = force_clip
        self.temperature = temperature
        self.ema_alpha = ema_alpha
        self.use_local_gravity = use_local_gravity
        self.use_nesting = use_nesting
        self.use_nested_bridges = use_nested_bridges
        self.bridge_strength = bridge_strength
        self.use_repulsion = use_repulsion

        if use_nested_bridges and num_parent_centers < 1:
            raise ValueError("num_parent_centers must be positive when nested bridges are enabled")

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_net = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, gravity_dim),
        )
        self.local_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + gravity_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.local_direction = nn.Linear(gravity_dim, hidden_dim)
        self.local_value = nn.Linear(hidden_dim, hidden_dim)
        self.center_direction = nn.Linear(gravity_dim, hidden_dim)
        self.center_value_proj = nn.Linear(hidden_dim, hidden_dim)
        if self.use_nested_bridges:
            self.parent_centers = nn.Parameter(torch.randn(num_parent_centers, gravity_dim) * 0.1)
            self.bridge_value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.bridge_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru_cell = nn.GRUCell(embedding_dim + hidden_dim + hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.centers = nn.Parameter(torch.randn(num_centers, gravity_dim) * 0.1)
        self.register_buffer("center_values", torch.zeros(num_centers, hidden_dim))
        self.latest_metrics: Dict[str, float] = {}
        self.latest_trace: Dict[str, list] = {}

    def _kernel(self, squared_distance: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.pow(squared_distance + self.epsilon ** 2, self.gravity_power / 2.0)

    def _update_centers(
        self,
        center_values: torch.Tensor,
        assignments: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            assignment_mass = assignments.sum(dim=0).unsqueeze(-1).clamp_min(1e-6)
            weighted_hidden = assignments.transpose(0, 1) @ hidden_state
            weighted_hidden = weighted_hidden / assignment_mass
            decay = self.ema_alpha * assignments.mean(dim=0).unsqueeze(-1)
            return center_values * (1.0 - decay) + decay * weighted_hidden

    def repulsion_regularization(self) -> torch.Tensor:
        if not self.use_repulsion or self.num_centers < 2:
            return self.centers.new_tensor(0.0)
        delta = self.centers.unsqueeze(1) - self.centers.unsqueeze(0)
        dist2 = delta.pow(2).sum(dim=-1) + torch.eye(self.num_centers, device=self.centers.device)
        inv = 1.0 / torch.pow(dist2 + self.epsilon ** 2, (self.gravity_power + 2.0) / 2.0)
        mask = 1.0 - torch.eye(self.num_centers, device=self.centers.device)
        return (delta.pow(2).sum(dim=-1) * inv * mask).mean()

    def center_norm_regularization(self) -> torch.Tensor:
        return ((self.centers.norm(dim=-1) - 1.0) ** 2).mean()

    def forward(self, tokens: torch.Tensor, return_metrics: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, sequence_length = tokens.shape
        device = tokens.device
        embedded = self.dropout(self.embedding(tokens))
        h_prev = embedded.new_zeros(batch_size, self.hidden_dim)
        hidden_history = []
        position_history = []
        logits = []
        local_force_norms = []
        nesting_force_norms = []
        bridge_force_norms = []
        parent_entropies = []
        center_entropies = []
        clipped_fraction = []
        token_center_distances = []
        token_token_distances = []
        assignment_history = []
        center_values = self.center_values.detach().clone()
        positions_trace = []
        assignment_trace = []
        entropy_trace = []
        local_force_trace = []
        nesting_force_trace = []
        bridge_force_trace = []
        interaction_trace = embedded.new_zeros(sequence_length, sequence_length)

        for t in range(sequence_length):
            embedding_t = embedded[:, t, :]
            position_input = torch.cat([embedding_t, h_prev], dim=-1)
            r_t = self.position_net(position_input)

            local_force = embedded.new_zeros(batch_size, self.hidden_dim)
            if self.use_local_gravity and hidden_history:
                start = max(0, len(hidden_history) - self.local_window)
                prev_hidden = torch.stack(hidden_history[start:], dim=1)
                prev_positions = torch.stack(position_history[start:], dim=1)
                delta = prev_positions - r_t.unsqueeze(1)
                dist2 = delta.pow(2).sum(dim=-1)
                token_token_distances.append(torch.sqrt(dist2 + self.epsilon ** 2).mean())
                kernel = self._kernel(dist2)
                cosine = F.cosine_similarity(
                    r_t.unsqueeze(1).expand_as(prev_positions),
                    prev_positions,
                    dim=-1,
                    eps=1e-8,
                )
                gate_input = torch.cat(
                    [
                        h_prev.unsqueeze(1).expand(-1, prev_hidden.size(1), -1),
                        prev_hidden,
                        delta,
                        kernel.unsqueeze(-1),
                        cosine.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                gate = torch.sigmoid(self.local_gate(gate_input)).squeeze(-1)
                direction = torch.tanh(self.local_direction(delta))
                values = self.local_value(prev_hidden)
                raw_force = ((gate * kernel).unsqueeze(-1) * direction * values).sum(dim=1)
                local_force = torch.clamp(raw_force, min=-self.force_clip, max=self.force_clip)
                clipped_fraction.append((raw_force.abs() > self.force_clip).float().mean())
                interaction_trace[t, start : start + prev_hidden.size(1)] = (gate[0] * kernel[0]).detach()
            else:
                token_token_distances.append(embedded.new_tensor(0.0))
                clipped_fraction.append(embedded.new_tensor(0.0))

            logits_scores = (r_t @ self.centers.t()) / max(self.temperature, 1e-6)
            assignments = torch.softmax(logits_scores, dim=-1)
            assignment_history.append(assignments)
            entropy = -(assignments * torch.log(assignments.clamp_min(1e-8))).sum(dim=-1)
            center_entropies.append(entropy.mean())
            entropy_trace.append(entropy[0].detach())
            center_delta = self.centers.unsqueeze(0) - r_t.unsqueeze(1)
            center_dist2 = center_delta.pow(2).sum(dim=-1)
            token_center_distances.append(torch.sqrt(center_dist2 + self.epsilon ** 2).mean())

            nesting_force = embedded.new_zeros(batch_size, self.hidden_dim)
            bridge_force = embedded.new_zeros(batch_size, self.hidden_dim)
            if self.use_nesting:
                center_kernel = self._kernel(center_dist2)
                direction = torch.tanh(self.center_direction(center_delta))
                projected_center_values = self.center_value_proj(center_values).unsqueeze(0).expand(batch_size, -1, -1)
                raw_force = (assignments * center_kernel).unsqueeze(-1) * direction * projected_center_values
                nesting_force = torch.clamp(raw_force.sum(dim=1), min=-self.force_clip, max=self.force_clip)

                if self.use_nested_bridges:
                    bridge_scores = (self.centers @ self.parent_centers.t()) / max(self.temperature, 1e-6)
                    child_to_parent = torch.softmax(bridge_scores, dim=-1)
                    parent_mass = child_to_parent.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
                    parent_values = (child_to_parent.t() @ center_values) / parent_mass
                    bridged_center_values = child_to_parent @ parent_values
                    bridge_context = assignments @ bridged_center_values
                    bridge_gate = torch.sigmoid(self.bridge_gate(torch.cat([h_prev, bridge_context], dim=-1)))
                    bridge_force = bridge_gate * torch.tanh(self.bridge_value_proj(bridge_context))
                    bridge_force = torch.clamp(
                        self.bridge_strength * bridge_force,
                        min=-self.force_clip,
                        max=self.force_clip,
                    )
                    nesting_force = torch.clamp(
                        nesting_force + bridge_force,
                        min=-self.force_clip,
                        max=self.force_clip,
                    )
                    parent_entropy = -(child_to_parent * torch.log(child_to_parent.clamp_min(1e-8))).sum(dim=-1)
                    parent_entropies.append(parent_entropy.mean())

            cell_input = torch.cat([embedding_t, local_force, nesting_force], dim=-1)
            h_t = self.gru_cell(cell_input, h_prev)
            h_t = self.layer_norm(h_t)

            if self.use_nesting:
                center_values = self._update_centers(center_values, assignments, h_t)

            hidden_history.append(h_t)
            position_history.append(r_t)
            local_force_norms.append(local_force.norm(dim=-1).mean())
            nesting_force_norms.append(nesting_force.norm(dim=-1).mean())
            bridge_force_norms.append(bridge_force.norm(dim=-1).mean())
            positions_trace.append(r_t[0].detach())
            assignment_trace.append(assignments[0].detach())
            local_force_trace.append(local_force[0].norm().detach())
            nesting_force_trace.append(nesting_force[0].norm().detach())
            bridge_force_trace.append(bridge_force[0].norm().detach())
            logits.append(self.output_projection(h_t))
            h_prev = h_t

        stacked_logits = torch.stack(logits, dim=1)
        assignment_tensor = torch.stack(assignment_history, dim=1)
        mean_q = assignment_tensor.mean(dim=(0, 1)) if assignment_history else torch.full((self.num_centers,), 1.0 / self.num_centers, device=device)
        effective_num_centers = torch.exp(-(mean_q * torch.log(mean_q.clamp_min(1e-8))).sum())
        center_usage = assignment_tensor.argmax(dim=-1).reshape(-1).bincount(minlength=self.num_centers).float()
        center_usage = center_usage / center_usage.sum().clamp_min(1.0)
        self.latest_metrics = {
            "mean_local_force_norm": torch.stack(local_force_norms).mean().item(),
            "mean_nesting_force_norm": torch.stack(nesting_force_norms).mean().item(),
            "mean_bridge_force_norm": torch.stack(bridge_force_norms).mean().item(),
            "parent_center_entropy": torch.stack(parent_entropies).mean().item() if parent_entropies else 0.0,
            "center_entropy": torch.stack(center_entropies).mean().item(),
            "effective_num_centers": effective_num_centers.item(),
            "clipped_force_fraction": torch.stack(clipped_fraction).mean().item(),
            "mean_token_center_distance": torch.stack(token_center_distances).mean().item(),
            "mean_token_token_distance": torch.stack(token_token_distances).mean().item(),
            "center_usage": center_usage.detach().cpu().tolist(),
            "center_norm_regularization": self.center_norm_regularization().item(),
            "repulsion_regularization": self.repulsion_regularization().item(),
        }
        self.latest_trace = {
            "positions": torch.stack(positions_trace).cpu().tolist(),
            "assignments": torch.stack(assignment_trace).cpu().tolist(),
            "center_entropy_by_position": torch.stack(entropy_trace).cpu().tolist(),
            "local_force_norms_by_position": torch.stack(local_force_trace).cpu().tolist(),
            "nesting_force_norms_by_position": torch.stack(nesting_force_trace).cpu().tolist(),
            "bridge_force_norms_by_position": torch.stack(bridge_force_trace).cpu().tolist(),
            "interaction_strength": interaction_trace.cpu().tolist(),
        }
        if self.training and self.use_nesting:
            with torch.no_grad():
                self.center_values.copy_(center_values)
        if return_metrics:
            return stacked_logits, dict(self.latest_metrics)
        return stacked_logits
